"""Tables across jobs and accounts: list, running, gpus and account list."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Annotated, get_args

import typer
from rich.console import Group, RenderableType
from rich.live import Live
from rich.text import Text

from ..models import TERMINAL, JobState, checked_states
from . import options
from .common import client, output
from .options import AccountOption, LiveOption, ResourceArgument, StatesOption
from .output import ago, duration, hours, job_dict, table, until

UNFINISHED = sorted(set(get_args(JobState)) - TERMINAL)
RESOURCES = {"cpu": "CPU", "gpu": "GPU", "unknown": "?"}


# Jobs -----------------------------------------------------------------------------------------


def list_jobs(
    ctx: typer.Context,
    resource: ResourceArgument = "all",
    state: StatesOption = None,
    account: AccountOption = None,
    active: Annotated[bool, typer.Option("--active", help="Only unfinished jobs; --state overrides")] = False,
    limit: Annotated[int | None, typer.Option(min=1, help="Only the newest N jobs")] = None,
):
    """List jobs, oldest first: every job, or those of an account, a resource or some states."""
    store = client(ctx).store
    _, jobs = store.page(
        states=checked_states(state or None) or (UNFINISHED if active else None),
        account=account and client(ctx).config.account(account).id,
        pool=options.resource(resource),
        limit=limit or -1,
        newest_first=limit is not None,
    )
    jobs = jobs[::-1] if limit else jobs
    out = output(ctx)
    if out.json:
        out.emit([job_dict(job) for job in jobs])
        return
    headers = ("Job", "Name", "Run", "State", "Account", "Resource", "Outputs", "Waiting / error")
    out.show(table(headers, map(_job_cells, jobs), wrap=("Name", "Waiting / error")))


def _job_cells(job) -> list[str]:
    return [
        job.id[:12],
        job.spec.name,
        job.result_dir.name if job.run is not None else "",
        job.state,
        job.account,
        job.spec.accelerator or job.pool.upper(),
        job.download_state,
        job.error or job.wait_reason or "",
    ]


# Runs -----------------------------------------------------------------------------------------


def running(
    ctx: typer.Context,
    resource: ResourceArgument = "all",
    account: AccountOption = None,
    live: LiveOption = False,
    watch: Annotated[bool, typer.Option("--watch", help="Redraw the table until Ctrl-C")] = False,
    interval: Annotated[
        float | None, typer.Option(min=1, help="Seconds between redraws; default the polling interval")
    ] = None,
):
    """Runs holding each account's slots: this queue's jobs and runs started elsewhere."""
    agent = client(ctx).agent()
    stale_after = client(ctx).config.discovery_seconds

    def fetch() -> dict:
        return agent.running(options.resource(resource), account=account, live=live)

    out = output(ctx)
    if out.json:
        out.emit(fetch())
    elif watch:
        # Asking the providers takes several requests per account, so live views redraw slowly.
        seconds = interval or client(ctx).config.poll_seconds
        _watch(
            out.console,
            lambda: Group(*_running_view(fetch(), live, stale_after)),
            max(seconds, 60) if live else seconds,
        )
    else:
        out.show(*_running_view(fetch(), live, stale_after))


def _watch(console, render: Callable[[], RenderableType], seconds: float) -> None:
    with Live(render(), console=console, auto_refresh=False) as display:
        try:
            while True:
                time.sleep(seconds)
                display.update(render(), refresh=True)
        except KeyboardInterrupt:
            pass


def _running_view(result: dict, live: bool, stale_after: float) -> list[RenderableType]:
    parts = []
    if result["runs"]:
        headers = ("Account", "Location", "Notebook / run", "Name", "Type", "State", "Elapsed", "Job")
        parts.append(table(headers, map(_run_cells, result["runs"]), wrap=("Notebook / run", "Name")))
    counts = result["counts"]
    summary = [f"{counts[kind]} {RESOURCES[kind]}" for kind in ("gpu", "cpu") if counts[kind]]
    if counts["unknown"]:
        summary.append(f"{counts['unknown']} of unknown type, counted as both")
    parts.append(Text(f"{result['total']} running" + (": " + ", ".join(summary) if summary else "")))
    parts += [Text(note, style="dim") for note in _discovery_notes(result["discovery"], live, stale_after)]
    return parts


def _run_cells(run: dict) -> list[str]:
    ours = "job_id" in run
    return [
        run["account"],
        run["provider"],
        run["ref"],
        run["name"] if ours else "(external)",
        run.get("accelerator") or RESOURCES[run["resource"]],
        run.get("state", ""),
        duration(run["elapsed_seconds"]) if ours else "",
        run["job_id"][:12] if ours else "",
    ]


def _discovery_notes(checks: dict, live: bool, stale_after: float) -> list[str]:
    """Accounts whose runs started elsewhere are unknown, or were last read long ago."""
    notes = [
        f"{account}: last check failed: {check['error']}"
        if "error" in check
        else f"{account}: not checked yet"
        for account, check in checks.items()
        if "error" in check or check["checked_age_seconds"] is None
    ]
    ages = [
        check["checked_age_seconds"] for check in checks.values() if check["checked_age_seconds"] is not None
    ]
    # The worker checks accounts only while it has jobs to place, so an idle queue's check ages.
    if ages and max(ages) > stale_after:
        notes.append(f"Accounts last checked up to {duration(max(ages))} ago")
    if notes and not live:
        notes.append("--live asks the providers now")
    return notes


# Accounts -------------------------------------------------------------------------------------


def gpus(ctx: typer.Context, live: LiveOption = False):
    """GPU slots in use and free on each account, the GPU time left and when it resets, and totals."""
    result = client(ctx).agent().accounts(live=live)
    out = output(ctx)
    if out.json:
        out.emit(result)
        return
    headers = ("Account", "Location", "In use", "Free", "Time left", "Resets", "Devices", "Checked")
    rows = [
        [
            account["id"],
            account["provider"],
            _slots(account["gpu"]),
            _free(account["gpu"]),
            _time_left(account),
            until(account.get("gpu_refresh_at")),
            ", ".join(account.get("devices", [])),
            ago(account["checked_age_seconds"]),
        ]
        for account in result["accounts"]
    ]
    totals, bound = result["totals"], _bound(result["totals"])
    total = ["Total", "", _slots(totals["gpu"]), _free(totals["gpu"], bound), _total_time(result), "", "", ""]
    out.show(table(headers, rows, wrap=("Devices",), total=total))
    _account_notes(ctx, result, live)


def account_list(ctx: typer.Context, live: LiveOption = False):
    """Accounts in preference order with CPU and GPU slots in use and free, and GPU time left."""
    result = client(ctx).agent().accounts(live=live)
    out = output(ctx)
    if out.json:
        out.emit(result)
        return
    headers = ("Account", "Location", "CPU", "CPU free", "GPU", "GPU free", "GPU time left", "Checked")
    rows = [
        [
            account["id"],
            account["provider"],
            *_pool_cells(account),
            _time_left(account),
            ago(account["checked_age_seconds"]),
        ]
        for account in result["accounts"]
    ]
    totals = result["totals"]
    total = ["Total", "", *_pool_cells(totals, _bound(totals)), _total_time(result), ""]
    out.show(table(headers, rows, total=total))
    out.line(f"Failover: {result['failover']}. The first account is the default.", style="dim")
    _account_notes(ctx, result, live)


def _account_notes(ctx, result: dict, live: bool) -> None:
    checks = {account["id"]: account for account in result["accounts"]}
    output(ctx).notes(_discovery_notes(checks, live, client(ctx).config.discovery_seconds))


def _pool_cells(counts: dict, bound: str = "") -> list[str]:
    return [cell for pool in ("cpu", "gpu") for cell in (_slots(counts[pool]), _free(counts[pool], bound))]


def _slots(slots: dict) -> str:
    return f"{slots['used']}/{slots['limit']}"


def _free(slots: dict, bound: str = "") -> str:
    return "?" if slots["free"] is None else f"{bound}{slots['free']}"


def _bound(totals: dict) -> str:
    """Totals are lower bounds while some account is unknown."""
    return "" if totals["complete"] else ">="


def _time_left(account: dict) -> str:
    if not account["gpu_quota_limited"]:
        return "no limit" if account["gpu"]["limit"] else "-"
    seconds = account["gpu_quota_seconds"]
    return "?" if seconds is None else hours(seconds)


def _total_time(result: dict) -> str:
    totals = result["totals"]
    if totals["gpu_quota_seconds"] is not None:
        return _bound(totals) + hours(totals["gpu_quota_seconds"])
    return "?" if any(account["gpu_quota_limited"] for account in result["accounts"]) else "no limit"

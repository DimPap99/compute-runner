"""Finding, and on request deleting, what the runner left behind."""

from __future__ import annotations

from typing import Annotated

import typer

from ..cleanup import Cleanup
from .common import client, output
from .output import size, table


def cleanup(
    ctx: typer.Context,
    older_than: Annotated[
        float, typer.Option(min=0, help="Days since a job finished before its leftovers may go")
    ] = 7,
    account: Annotated[
        list[str] | None, typer.Option("--account", help="Only this account; repeatable")
    ] = None,
    local: Annotated[bool, typer.Option("--local/--no-local", help="Include the state directory")] = True,
    remote: Annotated[
        bool, typer.Option("--remote/--no-remote", help="Ask the accounts for their items")
    ] = True,
    include_snapshots: Annotated[
        bool, typer.Option("--include-snapshots", help="Also local snapshots that retry and continue need")
    ] = False,
    delete: Annotated[
        bool, typer.Option("--delete", help="Delete what may go; otherwise only report")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Delete without asking")] = False,
):
    """Report what this runner left behind and what may go; --delete removes it.

    Covers only runner-made items: launch notebooks, bundle datasets, SSH run folders and
    bundles, and the state directory's snapshots, staging folders, uploads and log copies. An
    item may go once every job using it finished before the cutoff with its outputs downloaded.
    Remote items this queue did not create are never deleted.
    """
    planner = Cleanup(
        client(ctx),
        older_than_days=older_than,
        include_snapshots=include_snapshots,
        accounts=account if remote else [],
        local=local,
    )
    items = planner.items()
    report = planner.report(items)
    reclaimable = report["totals"]["reclaimable"]
    if delete and reclaimable["count"]:
        if not yes:
            question = f"Delete {reclaimable['count']} items ({size(reclaimable['bytes'])})?"
            typer.confirm(question, abort=True, err=True)
        report["deleted"] = planner.delete(items)
    out = output(ctx)
    if out.json:
        out.emit(report)
    else:
        _show(out, report)


def _show(out, report: dict) -> None:
    rows = [
        [item["location"], item["kind"], item["name"], size(item.get("bytes")), item["reason"]]
        for item in report["items"]
    ]
    if rows:
        out.show(table(("Where", "Kind", "Name", "Size", "Why"), rows, wrap=("Name", "Why")))
    totals = report["totals"]
    reclaimable = totals["reclaimable"]
    out.line(
        f"{reclaimable['count']} items may go ({size(reclaimable['bytes'])}); "
        f"{totals['kept']['count']} are kept and {totals['unknown']['count']} were not made by this queue"
    )
    if "deleted" in report:
        deleted = report["deleted"]
        out.line(f"Deleted {deleted['deleted']} items ({size(deleted['bytes'])})")
        out.notes(f"{name}: {error}" for name, error in deleted["failed"].items())
    elif reclaimable["count"]:
        out.line("--delete removes them", style="dim")
    out.notes(f"{account}: {error}" for account, error in report.get("errors", {}).items())

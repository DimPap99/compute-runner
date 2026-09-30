"""Read-only views across accounts, built on the worker's own capacity rules.

Local by default: this queue's jobs and the worker's last discovery. Live views ask every
provider now instead (on Kaggle, a request per notebook run in the last 24 hours).
"""

from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .models import ACTIVE, Account, Pool
from .providers import safe_message, short
from .worker.capacity import POOLS, Capacity, Discovery, DiscoveryFile


class CapacityView:
    """Each account's slots, GPU time and the runs holding them, at one moment."""

    def __init__(self, client, *, live: bool = False, account: str | None = None):
        """account limits the view, and a live one's requests, to that account."""
        self.client = client
        self.config = client.config
        self.selected = self.config.account(account).id if account else None
        self.accounts = [item for item in self.config.accounts if self.selected in (None, item.id)]
        self.found = self._discover(live)
        self.now = time.time()
        self.jobs = client.store.list(ACTIVE)
        self.capacity = Capacity(self.config, self.found, self.jobs)

    def _discover(self, live: bool) -> dict[str, Discovery]:
        ids = [account.id for account in self.accounts]
        if not live:
            saved = DiscoveryFile(self.config.state_dir).load()
            return {account: saved.get(account) or Discovery() for account in ids}
        # Accounts answer independently, and a Kaggle account's discovery takes several requests.
        with ThreadPoolExecutor(max_workers=max(1, len(ids))) as pool:
            return dict(zip(ids, pool.map(self._ask, ids), strict=True))

    def _ask(self, account: str) -> Discovery:
        found = Discovery()
        try:
            found.record(self.client.provider(account).inventory(), time.time())
        except Exception as error:  # As in the worker's discovery: one broken account hides no other.
            found.error = safe_message(error)
        return found

    def checked(self, account: str) -> dict:
        """How old the account's last discovery is (None: never made), and why the latest one failed."""
        found = self.found[account]
        value = dict(checked_age_seconds=found.age(self.now))
        if found.error:
            value["error"] = short(found.error)
        return value

    # Accounts ---------------------------------------------------------------------------------

    def account_rows(self) -> list[dict]:
        """Each account's slots in use, their limit, and how many a new job could take now."""
        return [self._account_row(account) for account in self.accounts]

    def _account_row(self, account: Account) -> dict:
        found = self.found[account.id]
        used = self.capacity.used(account.id)
        row = dict(id=account.id, provider=account.provider)
        for pool in POOLS:
            limit, free = self.capacity.limit(account.id, pool), self.capacity.free(account.id, pool)
            row[pool] = dict(used=used[pool], limit=limit, free=free)
        # SSH machines have no GPU time limit; null elsewhere means not checked yet.
        row["gpu_quota_limited"] = account.provider != "ssh"
        row["gpu_quota_seconds"] = None if found.gpu_seconds is None else round(found.gpu_seconds)
        if found.gpu_refresh_at is not None:
            row["gpu_refresh_at"] = found.gpu_refresh_at
        if found.devices:
            row["devices"] = found.devices
        return row | self.checked(account.id)

    @staticmethod
    def totals(rows: list[dict]) -> dict:
        """Slots and GPU time over every account.

        free and gpu_quota_seconds add up the accounts checked successfully (None when there are
        none); complete says whether that is every account, so otherwise they are lower bounds.
        """
        known = [row for row in rows if row["gpu"]["free"] is not None]
        quotas = [row["gpu_quota_seconds"] for row in known if row["gpu_quota_limited"]]
        value = {
            pool: dict(
                used=sum(row[pool]["used"] for row in rows),
                limit=sum(row[pool]["limit"] for row in rows),
                free=sum(row[pool]["free"] for row in known) if known else None,
            )
            for pool in POOLS
        }
        return value | dict(
            gpu_quota_seconds=sum(quotas) if quotas else None, complete=len(known) == len(rows)
        )

    # Runs -------------------------------------------------------------------------------------

    def runs(self, resource: Pool | None = None) -> list[dict]:
        """Runs holding slots, in account preference order: this queue's first, then others found.

        A discovered run of unknown resource holds both pools, as the worker counts it.
        """
        ours = [
            self._our_run(job)
            for job in self.jobs
            if job.outstanding and self._shown(job.attempts[-1].account)
        ]
        taken = {(run["account"], run["ref"].lower()) for run in ours}
        others = [
            dict(account=account, ref=ref, resource=kind if kind in POOLS else "unknown")
            for account, found in self.found.items()
            for ref, kind in found.runs.items()
            if (account, ref) not in taken
        ]
        runs = [
            dict(run, provider=run["account"].partition(":")[0])
            for run in [*ours, *others]
            if resource is None or run["resource"] in (resource, "unknown")
        ]
        # Stable: this queue's runs stay oldest first. Accounts no longer configured go last.
        order = {account: index for index, account in enumerate(self.found)}
        return sorted(runs, key=lambda run: (order.get(run["account"], len(order)), "job_id" not in run))

    def _shown(self, account: str) -> bool:
        return self.selected in (None, account)

    def _our_run(self, job) -> dict:
        attempt = job.attempts[-1]
        run = dict(
            account=attempt.account,
            ref=attempt.ref,
            resource=job.pool,
            job_id=job.id,
            name=short(job.spec.name, 100),
            state=job.state,
            elapsed_seconds=max(0, round(self.now - attempt.started_at)),
        )
        extra = dict(accelerator=short(job.spec.accelerator, 100), url=job.url)
        return run | {key: item for key, item in extra.items() if item}

    @staticmethod
    def counts(runs: list[dict]) -> dict:
        found = Counter(run["resource"] for run in runs)
        return {kind: found[kind] for kind in (*POOLS, "unknown")}

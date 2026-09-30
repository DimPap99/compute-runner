"""Each account's slots: what holds them, what the worker last discovered, and who may start now.

The worker's admission rule lives here once. The worker asks obstacle() before a launch;
read views ask free(), so what they show is what the worker would do.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from ..models import Config, JobRecord, Pool
from ..providers import Inventory, safe_message
from ..store import atomic_json
from .context import Component

# Each account's last Discovery, written by the worker for local readers.
DISCOVERY = "accounts.json"
POOLS: tuple[Pool, ...] = ("cpu", "gpu")


def waiting_for_capacity(account: str, pool: Pool) -> str:
    return f"Waiting for {pool.upper()} capacity on {account}"


@dataclass
class Discovery:
    """What the worker last learned about one account's remote capacity (see providers.Inventory)."""

    runs: dict = field(default_factory=dict)
    checked_at: float | None = None
    error: str | None = None
    retry_at: float = 0
    gpu_seconds: float | None = None
    gpu_refresh_at: float | None = None
    devices: list = field(default_factory=list)

    @classmethod
    def from_json(cls, value) -> Discovery:
        """A saved discovery; fields it lacks, as older workers saved them, take their defaults."""
        if not isinstance(value, dict):
            return cls()
        names = {item.name for item in fields(cls)}
        return cls(**{name: item for name, item in value.items() if name in names})

    @property
    def known(self) -> bool:
        """Whether the account's runs are known: checked, and the latest check succeeded."""
        return self.checked_at is not None and not self.error

    @property
    def gpu_spent(self) -> bool:
        # None: the account has no GPU time limit.
        return self.gpu_seconds is not None and self.gpu_seconds <= 0

    def record(self, found: Inventory, now: float) -> None:
        self.runs, self.gpu_seconds = found.runs, found.gpu_seconds
        self.gpu_refresh_at, self.devices = found.gpu_refresh_at, found.devices
        self.checked_at, self.error = now, None

    def age(self, now: float) -> int | None:
        return None if self.checked_at is None else max(0, round(now - self.checked_at))


class DiscoveryFile:
    """accounts.json in the state directory: the worker's last Discovery of each account."""

    def __init__(self, state_dir: Path):
        self.path = state_dir / DISCOVERY

    def load(self) -> dict[str, Discovery]:
        try:
            saved = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return (
            {account: Discovery.from_json(value) for account, value in saved.items()}
            if isinstance(saved, dict)
            else {}
        )

    def save(self, found: dict[str, Discovery]) -> None:
        atomic_json(self.path, {account: asdict(item) for account, item in found.items()})


class Capacity:
    """Slots on each account: this queue's outstanding runs plus other runs discovered there."""

    def __init__(self, config: Config, discovery: dict[str, Discovery], jobs: list[JobRecord]):
        """jobs are this queue's active jobs; the outstanding ones hold slots."""
        self.config = config
        self.discovery = discovery
        self.jobs = jobs

    def _found(self, account: str) -> Discovery:
        return self.discovery.get(account) or Discovery()

    def limit(self, account: str, pool: Pool) -> int:
        return getattr(self.config.account(account), pool + "_limit")

    def used(self, account: str) -> dict[Pool, int]:
        """Runs holding the account's CPU and GPU slots. A run of unknown resource holds both."""
        ours = [job for job in self.jobs if job.outstanding and job.attempts[-1].account == account]
        refs = {job.remote_ref.lower() for job in ours}
        counts = {pool: sum(job.pool == pool for job in ours) for pool in POOLS}
        for ref, resource in self._found(account).runs.items():
            if ref.lower() not in refs:
                for pool in POOLS:
                    counts[pool] += resource in {pool, "unknown"}
        return counts

    def full(self, account: str, pool: Pool, *, reserved: int = 0) -> bool:
        """reserved counts jobs about to take slots there, such as ones preparing."""
        return self.used(account)[pool] + reserved >= self.limit(account, pool)

    def obstacle(self, account: str, pool: Pool, *, reserved: int = 0) -> str | None:
        """Why no new run can start on the account now; None if one can."""
        found = self._found(account)
        if not found.known:
            return f"Run discovery on {account} unavailable; waiting before new launches"
        if self.full(account, pool, reserved=reserved):
            return waiting_for_capacity(account, pool)
        if pool == "gpu" and found.gpu_spent:
            return f"Waiting for available GPU quota on {account}"
        return None

    def free(self, account: str, pool: Pool) -> int | None:
        """How many new runs could start on the account now; None while its runs are unknown."""
        found = self._found(account)
        if not found.known:
            return None
        if pool == "gpu" and found.gpu_spent:
            return 0
        return max(0, self.limit(account, pool) - self.used(account)[pool])


class Discoverer(Component):
    """Reads each account's inventory at most every discovery_seconds, backing off after failures."""

    def __init__(self, context):
        super().__init__(context)
        self.found: dict[str, Discovery] = defaultdict(Discovery)

    def refresh(self, account: str) -> Discovery:
        found = self.found[account]
        now = time.time()
        due = found.checked_at is None or now - found.checked_at >= self.config.discovery_seconds
        if now < found.retry_at or not due:
            return found
        try:
            self.heartbeat(stage=f"discovering runs on {account}")
            # Quota is read with the runs, not per launch: failover weighs every account each cycle.
            found.record(self.provider(account).inventory(), now)
        except Exception as error:
            found.error = safe_message(error)
            found.retry_at = now + self.config.retry_seconds
        return found

    def forget(self, account: str, ref: str) -> None:
        """A run of this queue ended, so it no longer holds a slot there."""
        self.found[account].runs.pop(ref.lower(), None)

    def save(self) -> None:
        DiscoveryFile(self.config.state_dir).save(self.found)

"""Retention: with retention_days set, the worker deletes old local leftovers and job records daily."""

from __future__ import annotations

import logging
import time

from ..cleanup import Cleanup
from .context import Component

logger = logging.getLogger(__name__)
DAY = 86400


class Retention(Component):
    """What `cleanup --older-than DAYS --delete` would delete locally, once a day.

    Remote items stay the user's to delete, so a job's record goes only once a check of every
    account finds none left (see cleanup). Snapshots go once no remaining job uses them.
    """

    def run_due(self) -> None:
        days = self.config.retention_days
        now = time.time()
        if days is None or now - float(self.store.meta("retention_at", 0)) < DAY:
            return
        # Saved first: a pass that fails is tried again tomorrow, not every cycle.
        self.store.set_meta("retention_at", now)
        self.heartbeat(stage="retention")
        planner = Cleanup(self.context, older_than_days=days)
        result = planner.delete([item for item in planner.items() if item.location == "local"])
        if result["deleted"] or result["failed"]:
            logger.info(
                "Retention deleted %d items (%d bytes); %d failed: %s",
                result["deleted"],
                result["bytes"],
                len(result["failed"]),
                result["failed"],
            )

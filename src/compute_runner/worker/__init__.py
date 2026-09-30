"""The worker: the one scheduler that launches, follows and collects every account's jobs.

capacity: each account's slots and discovered runs. datasets: attaching or copying datasets.
preparation: uploads before launch. submission: launching attempts. polling: following runs.
dispatch: placing pending jobs, with failover. downloads: collecting outputs. loop: the Worker.
"""

from .capacity import DISCOVERY, Capacity, Discovery, DiscoveryFile, waiting_for_capacity
from .downloads import collect_outputs
from .loop import Worker

__all__ = [
    "DISCOVERY",
    "Capacity",
    "Discovery",
    "DiscoveryFile",
    "Worker",
    "collect_outputs",
    "waiting_for_capacity",
]

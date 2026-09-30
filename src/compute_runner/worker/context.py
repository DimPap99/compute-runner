"""What every part of the worker shares: the configuration, the providers and the queue."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..models import Config
from ..providers import Provider
from ..store import Store


@dataclass(frozen=True)
class WorkerContext:
    config: Config
    # Maps an account ID to its Provider.
    provider: Callable[[str], Provider]
    store: Store


class Component:
    """One responsibility of the worker, working on the shared context."""

    def __init__(self, context: WorkerContext):
        self.context = context
        self.config = context.config
        self.provider = context.provider
        self.store = context.store

    def heartbeat(self, **stage):
        """Tell health probes what the worker is doing, such as stage="uploading"."""
        self.store.heartbeat(state="running", **stage)

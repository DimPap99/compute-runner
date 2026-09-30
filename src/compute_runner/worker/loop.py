"""The worker loop: each cycle polls runs, dispatches pending jobs and schedules downloads."""

from __future__ import annotations

import contextlib
import logging
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from ..models import ACTIVE, PENDING, Config
from ..results import publish
from ..store import Store
from .capacity import Discoverer
from .context import WorkerContext
from .datasets import DatasetAccess
from .dispatch import Dispatcher
from .downloads import DownloadScheduler
from .polling import Poller
from .preparation import Preparer
from .submission import Submitter

logger = logging.getLogger(__name__)


class Worker:
    """The single scheduler of a state directory; its lock keeps a second one out."""

    def __init__(self, config: Config, provider, store=None):
        """provider maps an account ID to its Provider."""
        self.config = config
        self.store = store or Store(config.state_dir)
        self.stop_event = threading.Event()
        context = WorkerContext(config, provider, self.store)
        self.discoverer = Discoverer(context)
        datasets = DatasetAccess(context)
        self.poller = Poller(context, self.discoverer)
        self.dispatcher = Dispatcher(
            context,
            discoverer=self.discoverer,
            datasets=datasets,
            preparer=Preparer(context, datasets),
            submitter=Submitter(context),
            stop=self.stop_event,
        )
        self.downloads = DownloadScheduler(context)

    @property
    def discovery(self):
        """What the worker last learned about each account's capacity, by account ID."""
        return self.discoverer.found

    def tick(self):
        """One complete, locked cycle; also useful for cron and deterministic tests."""
        with self.store.worker_lock():
            try:
                self._tick()
            finally:
                self.store.heartbeat(state="stopped", mode="once")

    def run(self):
        with self.store.worker_lock(), self._stop_on_signals():
            try:
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="kgr-download") as pool:
                    self.downloads.pool = pool
                    while not self.stop_event.is_set():
                        self._cycle()
                        self.stop_event.wait(self.config.poll_seconds)
            finally:
                self.downloads.pool = None
                self.store.heartbeat(state="stopped")

    def _cycle(self):
        try:
            self._tick()
        except Exception:
            logger.exception("Worker cycle failed; saved work will be retried")
            self.store.heartbeat(state="running", error="Worker cycle failed; see service logs")

    @contextlib.contextmanager
    def _stop_on_signals(self):
        """SIGINT and SIGTERM end the loop after the current step; handlers are restored afterwards."""
        previous = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, lambda *_: self.stop_event.set())
        try:
            yield
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def _tick(self):
        self.store.heartbeat(state="running", stage="monitoring")
        now = time.time()
        for job in self.store.list(ACTIVE):
            if job.outstanding and job.next_action_at <= now:
                self.poller.poll(job)
        pending = self.store.list(PENDING)
        if pending and not self.stop_event.is_set():
            self.dispatcher.dispatch(pending)
        self.downloads.schedule()
        publish(self.store)
        self.discoverer.save()
        self.store.heartbeat(state="running", stage="idle")

"""Transactional shared state. Only the worker performs remote mutations."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import logging
import os
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .models import MOVABLE, BatchRecord, Config, JobRecord
from .paths import application_dir, run_folder, run_number

logger = logging.getLogger(__name__)
LOW_DISK_FRACTION = 0.10
DISK_WRITE_HEADROOM = 16 * 1024 * 1024
_LOW_DISK_DEVICES = set()
_LOW_DISK_LOCK = threading.Lock()


def config_path() -> Path:
    return application_dir("CONFIG") / "config.json"


def try_lock(stream) -> bool:
    """Take an exclusive lock without waiting; closing the file releases it."""
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def load_config() -> Config:
    path = config_path()
    return Config.model_validate_json(path.read_text()) if path.exists() else Config()


def _disk_status(path: Path):
    usage = shutil.disk_usage(path.parent)
    device = path.parent.stat().st_dev
    low = usage.total > 0 and usage.free / usage.total < LOW_DISK_FRACTION
    with _LOW_DISK_LOCK:
        warned = device in _LOW_DISK_DEVICES
        if low:
            _LOW_DISK_DEVICES.add(device)
        else:
            _LOW_DISK_DEVICES.discard(device)
    if low and not warned:
        logger.warning(
            "Low disk space for %s: %.1f%% free (%s of %s bytes)",
            path,
            usage.free / usage.total * 100,
            usage.free,
            usage.total,
        )
    return usage


def _require_disk_space(path: Path, required: int):
    usage = _disk_status(path)
    needed = required + DISK_WRITE_HEADROOM
    if usage.free < needed:
        raise OSError(
            errno.ENOSPC,
            f"Insufficient disk space: need {needed} bytes including write headroom, have {usage.free}",
            str(path),
        )
    return usage


def atomic_write(path: Path, data, *, check_space=False, expected_bytes=None):
    """Privately replace path with bytes or byte chunks. A failure midway keeps the old file."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if expected_bytes is not None:
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise ValueError("expected_bytes must be a nonnegative integer")
        _require_disk_space(path, expected_bytes)
    elif check_space:
        _require_disk_space(path, 0)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            os.chmod(temporary, 0o600)
            for chunk in [data] if isinstance(data, bytes) else data:
                if check_space:
                    _require_disk_space(path, len(chunk))
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if check_space or expected_bytes is not None:
            _disk_status(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value):
    atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False).encode())


class Store:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = root / "queue.sqlite3"
        with self.connection() as db:
            self._migrate(db)
        os.chmod(self.db, 0o600)

    @staticmethod
    def _migrate(db):
        """Create the schema, or upgrade an older one; a newer schema is refused."""
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT OR IGNORE INTO meta VALUES ('schema_version', '1');
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, created REAL NOT NULL, state TEXT NOT NULL, record TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS job_state ON jobs(state, created);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                timestamp REAL NOT NULL, state TEXT NOT NULL, detail TEXT
            );
        """)
        version = db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        # 3: job records name an account; 4: run folders, output receipts and dataset copies.
        # Older versions must not open newer queues.
        if version not in {"1", "2", "3", "4"}:
            raise RuntimeError(f"Unsupported state schema {version}; do not open with this version")
        db.executescript("""
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS batches (
                id TEXT PRIMARY KEY, created REAL NOT NULL,
                request_key TEXT UNIQUE, fingerprint TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS batch_jobs (
                batch_id TEXT NOT NULL, job_id TEXT NOT NULL UNIQUE, position INTEGER NOT NULL,
                PRIMARY KEY (batch_id, position)
            );
            CREATE INDEX IF NOT EXISTS event_job ON events(job_id, id);
            CREATE TABLE IF NOT EXISTS runs (
                experiment TEXT NOT NULL, number INTEGER NOT NULL, job_id TEXT NOT NULL UNIQUE,
                PRIMARY KEY (experiment, number)
            );
            CREATE TABLE IF NOT EXISTS outputs (
                job_id TEXT NOT NULL, name TEXT NOT NULL, path TEXT NOT NULL,
                bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY (job_id, name)
            );
            CREATE TABLE IF NOT EXISTS dataset_copies (
                source TEXT PRIMARY KEY, digest TEXT NOT NULL, bytes INTEGER NOT NULL,
                created REAL NOT NULL
            );
            UPDATE meta SET value='4' WHERE key='schema_version';
            COMMIT;
        """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.db, timeout=30)
        try:
            db.execute("PRAGMA synchronous=FULL")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _event(db, job, detail):
        db.execute(
            "INSERT INTO events(job_id,timestamp,state,detail) VALUES(?,?,?,?)",
            (job.id, time.time(), job.state, detail),
        )

    @staticmethod
    def _batch(db, batch_id, *, replayed=False):
        row = db.execute("SELECT created FROM batches WHERE id=?", (batch_id,)).fetchone()
        if not row:
            raise KeyError(f"No batch {batch_id}")
        rows = db.execute(
            "SELECT j.record FROM batch_jobs b JOIN jobs j ON j.id=b.job_id "
            "WHERE b.batch_id=? ORDER BY b.position",
            (batch_id,),
        ).fetchall()
        return BatchRecord(
            id=batch_id,
            created_at=row[0],
            replayed=replayed,
            jobs=[JobRecord.model_validate_json(row[0]) for row in rows],
        )

    @classmethod
    def _request(cls, db, request_key, fingerprint):
        if request_key is None:
            return None
        row = db.execute("SELECT id,fingerprint FROM batches WHERE request_key=?", (request_key,)).fetchone()
        if row:
            if row[1] != fingerprint:
                raise ValueError("Request key already belongs to a different request; use a new key")
            return cls._batch(db, row[0], replayed=True)
        return None

    def request(self, request_key, fingerprint):
        with self.connection() as db:
            db.execute("BEGIN")
            return self._request(db, request_key, fingerprint)

    def add_batch(self, batch: BatchRecord, *, request_key, fingerprint, experiments=None):
        """Commit a batch; jobs listed in experiments ({job ID: folder}) get their run folder now.

        Snapshots happen before this transaction. No job is visible to the worker
        until the entire batch and its idempotency receipt have committed.
        """
        created = []
        try:
            with self.connection() as db:
                db.execute("BEGIN IMMEDIATE")
                previous = self._request(db, request_key, fingerprint)
                if previous is not None:
                    return previous
                db.execute(
                    "INSERT INTO batches VALUES (?,?,?,?)",
                    (batch.id, batch.created_at, request_key, fingerprint),
                )
                for position, job in enumerate(batch.jobs):
                    if experiments and job.id in experiments:
                        self._number(db, job, experiments[job.id], created)
                    db.execute(
                        "INSERT INTO jobs VALUES (?,?,?,?)",
                        (job.id, job.created_at, job.state, job.model_dump_json()),
                    )
                    self._event(db, job, "submitted locally")
                    db.execute("INSERT INTO batch_jobs VALUES (?,?,?)", (batch.id, job.id, position))
        except BaseException:
            # The batch did not commit; neither do the run folders it reserved, which are still empty.
            for folder in created:
                with contextlib.suppress(OSError):
                    folder.rmdir()
            raise
        return batch

    @staticmethod
    def _number(db, job, experiment: Path, created: list):
        """Give a new job the next run number in its experiment folder, and create that run's folder.

        Inside the batch transaction, so submitters to one queue never share a number. Folders
        on disk count too, and the folder is created under a lock in the experiment folder, so
        queues of other state directories writing to the same experiment cannot share one either.
        The new folder, still empty, is appended to created as soon as it exists.
        """
        experiment.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (experiment / ".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            row = db.execute("SELECT MAX(number) FROM runs WHERE experiment=?", (str(experiment),)).fetchone()
            seen = [run_number(path.name) for path in experiment.iterdir()]
            job.run = max([row[0] or 0, *(number for number in seen if number is not None)]) + 1
            job.result_dir = experiment / run_folder(job.run, job.created_at)
            job.result_dir.mkdir(mode=0o700)
            created.append(job.result_dir)
        db.execute("INSERT INTO runs VALUES (?,?,?)", (str(experiment), job.run, job.id))

    def experiment(self, folder: Path) -> list[JobRecord]:
        """The jobs numbered in an experiment folder, by run number."""
        with self.connection() as db:
            rows = db.execute(
                "SELECT j.record FROM runs r JOIN jobs j ON j.id=r.job_id "
                "WHERE r.experiment=? ORDER BY r.number",
                (str(folder),),
            ).fetchall()
        return [JobRecord.model_validate_json(row[0]) for row in rows]

    def output_receipts(self, job_id) -> dict:
        """Outputs already saved for a job: {remote name: {path in its run folder, bytes, sha256}}."""
        with self.connection() as db:
            rows = db.execute(
                "SELECT name,path,bytes,sha256 FROM outputs WHERE job_id=?", (job_id,)
            ).fetchall()
        return {name: dict(path=path, bytes=size, sha256=digest) for name, path, size, digest in rows}

    def record_output(self, job_id, name, *, path, bytes, sha256):
        with self.connection() as db:
            db.execute(
                "INSERT OR REPLACE INTO outputs VALUES (?,?,?,?,?)", (job_id, name, path, bytes, sha256)
            )

    def dataset_copy(self, source) -> dict | None:
        """The local bundle made from a provider dataset version, if one was made."""
        with self.connection() as db:
            row = db.execute("SELECT digest,bytes FROM dataset_copies WHERE source=?", (source,)).fetchone()
        return dict(digest=row[0], bytes=row[1]) if row else None

    def save_dataset_copy(self, source, *, digest, bytes):
        with self.connection() as db:
            db.execute(
                "INSERT OR REPLACE INTO dataset_copies VALUES (?,?,?,?)", (source, digest, bytes, time.time())
            )

    def meta(self, key, default=None):
        with self.connection() as db:
            row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key, value):
        with self.connection() as db:
            db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, str(value)))

    @contextmanager
    def download_claim(self, job_id):
        """Yield whether this process may collect the job's outputs now; one collector per job.

        A lock file in the state directory, released if the process dies. A shared database
        could hold the same claim as a leased row.
        """
        folder = self.root / "jobs" / job_id
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (folder / "download.lock").open("a+") as lock:
            yield try_lock(lock)

    def batch(self, batch_id):
        with self.connection() as db:
            db.execute("BEGIN")
            return self._batch(db, batch_id)

    def resolve_id(self, prefix):
        return self.resolve_ids([prefix])[0]

    def resolve_ids(self, prefixes):
        """Resolve unambiguous job ID prefixes using one connection."""
        if not all(prefixes):
            raise ValueError("A job ID is required")
        result = []
        with self.connection() as db:
            for prefix in prefixes:
                rows = db.execute(
                    "SELECT id FROM jobs WHERE substr(id,1,?)=? LIMIT 2", (len(prefix), prefix)
                ).fetchall()
                if len(rows) != 1:
                    raise ValueError(f"Expected one matching job for {prefix}; found {len(rows)}")
                result.append(rows[0][0])
        return result

    @staticmethod
    def _filters(db, batch_id=None, job_ids=None, states=None, account=None, pool=None):
        clauses, args = [], []
        if account is not None:
            clauses.append("lower(json_extract(j.record,'$.account'))=lower(?)")
            args.append(account)
        if pool is not None:
            clauses.append("json_extract(j.record,'$.spec.gpu')=?")
            args.append(int(pool == "gpu"))
        if batch_id is not None:
            if not db.execute("SELECT 1 FROM batches WHERE id=?", (batch_id,)).fetchone():
                raise KeyError(f"No batch {batch_id}")
            clauses.append("j.id IN (SELECT job_id FROM batch_jobs WHERE batch_id=?)")
            args.append(batch_id)
        if job_ids is not None:
            for job_id in job_ids:
                if not db.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone():
                    raise KeyError(f"No job {job_id}")
            clauses.append("j.id IN (" + ",".join("?" for _ in job_ids) + ")")
            args.extend(job_ids)
        if states is not None:
            clauses.append("j.state IN (" + ",".join("?" for _ in states) + ")")
            args.extend(states)
        return " AND ".join(clauses) or "1", args

    def page(
        self,
        *,
        batch_id=None,
        job_ids=None,
        states=None,
        account=None,
        pool=None,
        limit=20,
        offset=0,
        newest_first=False,
    ):
        """Counts by state over the selection, and one page of it; limit=-1 reads every job.

        A batch lists in submission order; otherwise by creation time, oldest first unless
        newest_first. account and pool ("cpu" or "gpu") narrow the selection.
        """
        with self.connection() as db:
            db.execute("BEGIN")
            where, args = self._filters(db, batch_id, job_ids, states, account, pool)
            counts = dict(
                db.execute(
                    f"SELECT j.state,COUNT(*) FROM jobs j WHERE {where} GROUP BY j.state", args
                ).fetchall()
            )
            if batch_id:
                order = "(SELECT position FROM batch_jobs b WHERE b.job_id=j.id)"
            else:
                order = "j.created DESC,j.id DESC" if newest_first else "j.created,j.id"
            rows = db.execute(
                f"SELECT j.record FROM jobs j WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?",
                [*args, limit, offset],
            ).fetchall()
        return counts, [JobRecord.model_validate_json(row[0]) for row in rows]

    def changes(self, *, after=0, batch_id=None, limit=20):
        """Coalesced current states, not historical records. Cursor and rows share a snapshot."""
        with self.connection() as db:
            db.execute("BEGIN")
            where, args = self._filters(db, batch_id)
            high = db.execute("SELECT COALESCE(MAX(id),0) FROM events").fetchone()[0]
            if after > high:
                raise ValueError(
                    "Cursor is ahead of this queue; use the original state directory or reset to 0"
                )
            rows = db.execute(
                f"SELECT MAX(e.id),j.record FROM events e JOIN jobs j ON j.id=e.job_id "
                f"WHERE e.id>? AND {where} GROUP BY j.id ORDER BY MAX(e.id) LIMIT ?",
                [after, *args, limit + 1],
            ).fetchall()
            more = len(rows) > limit
            rows = rows[:limit]
            cursor = rows[-1][0] if more else high
        return cursor, more, [JobRecord.model_validate_json(row[1]) for row in rows]

    def get(self, job_id: str) -> JobRecord:
        with self.connection() as db:
            row = db.execute("SELECT record FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise KeyError(f"No job {job_id}")
        return JobRecord.model_validate_json(row[0])

    def list(self, states=None, *, account=None, pool=None) -> list[JobRecord]:
        return self.page(states=states, account=account, pool=pool, limit=-1)[1]

    def counts_by_account(self) -> dict[str, dict[str, int]]:
        """How many jobs each account has in each state: {account: {state: count}}."""
        with self.connection() as db:
            rows = db.execute(
                "SELECT json_extract(record,'$.account'),state,COUNT(*) FROM jobs GROUP BY 1,2"
            ).fetchall()
        found = {}
        for account, state, count in rows:
            found.setdefault(account, {})[state] = count
        return found

    def attention(self, limit: int) -> tuple[int, list[JobRecord]]:
        """Jobs waiting on someone, newest first: (how many, at most limit of them).

        Blocked jobs, jobs needing attention, failed downloads, and pending jobs another account
        could start (the failover policy asks first).
        """
        where = (
            "state IN ('blocked','needs_attention') OR json_extract(record,'$.download_state')='error' "
            "OR (state IN ('queued','preparing') AND json_extract(record,'$.suggested_account') IS NOT NULL)"
        )
        with self.connection() as db:
            db.execute("BEGIN")
            total = db.execute(f"SELECT COUNT(*) FROM jobs WHERE {where}").fetchone()[0]
            rows = db.execute(
                f"SELECT record FROM jobs WHERE {where} ORDER BY created DESC,id DESC LIMIT ?", (limit,)
            ).fetchall()
        return total, [JobRecord.model_validate_json(row[0]) for row in rows]

    def batch_positions(self, job_ids) -> dict[str, tuple[str, int]]:
        """Each listed job's batch and its place there: {job ID: (batch ID, position)}."""
        if not job_ids:
            return {}
        marks = ",".join("?" for _ in job_ids)
        with self.connection() as db:
            rows = db.execute(
                f"SELECT job_id,batch_id,position FROM batch_jobs WHERE job_id IN ({marks})", list(job_ids)
            ).fetchall()
        return {job_id: (batch_id, position) for job_id, batch_id, position in rows}

    def update(self, job_id: str, *, expected=None, **changes) -> JobRecord | None:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT record FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                raise KeyError(job_id)
            job = JobRecord.model_validate_json(row[0])
            if expected is not None and job.state not in expected:
                return None
            previous = self._observable(job)
            raw = job.model_dump(mode="json")
            raw.update(changes, updated_at=time.time())
            job = JobRecord.model_validate(raw)
            db.execute(
                "UPDATE jobs SET state=?,record=? WHERE id=?", (job.state, job.model_dump_json(), job.id)
            )
            if previous != self._observable(job):
                self._event(db, job, job.error or job.wait_reason or job.download_state)
            return job

    def place(self, job_id: str, account: str, reason: str, *, transfer=None) -> JobRecord | None:
        """Put a job that has no remote run on another account; None if it changed meanwhile.

        Requeueing also stops a preparation in progress, whose updates expect "preparing".
        transfer, when given, sets whether its datasets may be copied there.
        """
        changes = {} if transfer is None else {"transfer": transfer}
        return self.update(
            job_id,
            expected=MOVABLE,
            account=account,
            state="queued",
            # Uploads and attached datasets belong to the previous account; copies stay cached locally.
            upload_refs={},
            transfers={},
            suggested_account=None,
            suggested_transfer=False,
            error=None,
            wait_reason=reason,
            next_action_at=0,
            **changes,
        )

    @staticmethod
    def _observable(job):
        return (
            job.state,
            job.account,
            job.suggested_account,
            job.suggested_transfer,
            job.remote_state,
            job.wait_reason,
            job.error,
            job.download_state,
            job.download_error,
            job.attempts,
            job.finished_at,
        )

    @contextmanager
    def worker_lock(self):
        with (self.root / "worker.lock").open("a+") as lock:
            # Health probes hold this lock for an instant; do not mistake one for a worker.
            for _ in range(20):
                if try_lock(lock):
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError("A worker already owns this state directory")
            yield

    def heartbeat(self, **extra):
        atomic_json(self.root / "worker.json", dict(pid=os.getpid(), timestamp=time.time(), **extra))

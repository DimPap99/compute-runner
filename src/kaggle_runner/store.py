"""Transactional shared state. Only the worker performs remote mutations."""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .models import BatchRecord, Config, JobRecord


def config_path() -> Path:
    return (
        Path(
            os.environ.get(
                "KGR_CONFIG_DIR",
                str(Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "kaggle-runner"),
            )
        )
        / "config.json"
    )


def load_config() -> Config:
    path = config_path()
    return Config.model_validate_json(path.read_text()) if path.exists() else Config()


def atomic_json(path: Path, value):
    import uuid

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            os.chmod(temporary, 0o600)
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Store:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = root / "queue.sqlite3"
        with self.connection() as db:
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
            if version not in {"1", "2"}:
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
                UPDATE meta SET value='2' WHERE key='schema_version';
                COMMIT;
            """)
        os.chmod(self.db, 0o600)

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

    def add(self, job: JobRecord):
        with self.connection() as db:
            self._insert_job(db, job)
        return job

    @staticmethod
    def _insert_job(db, job):
        db.execute(
            "INSERT INTO jobs VALUES (?,?,?,?)",
            (job.id, job.created_at, job.state, job.model_dump_json()),
        )
        db.execute(
            "INSERT INTO events(job_id,timestamp,state,detail) VALUES(?,?,?,?)",
            (job.id, time.time(), job.state, "submitted locally"),
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

    def add_batch(self, batch: BatchRecord, *, request_key, fingerprint):
        # Snapshots happen before this transaction. No job is visible to the worker
        # until the entire batch and its idempotency receipt have committed.
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
                self._insert_job(db, job)
                db.execute("INSERT INTO batch_jobs VALUES (?,?,?)", (batch.id, job.id, position))
        return batch

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
    def _filters(db, batch_id=None, job_ids=None, states=None):
        clauses, args = [], []
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

    def page(self, *, batch_id=None, job_ids=None, states=None, limit=20, offset=0):
        with self.connection() as db:
            db.execute("BEGIN")
            where, args = self._filters(db, batch_id, job_ids, states)
            counts = dict(
                db.execute(
                    f"SELECT j.state,COUNT(*) FROM jobs j WHERE {where} GROUP BY j.state", args
                ).fetchall()
            )
            order = (
                "(SELECT position FROM batch_jobs b WHERE b.job_id=j.id)" if batch_id else "j.created,j.id"
            )
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

    def list(self, states=None) -> list[JobRecord]:
        with self.connection() as db:
            rows = db.execute("SELECT record FROM jobs ORDER BY created,id").fetchall()
        jobs = [JobRecord.model_validate_json(r[0]) for r in rows]
        return [j for j in jobs if states is None or j.state in states]

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
                db.execute(
                    "INSERT INTO events(job_id,timestamp,state,detail) VALUES (?,?,?,?)",
                    (job.id, time.time(), job.state, job.error or job.wait_reason or job.download_state),
                )
            return job

    @staticmethod
    def _observable(job):
        return (
            job.state,
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
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("A worker already owns this state directory") from None
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def heartbeat(self, **extra):
        atomic_json(self.root / "worker.json", dict(pid=os.getpid(), timestamp=time.time(), **extra))

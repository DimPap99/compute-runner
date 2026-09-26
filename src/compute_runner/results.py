"""Experiment folders: where a run's outputs land, and the records written beside them.

Submission fixes each job's run folder, RESULTS/EXPERIMENT/NNN_TIMESTAMP, before anything runs.
The queue database is the source of truth: the worker downloads outputs into the folder and
writes job.json and the experiment's runs.md from the database. Agents read these files; they
never create, move or rename them.
"""

from __future__ import annotations

import fnmatch
import json
import logging
from datetime import datetime
from pathlib import Path

from .models import Config, JobRecord, JobSpec
from .paths import experiment_folder
from .runtime import file_digest, safe_relative
from .security import redact_secrets
from .store import atomic_json, atomic_write

logger = logging.getLogger(__name__)
RUN_RECORD = "job.json"
INDEX = "runs.md"


def results_root(spec: JobSpec, config: Config) -> Path:
    """The workload's own results_dir, else the configured one, else results/ beside its code."""
    if spec.results_dir is not None:
        return spec.results_dir.expanduser().absolute()
    if config.results_dir is not None:
        return config.results_dir
    source = spec.source.expanduser().absolute()
    return (source if source.is_dir() else source.parent) / "results"


def experiment_dir(spec: JobSpec, config: Config) -> Path:
    return results_root(spec, config) / experiment_folder(spec.name)


def outputs_dir(job: JobRecord) -> Path:
    """The local copy of what the workload wrote to KGR_OUTPUT_DIR."""
    return job.result_dir / "outputs" / ("outputs" if job.run is None else "")


def source_copies(job):
    """Match output names of the project copy the runtime made; saved bundles already hold them.

    Files the workload creates under the project folder are still downloaded, except
    __pycache__ bytecode. In-place edits of snapshot files are not collected.
    """
    names = {f"project/{name}" for name in job.snapshot["source"]["files"]}
    return lambda name: name in names or (name.startswith("project/") and "__pycache__" in name.split("/"))


class OutputSink:
    """Chooses which remote outputs to fetch and where they go; providers only move bytes.

    Remote names are relative to the run's working directory. What the workload wrote to
    KGR_OUTPUT_DIR (outputs/NAME) lands in ROOT/outputs/NAME, anything else it left there in
    ROOT/working/NAME, and the log in ROOT/run.log. receipts holds files already saved and
    verified, so an interrupted collection resumes; this base class keeps them in memory.
    """

    def __init__(self, root: Path, *, patterns=None, skip=None, strict=False):
        self.root = Path(root)
        self.patterns, self.skip, self.strict = patterns, skip, strict
        self.receipts: dict[str, dict] = {}

    def place(self, name: str) -> str:
        return name if name.startswith("outputs/") else "working/" + name

    def target(self, name: str) -> Path | None:
        """The file to write for a remote output, or None to leave it: filtered or already saved."""
        name = safe_relative(name)
        if self.skip is not None and self.skip(name):
            return None
        if self.patterns is not None and not any(fnmatch.fnmatchcase(name, p) for p in self.patterns):
            return None
        relative = self.place(name)
        folder = self.root / relative.split("/")[0]
        target = self.root / relative
        if not target.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Output resolves outside the destination")
        receipt = self.receipts.get(name)
        if receipt and target.is_file() and file_digest(target) == receipt["sha256"]:
            return None
        return target

    def log(self, text: str) -> None:
        data = redact_secrets(text, strict=self.strict).encode()
        atomic_write(self.root / "run.log", data, check_space=True, expected_bytes=len(data))

    def saved(self, name: str, target: Path, sha256: str) -> None:
        self.receipts[name] = dict(
            path=target.relative_to(self.root).as_posix(), bytes=target.stat().st_size, sha256=sha256
        )


class JobOutputs(OutputSink):
    """One job's outputs, with receipts kept in the queue database."""

    def __init__(self, store, job: JobRecord, *, strict=False):
        super().__init__(
            job.result_dir, patterns=job.spec.output_patterns, skip=source_copies(job), strict=strict
        )
        self.store, self.job = store, job
        self.receipts = store.output_receipts(job.id)

    def place(self, name):
        # Jobs saved before run folders kept every remote file under outputs/.
        return "outputs/" + name if self.job.run is None else super().place(name)

    def saved(self, name, target, sha256):
        super().saved(name, target, sha256)
        self.store.record_output(self.job.id, name, **self.receipts[name])


def listed_outputs(job: JobRecord):
    """(root, [(path relative to root, bytes)]) for a job's downloaded files; logs and records excluded."""
    root = job.result_dir / "outputs" if job.run is None else job.result_dir
    folders = [root] if job.run is None else [root / "outputs", root / "working"]
    files = sorted(
        (path.relative_to(root).as_posix(), path.stat().st_size)
        for folder in folders
        if folder.is_dir()
        for path in folder.rglob("*")
        if path.is_file() and not (path.name.startswith(".") and path.name.endswith(".tmp"))
    )
    return root, files


def verified_checkpoint(folder: Path) -> str:
    """The checkpoint file named by folder/latest.json, after checking its SHA-256.

    latest.json is the manifest written by the skill's checkpointing helper.
    """
    manifest = folder / "latest.json"
    try:
        metadata = json.loads(manifest.read_text())
        name, expected = metadata["file"], metadata["sha256"]
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError(f"No readable checkpoint manifest at {manifest}") from None
    if not isinstance(name, str) or Path(name).name != name or name == "latest.json":
        raise ValueError(f"Unsafe checkpoint name in {manifest}")
    checkpoint = folder / name
    if not checkpoint.is_file() or file_digest(checkpoint) != expected:
        raise ValueError(f"Checkpoint {checkpoint} is missing or does not match latest.json")
    return name


def _time(value):
    return datetime.fromtimestamp(value).astimezone().isoformat(timespec="seconds") if value else None


def run_record(job: JobRecord) -> dict:
    """job.json: what the run is, how it was started, and where it stands."""
    spec = job.spec
    value = dict(
        job_id=job.id,
        experiment=spec.name,
        run=job.run,
        folder=job.result_dir.name,
        params=spec.params,
        state=job.state,
        error=job.error,
        account=job.account,
        url=job.url,
        submitted_at=_time(job.created_at),
        finished_at=_time(job.finished_at),
        parent_id=job.parent_id,
        command={
            key: item
            for key, item in [
                ("entrypoint", job.snapshot.get("entrypoint")),
                ("module", job.snapshot.get("module")),
                ("args", spec.command_args()),
            ]
            if item is not None
        },
        source=dict(path=str(spec.source), digest=job.snapshot["source"]["digest"]),
        inputs={alias: str(value) for alias, value in spec.inputs.items()},
        datasets=spec.datasets,
        copied_datasets=job.transfers,
        resources=dict(
            gpu=spec.gpu,
            accelerator=spec.accelerator,
            internet=spec.internet,
            timeout_seconds=spec.timeout_seconds,
        ),
        # Names only: values stay in the queue.
        env=sorted(spec.env),
        downloads=job.download_state,
        download_error=job.download_error,
        attempts=[
            dict(number=a.number, account=a.account, url=a.url, state=a.state, started_at=_time(a.started_at))
            for a in job.attempts
        ],
    )
    return {key: item for key, item in value.items() if item not in (None, {}, [])}


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_index(folder: Path, jobs: list[JobRecord]) -> None:
    """runs.md: one row per run of the experiment, so its params and state are visible at a glance."""
    runs = {job.id: job.result_dir.name.split("_")[0] for job in jobs}
    lines = [
        f"# {folder.name}",
        "",
        "Runs of this experiment. Compute Runner rewrites this file from its queue; edits are overwritten.",
        "",
        "| Run | State | Params | Outputs | Account | From | Job |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for job in jobs:
        params = ", ".join(f"{name}={value}" for name, value in job.spec.params.items())
        parent = runs.get(job.parent_id) or (job.parent_id or "")[:12]
        row = [job.result_dir.name, job.state, params, job.download_state, job.account, parent, job.id[:12]]
        lines.append("| " + " | ".join(_cell(item) for item in row) + " |")
    atomic_write(folder / INDEX, ("\n".join(lines) + "\n").encode())


def publish(store) -> None:
    """Rewrite job.json and runs.md for runs whose state changed since the last pass.

    Follows the queue's change cursor, kept in the database, so each change is written once
    and a restart resumes where it stopped. A folder that cannot be written is logged and
    rewritten on the job's next change.
    """
    cursor = int(store.meta("views_cursor", 0))
    folders = set()
    while True:
        try:
            cursor, more, jobs = store.changes(after=cursor, limit=100)
        except ValueError:  # The queue was replaced; rewrite everything.
            cursor, more, jobs = store.changes(after=0, limit=100)
        for job in jobs:
            if job.run is None:
                continue
            folders.add(job.result_dir.parent)
            try:
                atomic_json(job.result_dir / RUN_RECORD, run_record(job))
            except OSError as error:
                logger.warning("Could not write %s: %s", job.result_dir / RUN_RECORD, error)
        if not more:
            break
    for folder in folders:
        try:
            write_index(folder, store.experiment(folder))
        except OSError as error:
            logger.warning("Could not write %s: %s", folder / INDEX, error)
    store.set_meta("views_cursor", cursor)

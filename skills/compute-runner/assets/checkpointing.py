"""Small checkpoint helper to copy and adapt inside a resumable workload.

The module is framework-neutral: pass serializer callbacks such as ``torch.save``
and ``lambda path: torch.load(path, map_location="cpu", weights_only=False)``.
The caller remains responsible for putting every required training state object in
the saved payload and restoring it to the live model, optimizer, and data loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


SCHEMA_VERSION = 1


class CheckpointError(RuntimeError):
    """A requested checkpoint is missing, corrupt, or incompatible."""


@dataclass(frozen=True)
class RestoredCheckpoint:
    state: Any
    path: Path
    metadata: dict[str, Any]


class CheckpointCadence:
    """Decide when a safe training boundary is due for a checkpoint."""

    def __init__(
        self,
        mode: str,
        every: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if mode not in {"minutes", "epochs"}:
            raise ValueError("checkpoint mode must be 'minutes' or 'epochs'")
        if isinstance(every, bool) or not isinstance(every, (int, float)) or every <= 0:
            raise ValueError("checkpoint interval must be a positive number")
        if mode == "epochs" and not float(every).is_integer():
            raise ValueError("an epoch checkpoint interval must be a positive integer")
        self.mode = mode
        self.every = int(every) if mode == "epochs" else float(every)
        self._clock = clock
        self._last_saved_at = clock()

    def due(self, *, completed_epochs: int | None = None) -> bool:
        if self.mode == "minutes":
            return self._clock() - self._last_saved_at >= self.every * 60
        if completed_epochs is None:
            raise ValueError("completed_epochs is required for epoch cadence")
        return completed_epochs > 0 and completed_epochs % self.every == 0

    def mark_saved(self) -> None:
        self._last_saved_at = self._clock()


class CheckpointManager:
    """Write atomic checkpoints and discover a verified checkpoint to restore."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        *,
        resume_input: str | os.PathLike[str] | None = None,
        compatibility: Mapping[str, Any] | None = None,
        suffix: str = ".pt",
    ) -> None:
        if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
            raise ValueError("checkpoint suffix must be a simple extension such as '.pt'")
        self.output_dir = Path(output_dir)
        configured_input = resume_input if resume_input is not None else os.environ.get("KGR_INPUT_RESUME")
        self.resume_input = Path(configured_input) if configured_input else None
        self.compatibility = dict(compatibility) if compatibility is not None else None
        self.compatibility_sha256 = _json_digest(self.compatibility) if compatibility is not None else None
        self.suffix = suffix

    def restore(
        self,
        mode: str,
        *,
        load: Callable[[Path], Any],
    ) -> RestoredCheckpoint | None:
        """Restore `auto`, `required`, `never`, or an explicit file/manifest path."""

        if mode == "never":
            return None
        if mode in {"auto", "required"}:
            source = self.resume_input
            if source is None:
                if mode == "auto":
                    return None
                raise CheckpointError("resume is required but KGR_INPUT_RESUME is not available")
        else:
            source = Path(mode)

        checkpoint, metadata = self._resolve(source)
        expected = metadata.get("sha256")
        if expected and _sha256(checkpoint) != expected:
            raise CheckpointError(f"checkpoint checksum does not match latest.json: {checkpoint}")
        self._check_compatibility(metadata)

        try:
            state = load(checkpoint)
        except Exception as error:
            raise CheckpointError(f"could not load checkpoint {checkpoint}: {error}") from error
        step = metadata.get("global_step", "unknown")
        print(f"RESUMED_FROM step={step} path={checkpoint}", flush=True)
        return RestoredCheckpoint(state=state, path=checkpoint, metadata=metadata)

    def save(
        self,
        state: Any,
        *,
        dump: Callable[[Any, Path], None],
        global_step: int,
        completed_epochs: int,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Atomically save a numbered checkpoint, then publish `latest.json`."""

        if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
            raise ValueError("global_step must be a non-negative integer")
        if (
            isinstance(completed_epochs, bool)
            or not isinstance(completed_epochs, int)
            or completed_epochs < 0
        ):
            raise ValueError("completed_epochs must be a non-negative integer")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"checkpoint-step-{global_step:012d}{self.suffix}"
        checkpoint = self.output_dir / filename
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{filename}.", suffix=".tmp", dir=self.output_dir
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            dump(state, temporary)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise CheckpointError("checkpoint serializer did not produce a non-empty file")
            _fsync_file(temporary)
            os.replace(temporary, checkpoint)
            _fsync_directory(self.output_dir)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "file": filename,
            "sha256": _sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "global_step": global_step,
            "completed_epochs": completed_epochs,
            "saved_at": time.time(),
        }
        if self.compatibility is not None:
            metadata["compatibility"] = self.compatibility
            metadata["compatibility_sha256"] = self.compatibility_sha256
        if extra_metadata is not None:
            metadata["extra"] = dict(extra_metadata)
        _atomic_json(self.output_dir / "latest.json", metadata)
        print(f"CHECKPOINT_SAVED step={global_step} path={checkpoint}", flush=True)
        return checkpoint

    def _resolve(self, source: Path) -> tuple[Path, dict[str, Any]]:
        if not source.exists():
            raise CheckpointError(f"resume path does not exist: {source}")
        if source.is_file() and source.name != "latest.json":
            sibling_manifest = source.parent / "latest.json"
            if sibling_manifest.is_file():
                metadata = _read_manifest(sibling_manifest)
                if metadata["file"] == source.name:
                    return source, metadata
            return source, {}

        if source.is_file():
            manifest = source
        else:
            direct = source / "latest.json"
            if direct.is_file():
                manifest = direct
            else:
                manifests = list(source.rglob("latest.json"))
                if len(manifests) != 1:
                    raise CheckpointError(f"expected one latest.json under {source}, found {len(manifests)}")
                manifest = manifests[0]
        metadata = _read_manifest(manifest)
        checkpoint = manifest.parent / metadata["file"]
        if not checkpoint.is_file():
            raise CheckpointError(f"checkpoint named by latest.json is missing: {checkpoint}")
        return checkpoint, metadata

    def _check_compatibility(self, metadata: Mapping[str, Any]) -> None:
        if self.compatibility_sha256 is None:
            return
        saved = metadata.get("compatibility_sha256")
        if saved is None:
            raise CheckpointError("checkpoint has no compatibility metadata")
        if saved != self.compatibility_sha256:
            raise CheckpointError("checkpoint is incompatible with the requested training configuration")


def add_checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the standard resumability interface to an ArgumentParser."""

    parser.add_argument("--resume", default="auto", help="auto|required|never|PATH")
    parser.add_argument("--checkpoint-mode", choices=("minutes", "epochs"), required=True)
    parser.add_argument("--checkpoint-every", type=float, required=True)


def cadence_from_args(args: argparse.Namespace) -> CheckpointCadence:
    return CheckpointCadence(args.checkpoint_mode, args.checkpoint_every)


def capture_torch_rng_state() -> dict[str, Any]:
    """Return RNG state suitable for inclusion in a torch-serialized payload."""

    import torch

    state: dict[str, Any] = {"python": random.getstate(), "torch_cpu": torch.get_rng_state()}
    try:
        import numpy
    except ImportError:
        pass
    else:
        state["numpy"] = numpy.random.get_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_torch_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG state previously returned by `capture_torch_rng_state`."""

    import torch

    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if "numpy" in state:
        import numpy

        numpy.random.set_state(state["numpy"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise CheckpointError(f"could not read checkpoint manifest {path}: {error}") from error
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointError(f"unsupported checkpoint manifest: {path}")
    filename = metadata.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise CheckpointError(f"unsafe checkpoint filename in manifest: {path}")
    checksum = metadata.get("sha256")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise CheckpointError(f"invalid checkpoint checksum in manifest: {path}")
    return metadata


def _json_digest(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    except (TypeError, ValueError) as error:
        raise ValueError(f"compatibility metadata must be JSON-serializable: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    data = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

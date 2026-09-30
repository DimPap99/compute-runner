"""Deterministic, credential-excluding snapshots; no remote calls."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import nbformat
import pathspec

from .models import JobSpec
from .runtime import MANIFEST, json_digest, manifest_records, safe_relative, verify_files
from .security import detected_secret, secret_filename
from .store import atomic_json

PROTECTED = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".kaggle",
    ".ssh",
    ".aws",
    ".azure",
    ".docker",
    ".gnupg",
    ".kube",
    ".password-store",
    "node_modules",
}


def _protected(path: Path) -> bool:
    """Version control, environments, caches and credentials never enter a bundle."""
    return any(part.casefold() in PROTECTED for part in path.parts) or secret_filename(path.name)


class SourceScan:
    """The files of a source or input to snapshot: what lies below it, less ignored and protected files.

    exclude and the ignore files hold gitignore patterns. skip lists folders left out wherever
    they appear inside the source, such as its results folder. Links are refused, not followed.
    """

    def __init__(self, source: Path, exclude, *, ignore_files=(".gitignore", ".kgrignore"), skip=()):
        self.skipped = {Path(os.path.realpath(folder)) for folder in skip}
        source = source.expanduser().absolute()
        if source.is_symlink():
            raise ValueError(f"Symlink source is not supported: {source}")
        self.source = source.resolve(strict=True)
        self.root = self.source if self.source.is_dir() else self.source.parent
        patterns = [*exclude, *self._ignore_patterns(ignore_files)]
        self.ignored = pathspec.PathSpec.from_lines("gitwildmatch", patterns)

    def _ignore_patterns(self, names) -> list[str]:
        patterns = []
        for name in names:
            file = self.root / name
            if file.is_file() and not file.is_symlink():
                patterns.extend(file.read_text().splitlines())
        return patterns

    def files(self) -> list[Path]:
        """The selected files, relative to root, in sorted order."""
        candidates = self._walk() if self.source.is_dir() else [Path(self.source.name)]
        selected = [relative for relative in sorted(candidates) if self._selected(relative)]
        if not selected:
            raise ValueError(f"No uploadable files in {self.source}")
        return selected

    def _walk(self) -> list[Path]:
        candidates = []
        for current, dirs, files in os.walk(self.root, followlinks=False):
            relative = Path(current).relative_to(self.root)
            dirs[:] = [name for name in sorted(dirs) if self._entered(relative / name)]
            candidates.extend(relative / name for name in sorted(files))
        return candidates

    def _entered(self, relative: Path) -> bool:
        if _protected(relative) or self.ignored.match_file(relative.as_posix() + "/"):
            return False
        if self.root / relative in self.skipped:
            return False
        if (self.root / relative).is_symlink():
            raise ValueError(f"Symlinks are not supported: {relative}")
        return True

    def _selected(self, relative: Path) -> bool:
        if _protected(relative) or self.ignored.match_file(relative.as_posix()):
            return False
        full = self.root / relative
        if full.is_symlink():
            raise ValueError(f"Symlinks are not supported: {relative}")
        if not full.is_file():
            raise ValueError(f"Only regular files are supported: {relative}")
        if relative.as_posix() == MANIFEST:
            raise ValueError(f"{MANIFEST} is reserved; rename it or exclude it")
        return True


def scan(
    source: Path, exclude, *, ignore_files=(".gitignore", ".kgrignore"), skip=()
) -> tuple[Path, list[Path]]:
    """(root, files relative to it) of a source or input to snapshot; see SourceScan."""
    found = SourceScan(source, exclude, ignore_files=ignore_files, skip=skip)
    return found.root, found.files()


def clean_notebook(path: Path, *, python=False) -> bytes:
    # nbformat raises jsonschema's ValidationError, not a ValueError, while reading and validating.
    try:
        notebook = nbformat.read(path, as_version=4)
        language = notebook.metadata.get("kernelspec", {}).get("language", "python")
        if python and language.lower() != "python":
            raise ValueError("Only Python notebooks are supported")
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        notebook.metadata.pop("widgets", None)
        nbformat.validate(notebook)
    except nbformat.ValidationError as error:
        raise ValueError(f"Invalid notebook {path.name}: {error.message}") from None
    return nbformat.writes(notebook).encode()


def describe(spec: JobSpec, *, skip=()) -> dict:
    """Plan the snapshot of a spec whose inputs are all local paths."""
    root, files = scan(spec.source, spec.exclude, skip=skip)
    source = spec.source.expanduser().resolve()
    entrypoint, kind = _entry(spec, source, root, files)
    if spec.requirements and Path(safe_relative(spec.requirements)) not in files:
        raise ValueError("requirements must identify an included project file")
    return dict(
        root=str(root),
        kind=kind,
        single_file=source.is_file(),
        entrypoint=entrypoint,
        module=spec.module,
        files=[p.as_posix() for p in files],
        bytes=sum((root / p).stat().st_size for p in files),
        inputs={alias: _describe_input(path, skip) for alias, path in spec.inputs.items()},
        gpu=spec.gpu,
        accelerator=spec.accelerator,
        internet=spec.internet,
        timeout_seconds=spec.timeout_seconds,
        private=True,
        datasets=spec.datasets,
        environment_keys=list(spec.env),
    )


def _entry(spec: JobSpec, source: Path, root: Path, files: list[Path]) -> tuple[str | None, str]:
    """(entrypoint, kind) of what runs: a file source itself, a module, or an entrypoint in the folder."""
    entrypoint = spec.entrypoint
    if source.is_file():
        if entrypoint or spec.module:
            raise ValueError("For a file source, omit entrypoint/module; its filename is the entrypoint")
        entrypoint = source.name
    if spec.module:
        module_path = spec.module.replace(".", "/")
        if not any(Path(p) in files for p in (module_path + ".py", module_path + "/__main__.py")):
            raise ValueError(f"Module {spec.module} is absent or excluded from the snapshot")
        return entrypoint, "script"
    if not entrypoint:
        raise ValueError("Folder sources require entrypoint or module")
    entrypoint = safe_relative(entrypoint)
    if Path(entrypoint) not in files:
        raise ValueError(f"Entrypoint is absent or excluded: {entrypoint}")
    if Path(entrypoint).suffix not in {".py", ".ipynb"}:
        raise ValueError("Entrypoint must be a .py or .ipynb file")
    if entrypoint.endswith(".ipynb"):
        clean_notebook(root / entrypoint, python=True)
        return entrypoint, "notebook"
    return entrypoint, "script"


def _describe_input(path: Path, skip) -> dict:
    # Data folders often .gitignore exactly the files they exist to carry; only .kgrignore applies.
    root, files = scan(path, [], ignore_files=(".kgrignore",), skip=skip)
    return dict(
        root=str(root),
        files=[p.as_posix() for p in files],
        bytes=sum((root / p).stat().st_size for p in files),
    )


def plain_files(root: Path, *, allow_bundle_manifest=False) -> list[str]:
    """Files for a dataset copy, optionally unwrapping a verified runner bundle.

    A runner-created dataset already contains our manifest. Verify its complete
    inventory and hashes before regenerating that metadata in the new snapshot.
    Arbitrary user inputs still reject the reserved name by default.
    """
    files = []
    for current, dirs, names in os.walk(root):
        for name in [*dirs, *names]:
            if (Path(current) / name).is_symlink():
                raise ValueError(f"Symlinks are not supported: {name}")
        files += [(Path(current) / name).relative_to(root).as_posix() for name in names]
    if MANIFEST in files:
        if not allow_bundle_manifest:
            raise ValueError(f"{MANIFEST} is reserved")
        try:
            metadata = json.loads((root / MANIFEST).read_text())
            records = manifest_records(metadata, metadata["digest"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid existing bundle manifest") from error
        if set(files) != set(records) | {MANIFEST} or MANIFEST in records:
            raise ValueError("Existing bundle inventory differs from manifest")
        verify_files(root, records)
        files = list(records)
    if not files:
        raise ValueError("The dataset has no files")
    return sorted(files)


def snapshot_bundle(root: Path, files: list, bundle_root: Path, *, notebooks=False, screen=True) -> dict:
    """Copy an inventory of files into an immutable, content-addressed bundle; see BundleWriter."""
    return BundleWriter(bundle_root, notebooks=notebooks, screen=screen).write(root, files)


class BundleWriter:
    """Writes immutable, content-addressed bundles: bundle_root/DIGEST/{files/, payload.zip}.

    screen rejects files containing recognizable credentials; copies of provider datasets,
    which are already on the provider, are not screened. notebooks clears notebooks' outputs.
    """

    def __init__(self, bundle_root: Path, *, notebooks=False, screen=True):
        self.bundle_root = bundle_root
        self.notebooks = notebooks
        self.screen = screen

    def write(self, root: Path, files: list) -> dict:
        self.bundle_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryDirectory(prefix=".building-", dir=self.bundle_root) as temporary:
            stage = Path(temporary)
            records = {
                relative.as_posix(): self._copy(root / relative, stage / "files" / relative, relative)
                for relative in map(Path, files)
            }
            digest = json_digest(records)
            atomic_json(stage / "files" / MANIFEST, dict(schema_version=1, digest=digest, files=records))
            self._archive(stage, records)
            self._publish(stage, digest)
        return dict(digest=digest, bytes=sum(record["size"] for record in records.values()), files=records)

    def _copy(self, original: Path, destination: Path, relative: Path) -> dict:
        """Copy one file, screened; it must not change meanwhile. Returns its manifest record."""
        before = original.stat()
        destination.parent.mkdir(parents=True, exist_ok=True)
        cleaned = self._cleaned(original)
        with destination.open("wb") as output:
            if cleaned is not None:
                self._check(cleaned, relative)
                output.write(cleaned)
                digest, size = hashlib.sha256(cleaned), len(cleaned)
            else:
                digest, size = self._stream(original, output, relative)
            output.flush()
            os.fsync(output.fileno())
        after = original.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise ValueError(f"File changed during snapshot; submit again: {original}")
        return {"sha256": digest.hexdigest(), "size": size}

    def _cleaned(self, original: Path) -> bytes | None:
        if not (self.notebooks and original.suffix == ".ipynb"):
            return None
        try:
            return clean_notebook(original)
        except ValueError:
            return None  # Only the entrypoint runs, and describe() validated it; copy others unchanged.

    def _stream(self, original: Path, output, relative: Path):
        digest, size, tail = hashlib.sha256(), 0, b""
        with original.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                # A credential split between two chunks is still found in the previous chunk's tail.
                self._check(tail + chunk, relative)
                digest.update(chunk)
                output.write(chunk)
                size += len(chunk)
                tail = chunk[-512:]
        return digest, size

    def _check(self, data: bytes, relative: Path) -> None:
        if self.screen and (kind := detected_secret(data)):
            raise ValueError(f"Detected {kind} in {relative}; remove or exclude that credential")

    @staticmethod
    def _archive(stage: Path, records: dict) -> None:
        """payload.zip: every file and the manifest, byte-for-byte reproducible."""
        payload = stage / "files"
        with zipfile.ZipFile(stage / "payload.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in [*sorted(records), MANIFEST]:
                info = zipfile.ZipInfo(relative, date_time=(2020, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                with (payload / relative).open("rb") as source_file, archive.open(info, "w") as target:
                    shutil.copyfileobj(source_file, target, 1024 * 1024)

    def _publish(self, stage: Path, digest: str) -> None:
        """Move the finished bundle into place; one that already exists is kept, as it is identical."""
        target = self.bundle_root / digest
        if target.exists():
            return
        try:
            os.rename(stage, target)
        except OSError:
            if not target.exists():
                raise


def snapshot(spec: JobSpec, state_dir: Path, *, skip=()) -> dict:
    plan = describe(spec, skip=skip)
    parts = [plan, *plan["inputs"].values()]
    if any((Path(part["root"]) / name).is_relative_to(state_dir) for part in parts for name in part["files"]):
        raise ValueError("The state directory must not be inside a source or input folder")

    def save(part, **options):
        return snapshot_bundle(Path(part["root"]), part["files"], state_dir / "bundles", **options)

    return {key: plan[key] for key in ("kind", "single_file", "entrypoint", "module")} | {
        "source": save(plan, notebooks=True),
        "inputs": {alias: save(part) for alias, part in plan["inputs"].items()},
    }

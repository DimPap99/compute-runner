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
from .runtime import MANIFEST, _manifest, _verify, json_digest, safe_relative
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


def inventory(
    source: Path, exclude: list[str], ignore_files=(".gitignore", ".kgrignore"), skip=()
) -> tuple[Path, list[Path]]:
    """skip lists folders left out wherever they appear inside source, such as its results folder."""
    source = source.expanduser().absolute()
    skipped = {Path(os.path.realpath(folder)) for folder in skip}
    if source.is_symlink():
        raise ValueError(f"Symlink source is not supported: {source}")
    source = source.resolve(strict=True)
    root = source if source.is_dir() else source.parent
    patterns = list(exclude)
    for name in ignore_files:
        file = root / name
        if file.is_file() and not file.is_symlink():
            patterns.extend(file.read_text().splitlines())
    ignored = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
    selected = []

    def protect(path):
        return any(p.casefold() in PROTECTED for p in path.parts) or secret_filename(path.name)

    candidates = []
    if source.is_dir():
        for current, dirs, files in os.walk(root, followlinks=False):
            relative = Path(current).relative_to(root)
            allowed = []
            for name in sorted(dirs):
                rel = relative / name
                if protect(rel) or ignored.match_file(rel.as_posix() + "/") or root / rel in skipped:
                    continue
                if (root / rel).is_symlink():
                    raise ValueError(f"Symlinks are not supported: {rel}")
                allowed.append(name)
            dirs[:] = allowed
            candidates.extend(relative / name for name in sorted(files))
    else:
        candidates = [Path(source.name)]
    for rel in sorted(candidates):
        if protect(rel) or ignored.match_file(rel.as_posix()):
            continue
        full = root / rel
        if full.is_symlink():
            raise ValueError(f"Symlinks are not supported: {rel}")
        if not full.is_file():
            raise ValueError(f"Only regular files are supported: {rel}")
        if rel.as_posix() == MANIFEST:
            raise ValueError(f"{MANIFEST} is reserved; rename it or exclude it")
        selected.append(rel)
    if not selected:
        raise ValueError(f"No uploadable files in {source}")
    return root, selected


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
    root, files = inventory(spec.source, spec.exclude, skip=skip)
    source = spec.source.expanduser().resolve()
    entrypoint = spec.entrypoint
    if source.is_file():
        if entrypoint or spec.module:
            raise ValueError("For a file source, omit entrypoint/module; its filename is the entrypoint")
        entrypoint = source.name
    if spec.module:
        module_path = spec.module.replace(".", "/")
        if not any(Path(p) in files for p in (module_path + ".py", module_path + "/__main__.py")):
            raise ValueError(f"Module {spec.module} is absent or excluded from the snapshot")
        kind = "script"
    elif entrypoint:
        entrypoint = safe_relative(entrypoint)
        if Path(entrypoint) not in files:
            raise ValueError(f"Entrypoint is absent or excluded: {entrypoint}")
        if Path(entrypoint).suffix not in {".py", ".ipynb"}:
            raise ValueError("Entrypoint must be a .py or .ipynb file")
        kind = "notebook" if entrypoint.endswith(".ipynb") else "script"
        if kind == "notebook":
            clean_notebook(root / entrypoint, python=True)
    else:
        raise ValueError("Folder sources require entrypoint or module")
    if spec.requirements and Path(safe_relative(spec.requirements)) not in files:
        raise ValueError("requirements must identify an included project file")
    inputs = {}
    for alias, source_path in spec.inputs.items():
        # Data folders often .gitignore exactly the files they exist to carry; only .kgrignore applies.
        data_root, data_files = inventory(source_path, [], ignore_files=(".kgrignore",), skip=skip)
        inputs[alias] = dict(
            root=str(data_root),
            files=[p.as_posix() for p in data_files],
            bytes=sum((data_root / p).stat().st_size for p in data_files),
        )
    return dict(
        root=str(root),
        kind=kind,
        single_file=source.is_file(),
        entrypoint=entrypoint,
        module=spec.module,
        files=[p.as_posix() for p in files],
        bytes=sum((root / p).stat().st_size for p in files),
        inputs=inputs,
        gpu=spec.gpu,
        accelerator=spec.accelerator,
        internet=spec.internet,
        timeout_seconds=spec.timeout_seconds,
        private=True,
        datasets=spec.datasets,
        environment_keys=list(spec.env),
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
            records = _manifest(metadata, metadata["digest"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid existing bundle manifest") from error
        if set(files) != set(records) | {MANIFEST} or MANIFEST in records:
            raise ValueError("Existing bundle inventory differs from manifest")
        _verify(root, records)
        files = list(records)
    if not files:
        raise ValueError("The dataset has no files")
    return sorted(files)


def snapshot_bundle(root: Path, files: list, bundle_root: Path, *, notebooks=False, screen=True) -> dict:
    """Copy an inventory of files into an immutable, content-addressed bundle.

    screen rejects files containing recognizable credentials; copies of provider datasets,
    which are already on the provider, are not screened.
    """
    bundle_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix=".building-", dir=bundle_root) as temporary:
        stage = Path(temporary)
        payload = stage / "files"
        records = {}
        for relative in map(Path, files):
            original = root / relative
            before = original.stat()
            destination = payload / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            data = None
            if notebooks and original.suffix == ".ipynb":
                try:
                    data = clean_notebook(original)
                except ValueError:
                    pass  # Only the entrypoint runs, and describe() validated it; copy others unchanged.
            with destination.open("wb") as output:
                if data is not None:
                    if screen and (kind := detected_secret(data)):
                        raise ValueError(f"Detected {kind} in {relative}; remove or exclude that credential")
                    output.write(data)
                    digest.update(data)
                    size = len(data)
                else:
                    tail = b""
                    with original.open("rb") as input_file:
                        while chunk := input_file.read(1024 * 1024):
                            if screen and (kind := detected_secret(tail + chunk)):
                                raise ValueError(
                                    f"Detected {kind} in {relative}; remove or exclude that credential"
                                )
                            digest.update(chunk)
                            output.write(chunk)
                            size += len(chunk)
                            tail = chunk[-512:]
                output.flush()
                os.fsync(output.fileno())
            after = original.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ino,
            ):
                raise ValueError(f"File changed during snapshot; submit again: {original}")
            records[relative.as_posix()] = {"sha256": digest.hexdigest(), "size": size}
        fingerprint = json_digest(records)
        manifest = dict(schema_version=1, digest=fingerprint, files=records)
        atomic_json(payload / MANIFEST, manifest)
        with zipfile.ZipFile(stage / "payload.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in [*sorted(records), MANIFEST]:
                info = zipfile.ZipInfo(relative, date_time=(2020, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                with (payload / relative).open("rb") as source_file, archive.open(info, "w") as target:
                    shutil.copyfileobj(source_file, target, 1024 * 1024)
        target = bundle_root / fingerprint
        if not target.exists():
            try:
                os.rename(stage, target)
            except OSError:
                if not target.exists():
                    raise
        return dict(digest=fingerprint, bytes=sum(r["size"] for r in records.values()), files=records)


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

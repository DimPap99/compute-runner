"""Deterministic, credential-excluding snapshots; no Kaggle calls."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import nbformat
import pathspec

from .models import JobSpec
from .store import atomic_json

MANIFEST = "kgr-manifest.json"
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
    ".gnupg",
    "node_modules",
}
SECRET_NAMES = {
    ".env",
    "kaggle.json",
    "access_token",
    "credentials",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
}


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in value:
        raise ValueError(f"Path must stay inside the project: {value}")
    return path.as_posix()


def inventory(source: Path, exclude: list[str]) -> tuple[Path, list[Path]]:
    source = source.expanduser().absolute()
    if source.is_symlink():
        raise ValueError(f"Symlink source is not supported: {source}")
    source = source.resolve(strict=True)
    root = source if source.is_dir() else source.parent
    patterns = list(exclude)
    for name in (".gitignore", ".kgrignore"):
        file = root / name
        if file.is_file() and not file.is_symlink():
            patterns.extend(file.read_text().splitlines())
    ignored = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
    selected = []

    def protect(path):
        return (
            any(p in PROTECTED for p in path.parts)
            or path.name in SECRET_NAMES
            or path.name.startswith(".env.")
            or path.suffix in {".pem", ".key", ".p12"}
        )

    candidates = []
    if source.is_dir():
        for current, dirs, files in os.walk(root, followlinks=False):
            relative = Path(current).relative_to(root)
            allowed = []
            for name in sorted(dirs):
                rel = relative / name
                if protect(rel) or ignored.match_file(rel.as_posix() + "/"):
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


def clean_notebook(path: Path) -> bytes:
    notebook = nbformat.read(path, as_version=4)
    language = notebook.metadata.get("kernelspec", {}).get("language", "python")
    if language.lower() != "python":
        raise ValueError("Only Python notebooks are supported")
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    notebook.metadata.pop("widgets", None)
    nbformat.validate(notebook)
    return nbformat.writes(notebook).encode()


def describe(spec: JobSpec) -> dict:
    root, files = inventory(spec.source, spec.exclude)
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
            clean_notebook(root / entrypoint)
    else:
        raise ValueError("Folder sources require entrypoint or module")
    if spec.requirements and Path(safe_relative(spec.requirements)) not in files:
        raise ValueError("requirements must identify an included project file")
    inputs = {}
    for alias, source_path in spec.inputs.items():
        data_root, data_files = inventory(source_path, [])
        inputs[alias] = dict(
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


def snapshot_bundle(source: Path, exclude: list[str], bundle_root: Path, *, notebooks=False) -> dict:
    root, files = inventory(source, exclude)
    bundle_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix=".building-", dir=bundle_root) as temporary:
        stage = Path(temporary)
        payload = stage / "files"
        records = {}
        for relative in files:
            original = root / relative
            before = original.stat()
            destination = payload / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with destination.open("wb") as output:
                if notebooks and original.suffix == ".ipynb":
                    data = clean_notebook(original)
                    output.write(data)
                    digest.update(data)
                    size = len(data)
                else:
                    with original.open("rb") as input_file:
                        while chunk := input_file.read(1024 * 1024):
                            digest.update(chunk)
                            output.write(chunk)
                            size += len(chunk)
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
        serialized = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        fingerprint = hashlib.sha256(serialized).hexdigest()
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


def snapshot(spec: JobSpec, state_dir: Path) -> dict:
    plan = describe(spec)
    root = spec.source.expanduser().resolve()
    if root.is_dir() and state_dir.is_relative_to(root):
        raise ValueError("The state directory must not be inside a source folder")
    source = snapshot_bundle(spec.source, spec.exclude, state_dir / "bundles", notebooks=True)
    inputs = {}
    for alias, path in spec.inputs.items():
        if path.expanduser().resolve().is_dir() and state_dir.is_relative_to(path.expanduser().resolve()):
            raise ValueError("The state directory must not be inside an input folder")
        inputs[alias] = snapshot_bundle(path, [], state_dir / "bundles")
    return {key: plan[key] for key in ("kind", "single_file", "entrypoint", "module")} | {
        "source": source,
        "inputs": inputs,
    }

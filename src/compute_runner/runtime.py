"""Standard-library-only bootstrap copied into every workload's launch package.

Runs on the provider's machine (Python 3.9 or newer). Paths in its configuration may be relative
to the directory it starts in.
"""

import base64
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

MANIFEST = "kgr-manifest.json"


def safe_relative(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name or not p.parts:
        raise ValueError("Path must stay inside the project: " + name)
    return p.as_posix()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tree_size(path):
    """(bytes, latest change) of a file, or of every file below a folder; links are not followed."""
    stat = os.lstat(path)
    size, latest = (0 if os.path.isdir(path) else stat.st_size), stat.st_mtime
    if os.path.isdir(path) and not os.path.islink(path):
        for current, _dirs, files in os.walk(path):
            for name in files:
                try:
                    found = os.lstat(os.path.join(current, name))
                except OSError:  # Gone meanwhile.
                    continue
                size += found.st_size
                latest = max(latest, found.st_mtime)
    return size, latest


def manifest_records(data, digest):
    """The file records of a bundle manifest, after checking them against the bundle's digest."""
    records = data["files"]
    if json_digest(records) != digest or data.get("digest") != digest or data.get("schema_version") != 1:
        raise ValueError("Bundle manifest hash mismatch")
    for name in records:
        safe_relative(name)
    return records


def verify_files(root, records):
    """Check each recorded file: a regular file inside root, with its recorded size and SHA-256."""
    for name, record in records.items():
        file = root / name
        if (
            file.is_symlink()
            or not file.is_file()
            or not file.resolve().is_relative_to(root.resolve())
            or file.stat().st_size != record["size"]
        ):
            raise ValueError("Bundle file invalid: " + name)
        if file_digest(file) != record["sha256"]:
            raise ValueError("Bundle checksum mismatch: " + name)


def unpack_bundle(archive_path, digest, target):
    """Extract a bundle archive into target, refusing members its manifest does not list."""
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read(MANIFEST))
        records = manifest_records(manifest, digest)
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(records) | {MANIFEST}:
            raise ValueError("Unexpected or duplicate archive members")
        for info in archive.infolist():
            safe_relative(info.filename)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Symlinks are forbidden in bundles")
            if info.filename == MANIFEST:
                continue
            if info.file_size != records[info.filename]["size"]:
                raise ValueError("Archive size differs from manifest")
            output = target / info.filename
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, output.open("wb") as sink:
                shutil.copyfileobj(source, sink, 1024 * 1024)
    verify_files(target, records)
    return target


def _mounts(ref, input_root):
    """Where Kaggle may have mounted a dataset. It lowercases the path, whatever the reference's casing."""
    owner, slug = ref.lower().split("/")[:2]
    return [input_root / "datasets" / owner / slug, input_root / slug]


def _find_bundle(ref, digest, *, input_root=Path("/kaggle/input")):
    roots = _mounts(ref, input_root)
    candidates = []
    for root in roots:
        if not root.is_dir():
            continue
        for manifest_file in root.rglob(MANIFEST):
            try:
                records = manifest_records(json.loads(manifest_file.read_text()), digest)
            except (KeyError, ValueError):
                continue
            verify_files(manifest_file.parent, records)
            candidates.append(("directory", manifest_file.parent, records))
        if not candidates:
            for archive_file in root.rglob("payload.zip"):
                with zipfile.ZipFile(archive_file) as archive:
                    records = manifest_records(json.loads(archive.read(MANIFEST)), digest)
                candidates.append(("archive", archive_file, records))
    # Resolve duplicate paths if two mount conventions alias the same location.
    candidates = list({str(item[1].resolve()): item for item in candidates}.values())
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one verified bundle for {ref}; found {len(candidates)}")
    return candidates[0]


def _local_bundle(path, digest):
    """A bundle the provider already unpacked and verified on this machine."""
    root = Path(os.path.abspath(path))
    return "directory", root, manifest_records(json.loads((root / MANIFEST).read_text()), digest)


def _find_dataset(ref, *, input_root=Path("/kaggle/input")):
    """Where Kaggle mounted an attached dataset, under either of its mount conventions."""
    for root in _mounts(ref, input_root):
        if root.is_dir():
            return root
    raise FileNotFoundError(f"Dataset {ref} is not attached; this account may not be able to read it")


def bootstrap(config):
    """Restore the project and its inputs, set the workload's environment, and enter the project."""
    _announce_session(config["job_id"])
    project = Path(os.path.abspath(config.get("working_root", "/kaggle/working"))) / "project"
    project.mkdir(parents=True, exist_ok=False)
    _restore_source(config, project)
    inputs = {alias: _resolve_input(bundle) for alias, bundle in config["inputs"].items()}
    _export_environment(config, inputs, project.parent / "outputs")
    os.chdir(project)
    sys.path.insert(0, str(project))
    if config.get("requirements"):
        _install_requirements(project / safe_relative(config["requirements"]))
    sys.argv = [config.get("entrypoint") or config["module"], *config["args"]]
    print("KGR workload", config["job_id"], "inputs", json.dumps(inputs), flush=True)
    return project


def _announce_session(job_id):
    """Print Kaggle's session ID, which its public API never returns; the runner reads it to cancel."""
    session = re.search(r"-(\d+)-\w+$", os.environ.get("KAGGLE_CONTAINER_NAME", ""))
    if session:
        print(f"KGR workload {job_id} session {session.group(1)}", flush=True)


def _restore_source(config, project):
    """Recreate the project from its embedded files or its bundle, verified against the snapshot."""
    embedded = config.get("inline") or config.get("inline_gzip")
    if embedded:
        for name, value in embedded.items():
            target = project / safe_relative(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            content = base64.b64decode(value)
            target.write_bytes(gzip.decompress(content) if config.get("inline_gzip") else content)
        verify_files(project, config["source_files"])
        return
    if config.get("source_local"):
        kind, location, records = _local_bundle(config["source_local"], config["source_digest"])
    else:
        kind, location, records = _find_bundle(config["source_ref"], config["source_digest"])
    if kind == "archive":
        unpack_bundle(location, config["source_digest"], project)
        return
    for name in records:
        target = project / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(location / name, target)
    verify_files(project, records)


def _resolve_input(bundle):
    """Where one input is on this machine: an attached dataset, a path, or a verified bundle."""
    if "dataset" in bundle:
        return str(_find_dataset(bundle["dataset"]))
    if "path" in bundle:
        return _attached_path(bundle["path"])
    if "local" in bundle:
        return str(_local_bundle(bundle["local"], bundle["digest"])[1])
    kind, location, _ = _find_bundle(bundle["ref"], bundle["digest"])
    if kind == "archive":
        location = unpack_bundle(location, bundle["digest"], Path(tempfile.mkdtemp(prefix="kgr-input-")))
    return str(location)


def _attached_path(path):
    """Data already on this machine, attached where it is; a single file arrives in a folder, as copies do."""
    location = os.path.abspath(path)
    if os.path.isfile(location):
        folder = Path(tempfile.mkdtemp(prefix="kgr-input-"))
        (folder / os.path.basename(location)).symlink_to(location)
        location = str(folder)
    return location


def _export_environment(config, inputs, output):
    os.environ.update(config["env"])
    os.environ["KGR_INPUTS_JSON"] = json.dumps(inputs)
    os.environ["KGR_PARAMS_JSON"] = json.dumps(config.get("params", {}))
    for alias, location in inputs.items():
        os.environ["KGR_INPUT_" + alias.upper()] = location
    output.mkdir(exist_ok=True)
    os.environ["KGR_OUTPUT_DIR"] = str(output)
    os.environ["KGR_JOB_ID"] = config["job_id"]
    os.environ["PYTHONUNBUFFERED"] = "1"


def _install_requirements(requirements):
    pip = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
    subprocess.run([*pip, "-r", str(requirements)], check=True)


def run_script(config):
    bootstrap(config)
    command = [sys.executable, "-u"]
    command += ["-m", config["module"]] if config.get("module") else [config["entrypoint"]]
    subprocess.run(command + config["args"], check=True)

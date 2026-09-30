"""Commands the SSH adapter runs on the machine: python HELPER COMMAND JSON, answering in JSON.

Standard library only (Python 3.9 or newer). The adapter uploads this file appended to
runtime.py, so bundle checks are the runtime's own. Layout under the work directory:

    bundles/DIGEST/payload.zip     uploaded bundle, kept for reuse
    bundles/DIGEST/files/          unpacked, verified and read-only; runs read inputs here
    bundles/DIGEST/ready           written once files/ is complete
    runs/REF/                      one attempt: its launcher, run.json, run.log and state.json
    runs/REF/started               created atomically, so an attempt starts at most once
    runs/REF/working/              the runtime's working folder; outputs are downloaded from it
"""

import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# Imported on its own, as in tests; on the machine this file follows runtime.py.
if "unpack_bundle" not in globals():
    from compute_runner.runtime import MANIFEST, file_digest, tree_size, unpack_bundle

STOP_GRACE_SECONDS = 20
# How long start may take between marking a run started and recording its supervisor.
STARTING_SECONDS = 60
# Every process a run starts inherits this variable, so the supervisor can find them all.
MARKER = "KGR_RUN_MARKER"
# Exit status of a command that failed for a passing reason (see __main__).
TRANSIENT_EXIT = 3


def _write_json(path, value, *, replace=True):
    """Write path atomically; with replace=False, only if it does not exist yet."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value))
    if replace:
        os.replace(temporary, path)
        return
    try:
        os.link(temporary, path)
    except FileExistsError:
        pass
    finally:
        temporary.unlink()


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _alive(pid):
    """Whether pid is still a supervisor of this helper, not a reused process ID."""
    try:
        return "supervise" in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except OSError:
        return False


def _members(marker, group):
    """Live processes of one run: in its process group, or carrying the marker it inherits.

    The marker also finds processes that start a session of their own, such as a notebook's
    kernel. Zombies are left out, since they hold nothing and may never be reaped, and so are
    other users' processes, which this user cannot stop.
    """
    needle = f"{MARKER}={marker}".encode()
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            if os.stat(f"/proc/{entry}").st_uid != os.getuid():
                continue
            state, _, pgrp = Path(f"/proc/{entry}/stat").read_text().rsplit(")", 1)[1].split()[:3]
            if state != "Z" and (
                int(pgrp) == group or needle in Path(f"/proc/{entry}/environ").read_bytes().split(b"\0")
            ):
                found.append(int(entry))
        except (OSError, ValueError):  # Gone meanwhile.
            pass
    return found


def _ready(target):
    return target.with_name("ready")


def unpack(archive, digest, target):
    """Unpack and verify a bundle once, then make it read-only so runs cannot change it.

    A lock makes a second unpack, such as one retried after a timeout, wait for the first.
    """
    target = Path(target)
    # Only a bundle's own folder is ever replaced, so a wrong target cannot remove other files.
    if (target.name, target.parent.name, target.parent.parent.name) != ("files", digest, "bundles"):
        raise ValueError(f"Not a bundle folder for {digest}: {target}")
    with open(target.with_name("unpack.lock"), "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _ready(target).is_file():
            return {"ready": True}
        return _unpack_once(archive, digest, target)


def _unpack_once(archive, digest, target):
    staging = target.with_name(f".{target.name}.{os.getpid()}")
    if staging.exists():
        _make_writable(staging)
        shutil.rmtree(staging)
    unpack_bundle(Path(archive), digest, staging)
    with zipfile.ZipFile(archive) as bundle:
        (staging / MANIFEST).write_bytes(bundle.read(MANIFEST))
    for root, _dirs, files in os.walk(staging):
        for name in files:
            os.chmod(os.path.join(root, name), 0o444)
        os.chmod(root, 0o555)
    if target.exists():  # Left by an unpack that stopped before marking it ready.
        _make_writable(target)
        shutil.rmtree(target)
    os.rename(staging, target)
    _ready(target).write_text("")
    return {"ready": True}


def _make_writable(root):
    for current, _dirs, _files in os.walk(root):
        os.chmod(current, 0o755)


def bundle_ready(target):
    return {"ready": _ready(Path(target)).is_file()}


def start(run, env):
    """Start the attempt's supervisor in its own session, once; it outlives this connection."""
    run = Path(run)
    try:
        os.mkdir(run / "started")
    except FileExistsError:
        return {"started": False}
    with open(run / "run.log", "ab") as log:
        process = subprocess.Popen(
            [sys.executable, __file__, "supervise", json.dumps({"run": str(run), "env": env})],
            cwd=run,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    # Recorded now, so status and runs() see the run at once. The supervisor writes the same
    # record when it starts; this one never replaces what it wrote, even its final state.
    _write_json(run / "state.json", _running(process.pid, env), replace=False)
    return {"started": True, "pid": process.pid}


def _running(pid, env):
    return {"state": "running", "pid": pid, "started_at": time.time(), "gpu": env.get("CUDA_VISIBLE_DEVICES")}


def supervise(run, env):
    return Supervisor(run, env).supervise()


class Supervisor:
    """Runs one attempt's workload with its timeout and records how it ended in state.json.

    A stop, from cancel or the timeout, sends SIGTERM to every process of the run and SIGKILL
    after STOP_GRACE_SECONDS; so does a workload that exits and leaves processes behind. The
    result is recorded only once they are all gone, so a GPU stays counted until it is free.
    The cancel file is watched too, so a cancel that arrives before the SIGTERM handler exists
    is not lost.
    """

    def __init__(self, run, env):
        self.run = Path(run)
        self.env = env
        self.settings = json.loads((self.run / "run.json").read_text())
        self.marker = f"{self.run.name}-{os.getpid()}-{time.time_ns()}"
        self.stop_at = None  # When a stop was requested, on the monotonic clock.
        self.child = None
        self.outcome = {"state": "failed", "exit_code": None, "error": None}

    def supervise(self):
        signal.signal(signal.SIGTERM, self._terminate)
        signal.signal(signal.SIGINT, self._terminate)
        # supervised: from now on SIGTERM stops the run; before, it would kill this process unrecorded.
        record = {**_running(os.getpid(), self.env), "supervised": True}
        _write_json(self.run / "state.json", record)
        try:
            self._run_workload()
            self._conclude()
        except Exception as error:  # Recorded rather than lost: nobody is attached to this process.
            self.outcome["error"] = f"{type(error).__name__}: {error}"
        _write_json(self.run / "state.json", {**record, **self.outcome, "finished_at": time.time()})
        return self.outcome

    def _run_workload(self):
        command = self._command(self._python())
        if self._cancelled():
            return
        environment = {**os.environ, **self.env, MARKER: self.marker}
        self.child = subprocess.Popen(command, cwd=self.run, env=environment, start_new_session=True)
        self._watch()
        self.outcome["exit_code"] = self.child.returncode

    def _python(self):
        """The interpreter to run with: a virtual environment of the run when it installs requirements."""
        if not self.settings["requirements"] or self._cancelled():
            return sys.executable
        venv = self.run / "venv"
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], check=True)
        return str(venv / "bin" / "python")

    def _command(self, python):
        code = self.settings["code_file"]
        if self.settings["kind"] != "notebook":
            return [python, "-u", code]
        output = str(self.run / "working" / "notebook.ipynb")
        execute = [python, "-m", "nbconvert", "--to", "notebook", "--execute", code]
        return [*execute, "--output", output, "--ExecutePreprocessor.timeout=-1"]

    def _watch(self):
        """Wait until every process of the run is gone, stopping them once the run must end."""
        deadline = time.monotonic() + self.settings["timeout_seconds"]
        grace = self.settings.get("stop_grace_seconds", STOP_GRACE_SECONDS)
        stopping = None  # When SIGTERM went to the run's processes.
        while self.child.poll() is None or _members(self.marker, self.child.pid):
            now = time.monotonic()
            if stopping is None and self._must_stop(now, deadline):
                stopping = self.stop_at or now
                self._signal_all(signal.SIGTERM)
            elif stopping is not None and now - stopping >= grace:
                self._signal_all(signal.SIGKILL)
            time.sleep(0.5)

    def _must_stop(self, now, deadline):
        """Whether the run ends now: cancelled, timed out, or its workload exited."""
        if self.child.returncode is None and now >= deadline and not self._cancelled():
            self.outcome["error"] = f"Timed out after {self.settings['timeout_seconds']} seconds"
        return self._cancelled() or self.outcome["error"] is not None or self.child.returncode is not None

    def _conclude(self):
        if self._cancelled():
            self.outcome["state"] = "cancelled"
        elif self.outcome["error"] is None and self.child.returncode == 0:
            self.outcome["state"] = "succeeded"
        elif self.outcome["error"] is None:
            self.outcome["error"] = f"The workload exited with status {self.child.returncode}"

    def _terminate(self, *_):
        if self.stop_at is None:
            self.stop_at = time.monotonic()
            if self.child is not None:
                self._signal_all(signal.SIGTERM)

    def _cancelled(self):
        if self.stop_at is None and (self.run / "cancel").exists():
            self._terminate()
        return self.stop_at is not None

    def _signal_all(self, sig):
        for pid in _members(self.marker, self.child.pid):
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass


def status(run):
    run = Path(run)
    if not run.is_dir():
        return {"missing": True}
    record = _read_json(run / "state.json")
    if record is None:
        if not (run / "started").is_dir():
            return {"state": "staged"}
        if time.time() - (run / "started").stat().st_mtime > STARTING_SECONDS:
            return {
                "state": "failed",
                "error": "The run's supervisor never started (its start was interrupted)",
            }
        return {"state": "queued"}
    if record["state"] == "running" and not _alive(record["pid"]):
        # The supervisor may have recorded its result and exited since the first read.
        record = _read_json(run / "state.json") or record
        if record["state"] == "running":
            return {
                "state": "failed",
                "error": "The run stopped without recording a result (was the machine restarted?)",
            }
    return {"state": record["state"], "error": record.get("error")}


def cancel(run):
    """Ask the supervisor to stop the workload; one that has not started yet stops at once.

    A supervisor still starting up sees the cancel file before it starts the workload.
    """
    run = Path(run)
    (run / "cancel").touch()
    record = _read_json(run / "state.json")
    if record and record["state"] == "running" and record.get("supervised") and _alive(record["pid"]):
        try:
            os.kill(record["pid"], signal.SIGTERM)
            return {"signalled": True}
        except ProcessLookupError:  # It finished in the meantime.
            pass
    return {"signalled": False}


def runs(root):
    """Attempts that hold resources: {ref: {"resource": cpu|gpu, "gpu": device or None}}.

    A run counts while its supervisor is alive; start records it before returning.
    """
    found = {}
    for run in Path(root).iterdir() if Path(root).is_dir() else []:
        record = _read_json(run / "state.json")
        if record is None or record["state"] != "running" or not _alive(record["pid"]):
            continue
        settings = _read_json(run / "run.json") or {}
        found[run.name] = {"resource": "gpu" if settings.get("gpu") else "cpu", "gpu": record.get("gpu")}
    return found


def files(root):
    """Regular files below root with their size and SHA-256; symbolic links are left out."""
    root = Path(root)
    found = []
    for current, dirs, names in os.walk(root):
        dirs[:] = [name for name in dirs if not os.path.islink(os.path.join(current, name))]
        for name in names:
            path = Path(current) / name
            if path.is_symlink() or not path.is_file():
                continue
            found.append(
                {
                    "name": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": file_digest(path),
                }
            )
    return {"files": found}


def version(path):
    """Whether path exists, and a fingerprint of its files' names, sizes and modification times.

    Folders on a machine change in place; the fingerprint pins what a copy was made from. Links
    count as what they point to, as a copy follows them; broken links are left out, as a copy
    leaves them out.
    """
    if not os.path.exists(path):
        return {"exists": False}
    digest = hashlib.sha256()
    entries = [(path, os.stat(path))] if os.path.isfile(path) else []
    for current, dirs, names in os.walk(path):
        dirs.sort()
        for name in sorted(names):
            try:
                entries.append((os.path.join(current, name), os.stat(os.path.join(current, name))))
            except FileNotFoundError:
                pass
    for name, found in entries:
        digest.update(f"{os.path.relpath(name, path)}\0{found.st_size}\0{found.st_mtime_ns}\n".encode())
    return {"exists": True, "version": digest.hexdigest()[:16]}


# The work directory's folders a cleanup may list and remove, by artifact kind.
ARTIFACT_FOLDERS = {"run": "runs", "bundle": "bundles"}


def artifacts(root):
    """Run folders and bundles in the work directory, with their size and latest change."""
    found = {}
    for folder in ARTIFACT_FOLDERS.values():
        parent = Path(root) / folder
        entries = sorted(parent.iterdir()) if parent.is_dir() else []
        found[folder] = [
            _measure(path) for path in entries if path.is_dir() and not path.name.startswith(".")
        ]
    return found


def _measure(path):
    size, latest = tree_size(path)
    return {"name": path.name, "bytes": size, "modified_at": latest}


def remove(root, kind, name):
    """Delete one run folder or bundle, and nothing else; a run still running is refused."""
    if not name or name.startswith(".") or "/" in name or kind not in ARTIFACT_FOLDERS:
        raise ValueError(f"Not a {kind} of this runner: {name}")
    target = Path(root) / ARTIFACT_FOLDERS[kind] / name
    if target.is_symlink() or not target.is_dir():
        return {"removed": False}
    record = _read_json(target / "state.json") if kind == "run" else None
    if record and record["state"] == "running" and _alive(record["pid"]):
        raise ValueError(f"Run {name} is still running")
    _make_writable(target)  # Unpacked bundles are read-only.
    shutil.rmtree(target)
    return {"removed": True}


def info():
    """Facts the adapter and doctor report: Python version, home folder and GPUs."""
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        gpus = [line.split(", ", 1)[1] for line in result.stdout.splitlines() if ", " in line]
    except (OSError, subprocess.SubprocessError):
        pass
    return {"python": ".".join(map(str, sys.version_info[:3])), "home": str(Path.home()), "gpus": gpus}


COMMANDS = {
    "unpack": unpack,
    "bundle_ready": bundle_ready,
    "start": start,
    "supervise": supervise,
    "status": status,
    "cancel": cancel,
    "runs": runs,
    "files": files,
    "version": version,
    "info": info,
    "artifacts": artifacts,
    "remove": remove,
}

if __name__ == "__main__":
    try:
        result = COMMANDS[sys.argv[1]](**json.loads(sys.argv[2] if len(sys.argv) > 2 else "{}"))
    except OSError as error:  # Such as a full disk: the command may succeed later. Others are verdicts.
        if isinstance(error, (FileNotFoundError, PermissionError, NotADirectoryError, IsADirectoryError)):
            raise
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(TRANSIENT_EXIT)
    if sys.argv[1] != "supervise":  # The supervisor's output is the run's log.
        print(json.dumps(result))

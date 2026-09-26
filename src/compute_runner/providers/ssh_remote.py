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

if "_unpack" not in globals():  # Imported on its own, as in tests; on the machine it follows runtime.py.
    from compute_runner.runtime import MANIFEST, _unpack

STOP_GRACE_SECONDS = 20
STARTING_SECONDS = 60


def _write_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value))
    os.replace(temporary, path)


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


def _ready(target):
    return target.with_name("ready")


def unpack(archive, digest, target):
    """Unpack and verify a bundle once, then make it read-only so runs cannot change it.

    A lock makes a second unpack, such as one retried after a timeout, wait for the first.
    """
    target = Path(target)
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
    _unpack(Path(archive), digest, staging)
    with zipfile.ZipFile(archive) as bundle:
        (staging / MANIFEST).write_bytes(bundle.read(MANIFEST))
    for root, dirs, files in os.walk(staging):
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
    for current, dirs, files in os.walk(root):
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
    # Counted by runs() before the supervisor announces itself, so a GPU is never given out twice.
    _write_json(run / "assigned.json", {"gpu": env.get("CUDA_VISIBLE_DEVICES")})
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
    return {"started": True, "pid": process.pid}


def supervise(run, env):
    """Run the workload with its timeout and record how it ended in state.json.

    A stop, from cancel or the timeout, sends SIGTERM to the workload's process group and
    SIGKILL after STOP_GRACE_SECONDS. The cancel file is watched too, so a cancel that arrives
    before the SIGTERM handler exists is not lost.
    """
    run = Path(run)
    settings = json.loads((run / "run.json").read_text())
    state = run / "state.json"
    stop = {"at": None}
    child = None

    def signal_group(sig):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass

    def terminate(*_):
        if stop["at"] is None:
            stop["at"] = time.monotonic()
            if child is not None and child.poll() is None:
                signal_group(signal.SIGTERM)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    record = {
        "state": "running",
        "pid": os.getpid(),
        "started_at": time.time(),
        "gpu": env.get("CUDA_VISIBLE_DEVICES"),
    }
    _write_json(state, record)

    def cancelled():
        if stop["at"] is None and (run / "cancel").exists():
            terminate()
        return stop["at"] is not None

    outcome = {"state": "failed", "exit_code": None, "error": None}
    try:
        python = sys.executable
        if settings["requirements"] and not cancelled():
            subprocess.run([python, "-m", "venv", "--system-site-packages", str(run / "venv")], check=True)
            python = str(run / "venv" / "bin" / "python")
        if settings["kind"] == "notebook":
            command = [python, "-m", "nbconvert", "--to", "notebook", "--execute", settings["code_file"]]
            command += [
                "--output",
                str(run / "working" / "notebook.ipynb"),
                "--ExecutePreprocessor.timeout=-1",
            ]
        else:
            command = [python, "-u", settings["code_file"]]
        if not cancelled():
            child = subprocess.Popen(command, cwd=run, env={**os.environ, **env}, start_new_session=True)
            deadline = time.monotonic() + settings["timeout_seconds"]
            grace = settings.get("stop_grace_seconds", STOP_GRACE_SECONDS)
            timed_out_at = None
            while child.poll() is None:
                now = time.monotonic()
                if now >= deadline and timed_out_at is None and not cancelled():
                    outcome["error"] = f"Timed out after {settings['timeout_seconds']} seconds"
                    timed_out_at = now
                    signal_group(signal.SIGTERM)
                stopping = stop["at"] if cancelled() else timed_out_at
                if stopping is not None and now - stopping >= grace:
                    signal_group(signal.SIGKILL)
                time.sleep(0.5)
            outcome["exit_code"] = child.returncode
        if cancelled():
            outcome["state"] = "cancelled"
        elif outcome["error"] is None and child.returncode == 0:
            outcome["state"] = "succeeded"
        elif outcome["error"] is None:
            outcome["error"] = f"The workload exited with status {child.returncode}"
    except Exception as error:  # Recorded rather than lost: nobody is attached to this process.
        outcome["error"] = f"{type(error).__name__}: {error}"
    _write_json(state, {**record, **outcome, "finished_at": time.time()})
    return outcome


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
                "error": "The run's supervisor never started (was the machine restarted?)",
            }
        return {"state": "queued"}
    if record["state"] == "running" and not _alive(record["pid"]):
        return {
            "state": "failed",
            "error": "The run stopped without recording a result (was the machine restarted?)",
        }
    return {"state": record["state"], "error": record.get("error")}


def cancel(run):
    """Ask the supervisor to stop the workload; one that has not started yet stops at once."""
    run = Path(run)
    (run / "cancel").touch()
    record = _read_json(run / "state.json")
    if record and record["state"] == "running" and _alive(record["pid"]):
        os.kill(record["pid"], signal.SIGTERM)
        return {"signalled": True}
    return {"signalled": False}


def runs(root):
    """Attempts that hold resources: {ref: {"resource": cpu|gpu, "gpu": device or None}}.

    A run counts while its supervisor is alive, and while it is starting: started, but its
    supervisor has not written state.json yet (for at most STARTING_SECONDS).
    """
    found = {}
    for run in Path(root).iterdir() if Path(root).is_dir() else []:
        if not (run / "started").is_dir():
            continue
        record = _read_json(run / "state.json")
        if record is None:
            if time.time() - (run / "started").stat().st_mtime > STARTING_SECONDS:
                continue
            gpu = (_read_json(run / "assigned.json") or {}).get("gpu")
        elif record["state"] == "running" and _alive(record["pid"]):
            gpu = record.get("gpu")
        else:
            continue
        settings = _read_json(run / "run.json") or {}
        found[run.name] = {"resource": "gpu" if settings.get("gpu") else "cpu", "gpu": gpu}
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
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            found.append(
                {
                    "name": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
    return {"files": found}


def version(path):
    """Whether path exists, and a fingerprint of its files' names, sizes and modification times.

    Folders on a machine change in place; the fingerprint pins what a copy was made from.
    """
    if not os.path.exists(path):
        return {"exists": False}
    digest = hashlib.sha256()
    entries = [(path, os.stat(path))] if os.path.isfile(path) else []
    for current, dirs, names in os.walk(path):
        dirs.sort()
        entries += [
            (os.path.join(current, name), os.stat(os.path.join(current, name))) for name in sorted(names)
        ]
    for name, found in entries:
        digest.update(f"{os.path.relpath(name, path)}\0{found.st_size}\0{found.st_mtime_ns}\n".encode())
    return {"exists": True, "version": digest.hexdigest()[:16]}


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
}

if __name__ == "__main__":
    result = COMMANDS[sys.argv[1]](**json.loads(sys.argv[2] if len(sys.argv) > 2 else "{}"))
    if sys.argv[1] != "supervise":  # The supervisor's output is the run's log.
        print(json.dumps(result))

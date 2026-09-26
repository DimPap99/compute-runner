"""SSH adapter: runs workloads on a machine you can log in to, as your user, under a supervisor.

Everything happens in one work directory on the machine (see ssh_remote for its layout). The
adapter uploads a helper script there and asks it, over the same SSH connection, to unpack
bundles, start runs detached from the connection, and report their state. Downloads use SFTP.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import posixpath
import re
import shlex
import socket
import stat
import threading
import time
from pathlib import Path

from .. import runtime
from ..models import Account, JobRecord, JobSpec
from ..security import redact_secrets
from ..store import atomic_json, atomic_write, config_path
from . import RemoteError
from .launch import launch_config, write_launcher
from .ssh_remote import TRANSIENT_EXIT

try:
    import paramiko
except ImportError:  # The ssh extra is optional; Kaggle-only installations do without it.
    paramiko = None

# How a lost connection shows up during SFTP: a transient failure, never a verdict on the job.
TRANSPORT_ERRORS = (OSError, EOFError) + ((paramiko.SSHException,) if paramiko else ())

HELPER = Path(__file__).with_name("ssh_remote.py")
LOG_TAIL_BYTES = 256 * 1024
COMMAND_SECONDS = 600
LONG_COMMAND_SECONDS = 6 * 3600
LONG = {"unpack", "files", "version"}
STATES = {"queued", "running", "succeeded", "failed", "cancelled"}


def known_hosts_path() -> Path:
    """Host keys accepted with --trust-new-host, beside the configuration."""
    return config_path().parent / "known_hosts"


# NAME:/PATH#FINGERPRINT as resolve_dataset pins it; NAME: and #FINGERPRINT are optional.
PINNED = re.compile(r"^(?:(?P<machine>[A-Za-z0-9_-]+):)?(?P<path>/.*?)(?:#(?P<version>[0-9a-f]{16}))?$")


def _location(ref: str) -> tuple[str | None, str]:
    """The machine a reference names, if any, and its path."""
    match = PINNED.match(ref)
    if match is None:
        raise ValueError(f"Not a path on an SSH machine: {ref}")
    return match["machine"], match["path"]


def _path(ref: str) -> str:
    return _location(ref)[1]


class SshProvider:
    # How long a stopped workload may take to exit after SIGTERM before it is killed.
    stop_grace_seconds = 20

    def __init__(self, account: Account, state_dir: Path, *, strict=False):
        if paramiko is None:
            raise RemoteError("SSH accounts need the ssh extra: pip install 'compute-runner[ssh]'", "invalid")
        self.account = account
        self.settings = account.ssh
        self.state_dir = state_dir
        self.strict = strict
        self._client = None
        self._lock = threading.Lock()
        self._workdir = None
        self._helper = None

    # Connection -------------------------------------------------------------------------------

    @property
    def _host_key_name(self):
        host, port = self.settings.host, self.settings.port
        return host if port == 22 else f"[{host}]:{port}"

    def _changed_host_key(self):
        return RemoteError(
            f"The host key of {self._host_key_name} does not match the saved one. If the machine "
            "was reinstalled, remove its old key from known_hosts and trust it again; otherwise "
            "do not connect",
            "auth",
            definitive=True,
        )

    def _connect(self, policy):
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        if known_hosts_path().is_file():
            client.load_host_keys(str(known_hosts_path()))
        client.set_missing_host_key_policy(policy)
        password = None
        if self.settings.password_file is not None:
            path = self.settings.password_file.expanduser()
            try:
                if path.stat().st_mode & 0o077:
                    raise RemoteError(
                        f"Password file {path} is readable by other users; run: chmod 600 {path}",
                        "auth",
                        definitive=True,
                    )
                password = path.read_text().rstrip("\n")
            except OSError as error:
                raise RemoteError(
                    f"Cannot read password file {path}: {error}", "auth", definitive=True
                ) from error
        key = str(self.settings.key.expanduser()) if self.settings.key else None
        try:
            client.connect(
                self.settings.host,
                port=self.settings.port,
                username=self.settings.username,
                key_filename=key,
                password=password,
                allow_agent=key is None and password is None,
                look_for_keys=key is None and password is None,
                timeout=30,
                banner_timeout=30,
                auth_timeout=30,
            )
        except paramiko.BadHostKeyException as error:
            raise self._changed_host_key() from error
        except paramiko.AuthenticationException as error:
            raise RemoteError(
                f"SSH login to {self.settings.username}@{self._host_key_name} was refused; check the key or "
                "password file",
                "auth",
                definitive=True,
            ) from error
        except paramiko.SSHException as error:
            if "not found in known_hosts" in str(error):
                raise RemoteError(
                    f"{self._host_key_name} is not a known host. Check its fingerprint, then run: "
                    f"compute-runner account add ssh {self.account.user} --trust-new-host",
                    "auth",
                    definitive=True,
                ) from error
            raise RemoteError(f"SSH to {self._host_key_name} failed: {error}") from error
        except (OSError, EOFError) as error:
            raise RemoteError(f"Cannot reach {self._host_key_name}: {error}") from error
        client.get_transport().set_keepalive(30)
        return client

    def _ssh(self):
        with self._lock:
            transport = self._client.get_transport() if self._client else None
            if transport is None or not transport.is_active():
                self._client = self._connect(paramiko.RejectPolicy())
                self._helper = None
            return self._client

    def trust_host(self) -> str:
        """Accept the machine's host key if none is known for it yet, and return its fingerprint.

        The key is read without logging in, so no password or key reaches an unverified host.
        A key that differs from a known one is refused, as on every connection.
        """
        known = paramiko.HostKeys()
        for path in (Path("~/.ssh/known_hosts").expanduser(), known_hosts_path()):
            if path.is_file():
                known.load(str(path))
        saved = known.lookup(self._host_key_name) or {}
        try:
            with socket.create_connection((self.settings.host, self.settings.port), timeout=30) as sock:
                transport = paramiko.Transport(sock)
                try:
                    # Ask for a key type already saved for the host first, as ssh does.
                    options = transport.get_security_options()
                    options.key_types = [
                        *(kind for kind in options.key_types if kind in saved),
                        *(kind for kind in options.key_types if kind not in saved),
                    ]
                    transport.start_client(timeout=30)
                    key = transport.get_remote_server_key()
                finally:
                    transport.close()
        except TRANSPORT_ERRORS as error:
            raise RemoteError(f"Cannot read the host key of {self._host_key_name}: {error}") from error
        if saved and saved.get(key.get_name()) != key:
            raise self._changed_host_key()
        if not saved:
            own = paramiko.HostKeys()
            path = known_hosts_path()
            if path.is_file():
                own.load(str(path))
            own.add(self._host_key_name, key.get_name(), key)
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            own.save(str(path))
            path.chmod(0o600)
        return key.fingerprint

    def _exec(self, command, *, timeout=COMMAND_SECONDS):
        try:
            _, stdout, stderr = self._ssh().exec_command(command, timeout=timeout)
            output = stdout.read().decode(errors="replace")
            error = stderr.read().decode(errors="replace")
            code = stdout.channel.recv_exit_status()
        except (paramiko.SSHException, OSError, EOFError, socket.timeout) as error:
            raise RemoteError(f"SSH command on {self._host_key_name} failed: {error}") from error
        if code == -1:  # No exit status: the connection dropped, or a signal killed the command.
            raise RemoteError(f"SSH command on {self._host_key_name} ended without an exit status")
        return code, output, error

    def _sftp(self):
        try:
            return self._ssh().open_sftp()
        except (paramiko.SSHException, OSError, EOFError) as error:
            raise RemoteError(f"SFTP to {self._host_key_name} failed: {error}") from error

    # Remote helper ----------------------------------------------------------------------------

    @property
    def workdir(self) -> str:
        """The absolute work directory on the machine."""
        if self._workdir is None:
            sftp = self._sftp()
            try:
                home = sftp.normalize(".")
            except TRANSPORT_ERRORS as error:
                raise RemoteError(
                    f"Finding the home folder on {self._host_key_name} failed: {error}"
                ) from error
            finally:
                sftp.close()
            self._workdir = posixpath.join(home, self.settings.workdir)
        return self._workdir

    def _run_dir(self, ref):
        return posixpath.join(self.workdir, "runs", ref)

    def _bundle_dir(self, digest):
        return posixpath.join(self.workdir, "bundles", digest)

    def _helper_path(self):
        """Upload runtime.py with the helper appended, once per version, and return its path."""
        if self._helper is None:
            code = Path(runtime.__file__).read_text() + "\n" + HELPER.read_text()
            name = f"kgr_helper_{hashlib.sha256(code.encode()).hexdigest()[:16]}.py"
            path = posixpath.join(self.workdir, "lib", name)
            sftp = self._sftp()
            try:
                self._mkdirs(sftp, posixpath.dirname(path))
                try:
                    sftp.stat(path)
                except FileNotFoundError:
                    self._put_bytes(sftp, code.encode(), path)
            except TRANSPORT_ERRORS as error:
                raise RemoteError(f"Uploading the helper to {self._host_key_name} failed: {error}") from error
            finally:
                sftp.close()
            self._helper = path
        return self._helper

    def _call(self, command, **arguments):
        line = " ".join(
            shlex.quote(part)
            for part in [self.settings.python, self._helper_path(), command, json.dumps(arguments)]
        )
        # Unpacking, hashing outputs and fingerprinting folders scale with the data.
        code, output, error = self._exec(
            line, timeout=LONG_COMMAND_SECONDS if command in LONG else COMMAND_SECONDS
        )
        if code != 0:
            detail = (
                f"{command} on {self._host_key_name} failed: "
                + (error.strip().splitlines() or ["no output"])[-1]
            )
            if code == TRANSIENT_EXIT:
                raise RemoteError(detail)
            raise RemoteError(detail, "invalid", definitive=True)
        return json.loads(output.strip().splitlines()[-1])

    @staticmethod
    def _mkdirs(sftp, path):
        parts, current = path.strip("/").split("/"), ""
        for part in parts:
            current += "/" + part
            try:
                sftp.stat(current)
            except FileNotFoundError:
                sftp.mkdir(current, mode=0o700)

    @staticmethod
    def _put_bytes(sftp, data, path):
        temporary = f"{path}.{time.time_ns()}.part"
        with sftp.open(temporary, "wb") as stream:
            stream.write(data)
        sftp.posix_rename(temporary, path)

    # Provider contract ------------------------------------------------------------------------

    def check(self, spec: JobSpec):
        if not spec.internet:
            raise ValueError("SSH machines cannot block network access; set internet: true for SSH accounts")
        if spec.accelerator:
            raise ValueError("SSH machines take gpu: true, not an accelerator ID")
        if spec.gpu and self.account.gpu_limit == 0:
            raise ValueError(f"{self.account.id} has no GPU slots; add them with account add --gpu-limit N")
        if spec.datasets:
            raise ValueError("SSH jobs take data as inputs, such as inputs: {data: 'ssh:/path/on/machine'}")

    def url(self, ref):
        address = (
            f"{self.settings.host}:{self.settings.port}" if self.settings.port != 22 else self.settings.host
        )
        return f"ssh://{self.settings.username}@{address}/~/{self.settings.workdir}/runs/{ref}"

    def ensure_bundle(self, bundle) -> str | None:
        """Upload the bundle once and unpack it verified; its digest is its reference."""
        digest = bundle["digest"]
        target = self._bundle_dir(digest)
        if self._call("bundle_ready", target=posixpath.join(target, "files"))["ready"]:
            return digest
        archive = posixpath.join(target, "payload.zip")
        local = self.state_dir / "bundles" / digest / "payload.zip"
        sftp = self._sftp()
        try:
            self._mkdirs(sftp, target)
            temporary = f"{archive}.{time.time_ns()}.part"
            sftp.put(str(local), temporary, confirm=True)
            sftp.posix_rename(temporary, archive)
        except TRANSPORT_ERRORS as error:
            raise RemoteError(f"Uploading bundle {digest[:12]} failed: {error}") from error
        finally:
            sftp.close()
        self._call("unpack", archive=archive, digest=digest, target=posixpath.join(target, "files"))
        return digest

    def resolve_dataset(self, ref):
        """A path on this machine, readable when it exists; pinned as NAME:PATH#FINGERPRINT.

        A path on another machine is not readable here, even if the same path exists.
        """
        machine, path = _location(ref)
        if machine is not None and machine.casefold() != self.account.user.casefold():
            return None
        found = self._call("version", path=path)
        return f"{self.account.user}:{path}#{found['version']}" if found["exists"] else None

    def fetch_dataset(self, ref, destination: Path):
        """Copy a file or folder from this machine, as plain files."""
        sftp = self._sftp()
        try:
            self._get_tree(sftp, _path(ref), destination)
        except TRANSPORT_ERRORS as error:
            raise RemoteError(f"Copying {ref} from {self._host_key_name} failed: {error}") from error
        finally:
            sftp.close()

    def _get_tree(self, sftp, path, destination):
        """Copy path as the job on this machine sees it: links to files count as the files.

        Broken links are left out, as the fingerprint leaves them out. Links to folders could
        loop, so they are refused.
        """
        if stat.S_ISREG(sftp.stat(path).st_mode):
            sftp.get(path, str(destination / posixpath.basename(path)))
            return
        for entry in sftp.listdir_attr(path):
            child = posixpath.join(path, entry.filename)
            mode = entry.st_mode
            if stat.S_ISLNK(mode):
                try:
                    mode = sftp.stat(child).st_mode
                except FileNotFoundError:
                    continue
                if stat.S_ISDIR(mode):
                    raise ValueError(f"Links to folders cannot be copied: {child}")
            if stat.S_ISDIR(mode):
                (destination / entry.filename).mkdir()
                self._get_tree(sftp, child, destination / entry.filename)
            elif stat.S_ISREG(mode):
                sftp.get(child, str(destination / entry.filename))
            else:
                raise ValueError(f"Only files and folders can be copied: {child}")

    def _folder(self, job, number):
        return self.state_dir / "jobs" / job.id / f"attempt-{number}"

    def stage(self, job: JobRecord, number: int) -> str:
        """Build the launch package locally; paths in it are relative to the run's folder."""
        name = re.sub(r"[^a-z0-9]+", "-", job.spec.name.lower())[:16].strip("-") or "workload"
        ref = f"kgr-{name}-{job.id[:12]}-a{number}"
        folder = self._folder(job, number)
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        snapshot = job.snapshot
        inputs = {}
        for alias, bundle in [*snapshot["inputs"].items(), *job.transfers.items()]:
            inputs[alias] = dict(local=f"../../bundles/{bundle['digest']}/files", digest=bundle["digest"])
        for alias in job.spec.dataset_inputs():
            if alias not in inputs:  # Not copied, so a path on this machine.
                inputs[alias] = dict(path=_path(job.upload_refs["input:" + alias]))
        location = {"working_root": "working"}
        if not snapshot["single_file"]:
            location["source_local"] = f"../../bundles/{snapshot['source']['digest']}/files"
        config = launch_config(job, self.state_dir, inputs, **location)
        code_file = write_launcher(job, config, folder, self.state_dir)
        settings = dict(
            kind=snapshot["kind"],
            code_file=code_file,
            requirements=bool(job.spec.requirements),
            timeout_seconds=job.spec.timeout_seconds,
            stop_grace_seconds=self.stop_grace_seconds,
            gpu=job.spec.gpu,
        )
        atomic_json(folder / "run.json", settings)
        return ref

    def submit(self, job: JobRecord) -> dict:
        attempt = job.attempts[-1]
        run = self._run_dir(attempt.ref)
        folder = self._folder(job, attempt.number)
        # Until the start command runs, nothing can have launched: failures are definitive.
        try:
            env = {"CUDA_VISIBLE_DEVICES": self._free_gpu() if job.spec.gpu else ""}
            sftp = self._sftp()
            try:
                self._mkdirs(sftp, posixpath.join(run, "working"))
                for path in sorted(folder.iterdir()):
                    if path.is_file():
                        self._put_bytes(sftp, path.read_bytes(), posixpath.join(run, path.name))
            finally:
                sftp.close()
        except RemoteError as error:
            raise RemoteError(str(error), error.kind, definitive=True) from error
        except TRANSPORT_ERRORS as error:
            raise RemoteError(f"Uploading the launch package failed: {error}", definitive=True) from error
        self._call("start", run=run, env=env)
        return {}

    def _free_gpu(self) -> str:
        used = {
            found["gpu"] for found in self._call("runs", root=posixpath.join(self.workdir, "runs")).values()
        }
        for index in range(self.account.gpu_limit):
            if str(index) not in used:
                return str(index)
        raise RemoteError(f"Every GPU on {self.account.id} is in use", "capacity", definitive=True)

    def status(self, ref):
        found = self._call("status", run=self._run_dir(ref))
        # A package uploaded but never started is no run either.
        if found.get("missing") or found["state"] == "staged":
            raise RemoteError(f"No run {ref} on {self._host_key_name}", "missing", definitive=True)
        state = found["state"] if found["state"] in STATES else None
        return dict(state=state, detail=found["state"], error=found.get("error"))

    def cancel(self, ref, job_id):
        """Signal the supervisor, which stops every process of the run; polling sees the end."""
        self._call("cancel", run=self._run_dir(ref))
        return False

    def active_runs(self):
        found = self._call("runs", root=posixpath.join(self.workdir, "runs"))
        return {ref: run["resource"] for ref, run in found.items()}

    def quota(self):
        """No GPU time limit; GPU slots are the account's gpu_limit."""
        return {"gpu": {"available_seconds": None} if self.account.gpu_limit else None, "refresh_at": None}

    def info(self) -> dict:
        """Python version, home folder and GPUs on the machine, for doctor."""
        return self._call("info")

    def _read(self, ref, *, tail=None):
        sftp = self._sftp()
        try:
            with sftp.open(posixpath.join(self._run_dir(ref), "run.log"), "rb") as stream:
                if tail is not None:
                    size = stream.stat().st_size
                    stream.seek(max(0, size - tail))
                return stream.read().decode(errors="replace")
        except FileNotFoundError:
            return ""
        except TRANSPORT_ERRORS as error:
            raise RemoteError(f"Reading the log of {ref} failed: {error}") from error
        finally:
            sftp.close()

    def live_log(self, ref):
        return redact_secrets(self._read(ref, tail=LOG_TAIL_BYTES), strict=self.strict)

    def logs(self, ref, *, follow=False):
        if not follow:
            yield redact_secrets(self._read(ref), strict=self.strict)
            return
        # Read only what was appended, and never split a UTF-8 character between reads.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        offset = 0
        sftp = self._sftp()
        try:
            while True:
                finished = self.status(ref)["state"] not in {"queued", "running"}
                try:
                    with sftp.open(posixpath.join(self._run_dir(ref), "run.log"), "rb") as stream:
                        stream.seek(offset)
                        data = stream.read()
                except FileNotFoundError:
                    data = b""
                offset += len(data)
                if text := decoder.decode(data, final=finished):
                    yield redact_secrets(text, strict=self.strict)
                if finished:
                    return
                time.sleep(2)
        except TRANSPORT_ERRORS as error:
            raise RemoteError(f"Following the log of {ref} failed: {error}") from error
        finally:
            sftp.close()

    def download(self, ref, sink):
        run = self._run_dir(ref)
        sink.log(self._read(ref))
        listed = self._call("files", root=posixpath.join(run, "working"))["files"]
        sftp = self._sftp()
        try:
            for item in listed:
                target = sink.target(item["name"])
                if target is None:
                    continue
                with sftp.open(posixpath.join(run, "working", item["name"]), "rb") as stream:
                    stream.prefetch()
                    atomic_write(
                        target, _checked(stream, item), check_space=True, expected_bytes=item["bytes"]
                    )
                sink.saved(item["name"], target, item["sha256"])
        except TRANSPORT_ERRORS as error:
            raise RemoteError(f"Downloading outputs of {ref} failed: {error}") from error
        finally:
            sftp.close()


def _checked(stream, item):
    """Yield a download; fail before the file is replaced if it differs from what was listed."""
    digest, size = hashlib.sha256(), 0
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
        yield chunk
    if size != item["bytes"] or digest.hexdigest() != item["sha256"]:
        raise RemoteError(f"{item['name']} changed while it was downloaded; retrying later")

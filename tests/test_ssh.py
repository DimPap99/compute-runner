"""SSH accounts: the adapter runs for real, with its commands and SFTP served by this machine.

LocalSsh keeps everything the adapter does (uploads, the helper, detached supervisors, downloads)
and only replaces the transport, so these tests need no SSH server. test_real_ssh_server runs the
same flow over SSH when KGR_TEST_SSH names a machine.
"""

import json
import os
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from compute_runner import Account, Client, Config, JobSpec
from compute_runner.bundle import snapshot_bundle
from compute_runner.cli import app
from compute_runner.credentials import account_secrets, credentials_path
from compute_runner.models import SshSettings, input_reference
from compute_runner.providers import Artifact, RemoteError, ssh_remote
from compute_runner.providers.ssh import HELPER, SshProvider
from compute_runner.runtime import __file__ as RUNTIME

from conftest import FakeProvider, due


class LocalSftp:
    """The SFTP calls the adapter makes, on the local filesystem."""

    def __init__(self, home):
        self.home = home

    def normalize(self, path):
        return str(self.home)

    def stat(self, path):
        return os.stat(path)

    def lstat(self, path):
        return os.lstat(path)

    def mkdir(self, path, mode=0o777):
        os.mkdir(path, mode)

    def open(self, path, mode="r"):
        stream = open(path, mode)
        stream.prefetch = lambda: None
        stream.stat = lambda: os.fstat(stream.fileno())
        return stream

    def put(self, local, remote, confirm=True):
        shutil.copyfile(local, remote)

    def get(self, remote, local):
        shutil.copyfile(remote, local)

    def posix_rename(self, source, target):
        os.replace(source, target)

    def listdir_attr(self, path):
        return [
            SimpleNamespace(filename=name, st_mode=os.lstat(os.path.join(path, name)).st_mode)
            for name in sorted(os.listdir(path))
        ]

    def close(self):
        pass


class LocalSsh(SshProvider):
    def _exec(self, command, *, timeout=600):
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr

    def _sftp(self):
        return LocalSftp(self.home)


def ssh_account(**changes):
    settings = SshSettings(host="lab.example", username="me", python=sys.executable)
    return Account(provider="ssh", user="lab", ssh=settings, **changes)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """An SSH account whose machine is a folder here; kaggle:tester is also connected."""
    monkeypatch.setenv("KGR_CONFIG_DIR", str(tmp_path / "config"))
    config = Config(
        accounts=[ssh_account(gpu_limit=2), Account(user="tester")],
        state_dir=tmp_path / "state",
        poll_seconds=1,
        retry_seconds=1,
        reconcile_seconds=1,
    )
    machine = LocalSsh(config.accounts[0], config.state_dir)
    machine.home = tmp_path / "home"
    machine.home.mkdir()
    kaggle = FakeProvider(config.state_dir)
    client = Client(config=config, providers={"ssh:lab": machine, "kaggle:tester": kaggle})
    project = tmp_path / "project"
    project.mkdir()
    return SimpleNamespace(client=client, machine=machine, kaggle=kaggle, project=project, tmp=tmp_path)


DONE = {"complete", "error", "disabled"}


def settle(client, *job_ids, seconds=60):
    """Run worker cycles until the jobs finish and their downloads end."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        client.worker().tick()
        jobs = [client.get(job_id) for job_id in job_ids]
        if all(job.state == "blocked" or job.terminal and job.download_state in DONE for job in jobs):
            return jobs
        for job_id in job_ids:
            due(client, job_id)
        time.sleep(0.2)
    raise AssertionError([(job.state, job.download_state, job.error) for job in jobs])


def running(client, job_id, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        client.worker().tick()
        if client.get(job_id).state == "running":
            return
        due(client, job_id)
        time.sleep(0.2)
    raise AssertionError(client.get(job_id).state)


def test_a_workload_runs_detached_and_its_outputs_come_back(lab):
    (lab.project / "train.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "out = Path(os.environ['KGR_OUTPUT_DIR'])\n"
        "data = os.environ['KGR_INPUT_DATA']\n"
        "(out / 'seen.json').write_text(json.dumps({'argv': sys.argv[1:], 'data': sorted(os.listdir(data)),"
        " 'cuda': os.environ['CUDA_VISIBLE_DEVICES']}))\n"
        "Path('../scratch.txt').write_text('left')\nprint('trained')\n"
    )
    (lab.tmp / "data").mkdir()
    (lab.tmp / "data/rows.csv").write_text("a\n")
    spec = JobSpec(
        source=lab.project,
        entrypoint="train.py",
        internet=True,
        params={"lr": 0.1},
        inputs={"data": lab.tmp / "data"},
    )
    [job] = settle(lab.client, lab.client.submit(spec).id)
    assert job.state == "succeeded" and job.download_state == "complete", job.error
    assert job.url.startswith("ssh://me@lab.example/~/.compute-runner/runs/kgr-")
    seen = json.loads((job.result_dir / "outputs/seen.json").read_text())
    # Inputs hold only the data and its manifest, as on Kaggle; CPU jobs get no GPU.
    assert seen == {"argv": ["--lr", "0.1"], "data": ["kgr-manifest.json", "rows.csv"], "cuda": ""}
    assert (job.result_dir / "working/scratch.txt").read_text() == "left"
    assert "trained" in (job.result_dir / "run.log").read_text()
    # Bundles are unpacked once, verified and read-only, so a run cannot change another's inputs.
    bundles = [
        path for path in (lab.machine.home / ".compute-runner/bundles").iterdir() if (path / "ready").exists()
    ]
    assert bundles and all(not os.stat(path / "files").st_mode & stat.S_IWUSR for path in bundles)


def test_requirements_install_into_a_virtual_environment_of_the_run(lab):
    (lab.project / "requirements.txt").write_text("")
    (lab.project / "env.py").write_text(
        "import os, sys\nfrom pathlib import Path\n"
        "Path(os.environ['KGR_OUTPUT_DIR'], 'python.txt').write_text(sys.prefix)\n"
    )
    spec = JobSpec(source=lab.project, entrypoint="env.py", internet=True, requirements="requirements.txt")
    [job] = settle(lab.client, lab.client.submit(spec).id, seconds=180)
    assert job.state == "succeeded", (job.result_dir / "run.log").read_text()
    assert (job.result_dir / "outputs/python.txt").read_text().endswith("/venv")


def test_cancel_stops_the_whole_process_group(lab):
    (lab.project / "sleepy.py").write_text(
        "import os, subprocess, time\nfrom pathlib import Path\n"
        "child = subprocess.Popen(['sleep', '300'])\n"
        "Path(os.environ['KGR_OUTPUT_DIR'], 'child.pid').write_text(str(child.pid))\ntime.sleep(300)\n"
    )
    job = lab.client.submit(JobSpec(source=lab.project, entrypoint="sleepy.py", internet=True))
    running(lab.client, job.id)
    run = lab.machine.home / ".compute-runner/runs" / lab.client.get(job.id).remote_ref
    deadline = time.monotonic() + 30
    while not (run / "working/outputs/child.pid").exists() and time.monotonic() < deadline:
        time.sleep(0.2)
    child = int((run / "working/outputs/child.pid").read_text())
    assert lab.client.cancel(job.id).wait_reason == "Cancellation requested on ssh:lab"
    [job] = settle(lab.client, job.id)
    assert job.state == "cancelled"
    # The workload's own child ends too; an ended process may linger as a zombie until reaped.
    deadline = time.monotonic() + 10
    while alive(child) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not alive(child)


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


STUBBORN = (
    "import os, subprocess, sys, time\nfrom pathlib import Path\n"
    "code = 'import signal, time\\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\\ntime.sleep(300)'\n"
    # Its own session, as a notebook's kernel has: outside the workload's process group.
    "child = subprocess.Popen([sys.executable, '-c', code], start_new_session=True)\n"
    "time.sleep(1)  # Let it install its handler.\n"
    "Path(os.environ['KGR_OUTPUT_DIR'], 'child.pid').write_text(str(child.pid))\n"
)


def child_of(lab, job_id):
    run = lab.machine.home / ".compute-runner/runs" / lab.client.get(job_id).remote_ref
    deadline = time.monotonic() + 30
    while not (run / "working/outputs/child.pid").exists() and time.monotonic() < deadline:
        time.sleep(0.2)
    return int((run / "working/outputs/child.pid").read_text())


def test_a_workload_that_ignores_sigterm_is_killed_after_the_grace_period(lab):
    lab.machine.stop_grace_seconds = 1
    (lab.project / "stubborn.py").write_text(STUBBORN + "time.sleep(300)\n")
    job = lab.client.submit(JobSpec(source=lab.project, entrypoint="stubborn.py", internet=True))
    running(lab.client, job.id)
    child = child_of(lab, job.id)
    lab.client.cancel(job.id)
    [job] = settle(lab.client, job.id, seconds=30)
    # Recorded only once every process of the run is gone, even one in a session of its own.
    assert job.state == "cancelled" and not alive(child)


def test_processes_a_finished_workload_leaves_behind_are_stopped(lab):
    lab.machine.stop_grace_seconds = 1
    (lab.project / "leaves.py").write_text(STUBBORN + "print('done')\n")
    job = lab.client.submit(JobSpec(source=lab.project, entrypoint="leaves.py", internet=True))
    [job] = settle(lab.client, job.id, seconds=30)
    child = int((job.result_dir / "outputs/child.pid").read_text())
    assert job.state == "succeeded" and not alive(child)


def test_following_a_log_returns_exactly_what_was_written(lab):
    (lab.project / "talk.py").write_text(
        "import time\nfor i in range(5):\n"
        "    print(f'line {i} \u00e9\u00e8', flush=True)\n    time.sleep(0.4)\n"
    )
    job = lab.client.submit(JobSpec(source=lab.project, entrypoint="talk.py", internet=True))
    running(lab.client, job.id)
    followed = "".join(lab.machine.logs(lab.client.get(job.id).remote_ref, follow=True))
    assert (
        followed == lab.machine._read(lab.client.get(job.id).remote_ref) and "line 4 \u00e9\u00e8" in followed
    )


def test_timeouts_and_failures_are_recorded(lab):
    (lab.project / "slow.py").write_text("import time\ntime.sleep(60)\n")
    (lab.project / "bad.py").write_text("raise SystemExit(7)\n")
    slow = lab.client.submit(
        JobSpec(source=lab.project, entrypoint="slow.py", internet=True, timeout_seconds=2)
    )
    bad = lab.client.submit(JobSpec(source=lab.project, entrypoint="bad.py", internet=True))
    slow, bad = settle(lab.client, slow.id, bad.id, seconds=90)
    assert slow.state == "failed" and slow.error == "Timed out after 2 seconds"
    assert bad.state == "failed" and "exited with status 1" in bad.error


def test_a_run_whose_machine_restarted_is_reported_failed(lab):
    (lab.project / "sleepy.py").write_text("import time\ntime.sleep(300)\n")
    job = lab.client.submit(JobSpec(source=lab.project, entrypoint="sleepy.py", internet=True))
    running(lab.client, job.id)
    run = lab.machine.home / ".compute-runner/runs" / lab.client.get(job.id).remote_ref
    record = json.loads((run / "state.json").read_text())
    # Kill the supervisor and its workload without letting it record anything, as a reboot would.
    os.kill(record["pid"], signal.SIGKILL)
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            if os.readlink(f"/proc/{pid}/cwd") == str(run):
                os.kill(int(pid), signal.SIGKILL)
        except OSError:
            pass
    time.sleep(0.5)
    [job] = settle(lab.client, job.id)
    assert job.state == "failed" and "stopped without recording a result" in job.error


def test_gpu_jobs_get_their_own_device_and_wait_when_all_are_busy(lab):
    (lab.project / "gpu.py").write_text(
        "import os, time\nfrom pathlib import Path\n"
        "Path(os.environ['KGR_OUTPUT_DIR'], 'cuda.txt').write_text(os.environ['CUDA_VISIBLE_DEVICES'])\n"
        "time.sleep(3)\n"
    )
    ids = [
        lab.client.submit(
            JobSpec(source=lab.project, entrypoint="gpu.py", internet=True, gpu=True, name=f"g{i}")
        ).id
        for i in range(3)
    ]
    lab.client.worker().tick()
    states = [lab.client.get(job_id).state for job_id in ids]
    assert states.count("remote_queued") == 2 and states.count("queued") == 1, states
    jobs = settle(lab.client, *ids, seconds=90)
    devices = [(job.result_dir / "outputs/cuda.txt").read_text() for job in jobs]
    assert all(job.state == "succeeded" for job in jobs) and set(devices) == {"0", "1"}


def test_folders_on_the_machine_are_attached_where_they_are(lab):
    shared = lab.machine.home / "shared"
    shared.mkdir()
    (shared / "big.bin").write_text("x")
    (lab.project / "use.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "Path(os.environ['KGR_OUTPUT_DIR'], 'data.txt').write_text(os.environ['KGR_INPUT_DATA'])\n"
    )
    job = lab.client.submit(
        JobSpec(
            source=lab.project, entrypoint="use.py", internet=True, inputs={"data": Path(f"ssh:{shared}")}
        )
    )
    [job] = settle(lab.client, job.id)
    assert (job.result_dir / "outputs/data.txt").read_text() == str(shared)
    missing = lab.client.submit(
        JobSpec(source=lab.project, entrypoint="use.py", internet=True, inputs={"data": Path("ssh:/nowhere")})
    )
    [missing] = settle(lab.client, missing.id)
    assert str(missing.spec.inputs["data"]) == "ssh:lab:/nowhere"
    assert (
        missing.state == "blocked"
        and "find dataset lab:/nowhere. Check the reference (MACHINE:" in missing.error
    )


def test_a_changed_folder_is_copied_again(lab):
    shared = lab.machine.home / "shared"
    shared.mkdir()
    (shared / "a.csv").write_text("1\n")
    lab.client.config.transfer = True
    (lab.project / "use.py").write_text("print('ok')\n")
    spec = JobSpec(source=lab.project, entrypoint="use.py", inputs={"data": Path(f"ssh:lab:{shared}")})
    # On another account, a folder must name its machine.
    with pytest.raises(ValueError, match="Name the machine of input data"):
        lab.client.submit(
            spec.model_copy(update={"inputs": {"data": Path(f"ssh:{shared}")}}), account="kaggle:tester"
        )

    def copied():
        job = lab.client.submit(spec, account="kaggle:tester")
        lab.client.worker().tick()
        return lab.client.get(job.id).transfers["data"]

    first = copied()
    assert copied() == first
    (shared / "a.csv").write_text("2\n")
    changed = copied()
    assert changed["source"] != first["source"] and changed["digest"] != first["digest"]


def test_a_kaggle_dataset_is_copied_onto_the_machine(lab):
    lab.client.config.transfer = True
    (lab.project / "use.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "listed = ' '.join(sorted(os.listdir(os.environ['KGR_INPUT_DATA'])))\n"
        "Path(os.environ['KGR_OUTPUT_DIR'], 'files.txt').write_text(listed)\n"
    )
    spec = JobSpec(
        source=lab.project, entrypoint="use.py", internet=True, inputs={"data": Path("kaggle:tester/private")}
    )
    [job] = settle(lab.client, lab.client.submit(spec, account="ssh:lab").id)
    assert job.state == "succeeded", job.error
    assert (job.result_dir / "outputs/files.txt").read_text() == "kgr-manifest.json train.csv"
    assert lab.kaggle.fetched == ["tester/private/7"]


def test_a_path_on_one_machine_is_not_readable_on_another(lab):
    shared = lab.machine.home / "shared"
    shared.mkdir()
    other = LocalSsh(ssh_account().model_copy(update={"user": "lab2"}), lab.tmp / "state")
    other.home = lab.machine.home  # Even where the same path exists, it is other data.
    assert lab.machine.resolve_dataset(f"lab:{shared}").startswith(f"lab:{shared}#")
    assert other.resolve_dataset(f"lab:{shared}") is None


def test_an_interrupted_submission_that_never_started_is_not_mistaken_for_a_run(lab):
    (lab.project / "t.py").write_text("print(1)\n")
    job = lab.client.submit(JobSpec(source=lab.project, entrypoint="t.py", internet=True))
    call = lab.machine._call

    def connection_lost_at_start(command, **arguments):
        if command == "start":
            raise RemoteError("connection lost")
        return call(command, **arguments)

    lab.machine._call = connection_lost_at_start
    lab.client.worker().tick()
    assert lab.client.get(job.id).attempts[-1].state == "uncertain"
    lab.machine._call = call
    ref = lab.client.get(job.id).remote_ref
    with pytest.raises(RemoteError) as error:
        lab.machine.status(ref)
    assert error.value.kind == "missing"


def test_a_dropped_connection_during_an_upload_is_retried(lab, monkeypatch):
    import paramiko

    (lab.tmp / "data").mkdir()
    (lab.tmp / "data/x.csv").write_text("1\n")
    (lab.project / "t.py").write_text("print(1)\n")

    def dropped(self, local, remote, confirm=True):
        raise paramiko.SSHException("Server connection dropped")

    monkeypatch.setattr(LocalSftp, "put", dropped)
    job = lab.client.submit(
        JobSpec(source=lab.project, entrypoint="t.py", internet=True, inputs={"data": lab.tmp / "data"})
    )
    lab.client.worker().tick()
    job = lab.client.get(job.id)
    assert job.state == "preparing" and job.wait_reason == "Upload retry pending", job.error
    monkeypatch.undo()
    [job] = settle(lab.client, job.id)
    assert job.state == "succeeded"


def test_a_run_whose_supervisor_never_started_is_reported_failed(lab):
    run = lab.machine.home / ".compute-runner/runs/kgr-x-a1"
    (run / "started").mkdir(parents=True)
    assert lab.machine.status("kgr-x-a1")["state"] == "queued"
    old = time.time() - 3600
    os.utime(run / "started", (old, old))
    found = lab.machine.status("kgr-x-a1")
    assert found["state"] == "failed" and "never started" in found["error"]


def test_ssh_accounts_refuse_what_they_cannot_honour(lab):
    machine = lab.machine
    with pytest.raises(ValueError, match="cannot block network access"):
        machine.check(JobSpec(source=lab.project))
    with pytest.raises(ValueError, match="not an accelerator"):
        machine.check(JobSpec(source=lab.project, internet=True, accelerator="NvidiaTeslaT4"))
    with pytest.raises(ValueError, match="as inputs"):
        machine.check(JobSpec(source=lab.project, internet=True, datasets=["a/b"]))
    no_gpus = SshProvider(ssh_account(), lab.tmp)
    with pytest.raises(ValueError, match="no GPU slots"):
        no_gpus.check(JobSpec(source=lab.project, internet=True, gpu=True))
    assert no_gpus.quota()["gpu"] is None and machine.quota()["gpu"] == {"available_seconds": None}


def test_offline_jobs_never_fail_over_to_an_ssh_machine(lab):
    lab.client.config.failover = "auto"
    lab.kaggle.external = {"tester/other": "cpu"}
    lab.client.config.accounts[1] = lab.client.config.accounts[1].model_copy(update={"cpu_limit": 1})
    (lab.project / "t.py").write_text("print(1)\n")
    offline = lab.client.submit(JobSpec(source=lab.project, entrypoint="t.py"), account="kaggle:tester")
    online = lab.client.submit(
        JobSpec(source=lab.project, entrypoint="t.py", internet=True), account="kaggle:tester"
    )
    lab.client.worker().tick()
    assert (
        lab.client.get(offline.id).account == "kaggle:tester"
        and lab.client.get(online.id).account == "ssh:lab"
    )


def test_ssh_account_settings_and_references():
    assert ssh_account().gpu_limit == 0 and ssh_account(gpu_limit=2).gpu_limit == 2
    assert ssh_account().id == "ssh:lab"
    with pytest.raises(ValueError, match="need host settings"):
        Account(provider="ssh", user="lab")
    with pytest.raises(ValueError, match="need host settings"):
        Account(user="k", ssh=ssh_account().ssh)
    assert input_reference("ssh:/data/set") == ("ssh", "/data/set")
    assert input_reference("ssh:relative") is None and input_reference("kaggle:/abs") is None


def test_account_add_saves_ssh_settings_and_its_secrets_apart(tmp_path, monkeypatch):
    monkeypatch.setenv("KGR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("KGR_STATE_DIR", str(tmp_path / "state"))
    runner = CliRunner()
    password, key = tmp_path / "password", tmp_path / "key"
    password.write_text("secret\n")
    key.write_text("key")
    add = ["--json", "account", "add", "ssh", "lab", "--host", "10.0.0.5", "--login", "me"]
    assert runner.invoke(app, [*add, "--password-file", str(password), "--gpu-limit", "1"]).exit_code == 0
    assert account_secrets("ssh:lab") == {"password": "secret"}
    # Updating keeps other settings; a key replaces the saved password.
    assert runner.invoke(app, ["--json", "account", "add", "ssh", "lab", "--key", str(key)]).exit_code == 0
    [account] = Client().config.accounts
    assert account.ssh.host == "10.0.0.5" and account.gpu_limit == 1
    assert account_secrets("ssh:lab") == {"key": str(key)}
    assert "secret" not in (tmp_path / "config/config.json").read_text()
    both = runner.invoke(app, [*add, "--key", str(key), "--password-file", str(password)])
    assert both.exit_code != 0 and "Choose one of" in str(both.exception)
    kaggle = runner.invoke(app, ["--json", "account", "add", "kaggle", "someone", "--host", "x"])
    assert kaggle.exit_code != 0 and "SSH accounts only" in str(kaggle.exception)
    login = runner.invoke(app, ["--json", "account", "add", "kaggle", "someone", "--login", "x"])
    assert "--login applies to SSH accounts only" in str(login.exception)
    # Refused before anything is saved.
    trusted = runner.invoke(app, ["--json", "account", "add", "kaggle", "someone", "--trust-new-host"])
    assert trusted.exit_code != 0 and [a.id for a in Client().config.accounts] == ["ssh:lab"]


def test_the_remote_helper_is_the_runtime_plus_commands(tmp_path):
    # The machine gets runtime.py with the helper appended; bundles are checked by the runtime itself.
    helper = tmp_path / "helper.py"
    helper.write_text(Path(RUNTIME).read_text() + "\n" + HELPER.read_text())
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.txt").write_text("a")
    bundle = snapshot_bundle(tmp_path / "src", ["a.txt"], tmp_path / "bundles")
    archive = tmp_path / "bundles" / bundle["digest"] / "payload.zip"

    def call(command, **arguments):
        result = subprocess.run(
            [sys.executable, str(helper), command, json.dumps(arguments)], capture_output=True, text=True
        )
        return result.returncode, result.stdout, result.stderr

    target = tmp_path / "remote/bundles" / bundle["digest"] / "files"
    target.parent.mkdir(parents=True)
    code, out, _ = call("unpack", archive=str(archive), digest="0" * 64, target=str(target))
    assert code != 0 and not target.exists()
    # Nothing outside a bundle's own folder is ever unpacked into, or removed.
    home = tmp_path / "remote/data"
    home.mkdir()
    (home / "keep.txt").write_text("mine")
    code, out, _ = call("unpack", archive=str(archive), digest=bundle["digest"], target=str(home))
    assert code != 0 and "Not a bundle folder" in _ and (home / "keep.txt").read_text() == "mine"
    code, out, _ = call("unpack", archive=str(archive), digest=bundle["digest"], target=str(target))
    assert json.loads(out) == {"ready": True} and (target / "a.txt").read_text() == "a"


def test_a_file_on_the_machine_arrives_as_a_folder_holding_it_as_copies_do(lab):
    (lab.machine.home / "train.csv").write_text("a\n")
    (lab.project / "use.py").write_text(
        "import os\nfrom pathlib import Path\ndata = Path(os.environ['KGR_INPUT_DATA'])\n"
        "Path(os.environ['KGR_OUTPUT_DIR'], 'seen.txt').write_text((data / 'train.csv').read_text())\n"
    )
    inputs = {"data": Path(f"ssh:{lab.machine.home}/train.csv")}
    [job] = settle(
        lab.client,
        lab.client.submit(JobSpec(source=lab.project, entrypoint="use.py", internet=True, inputs=inputs)).id,
    )
    assert job.state == "succeeded" and (job.result_dir / "outputs/seen.txt").read_text() == "a\n", job.error


def test_links_are_followed_alike_when_attaching_fingerprinting_and_copying(lab):
    real = lab.machine.home / "real"
    real.mkdir()
    (real / "a.csv").write_text("1\n")
    (real / "gone").symlink_to(lab.machine.home / "deleted")
    (real / "b.csv").symlink_to(real / "a.csv")
    (lab.machine.home / "data").symlink_to(real)
    pinned = lab.machine.resolve_dataset(f"ssh:LAB:{lab.machine.home}/data".removeprefix("ssh:"))
    assert pinned and pinned.startswith(f"lab:{lab.machine.home}/data#")
    copy = lab.tmp / "copy"
    copy.mkdir()
    lab.machine.fetch_dataset(pinned, copy)
    assert sorted(os.listdir(copy)) == ["a.csv", "b.csv"] and (copy / "b.csv").read_text() == "1\n"
    (real / "loop").symlink_to(real)
    (lab.tmp / "copy2").mkdir()
    with pytest.raises(ValueError, match="Links to folders"):
        lab.machine.fetch_dataset(pinned, lab.tmp / "copy2")


def test_a_file_that_changes_while_downloaded_does_not_replace_the_saved_one(lab):
    (lab.project / "w.py").write_text(
        "import os\nfrom pathlib import Path\nPath(os.environ['KGR_OUTPUT_DIR'], 'm.pt').write_text('new')\n"
    )
    job = lab.client.submit(JobSpec(source=lab.project, entrypoint="w.py", internet=True))
    call = lab.machine._call

    def listed_before_a_change(command, **arguments):
        found = call(command, **arguments)
        if command == "files":
            for item in found["files"]:
                item["sha256"] = "0" * 64
        return found

    lab.machine._call = listed_before_a_change
    lab.client.worker().tick()
    for _ in range(100):
        job = lab.client.get(job.id)
        if job.download_error:
            break
        due(lab.client, job.id)
        lab.client.worker().tick()
        time.sleep(0.2)
    assert "changed while it was downloaded" in job.download_error
    assert not (job.result_dir / "outputs/m.pt").exists()
    lab.machine._call = call
    due(lab.client, job.id)
    lab.client.worker().tick()
    job = lab.client.get(job.id)
    assert job.download_state == "complete" and (job.result_dir / "outputs/m.pt").read_text() == "new"


class Channel:
    def __init__(self, code):
        self.code = code

    def recv_exit_status(self):
        return self.code


class Stream:
    def __init__(self, code, data=b""):
        self.channel, self.data = Channel(code), data

    def read(self):
        return self.data


def test_commands_without_an_exit_status_or_with_passing_failures_are_transient(lab, monkeypatch):
    def exits(code):
        client = SimpleNamespace(
            exec_command=lambda command, timeout: (None, Stream(code), Stream(code, b"boom"))
        )
        monkeypatch.setattr(SshProvider, "_ssh", lambda self: client)
        return SshProvider._exec(lab.machine, "true")

    with pytest.raises(RemoteError) as dropped:
        exits(-1)
    assert not dropped.value.definitive
    monkeypatch.undo()
    for code, definitive in [(ssh_remote.TRANSIENT_EXIT, False), (1, True)]:
        lab.machine._exec = lambda command, timeout=600, code=code: (code, "", "OSError: disk full")
        with pytest.raises(RemoteError) as error:
            lab.machine._call("info")
        assert error.value.definitive is definitive


def test_sftp_failures_of_the_helper_upload_are_remote_errors(lab, monkeypatch):
    def full(self, path, mode="r"):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(LocalSftp, "open", full)
    with pytest.raises(RemoteError, match="Uploading the helper"):
        lab.machine.info()


def test_a_supervisor_that_records_its_result_as_status_reads_is_not_a_failure(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "started").mkdir(parents=True)
    (run / "state.json").write_text(json.dumps({"state": "running", "pid": 1}))

    def finished_meanwhile(pid):
        (run / "state.json").write_text(json.dumps({"state": "succeeded", "pid": 1}))
        return False

    monkeypatch.setattr(ssh_remote, "_alive", finished_meanwhile)
    assert ssh_remote.status(str(run)) == {"state": "succeeded", "error": None}
    monkeypatch.setattr(ssh_remote, "_alive", lambda pid: True)
    signals = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: signals.append(pid))
    # Recorded by start: the supervisor may not handle SIGTERM yet, so only the cancel file asks it.
    (run / "state.json").write_text(json.dumps({"state": "running", "pid": 1}))
    assert ssh_remote.cancel(str(run)) == {"signalled": False} and signals == []
    assert (run / "cancel").exists()
    (run / "state.json").write_text(json.dumps({"state": "running", "pid": 1, "supervised": True}))
    assert ssh_remote.cancel(str(run)) == {"signalled": True} and signals == [1]
    monkeypatch.setattr(os, "kill", lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()))
    assert ssh_remote.cancel(str(run)) == {"signalled": False}


def test_logins_use_the_credentials_file_then_an_older_password_file(lab, tmp_path):
    password = tmp_path / "password"
    password.write_text("older\n")
    settings = lab.machine.settings.model_copy(update={"password_file": password})
    machine = SshProvider(lab.machine.account.model_copy(update={"ssh": settings}), lab.tmp)
    assert machine._secrets() == {"password": "older"}
    credentials_path().parent.mkdir(parents=True, exist_ok=True)
    credentials_path().write_text(json.dumps({"ssh:lab": {"key": "~/.ssh/lab", "passphrase": "p"}}))
    assert machine._secrets() == {"key": "~/.ssh/lab", "passphrase": "p"}
    credentials_path().write_text("{broken")
    with pytest.raises(RemoteError, match="not valid JSON") as error:
        machine._connect(None)
    assert error.value.kind == "auth" and error.value.definitive


def test_the_work_directory_is_a_folder_of_its_own_below_home():
    for bad in ["", ".", "..", "/", "/home/me", "a/../..", "~"]:
        if bad == "~":
            continue
        with pytest.raises(ValueError, match="below the home folder"):
            SshSettings(host="h", username="u", workdir=bad)
    assert SshSettings(host="h", username="u", workdir="./jobs/runner/").workdir == "jobs/runner"


def test_agent_accounts_tell_an_unlimited_gpu_quota_from_an_unchecked_one(lab):
    accounts = {a["id"]: a for a in lab.client.agent().accounts()["accounts"]}
    assert accounts["ssh:lab"]["gpu_quota_limited"] is False
    assert accounts["kaggle:tester"]["gpu_quota_limited"] is True


def test_gpus_count_a_machines_slots_without_a_time_limit(lab, monkeypatch):
    monkeypatch.setattr("compute_runner.cli.Client", lambda **_: lab.client)
    result = json.loads(CliRunner().invoke(app, ["--json", "gpus", "--live"]).output)
    machine = {account["id"]: account for account in result["accounts"]}["ssh:lab"]
    assert machine["gpu"] == {"used": 0, "limit": 2, "free": 2} and machine["gpu_quota_seconds"] is None
    totals = result["totals"]
    assert totals["gpu"] == {"used": 0, "limit": 3, "free": 3}
    assert totals["gpu_quota_seconds"] == 100000 and totals["complete"]


@pytest.mark.skipif(
    not os.environ.get("KGR_TEST_SSH"), reason="set KGR_TEST_SSH=user@host:port and KGR_TEST_SSH_KEY"
)
def test_real_ssh_server(tmp_path, monkeypatch):
    """The same flow over real SSH; the machine's host key must already be trusted."""
    login, _, address = os.environ["KGR_TEST_SSH"].partition("@")
    host, _, port = address.partition(":")
    monkeypatch.setenv("KGR_CONFIG_DIR", os.environ.get("KGR_TEST_SSH_CONFIG_DIR", str(tmp_path / "config")))
    settings = SshSettings(
        host=host,
        port=int(port or 22),
        username=login,
        key=Path(os.environ["KGR_TEST_SSH_KEY"]),
        workdir=f".compute-runner-test-{os.getpid()}",
    )
    config = Config(
        accounts=[Account(provider="ssh", user="real", ssh=settings)], state_dir=tmp_path / "state"
    )
    client = Client(config=config)
    (tmp_path / "p").mkdir()
    (tmp_path / "p/t.py").write_text(
        "import os\nfrom pathlib import Path\nPath(os.environ['KGR_OUTPUT_DIR'], 'ok.txt').write_text('ok')\n"
    )
    job = client.submit(JobSpec(source=tmp_path / "p", entrypoint="t.py", internet=True))
    try:
        [job] = settle(client, job.id, seconds=120)
        assert job.state == "succeeded" and (job.result_dir / "outputs/ok.txt").read_text() == "ok"
    finally:
        machine = client.provider("ssh:real")
        # Removes only this test's own work directory.
        assert settings.workdir.startswith(".compute-runner-test-")
        folder = shlex.quote(machine.workdir)
        machine._exec(f"chmod -R u+w {folder} && rm -rf {folder}")


def test_doctor_warns_when_the_machine_has_fewer_gpus_than_its_slots(lab):
    # The test machine has no nvidia-smi, so its two GPU slots are more than it has.
    found = lab.machine.diagnose()
    assert found["machine"]["gpus"] == []
    assert found["warning"] == "gpu_limit is 2, but nvidia-smi lists 0 GPUs"
    assert lab.machine.inventory().devices == []


def test_run_folders_and_bundles_are_listed_and_removed_but_never_while_running(lab):
    (lab.project / "wait.py").write_text("import time\ntime.sleep(3)\n")
    job = lab.client.submit(JobSpec(source=lab.project, entrypoint="wait.py", internet=True))
    running(lab.client, job.id)
    ref = lab.client.get(job.id).remote_ref
    found = {(item.kind, item.name): item for item in lab.machine.artifacts()}
    digest = job.snapshot["source"]["digest"]
    assert found[("run", ref)].attempt == ref and found[("bundle", digest)].digest == digest
    assert found[("bundle", digest)].bytes > 0
    with pytest.raises(RemoteError, match="still running"):
        lab.machine.delete_artifact(found[("run", ref)])
    settle(lab.client, job.id)
    for artifact in (found[("run", ref)], found[("bundle", digest)]):
        lab.machine.delete_artifact(artifact)
    assert lab.machine.artifacts() == []
    with pytest.raises(RemoteError, match="Not a run"):
        lab.machine.delete_artifact(Artifact("run", "../bundles"))

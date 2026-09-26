"""Preserve saved work and service ownership across the application rename."""

import pytest
from typer.testing import CliRunner

from compute_runner import Client, Config, JobSpec
from compute_runner import service
from compute_runner.cli import app
from compute_runner.paths import APP_NAME, LEGACY_APP_NAME, application_dir
from compute_runner.store import atomic_json, config_path


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    for prefix in ("COMPUTE_RUNNER", "KGR"):
        for kind in ("CONFIG", "STATE"):
            monkeypatch.delenv(f"{prefix}_{kind}_DIR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    return tmp_path


def test_fresh_install_uses_new_paths(isolated_paths):
    root = isolated_paths
    assert config_path() == root / "config" / APP_NAME / "config.json"
    assert Config().state_dir == root / "data" / APP_NAME


def test_upgrade_reopens_saved_jobs_and_request_receipts(isolated_paths):
    root = isolated_paths
    config = Config(owner="tester", state_dir=root / "data" / LEGACY_APP_NAME)
    atomic_json(root / "config" / LEGACY_APP_NAME / "config.json", config.model_dump(mode="json"))
    source = root / "train.py"
    source.write_text("print('saved workload')")
    original = Client(config=config)
    job = original.submit(JobSpec(source=source), request_key="preserve-intent")
    result = job.result_dir / "outputs/result.txt"
    result.parent.mkdir(parents=True)
    result.write_text("saved result")

    # An empty new directory must not hide a usable old queue/configuration.
    (root / "config" / APP_NAME).mkdir()
    (root / "data" / APP_NAME).mkdir()
    restored = Client()
    replay = restored.submit(JobSpec(source=source), request_key="preserve-intent")
    assert replay.id == job.id
    assert len(restored.list()) == 1
    assert (restored.get(job.id).result_dir / "outputs/result.txt").read_text() == "saved result"
    assert Config().state_dir == config.state_dir


def test_new_configuration_wins_when_both_exist(isolated_paths):
    root = isolated_paths
    for name in (LEGACY_APP_NAME, APP_NAME):
        config = Config(owner="tester", state_dir=root / "data" / name)
        atomic_json(root / "config" / name / "config.json", config.model_dump(mode="json"))
        Client(config=config)
    assert config_path() == root / "config" / APP_NAME / "config.json"
    assert Config().state_dir == root / "data" / APP_NAME


@pytest.mark.parametrize("kind", ["CONFIG", "STATE"])
def test_explicit_paths_and_legacy_environment_aliases(isolated_paths, monkeypatch, kind):
    root = isolated_paths
    monkeypatch.setenv(f"KGR_{kind}_DIR", str(root / "legacy-override"))
    assert application_dir(kind) == root / "legacy-override"
    monkeypatch.setenv(f"COMPUTE_RUNNER_{kind}_DIR", str(root / "new-override"))
    assert application_dir(kind) == root / "new-override"


def test_cli_config_directory_overrides_environment(isolated_paths, monkeypatch):
    root = isolated_paths
    monkeypatch.setenv("COMPUTE_RUNNER_CONFIG_DIR", str(root / "from-environment"))
    result = CliRunner().invoke(
        app, ["--config-dir", str(root / "from-cli"), "account", "add", "kaggle", "tester"]
    )
    assert result.exit_code == 0, result.output
    assert (root / "from-cli/config.json").is_file()
    assert not (root / "from-environment/config.json").exists()


def test_service_upgrade_disables_previous_worker_before_enabling_new(isolated_paths, monkeypatch):
    root = isolated_paths / "config/systemd/user"
    root.mkdir(parents=True)
    legacy = root / service.LEGACY_UNIT_NAME
    legacy.write_text(service.LEGACY_DESCRIPTION)
    calls = []
    monkeypatch.setattr(service.subprocess, "run", lambda command, **_: calls.append(command))
    path = service.install(Config(state_dir=isolated_paths / "state"))
    assert path.name == service.UNIT_NAME
    assert "-m compute_runner" in path.read_text()
    assert calls == [
        ["systemctl", "--user", "disable", "--now", service.LEGACY_UNIT_NAME],
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", service.UNIT_NAME],
    ]


def test_service_control_works_before_reinstallation(isolated_paths, monkeypatch):
    root = isolated_paths / "config/systemd/user"
    root.mkdir(parents=True)
    (root / service.LEGACY_UNIT_NAME).write_text(service.LEGACY_DESCRIPTION)
    calls = []
    monkeypatch.setattr(service.subprocess, "run", lambda command, **_: calls.append(command))
    service.control("status")
    assert calls[-1][3] == service.LEGACY_UNIT_NAME
    # Its ExecStart names the removed kaggle_runner package, so it must not be started again.
    with pytest.raises(ValueError, match="service install"):
        service.control("start")
    (root / service.UNIT_NAME).write_text(service.DESCRIPTION)
    service.control("status")
    assert calls[-1][3] == service.UNIT_NAME


def test_service_install_leaves_unrelated_units_alone(isolated_paths, monkeypatch):
    root = isolated_paths / "config/systemd/user"
    root.mkdir(parents=True)
    (root / service.LEGACY_UNIT_NAME).write_text("Description=Something else")
    calls = []
    monkeypatch.setattr(service.subprocess, "run", lambda command, **_: calls.append(command))
    service.install(Config(state_dir=isolated_paths / "state"), start=False)
    assert not any("disable" in command for command in calls)
    assert "--now" not in calls[-1]
    (root / service.UNIT_NAME).write_text("Description=Something else")
    with pytest.raises(ValueError, match="unrelated service"):
        service.install(Config(state_dir=isolated_paths / "state"))

import json

from typer.testing import CliRunner

from compute_runner import Client
from compute_runner.cli import app
from compute_runner.workloads import load_specs
from compute_runner.service import unit_text


def test_cli_init_dry_run_and_submit(tmp_path, monkeypatch):
    monkeypatch.setenv("KGR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("KGR_STATE_DIR", str(tmp_path / "state"))
    runner = CliRunner()
    result = runner.invoke(app, ["--json", "account", "add", "kaggle", "tester"])
    assert result.exit_code == 0, result.output
    script = tmp_path / "hello.py"
    script.write_text("print(42)")
    result = runner.invoke(app, ["--json", "submit", str(script), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["private"] is True
    assert not (tmp_path / "state/bundles").exists()
    result = runner.invoke(app, ["--json", "submit", str(script)])
    assert result.exit_code == 0, result.output
    job_id = json.loads(result.output)[0]["id"]
    result = runner.invoke(app, ["--json", "status", job_id[:12]])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["state"] == "queued"


def test_cli_status_redacts_nonsecret_environment_values(tmp_path, monkeypatch):
    monkeypatch.setenv("KGR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("KGR_STATE_DIR", str(tmp_path / "state"))
    runner = CliRunner()
    assert runner.invoke(app, ["--json", "account", "add", "kaggle", "tester"]).exit_code == 0
    script = tmp_path / "hello.py"
    script.write_text("print(42)")
    yaml = tmp_path / "job.yaml"
    yaml.write_text(f"source: {script}\nenv:\n  EXPERIMENT_SEED: private-display-value\n")
    submitted = runner.invoke(app, ["--json", "submit", str(yaml)])
    assert submitted.exit_code == 0, submitted.output
    job_id = json.loads(submitted.output)[0]["id"]
    assert "private-display-value" not in submitted.output
    status = runner.invoke(app, ["--json", "status", job_id])
    assert status.exit_code == 0, status.output
    assert json.loads(status.output)["spec"]["env"] == {"EXPERIMENT_SEED": "[redacted]"}


def test_cli_rejected_secret_env_does_not_echo_value(tmp_path, monkeypatch):
    monkeypatch.setenv("KGR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("KGR_STATE_DIR", str(tmp_path / "state"))
    runner = CliRunner()
    assert runner.invoke(app, ["--json", "account", "add", "kaggle", "tester"]).exit_code == 0
    script = tmp_path / "hello.py"
    script.write_text("print(42)")
    yaml = tmp_path / "job.yaml"
    rejected_value = "should-never-appear-in-errors"
    yaml.write_text(f"source: {script}\nenv:\n  API_TOKEN: {rejected_value}\n")
    result = runner.invoke(app, ["--json", "submit", str(yaml)])
    assert result.exit_code == 1
    assert rejected_value not in result.output


def test_yaml_relative_paths(tmp_path):
    config = tmp_path / "workload.yaml"
    config.write_text("source: project\nmodule: demo.main\ninputs:\n  data: input\n")
    spec = load_specs(config)[0]
    assert spec.source == tmp_path / "project"
    assert spec.inputs["data"] == tmp_path / "input"


def test_yaml_expands_home_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = tmp_path / "workload.yaml"
    config.write_text("source: ~/project\nmodule: demo.main\n")
    assert load_specs(config)[0].source == tmp_path / "home/project"


def test_settings_and_account_limits_persist(tmp_path, monkeypatch):
    monkeypatch.setenv("KGR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("KGR_STATE_DIR", str(tmp_path / "state"))
    runner = CliRunner()
    assert runner.invoke(app, ["account", "add", "kaggle", "tester"]).exit_code == 0
    assert Client().provider().strict is False  # Permissive unless requested.
    assert runner.invoke(app, ["account", "add", "kaggle", "tester", "--cpu-limit", "2"]).exit_code == 0
    assert runner.invoke(app, ["init", "--strict", "--failover", "auto"]).exit_code == 0
    assert runner.invoke(app, ["account", "add", "kaggle", "tester"]).exit_code == 0
    assert runner.invoke(app, ["init"]).exit_code == 0
    saved = json.loads((tmp_path / "config/config.json").read_text())
    assert [(a["user"], a["cpu_limit"]) for a in saved["accounts"]] == [("tester", 2)]
    assert saved["strict"] is True and saved["failover"] == "auto"
    assert Client().provider().strict is True
    result = runner.invoke(app, ["init", "--failover", "sometimes"])
    assert isinstance(result.exception, ValueError) and "failover" in str(result.exception)


def test_service_escapes_paths_and_uses_venv(setup):
    client, _, _ = setup
    unit = unit_text(client.config, python="/some path/with%percent/python")
    assert '"/some path/with%%percent/python"' in unit
    assert "Restart=on-failure" in unit
    assert "UMask=0077" in unit


def test_service_working_directory_is_not_quoted(setup):
    client, _, _ = setup
    assert f"WorkingDirectory={client.config.state_dir}\n" in unit_text(client.config)


def test_configuration_changes_remind_a_running_worker_to_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("KGR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("KGR_STATE_DIR", str(tmp_path / "state"))
    runner = CliRunner()
    assert "restart" not in runner.invoke(app, ["--json", "account", "add", "kaggle", "tester"]).stderr
    with Client().store.worker_lock():
        result = runner.invoke(app, ["--json", "account", "add", "kaggle", "other"])
    assert result.exit_code == 0 and "compute-runner service restart" in result.stderr
    assert json.loads(result.stdout)["accounts"] == ["kaggle:tester", "kaggle:other"]


def test_one_off_state_dir_is_not_saved_and_dry_run_checks_the_account(tmp_path, monkeypatch):
    monkeypatch.setenv("KGR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("KGR_STATE_DIR", str(tmp_path / "state"))
    runner = CliRunner()
    scratch = tmp_path / "scratch"
    result = runner.invoke(app, ["--state-dir", str(scratch), "account", "add", "kaggle", "tester"])
    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / "config/config.json").read_text())
    assert saved["state_dir"] == str((tmp_path / "state").resolve())
    script = tmp_path / "hello.py"
    script.write_text("print(42)")
    result = runner.invoke(app, ["--json", "submit", str(script), "--dry-run", "--timeout", "90000"])
    assert result.exit_code != 0 and "43200" in str(result.exception)

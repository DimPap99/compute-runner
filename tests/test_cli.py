import json

from typer.testing import CliRunner

from kaggle_runner.cli import app, load_specs
from kaggle_runner.service import unit_text


def test_cli_init_dry_run_and_submit(tmp_path, monkeypatch):
    monkeypatch.setenv("KGR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("KGR_STATE_DIR", str(tmp_path / "state"))
    runner = CliRunner()
    result = runner.invoke(app, ["--json", "init", "--owner", "tester"])
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
    assert runner.invoke(app, ["--json", "init", "--owner", "tester"]).exit_code == 0
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
    assert runner.invoke(app, ["--json", "init", "--owner", "tester"]).exit_code == 0
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


def test_service_escapes_paths_and_uses_venv(setup):
    client, _, _ = setup
    unit = unit_text(client.config, python="/some path/with%percent/python")
    assert '"/some path/with%%percent/python"' in unit
    assert "Restart=on-failure" in unit
    assert "UMask=0077" in unit


def test_service_working_directory_is_not_quoted(setup):
    client, _, _ = setup
    assert f"WorkingDirectory={client.config.state_dir}\n" in unit_text(client.config)

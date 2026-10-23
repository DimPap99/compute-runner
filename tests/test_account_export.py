"""Moving accounts to another computer: account export and account import."""

import base64
import json

import paramiko
import pytest
from typer.testing import CliRunner

from compute_runner import Account, Client, account_export
from compute_runner.cli import app
from compute_runner.credentials import account_secrets, credentials_path, imported_key_path
from compute_runner.models import SshSettings
from compute_runner.providers import RemoteError
from compute_runner.providers.ssh import import_host_keys, known_hosts_path
from compute_runner.security import detected_secret

PASSPHRASE = "correct horse battery"
TWICE = f"{PASSPHRASE}\n{PASSPHRASE}\n"
# An SSH key file as it is, CRLF and all; the export carries its bytes unchanged.
KEY = b"-----BEGIN OPENSSH PRIVATE KEY-----\r\nb3BlbnNzaC1rZXktdjE=\r\n-----END OPENSSH PRIVATE KEY-----\r\n"


@pytest.fixture
def computer(tmp_path, monkeypatch):
    """use(NAME) switches to a computer of its own: home folder, configuration and state."""
    monkeypatch.setattr(account_export, "SCRYPT", dict(n=2**10, r=8, p=1))  # Quick, for tests only.

    def use(name):
        root = tmp_path / name
        monkeypatch.setenv("HOME", str(root / "home"))
        monkeypatch.setenv("KGR_CONFIG_DIR", str(root / "config"))
        monkeypatch.setenv("KGR_STATE_DIR", str(root / "state"))
        return root

    return use


def run(*args, input=None):
    result = CliRunner().invoke(app, ["--json", *args], input=input)
    if result.exception and not isinstance(result.exception, SystemExit):
        raise result.exception
    return json.loads(result.output.splitlines()[-1])


def trust(path, name, key):
    known = paramiko.HostKeys()
    if path.is_file():
        known.load(str(path))
    known.add(name, key.get_name(), key)
    path.parent.mkdir(parents=True, exist_ok=True)
    known.save(str(path))


def known_key(name):
    return dict(paramiko.HostKeys(str(known_hosts_path())).lookup(name) or {})


@pytest.fixture
def old_computer(computer, tmp_path):
    """Accounts with every kind of login, and an export of them in tmp_path/accounts.export."""
    root = computer("old")
    kaggle_json = tmp_path / "kaggle.json"
    kaggle_json.write_text(json.dumps({"username": "alice", "key": "a" * 32}))
    run("account", "add", "kaggle", "alice", "--credentials", str(kaggle_json), "--gpu-limit", "2")
    run("account", "add", "kaggle", "bob", "--enter-key", input="KGAT_typed_token\n")
    run("account", "add", "kaggle", "carol")  # Kaggle's defaults, which stay on this computer.
    key = root / "home/.ssh/id_lab"
    key.parent.mkdir(parents=True)
    key.write_bytes(KEY)
    lab = ["ssh", "lab", "--host", "lab.example", "--login", "me", "--workdir", "jobs", "--gpu-limit", "1"]
    run("account", "add", *lab, "--key", str(key))
    cpu = ["ssh", "cpu", "--host", "cpu.example", "--port", "2222", "--login", "me", "--enter-password"]
    run("account", "add", *cpu, input="hunter2\n")
    saved = json.loads(credentials_path().read_text())
    saved["ssh:lab"]["passphrase"] = "unlocks the key"
    credentials_path().write_text(json.dumps(saved))
    # One machine trusted by the runner, the other by ssh itself.
    lab_key, cpu_key = paramiko.ECDSAKey.generate(), paramiko.ECDSAKey.generate()
    trust(known_hosts_path(), "lab.example", lab_key)
    trust(root / "home/.ssh/known_hosts", "[cpu.example]:2222", cpu_key)
    file = tmp_path / "accounts.export"
    result = run("account", "export", str(file), input=TWICE)
    return result, file, lab_key, cpu_key


IDS = ["kaggle:alice", "kaggle:bob", "kaggle:carol", "ssh:lab", "ssh:cpu"]


def test_accounts_move_to_another_computer_with_their_secrets_and_host_keys(old_computer, computer, tmp_path):
    exported, file, lab_key, cpu_key = old_computer
    assert exported["accounts"] == IDS and exported["file"] == str(file)
    assert exported["without_secrets"] == {"kaggle:carol": "Kaggle's default (~/.kaggle, KAGGLE_*)"}
    assert "without_host_keys" not in exported
    text = file.read_bytes()
    for secret in [b"a" * 32, b"KGAT_typed_token", b"hunter2", b"unlocks the key", b"b3BlbnNzaC1rZXktdjE"]:
        assert secret not in text
    assert oct(file.stat().st_mode & 0o777) == "0o600"
    # A source or input bundle that holds an export is refused, encrypted or not.
    assert detected_secret(text) == "account export"

    new = computer("new")
    imported = run("account", "import", str(file), input=PASSPHRASE + "\n")
    assert imported["added"] == IDS and imported["updated"] == [] and imported["accounts"] == IDS
    assert imported["host_keys"] == {"ssh:lab": "saved", "ssh:cpu": "saved"}
    config = Client().config
    assert config.account("kaggle:alice").gpu_limit == 2
    lab = config.account("ssh:lab")
    assert (lab.ssh.host, lab.ssh.workdir, lab.gpu_limit, lab.ssh.key) == ("lab.example", "jobs", 1, None)
    assert config.account("ssh:cpu").ssh.port == 2222
    assert account_secrets("kaggle:alice") == {"username": "alice", "key": "a" * 32}
    assert account_secrets("kaggle:bob") == {"token": "KGAT_typed_token"}
    assert account_secrets("kaggle:carol") == {}
    assert account_secrets("ssh:cpu") == {"password": "hunter2"}
    key = imported_key_path("ssh:lab")
    assert account_secrets("ssh:lab") == {"key": str(key), "passphrase": "unlocks the key"}
    assert key.read_bytes() == KEY and oct(key.stat().st_mode & 0o777) == "0o600"
    assert known_key("lab.example") == {lab_key.get_name(): lab_key}
    assert known_key("[cpu.example]:2222") == {cpu_key.get_name(): cpu_key}
    # Nothing here points back at the old computer.
    for path in (credentials_path(), new / "config/config.json"):
        assert str(tmp_path / "old") not in path.read_text()
    # Importing again changes nothing.
    again = run("account", "import", str(file), input=PASSPHRASE + "\n")
    assert again["added"] == [] and again["updated"] == IDS
    assert again["host_keys"] == {"ssh:lab": "already known", "ssh:cpu": "already known"}


def test_import_updates_accounts_already_here_and_keeps_what_the_export_lacks(old_computer, computer):
    _, file, *_ = old_computer
    computer("new")
    run("account", "add", "kaggle", "zed")
    run("account", "add", "kaggle", "CAROL", "--enter-key", input="KGAT_carol_here\n")
    run("account", "add", "ssh", "lab", "--host", "lab.example", "--login", "me", "--cpu-limit", "9")
    # This computer knows the machine by another key: it stays, and the import says why.
    other = paramiko.ECDSAKey.generate()
    trust(known_hosts_path(), "lab.example", other)
    imported = run("account", "import", str(file), input=PASSPHRASE + "\n")
    assert imported["updated"] == ["kaggle:CAROL", "ssh:lab"]
    assert imported["added"] == ["kaggle:alice", "kaggle:bob", "ssh:cpu"]
    # Accounts already here keep their place and ID; new ones follow in the export's order.
    assert imported["accounts"] == [
        "kaggle:zed",
        "kaggle:CAROL",
        "ssh:lab",
        "kaggle:alice",
        "kaggle:bob",
        "ssh:cpu",
    ]
    assert "does not match the saved one" in imported["host_keys"]["ssh:lab"]
    assert imported["host_keys"]["ssh:cpu"] == "saved"
    assert known_key("lab.example") == {other.get_name(): other}
    # carol was exported without secrets, so hers here are kept; lab takes the export's settings.
    assert account_secrets("kaggle:carol") == {"token": "KGAT_carol_here"}
    assert Client().config.account("ssh:lab").cpu_limit == 5
    assert account_secrets("ssh:lab")["key"] == str(imported_key_path("ssh:lab"))


def test_an_imported_key_is_deleted_with_the_secrets_that_name_it(old_computer, computer, tmp_path):
    _, file, *_ = old_computer
    computer("new")
    run("account", "import", str(file), input=PASSPHRASE + "\n")
    key = imported_key_path("ssh:lab")
    password = tmp_path / "password"
    password.write_text("swordfish\n")
    run("account", "add", "ssh", "lab", "--password-file", str(password))
    assert not key.exists() and account_secrets("ssh:lab") == {"password": "swordfish"}
    run("account", "import", str(file), input=PASSPHRASE + "\n")
    assert key.exists()
    run("account", "remove", "ssh:lab")
    assert not key.exists() and account_secrets("ssh:lab") == {}


def test_export_of_chosen_accounts_and_what_it_refuses(old_computer, tmp_path):
    _, file, *_ = old_computer
    with pytest.raises(ValueError, match="already exists"):
        run("account", "export", str(file), input=TWICE)
    chosen = tmp_path / "chosen.export"
    result = run(
        "account", "export", str(chosen), "--account", "KAGGLE:BOB", "--account", "ssh:cpu", input=TWICE
    )
    assert result["accounts"] == ["kaggle:bob", "ssh:cpu"] and "without_secrets" not in result
    with pytest.raises(ValueError, match="Unknown account kaggle:nobody"):
        run("account", "export", str(tmp_path / "x"), "--account", "kaggle:nobody", input=TWICE)
    with pytest.raises(ValueError, match="at least 8 characters"):
        run("account", "export", str(tmp_path / "short"), input="short\nshort\n")
    assert not (tmp_path / "short").exists()
    # A key file that is gone fails the export before the passphrase is asked for.
    (tmp_path / "old/home/.ssh/id_lab").unlink()
    with pytest.raises(ValueError, match="Cannot read the credentials of ssh:lab"):
        run("account", "export", str(tmp_path / "broken"))
    assert not (tmp_path / "broken").exists()


def test_import_refuses_wrong_passphrases_damage_and_other_files(old_computer, computer, tmp_path):
    _, file, *_ = old_computer
    computer("new")
    with pytest.raises(ValueError, match="Wrong passphrase, or the export is damaged"):
        run("account", "import", str(file), input="not the passphrase\n")
    envelope = json.loads(file.read_text())
    data = bytearray(base64.b64decode(envelope["data"]))
    data[-1] ^= 1
    damaged = tmp_path / "damaged.export"
    damaged.write_text(json.dumps(envelope | {"data": base64.b64encode(data).decode()}))
    with pytest.raises(ValueError, match="Wrong passphrase, or the export is damaged"):
        run("account", "import", str(damaged), input=PASSPHRASE + "\n")
    newer = tmp_path / "newer.export"
    newer.write_text(json.dumps(envelope | {"version": 2}))
    with pytest.raises(ValueError, match="another version of compute-runner"):
        run("account", "import", str(newer))
    for text in ["{}", "not json", '["compute-runner-accounts"]']:
        (tmp_path / "other").write_text(text)
        with pytest.raises(ValueError, match="is not an account export"):
            run("account", "import", str(tmp_path / "other"))
    assert Client().config.accounts == [] and not credentials_path().exists()


def test_host_keys_of_types_this_paramiko_cannot_read_are_skipped(computer):
    computer("new")
    account = Account(provider="ssh", user="lab", ssh=SshSettings(host="lab.example", username="me"))
    key = paramiko.ECDSAKey.generate()
    # Such as DSA host keys exported by paramiko 3, which paramiko 4 cannot read.
    unknown = {"ssh-future@example.com": base64.b64encode(b"key").decode()}
    with pytest.raises(RemoteError, match="No usable host key of lab.example"):
        import_host_keys(account, unknown)
    with pytest.raises(RemoteError, match="host key of lab.example is damaged"):
        import_host_keys(account, {key.get_name(): base64.b64encode(b"junk").decode()})
    assert import_host_keys(account, unknown | {key.get_name(): key.get_base64()})
    assert known_key("lab.example") == {key.get_name(): key}

import importlib.util
import pickle
import sys
from pathlib import Path

import pytest


ASSET = Path(__file__).parents[1] / "skills/compute-runner/assets/checkpointing.py"


@pytest.fixture(scope="module")
def checkpointing():
    spec = importlib.util.spec_from_file_location("checkpointing_asset", ASSET)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dump_pickle(value, path):
    with path.open("wb") as stream:
        pickle.dump(value, stream)


def load_pickle(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def test_minute_and_epoch_cadence(checkpointing):
    now = [100.0]
    minutes = checkpointing.CheckpointCadence("minutes", 2, clock=lambda: now[0])
    assert not minutes.due()
    now[0] = 220.0
    assert minutes.due()
    minutes.mark_saved()
    assert not minutes.due()

    epochs = checkpointing.CheckpointCadence("epochs", 3)
    assert not epochs.due(completed_epochs=2)
    assert epochs.due(completed_epochs=3)


@pytest.mark.parametrize(
    ("mode", "every"),
    [("steps", 1), ("minutes", 0), ("epochs", 1.5)],
)
def test_invalid_cadence_is_rejected(checkpointing, mode, every):
    with pytest.raises(ValueError):
        checkpointing.CheckpointCadence(mode, every)


def test_save_and_required_restore(checkpointing, tmp_path):
    compatibility = {"model": "tiny", "optimizer": "adamw"}
    writer = checkpointing.CheckpointManager(tmp_path / "new", compatibility=compatibility, suffix=".pickle")
    saved = writer.save(
        {"weights": [1, 2], "global_step": 9},
        dump=dump_pickle,
        global_step=9,
        completed_epochs=2,
    )
    assert saved.name == "checkpoint-step-000000000009.pickle"
    assert (saved.parent / "latest.json").is_file()
    assert not list(saved.parent.glob(".*.tmp"))

    reader = checkpointing.CheckpointManager(
        tmp_path / "replacement",
        resume_input=saved.parent,
        compatibility=compatibility,
        suffix=".pickle",
    )
    restored = reader.restore("required", load=load_pickle)
    assert restored.state["global_step"] == 9
    assert restored.metadata["completed_epochs"] == 2


def test_auto_without_input_starts_fresh(checkpointing, tmp_path, monkeypatch):
    monkeypatch.delenv("KGR_INPUT_RESUME", raising=False)
    manager = checkpointing.CheckpointManager(tmp_path / "new")
    assert manager.restore("auto", load=load_pickle) is None
    with pytest.raises(checkpointing.CheckpointError, match="required"):
        manager.restore("required", load=load_pickle)


def test_corruption_and_incompatibility_fail_loudly(checkpointing, tmp_path):
    writer = checkpointing.CheckpointManager(tmp_path / "saved", compatibility={"model": "a"})
    saved = writer.save({}, dump=dump_pickle, global_step=1, completed_epochs=0)

    incompatible = checkpointing.CheckpointManager(
        tmp_path / "new", resume_input=saved.parent, compatibility={"model": "b"}
    )
    with pytest.raises(checkpointing.CheckpointError, match="incompatible"):
        incompatible.restore("required", load=load_pickle)

    saved.write_bytes(b"corrupt")
    reader = checkpointing.CheckpointManager(
        tmp_path / "new", resume_input=saved.parent, compatibility={"model": "a"}
    )
    with pytest.raises(checkpointing.CheckpointError, match="checksum"):
        reader.restore("required", load=load_pickle)

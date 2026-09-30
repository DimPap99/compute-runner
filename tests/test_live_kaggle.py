"""The main user journey on a real Kaggle account, paced so Kaggle does not rate-limit it.

Opt-in, since it launches notebooks and creates private datasets on the account:

    KGR_TEST_KAGGLE=USERNAME KGR_TEST_KAGGLE_CREDENTIALS=path/to/kaggle.json-or-token \\
        pytest tests/test_live_kaggle.py

KGR_TEST_KAGGLE_GPU=1 trains on a GPU (a minute or two of GPU quota). KGR_TEST_KAGGLE_PACE sets the
seconds between worker cycles (default 20).

It runs like the service: one long-lived worker, whose run discovery is reused for
discovery_seconds. Starting a fresh worker every few seconds rediscovers every recent notebook,
one status call each, and Kaggle answers 429 Too Many Requests.
"""

import json
import os
import shutil
import time
from pathlib import Path

import pytest

from compute_runner import Account, Client, Config, JobSpec

ACCOUNT = os.environ.get("KGR_TEST_KAGGLE")
PACE = float(os.environ.get("KGR_TEST_KAGGLE_PACE", "20"))
GPU = os.environ.get("KGR_TEST_KAGGLE_GPU") == "1"
CHECKPOINTING = Path(__file__).parents[1] / "skills/compute-runner/assets/checkpointing.py"

pytestmark = pytest.mark.skipif(
    not (ACCOUNT and os.environ.get("KGR_TEST_KAGGLE_CREDENTIALS")),
    reason="set KGR_TEST_KAGGLE=USERNAME and KGR_TEST_KAGGLE_CREDENTIALS=FILE to run on real Kaggle",
)

TRAIN = """
import argparse, json, os, sys
from pathlib import Path
from checkpointing import CheckpointManager, add_checkpoint_arguments

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=2)
add_checkpoint_arguments(parser)
args = parser.parse_args()
if os.environ.get("KGR_TEST_GPU") == "1":
    import torch
    assert torch.cuda.is_available(), "no GPU"
out = Path(os.environ["KGR_OUTPUT_DIR"])
manager = CheckpointManager(out / "checkpoints")
restored = manager.restore(args.resume, load=lambda p: json.loads(p.read_text()))
start = restored.state["epoch"] if restored else 0
manager.save({"epoch": start + 1}, dump=lambda s, p: p.write_text(json.dumps(s)),
             global_step=start + 1, completed_epochs=start + 1)
if start + 1 < args.epochs:
    sys.exit(3)  # A session limit, halfway through.
(out / "metrics.json").write_text(json.dumps({"resumed_from": start}))
"""

READ = """
import json, os
from pathlib import Path
data = Path(os.environ["KGR_INPUT_DATA"])
lines = sum(len(p.read_text().splitlines()) for p in data.rglob("*.csv"))
Path(os.environ["KGR_OUTPUT_DIR"], "data.json").write_text(json.dumps({"csv_lines": lines}))
"""


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setenv("KGR_CONFIG_DIR", str(tmp_path / "config"))
    config = Config(
        accounts=[Account(user=ACCOUNT, credentials=Path(os.environ["KGR_TEST_KAGGLE_CREDENTIALS"]))],
        state_dir=tmp_path / "state",
        poll_seconds=PACE,
        failover="off",
    )
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy(CHECKPOINTING, project)
    (project / "train.py").write_text(TRAIN)
    (project / "read.py").write_text(READ)
    client = Client(config=config)
    worker = client.worker()
    # Every discovery answer over the whole run, so a rate limit that later cleared is still seen.
    worker.limited = []
    refresh = worker.discoverer.refresh

    def watched(account):
        found = refresh(account)
        if found.error and "429" in found.error:
            worker.limited.append(found.error)
        return found

    worker.discoverer.refresh = watched
    return client, worker, project


def settle(client, worker, *job_ids, minutes=30):
    """Paced cycles of one worker until the jobs finish and download, or are blocked."""
    deadline = time.monotonic() + minutes * 60
    while time.monotonic() < deadline:
        worker.tick()
        jobs = [client.get(job_id) for job_id in job_ids]
        done = [job.state == "blocked" or job.terminal and job.download_state != "pending" for job in jobs]
        if all(done) and all(job.download_state != "downloading" for job in jobs):
            return jobs
        time.sleep(PACE)
    raise AssertionError([(job.spec.name, job.state, job.download_state, job.error) for job in jobs])


def test_train_continue_datasets_and_blocking_on_real_kaggle(live):
    client, worker, project = live
    env = {"KGR_TEST_GPU": "1"} if GPU else {}
    train = client.submit(
        JobSpec(
            source=project,
            entrypoint="train.py",
            name="live-train",
            gpu=GPU,
            env=env,
            timeout_seconds=1200,
            args=["--checkpoint-mode", "epochs", "--checkpoint-every", "1"],
        )
    )
    public = client.submit(
        JobSpec(
            source=project, entrypoint="read.py", name="live-data", inputs={"data": Path("kaggle:uciml/iris")}
        )
    )
    typo = client.submit(
        JobSpec(
            source=project,
            entrypoint="read.py",
            name="live-typo",
            inputs={"data": Path("kaggle:uciml/irs-nope")},
        )
    )
    train, public, typo = settle(client, worker, train.id, public.id, typo.id)

    # A misspelled dataset is refused before anything is launched.
    assert typo.state == "blocked" and "No connected account can find dataset" in typo.error
    assert not typo.attempts
    # A public dataset is attached and read under its alias.
    assert public.state == "succeeded", public.error
    assert json.loads((public.result_dir / "outputs/data.json").read_text())["csv_lines"] > 100
    # The first session stops at its limit with a checkpoint; continue resumes from it.
    assert train.state == "failed" and (train.result_dir / "outputs/checkpoints/latest.json").is_file()
    resumed = client.continue_run(train.id)
    [resumed] = settle(client, worker, resumed.id)
    assert resumed.state == "succeeded", resumed.error
    assert json.loads((resumed.result_dir / "outputs/metrics.json").read_text()) == {"resumed_from": 1}
    assert "RESUMED_FROM step=1" in (resumed.result_dir / "run.log").read_text()

    # Paced like the service, the account was never rate-limited.
    assert worker.limited == []

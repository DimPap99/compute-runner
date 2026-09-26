# Kaggle Runner

A local Python API and CLI for running private Python projects, scripts, and notebooks on Kaggle. Submit a workload once; a persistent worker waits for capacity, uploads inputs, follows its status, and downloads results. Source files are snapshotted at submission, so editing your project cannot change an already queued run.

## Quick start

Python 3.12 or newer is required. This project has its own environment and does not depend on a research repository.

```bash
cd ~/kaggle-runner
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e . --no-deps
source .venv/bin/activate

# Reuses standard Kaggle credentials; never embeds them in workloads.
kgr init --owner YOUR_KAGGLE_USERNAME
kgr doctor
kgr service install

kgr submit examples/hello.py --dry-run
kgr submit examples/hello.py
kgr list
kgr status JOB_ID
kgr logs JOB_ID --follow
kgr wait JOB_ID
```

`kgr service install` enables and starts a **systemd user service**. It survives terminal closure, restarts after a process crash, and starts on login. It does not enable user lingering: scheduling can stop after logout and while the machine is off. Already submitted Kaggle jobs continue remotely; the worker reconciles them when it resumes. Alternatively, run `kgr worker run` in a terminal, or `kgr worker run --once` for one complete cycle. Only one worker may own a state directory.

Use `kgr service stop`, `start`, `restart`, and `status` to manage the service. Worker diagnostics are also available with `kgr worker status` and `journalctl --user -u kaggle-runner.service -f`.

## Python API

Install this package into another Python environment with `pip install -e ~/kaggle-runner` if you want to import it there. The CLI's own environment is isolated.

```python
from kaggle_runner import Client, JobSpec

client = Client()
job = client.submit(JobSpec(
    source="/path/to/project",
    module="experiments.train",
    args=["--epochs", "20"],
    gpu=True,
    accelerator="NvidiaTeslaT4",
    internet=False,
    datasets=["yourname/prepared-data"],
    timeout_seconds=3600,
))

print(job.id)
finished = client.wait(job.id, timeout=7200)
print(finished.state, finished.error, finished.result_dir)
```

`submit` only snapshots files and adds a local queue entry. A running worker is required to launch jobs. Importing the package, creating a client, and previewing a workload do not authenticate to Kaggle.

Available operations:

| Operation | Behavior |
| --- | --- |
| `preview(spec)` | Inspect selected paths, sizes, resources, and attachments without uploading |
| `submit(spec, request_key=None)`, `submit_many(specs, request_key=None)` | Snapshot and atomically queue jobs; optionally deduplicate requests |
| `submit_batch(specs, request_key=None)`, `batch(id)` | Submit/read a durable batch with ordered jobs and a stable batch ID |
| `agent()` | Compact, bounded JSON-compatible API for LLMs and other automation |
| `get(id)`, `list(states=None)` | Read the worker's saved state |
| `wait(id, timeout=None, downloads=True)` | Wait for execution and automatic downloads; return early for blocked/uncertain work |
| `logs(id, follow=False)` | Yield persisted logs or attach to the live log stream |
| `download(id)` | Retrieve outputs for a terminal run, even when automatic downloads were disabled |
| `retry(id)` | Explicitly create a new job from a terminal/blocked job's saved files and settings |
| `cancel(id)` | Cancel work that has not been remotely submitted |
| `quota()` | Query GPU/TPU usage, reservations, remaining time, and refresh time |
| `worker_health()` | Report lock ownership, heartbeat age, and current worker activity |

Retries do not reread the original project. To change code or configuration, submit a new `JobSpec`. Batch submission snapshots every specification before committing jobs and their batch receipt in one SQLite transaction. If snapshotting or insertion fails, no jobs in that batch are queued (unused local content-addressed bundles may remain). Runs progress independently after the batch is committed.

## Agent interface

`kgr agent` emits one compact JSON object without needing `--json`. It uses the same persistent worker as the normal CLI; it does not call any model or consume LLM tokens for background monitoring. Full job records, source inventories, and environment values are omitted from summaries. The ordinary CLI's output remains compatible.

```bash
# One request can contain many jobs, with their own GPU/internet settings.
kgr agent submit examples/batch.yaml --request-key experiment-v1 --dry-run
kgr agent submit examples/batch.yaml --request-key experiment-v1

# Recover current state even in a new conversation.
kgr agent status
kgr agent status --batch BATCH_ID
kgr agent status JOB_ID_1 JOB_ID_2
kgr agent status --state running --state failed

# Start from 0, then reuse the returned cursor with the same batch filter.
kgr agent changes --batch BATCH_ID --after 0
kgr agent changes --batch BATCH_ID --after RETURNED_CURSOR

# First call fetches a snapshot; later reads use the private cache.
kgr agent logs JOB_ID --tail 50 --max-bytes 8192
kgr agent logs JOB_ID --refresh
kgr agent health

# Explicit retry of the saved code. Repeating this retry key is safe.
kgr agent retry JOB_ID --request-key experiment-retry-v1
kgr agent cancel JOB_ID
```

The submit command also accepts `--gpu/--cpu`, `--internet/--no-internet`, `--module`, `--entrypoint`, `--accelerator`, `--timeout`, and repeated `--arg`. CLI overrides apply to every job in the input YAML. `--dry-run` reports total files/bytes and resource counts; use ordinary `kgr submit --dry-run` when the full upload inventory is needed.

### Response and cursor contract

- All successful agent responses have `schema_version: 1`. Operation errors return a short JSON `error` and exit status 1. CLI syntax errors use the standard CLI error output.
- Job summaries contain `id`, `batch_id`, `name`, `state`, `resource`, `internet`, `downloads`, and `outputs_ready`. Jobs belonging to a batch also carry a zero-based `batch_index` matching their input position; batch status preserves that order. When available, summaries include a Kaggle `url`, bounded `reason`/`error`/`download_error`, and `output_dir` after downloads complete. A computation finishing successfully does not imply its downloads are ready.
- Status and submission responses include `total`, aggregate `counts`, `jobs`, `next_offset`, and compact worker health. Default page size is 20, maximum 100. Follow `next_offset` with `status --batch ID --offset N`; counts cover the whole selected batch. Job-ID selections accept at most 100 IDs. Global status includes older jobs whose `batch_id` is null.
- `changes` returns `cursor`, `has_more`, and the latest summary of each job with an event after the input cursor. Repeated events for a job are coalesced; this is not a historical event log. Read more pages while `has_more` is true. Each page and cursor share a SQLite read snapshot, so changes occurring during a read remain visible on a later call. A job that changes between pages may appear again.
- Keep a cursor per state directory and batch filter. Reset to 0 after changing filters or losing the cursor. Unrelated batches can advance a filtered cursor without returning jobs. Idle polls and worker heartbeats do not create job events; download errors do. Do not poll through an LLM for every worker cycle.
- Log `text` defaults to the last 50 lines, capped at 8192 UTF-8 bytes (maximum 500 lines / 65536 bytes). Results indicate `truncated`, `total_bytes`, cache time, and a `path` to the full text. A byte boundary can leave a partial first line. Log retrieval never follows a stream. `--refresh` fetches again; otherwise cached reads work offline. A failed refresh preserves the old cache. Full log fetching may still transfer a large response from Kaggle; only the agent-facing output is bounded.

### Duplicate-safe requests

Agent submit and retry require `--request-key`; ordinary `kgr submit` also accepts it. Keys are 1–128 letters, digits, dots, underscores, colons, slashes, or hyphens, starting with a letter or digit. They are unique within the state directory and shared by submit/retry operations.

Use one stable key for one intended submission. Replaying the same normalized specs with the same key returns the original batch and `replayed: true`, including after process restarts or completion. Concurrent replays commit only one batch. A key reused with different settings fails. The key identifies the request, **not current source contents**: changed or missing local files do not change an already accepted request. Use a new key for new code or an intentional rerun. Receipts remain on disk; no automatic expiration or deletion is performed.

Python callers can use the same compact interface:

```python
from kaggle_runner import Client, JobSpec

agent = Client().agent()  # Or AgentClient() for the default configuration.
submitted = agent.submit([
    JobSpec(source="/path/analyze.py"),
    JobSpec(source="/path/train.py", gpu=True, internet=True),
], request_key="experiment-v1")
batch_id = submitted["batch_id"]
page = agent.changes(batch_id=batch_id, after=0)
cursor = page["cursor"]
# Later: agent.changes(batch_id=batch_id, after=cursor)
# If page["has_more"], read the remaining pages immediately with that cursor.
```

`agent.status(job_ids=None, batch_id=None, states=None, limit=20, offset=0)`, `agent.logs(id, tail=50, max_bytes=8192, refresh=False)`, `agent.retry(id, request_key=...)`, `agent.cancel(id)`, and `agent.health()` round out the API. Read/status/change operations do not contact Kaggle. Submission is local, with remote work performed by the daemon.

### Codex skill

The reusable skill is in `skills/kaggle-runner/` and is installed locally at `~/.codex/skills/kaggle-runner/`. It teaches compact commands, batching, cursor use, and recovery. Use `$kaggle-runner` or ask to run/monitor Kaggle workloads in a session that discovers the skill. If the current skill list is already loaded, start a new session or explicitly reference the installed `SKILL.md`.

For another local installation, copy that skill folder into your Codex skills directory. The skill expects `kgr` on PATH or `~/kaggle-runner/.venv/bin/kgr`. Any agent with local shell access can use the CLI; no MCP server or LLM API key is needed. An MCP wrapper for clients without shell access is not included.

This interface queues jobs as capacity allows. Timed/recurring schedules and a trigger to resume an LLM conversation when jobs finish are not implemented. The worker continues while it is running, independently of the conversation.

Existing v1 queue databases are migrated additively to schema v2, retaining job IDs, events, and snapshots. Older application versions cannot open the upgraded database. Stop the worker and back up the SQLite database before downgrading.

## Workload definitions

A file source uploads only that `.py` or `.ipynb` file. A folder source includes its selected files and needs either `entrypoint` or `module`.

```yaml
name: training-experiment
source: ./project
module: experiments.train
# Alternatively: entrypoint: scripts/train.py or notebooks/analysis.ipynb
args: ["--epochs", "20"]
env:
  EXPERIMENT_SEED: "42"
gpu: true
accelerator: NvidiaTeslaT4
internet: false
timeout_seconds: 3600

datasets: [yourname/prepared-data/3]
inputs:
  extra_data: ./data

# Optional; requires internet: true. Relative to the project root.
# requirements: requirements.txt

exclude: ["outputs/", "checkpoints/"]
auto_download: true
output_patterns: ["outputs/*.json", "outputs/*.pt"]
```

Run `kgr submit workload.yaml`. Paths in YAML are relative to that YAML file; the entrypoint and requirements paths are relative to the source folder. `examples/batch.yaml` demonstrates a `jobs:` list. CLI overrides include `--entrypoint`, `--module`, `--gpu/--cpu`, `--internet/--no-internet`, `--accelerator`, `--timeout`, and repeatable `--arg`.

Use `kgr --json list`, `kgr --json status ID`, or `kgr --json submit workload.yaml` for automation. `logs --follow` with `--json` emits one JSON object per log chunk. CLI job IDs may be unambiguous prefixes; Python API IDs are the complete returned identifiers.

### Files, inputs, and dependencies

- Source folder selection respects root `.gitignore`, `.kgrignore`, and `exclude` patterns. Add large generated output directories to these files or to `exclude`.
- Credentials such as `.env`, `.env.*`, `kaggle.json`, access-token files, private-key files, and credential directories are excluded regardless of ignore negations. Git metadata, virtual environments, caches, and `node_modules` are excluded. Symlinks are rejected. This is a filename-based exclusion policy, not a scanner for secrets embedded in ordinary source files.
- Notebook cell outputs and execution counts are cleared in the snapshot. The original notebook is unchanged.
- Single-file workloads embed their saved file in a generated private kernel. Folders and explicitly selected local inputs become separate private datasets. Identical content reuses the same dataset, and managed datasets are never updated in place. Their metadata preserves original licensing terms rather than granting a new license.
- Existing unversioned `datasets` references are resolved and recorded when the worker prepares the run. For a fixed dataset version from the outset, supply `owner/slug/version` explicitly.
- The bootstrap handles both Kaggle archive files and automatically expanded archives. It verifies manifest and file hashes before importing your project.
- Project code runs from `/kaggle/working/project`. Put output files in `Path(os.environ["KGR_OUTPUT_DIR"])`, which points to `/kaggle/working/outputs`. Other files saved under `/kaggle/working` are also available as Kaggle outputs.
- Named local inputs are available through `KGR_INPUT_<UPPERCASE_ALIAS>` and the JSON mapping `KGR_INPUTS_JSON`. Expanded datasets are read directly from their Kaggle mounts. Archives that remain compressed are extracted into temporary storage outside captured outputs. Existing dataset attachments remain available through Kaggle's native `/kaggle/input` paths.
- Use Kaggle's installed Python packages by default. An explicitly supplied requirements file is installed using Kaggle's Python with `pip`; it requires internet access. Local virtual environments are never copied. There is no automatic offline dependency resolution or custom-container support.
- Environment values in the job definition are ordinary, persisted configuration. Do not put secrets in `env`. Local process environment variables are not forwarded to the workload.

## Scheduling and failure behavior

Defaults are private visibility, CPU, internet disabled, a 12-hour requested session ceiling, up to five managed CPU runs, and one managed GPU run. Configure pool limits and polling with `kgr init --owner YOUR_USERNAME --cpu-limit 5 --gpu-limit 1 --poll-seconds 30`, then restart the service. A pool limit of zero pauses launches for that resource.

These are local admission limits, not a promise of Kaggle availability. The worker polls account notebooks to account for other active runs and checks GPU quota before GPU submission. Remote visibility is best effort: another client, an older concurrently executing notebook version, or an account change can race with discovery. Kaggle's capacity and quota rejection is authoritative. No weekly CPU allowance is invented. CPU and GPU queues are independent, and a blocked GPU queue does not stop CPU work.

Status polling is every 30 seconds by default. Dataset preparation and uploads run in the dispatcher and can extend a cycle; downloads run separately so large result files do not hold up new submissions. Account discovery refreshes every five minutes and can briefly underuse capacity after an external job finishes.

Every submission attempt has its own notebook slug, recorded **before** the upload request. The worker never overwrites an existing experiment notebook. It also checks error fields in successful HTTP responses.

- **Capacity rejection:** wait with backoff, then use a fresh slug. This accommodates Kaggle's reported problem with reusing a slug after a rejected creation.
- **Network interruption during submission:** reconcile the recorded slug. Do not launch another attempt merely because the response was lost. Unresolved cases become `needs_attention` and continue reserving capacity.
- **Code failure:** preserve the failure and download whatever outputs Kaggle exposes. Do not rerun the computation automatically.
- **Upload/authentication/storage problem:** expose the reason. Repair it and use `retry` when the job is blocked.
- **Download failure:** preserve the computation outcome, retain completed files with verified receipts, and retry collection independently. Incomplete files never replace completed outputs.

Inspect an unresolved notebook on Kaggle before acting. If you have independently confirmed that **no execution exists**, `kgr resolve JOB_ID --not-submitted` records that explicit operator assertion; it does not launch anything. Then `kgr retry JOB_ID` creates a fresh job. Never use this to bypass an active or uncertain remote execution.

`cancel` is for locally pending work. Active runs must be stopped in Kaggle's web interface; the worker continues monitoring until Kaggle reports termination. The tool does not delete notebooks as a substitute for cancellation.

Runs are independent. You can download checkpoints and attach them as inputs to a subsequent run; there is no automatic continuation, dependency graph, recurring scheduling, or HTTP server in v1.

## State and results

Configuration is in `~/.config/kaggle-runner/config.json`. The queue, immutable snapshots, receipts, and results are in `~/.local/share/kaggle-runner`. `--config-dir` and `--state-dir` provide isolated CLI profiles; Python can use `Client(config=Config(...))`. A state directory is bound to one account.

A result directory contains:

```text
results/JOB_ID/
  provenance.json           # source hashes, settings, attachments, attempts and remote identity
  run.log                   # final log exposed by Kaggle
  downloads.json            # checksums and sizes of completed downloads
  outputs/                  # preserves paths relative to /kaggle/working
    outputs/result.json     # an example file written via KGR_OUTPUT_DIR
    project/...             # source files are captured too unless filtered out
```

Artifact filters use glob patterns against the relative remote paths, for example `outputs/*.json`. Logs are retrieved even when no artifact path matches. Output availability after a timeout or failed execution is controlled by Kaggle; the tool cannot recover files Kaggle did not retain.

Execution state and download state are separate. `succeeded` means Kaggle reported completion; `download_state=complete` means collection succeeded. `wait(..., downloads=False)` returns as soon as execution terminates. No cleanup runs automatically: retained datasets, notebooks, snapshots and outputs consume storage until you deliberately remove them. Back up the entire state directory with the worker stopped when moving this controller to another machine.

The systemd service reads standard Kaggle credentials from your account configuration. If you authenticate only through transient shell environment variables, persist credentials using Kaggle's supported authentication setup before starting the service. Credential values are never written into the generated unit.

## Development and verification

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests
.venv/bin/python -m compileall -q src
```

The test suite uses an injected fake backend for scheduling and recovery, plus real subprocess execution of generated launchers. It covers admission limits, external jobs, independent resource queues, persistence, submission ambiguity, explicit retries, private packaging, corrupt archives, streaming downloads, and CLI behavior. Unit tests never create remote resources.

Live smoke examples are intentionally tiny: `examples/hello.py`, `examples/hello.ipynb`, `examples/project.yaml`, and `examples/gpu_smoke.py`. The GPU smoke checks NVIDIA availability and a small HTTPS request; request both `--gpu --internet` to run it. Uploading/running these examples creates private Kaggle resources and consumes the corresponding quota.

The adapter pins Kaggle 2.2.4 and kagglesdk 0.1.37. Unique notebook slugs avoid the released client's per-version status/log/output limitations. Transport retries are disabled for SDK requests; the dispatcher classifies retries by operation. The official client is only imported when a remote operation is needed.
# kaggle-runner

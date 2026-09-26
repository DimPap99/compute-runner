# Kaggle Runner

Run Python scripts, notebooks, and project directories on Kaggle through a local Python API or CLI. A background worker manages uploads, execution status, resource admission, and output downloads. Jobs and submission attempts are stored in SQLite. Source files are snapshotted when a job is submitted.

The agent interface provides compact JSON responses, persistent batches, idempotent submissions, paginated status queries, change cursors, and bounded log retrieval. It uses the same queue and worker as the standard CLI.

## Requirements

- Python 3.12 or later
- Linux with `systemd --user` for the background service
- Kaggle credentials available to the user running the worker
- A Kaggle account with access to the requested compute resources and datasets

The worker can also run in a terminal without systemd. Local process locking uses `fcntl`.

## Installation

From the project directory:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e . --no-deps
source .venv/bin/activate

kgr init --owner YOUR_KAGGLE_USERNAME
kgr doctor
kgr service install
```

`kgr init` saves the account name and local scheduling limits. It does not configure Kaggle credentials. `kgr doctor` checks the local configuration, remote quota information, and active runs. Use `kgr doctor --offline` for local checks only.

The service must be able to authenticate without an interactive shell. Credentials supplied only through temporary shell variables are not copied into the generated service unit.

## Quick start

Preview the files selected for upload, then submit a workload:

```bash
kgr submit examples/hello.py --dry-run
kgr submit examples/hello.py
kgr submit examples/project.yaml
kgr submit examples/batch.yaml
```

Inspect a submitted job using the ID returned by `submit`:

```bash
kgr list
kgr status JOB_ID
kgr logs JOB_ID --follow
kgr wait JOB_ID
```

Submission writes to the local queue. A running worker is required to upload and launch jobs. CLI job IDs may be unambiguous prefixes. Batch IDs must be complete.

The standard CLI supports JSON output through the global `--json` option:

```bash
kgr --json list
kgr --json status JOB_ID
```

These commands return full records. Use `kgr agent` for bounded responses intended for automation.

## Workload configuration

A source can be a `.py` file, an `.ipynb` file, or a directory. Directory sources require either `entrypoint` or `module`. Single-file sources use their own filename as the entrypoint.

```yaml
name: training-experiment
source: ./project
module: experiments.train
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

exclude: ["outputs/", "checkpoints/"]
auto_download: true
output_patterns: ["outputs/*.json", "outputs/*.pt"]
```

Source and input paths are relative to the YAML file. Entrypoint and requirements paths are relative to the source directory. For example, use `entrypoint: scripts/train.py` instead of `module` to run a file within a project.

| Field | Default | Description |
| --- | --- | --- |
| `source` | Required | Python file, notebook, or project directory |
| `name` | `workload` | Display name |
| `entrypoint` | Unset | Relative script or notebook path within a directory source |
| `module` | Unset | Python module to execute, such as `experiments.train` |
| `args` | `[]` | Arguments passed to the workload |
| `env` | `{}` | Persisted, nonsecret environment values |
| `gpu` | `false` | Request a GPU |
| `accelerator` | Unset | NVIDIA accelerator ID. Setting this also enables GPU use |
| `internet` | `false` | Enable network access in the workload |
| `timeout_seconds` | `43200` | Requested session timeout, from 1 to 43200 seconds |
| `datasets` | `[]` | Existing datasets as `owner/slug` or `owner/slug/version` |
| `inputs` | `{}` | Named local files or directories to upload as private datasets |
| `requirements` | Unset | Included requirements file to install with pip. Requires internet access |
| `exclude` | `[]` | Additional source exclusion patterns |
| `auto_download` | `true` | Download outputs after execution terminates |
| `output_patterns` | Unset | Glob filters for remote output paths. Unset selects all files |

Unknown fields are rejected. Input names must be unique ignoring case. Environment names beginning with `KGR_` are reserved.

For a batch, put workload mappings under `jobs`:

```yaml
jobs:
  - name: analysis
    source: ./analyze.py
    timeout_seconds: 1800
  - name: training
    source: ./train.py
    gpu: true
    internet: true
    timeout_seconds: 7200
```

A batch accepts 1 to 1000 jobs. Every source and local input is snapshotted before the jobs are committed in one database transaction. A snapshot or database failure leaves none of that batch's jobs queued. Unused local bundles may remain. Jobs execute independently after the commit.

Both submit commands accept `--entrypoint`, `--module`, `--gpu/--cpu`, `--internet/--no-internet`, `--accelerator`, `--timeout`, and repeated `--arg` options. Overrides apply to every job in the YAML file. `--cpu` clears a configured accelerator and cannot be combined with `--accelerator`.

### Packaging and runtime

Source selection respects the root `.gitignore`, `.kgrignore`, and `exclude` patterns. Credential filenames, Git metadata, virtual environments, caches, and `node_modules` are excluded. Before saving a snapshot, the runner also rejects high-confidence private keys and service-token patterns without printing the detected value. This screening reduces accidental disclosure but cannot recognize every possible credential, so keep secrets outside source and input folders. Symlinks are rejected. Notebook outputs and execution counts are removed from the saved snapshot.

Single files are embedded in a generated private kernel. Project directories and local inputs become private datasets. Identical content reuses the same dataset. Managed datasets are immutable and retain source license metadata.

Unversioned dataset references are resolved when the worker prepares the job. Supply a version to select a specific dataset revision. The runtime accepts expanded Kaggle inputs or archives and verifies bundle contents before execution.

Project code runs from `/kaggle/working/project`. Write result files under `KGR_OUTPUT_DIR`, which points to `/kaggle/working/outputs`. Output names are relative to `/kaggle/working`, so a file written to `KGR_OUTPUT_DIR` is saved locally as `results/JOB_ID/outputs/outputs/NAME`. Downloads skip the runtime's copy of the snapshot files under `project/` and its `__pycache__` bytecode; new files the workload writes under `project/` are still collected. Named inputs are exposed through `KGR_INPUT_<UPPERCASE_ALIAS>` and the `KGR_INPUTS_JSON` mapping. Existing Kaggle dataset attachments remain under `/kaggle/input`.

Output downloads and full log caches have no configured size limit. Before writing, the runner checks free space on the filesystem containing the state directory. Known download sizes are checked up front; unknown or compressed bodies and log streams are checked as chunks arrive. A write that cannot fit with 16 MiB of operational headroom is stopped without replacing an existing file. The worker emits one warning when that filesystem falls below 10% free space and can warn again after space recovers and crosses the threshold later.

The workload uses Kaggle's Python environment. A configured requirements file is installed before execution. Local virtual environments and process environment variables are not forwarded. `env` is only for nonsecret configuration: secret-like variable names and recognizable credential values are rejected because these values must be stored with the job and embedded in the private Kaggle workload.

## Python API

Install the package into the calling environment with `pip install -e /path/to/kaggle-runner`.

```python
from kaggle_runner import Client, JobSpec

client = Client()
batch = client.submit_batch(
    [
        JobSpec(source="/path/analyze.py", timeout_seconds=1800),
        JobSpec(source="/path/train.py", gpu=True, internet=True),
    ],
    request_key="experiment-v1",
)

print(batch.id)
for job in batch.jobs:
    print(job.id, job.state)

finished = client.wait(batch.jobs[0].id, timeout=7200)
print(finished.state, finished.download_state, finished.result_dir)
```

`JobRecord` contains the workload specification, source manifest, attempts, execution state, download state, and result path. `BatchRecord` contains `id`, `created_at`, ordered `jobs`, and `replayed`.

| Method | Behavior |
| --- | --- |
| `preview(spec)` | Return the selected files, sizes, inputs, and resource settings without uploading |
| `submit(spec, request_key=None)` | Queue one job and return its record |
| `submit_many(specs, request_key=None)` | Queue a batch and return its jobs in input order |
| `submit_batch(specs, request_key=None)` | Queue a batch and return a `BatchRecord` |
| `batch(batch_id)` | Read a batch and its current job records |
| `get(job_id)`, `list(states=None)` | Read saved job records |
| `wait(job_id, timeout=None, downloads=True)` | Wait for execution and downloads. Return early for blocked or unresolved work, or when output collection failed |
| `logs(job_id, follow=False)` | Yield persisted logs of a finished run, a bounded snapshot of an unfinished one, or follow the remote log stream |
| `download(job_id)` | Collect outputs from a submitted job whose execution has terminated |
| `retry(job_id, request_key=None)` | Create a job from the original saved files and settings |
| `retry_batch(job_id, request_key=None)` | Create a retry and return its single-job batch |
| `cancel(job_id)` | Cancel pending work locally, or ask Kaggle to stop a running job |
| `resolve_not_submitted(job_id)` | Record an operator's confirmation that an unresolved attempt created no remote execution |
| `quota()` | Query accelerator quota information |
| `worker_health()` | Read worker lock ownership and heartbeat data |
| `worker()` | Construct a worker for this configuration |
| `agent()` | Return the compact automation interface |

Use complete job IDs with `Client`. Optional parameters shown after the first argument are keyword arguments. Creating a client or reading local state does not authenticate to Kaggle. Submission is local, while logs, downloads, and quota queries access Kaggle when needed.

`wait` raises `TimeoutError` when its local wait deadline expires. This does not cancel the job. With `downloads=False`, it returns after execution terminates. A blocked or uncertain job is returned for inspection, as is a finished job whose `download_state` is `error`; the worker keeps retrying that download. Waiting raises an error once no worker has run for 30 seconds.

## Agent interface

`kgr agent` returns one compact JSON object per operation. It does not require `--json`. Responses omit source manifests, full specifications, and environment values. The background worker performs polling and downloads without model calls.

```bash
kgr agent submit examples/batch.yaml --request-key experiment-v1 --dry-run
kgr agent submit examples/batch.yaml --request-key experiment-v1
kgr agent status --batch BATCH_ID
kgr agent status JOB_ID_1 JOB_ID_2
kgr agent status --state running --state failed
kgr agent changes --batch BATCH_ID --after 0
kgr agent logs JOB_ID --tail 50 --max-bytes 8192
kgr agent wait --batch BATCH_ID --timeout 300
kgr agent outputs JOB_ID
kgr agent health
```

`--dry-run` returns aggregate file sizes, file counts, and resource counts without queueing jobs. The response includes `total`, `files`, `bytes`, `gpu_jobs`, `internet_jobs`, and `private`. Use the standard `kgr submit --dry-run` command for individual filenames.

### Submission keys

Agent submissions and retries require `--request-key`. Standard `kgr submit` and the Python submission methods also accept a key.

A key identifies one intended operation within a state directory. Repeating the same request returns the original batch with `replayed: true`. This holds across concurrent callers, process restarts, and completed runs. Reusing the key with different settings raises an error. Submit and retry operations share the same key namespace.

Keys must match `[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}`. The comparison uses normalized workload settings, including paths. It does not compare the current contents of source files. A replay uses the original snapshots even if those files have changed or disappeared. Use a new key to run changed code or intentionally repeat an experiment. Submission receipts do not expire automatically.

Retries use the original saved files:

```bash
kgr agent retry JOB_ID --request-key experiment-retry-v1
```

### Status and pagination

Status, submit, retry, and cancel responses contain `schema_version`, `batch_id`, `total`, `counts`, `jobs`, `next_offset`, and `worker`. Submit and retry also return `replayed`.

The default page size is 20 jobs, with a maximum of 100. `counts` and `total` cover the full selection. Follow `next_offset` until it is null:

```bash
kgr agent status --batch BATCH_ID --limit 20 --offset 0
kgr agent status --batch BATCH_ID --limit 20 --offset 20
```

Each job summary contains:

| Field | Meaning |
| --- | --- |
| `id`, `name`, `state` | Job identity and current execution state |
| `batch_id`, `batch_index` | Batch membership and zero-based input position |
| `resource`, `internet` | CPU or GPU configuration and network setting |
| `downloads`, `outputs_ready` | Download state and completion flag |
| `url` | Kaggle notebook URL, when an attempt exists |
| `reason`, `error`, `download_error` | Available diagnostic messages, limited to 400 characters each |
| `output_dir` | Local results directory after downloads complete |
| `parent_id` | Original job ID for a retry |

Optional fields are omitted when unavailable. Jobs created before batch support have a null `batch_id` and no `batch_index`. Batch status preserves input order. Status queries accept at most 100 explicit job IDs. Job names and resource labels are limited to 100 characters.

### Change cursors

`changes` returns the latest state of each job with an event after `--after`. Start at 0, then pass the returned `cursor` to the next call:

```bash
kgr agent changes --batch BATCH_ID --after 0
kgr agent changes --batch BATCH_ID --after RETURNED_CURSOR
```

Responses contain `schema_version`, `batch_id`, `cursor`, `has_more`, `jobs`, and `worker`. Drain additional pages while `has_more` is true. Page size defaults to 20 and is limited to 100.

Events for the same job are coalesced. A job that changes between pages can appear again. The returned records describe current state rather than every historical transition. Each query reads its records and cursor from one SQLite snapshot, so later changes remain visible to a subsequent query.

Keep a separate cursor for each state directory and batch filter. Reset to 0 when changing filters or recovering a lost cursor. Events from other batches may advance a filtered cursor without returning jobs. Unchanged status polls and worker heartbeats produce no job events. Download state and error changes do.

### Log retrieval

```bash
kgr agent logs JOB_ID --tail 50 --max-bytes 8192
kgr agent logs JOB_ID --refresh
```

Kaggle stores a session's log only after it ends. While a submitted job is unfinished, every call reads a live snapshot from Kaggle's log stream, which replays the log from the start. The read stops after 5 idle seconds or 20 seconds in total, and the response has `live: true`.

For a finished job, the first call fetches the stored log and saves a private local copy. Later calls read that copy unless `--refresh` is supplied or the job has finished since the copy was saved. A failed refresh preserves the existing cache.

The response contains `text`, `bytes`, `total_bytes`, `truncated`, `path`, `fetched`, `cached_at`, and `live`, together with `schema_version` and the job `id`. `path` identifies the full cached log. `cached_at` is its modification time as a Unix timestamp.

The default response contains at most 50 lines and 8192 UTF-8 bytes. The maximum permitted limits are 500 lines and 65536 bytes. Lines are split on `\n` only, so carriage-return progress bars count as one line. Byte truncation can leave a partial first line. The size limit applies to the returned text, not the download from Kaggle. Agent log retrieval does not follow a stream.

### Waiting

```bash
kgr agent wait --batch BATCH_ID --timeout 300
kgr agent wait JOB_ID_1 JOB_ID_2 --timeout 600 --no-downloads
```

`wait` blocks until every selected job settles or the timeout passes, then returns the same fields as `status` plus `settled`, `timed_out`, and `waited_seconds`. A job is settled when it is terminal and its downloads are complete, disabled, or failed, or when it is `blocked` or `needs_attention`. With `--no-downloads`, terminal state is enough. A timeout is not an error, so check `timed_out` and call again if needed. The timeout can be 0 to 86400 seconds. Keep it below the command timeout of the calling tool. `wait` reads local state only. It fails if the worker stays stopped for 30 seconds.

### Outputs

```bash
kgr agent outputs JOB_ID
kgr agent outputs JOB_ID --limit 100 --offset 100
```

`outputs` lists downloaded files without reading them. The response contains `root`, `total`, `files` (each with `path` relative to `root` and `bytes`), `next_offset`, `state`, `downloads`, `outputs_ready`, and, when available, `download_error` and `log_path`. Files written to `KGR_OUTPUT_DIR` appear as `outputs/NAME`. Read them from `root` with ordinary file tools. The listing can be partial until `outputs_ready` is true.

### Python access

```python
from kaggle_runner import Client

agent = Client().agent()
page = agent.changes(batch_id="BATCH_ID", after=0)
for job in page["jobs"]:
    print(job["id"], job["state"])

while page["has_more"]:
    page = agent.changes(batch_id="BATCH_ID", after=page["cursor"])
    for job in page["jobs"]:
        print(job["id"], job["state"])

cursor = page["cursor"]
```

`AgentClient()` is also available from `kaggle_runner` and uses the default configuration.

| Method | Result |
| --- | --- |
| `submit(specs, request_key=...)` | Batch status and replay flag |
| `preview(specs)` | Aggregate upload inventory |
| `status(job_ids=None, batch_id=None, states=None, limit=20, offset=0)` | Paginated job summaries and counts |
| `changes(after=0, batch_id=None, limit=20)` | Changed jobs and the next event cursor |
| `logs(job_id, tail=50, max_bytes=8192, refresh=False)` | Bounded text and cache metadata |
| `wait(job_ids=None, batch_id=None, timeout=300, downloads=True, limit=20)` | Status once the selection settles or the timeout passes |
| `outputs(job_id, limit=100, offset=0)` | Downloaded file listing |
| `retry(job_id, request_key=...)` | Retry batch status and replay flag |
| `cancel(job_id)` | Updated job status. Repeated cancellation of a cancelled job is accepted |
| `health()` | Worker lock and heartbeat summary |

Agent methods return JSON-compatible dictionaries and raise Python exceptions on errors. CLI responses use `schema_version: 1`. Handled operation errors return a JSON `error` and exit status 1. Argument parsing and startup failures use the standard CLI error output.

### Codex skill

The [bundled skill](skills/kaggle-runner/SKILL.md) documents the agent commands, request keys, cursor handling, and recovery workflow. Install the `skills/kaggle-runner` directory into your Codex skills directory, normally `~/.codex/skills`.

Invoke it as `$kaggle-runner` in a session where the skill is available. The skill expects `kgr` on PATH or at `~/kaggle-runner/.venv/bin/kgr`. Other agents with local shell access can use the same CLI. No MCP server or model API key is required.

## Worker configuration

`kgr service install` enables and starts `kaggle-runner.service` as a systemd user service. It starts on login and restarts after a process failure. Installation does not enable user lingering, so processing may stop after logout. The host must remain running and connected to submit jobs and collect results. Submitted Kaggle jobs continue remotely and are reconciled when the worker resumes.

```bash
kgr service status
kgr service restart
kgr service stop
kgr service start
kgr worker status
journalctl --user -u kaggle-runner.service -f
```

Use `kgr worker run` to run in the foreground, or `kgr worker run --once` for one dispatch cycle. Only one worker may hold the lock for a state directory.

| Setting | Default |
| --- | --- |
| Managed CPU concurrency | 5 |
| Managed GPU concurrency | 1 |
| Status polling interval | 30 seconds |
| Account discovery interval | 300 seconds |
| Workload visibility | Private |

Set resource limits and polling through `kgr init`, then restart the worker:

```bash
kgr init --owner YOUR_KAGGLE_USERNAME --cpu-limit 5 --gpu-limit 1 --poll-seconds 30
kgr service restart
```

A resource limit of zero pauses launches for that pool. CPU and GPU queues are independent. The worker accounts for discovered external runs and checks GPU quota before admission. Discovery checks only notebooks run within the last 24 hours, which keeps it within Kaggle's rate limits. Kaggle's notebook listing reports every notebook as CPU, so discovery reads each active run's own settings once to count GPU runs correctly. Discovery can be stale, so Kaggle's capacity and quota responses remain authoritative. Local limits do not guarantee available resources or an unlimited CPU allowance.

Dataset preparation and uploads run in the dispatcher and can extend a polling cycle. Output downloads run separately.

## Failure handling

| Condition | Behavior |
| --- | --- |
| Capacity rejection | Back off and retry with a fresh notebook slug |
| Uncertain submission | Query the recorded notebook reference before attempting another submission |
| Unresolved remote execution | Set `needs_attention` and continue reserving capacity |
| Workload failure | Set `failed` and collect available outputs without rerunning the computation |
| Nonretryable upload or authentication error | Set `blocked` and retain the diagnostic message |
| Download failure | Preserve execution status and retry output collection independently |

Each attempt records its notebook slug before the remote request. The worker creates a new slug for each attempt and does not overwrite an existing experiment notebook.

`retry` accepts a terminal or blocked job when no execution remains outstanding. It uses saved snapshots. Submit a new workload to change the code or settings.

For an unresolved submission, inspect its Kaggle URL first. If no remote execution exists, record that confirmation before retrying:

```bash
kgr resolve JOB_ID --not-submitted
kgr agent retry JOB_ID --request-key resolved-retry-v1
```

`resolve` records an operator assertion and does not launch a job. It must not be used to bypass an active or uncertain execution.

`kgr cancel JOB_ID` and `kgr agent cancel JOB_ID` cancel pending work locally. For a running job, they ask Kaggle to stop the session. The job shows the reason `Cancellation requested on Kaggle` until the worker sees the run end, usually within a minute. It then becomes `cancelled`, and its partial outputs and log are collected. Kaggle's public API does not return session IDs, so the runtime prints its own session ID at startup and cancellation reads it from the live log. A job that has not started on Kaggle yet, or was submitted by an older version of the runner, cannot be cancelled this way; stop it on its Kaggle page. A submission whose outcome is uncertain is never cancelled automatically.

## State and outputs

Default locations:

| Path | Contents |
| --- | --- |
| `~/.config/kaggle-runner/config.json` | Account and worker configuration |
| `~/.local/share/kaggle-runner/queue.sqlite3` | Jobs, batches, request receipts, and events |
| `~/.local/share/kaggle-runner/bundles/` | Immutable source and input snapshots |
| `~/.local/share/kaggle-runner/logs/JOB_ID.log` | Agent log cache |
| `~/.local/share/kaggle-runner/results/JOB_ID/` | Downloaded files and provenance |

Global `--config-dir` and `--state-dir` options select alternate CLI locations. Python callers can use `Client(config=Config(...))` or a `state_dir` override. Each state directory is bound to one Kaggle account.

A completed download has this layout:

```text
results/JOB_ID/
  provenance.json
  downloads.json
  run.log
  outputs/
    outputs/result.json
    project/...
```

`provenance.json` records source hashes, settings, dataset references, attempts, and remote identity. `downloads.json` records verified file sizes and checksums. `run.log` is present when Kaggle exposes a log. Files under the local `outputs` directory preserve paths relative to `/kaggle/working`. A file written to `KGR_OUTPUT_DIR/result.json` therefore appears at `outputs/outputs/result.json`.

Output filters match remote relative paths such as `outputs/*.json`. Logs are collected independently of those filters. Files unavailable from Kaggle after a failure or timeout cannot be recovered by the controller.

Execution and download states are separate. `succeeded` means Kaggle reported completion. `download_state=complete`, exposed as `outputs_ready: true` by the agent interface, means output collection completed. Downloads can be pending or failed after a successful computation.

No automatic cleanup removes notebooks, datasets, snapshots, logs, or results. Stop the worker before backing up or moving the complete state directory.

### Database upgrades

Version 0.2 upgrades schema version 1 to version 2 by adding batch and request records. Existing job IDs, events, and snapshots are preserved. Older application versions cannot open the upgraded database.

Back up the database before upgrading. To restore an older application version, stop the worker and restore a compatible backup. Preserve any jobs and artifacts created after that backup separately.

## Limitations

- Scheduling starts jobs when capacity becomes available. Start times, recurring schedules, and dependency graphs are not implemented.
- Checkpoint continuation requires downloading a checkpoint and attaching it to a new job.
- Remote cancellation needs the run to have started and to have been submitted by this version of the runner.
- HTTP and MCP servers are not included.
- Job completion does not automatically resume an LLM conversation.
- Custom containers and automatic offline dependency installation are not supported.

## Development

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests
.venv/bin/python -m compileall -q src
```

Tests use a fake Kaggle backend and local execution of generated launchers. They cover scheduling, restart recovery, submission ambiguity, batch transactions, idempotency, cursor pagination, packaging, downloads, and CLI behavior. Automated tests do not create remote resources.

The adapter pins `kaggle==2.2.4` and `kagglesdk==0.1.37`. SDK transport retries are disabled. The worker determines whether a remote operation can be retried. The Kaggle client is imported when a remote operation is required.

Example workloads are under [examples](examples). The GPU smoke test requires `--gpu --internet`. Running examples on Kaggle creates private resources and uses the corresponding compute allocation. See [VALIDATION.md](VALIDATION.md) for recorded test results and live checks.

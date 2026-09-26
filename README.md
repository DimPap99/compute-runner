# Compute Runner

Submit, monitor, and resume Python workloads through a local Python API or CLI.

Jobs run on provider accounts you connect. Kaggle is the only provider adapter today; you can connect several Kaggle accounts, and a job that cannot start on one can move to another.

A background worker manages uploads, execution status, resource admission, and output downloads. Jobs and submission attempts are stored in SQLite. Source files are snapshotted when a job is submitted. Resuming training requires checkpoint support in the workload.

The agent interface provides compact JSON responses, persistent batches, idempotent submissions, paginated status queries, change cursors, and bounded log retrieval. It uses the same queue and worker as the standard CLI.

## Requirements

- Python 3.12 or later
- Linux with `systemd --user` for the background service
- Credentials for each connected account, readable by the user running the worker
- Accounts with access to the requested compute resources and datasets

The worker can also run in a terminal without systemd. Local process locking uses `fcntl`.

## Installation

From the project directory:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e . --no-deps
source .venv/bin/activate

compute-runner account add kaggle YOUR_KAGGLE_USERNAME
compute-runner doctor
compute-runner service install
```

`compute-runner account add` connects an account and its local scheduling limits; see [Accounts and failover](#accounts-and-failover). `compute-runner init` saves worker-wide settings: the failover policy, polling interval, and [strict mode](#strict-mode). Rerunning either command changes only the options you pass. `compute-runner doctor` checks the local configuration and, for each account, remote quota information and active runs. Use `compute-runner doctor --offline` for local checks only.

`compute-runner` is the primary command; `kgr` remains an alias for existing scripts. Python callers import `compute_runner`.

The service must be able to authenticate without an interactive shell. Credentials supplied only through temporary shell variables are not copied into the generated service unit.

## Quick start

Preview the files selected for upload, then submit a workload:

```bash
compute-runner submit examples/hello.py --dry-run
compute-runner submit examples/hello.py
compute-runner submit examples/project.yaml
compute-runner submit examples/batch.yaml
```

Each job's results go to a run folder chosen at submission, such as `examples/results/hello/001_2026-09-26_14-30-12/`; see [Results](#results). Inspect a submitted job using the ID returned by `submit`:

```bash
compute-runner list
compute-runner status JOB_ID
compute-runner logs JOB_ID --follow
compute-runner wait JOB_ID
```

Submission writes to the local queue. A running worker is required to upload and launch jobs. CLI job IDs may be unambiguous prefixes. Batch IDs must be complete.

The standard CLI supports JSON output through the global `--json` option:

```bash
compute-runner --json list
compute-runner --json status JOB_ID
```

These commands return full records. Use `compute-runner agent` for bounded responses intended for automation.

## Accounts and failover

An account is one set of credentials on one provider. Its ID is `PROVIDER:USER`, such as `kaggle:alice`. Accounts are kept in order of preference, and the first is the default for new jobs:

```bash
compute-runner account add kaggle alice                   # standard Kaggle credentials
compute-runner account add kaggle bob --credentials ~/.config/kaggle-bob/kaggle.json --gpu-limit 1
compute-runner account add kaggle bob --default          # prefer bob from now on
compute-runner account list
compute-runner account remove kaggle:bob
```

Without `--credentials`, the account uses Kaggle's usual discovery (`KAGGLE_*` variables, `~/.kaggle`). A credentials file is used for that account alone and can hold a `kaggle.json` username and key or a Kaggle access token. Only its path is saved. Every account checks that its credentials authenticate as its user, so a global token cannot act for another account. An account cannot be removed while unfinished jobs, or pending downloads, use it. Failed downloads of its runs are not retried while it is removed and resume if it is added again.

The worker reads accounts and settings when it starts. After `account add`, `account remove`, or `init`, apply the change with `compute-runner service restart`; these commands print that reminder while a worker is running. Until then, a job submitted to a newly added account waits with a reason saying the running worker does not know its account.

Each job records its account, and each attempt records the account, remote reference, and URL it ran on. Choose an account with `--account` on `submit` and `retry`, or move a job that has not been submitted yet:

```bash
compute-runner submit train.py --gpu --account kaggle:bob
compute-runner move JOB_ID --account kaggle:alice
```

When a waiting job cannot start on its account, because its slots are busy, its GPU quota is exhausted, or the account cannot be checked, the failover policy decides what happens:

| Policy | Behavior |
| --- | --- |
| `off` | The job waits on its account |
| `ask` (default) | The job waits and shows `suggested_account`: the first other account, in preference order, that can start it now. An agent asks the user before moving it |
| `auto` | The worker moves the job to that account. Its `account` changes, and its event history records `Moved from ACCOUNT: reason` |

```bash
compute-runner init --failover auto
compute-runner service restart
```

Failover counts jobs already preparing on the other account, so a burst moves only as many jobs as that account can start. A job whose inputs are already uploading stays on its account, and jobs queued behind it wait for it rather than failing over. An account that rejected one of a job's launches is not chosen for that job again, so a job cannot bounce between two full accounts. Moving a job uploads its inputs again, because private datasets belong to one account. Failover checks that the other account can read the job's provider datasets; see [Datasets across accounts](#datasets-across-accounts). Runs that have started never move; continue a stopped resumable run on another account with `compute-runner continue JOB_ID --account ID`, as described in [Optional resumable training](#optional-resumable-training).

### Datasets across accounts

A job can attach existing provider datasets, such as another account's private dataset, and the account that runs it may not be able to read them. Before attaching a dataset, the worker asks the job's account whether it can read it, then:

| Situation | Behavior |
| --- | --- |
| The job's account can read it | Attach it directly, pinned to its current version |
| Another connected account can read it, and copies are allowed | Download it once through that account, then upload the copy to the job's account like a local input |
| Another connected account can read it, and copies are not allowed | Block the job; the error names the account that can read it |
| No connected account can read it, or that version does not exist | Block the job before it runs; the error asks to check the reference or the accounts' access |

Copies are off by default because they place the data in another account and take its storage. Allow them for one job with `compute-runner agent move JOB_ID --account ACCOUNT --transfer` (the job's own account also works), or for every job with `compute-runner init --transfer`. Only aliased inputs (`inputs: {data: "kaggle:owner/slug"}`) can be copied, because the workload finds a copy through `KGR_INPUT_DATA`; the unaliased `datasets` list cannot. A copied dataset version is cached in the state directory and reused. Check the dataset's license before copying it.

Failover prefers an account that can read every dataset. Under `ask`, a suggested account that would need a copy is shown with `suggested_transfer: true`; under `auto`, the worker moves a job there only when copies are allowed.

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

params: {lr: 0.01, batch-size: 128}
inputs:
  extra_data: ./data
  prepared: "kaggle:yourname/prepared-data/3"

exclude: ["checkpoints/"]
auto_download: true
output_patterns: ["outputs/*.json", "outputs/*.pt"]
```

Source and input paths are relative to the YAML file. Entrypoint and requirements paths are relative to the source directory. For example, use `entrypoint: scripts/train.py` instead of `module` to run a file within a project.

| Field | Default | Description |
| --- | --- | --- |
| `source` | Required | Python file, notebook, or project directory |
| `name` | `workload`; a script's file name | Experiment name, and the name of its results folder |
| `entrypoint` | Unset | Relative script or notebook path within a directory source |
| `module` | Unset | Python module to execute, such as `experiments.train` |
| `args` | `[]` | Arguments passed to the workload |
| `params` | `{}` | Named settings, passed after `args` as `--NAME VALUE` (`true` as `--NAME`, `false` omitted), exposed as `KGR_PARAMS_JSON`, and recorded with the results |
| `env` | `{}` | Persisted, nonsecret environment values |
| `gpu` | `false` | Request a GPU |
| `accelerator` | Unset | Provider accelerator ID; Kaggle accepts NVIDIA IDs. Setting this also enables GPU use |
| `internet` | `false` | Enable network access in the workload |
| `timeout_seconds` | `43200` | Requested session timeout; Kaggle accepts 1 to 43200 seconds |
| `datasets` | `[]` | Existing provider datasets without an alias; on Kaggle `owner/slug` or `owner/slug/version` |
| `inputs` | `{}` | Named inputs: a local file or directory (uploaded as a private dataset), a provider dataset such as `kaggle:owner/slug/3`, or a finished job's outputs as `job:JOB_ID/PATH` |
| `requirements` | Unset | Included requirements file to install with pip. Requires internet access |
| `exclude` | `[]` | Additional source exclusion patterns |
| `auto_download` | `true` | Download outputs after execution terminates |
| `output_patterns` | Unset | Glob filters for remote output paths. Unset selects all files |
| `results_dir` | Unset | Parent of the experiment folders for this workload; see [Results](#results) |

Unknown fields are rejected. Input names must be unique ignoring case. Environment names beginning with `KGR_` are reserved. Provider-specific values are checked against the job's account when it is submitted or moved.

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

Both submit commands accept `--entrypoint`, `--module`, `--gpu/--cpu`, `--internet/--no-internet`, `--accelerator`, `--timeout`, `--account`, and repeated `--arg` and `--param NAME=VALUE` options. `--param` values are read as YAML scalars, so `lr=0.01` is a number and `amp=true` a flag, and they are merged into each job's `params`. Overrides apply to every job in the YAML file. `--cpu` clears a configured accelerator and cannot be combined with `--accelerator`.

### Packaging and runtime

Source selection respects the root `.gitignore`, `.kgrignore`, and `exclude` patterns. Input folders honor only their own `.kgrignore`, because data folders often `.gitignore` the very files they carry. Credential filenames, Git metadata, virtual environments, caches, and `node_modules` are excluded. Before saving a snapshot, the runner also rejects high-confidence private keys and service-token patterns without printing the detected value. This screening reduces accidental disclosure but cannot recognize every possible credential, so keep secrets outside source and input folders. Symlinks are rejected. Notebook outputs and execution counts are removed from the saved snapshot. A notebook entrypoint must be a valid Python notebook; other notebooks in the project, such as R notebooks, are cleaned when possible and otherwise copied unchanged.

Single files are embedded in a generated private kernel. Project directories and local inputs become private datasets. Identical content reuses the same dataset. Managed datasets are immutable and retain source license metadata.

Unversioned dataset references are resolved when the worker prepares the job. Supply a version to select a specific dataset revision. The runtime accepts expanded Kaggle inputs or archives and verifies bundle contents before execution.

Project code runs from `/kaggle/working/project`. Write result files under `KGR_OUTPUT_DIR`, which points to `/kaggle/working/outputs`; a file written there is saved locally as `RUN_FOLDER/outputs/NAME`. Downloads skip the runtime's copy of the snapshot files under `project/` and its `__pycache__` bytecode; new files the workload writes under `project/` are still collected, into `RUN_FOLDER/working/project/`. Every named input, whether uploaded, attached from a provider, copied, or taken from another job, is exposed through `KGR_INPUT_<UPPERCASE_ALIAS>` and the `KGR_INPUTS_JSON` mapping, so workloads never depend on provider paths. Datasets in the unaliased `datasets` list remain under `/kaggle/input`. Parameters are in `KGR_PARAMS_JSON`.

Output downloads and full log caches have no configured size limit. Before writing, the runner checks free space on the filesystem containing the state directory. Known download sizes are checked up front; unknown or compressed bodies and log streams are checked as chunks arrive. A write that cannot fit with 16 MiB of operational headroom is stopped without replacing an existing file. The worker emits one warning when that filesystem falls below 10% free space and can warn again after space recovers and crosses the threshold later.

Downloads use the environment's proxy and certificate settings by default. [Strict mode](#strict-mode) ignores them and fetches only public HTTPS addresses.

The workload uses Kaggle's Python environment. A configured requirements file is installed before execution. Local virtual environments and process environment variables are not forwarded. `env` is only for nonsecret configuration: secret-like variable names and recognizable credential values are rejected because these values must be stored with the job and embedded in the private Kaggle workload.

### Optional resumable training

Resumability is a workload-code decision, not a runner default. When an agent is preparing a stateful workload and the user's choice is unclear, the bundled skill tells it to ask whether the job should be resumable. If the answer is yes and no cadence was given, it then asks whether to checkpoint by elapsed minutes or completed epochs and for the interval. An explicit non-resumable choice is respected.

Resumable scripts should expose `--resume auto|required|never|PATH`, `--checkpoint-mode minutes|epochs`, and `--checkpoint-every NUMBER`, and write checkpoints below `KGR_OUTPUT_DIR/checkpoints`. Continue a stopped run once its outputs are downloaded:

```bash
compute-runner agent continue JOB_ID --request-key continue-1                          # same account
compute-runner agent continue JOB_ID --request-key continue-1 --account kaggle:bob     # another account
```

`continue` verifies that `checkpoints/latest.json` names a checkpoint with a matching SHA-256, then queues the job's saved code and inputs as the next run of the same experiment. It attaches only `latest.json` and that checkpoint as the input `resume` (`KGR_INPUT_RESUME`) and passes `--resume required`, replacing another `--resume` value or the `resume` parameter, so a missing or invalid checkpoint cannot silently restart training. To continue with changed code, submit a new workload with `inputs: {resume: "job:JOB_ID/checkpoints"}` and `--resume required`.

The skill includes a framework-neutral helper at `skills/compute-runner/assets/checkpointing.py`. It provides cadence checks, atomic numbered files, a checksummed `latest.json`, compatibility validation, and resume discovery. Training code must still serialize and restore its framework-specific model, optimizer, scheduler, scaler, progress, RNG, and data-loader state. See `skills/compute-runner/references/resumability.md` for the complete agent and migration contract.

## Results

Each job's run folder is fixed when it is submitted, before anything runs, and returned by every status response as `run_dir`:

```text
RESULTS/NAME/
  runs.md
  001_2026-09-26_14-30-12/
    job.json
    run.log
    outputs/        files the workload wrote to KGR_OUTPUT_DIR
    working/        other files it left in /kaggle/working
  002_2026-09-26_15-02-47/
```

- `RESULTS` is the workload's `results_dir`, else the folder set with `compute-runner init --results-dir PATH` (`--results-dir ""` restores the default), else `results/` beside the workload's code: inside a source folder, or next to a single script. When it lies inside the source folder, it is left out of source snapshots.
- `NAME` is the job's name with characters other than letters, digits, `.`, `_` and `-` replaced by `-`.
- The run number counts up within the folder. The queue hands it out when the batch commits, so concurrent submitters never share one, and it continues after folders already on disk. The timestamp is the submission time in local time.
- `job.json` records the job ID, parameters, command, account, run URL, state, attempts, inputs and download state. `runs.md` lists every run of the experiment with its state, parameters, download state, parent run (for retries and continuations) and job ID.

Only the queue writes there. The worker downloads outputs into the folder (as does `compute-runner download`) and rewrites `job.json` and `runs.md` from the queue whenever a job changes, following the queue's change cursor, so a restart resumes where it stopped and edits to those two files are overwritten. Retries and continuations are the next runs of the same experiment. Jobs saved before run folders keep `STATE_DIR/results/JOB_ID/`, with `outputs/outputs/NAME` and `provenance.json`.

## Python API

Install the package into the calling environment with `pip install -e /path/to/compute-runner`.

```python
from compute_runner import Client, JobSpec

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

`JobRecord` contains the workload specification, source manifest, attempts, execution state, download state, run number, and run folder (`result_dir`). `BatchRecord` contains `id`, `created_at`, ordered `jobs`, and `replayed`.

| Method | Behavior |
| --- | --- |
| `preview(spec, account=None)` | Check the spec against the account and return the selected files, sizes, inputs, resource settings, and `experiment_dir` without uploading |
| `submit(spec, request_key=None, account=None)` | Queue one job and return its record |
| `submit_many(specs, request_key=None, account=None)` | Queue a batch and return its jobs in input order |
| `submit_batch(specs, request_key=None, account=None)` | Queue a batch and return a `BatchRecord` |
| `batch(batch_id)` | Read a batch and its current job records |
| `get(job_id)`, `list(states=None)` | Read saved job records |
| `wait(job_id, timeout=None, downloads=True)` | Wait for execution and downloads. Return early for blocked or unresolved work, or when output collection failed |
| `logs(job_id, follow=False)` | Yield persisted logs of a finished run, a bounded snapshot of an unfinished one, or follow the remote log stream |
| `download(job_id)` | Collect outputs from a submitted job whose execution has terminated |
| `retry(job_id, request_key=None, account=None)` | Create a job from the original saved files and settings, on the original account unless one is given |
| `retry_batch(job_id, request_key=None, account=None)` | Create a retry and return its single-job batch |
| `continue_run(job_id, request_key=None, account=None)` | Continue a stopped resumable run from its verified checkpoint as the next run of its experiment |
| `continue_batch(job_id, request_key=None, account=None)` | Create a continuation and return its single-job batch |
| `move(job_id, account, transfer=False)` | Place a job that has not been submitted on another account; `transfer` allows copying datasets it cannot read |
| `cancel(job_id)` | Cancel pending work locally, or ask the provider to stop a running job |
| `resolve_not_submitted(job_id)` | Record an operator's confirmation that an unresolved attempt created no remote execution |
| `quota(account=None)` | Query one account's accelerator quota, or every account's |
| `provider(account=None)` | Return the provider adapter for an account |
| `worker_health()` | Read worker lock ownership and heartbeat data |
| `worker()` | Construct a worker for this configuration |
| `agent()` | Return the compact automation interface |

Use complete job IDs with `Client`. Optional parameters shown after the first argument are keyword arguments. Creating a client or reading local state does not authenticate to any provider. Submission is local, while logs, downloads, and quota queries contact the job's provider when needed.

`wait` raises `TimeoutError` when its local wait deadline expires. This does not cancel the job. With `downloads=False`, it returns after execution terminates. A blocked or uncertain job is returned for inspection, as is a finished job whose `download_state` is `error`; the worker keeps retrying that download. Waiting raises an error once no worker has run for 30 seconds.

## Agent interface

`compute-runner agent` returns one compact JSON object per operation. It does not require `--json`. Responses omit source manifests, full specifications, and environment values. The background worker performs polling and downloads without model calls.

```bash
compute-runner agent submit examples/batch.yaml --request-key experiment-v1 --dry-run
compute-runner agent submit examples/batch.yaml --request-key experiment-v1
compute-runner agent status --batch BATCH_ID
compute-runner agent status JOB_ID_1 JOB_ID_2
compute-runner agent status --state running --state failed
compute-runner agent changes --batch BATCH_ID --after 0
compute-runner agent logs JOB_ID --tail 50 --max-bytes 8192
compute-runner agent wait --batch BATCH_ID --timeout 300
compute-runner agent outputs JOB_ID
compute-runner agent continue JOB_ID --request-key continue-1
compute-runner agent accounts
compute-runner agent move JOB_ID --account kaggle:bob
compute-runner agent health
```

`--dry-run` returns aggregate file sizes, file counts, and resource counts without queueing jobs. The response includes `total`, `experiment_dirs`, `files`, `bytes`, `gpu_jobs`, `internet_jobs`, and `private`. Use the standard `compute-runner submit --dry-run` command for individual filenames.

### Submission keys

Agent submissions and retries require `--request-key`. Standard `compute-runner submit` and the Python submission methods also accept a key.

A key identifies one intended operation within a state directory. Repeating the same request returns the original batch with `replayed: true`. This holds across concurrent callers, process restarts, and completed runs. Reusing the key with different settings raises an error. Submit and retry operations share the same key namespace.

Keys must match `[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}`. The comparison uses normalized workload settings, including paths. It does not compare the current contents of source files. A replay uses the original snapshots even if those files have changed or disappeared. Use a new key to run changed code or intentionally repeat an experiment. Submission receipts do not expire automatically.

Retries use the original saved files:

```bash
compute-runner agent retry JOB_ID --request-key experiment-retry-v1
```

`agent submit`, `agent retry` and `agent continue` accept `--account`. An explicit account is part of the request, so a replay must repeat it; requests without one keep their original fingerprint.

### Accounts

`agent accounts` reads local state only. It returns `failover`, `default`, and for each account in preference order: `id`, `provider`, `cpu` and `gpu` (`used` and `limit`), `gpu_quota_seconds` and `checked_age_seconds` from the worker's last check, and `error` when that check failed. `used` counts this queue's runs and other runs the worker discovered.

`agent move JOB_ID... --account ID` (or `--batch BATCH_ID`) moves the selected jobs that have not been submitted and returns their status with `moved`. Submitted and finished jobs, and jobs already on that account, stay where they are; `not_moved` lists jobs the target account cannot run. `--transfer` also allows copying datasets the account cannot read, including for jobs already on it.

### Status and pagination

Status, submit, retry, and cancel responses contain `schema_version`, `batch_id`, `total`, `counts`, `jobs`, `next_offset`, and `worker`. Submit and retry also return `replayed`.

The default page size is 20 jobs, with a maximum of 100. `counts` and `total` cover the full selection. Follow `next_offset` until it is null:

```bash
compute-runner agent status --batch BATCH_ID --limit 20 --offset 0
compute-runner agent status --batch BATCH_ID --limit 20 --offset 20
```

Each job summary contains:

| Field | Meaning |
| --- | --- |
| `id`, `name`, `state` | Job identity and current execution state |
| `account` | Account the job runs on |
| `batch_id`, `batch_index` | Batch membership and zero-based input position |
| `resource`, `internet` | CPU or GPU configuration and network setting |
| `downloads`, `outputs_ready` | Download state and completion flag |
| `run_dir` | The job's run folder, fixed at submission |
| `params` | The job's parameters |
| `url` | Provider URL of the run, when an attempt exists |
| `reason`, `error`, `download_error` | Available diagnostic messages, limited to 400 characters each |
| `suggested_account` | Another account that can start a waiting job now (failover `ask`) |
| `suggested_transfer` | Moving to `suggested_account` would copy datasets it cannot read |
| `parent_id` | Original job ID for a retry |

Optional fields are omitted when unavailable. Jobs created before batch support have a null `batch_id` and no `batch_index`. Batch status preserves input order. Status queries accept at most 100 explicit job IDs. Job names and resource labels are limited to 100 characters.

### Change cursors

`changes` returns the latest state of each job with an event after `--after`. Start at 0, then pass the returned `cursor` to the next call:

```bash
compute-runner agent changes --batch BATCH_ID --after 0
compute-runner agent changes --batch BATCH_ID --after RETURNED_CURSOR
```

Responses contain `schema_version`, `batch_id`, `cursor`, `has_more`, `jobs`, and `worker`. Drain additional pages while `has_more` is true. Page size defaults to 20 and is limited to 100.

Events for the same job are coalesced. A job that changes between pages can appear again. The returned records describe current state rather than every historical transition. Each query reads its records and cursor from one SQLite snapshot, so later changes remain visible to a subsequent query.

Keep a separate cursor for each state directory and batch filter. Reset to 0 when changing filters or recovering a lost cursor. Events from other batches may advance a filtered cursor without returning jobs. Unchanged status polls and worker heartbeats produce no job events. Download state, error, account, and `suggested_account` changes do.

### Log retrieval

```bash
compute-runner agent logs JOB_ID --tail 50 --max-bytes 8192
compute-runner agent logs JOB_ID --refresh
```

Kaggle stores a session's log only after it ends. While a submitted job is unfinished, every call reads a live snapshot from Kaggle's log stream, which replays the log from the start. The read stops after 5 idle seconds or 20 seconds in total, and the response has `live: true`.

For a finished job, the first call fetches the stored log and saves a private local copy. Later calls read that copy unless `--refresh` is supplied or the job has finished since the copy was saved. A failed refresh preserves the existing cache.

The response contains `text`, `bytes`, `total_bytes`, `truncated`, `path`, `fetched`, `cached_at`, and `live`, together with `schema_version` and the job `id`. `path` identifies the full cached log. `cached_at` is its modification time as a Unix timestamp.

The default response contains at most 50 lines and 8192 UTF-8 bytes. The maximum permitted limits are 500 lines and 65536 bytes. Lines are split on `\n` only, so carriage-return progress bars count as one line. Byte truncation can leave a partial first line. The size limit applies to the returned text, not the download from Kaggle. Agent log retrieval does not follow a stream.

Logs are cached and returned with known credential formats replaced by `[redacted]`: Kaggle tokens and keys, private key blocks, cloud and service API tokens, JSON web tokens, long values assigned to names such as `api_key` or `password`, and passwords in URLs. [Strict mode](#strict-mode) redacts more broadly.

### Waiting

```bash
compute-runner agent wait --batch BATCH_ID --timeout 300
compute-runner agent wait JOB_ID_1 JOB_ID_2 --timeout 600 --no-downloads
```

`wait` blocks until every selected job settles or the timeout passes, then returns the same fields as `status` plus `settled`, `timed_out`, and `waited_seconds`. A job is settled when it is terminal and its downloads are complete, disabled, or failed, or when it is `blocked` or `needs_attention`. With `--no-downloads`, terminal state is enough. A timeout is not an error, so check `timed_out` and call again if needed. The timeout can be 0 to 86400 seconds. Keep it below the command timeout of the calling tool. `wait` reads local state only. It fails if the worker stays stopped for 30 seconds.

### Outputs

```bash
compute-runner agent outputs JOB_ID
compute-runner agent outputs JOB_ID --limit 100 --offset 100
```

`outputs` lists downloaded files without reading them. The response contains `root` (the run folder), `total`, `files` (each with `path` relative to `root` and `bytes`), `next_offset`, `state`, `downloads`, `outputs_ready`, and, when available, `download_error`, `log_path`, and `record_path` (`job.json`). Files written to `KGR_OUTPUT_DIR` appear as `outputs/NAME`, and other files the run left in its working directory as `working/NAME`. Read them from `root` with ordinary file tools. The listing can be partial until `outputs_ready` is true.

### Python access

```python
from compute_runner import Client

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

`AgentClient()` is also available from `compute_runner` and uses the default configuration.

| Method | Result |
| --- | --- |
| `submit(specs, request_key=..., account=None)` | Batch status and replay flag |
| `preview(specs, account=None)` | Aggregate upload inventory |
| `status(job_ids=None, batch_id=None, states=None, limit=20, offset=0)` | Paginated job summaries and counts |
| `changes(after=0, batch_id=None, limit=20)` | Changed jobs and the next event cursor |
| `logs(job_id, tail=50, max_bytes=8192, refresh=False)` | Bounded text and cache metadata |
| `wait(job_ids=None, batch_id=None, timeout=300, downloads=True, limit=20)` | Status once the selection settles or the timeout passes |
| `outputs(job_id, limit=100, offset=0)` | Downloaded file listing |
| `accounts()` | Accounts, failover policy, slots in use, and last known GPU quota |
| `move(job_ids=None, batch_id=None, account=..., transfer=False, limit=20)` | Status of the selection and the number moved |
| `retry(job_id, request_key=..., account=None)` | Retry batch status and replay flag |
| `continue_run(job_id, request_key=..., account=None)` | Continuation batch status and replay flag |
| `cancel(job_id)` | Updated job status. Repeated cancellation of a cancelled job is accepted |
| `health()` | Worker lock and heartbeat summary |

Agent methods return JSON-compatible dictionaries and raise Python exceptions on errors. CLI responses use `schema_version: 1`. Handled operation errors return a JSON `error` and exit status 1. Argument parsing and startup failures use the standard CLI error output.

### Agent skill

The interface is model-agnostic: any LLM agent that can run shell commands can drive `compute-runner agent`. The [bundled skill](skills/compute-runner/SKILL.md) documents the commands, which actions need the user's approval, request keys, cursor handling, and recovery workflow in the standard `SKILL.md` format. Symlink it into your agent's skills directory so installed copies stay current:

```bash
ln -s ~/compute-runner/skills/compute-runner ~/.claude/skills/compute-runner  # Claude Code
ln -s ~/compute-runner/skills/compute-runner ~/.codex/skills/compute-runner   # Codex
```

Agents without skill support can be pointed at `SKILL.md` directly. The skill expects `compute-runner` on PATH or at `~/compute-runner/.venv/bin/compute-runner`. No MCP server or model API key is required.

## Worker configuration

`compute-runner service install` enables and starts `compute-runner.service` as a systemd user service. It starts on login and restarts after a process failure. Installation does not enable user lingering, so processing may stop after logout. The host must remain running and connected to submit jobs and collect results. Submitted jobs continue remotely and are reconciled when the worker resumes.

```bash
compute-runner service status
compute-runner service restart
compute-runner service stop
compute-runner service start
compute-runner worker status
journalctl --user -u compute-runner.service -f
```

Use `compute-runner worker run` to run in the foreground, or `compute-runner worker run --once` for one dispatch cycle. Only one worker may hold the lock for a state directory.

| Setting | Default |
| --- | --- |
| Managed CPU concurrency per account | 5 |
| Managed GPU concurrency per account | 1 |
| Failover policy | `ask` |
| Status polling interval | 30 seconds |
| Account discovery interval | 300 seconds |
| Workload visibility | Private |
| Strict mode | Off |
| Results folder | `results/` beside each workload's code |
| Dataset copies between accounts | Off |

Set an account's resource limits through `compute-runner account add`, and failover, polling, strict mode, the results folder, and dataset copies through `compute-runner init`, then restart the worker. Omitted options keep their saved values:

```bash
compute-runner account add kaggle YOUR_KAGGLE_USERNAME --cpu-limit 5 --gpu-limit 1
compute-runner init --failover ask --poll-seconds 30
compute-runner init --results-dir ~/experiments --transfer
compute-runner service restart
```

The results folder applies to jobs submitted after the change; existing jobs keep their run folders.

A resource limit of zero pauses launches for that account's pool. CPU and GPU queues are independent, and each account has its own. The worker writes each account's last discovery to `accounts.json` in the state directory. Each discovery also reads the account's GPU quota, and the worker accounts for discovered external runs and quota before admission. Discovery checks only notebooks run within the last 24 hours, which keeps it within Kaggle's rate limits. Kaggle's notebook listing reports every notebook as CPU, so discovery reads each active run's own settings once to count GPU runs correctly. Discovery can be stale, so Kaggle's capacity and quota responses remain authoritative. Local limits do not guarantee available resources or an unlimited CPU allowance.

Dataset preparation, uploads and dataset copies run in the dispatcher and can extend a polling cycle. A copy needs local disk space for about three times the dataset: the download, the snapshot, and its archive. Output downloads run separately.

### Strict mode

Strict mode is off by default, so ordinary workload output and network setups work unchanged. Turn it on when logs or downloads must be locked down, then restart the worker:

```bash
compute-runner init --strict      # --no-strict turns it off again
compute-runner service restart
```

| Behavior | Default | Strict |
| --- | --- | --- |
| Log redaction | Known credential formats and passwords in URLs | Also URL query strings, words after `Bearer`/`Basic`, and values of fields named like tokens, keys, secrets, or passwords. This can hide ordinary text such as `num_tokens=512` |
| Output downloads | Environment proxy, certificate, and `.netrc` settings apply; redirects are followed | Environment settings are ignored; every URL and redirect must be HTTPS on port 443 to a public address |

Error messages saved with jobs always receive the strict redaction, because they can contain signed download URLs. Logs cached before a change keep their previous redaction until fetched again with `compute-runner agent logs JOB_ID --refresh`. Source credential screening and the nonsecret `env` checks apply in both modes.

## Failure handling

| Condition | Behavior |
| --- | --- |
| Capacity or quota rejection | Back off and retry with a fresh notebook slug; the failover policy can move the job to another account |
| Uncertain submission | Query the recorded notebook reference before attempting another submission |
| Unresolved remote execution | Set `needs_attention` and continue reserving capacity |
| Workload failure | Set `failed` and collect available outputs without rerunning the computation |
| Transient upload error | Keep the job preparing on its account; completed uploads are kept and the rest retried |
| Nonretryable upload or authentication error | Set `blocked` and retain the diagnostic message |
| Dataset the job's account cannot read | Copy it when allowed; otherwise set `blocked` and name an account that can read it. See [Datasets across accounts](#datasets-across-accounts) |
| Dataset no connected account can find | Set `blocked` before launching; nothing runs without its data |
| Job's account unknown to the running worker | Keep the job queued with that reason; restart the worker after adding the account, or move the job |
| Download failure | Preserve execution status and retry output collection independently after 1, 2, 4, … minutes, then hourly |

Each attempt records its notebook slug before the remote request. The worker creates a new slug for each attempt and does not overwrite an existing experiment notebook.

`retry` accepts a terminal or blocked job when no execution remains outstanding. It uses saved snapshots. Submit a new workload to change the code or settings.

For an unresolved submission, inspect its URL first. If no remote execution exists, record that confirmation before retrying:

```bash
compute-runner resolve JOB_ID --not-submitted
compute-runner agent retry JOB_ID --request-key resolved-retry-v1
```

`resolve` records an operator assertion and does not launch a job. It must not be used to bypass an active or uncertain execution.

`compute-runner logs JOB_ID --follow` keeps waiting while a running session prints nothing, and gives up only after repeated connection failures.

`compute-runner cancel JOB_ID` and `compute-runner agent cancel JOB_ID` cancel pending work locally. For a running job, they ask Kaggle to stop the session. The job shows the reason `Cancellation requested on ACCOUNT` until the worker sees the run end, usually within a minute. It then becomes `cancelled`, and its partial outputs and log are collected. Kaggle's public API does not return session IDs, so the runtime prints its own session ID at startup and cancellation reads it from the live log. A job that has not started on Kaggle yet, or was submitted by an older version of the runner, cannot be cancelled this way; stop it on its Kaggle page. A submission whose outcome is uncertain is never cancelled automatically.

## State and outputs

Default locations:

| Path | Contents |
| --- | --- |
| `~/.config/compute-runner/config.json` | Account and worker configuration |
| `~/.local/share/compute-runner/queue.sqlite3` | Jobs, batches, request receipts, events, run numbers, downloaded-file receipts, and dataset copies |
| `~/.local/share/compute-runner/bundles/` | Immutable source, input, and dataset-copy snapshots |
| `~/.local/share/compute-runner/logs/JOB_ID.log` | Agent log cache |
| `~/.local/share/compute-runner/accounts.json` | Each account's last run discovery and GPU quota, written by the worker |
| `~/.local/share/compute-runner/jobs/JOB_ID/` | Staged launch packages and the job's download lock |
| `~/.local/share/compute-runner/results/JOB_ID/` | Results of jobs saved before run folders |

Global `--config-dir` and `--state-dir` options select alternate CLI locations. Python callers can use `Client(config=Config(...))` or a `state_dir` override. One state directory can hold jobs for several accounts.

Configurations and job records saved by single-account versions are read as the account `kaggle:OWNER`; their URLs and request keys keep working. `COMPUTE_RUNNER_CONFIG_DIR` and `COMPUTE_RUNNER_STATE_DIR` select alternate default directories; the older `KGR_CONFIG_DIR` and `KGR_STATE_DIR` variables remain supported. Existing installations are discovered automatically when the new default locations contain no configuration or queue. Installation of the renamed service disables the previous managed service so only one worker owns the queue. Workload-facing `KGR_*` variables, bundle formats, and remote artifact IDs remain stable for saved jobs and training scripts.

Results go to run folders outside the state directory; see [Results](#results). The queue database is the source of truth for them: it records each run's number and folder, and the size and SHA-256 of every file downloaded, so an interrupted download resumes without fetching verified files again. `run.log` is present when Kaggle exposes a log.

Output filters match remote relative paths such as `outputs/*.json`. Logs are collected independently of those filters. Files unavailable from Kaggle after a failure or timeout cannot be recovered by the controller.

Execution and download states are separate. `succeeded` means Kaggle reported completion. `download_state=complete`, exposed as `outputs_ready: true` by the agent interface, means output collection completed. Downloads can be pending or failed after a successful computation.

No automatic cleanup removes notebooks, datasets, snapshots, logs, or results. Stop the worker before backing up or moving the complete state directory.

### Database upgrades

Version 0.2 upgrades schema version 1 to version 2 by adding batch and request records. Version 0.3 upgrades it to version 3, whose job records name their account. Version 0.4 upgrades it to version 4, adding run numbers, download receipts, and dataset copies. Existing job IDs, events, and snapshots are preserved. Older application versions cannot open the upgraded database.

Back up the database before upgrading. To restore an older application version, stop the worker and restore a compatible backup. Preserve any jobs and artifacts created after that backup separately.

## Limitations

- Scheduling starts jobs when capacity becomes available. Start times, recurring schedules, and dependency graphs are not implemented.
- Checkpoint continuation requires the stopped run's outputs to be downloaded first.
- Remote cancellation needs the run to have started and to have been submitted by this version of the runner.
- HTTP and MCP servers are not included.
- Only aliased dataset inputs can be copied between accounts, and copying runs in the dispatcher.
- Job completion does not automatically resume an LLM conversation.
- Custom containers and automatic offline dependency installation are not supported.

## Development

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests
.venv/bin/python -m compileall -q src
```

Tests use a Kaggle adapter with simulated network calls and local execution of generated launchers. They cover scheduling, restart recovery, submission ambiguity, batch transactions, idempotency, cursor pagination, packaging, downloads, and CLI behavior. Automated tests do not create remote resources.

### Adding a provider

The queue talks to providers only through the `Provider` protocol in `src/compute_runner/providers/__init__.py`, one instance per account; `connect()` chooses the adapter by `Account.provider`. An adapter:

| Method | Contract |
| --- | --- |
| `check(spec)` | Raise `ValueError` for a specification the provider cannot run, including its dataset references |
| `ensure_bundle(bundle)` | Reuse a content-addressed bundle the account already holds, or upload it; `None` while it is still processing |
| `resolve_dataset(ref)` | Pin a dataset the account can read; `None` if it cannot. Every call checks access |
| `fetch_dataset(ref, destination)` | Download a readable dataset as plain files, so it can be copied to another account |
| `stage(job, number)` | Build the attempt's launch package locally and return its deterministic remote reference. It must expose every input as `KGR_INPUT_<ALIAS>` |
| `submit(job)` | Launch the staged attempt; raise `RemoteError` with `definitive=True` only when nothing was launched |
| `status(ref)` | Return a state of `queued`, `running`, `cancelling`, `succeeded`, `failed`, or `cancelled` (or `None`), the raw provider state, and an error |
| `download(ref, sink)` | Give the run's log and files to an output sink, which decides what to fetch and where it goes |
| `url`, `cancel`, `active_runs`, `quota`, `logs`, `live_log` | Links, cancellation, capacity discovery, quota, and logs |

The reference returned by `stage` is saved before `submit` runs, so an interrupted submission is reconciled through `status` rather than launched twice. Workloads read the provider-neutral `KGR_*` runtime variables; `providers/downloads.py` verifies outputs exposed as signed HTTPS URLs.

The Kaggle adapter pins `kaggle==2.2.4` and `kagglesdk==0.1.37`. SDK transport retries are disabled. The worker determines whether a remote operation can be retried. The Kaggle client is imported when a remote operation is required.

Example workloads are under [examples](examples). The GPU smoke test requires `--gpu --internet`. Running examples on Kaggle creates private resources and uses the corresponding compute allocation. See [VALIDATION.md](VALIDATION.md) for recorded test results and live checks.

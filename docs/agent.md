# Agent interface

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

## Submission keys

Agent submissions and retries require `--request-key`. Standard `compute-runner submit` and the Python submission methods also accept a key.

A key identifies one intended operation within a state directory. Repeating the same request returns the original batch with `replayed: true`. This holds across concurrent callers, process restarts, and completed runs. Reusing the key with different settings raises an error. Submit and retry operations share the same key namespace.

Keys must match `[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}`. The comparison uses normalized workload settings, including paths. It does not compare the current contents of source files. A replay uses the original snapshots even if those files have changed or disappeared. Use a new key to run changed code or intentionally repeat an experiment. Submission receipts do not expire automatically.

Retries use the original saved files:

```bash
compute-runner agent retry JOB_ID --request-key experiment-retry-v1
```

`agent submit`, `agent retry` and `agent continue` accept `--account`. An explicit account is part of the request, so a replay must repeat it; requests without one keep their original fingerprint.

## Accounts

`agent accounts` reads local state only. It returns `failover`, `default`, and for each account in preference order: `id`, `provider`, `cpu` and `gpu` (`used`, `limit`, and `free`: slots a new job could take now, null while the account's runs are unknown), `gpu_quota_limited` (false on SSH machines, which have no GPU time limit), `gpu_quota_seconds` and `checked_age_seconds` from the worker's last check (null when not checked yet), and `error` when that check failed. `used` counts this queue's runs and other runs the worker discovered.

`agent move JOB_ID... --account ID` (or `--batch BATCH_ID`) moves the selected jobs that have not been submitted and returns their status with `moved`. Submitted and finished jobs, and jobs already on that account, stay where they are; `not_moved` lists jobs the target account cannot run. `--transfer` also allows copying datasets the account cannot read, including for jobs already on it.

## Status and pagination

Status, submit, retry, and cancel responses contain `schema_version`, `batch_id`, `total`, `counts`, `jobs`, `next_offset`, and `worker`. Submit and retry also return `replayed`.

The default page size is 20 jobs, with a maximum of 100. A batch is listed in submission order; any other selection lists the newest jobs first. `counts` and `total` cover the full selection. Follow `next_offset` until it is null:

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

## Change cursors

`changes` returns the latest state of each job with an event after `--after`. Start at 0, then pass the returned `cursor` to the next call:

```bash
compute-runner agent changes --batch BATCH_ID --after 0
compute-runner agent changes --batch BATCH_ID --after RETURNED_CURSOR
```

Responses contain `schema_version`, `batch_id`, `cursor`, `has_more`, `jobs`, and `worker`. Drain additional pages while `has_more` is true. Page size defaults to 20 and is limited to 100.

Events for the same job are coalesced. A job that changes between pages can appear again. The returned records describe current state rather than every historical transition. Each query reads its records and cursor from one SQLite snapshot, so later changes remain visible to a subsequent query.

Keep a separate cursor for each state directory and batch filter. Reset to 0 when changing filters or recovering a lost cursor. Events from other batches may advance a filtered cursor without returning jobs. Unchanged status polls and worker heartbeats produce no job events. Download state, error, account, and `suggested_account` changes do.

## Log retrieval

```bash
compute-runner agent logs JOB_ID --tail 50 --max-bytes 8192
compute-runner agent logs JOB_ID --refresh
```

While a submitted job is unfinished, every call reads a live snapshot, and the response has `live: true`. Kaggle stores a session's log only after it ends, so on Kaggle the snapshot comes from its log stream, which replays the log from the start and is read for at most 20 seconds (5 when idle). On an SSH machine it is the end of the run's log file.

For a finished job, the first call fetches the stored log and saves a private local copy. Later calls read that copy unless `--refresh` is supplied or the job has finished since the copy was saved. A failed refresh preserves the existing cache.

The response contains `text`, `bytes`, `total_bytes`, `truncated`, `path`, `fetched`, `cached_at`, and `live`, together with `schema_version` and the job `id`. `path` identifies the full cached log. `cached_at` is its modification time as a Unix timestamp.

The default response contains at most 50 lines and 8192 UTF-8 bytes. The maximum permitted limits are 500 lines and 65536 bytes. Lines are split on `\n` only, so carriage-return progress bars count as one line. Byte truncation can leave a partial first line. The size limit applies to the returned text, not the download from the provider. Agent log retrieval does not follow a stream.

Logs are cached and returned with known credential formats replaced by `[redacted]`: Kaggle tokens and keys, private key blocks, cloud and service API tokens, JSON web tokens, long values assigned to names such as `api_key` or `password`, and passwords in URLs. [Strict mode](operations.md#strict-mode) redacts more broadly.

## Waiting

```bash
compute-runner agent wait --batch BATCH_ID --timeout 300
compute-runner agent wait JOB_ID_1 JOB_ID_2 --timeout 600 --no-downloads
```

`wait` blocks until every selected job settles or the timeout passes, then returns the same fields as `status` plus `settled`, `timed_out`, and `waited_seconds`. A job is settled when it is terminal and its downloads are complete, disabled, or failed, or when it is `blocked` or `needs_attention`. With `--no-downloads`, terminal state is enough. A timeout is not an error, so check `timed_out` and call again if needed. The timeout can be 0 to 86400 seconds. Keep it below the command timeout of the calling tool. `wait` reads local state only. It fails if the worker stays stopped for 30 seconds.

## Outputs

```bash
compute-runner agent outputs JOB_ID
compute-runner agent outputs JOB_ID --limit 100 --offset 100
```

`outputs` lists downloaded files without reading them. The response contains `root` (the run folder), `total`, `files` (each with `path` relative to `root` and `bytes`), `next_offset`, `state`, `downloads`, `outputs_ready`, and, when available, `download_error`, `log_path`, and `record_path` (`job.json`). Files written to `KGR_OUTPUT_DIR` appear as `outputs/NAME`, and other files the run left in its working directory as `working/NAME`. Read them from `root` with ordinary file tools. The listing can be partial until `outputs_ready` is true.

## Python access

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
| `accounts(live=False)` | Accounts, failover policy, slots in use and free, and last known GPU quota |
| `running(resource=None, account=None, live=False)` | Runs holding each account's slots, including runs started elsewhere (no `job_id`), and each account's discovery age |
| `move(job_ids=None, batch_id=None, account=..., transfer=False, limit=20)` | Status of the selection and the number moved |
| `retry(job_id, request_key=..., account=None)` | Retry batch status and replay flag |
| `continue_run(job_id, request_key=..., account=None)` | Continuation batch status and replay flag |
| `cancel(job_id)` | Updated job status. Repeated cancellation of a cancelled job is accepted |
| `health()` | Worker lock and heartbeat summary |

Agent methods return JSON-compatible dictionaries and raise Python exceptions on errors. CLI responses use `schema_version: 1`. Handled operation errors return a JSON `error` and exit status 1. Argument parsing and startup failures use the standard CLI error output.

## Agent skill

The interface works with any LLM agent that can run shell commands. The [bundled skill](../skills/compute-runner/SKILL.md) is plain Markdown with a short front matter (`name`, `description`). It documents the commands, request keys, cursor handling and recovery, and what the agent may do on its own. The agent leaves packaging, uploads, downloads and file handling to the application. It changes your code, for example to add checkpointing or to fix a job that failed, only after you approve. It never handles credentials. For agents that load skills from a folder, link it into that folder so installed copies stay current:

```bash
ln -s ~/compute-runner/skills/compute-runner PATH/TO/YOUR/AGENT/skills/compute-runner
```

Agents without skill support can be pointed at `SKILL.md` directly, for example from their instructions file. The skill expects `compute-runner` on PATH or at `~/compute-runner/.venv/bin/compute-runner`. No MCP server or model API key is required.

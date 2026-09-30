# Python API

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
| `retry(job_id, request_key=None, account=None)` | Create a job from the original saved files and settings, on the original account unless one is given; on the same account it keeps the job's dataset copy permission |
| `retry_batch(job_id, request_key=None, account=None)` | Create a retry and return its single-job batch |
| `retry_jobs(job_ids, request_key=None, account=None)` | Rerun several jobs as one batch, or none when one of them cannot be rerun |
| `continue_run(job_id, request_key=None, account=None)` | Continue a stopped resumable run from its verified checkpoint as the next run of its experiment |
| `continue_batch(job_id, request_key=None, account=None)` | Create a continuation and return its single-job batch |
| `move(job_id, account, transfer=False)` | Place a job that has not been submitted on another account; `transfer` allows copying datasets it cannot read |
| `cancel(job_id)` | Cancel pending work locally, or ask the provider to stop a running job |
| `cancel_many(job_ids)` | Cancel each job; returns `cancelled` and the `failed` ones with their errors |
| `resolve_not_submitted(job_id)` | Record an operator's confirmation that a job needing attention has no remote execution |
| `quota(account=None)` | Query one account's accelerator quota, or every account's |
| `cleanup(older_than_days=7, include_snapshots=False, accounts=None, local=True, delete=False, limit=None)` | Report what the runner left behind and what may go; `delete=True` removes it (see [Cleanup](operations.md#cleanup)) |
| `provider(account=None)` | Return the provider adapter for an account |
| `worker_health()` | Read worker lock ownership and heartbeat data |
| `worker()` | Construct a worker for this configuration |
| `agent()` | Return the compact automation interface |

Use complete job IDs with `Client`. Optional parameters shown after the first argument are keyword arguments. Creating a client or reading local state does not authenticate to any provider. Submission is local, while logs, downloads, and quota queries contact the job's provider when needed.

`wait` raises `TimeoutError` when its local wait deadline expires. This does not cancel the job. With `downloads=False`, it returns after execution terminates. A blocked or uncertain job is returned for inspection, as is a finished job whose `download_state` is `error`; the worker keeps retrying that download. Waiting raises an error once no worker has run for 30 seconds.

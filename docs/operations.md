# Operations

## Commands and settings

`compute-runner account add` connects an account and its local scheduling limits; see [Accounts and failover](accounts.md). `compute-runner init` saves worker-wide settings: the failover policy, polling interval, results folder, dataset copies and [strict mode](#strict-mode). Rerunning either command changes only the options you pass. `compute-runner doctor` checks the local configuration, where each account's login comes from and, for each account, remote quota information and active runs. Use `compute-runner doctor --offline` for local checks only.

The worker runs as a `systemd --user` service, or in a terminal without systemd; local process locking uses `fcntl`. The service must be able to authenticate without an interactive shell, so keep credentials in the [credentials file](accounts.md#credentials) or Kaggle's standard locations. Credentials supplied only through temporary shell variables are not copied into the generated service unit.

Submission writes to the local queue; a running worker is required to upload and launch jobs. CLI job IDs may be unambiguous prefixes; batch IDs must be complete. The standard CLI prints full records as JSON with the global `--json` option (`compute-runner --json status JOB_ID`); use `compute-runner agent` for bounded responses intended for automation.

## Looking across jobs and accounts

```bash
compute-runner list gpu --account kaggle:alice --active   # filter jobs: resource, account, state
compute-runner list --state failed --limit 20             # the newest 20 failed jobs
compute-runner running                                    # runs holding CPU and GPU slots on every account
compute-runner running gpu --watch                        # redraw every polling interval until Ctrl-C
compute-runner gpus                                       # GPU slots, GPU time left and when it resets
compute-runner account list                               # CPU and GPU slots of every account, with totals
compute-runner logs JOB_ID --tail 50                      # the end of a run's log
```

`running` lists this queue's submitted jobs with their notebook or run folder, type, state and time since launch, followed by runs started outside this queue (such as a notebook run on Kaggle's website), marked `(external)`. A run whose type discovery could not tell holds both CPU and GPU slots, so both filters show it. `gpus` shows, for each account, the GPU slots in use, how many a new job could take now (none once the account's GPU time is spent), the GPU time left, when Kaggle restores it, the GPU models of SSH machines, and when the worker last checked. `account list` shows the same for CPU and GPU slots. Totals read `>=` while an account is unknown. These views apply the worker's own admission rule, so a free slot is one the worker would use. They read the worker's last [discovery](#worker-configuration), which it refreshes only while it has jobs to place, and say when that is old or failed. `--live` asks every provider now instead; on Kaggle this costs a request per notebook run in the last 24 hours, so `running --watch --live` redraws at most once a minute.

## Acting on several jobs

```bash
compute-runner cancel JOB_ID JOB_ID                       # several jobs
compute-runner cancel --batch BATCH_ID                    # a batch's unfinished jobs
compute-runner cancel --account kaggle:alice --state queued
compute-runner retry --batch BATCH_ID --state failed      # rerun a batch's failures together
```

Cancelling more than one job asks for confirmation (`--yes` skips it) and lists the jobs that could not be cancelled. A retry of several jobs queues all of them as one batch, or none when one of them cannot be rerun.

## Cleanup

Nothing is removed automatically. `compute-runner cleanup` reports what the runner left behind: launch notebooks and bundle datasets on Kaggle, run folders and bundles on SSH machines, and the state directory's snapshots, staging folders, upload folders and log copies. Results folders are never touched.

```bash
compute-runner cleanup                       # report, with sizes and why each item stays or may go
compute-runner cleanup --no-remote           # the state directory only, without remote calls
compute-runner cleanup --older-than 30 --delete
```

An item may go once every job using it has finished, more than `--older-than` days ago (default 7), with its outputs downloaded or not wanted. A bundle shared with an unfinished job stays. Remote items this queue did not create are reported as unknown and never deleted, and a run still running on an SSH machine is refused. Local snapshots stay unless `--include-snapshots` is given, because `retry` and `continue` need them. `--delete` asks for confirmation (`--yes` skips it) and deletes only the items reported as reclaimable.

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
| Managed GPU concurrency per account | 1 on Kaggle, 0 on SSH machines |
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

A resource limit of zero pauses launches for that account's pool. CPU and GPU queues are independent, and each account has its own. The worker writes each account's last discovery to `accounts.json` in the state directory. Each discovery also reads the account's GPU quota, and the worker accounts for discovered external runs and quota before admission. On Kaggle, discovery checks only notebooks run within the last 24 hours, which keeps it within Kaggle's rate limits, and reads each active run's own settings once, because Kaggle's notebook listing reports every notebook as CPU. On an SSH machine, it lists the runner's own runs in the work directory. Discovery can be stale, so Kaggle's capacity and quota responses remain authoritative. Local limits do not guarantee available resources or an unlimited CPU allowance.

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
| Capacity or quota rejection | Back off and retry with a fresh remote reference; the failover policy can move the job to another account |
| Uncertain submission | Query the recorded remote reference before attempting another submission |
| Unresolved remote execution | Set `needs_attention` and continue reserving capacity |
| Workload failure | Set `failed` and collect available outputs without rerunning the computation |
| Transient upload error | Keep the job preparing on its account; completed uploads are kept and the rest retried |
| Nonretryable upload or authentication error | Set `blocked` and retain the diagnostic message |
| Dataset the job's account cannot read | Copy it when allowed; otherwise set `blocked` and name an account that can read it. See [Datasets across accounts](accounts.md#datasets-across-accounts) |
| Dataset no connected account can find | Set `blocked` before launching; nothing runs without its data |
| Job's account unknown to the running worker | Keep the job queued with that reason; restart the worker after adding the account, or move the job |
| Download failure | Preserve execution status and retry output collection independently after 1, 2, 4, … minutes, then hourly |

Each attempt records its remote reference (a Kaggle notebook slug, or a run folder on an SSH machine) before the remote request. The worker creates a new reference for each attempt and never overwrites an earlier one.

`retry` accepts a terminal or blocked job when no execution remains outstanding. It uses saved snapshots. Submit a new workload to change the code or settings.

For an unresolved submission, or a run the provider no longer knows (for example, a deleted notebook), inspect its URL first. If no remote execution exists, record that confirmation before retrying:

```bash
compute-runner resolve JOB_ID --not-submitted
compute-runner agent retry JOB_ID --request-key resolved-retry-v1
```

`resolve` records an operator assertion and does not launch a job. It must not be used to bypass an active or uncertain execution.

`compute-runner logs JOB_ID --follow` keeps waiting while a running session prints nothing, and gives up only after repeated connection failures.

`compute-runner cancel JOB_ID` and `compute-runner agent cancel JOB_ID` cancel pending work locally. For a running job on an SSH machine, the run's supervisor stops it (see [SSH machines](ssh.md)). On Kaggle, they ask Kaggle to stop the session. The job shows the reason `Cancellation requested on ACCOUNT` until the worker sees the run end, usually within a minute. It then becomes `cancelled`, and its partial outputs and log are collected. Kaggle's public API does not return session IDs, so the runtime prints its own session ID at startup and cancellation reads it from the live log. A job still queued on Kaggle has no session yet: cancellation deletes that attempt's launch notebook instead, which removes the run from Kaggle's queue, and the job becomes `cancelled` at once with the reason `Cancelled before it started on ACCOUNT`. A job that is starting but has not printed its session ID yet, or was submitted by an older version of the runner, cannot be cancelled; try again shortly or stop it on its Kaggle page. A submission whose outcome is uncertain is never cancelled automatically.

## State and outputs

Default locations:

| Path | Contents |
| --- | --- |
| `~/.config/compute-runner/config.json` | Account and worker configuration, without secrets |
| `~/.config/compute-runner/credentials.json` | Every account's keys, tokens and passwords; see [Credentials](accounts.md#credentials) |
| `~/.config/compute-runner/known_hosts` | SSH host keys accepted with `--trust-new-host` |
| `~/.local/share/compute-runner/queue.sqlite3` | Jobs, batches, request receipts, events, run numbers, downloaded-file receipts, and dataset copies |
| `~/.local/share/compute-runner/bundles/` | Immutable source, input, and dataset-copy snapshots |
| `~/.local/share/compute-runner/logs/JOB_ID.log` | Agent log cache |
| `~/.local/share/compute-runner/accounts.json` | Each account's last run discovery and GPU quota, written by the worker |
| `~/.local/share/compute-runner/jobs/JOB_ID/` | Staged launch packages and the job's download lock |
| `~/.local/share/compute-runner/results/JOB_ID/` | Results of jobs saved before run folders |

Global `--config-dir` and `--state-dir` options select alternate CLI locations. Python callers can use `Client(config=Config(...))` or a `state_dir` override. One state directory can hold jobs for several accounts.

Configurations and job records saved by single-account versions are read as the account `kaggle:OWNER`; their URLs and request keys keep working. `COMPUTE_RUNNER_CONFIG_DIR` and `COMPUTE_RUNNER_STATE_DIR` select alternate default directories; the older `KGR_CONFIG_DIR` and `KGR_STATE_DIR` variables remain supported. Existing installations are discovered automatically when the new default locations contain no configuration or queue. Installation of the renamed service disables the previous managed service so only one worker owns the queue. Workload-facing `KGR_*` variables, bundle formats, and remote artifact IDs remain stable for saved jobs and training scripts.

Results go to run folders outside the state directory; see [Results](results.md). The queue database is the source of truth for them: it records each run's number and folder, and the size and SHA-256 of every file downloaded, so an interrupted download resumes without fetching verified files again. `run.log` is present when the provider exposes a log.

Output filters match remote relative paths such as `outputs/*.json`. Logs are collected independently of those filters. Files the provider no longer has after a failure or timeout cannot be recovered.

Execution and download states are separate. `succeeded` means the provider reported completion. `download_state=complete`, exposed as `outputs_ready: true` by the agent interface, means output collection completed. Downloads can be pending or failed after a successful computation.

No automatic cleanup removes notebooks, datasets, SSH run folders, snapshots, logs, or results; `compute-runner cleanup` reports what may go and deletes it on request (see [Cleanup](#cleanup)). Stop the worker before backing up or moving the complete state directory.

### Database upgrades

Version 0.2 upgrades schema version 1 to version 2 by adding batch and request records. Version 0.3 upgrades it to version 3, whose job records name their account. Version 0.4 upgrades it to version 4, adding run numbers, download receipts, and dataset copies. Existing job IDs, events, and snapshots are preserved. Older application versions cannot open the upgraded database.

Back up the database before upgrading. To restore an older application version, stop the worker and restore a compatible backup. Preserve any jobs and artifacts created after that backup separately.

# Verification record

Verified on 2026-09-26 using the installed Python 3.12 environment. Sections are newest first.

## Review of experiment folders and datasets

- 208 automated tests passed; Ruff passed. Each new regression test fails on the previous code.
- A dataset that no connected account can find (a wrong reference, a pinned version past the latest, or no access) now blocks the job before anything is uploaded or launched, with an error and a reason that ask to check the reference; a retry runs once an account may read it. An account whose access check fails transiently keeps the job retrying; one with definitive errors, such as a revoked key, counts as unable to read.
- `continue` gives a trailing bare `--resume` the value `required`. A `results_dir` whose experiment folder would be the source folder is refused. Cancelling clears `suggested_transfer`. Hyphenated secret-looking parameter names such as `hf-token` are rejected.
- Simulated end to end through the CLI, with each launch executed locally against a fake `/kaggle` tree and remote notebooks holding their slots until finished: a six-job sweep over two accounts with two CPU slots each, one slot taken by an outside notebook, filling every free slot, failing over under `auto`, and starting waiting jobs as slots freed; a GPU job waiting for quota under `ask`, moved after approval; an account added while a worker ran, held until restart; a revoked key that holds the job under `failover: off`, a rotated key that launches it, and `auto` moving a job off the broken account; account removal refused while its jobs were unfinished; a checkpointed run continued twice, the last time on another account, resuming from the verified checkpoint each time; misspelled and nonexistent-version datasets blocked, the corrected reference attached and read, and a private dataset copied after `move --transfer`.
- Not validated live: Kaggle was unreachable from the review environment (its network policy denied `www.kaggle.com` and `api.kaggle.com`), so Kaggle's API behavior, dataset mount paths, `fetch_dataset`, and real slot and quota limits were exercised only through the simulation.

## Experiment folders and datasets across accounts (v0.4.0)

- 204 automated tests passed, including 25 new cases; Ruff, compilation, and `git diff --check` passed.
- Run folders: numbering inside the batch transaction, numbering after folders already on disk, the `NNN_YYYY-MM-DD_HH-MM-SS` format, results-folder precedence (workload, then configuration, then beside the code), leaving a results folder inside a project out of its snapshot, `job.json` and `runs.md` written by the worker and not rewritten after a restart unless the job changes, output placement in `outputs/` and `working/` with download receipts in the queue, and the older layout for existing jobs.
- Parameters reach a locally executed launcher as `--NAME VALUE` and `KGR_PARAMS_JSON`; `--param` values are typed like YAML and secret-looking ones are rejected. `job:ID/PATH` inputs resolve to downloaded outputs and are refused before downloads complete.
- `continue` refuses a running job, a missing manifest, and a checksum mismatch; it uploads only `latest.json` and the checkpoint it names, requires `--resume required`, replays by request key, and can run on another account.
- Datasets: an aliased Kaggle dataset is found by the launcher through `KGR_INPUT_<ALIAS>` under a simulated mount. A dataset the job's account cannot read blocks the job until copies are allowed, is then downloaded once through the account that can read it, and is uploaded and verified like a local input; a later job reuses the copy. A dataset no account can read is launched as given. Unaliased datasets are never copied. Failover prefers an account that can read the data, offers one that needs a copy with `suggested_transfer`, and moves there under `auto` only when copies are allowed. The Kaggle adapter checks access even for pinned references and treats 403 and 404 as unreadable.
- A throwaway queue with the CLI returned each job's `run_dir` and parameters at submission, and the dry run showed the experiment folder. The live queue's backup copy (schema 2, eight jobs) upgraded to schema 4; every job kept its previous results path, and publishing wrote nothing for them.
- The live queue, configuration, and service were not changed, and no remote workloads were submitted. `fetch_dataset` was not exercised against Kaggle.

## Review of the rename and accounts changes

- 179 automated tests passed; Ruff and compilation passed. Each regression test below fails on the previous code.
- Failover under `auto`: a job no longer bounces between two full accounts, because an account that rejected one of its launches is not chosen again. A job that is retrying uploads stays on its account and keeps completed uploads, and jobs queued behind it no longer fail over with a false capacity reason. Suggestions and moves see slots taken earlier in the same cycle. GPU quota is read with each discovery instead of every cycle on every account.
- A job submitted to an account added after the worker started now waits for a worker restart instead of being blocked; configuration commands remind a running worker to restart. `account add` in other casing keeps the saved ID. A one-off `--state-dir` is no longer saved by `account` or `init`. Dry runs check the spec against the account. `agent changes` reports new failover suggestions. Moving a job to its own account is refused. Failed downloads for a removed account are not retried until it returns. The legacy `kaggle-runner` unit can no longer be started, since its package is gone.
- The queue schema is now version 3 so versions without accounts refuse to open it; the package version is 0.3.0.
- Workload-file parsing moved to `workloads.py` and the shared operation errors to `agent.py`, removing the import cycle between the two CLIs. Every command has help text, and the skill lists which actions need the user's approval.
- No remote workloads were submitted; the running service was not restarted.

## Provider accounts and failover

- The queue now talks to providers only through the `Provider` protocol; Kaggle is its one adapter. 170 automated tests passed, including 20 new cases for accounts, failover under the `off`/`ask`/`auto` policies, moves, per-account credentials, discovery, and single-account upgrades. Ruff, compilation, and `git diff --check` passed.
- Backed up the live queue to `~/.local/share/compute-runner/queue.pre-accounts.sqlite3` before the service restart. The saved single-owner configuration is read as the account `kaggle:dimpap99`, and all seven existing job records report that account with unchanged notebook URLs. `compute-runner doctor` authenticated and passed the new username check. The service restarted on the new code and is running.
- A throwaway queue with a second account holding a fake key showed that `kagglesdk` sends any ambient access token (`~/.kaggle/access_token`) instead of an explicitly configured key. One smoke submission for that account reached Kaggle under the main account's token and was rejected for capacity; nothing was launched. The adapter now pins the transport's authentication to the credentials it verified. A rerun returned `401 Unauthorized` for the fake account, and failover `ask` suggested `kaggle:dimpap99`.
- Kaggle's notebook listing can start with empty placeholder entries dated 2010-04-01. Discovery stopped at them and reported no active runs while Kaggle enforced its 5-session CPU limit. Discovery now skips them and reported all five running research notebooks. A launch that the provider rejects for capacity or quota now also makes a waiting job eligible for failover.
- The live end-to-end job (`examples/hello.py`) did not run: the account was at Kaggle's 5-session CPU limit because of the user's own research notebooks. It was cancelled locally before launch, so it will not take a slot.

## Compute Runner rename

- 150 automated tests passed after the rename. These use fake remote services, transactional local queues, and local execution of generated launchers. Ruff checks, Python compilation, and `pip check` passed.
- `compute-runner` is installed at `~/.local/bin/compute-runner` and points to this project's isolated environment. The original `/home/dimpap/LocEstim` repository remains clean.
- Renamed the Python distribution, source package, CLI, documentation, notebook example, installed skill, project directory, and systemd user service. The README and command help say that Kaggle is the only supported provider.
- Rebuilt the virtual environment at `/home/dimpap/compute-runner/.venv` and installed the renamed package. Both `compute-runner` and the `kgr` compatibility alias work from outside the project directory.
- Moved local configuration and state to the new default directories. Compatibility links preserve paths in existing job records and source snapshots. All seven job records remain byte-for-byte identical, all six downloaded output directories remain accessible, and the queue database checksum was unchanged by the move.
- Added nine regression cases covering existing-queue discovery, request-key replay, saved outputs, new and legacy environment overrides, and service migration. The renamed service is enabled and running; its unit passed `systemd-analyze --user verify`.
- The installed skill is now a symlink to the bundled `skills/compute-runner` directory and passes skill validation. Ruff, `pip check`, and `git diff --check` passed.
- No new remote workloads were submitted for the rename. Existing workload environment variables, snapshot formats, and remote artifact identities were retained.

## Agent interface (v0.2.0)

- Tested atomic batch submission on snapshot and database failures, concurrent same-key submissions, conflicting request keys, explicit retries, restart/completion replay, and replay after source files change or disappear.
- Tested bounded status pages, deterministic batch order, per-batch cursors, coalesced changes during pagination, download-error events, quiet polling, and v1 queue migration.
- Tested bounded Unicode log tails, private full-log caching, offline cache reuse, and preserving an existing cache when a refresh fails. CLI tests cover structured operation errors and CPU overrides of GPU workload files.
- Validated the bundled agent skill. Its current installation is `~/.codex/skills/compute-runner`, linked to the repository so later updates stay current. Command examples were exercised by the CLI integration tests and read-only live checks.
- Backed up the live v1 database to `~/.local/share/compute-runner/queue.pre-agent-v1.sqlite3` before upgrading. All seven existing local records were preserved. Four pages of changes returned those seven jobs; a subsequent query returned no changed jobs.
- On those existing records, the full JSON was 12,057 bytes and compact status was 2,920 bytes (about 76% smaller). This is a response-size measurement on that sample, not a universal token-saving guarantee. A live three-line log query returned 187 bytes from a 528-byte log, retaining the full private cache.
- The systemd user service was restored and remains enabled. No additional Kaggle workloads were launched for this interface update; submission/worker integration was exercised with the fake backend, and remote log retrieval was checked against an existing successful run.

The agent interface is a CLI/Python API plus a model-agnostic agent skill. It does not provide an MCP server, timed/recurring jobs, or automatic conversation wakeups.

## Code review and strict mode

- A review of the whole codebase fixed notebook slugs ending in `-`, case-sensitive capacity accounting, worker-lock races with health probes, `compute-runner init` resetting saved limits, `~` in workload YAML, rejected attempts reported as remote runs, retryable dataset-inventory failures, and quota refresh times read as local time. Each fix has a regression test that fails on the previous code.
- Dispatch with 600 queued jobs dropped from about 50 seconds per worker cycle to under 1 second, measured with the fake backend.
- Tests cover strict mode on and off for log redaction and downloads, `compute-runner logs --follow` through quiet periods, non-Python supporting notebooks, input folders that `.gitignore` their data, and download retry backoff.

## Live Kaggle checks

| Workload | Kaggle notebook | Result |
| --- | --- | --- |
| CPU script | [aedcfc831544](https://www.kaggle.com/code/dimpap99/kgr-smoke-script-aedcfc831544-a1) | Succeeded; outputs downloaded; one attempt |
| Python notebook | [9b2ca69cbea6](https://www.kaggle.com/code/dimpap99/kgr-smoke-notebook-9b2ca69cbea6-a1) | Succeeded; outputs downloaded; one attempt |
| GPU and internet | [79c48e8f6c52](https://www.kaggle.com/code/dimpap99/kgr-smoke-gpu-79c48e8f6c52-a1) | Succeeded; outputs downloaded; one attempt |
| Python project with private local data | [1d82868d242c](https://www.kaggle.com/code/dimpap99/kgr-smoke-project-1d82868d242c-a1) | Succeeded; outputs downloaded; one attempt |
| Upload reuse and service restart | [0b7aaebd2087](https://www.kaggle.com/code/dimpap99/kgr-smoke-project-0b7aaebd2087-a1) | Succeeded; outputs downloaded; one attempt |
| Final response normalization | [dd59154ebdf3](https://www.kaggle.com/code/dimpap99/kgr-smoke-script-dd59154ebdf3-a1) | Succeeded; outputs downloaded; one attempt |

The GPU check detected two Tesla T4 devices and successfully made a small HTTPS request. The project check imported a sibling module, read a project-relative configuration file, and consumed an independently uploaded private dataset. Its repeat used exactly the same dataset references. The service restarted while that attempt was being reconciled and did not create a second attempt.

The final script check exercised the corrected handling of Kaggle's `/code/owner/slug` response: it recorded an accepted version-1 submission directly, without an uncertainty error. Tests cover that path, absolute URLs, bare slugs, and versioned references. Tests also cover Kaggle returning 403 for absent datasets; creation requires a successful inventory of the user's own datasets first.

One initial local project entry was blocked before submission by the absent-dataset 403 behavior. It was explicitly retried after the adapter fix and the superseded local entry was cancelled. All test jobs are now terminal. Private test notebooks and the two small reusable datasets are retained, consistent with the tool's no-automatic-deletion policy.

Results and source/provenance records are under `~/.local/share/compute-runner/results`. Existing research notebook executions were observed for capacity accounting and were not modified.

## Boundaries

Live validation used small workloads; multi-gigabyte transfers, platform-wide outages, and twelve-hour executions were not exercised. Crash recovery, pagination, partial downloads, quota/capacity failures, and concurrent submissions are covered by deterministic tests. Remote cancellation of running jobs and strict-mode downloads are covered by tests with a fake backend but were not exercised against live Kaggle runs. Scheduling depends on the local user service being online.

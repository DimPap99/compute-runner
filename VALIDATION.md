# Verification record

Verified on 2026-09-26 using the installed Python 3.12 environment.

- 78 automated tests passed. These use fake remote services, transactional local queues, and local execution of generated launchers.
- Ruff checks, Python compilation, and `pip check` passed.
- The systemd user unit passed `systemd-analyze --user verify`; the service is enabled and running.
- `kgr` is installed at `~/.local/bin/kgr` and points to this project's isolated environment.
- The original `/home/dimpap/LocEstim` repository remains clean.

## Agent interface (v0.2.0)

- Tested atomic batch submission on snapshot and database failures, concurrent same-key submissions, conflicting request keys, explicit retries, restart/completion replay, and replay after source files change or disappear.
- Tested bounded status pages, deterministic batch order, per-batch cursors, coalesced changes during pagination, download-error events, quiet polling, and v1 queue migration.
- Tested bounded Unicode log tails, private full-log caching, offline cache reuse, and preserving an existing cache when a refresh fails. CLI tests cover structured operation errors and CPU overrides of GPU workload files.
- Validated the installed `~/.codex/skills/kaggle-runner` skill; its copy under `skills/kaggle-runner` matches. Command examples were exercised by the CLI integration tests and read-only live checks.
- Backed up the live v1 database to `~/.local/share/kaggle-runner/queue.pre-agent-v1.sqlite3` before upgrading. All seven existing local records were preserved. Four pages of changes returned those seven jobs; a subsequent query returned no changed jobs.
- On those existing records, the full JSON was 12,057 bytes and compact status was 2,920 bytes (about 76% smaller). This is a response-size measurement on that sample, not a universal token-saving guarantee. A live three-line log query returned 187 bytes from a 528-byte log, retaining the full private cache.
- The systemd user service was restored and remains enabled. No additional Kaggle workloads were launched for this interface update; submission/worker integration was exercised with the fake backend, and remote log retrieval was checked against an existing successful run.

The agent interface is a CLI/Python API plus a Codex skill. It does not provide an MCP server, timed/recurring jobs, or automatic conversation wakeups.

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

Results and source/provenance records are under `~/.local/share/kaggle-runner/results`. Existing research notebook executions were observed for capacity accounting and were not modified.

## Boundaries

Live validation used small workloads; multi-gigabyte transfers, platform-wide outages, and twelve-hour executions were not exercised. Crash recovery, pagination, partial downloads, quota/capacity failures, and concurrent submissions are covered by deterministic tests. Active remote cancellation still requires Kaggle's web UI. Scheduling depends on the local user service being online.

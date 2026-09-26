# Verification record

Verified on 2026-09-26 using the installed Python 3.12 environment.

- 52 automated tests passed. These use fake remote services and local execution of generated launchers.
- Ruff checks, Python compilation, and `pip check` passed.
- The systemd user unit passed `systemd-analyze --user verify`; the service is enabled and running.
- `kgr` is installed at `~/.local/bin/kgr` and points to this project's isolated environment.
- The original `/home/dimpap/LocEstim` repository remains clean.

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

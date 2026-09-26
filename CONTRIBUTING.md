# Contributing

## Development

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests
.venv/bin/python -m compileall -q src
```

Tests use a Kaggle adapter with simulated network calls and local execution of generated launchers, and run the SSH adapter with its commands and SFTP served locally. `KGR_TEST_SSH=user@host:port` with `KGR_TEST_SSH_KEY` (and `KGR_TEST_SSH_CONFIG_DIR` holding a `known_hosts` that trusts the machine) also runs a job on a real SSH machine and removes its work directory afterwards. They cover scheduling, restart recovery, submission ambiguity, batch transactions, idempotency, cursor pagination, packaging, downloads, and CLI behavior. Automated tests do not create remote resources.

### Adding a provider

The queue talks to providers only through the `Provider` protocol in `src/compute_runner/providers/__init__.py`, one instance per account; `connect()` chooses the adapter by `Account.provider`. An adapter:

| Method | Contract |
| --- | --- |
| `check(spec)` | Raise `ValueError` for a specification the provider cannot run, including its dataset references |
| `ensure_bundle(bundle)` | Reuse a content-addressed bundle the account already holds, or upload it; `None` while it is still processing |
| `resolve_dataset(ref)` | Pin a dataset the account can read; `None` if it cannot, or if the dataset or pinned version does not exist. Every call checks access |
| `fetch_dataset(ref, destination)` | Download a readable dataset as plain files, so it can be copied to another account |
| `stage(job, number)` | Build the attempt's launch package locally and return its deterministic remote reference. It must expose every input as `KGR_INPUT_<ALIAS>` |
| `submit(job)` | Launch the staged attempt; raise `RemoteError` with `definitive=True` only when nothing was launched |
| `status(ref)` | Return a state of `queued`, `running`, `cancelling`, `succeeded`, `failed`, or `cancelled` (or `None`), the raw provider state, and an error |
| `download(ref, sink)` | Give the run's log and files to an output sink, which decides what to fetch and where it goes |
| `url`, `cancel`, `active_runs`, `quota`, `logs`, `live_log` | Links, cancellation, capacity discovery, quota (GPU `available_seconds: None` means no time limit), and logs |

`providers/launch.py` builds the runtime configuration and launcher that every adapter ships; `providers/ssh.py` is the second adapter and a compact example. The reference returned by `stage` is saved before `submit` runs, so an interrupted submission is reconciled through `status` rather than launched twice. Workloads read the provider-neutral `KGR_*` runtime variables; `providers/downloads.py` verifies outputs exposed as signed HTTPS URLs.

The Kaggle adapter pins `kaggle==2.2.4` and `kagglesdk==0.1.37`. SDK transport retries are disabled. The worker determines whether a remote operation can be retried. The Kaggle client is imported when a remote operation is required.

Example workloads are under [examples](examples). The GPU smoke test requires `--gpu --internet`. Running examples on Kaggle creates private resources and uses the corresponding compute allocation.

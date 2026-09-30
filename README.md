# Compute Runner

Run Python scripts, notebooks and projects on remote CPUs and GPUs from one local queue. Jobs run on accounts you connect: Kaggle accounts and your own machines over SSH. Every command, response and results folder works the same on both, and a job that cannot start on one account can move to another.

- **One queue, many accounts.** Connect several Kaggle accounts and SSH machines; the worker starts jobs as capacity frees up and can fail over between accounts.
- **Nothing to babysit.** A background worker uploads code and data, launches runs, tracks them and downloads outputs into a numbered run folder next to your code.
- **Safe to repeat.** Submissions are snapshotted and idempotent, so an interrupted call never launches a job twice.
- **Resumable training.** Continue a stopped run from its verified checkpoint, on the same account or another.
- **Built for agents too.** `compute-runner agent` gives LLM agents compact JSON responses, and a bundled skill tells them how to use it.

## Requirements

- Python 3.12 or later, on Linux (`systemd --user` runs the background worker; a terminal works too)
- A Kaggle account, or an SSH machine with Linux and Python 3.9 or later (plus `nbconvert` and `ipykernel` to run notebooks)

## Installation

From the project directory:

```bash
python3.12 -m venv .venv          # any Python 3.12 or newer; plain python3 may be older
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e . --no-deps
source .venv/bin/activate
```

`compute-runner` is the command; `kgr` remains an alias for existing scripts. Python callers import `compute_runner`. The lock file includes `paramiko` for SSH machines; other installations add it with `pip install 'compute-runner[ssh]'`.

## Connect an account

```bash
compute-runner account add kaggle YOUR_KAGGLE_USERNAME --enter-key       # type your API key or token
compute-runner account add ssh lab --host 10.0.0.5 --login you --key ~/.ssh/id_ed25519 --trust-new-host
compute-runner doctor                                                    # check the setup
compute-runner service install                                           # start the background worker
```

Keys, tokens and passwords are kept in one private file, `~/.config/compute-runner/credentials.json`, which never leaves your machine. See [Accounts and failover](docs/accounts.md) and [SSH machines](docs/ssh.md).

## Quick start

```bash
compute-runner submit examples/hello.py --dry-run     # preview what would be uploaded
compute-runner submit examples/hello.py
compute-runner submit examples/project.yaml
compute-runner submit train.py --gpu --internet --name cifar10-resnet18 --param lr=0.01 --input data=./data
```

Each job's results go to a run folder fixed at submission, such as `examples/results/hello/001_2026-09-26_14-30-12/`. Follow a job with the ID that `submit` prints:

```bash
compute-runner list
compute-runner status JOB_ID
compute-runner logs JOB_ID --follow
compute-runner wait JOB_ID
compute-runner running gpu     # what holds each account's GPU slots, including runs started elsewhere
compute-runner gpus            # GPU slots and time left on each account, and in total
```

Submission writes to the local queue; the worker uploads and launches jobs. Add `--json` before a command for machine-readable output.

## Using it from an LLM agent

Any agent that can run shell commands can drive `compute-runner agent`, which prints one compact JSON object per command. The [bundled skill](skills/compute-runner/SKILL.md) is plain Markdown that tells the agent how to use it. The agent leaves packaging, uploads and downloads to the application. It asks before changing your code, and it never handles credentials. See [Agent interface](docs/agent.md).

## Documentation

| Guide | Contents |
| --- | --- |
| [Accounts and failover](docs/accounts.md) | Connecting accounts, the credentials file, failover between accounts, datasets across accounts |
| [SSH machines](docs/ssh.md) | Running on your own machines, and how they differ from Kaggle |
| [Workloads](docs/workloads.md) | Workload YAML and submit options, packaging and runtime, resumable training |
| [Results](docs/results.md) | Run folders, `job.json` and `runs.md` |
| [Agent interface](docs/agent.md) | JSON commands, request keys, pagination, change cursors, logs, the agent skill |
| [Python API](docs/python-api.md) | `Client` and its methods |
| [Operations](docs/operations.md) | Commands and settings, the worker, strict mode, failure handling, state files, upgrades |
| [Contributing](CONTRIBUTING.md) | Tests and adding a provider |

## Limitations

- Jobs start when capacity is available; start times, recurring schedules and dependency graphs are not implemented.
- Continuing from a checkpoint needs the stopped run's outputs to be downloaded first.
- On Kaggle, cancelling a running job needs the session ID the runner prints at startup; a job still queued on Kaggle is cancelled by deleting its launch notebook.
- Only aliased dataset inputs can be copied between accounts.
- SSH machines run jobs as your login, isolated only by a folder and virtual environment per run, and cannot block network access.
- Custom containers, automatic offline dependency installation, and HTTP or MCP servers are not included.
- Job completion does not automatically resume an LLM conversation.

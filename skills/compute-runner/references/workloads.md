# Workload files

Most workloads need no file: `agent submit` takes `--name`, `--entrypoint` or `--module`, `--param`, `--input ALIAS=VALUE`, `--requirements`, `--gpu`, `--internet`, `--timeout` and `--arg`. Use a YAML for several jobs in one batch or for fields without a flag (`exclude`, `output_patterns`, `env`, `results_dir`, `auto_download`). It is a new file in the user's project, so write it only after the user agrees.

YAML accepts one job mapping or a `jobs:` list. Source and input paths are relative to the YAML file (`~` is expanded); entrypoint and requirements paths are relative to the source folder.

```yaml
jobs:
  - name: cifar10-analysis
    source: /absolute/path/to/project
    module: experiments.analyze
    params: {seed: 42}
    internet: false
    timeout_seconds: 3600
  - name: cifar10-resnet18
    source: /absolute/path/to/train.py
    params: {lr: 0.01, batch-size: 128, amp: true}
    gpu: true
    internet: true
    timeout_seconds: 7200
    inputs:
      data: /absolute/path/to/local/data
      pretrained: "kaggle:owner/existing-dataset/3"
```

Submit with `compute-runner agent submit /path/batch.yaml --request-key experiment-v1`. For a directory, choose `module: package.module` or `entrypoint: path/train.py` (or a Python notebook path). A single `.py`/`.ipynb` source does not take either option. Batch size is 1–1000 jobs, committed together; individual runs then progress independently. Request keys are 1–128 letters, digits, dots, underscores, colons, slashes, or hyphens, starting with a letter or digit.

## Names, parameters and results

`name` is the experiment: its runs share the folder `RESULTS/NAME/`, and each run gets `NNN_YYYY-MM-DD_HH-MM-SS` when it is submitted. Give every workload a name that says what it does, such as `cifar10-resnet18`, and reuse the name for repeats and sweeps of the same experiment. A single script submitted without YAML is named after its file.

Put the settings that distinguish runs in `params`, not in `args`. Each parameter reaches the program as `--NAME VALUE` after `args` (`true` passes `--NAME` alone; `false` omits it), is available as JSON in `KGR_PARAMS_JSON`, and is recorded in the run's `job.json` and the experiment's `runs.md`. On the command line, `--param lr=0.01` adds or overrides one; values are read as YAML scalars. Names are letters, digits, `_` and `-`; secret-looking names and values are rejected.

`RESULTS` defaults to `results/` beside the workload's code: inside a source folder, or next to a single script. The user can change it for everyone with `compute-runner init --results-dir PATH`, or for one workload with `results_dir:` (relative to the YAML file). Do not set `results_dir` unless the user asked for a location. A results folder inside the source is never uploaded with it.

## Inputs

Every input has an alias, and the workload reads it from `os.environ["KGR_INPUT_<ALIAS>"]` (for `data`, `KGR_INPUT_DATA`), or all of them from `KGR_INPUTS_JSON`. The value is always a folder; an input that is a single file is inside it under its own name. Never hardcode provider paths such as `/kaggle/input` or a machine's folders. An input is one of:

- A local file or folder. It becomes a private dataset of the job's account; identical snapshots reuse uploads.
- `"kaggle:OWNER/SLUG[/VERSION]"`: an existing dataset, attached directly when the job's account can read it. If it cannot, and another connected account can, the queue copies it only when the user allowed copies (see the skill's Accounts section). If no connected account can find it, the job is blocked before it runs.
- `"ssh:/ABSOLUTE/PATH"`: a file or folder already on the job's SSH machine, used where it is without an upload. Submission records the machine (`ssh:NAME:/ABSOLUTE/PATH`); a job on another account names it that way and uses the data only as a copy, which needs the user's approval like a dataset copy.
- `"job:JOB_ID[/PATH]"`: files a finished job wrote to `KGR_OUTPUT_DIR`, after its downloads completed. Use this to chain runs, instead of writing a results path.

Quote references in YAML. The older `datasets: [OWNER/SLUG/VERSION]` list still attaches datasets without an alias; they cannot be copied to another account, so prefer aliased inputs.

## Other fields

Optional fields: `accelerator: NvidiaTeslaT4`, `env: {SEED: "42"}`, `requirements: requirements.txt` (requires internet), `exclude: [checkpoints/]`, `auto_download: true`, `output_patterns: ["outputs/*.json"]`. Leave GPU accelerator unspecified unless a particular accelerator is needed. `accelerator`, dataset references, and the timeout limit are provider-specific and are checked against the job's account at submission; on Kaggle the timeout is 1–43200 seconds. SSH machines take no accelerator ID, require `internet: true`, and run notebooks only when the machine has `nbconvert`. The account is chosen with `--account`, not in YAML.

The runner excludes credential filenames, virtual environments, caches, and Git metadata. The source folder respects its `.gitignore`, `.kgrignore`, and `exclude`; input folders respect only `.kgrignore`. It does not detect secrets embedded in ordinary code. `env` is persisted configuration, so use it for nonsecret values only. No local environment or virtual environment is automatically forwarded. Runtime output goes in `os.environ["KGR_OUTPUT_DIR"]`. On Kaggle, code can use its preinstalled libraries without internet; on an SSH machine, packages installed in its Python are available, and `requirements` installs into a virtual environment of the run.

The ordinary CLI and Python API retain full records for deeper diagnostics. Read the project's `docs/` guides (such as `docs/workloads.md`) only for details outside this reference.

# Workload configuration

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
| `accelerator` | Unset | Provider accelerator ID; Kaggle accepts NVIDIA IDs, SSH machines none. Setting this also enables GPU use |
| `internet` | `false` | Enable network access in the workload. SSH machines cannot block it and require `true` |
| `timeout_seconds` | `43200` | Requested session timeout; Kaggle accepts 1 to 43200 seconds |
| `datasets` | `[]` | Existing provider datasets without an alias; on Kaggle `owner/slug` or `owner/slug/version` |
| `inputs` | `{}` | Named inputs: a local file or directory (uploaded as a private dataset), a provider dataset such as `kaggle:owner/slug/3`, a path on an SSH machine such as `ssh:/data/set` (on the job's machine) or `ssh:lab:/data/set`, or a finished job's outputs as `job:JOB_ID/PATH` |
| `requirements` | Unset | Included requirements file to install with pip. Requires internet access |
| `exclude` | `[]` | Additional source exclusion patterns |
| `auto_download` | `true` | Download outputs after execution terminates |
| `output_patterns` | Unset | Glob filters for remote output paths. Unset selects all files |
| `results_dir` | Unset | Parent of the experiment folders for this workload; see [Results](results.md) |

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

Both submit commands accept `--name`, `--entrypoint`, `--module`, `--requirements`, `--gpu/--cpu`, `--internet/--no-internet`, `--accelerator`, `--timeout`, `--account`, and repeated `--arg`, `--param NAME=VALUE` and `--input ALIAS=VALUE` options, so a single job rarely needs a YAML file. `--input` takes a local path, relative to the current folder, or a reference such as `kaggle:owner/slug/3`, `ssh:/data/set` or `job:JOB_ID/PATH`, and adds to the workload's `inputs`. `--param` values are read as YAML scalars, so `lr=0.01` is a number and `amp=true` a flag, and they are merged into each job's `params`. Overrides apply to every job in the YAML file. `--cpu` clears a configured accelerator and cannot be combined with `--accelerator`.

## Packaging and runtime

Source selection respects the root `.gitignore`, `.kgrignore`, and `exclude` patterns. Input folders honor only their own `.kgrignore`, because data folders often `.gitignore` the very files they carry. Credential filenames, Git metadata, virtual environments, caches, and `node_modules` are excluded. Before saving a snapshot, the runner also rejects high-confidence private keys and service-token patterns without printing the detected value. This screening reduces accidental disclosure but cannot recognize every possible credential, so keep secrets outside source and input folders. Symlinks are rejected. Notebook outputs and execution counts are removed from the saved snapshot. A notebook entrypoint must be a valid Python notebook; other notebooks in the project, such as R notebooks, are cleaned when possible and otherwise copied unchanged.

On Kaggle, single files are embedded in a generated private notebook, and project directories and local inputs become private datasets; identical content reuses the same dataset, and managed datasets are immutable and retain source license metadata. On an SSH machine, they are uploaded once into its work directory. Either way the runtime verifies every file against its checksum before the workload starts.

Unversioned dataset references are resolved when the worker prepares the job. Supply a version to select a specific dataset revision.

Project code runs from the `project/` folder of the run's working directory (`/kaggle/working` on Kaggle). Write result files under `KGR_OUTPUT_DIR`, the working directory's `outputs/`; a file written there is saved locally as `RUN_FOLDER/outputs/NAME`. Downloads skip the runtime's copy of the snapshot files under `project/` and its `__pycache__` bytecode; new files the workload writes under `project/` are still collected, into `RUN_FOLDER/working/project/`. Every named input, whether uploaded, attached from a provider, copied, or taken from another job, is exposed through `KGR_INPUT_<UPPERCASE_ALIAS>` and the `KGR_INPUTS_JSON` mapping, so workloads never depend on provider paths. Datasets in the unaliased `datasets` list remain under `/kaggle/input`. Parameters are in `KGR_PARAMS_JSON`.

Output downloads and full log caches have no configured size limit. Before writing, the runner checks free space on the filesystem containing the state directory. Known download sizes are checked up front; unknown or compressed bodies and log streams are checked as chunks arrive. A write that cannot fit with 16 MiB of operational headroom is stopped without replacing an existing file. The worker emits one warning when that filesystem falls below 10% free space and can warn again after space recovers and crosses the threshold later.

Downloads use the environment's proxy and certificate settings by default. [Strict mode](operations.md#strict-mode) ignores them and fetches only public HTTPS addresses.

The workload uses the provider's Python: Kaggle's environment, or the machine's `python3` (with a virtual environment of the run when it has requirements). A configured requirements file is installed before execution. Local virtual environments and process environment variables are not forwarded. `env` is only for nonsecret configuration: secret-like variable names and recognizable credential values are rejected because these values are stored with the job and shipped with the workload.

## Optional resumable training

Resumability is a workload-code decision, not a runner default. When an agent is preparing a stateful workload and the user's choice is unclear, the bundled skill tells it to ask whether the job should be resumable. If the answer is yes and no cadence was given, it then asks whether to checkpoint by elapsed minutes or completed epochs and for the interval. An explicit non-resumable choice is respected.

Resumable scripts should expose `--resume auto|required|never|PATH`, `--checkpoint-mode minutes|epochs`, and `--checkpoint-every NUMBER`, and write checkpoints below `KGR_OUTPUT_DIR/checkpoints`. Continue a stopped run once its outputs are downloaded:

```bash
compute-runner agent continue JOB_ID --request-key continue-1                          # same account
compute-runner agent continue JOB_ID --request-key continue-1 --account kaggle:bob     # another account
```

`continue` verifies that `checkpoints/latest.json` names a checkpoint with a matching SHA-256, then queues the job's saved code and inputs as the next run of the same experiment. It attaches only `latest.json` and that checkpoint as the input `resume` (`KGR_INPUT_RESUME`) and passes `--resume required`, replacing another `--resume` value or the `resume` parameter, so a missing or invalid checkpoint cannot silently restart training. To continue with changed code, submit a new workload with `inputs: {resume: "job:JOB_ID/checkpoints"}` and `--resume required`.

The skill includes a framework-neutral helper at `skills/compute-runner/assets/checkpointing.py`. It provides cadence checks, atomic numbered files, a checksummed `latest.json`, compatibility validation, and resume discovery. Training code must still serialize and restore its framework-specific model, optimizer, scheduler, scaler, progress, RNG, and data-loader state. See `skills/compute-runner/references/resumability.md` for the complete agent and migration contract.

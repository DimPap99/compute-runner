# Workload files

YAML accepts one job mapping or a `jobs:` list. Source and input paths are relative to the YAML file (`~` is expanded); entrypoint and requirements paths are relative to the source folder.

```yaml
jobs:
  - name: cpu-analysis
    source: /absolute/path/to/project
    module: experiments.analyze
    args: ["--seed", "42"]
    internet: false
    timeout_seconds: 3600
  - name: gpu-training
    source: /absolute/path/to/train.py
    gpu: true
    internet: true
    timeout_seconds: 7200
    inputs:
      data: /absolute/path/to/local/data
    datasets: [owner/existing-dataset/3]
```

Submit with `kgr agent submit /path/batch.yaml --request-key experiment-v1`. For a directory, choose `module: package.module` or `entrypoint: path/train.py` (or a Python notebook path). A single `.py`/`.ipynb` source does not take either option. Batch size is 1–1000 jobs, committed together; individual runs then progress independently. Request keys are 1–128 letters, digits, dots, underscores, colons, slashes, or hyphens, starting with a letter or digit.

Optional fields: `accelerator: NvidiaTeslaT4`, `env: {SEED: "42"}`, `requirements: requirements.txt` (requires internet), `exclude: [outputs/, checkpoints/]`, `auto_download: true`, `output_patterns: ["outputs/*.json"]`. Timeout is 1–43200 seconds. Leave GPU accelerator unspecified unless a particular accelerator is needed.

The runner excludes credential filenames, virtual environments, caches, and Git metadata. The source folder respects its `.gitignore`, `.kgrignore`, and `exclude`; input folders respect only `.kgrignore`. It does not detect secrets embedded in ordinary code. `env` is persisted configuration, so use it for nonsecret values only. No local environment or virtual environment is automatically forwarded.

Folders and local inputs become private datasets. Prefer existing Kaggle dataset references for large data already uploaded. Identical snapshots reuse uploads. Runtime output goes in `os.environ["KGR_OUTPUT_DIR"]`; named inputs are in `os.environ["KGR_INPUT_DATA"]` for the alias `data`. CPU code can also use Kaggle's preinstalled libraries without internet.

The ordinary CLI and Python API retain full records for deeper diagnostics. Read `~/kaggle-runner/README.md` only for details outside this reference.

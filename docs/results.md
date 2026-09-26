# Results

Each job's run folder is fixed and created, still empty, when it is submitted, before anything runs, and returned by every status response as `run_dir`:

```text
RESULTS/NAME/
  runs.md
  001_2026-09-26_14-30-12/
    job.json
    run.log
    outputs/        files the workload wrote to KGR_OUTPUT_DIR
    working/        other files it left in its working directory
  002_2026-09-26_15-02-47/
```

- `RESULTS` is the workload's `results_dir`, else the folder set with `compute-runner init --results-dir PATH` (`--results-dir ""` restores the default), else `results/` beside the workload's code: inside a source folder, or next to a single script. When it lies inside the source folder, it is left out of source snapshots.
- `NAME` is the job's name with characters other than letters, digits, `.`, `_` and `-` replaced by `-`.
- The run number counts up within the folder. The queue hands it out when the batch commits, so concurrent submitters never share one, and it continues after folders already on disk. The timestamp is the submission time in local time.
- `job.json` records the job ID, parameters, command, account, run URL, state, attempts, inputs and download state. `runs.md` lists every run of the experiment with its state, parameters, download state, parent run (for retries and continuations) and job ID.

Only the queue writes there. A hidden `.lock` file in each experiment folder keeps run numbers unique, also between queues of different state directories. The worker downloads outputs into the folder (as does `compute-runner download`) and rewrites `job.json` and `runs.md` from the queue whenever a job changes, following the queue's change cursor, so a restart resumes where it stopped and edits to those two files are overwritten. Retries and continuations are the next runs of the same experiment. Jobs saved before run folders keep `STATE_DIR/results/JOB_ID/`, with `outputs/outputs/NAME` and `provenance.json`.

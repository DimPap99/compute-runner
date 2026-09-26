# Resumable workloads

Resumability is opt-in. This reference applies after the user chooses it, or when an existing resumable job must be inspected, resumed, or migrated.

## Resolve the user's choice

Before editing or submitting a stateful workload, determine these independently:

1. Whether the workload should be resumable.
2. If yes, whether checkpoint cadence is based on elapsed minutes or completed epochs.
3. The positive cadence interval.

If item 1 is unspecified, ask: `Should this job be resumable?`

If resumability is selected and item 2 or 3 is unspecified, ask: `Should it checkpoint by elapsed minutes or completed epochs, and at what interval?`

Do not infer a default cadence. A user can explicitly choose non-resumable. For a stateless workload with no useful intermediate state, resumability is not applicable and no question is needed.

## Script contract

Prefer this interface unless the workload's framework has an established equivalent:

```text
--resume auto|required|never|PATH
--checkpoint-mode minutes|epochs
--checkpoint-every NUMBER
```

- `auto` loads a supplied checkpoint and otherwise starts fresh.
- `required` fails unless a valid, compatible checkpoint is available. Use it for an intended continuation or account migration.
- `never` deliberately starts fresh.
- `PATH` loads that exact checkpoint or manifest.
- `minutes` checkpoints at safe training boundaries after the requested elapsed interval.
- `epochs` checkpoints every requested number of completed epochs. Its interval must be an integer.

Use `KGR_OUTPUT_DIR/checkpoints` for new checkpoints. A named workload input called `resume` is exposed as `KGR_INPUT_RESUME`. Do not put checkpoints in the source directory or rely on a process-exit hook: cancellation can occur between hooks, so periodic checkpoints are the recovery mechanism.

For raw training loops, copy and adapt [../assets/checkpointing.py](../assets/checkpointing.py). It provides cadence calculation, atomic numbered checkpoint files, a checksummed `latest.json`, compatibility checks, and `auto`/`required`/`never`/explicit-path discovery. It deliberately accepts serializer callbacks so the training code remains responsible for framework state.

Use native checkpoint facilities instead when they preserve the required state correctly, such as Hugging Face Trainer or Lightning checkpoints. Keep the same command-line semantics and Kaggle input/output locations where practical.

A raw PyTorch integration should have this shape; adapt the state fields to the actual loop:

```python
from checkpointing import (
    CheckpointManager,
    add_checkpoint_arguments,
    cadence_from_args,
    capture_torch_rng_state,
    restore_torch_rng_state,
)

add_checkpoint_arguments(parser)
args = parser.parse_args()

compatibility = {
    "model": config.model_name,
    "optimizer": config.optimizer_name,
    "dataset": config.dataset_revision,
}
manager = CheckpointManager(
    Path(os.environ["KGR_OUTPUT_DIR"]) / "checkpoints",
    compatibility=compatibility,
)
cadence = cadence_from_args(args)
restored = manager.restore(
    args.resume,
    load=lambda path: torch.load(path, map_location="cpu", weights_only=False),
)
if restored:
    saved = restored.state
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    scaler.load_state_dict(saved["scaler"])
    restore_torch_rng_state(saved["rng"])
    start_epoch = saved["completed_epochs"]
    global_step = saved["global_step"]

# Call at the safe boundaries selected by the training loop.
state = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "scaler": scaler.state_dict(),
    "completed_epochs": completed_epochs,
    "global_step": global_step,
    "rng": capture_torch_rng_state(),
}
manager.save(
    state,
    dump=torch.save,
    global_step=global_step,
    completed_epochs=completed_epochs,
)
cadence.mark_saved()
```

The agent must integrate `cadence.due()` into the real loop: check it at safe batch boundaries for `minutes`, or pass `completed_epochs` at epoch boundaries for `epochs`. Optional scheduler or scaler objects should be handled explicitly rather than assumed to exist. Compatibility metadata and saved state must not contain credentials.

## State that must be preserved

For a true continuation, save and restore all state that affects the next update:

- model parameters;
- optimizer and learning-rate scheduler;
- mixed-precision gradient scaler, if used;
- completed epoch, global step, and best-metric or early-stopping state;
- Python, NumPy, framework CPU, and accelerator RNG state;
- data sampler position or batch-within-epoch for mid-epoch resume;
- model/data/tokenizer identity and a compatibility subset of the training configuration.

The total target epoch count and checkpoint cadence usually do not belong in the compatibility subset because they may intentionally change. Architecture, optimizer type, preprocessing identity, and dataset/split identity usually do.

Only the main process should write a checkpoint in distributed training. Synchronize workers around saving and restoring as required by the framework.

Minute-based checkpointing must occur at a safe boundary. If the workload cannot restore a data-loader position, either save only at epoch boundaries or explain that resuming replays part of the current epoch. Never describe that as exact mid-epoch continuation.

Emit concise machine-readable lines after successful operations:

```text
CHECKPOINT_SAVED step=12500 path=/kaggle/working/outputs/checkpoints/checkpoint-step-000000012500.pt
RESUMED_FROM step=12500 path=/kaggle/input/.../checkpoint-step-000000012500.pt
```

If resume input is present but corrupt or incompatible, fail loudly instead of silently starting over.

## Workload configuration

The first run has no `resume` input:

```yaml
name: model-training
source: ./project
entrypoint: train.py
args:
  - --resume
  - auto
  - --checkpoint-mode
  - minutes
  - --checkpoint-every
  - "10"
gpu: true
auto_download: true
output_patterns:
  - "outputs/checkpoints/*"
  - "outputs/metrics/*"
```

After the previous job's downloads complete, attach its checkpoint directory to the replacement job:

```yaml
name: model-training-resumed
source: ./project
entrypoint: train.py
args:
  - --resume
  - required
  - --checkpoint-mode
  - minutes
  - --checkpoint-every
  - "10"
gpu: true
inputs:
  resume: /absolute/path/to/results/JOB_ID/outputs/outputs/checkpoints
auto_download: true
output_patterns:
  - "outputs/checkpoints/*"
  - "outputs/metrics/*"
```

Use a new request key for the replacement submission. Reusing the original key replays the original batch rather than creating the intended continuation.

## Migration checks

Before submitting the continuation:

1. Stop the old job only when the user asked to stop it.
2. Wait until partial output downloads settle and `outputs_ready` is true.
3. Locate `latest.json`, verify its SHA-256 target, and load the checkpoint without starting training.
4. Confirm that saved progress and compatibility metadata match the intended workload.
5. Submit the continuation with `--resume required`, the downloaded checkpoint directory as input alias `resume`, and a new request key. To continue on another account, for example when the first account's GPU quota is exhausted, add `--account ID` after the user chooses it (see `compute-runner agent accounts`).
6. Inspect the initial log for `RESUMED_FROM` at the expected epoch or step.

If there is no valid checkpoint, tell the user what progress is recoverable. Do not label a restart from initial weights as a resume.

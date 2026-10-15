# Concurrent training queue

The Linux queue starts from one terminal and runs up to **six trainings at once**.
Each training gets one GPU, selected when it starts. One queue-managed training
can occupy each GPU. Completed slots are filled from the same job list, so a slow
image run does not hold up unrelated pose experiments.

The manager runs in the background without `nohup` or `tmux`. Closing a log
viewer or its terminal does not stop the queue. The Jupyter server/container
must remain running; this is not a system scheduler or a reservation service.

## Start and inspect

Copy and edit the example into an ignored local file. It contains two sample
jobs, not six mandatory jobs; add as many entries as needed.

```bash
cp docs/training_queue.example.json launchers/queue.local.json
bash launchers/QUEUE.sh start launchers/queue.local.json --dry-run
bash launchers/QUEUE.sh start launchers/queue.local.json
```

`--dry-run` validates the plan and paths and prints the resolved argument lists.
It does not inspect checkpoint contents, import PyTorch, contact GPUs, or start
training. The actual trainer validates checkpoint compatibility when launched.
The real start command also checks PyTorch-visible GPUs and NVIDIA telemetry.
The Python interpreter selected by the launcher is used for every training.

Watch all job states, assigned physical GPU indices, process IDs, and errors:

```bash
bash launchers/QUEUE.sh status --watch
```

Follow one training's normal tqdm terminal, from this terminal or a new one:

```bash
bash launchers/QUEUE.sh logs attention-small
```

**Ctrl+C detaches the viewer; it does not interrupt training.** Run `logs` with
another job ID to switch. Output is captured separately for each process, so
progress bars from different models are never mixed. A viewer opened before a
job starts waits for its output. A finished job's saved output remains readable.
The captured terminal uses 120 columns; use a terminal at least that wide for
the cleanest display. Raw ANSI and carriage-return bytes preserve tqdm updates;
these files are terminal recordings, not line-oriented text logs.

Commands default to the latest queue. Choose an older queue explicitly with
`status QUEUE_NAME`, or `logs JOB_ID --name QUEUE_NAME`.
The queue name is printed at start; `start --name NAME` supplies a unique name.

## Plan format

The example defines `version: 1`, `max_parallel: 6`, `poll_seconds: 10`,
`max_gpu_utilization: 100`, shared `defaults`, and a `jobs` list. Each job has:

- `id`: unique short name using letters, digits, dots, underscores or hyphens.
- `train`: training options using underscores, such as `attention_mlp_dropout`.
  These override `defaults`. Boolean options use JSON `true` or `false`.
- `min_free_mib`: minimum currently free GPU memory required before launch.

All paths inside training options are relative to the **project root**, not the
plan's location. Fresh outputs go to `trainings/QUEUE_NAME-JOB_ID`. The queue
supplies `run_dir`, `device=cuda`, and `progress=true`; these cannot be overridden
inside `train`. All other model settings retain the normal training CLI behavior.
Shell command strings are not accepted; arguments are passed directly to Python.

Use `start --max-parallel 3` to override the plan's concurrency limit for a batch.
Use `--trainings-dir PATH` if outputs belong outside the default directory.
The launcher honors the configured Python, diagnostics and trainings paths.

Memory thresholds are **launch guards, not measured peak requirements**. Defaults
are 4,096 MiB for pose models, 8,192 MiB for ResNet, and 16,384 MiB for DINOv3 or a
checkpoint whose model is not explicitly specified. Increase `min_free_mib` for
larger batches or backbone/input-layer training as needed. The example requests
16,384 MiB for ResNet input-head training. A job whose threshold exceeds every
accessible GPU's total capacity is rejected at startup.

To continue an existing run, use a job such as:

```json
{
  "id": "continue-dino",
  "min_free_mib": 16384,
  "train": {
    "resume": "trainings/PREVIOUS_RUN/checkpoints/last.pt",
    "epochs": 30
  }
}
```

Resume retains the existing run directory and checkpoint semantics. Keep shared
defaults compatible with that checkpoint; omit model-specific defaults when
mixing models. For a new training stage, use `init_from` instead of `resume` and
supply the desired training options. Checkpoint paths must already exist when
the plan is submitted. Do not queue a checkpoint that another process is still
resuming/writing. Duplicate output directories within one plan are rejected.

## GPU selection

The manager discovers the GPUs actually visible to PyTorch, respecting inherited
CUDA masks. Before each launch it reads current free memory and utilization from
`nvidia-smi`, excludes GPUs already used by its active jobs, and requires the
job's memory threshold. Among eligible GPUs it prefers the lowest utilization,
then the most free memory. The child receives the selected GPU's UUID through
`CUDA_VISIBLE_DEVICES`; inside that training it appears as `cuda:0`.

`max_gpu_utilization` can impose an upper limit. Its default of 100 allows a busy
GPU if memory is sufficient; this avoids waiting forever on a shared machine.
If no GPU qualifies, the job stays pending and the manager checks again later.
If telemetry fails, new launches wait. A smaller eligible job can pass one
waiting for more memory, while preserving list order among eligible jobs.

This does not reserve memory or prevent other users from starting jobs. Selection
is a snapshot, so another process can consume memory afterward. Running jobs
remain on their assigned GPU; no live migration occurs. A failed run, including
an out-of-memory error, is recorded and the queue continues with other jobs.
There are no automatic retries or silent checkpoint restarts.

Full GPUs are supported; MIG instances are currently rejected. An existing
training started outside this manager is reflected in GPU telemetry but does
not count toward the queue's six-process limit.

## Pause and stop

```bash
bash launchers/QUEUE.sh pause
bash launchers/QUEUE.sh resume
```

Pause prevents new launches while existing trainings continue. Resume unpauses
the live manager; it does not load a training checkpoint.

```bash
bash launchers/QUEUE.sh stop
```

Stop cancels pending jobs and lets active trainings finish. To also terminate
the queue's active processes, explicitly use `stop --terminate`. Only processes
started by this manager are signalled. Normal training checkpoints remain where
the trainer saved them; termination does not promise a checkpoint for the
partially completed epoch.

Only one manager may be active per project checkout. Plans are frozen at start;
editing the source JSON does not change an existing queue. Finished/stopped
queues cannot be restarted. Create a new plan with fresh runs or explicit
checkpoint resumes. After an unexpected manager/container failure, inspect
remaining processes before starting another queue; offline manager states are
marked potentially stale rather than treated as proof that training stopped.
Startup refuses to overlap a previous queue's recorded, still-live training
processes, verifying both process ID and Linux boot/start identity.

Queue plans, state, manager messages and per-job terminal recordings are saved
under `diagnostics/queues/QUEUE_NAME`. Normal training metrics and checkpoints
remain under their run directories, and the usual metrics export still works.
No extra packages are required beyond the existing runtime and `nvidia-smi`.

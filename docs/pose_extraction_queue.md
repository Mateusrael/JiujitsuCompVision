# Full-dataset pose extraction queue

The Linux extraction queue processes every image listed in `Data/annotations.json`
with both **YOLO26x-pose** and **RT-DETR + ViTPose++ Huge**. It uses one shared
pool of pending chunks and up to six GPU workers. Each worker keeps its model
loaded between chunks, so faster GPUs claim more work as they become available.

The default chunk size is **256 images**: roughly seven seconds of YOLO26 or a
minute of ViTPose at the measured extraction rates, before I/O overhead. This
keeps the final uneven workload small without repeatedly loading model weights.

## Start

Install the optional detector dependencies in the existing project environment:

```bash
.venv/bin/python -m pip install -r install/requirements-pose-detectors.txt
```

The queue respects the GPUs visible to PyTorch. If a previous command set
`CUDA_VISIBLE_DEVICES` to a single GPU, remove that shell override before starting
a queue that should use all GPUs exposed by the server:

```bash
unset CUDA_VISIBLE_DEVICES
bash launchers/EXTRACT_POSES.sh start --dry-run
bash launchers/EXTRACT_POSES.sh start
```

`--dry-run` validates and summarizes the dataset without probing GPUs, starting
models, or creating output files. By default the queue writes to
`Data/pose_detections/full_dataset`. Start refuses an existing output directory;
use `resume` to continue it or choose a different `--output-dir` for a new extraction.

GPUs with sufficient memory receive ViTPose work first. GPUs that cannot fit
ViTPose can receive YOLO26 work instead. Defaults require **8,192 MiB free for
ViTPose** or **2,048 MiB for YOLO26** before loading a model. These are scheduling
thresholds, not reservations or guarantees; other jobs may change GPU memory
availability. No GPU number is permanently restricted to either model.

To customize the configuration for a new queue:

```bash
bash launchers/EXTRACT_POSES.sh start \
  --output-dir Data/pose_detections/full_dataset_custom \
  --chunk-size 256 --max-parallel 6 \
  --vitpose-min-free-mib 8192 --yolo-min-free-mib 2048
```

`--annotations` and `--images-dir` select a different copy of the original dataset.
The immutable plan records the annotation checksum, image IDs, detector settings,
chunk boundaries and memory thresholds. The GPU list is discovered again on resume.

## Progress and logs

The manager runs in the background. Closing the launching terminal or a log viewer
does not stop extraction; shutting down the Jupyter server/container does.

```bash
bash launchers/EXTRACT_POSES.sh status --watch
```

Status shows completed image/chunk counts per detector and each worker's detector,
GPU, PID and current chunk. Image totals advance when a chunk finishes; the worker
log shows per-image tqdm progress within a chunk.

```bash
bash launchers/EXTRACT_POSES.sh logs
bash launchers/EXTRACT_POSES.sh logs WORKER_ID
```

The first command follows the manager log; replace `WORKER_ID` in the second with
an ID shown by `status`. Ctrl+C detaches either viewer. Use a terminal at least 120
columns wide for the captured worker progress bars. For a custom output directory,
pass the same `--output-dir` to each command.

## Stop and resume

```bash
bash launchers/EXTRACT_POSES.sh stop
bash launchers/EXTRACT_POSES.sh status --watch
```

Stop finishes active chunks and leaves pending work available. After the manager
has stopped, or after restarting a container, continue with:

```bash
bash launchers/EXTRACT_POSES.sh resume
```

Resume checks for surviving workers before assigning work. Complete per-image rows
in interrupted chunks are verified and reused. Each chunk has a single writer;
workers never append concurrently to the same cache. A changed annotation file,
detector configuration, weights or cached image identity is rejected.

## Outputs and training

Successful extraction creates:

```text
Data/pose_detections/full_dataset/
  yolo.jsonl
  yolo.meta.json
  vitpose.jsonl
  vitpose.meta.json
  plan.json
  state.json
  manager.log
  logs/
  chunks/
```

Final caches are published only after their chunks pass completeness and identity
checks. They contain one record per annotated image, including zero- and one-pose
results. Each record contains pixel coordinates, confidence, image checksum and
extraction time. Metadata records the detector configuration, package versions
and weight identity. Keep the chunk files until extraction is complete so it can
resume without repeating inference.

Extraction includes training, validation, test, excluded/buffer frames, and images
with missing supplied athlete poses. It does **not** change class labels, split
assignments, or training eligibility. Full caches can be reused for different
classifiers without rerunning the detectors.

For a new pose training, use either completed cache:

```bash
bash launchers/START_TRAINING.sh --model pose-wide \
  --pose-detections Data/pose_detections/full_dataset/yolo.jsonl
```

Use `vitpose.jsonl` for ViTPose inputs. The current `--pose-detections` option changes
**training inputs only**; epoch validation and final test still use supplied poses.
Use the [pose-source comparison](external_evaluation.md) to evaluate saved
classifiers with detected poses as well. Training still follows the selected
split and its existing supplied-pose eligibility rules; extracting every image does
not add buffer frames to training. Do not train from an unfinished or changing cache:
training and checkpoint resume verify its SHA-256.

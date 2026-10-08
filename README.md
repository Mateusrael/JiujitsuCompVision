# Jiu-jitsu position classification

Compare three classifiers using supplied athlete poses with an image classifier.
The target environment is Linux with Python 3.11, PyTorch 2.11.0,
torchvision 0.26.0 and CUDA 12.8. Setup creates an isolated project virtual
environment. Training uses one GPU per process.

Code is organized into thin launchers and scripts, reusable feature modules,
and separate per-experiment outputs.

```text
install/        environment probe and Linux dependency setup
launchers/      editable paths and commands to run common operations
src/
  Dataset/      annotation validation, audit, download, shared evaluation splits
  Loader/       paired-pose features and image loading
  Modules/      two pose MLPs, joint self-attention and ResNet-18
  Train/        training, checkpointing, experiment metadata
  Eval/         classification metrics and evaluation
  Scripts/      command-line entry points and upload bundle creation
tests/          small checks using synthetic data
docs/           recording-map template
Data/           remote dataset and split manifests (ignored by Git)
trainings/      per-run config, checkpoints and diagnostics (ignored)
diagnostics/    environment and dataset reports (ignored)
dist/           upload ZIPs generated locally (ignored)
```

## First deployment to JupyterLab

Open a **Terminal** from JupyterLab's Launcher and clone into the persistent
folder. Run each command separately. Skip cloning if the checkout already exists:

```bash
cd /home/jovyan/privado
git clone --branch master https://github.com/Mateusrael/JiujitsuCompVision.git
cd JiujitsuCompVision
bash bootstrap.sh --check-only
```

For an existing checkout, finish active runs, inspect `git status`, and update
with `git pull --ff-only`. As a fallback, upload `dist/VisaoComp-source.zip` to a
fresh `privado/JiujitsuCompVision` folder and extract it there with
`python3 -m zipfile -e VisaoComp-source.zip .`. The ZIP has a flat project root
and contains neither Git history nor datasets or runs. Review local source edits
before replacing an older upload. A terminal may start outside the folder shown
in JupyterLab's file browser; use `cd` explicitly.

`--check-only` saves `diagnostics/environment.json`, checks Git and optional
outbound HTTP connectivity, and inspects Python/PyTorch/GPU availability. It does
not install packages, download the dataset or train. It can run without PyTorch.
You can also upload only `install/check_environment.py` and run that file first.

For environment choices or debugging, start with this report. A successful
GitHub HEAD check shows HTTP connectivity only, not access to a private repository.
GPU visibility is not confirmation of a long-running allocation; follow the
lab's session and job rules.

## Setup on the DGX

Python 3.10 or newer and Bash are required; Python 3.11 is the target version.
Run these commands in the project directory on the DGX:

```bash
bash bootstrap.sh
source .venv/bin/activate
bash launchers/CHECK_ENVIRONMENT.sh --require-torch --require-cuda
```

The default setup creates `.venv` without system-site packages and installs
`torch==2.11.0` and `torchvision==0.26.0` from the official CUDA 12.8 wheel index,
plus the pinned `tqdm` progress dependency from PyPI.
It checks package versions, their imports and the PyTorch CUDA runtime build.
No system packages, NVIDIA drivers or system CUDA toolkit are installed.
The corresponding explicit installer command is:

```bash
bash install/install.sh --torch cu128
```

After updating an existing DGX checkout, install the progress dependency without
reinstalling PyTorch:

```bash
.venv/bin/python -m pip install -r install/requirements-runtime.txt
```

The pinned package pair and CUDA 12.8 wheels are listed in
[PyTorch's versioned installation instructions](https://pytorch.org/get-started/previous-versions/#v2-11-0).
The target DGX driver, 570.195.03, supports this CUDA build. Optional `cpu` and
`cu126` modes use the same package versions; `--torch existing` is an explicit
opt-in to inherit lab packages instead of installing this pinned environment.

If `.venv` already exists from a different setup mode, it is preserved and setup
stops. Create a separate environment with
`bash install/install.sh --torch cu128 --venv .venv-cu128`, then point the launcher
configuration's `PYTHON` at its absolute `.venv-cu128/bin/python` path. Use the
launchers directly for this custom environment; `bootstrap.sh` manages `.venv`.

Path defaults are relative to this repository. To use other persistent paths or
a custom Python environment, copy and edit the local configuration:

```bash
cp launchers/local_config.example.sh launchers/local_config.sh
```

Set `DATA_DIR`, `DIAGNOSTICS_DIR`, `TRAININGS_DIR`, `SPLIT_FILE`, and `PYTHON` as
needed. `local_config.sh` is machine-specific, sourced by
the launchers, and excluded from Git and ZIPs. Caller environment variables take
precedence over this file, then versioned defaults fill missing values. Do not put
credentials in it. Launchers automatically select the project `.venv` unless
`PYTHON` is explicitly overridden.

## Obtain and inspect data remotely

Start with annotations, which serve all four models:

```bash
bash launchers/DOWNLOAD_DATA.sh
bash launchers/AUDIT_DATASET.sh
```

**Default bootstrap and the synthetic GPU smoke test do not download images.** The
default `DOWNLOAD_DATA.sh` command above downloads annotations only. Prepare the
images on the DGX with:

```bash
bash launchers/DOWNLOAD_IMAGES.sh
```

This launcher downloads the archive with progress reporting, extracts images,
and audits every annotated image path. Its completion message confirms all three
steps succeeded; the report is saved to `diagnostics/dataset_audit.json` (or your
configured diagnostics directory). It does not start training.
The observed image archive has root-level JPEG names such as `0101166.jpg`; the
downloader extracts these into `Data/images/`. It keeps the archive, resumes
interrupted HTTP downloads when the server supports it, checks extraction paths
and CRCs, and records local hashes/source URLs. A locally calculated hash is not
an independently published source checksum.

Observed server sizes on 2026-10-07 were 208,848,730 bytes for annotations and
10,043,058,772 bytes for the compressed image ZIP (about 9.35 GiB). Extracted size
is checked from the archive before extraction; do not treat compressed size as
the total storage requirement. Account for ZIP + images + environments + weights
+ checkpoints. Filesystem free space does not establish your storage quota.
The complete JSON array is loaded into RAM during audit and training.

Dataset source: [ViCoS Brazilian Jiu-Jitsu Positions Dataset](https://www2.vicos.si/resources/jiujitsu/).
It is licensed CC BY-NC-SA 4.0. Retain attribution to Valter Hudovernik and Danijel
Skočaj, *Video-Based Detection of Combat Positions and Automatic Scoring in
Jiu-jitsu*, MMSports 2022. Dataset files are downloaded separately and are not
redistributed in this repository or bundle.

## Create one split for all models

The default is a **single-view temporal holdout within known videos**. It compares
the models on later examples of the same classes in selected videos. It does not
measure generalization to new matches or athletes. The supplied audit shows that
some classes appear in only two video prefixes, so assigning complete videos to
three partitions cannot give every partition all ten classes. Camera grouping
and synchronization are not independently verified.

```bash
bash launchers/PREPARE_SPLITS.sh
```

This creates `Data/splits/temporal_split.json`. The source selection uses one
video prefix for each normalized class:

| Video prefix | Selected classes |
|---|---|
| `00` | open guard, closed guard |
| `03` | 50-50 guard, half guard |
| `06` | mount, side control |
| `09` | turtle |
| `11` | standing, takedown |
| `14` | back |

All other source views are excluded, as are incidental occurrences of a class
outside its selected prefix. For example, a standing annotation in prefix `00`
is excluded. All four classifiers use exactly the same retained examples.

Within each class, records are ordered by their original frame numbers and
provisionally divided into **70% train, 15% validation, and 15% test**. Around
every change of partition within each selected video, including changes between
different classes, the tool removes an embargo of 75 frames on either side of
the boundary midpoint. It then verifies that retained frames in different
partitions are **more than 150 frames apart**, approximately five seconds at the
dataset's roughly 30 fps. Ratios change after these exclusions. This interval
reduces neighboring-frame overlap; it does not establish event independence.

Validation on **2026-10-07** against all 120,279 official annotations produced:

| Partition | Retained examples |
|---|---:|
| Train | 31,093 |
| Validation | 5,235 |
| Test | 6,138 |

All ten classes are present in every partition; the smallest class/partition
combination is validation takedown with 106 examples. The minimum separation is
151 frames in every selected prefix. Another 77,813 examples are excluded:
74,675 by source selection and 3,138 by the temporal embargo. These counts apply
to annotation SHA-256
`b7633ed161372bef7d0dcdd2cb3bc69400147b6a2532a4a5b8070048a1dd62bc`.

The tool requires at least 20 retained examples of every class in each partition
and stops if coverage or separation fails. It never silently reduces the gap or
falls back to random frames. The manifest stores every image assignment,
exclusion reason, source selection, boundary, coverage count and annotation hash.
The pose and image loaders reconstruct and verify the manifest before using it. Existing
manifests are never overwritten; use a new output path and update `SPLIT_FILE`
when deliberately changing the protocol. See `--help` for explicit options.

For data with verified recording metadata and enough independent recordings per
class, the original recording-group method remains available explicitly:

```bash
bash launchers/PREPARE_SPLITS.sh --method recording-group --groups "$PWD/Data/recording_groups.json" --output "$PWD/Data/splits/recording_split.json"
```

Use `docs/recording_groups.example.json` as the metadata template and set
`SPLIT_FILE` to this separate manifest in your local launcher configuration.
That method keeps all views of a recording together and still requires every
class in every partition.

## Smoke tests and training

The default comparison has **10 identity-independent position categories**. The
18 raw labels are merged with an explicit mapping. This avoids requiring an RGB
classifier to infer an unobservable athlete numbering convention.

Run a synthetic forward/backward smoke check inside the approved allocation:

```bash
.venv/bin/python -m src.Scripts.smoke_test --device cuda
```

For a small CPU check when a GPU is not allocated, use `--device cpu` explicitly.
The default smoke check exercises all four models using synthetic inputs and
untrained image weights. Select one model with `--model pose`, `pose-wide`,
`pose-attention`, or `image`; `--model both` retains the original pose/image check.
Full image training uses pretrained ResNet-18 weights, requiring a download/cache
on first use.

To check optional whole-model compilation on the DGX as well:

```bash
.venv/bin/python -m src.Scripts.smoke_test --device cuda --compile
```

This exercises both training forward/backward and evaluation forward passes.
The initial passes can take longer while PyTorch compiles the model graphs.

### Model choices

Every model classifies one frame into the same ten position categories. The three
pose models receive the same 102 features: `(x, y, confidence)` for 17 joints of
each athlete. Both athletes are centered and scaled together, missing joints
remain zero, and confidence is clipped only during featurization. Athlete slots
are randomly swapped with probability 0.5 during training by default. None of
these models extracts poses from images or models changes across video frames.

| CLI model | Architecture | Parameters updated during training |
|---|---|---|
| `pose` | `102 → 102 → 34 → 10`, ReLU after both hidden layers | All parameters, initialized randomly |
| `pose-wide` | `102 → 512 → 128 → 32 → 10`, ReLU after all three hidden layers | All parameters, initialized randomly |
| `pose-attention` | 34 joint tokens, four attention/MLP blocks, pooled features → 10 | All parameters, initialized randomly |
| `image` | ImageNet-pretrained ResNet-18 with a replacement `512 → 10` classifier | New classifier only; `--fine-tune` also updates the backbone |

Horizontal mirroring and dropout are optional for all four models and are
**disabled by default**:

| Option | Default | Meaning |
|---|---:|---|
| `--horizontal-flip-prob` | `0.0` | Probability of mirroring a training example; `0.5` enables a 50% chance |
| `--dropout` | `0.0` | Dropout probability during training; valid range [0, 1) |

Image mirroring reverses the full frame horizontally before resize-and-pad.
Pose mirroring negates normalized x and exchanges the COCO left/right joint
slots within each athlete; y and confidence move with their joint unchanged.
The mirror keeps athlete slots in place and is independent of the existing
athlete-swap augmentation. No vertical flip is applied. Validation and test
examples are never randomly mirrored or athlete-swapped.

For the MLPs, dropout follows each hidden ReLU. For the image model, it acts on
the 512 pooled features before the classifier, including when the backbone is
frozen. For attention, it acts inside the attention/MLP blocks as described below.
Dropout is disabled during validation and test evaluation; logits are never
passed through a dropout layer.

The attention model reshapes the common pose input into 34 joints. A shared
`3 → 128` projection embeds each joint, then learned slot embeddings identify
the joint and athlete slot. Each of its four blocks applies LayerNorm, four-head
self-attention and a residual addition, followed by LayerNorm, a
`128 → 512 → 128` MLP with GELU and another residual addition. Final LayerNorm
and a mean over observed joints produce 128 features for the `128 → 10` classifier.
Attention connects joints from both athletes within a single frame. Joints with
nonpositive confidence are masked from attention keys and mean pooling. If all
joints are missing, mean pooling returns zero and the classifier returns its bias.

These defaults can be configured at the start of an attention run:

| Option | Default | Meaning |
|---|---:|---|
| `--attention-dim` | `128` | Joint embedding width; must be divisible by head count |
| `--attention-heads` | `4` | Attention heads per block |
| `--attention-layers` | `4` | Number of residual attention/MLP blocks |
| `--attention-mlp-dim` | `512` | Hidden width of each block's MLP |
| `--attention-dropout` | `0.0` | Attention weights and attention residual-output dropout |
| `--attention-mlp-dropout` | `0.0` | Hidden and output dropout inside each block's MLP |
| `--attention-pooling` | `mean` | Mean over observed joints; `cls` uses a learned classification token |

The two attention dropout rates are independent and must be in [0, 1).
`--dropout` provides a shared fallback for any branch whose specific flag is
omitted. Each branch flag overrides that fallback, including an explicit zero.
For example, `--dropout 0.2 --attention-dropout 0` disables attention dropout
and keeps the block MLP dropout at 0.2. Without any dropout flags, both are zero.

`--attention-pooling cls` prepends a learned token, making 35 tokens in total,
and classifies its final representation instead of averaging the joints. All
architecture choices are saved in the checkpoint and cannot change on resume.

The image model preserves both athletes using resize-and-pad to 224 × 224 and
ImageNet normalization. Its original ImageNet `512 → 1000` layer is removed;
the new `512 → 10` classifier takes its place. The frozen backbone, including
BatchNorm statistics, produces the 512 features. Use `--fine-tune` for a separate
image experiment that updates the entire network, or `--no-pretrained` for an
explicitly untrained image baseline or smoke check. See
[the module documentation](src/Modules/README.md) for the architecture contracts.

The following are training examples to run after preparing the data and shared
split and choosing the model experiment. Setup and split preparation do not
start them automatically:

```bash
RUN_NAME=pose-baseline bash launchers/START_TRAINING.sh --model pose --epochs 30
```

```bash
RUN_NAME=pose-wide bash launchers/START_TRAINING.sh --model pose-wide --epochs 30
```

```bash
RUN_NAME=pose-attention bash launchers/START_TRAINING.sh --model pose-attention --epochs 30
```

```bash
RUN_NAME=image-baseline bash launchers/START_TRAINING.sh --model image --epochs 30
```

For an optional experiment with horizontal mirroring and dropout enabled:

```bash
RUN_NAME=pose-wide-mirror-dropout bash launchers/START_TRAINING.sh --model pose-wide --epochs 30 --horizontal-flip-prob 0.5 --dropout 0.2
```

Both flags also work with `pose`, `pose-attention`, and `image`. The examples
without these flags retain mirroring and dropout at zero.

For attention dropout of 0.1 and block MLP dropout of 0.3:

```bash
RUN_NAME=pose-attention-dropout bash launchers/START_TRAINING.sh --model pose-attention --epochs 30 --attention-dropout 0.1 --attention-mlp-dropout 0.3
```

Training shows an overall epoch progress bar and a batch bar for each training
and validation phase. Batch bars include percentage, batch count, elapsed time,
estimated remaining time, throughput, running loss and running accuracy. The
epoch bar and printed epoch summary show training loss and accuracy, validation
loss and accuracy, and validation macro F1. All five metrics are also saved in
`diagnostics/metrics.jsonl`, including in existing runs. Training metrics are
averaged during updates with augmentation/dropout active; validation uses the
final epoch weights in evaluation mode. `--no-progress`
disables the bars, which is useful for redirected logs; `--progress` enables them
again. Startup data/model preparation and compilation can precede the first
completed batch.

Compilation is optional and **off by default** for all four models. Add
`--compile` to pass the **whole model** to `torch.compile` using PyTorch's
default Inductor backend; there are no block-level compile decorators:

```bash
RUN_NAME=pose-attention-compiled bash launchers/START_TRAINING.sh --model pose-attention --epochs 30 --compile
```

`--compile-mode` accepts `default` (the default), `reduce-overhead`,
`max-autotune`, or `max-autotune-no-cudagraphs`. A nondefault mode requires
compilation to be enabled. Initial training/validation passes and new input
shapes can trigger compilation, so early ETA estimates can be inflated. Speed
gains depend on the model and workload. See the
[PyTorch compile options](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html).
Compiler errors are surfaced; the project does not silently switch compilation off.

CUDA is required by default; unavailable CUDA fails clearly. Data-loading workers,
batch size, seed, learning rate and weight decay can be set with CLI flags; see `--help`. Run
directories are separate and must be new. Keep the seed and split fixed when
comparing models. Missing poses are represented explicitly instead of excluding
those samples from one side of the comparison.

AdamW defaults to `--lr 0.001 --weight-decay 0.0001`. Weight decay must be finite
and nonnegative; zero disables it. It applies to all trainable parameters,
including biases and normalization parameters. Both optimizer settings are saved
and must stay fixed when resuming. Start a new run to compare different settings.
For example, an attention experiment with lower learning rate, higher weight decay,
mirroring, attention dropout 0.1 and MLP dropout 0.15 is:

```bash
RUN_NAME=pose-attention-regularized bash launchers/START_TRAINING.sh --model pose-attention --epochs 30 --compile --lr 0.0003 --weight-decay 0.001 --horizontal-flip-prob 0.5 --attention-dropout 0.1 --attention-mlp-dropout 0.15
```

These are experiment settings, not established optimal values. Lower learning
rates may need more epochs to converge. Changing several settings together tests
the combination; separate runs are needed to isolate each setting's contribution.

Resume from the last checkpoint and extend the total epoch count:

```bash
bash launchers/CONTINUE_TRAINING.sh --resume trainings/pose-baseline/checkpoints/last.pt --epochs 60
```

Checkpoints include model configuration, optimizer and random states. Saved
configuration is reused on resume, including the mirror probability and dropout
rates; changing these requires a new run. Attention checkpoints record and
restore the two effective branch rates independently. Best checkpoints are selected on
validation data; test data is used by the explicit
evaluation command:

```bash
bash launchers/EVALUATE.sh --checkpoint trainings/pose-baseline/checkpoints/best.pt
```

Compilation and progress are execution settings: they can be overridden on resume
with `--compile` / `--no-compile`, `--compile-mode`, and
`--progress` / `--no-progress`. Omitted flags reuse the run's saved choices.
Checkpoints always save the original model's state, so they can be resumed or
evaluated with or without compilation. Changing execution mode can change floating
point results and random-number behavior; bitwise reproducibility is not promised
across compiled and uncompiled execution.

Evaluation has its own batch progress bar and uses uncompiled execution by default,
even for a checkpoint trained with compilation. Add `--compile` and optionally
`--compile-mode` to the evaluation command to compile the reconstructed whole model,
or use `--no-progress` to suppress its bar.

```bash
bash launchers/EVALUATE.sh --checkpoint trainings/pose-wide/checkpoints/best.pt
```

```bash
bash launchers/EVALUATE.sh --checkpoint trainings/pose-attention/checkpoints/best.pt
```

```bash
bash launchers/EVALUATE.sh --checkpoint trainings/image-baseline/checkpoints/best.pt
```

Results include accuracy, macro-F1, per-class precision/recall/support and a
confusion matrix. Supplied-pose model results assume pose annotations already exist;
they do not measure an image-to-pose pipeline or end-to-end inference speed.

## Git workflow and source bundles

Keep code and shared configuration in Git. Dataset files, local settings,
session handoff notes and generated results remain ignored and excluded from
upload bundles.

If Git is installed in the Jupyter terminal, the repository host is reachable and
you have repository access, clone through HTTPS into a new/empty folder under
`privado` and run the same setup commands.
For updates, finish active runs, check local changes, then use `git pull --ff-only`.
Do not embed access tokens in clone URLs, scripts or notebooks.

For now, create an upload bundle on this development PC with:

```powershell
python -B src/Scripts/bundle_project.py
```

Use `--output dist/another-name.zip` for another snapshot. Each ZIP includes a
`source_manifest.json` with source hashes and Git state, even before an initial
commit. The supplied training provenance records Git when available; retain this
bundle manifest alongside archived results when deploying by upload.

## Validation status

The user-reported DGX environment check confirmed PyTorch 2.11.0 + CUDA 12.8,
torchvision 0.26.0, and GPU access. On 2026-10-07 the user reported passing GPU
smoke tests for all four models: `pose`, `pose-wide`, `pose-attention`, and `image`.
These checks used uncompiled execution. Compiled CUDA execution still needs its
separate smoke check in the target environment.
The full remote annotation audit found 120,279 records, all ten normalized
classes, no malformed records, and no duplicate image IDs. That audit did not
check image files; the full image download and image audit are not yet confirmed.
No real-data training results have been reported.

The default temporal split also passed validation against the complete official
annotation content on 2026-10-07, with the same SHA-256 as the DGX audit and the
counts reported above. The 208,848,730 annotation bytes were read into memory;
the annotation file and image archive were not saved on this development PC.
Only the derived report was saved to `diagnostics/temporal_split_validation.json`.
The user also generated the shared DGX manifest successfully: 31,093 training,
5,235 validation, 6,138 test and 77,813 excluded examples, with a minimum retained
cross-partition separation of 151 frames.

Local synthetic checks cover temporal exclusions, cross-class boundary gaps,
shared assignments and manifest validation:

```bash
python -B -m unittest discover -s tests -v
```

Small CPU train/resume/evaluation integration tests are opt-in and use synthetic
data only, without pretrained downloads:

```bash
JIUJITSU_RUN_TORCH_TESTS=1 python -B -m unittest discover -s tests -v
```

These require an environment with PyTorch, torchvision and tqdm; other checks use
Python's standard library. On 2026-10-07, all 91 tests passed locally with no
skips on CPU using PyTorch 2.8 and torchvision 0.23. They covered training,
resume and evaluation for all four models with horizontal-mirror probability
0.5 and dropout 0.1 (attention weight/output dropout 0.1 and block MLP dropout
0.2), plus checkpoint reconstruction with nondefault attention settings.
Separate checks covered independent dropout branches, explicit-zero overrides,
and dropout being disabled during evaluation. Resumed CPU weights matched
uninterrupted training exactly for every model, including the frozen ResNet
classifier. The separate CPU
forward/backward smoke check also passed for all four models.
Progress-on/off checks preserve weights, metrics and RNG state exactly.
Whole-model compilation checks exercise real TorchDynamo graph capture with
the CPU `eager` backend for all four models, plus checkpoint portability and
runtime-option overrides. These do not validate native CUDA/Inductor compilation
or establish a speedup; use the compiled DGX smoke command for that execution path.
Synthetic execution checks do not measure model quality on the real dataset.

# Jiu-jitsu position classification

Compare three classifiers using supplied athlete poses with ResNet-18 and
DINOv3 ViT-B/16 image classifiers.
The target environment is Linux with Python 3.11, PyTorch 2.11.0,
torchvision 0.26.0 and CUDA 12.8. Setup creates an isolated project virtual
environment. Training uses one GPU per process.

Code is organized into thin launchers and scripts, reusable feature modules,
and separate per-experiment outputs.

For a batch of experiments, use the [concurrent training queue](docs/training_queue.md).
It runs up to six trainings at once from one terminal, selects a GPU at each
launch using current memory/utilization, and keeps a separate attachable tqdm
log for every job. The manager continues in the background when a viewer closes.

Use the [training figure generator](docs/plot_training_metrics.md) to reproduce
loss curves, validation metrics, and per-class comparisons from saved run data.
It exports PNG/SVG figures and the source data for editable charts.

Open the [model explorer notebook](notebooks/model_explorer.ipynb)
in JupyterLab to browse curves, inspect class metrics, and predict random
samples with image/pose overlays. See the [notebook setup](docs/model_explorer.md)
for the project kernel and controls.

```text
install/        environment probe and Linux dependency setup
launchers/      editable paths and commands to run common operations
src/
  Dataset/      annotation validation, audit, download, shared evaluation splits
  Loader/       paired-pose features and image loading
  Modules/      two pose MLPs, joint self-attention, ResNet-18 and DINOv3
  Train/        training, checkpointing, experiment metadata
  Eval/         classification metrics and evaluation
  Scripts/      command-line entry points and upload bundle creation
tests/          small checks using synthetic data
docs/           versioned multiview section plan and recording-map template
notebooks/      interactive model exploration notebooks
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
plus runtime dependencies from PyPI, including `tqdm` and `transformers==4.57.1`.
It checks package versions, their imports and the PyTorch CUDA runtime build.
No system packages, NVIDIA drivers or system CUDA toolkit are installed.
The corresponding explicit installer command is:

```bash
bash install/install.sh --torch cu128
```

After updating an existing DGX checkout, update runtime dependencies without
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

Start with annotations, which serve all five models:

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

The default **multiview section split** targets **80% training, 10% validation,
and 10% test** among retained annotated images. It uses all 16 sources, grouped
as `00–02`, `03–05`, `06–08`, `09–10`, `11–13`, and `14–15`. Contiguous evaluation
sections are distributed throughout the recordings instead of always holding
out the latest examples of each class. All five models share the section
boundaries and train/validation/test roles; pose models additionally require
both athletes' supplied poses in every partition. Image training retains
single-pose images. The split balances both eligible populations toward the
80/10/10 target, including the retained examples of each class separately, so
their exact sample counts differ. A class's training percentage is its retained
training count divided by its retained train + validation + test count; it is
not that class's share of the entire training set.

```bash
bash launchers/PREPARE_SPLITS.sh
```

This creates `Data/splits/multiview_sections.json` from the versioned plan in
[`docs/multiview_sections.json`](docs/multiview_sections.json). A deterministic
assignment process balances the plan's candidate sections against the image
and pose populations together, using annotation counts and pose availability.
It does not use model predictions or evaluation scores. Each validation and test partition contains two
separately reported subsets:

| Evaluation type | What training can see | Target share of all retained images |
|---|---|---:|
| Validation: `held_out_view` | Other camera views of the same section | 5% |
| Validation: `unseen_moment` | No camera view from that section | 5% |
| Test: `held_out_view` | Other camera views of the same section | 5% |
| Test: `unseen_moment` | No camera view from that section | 5% |

Held-out views rotate across cameras. Unseen-moment sections withhold every
available view together. Sections use class transitions and large annotation
gaps where possible, with additional boundaries inside long class stretches.
The default buffer removes 75 reference frames on each side of boundaries
touching an evaluation section, for a total width of 150 frames across views.
Held-out-view samples require another training camera with the same normalized
class within three reference frames; samples without a counterpart are excluded.
Missing-pose frames can occupy the temporal buffers, reducing how many usable
pose examples a boundary removes. Those frames remain usable by the image models
outside buffers, so they do not replace the temporal gap required for images.
The buffer audit distinguishes annotated images already missing an athlete
from buffered images with both poses.

Optional group `boundary_adjustments` move an internal section boundary after
the base allocation while preserving every section's train/validation/test role.
The neighboring section endpoint moves with it, and memberships and temporal
buffers are rebuilt. The current plan shifts half-guard validation section
`03-05-013` 50 reference frames earlier, from `[3825, 4087)` to `[3775, 4037)`.
Its retained interior becomes `3850 < frame < 3962`; the ambiguous ending falls
inside the ordinary buffer. Sample counts can change because annotations and
complete poses are unevenly distributed. Final balance, coverage and separation
checks still apply; the manifest records the effective sections and adjustment audit.

Whole sections make the ratios approximate. The plan's `max_class_deviation`
sets the maximum absolute difference between each class's retained train/val/test
shares and the requested targets, separately for image and pose populations.
The versioned default plan sets `0.025`, allowing at most 2.5 percentage points
of deviation. Custom plans that omit the field use `0.04` (4 percentage points).
Allocation fails without creating a
manifest if it cannot meet the limit and the coverage/separation requirements;
revise the candidate sections before retrying. It does not silently relax the
limit or split individual frames at random.

The preparation command reports counts and percentages for every class, the
worst and RMS class deviations in percentage points, buffer costs, exclusions,
and coverage for each evaluation type in both populations. For pose
held-out-view evaluation, an eligible training view must also have both poses.

The plan defines sections on an estimated common reference clock. Each source's
original frame number is mapped with `reference = (frame - offset) / scale`;
the scale and offset were inferred from label transitions and annotation gaps.
Related cameras need not share original frame numbers. The source groups were reviewed visually, but
exact frame synchronization has not been independently verified. Gray gaps in
the annotation timeline mean no released annotation, not necessarily an
unclassified original frame. Neither evaluation type holds out whole matches
or athletes; report results as alternative-view and unseen-section evaluation
within the known recordings, with this alignment limitation.

The schema-3 manifest stores the complete plan, annotation hash, image
assignments, pose eligibility, exclusion reasons, and pooled/per-type class
counts for both populations. It also audits retained temporal separation and
paired-view distances, per-class balance, and the pose eligibility of buffered
images. For substantial class stretches, the distribution audit checks combined
validation/test coverage in the early, middle and late thirds separately for
images and eligible poses. Image checks require at least 20 released annotations
in all three thirds and then require 20 evaluation images per third. Pose checks
use those same thirds and require 20 evaluation examples in each third with at
least 20 available complete-pose examples; an unsupported pose third is skipped
individually.
Both populations use the same estimated reference clock. The pose and image loaders rebuild and verify
these derived values before using the split.
Existing manifests are never overwritten. To try a revised section plan, use
`--section-plan path/to/plan.json --output path/to/new_split.json` and point
`SPLIT_FILE` at the new manifest.

**Start fresh training runs after changing the split.** Models from the previous
split may already have trained on images assigned to the new validation/test
sets. Do not resume those checkpoints or use them for the new held-out results.
Checkpoint selection still uses pooled validation macro F1, with the two
validation subsets also reported separately each epoch. Test remains reserved
for explicit final evaluation.

### Legacy split methods

The previous single-view chronological 70/15/15 method remains available
explicitly. It keeps one source per class and removes a 150-frame temporal
embargo around partition changes. Existing `temporal_split.json` files are
left untouched:

```bash
bash launchers/PREPARE_SPLITS.sh --method single-view-temporal --output "$PWD/Data/splits/legacy_temporal_split.json"
```

Only that method accepts `--class-sources`, `--fractions`, `--gap-frames`, and
`--min-samples-per-class`. Multiview parameters belong in its section plan.
To use an existing legacy manifest, explicitly set `SPLIT_FILE` to its path.
Check any existing `launchers/local_config.sh` or environment override before
starting a new run, because explicit paths take precedence over the new default.

For verified recording metadata and enough independent recordings per class,
the recording-group method keeps all views of each recording together and
requires every class in every partition:

```bash
bash launchers/PREPARE_SPLITS.sh --method recording-group --groups "$PWD/Data/recording_groups.json" --output "$PWD/Data/splits/recording_split.json"
```

Use `docs/recording_groups.example.json` as the metadata template and set
`SPLIT_FILE` to that separate manifest. This method evaluates different recording
groups rather than the within-recording sections used by the default.

## Smoke tests and training

The default comparison has **10 identity-independent position categories**. The
18 raw labels are merged with an explicit mapping. This avoids requiring an RGB
classifier to infer an unobservable athlete numbering convention.

Run a synthetic forward/backward smoke check inside the approved allocation:

```bash
.venv/bin/python -m src.Scripts.smoke_test --device cuda
```

For a small CPU check when a GPU is not allocated, use `--device cpu` explicitly.
The default smoke check exercises all five models using synthetic inputs and
untrained image weights. Select one model with `--model pose`, `pose-wide`,
`pose-attention`, `image`, or `image-dinov3`; `--model both` retains the original
pose/ResNet check. Fresh image training uses pretrained backbone weights,
requiring a download/cache on first use. DINOv3 access setup is described below.

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
| `image` | ImageNet-pretrained ResNet-18 with a replacement `512 → 10` classifier | `head` (default), `input-head`, `layer4` plus classifier, or `full` |
| `image-dinov3` | DINOv3-pretrained ViT-B/16 CLS features → `768 → 10` classifier | `head` (default) or patch projection plus classifier with `input-head` |

Horizontal mirroring and dropout are optional for all five models and are
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

For the MLPs, dropout follows each hidden ReLU. For the image models, it acts on
the features before the classifier: 512 pooled features for ResNet-18 and 768
CLS features for DINOv3. For pose attention, it acts inside the attention/MLP
blocks as described below.
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

Both image models preserve both athletes using resize-and-pad to 224 × 224 and
ImageNet normalization. ResNet-18's original ImageNet `512 → 1000` layer is removed;
the new `512 → 10` classifier takes its place. Choose the trainable scope with
`--image-training`:

- `head` (default): freeze the entire backbone, including BatchNorm statistics.
- `input-head`: train the first convolution and classifier. All other backbone
  weights and all BatchNorm statistics stay frozen. For ResNet-18 this trains
  **14,538 parameters**.
- `layer4`: train the last residual stage (both blocks) and the classifier.
  The initial convolution and stages `layer1`–`layer3` stay frozen, including
  their BatchNorm statistics. BatchNorm in `layer4` updates during training.
- `full`: train the whole backbone and classifier.

For partial fine-tuning in a fresh run:

```bash
bash launchers/START_TRAINING.sh --model image --pretrained \
    --image-training layer4 --epochs 6 --lr 3e-5 \
    --weight-decay 0.001 --dropout 0.15
```

The selected scope and trainable/frozen parameter counts are saved with the run;
the counts are also printed at startup. `--lr` applies to all selected parameters.
Start a new run to change scope. Existing `--fine-tune` and `--no-fine-tune`
remain aliases for `full` and `head`; do not combine them with `--image-training`.
Use `--no-pretrained` for an explicitly untrained image baseline or smoke check. See
[the module documentation](src/Modules/README.md) for the architecture contracts.

### DINOv3 image features

`--model image-dinov3` uses the official
[`facebook/dinov3-vitb16-pretrain-lvd1689m` weights](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
through Transformers 4.57.1. The ViT-B/16 backbone has 12 blocks, 12 attention
heads, four register tokens, and 768-dimensional features. Its normalized CLS
token feeds optional dropout and a new `768 → 10` linear classifier.
`--image-training head` (default) trains only the classifier's **7,690 parameters**.
`--image-training input-head` also trains the existing 16 × 16 patch projection,
for **598,282 trainable parameters**. Transformer blocks, normalization, and
special tokens remain frozen. Both modes keep the backbone in evaluation mode
to disable positional augmentation. DINOv3 does not support `layer4`, `full`,
or the legacy `--fine-tune` alias.

Input-head training must propagate gradients through the frozen intermediate
layers to reach the input projection. It requires more computation and activation
memory than head-only training despite updating relatively few parameters.

The official weights require access approval. Review the conditions on the model
page using your Hugging Face account, then authenticate interactively on the DGX:

```bash
.venv/bin/hf auth login
```

Enter the token at the interactive prompt, not in command arguments or project
files; Git credential storage is not needed. The first pretrained run downloads
and caches the weights. Bootstrap installs dependencies but does not download
these weights. Fresh DINOv3 runs default to pretrained weights; `--no-pretrained`
is available for offline execution checks, not a meaningful frozen-feature baseline.

DINOv3 deliberately uses the same letterboxed image input as ResNet-18 to retain
both athletes. This is the project's preprocessing choice, rather than the
official processor's direct resize. The backbone is recomputed per batch; feature
caching is not implemented. Training, validation metrics, checkpoints and AdamW
options use the shared pipeline. Full checkpoints include the frozen weights,
so resume and evaluation reconstruct the model offline without Hub access.

```bash
RUN_NAME=dinov3-head bash launchers/START_TRAINING.sh \
    --model image-dinov3 --epochs 20 --lr 0.001 --weight-decay 0.0001
```

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

Both flags also work with `pose`, `pose-attention`, `image`, and `image-dinov3`.
The examples without these flags retain mirroring and dropout at zero.

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

Multiview runs also print validation macro F1 for `held_out_view` and
`unseen_moment` and save each subset's full metrics under
`val.by_evaluation_type`. The usual `val` metrics pool both subsets, and pooled
validation macro F1 selects `best.pt`.

Compilation is optional and **off by default** for all five models. Add
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
DINOv3 uses this same whole-model option. Its CUDA compilation and speedup must
be checked on the DGX; earlier smoke results for the other four models do not
validate this new backbone. To check it explicitly without downloading weights:

```bash
.venv/bin/python -m src.Scripts.smoke_test --model image-dinov3 --device cuda --compile
```

CUDA is required by default; unavailable CUDA fails clearly. Data-loading workers,
batch size, seed, learning rate and weight decay can be set with CLI flags; see `--help`. Run
directories are separate and must be new. Keep the seed and split fixed when
comparing models. Pose models exclude samples missing either athlete's pose
from training, validation and test. Images retain these samples, so compare
results alongside their reported sample counts and eligibility rules.

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

To adapt the input layer using an already trained image classifier, start a
new run with `--init-from` and a trusted project `best.pt` or `last.pt` checkpoint:

```bash
.venv/bin/python -m src.Scripts.train \
    --init-from trainings/dinov3-head/checkpoints/best.pt \
    --run-dir trainings/dinov3-input-head \
    --image-training input-head --epochs 5 --lr 1e-5 \
    --weight-decay 0.0001 --dropout 0.15 --horizontal-flip-prob 0.5
```

The same option works for ResNet checkpoints. This copies model weights into
a new run, including the trained classifier, and starts a fresh optimizer,
epoch counter, random state and best-validation score. Model and data paths
can be omitted to use the source checkpoint's values. Annotation/split hashes,
class definitions, image directory and model architecture must match; training
scope and classifier dropout can change. Training hyperparameters use fresh-run
defaults unless explicitly supplied. The pretrained-weight origin is inherited
from the source and cannot be overridden. The run records the source checkpoint's
path, epoch and SHA-256. The example's low learning rate is an experiment setting.
For ResNet, subsequent stages can also select `layer4` or `full`; DINOv3 currently
supports only `head` and `input-head`.

`--resume` continues an existing run with its optimizer and settings, and cannot
be combined with `--init-from`. Resume from the last checkpoint and extend the
total epoch count:

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
confusion matrix. Multiview evaluation saves pooled test scores and a
`by_evaluation_type` breakdown for held-out views and unseen moments, including
each subset's loss and class support. Supplied-pose model results assume pose
annotations already exist; they do not measure an image-to-pose pipeline or
end-to-end inference speed.

## Export training metrics

Collect every run under `TRAININGS_DIR` in one ZIP:

```bash
bash launchers/EXPORT_METRICS.sh
```

The command prints the archive path under `diagnostics/` (or `DIAGNOSTICS_DIR`
if configured). Original run directory names are preserved in the archive.

The archive includes:

- Full per-epoch training/validation metrics, per-class scores and confusion matrices.
- Start and resume configurations, including hyperparameters, split hashes and provenance.
- Existing `diagnostics/test_metrics.json`, if evaluation was already run.
- `summary.csv` with one row per run and best/final logged validation metrics.
- `manifest.json` with file hashes and warnings about missing or incomplete records.

This command reads result files only. It does not load checkpoints, start training
or evaluation, use a GPU, or collect images, model weights, credentials or arbitrary
files. Exports remain ignored by Git. Custom evaluation output paths are not
collected; use the standard `diagnostics/test_metrics.json` for inclusion.

Exports can be taken during training. They are snapshots of the bytes read from
each file, not a simultaneous snapshot of all processes. An unfinished trailing
metrics line is preserved in the ZIP but omitted from the summary with a warning.
The last logged epoch does not prove that a checkpoint was saved or the process
finished. Export again after the batch completes for a final comparison.
Missing settings remain blank, including weight decay in runs logged before it
became configurable. Original configurations are preserved alongside the metrics.

The exporter uses only Python's standard library and also runs directly:

```bash
python3 src/Scripts/export_metrics.py --trainings-dir trainings --output diagnostics/metrics-export.zip
```

Choose a new output name for each snapshot; existing archives are never replaced.

## Git workflow and source bundles

Keep code and shared configuration in Git. Dataset files, local settings and
generated results remain ignored and excluded from upload bundles.

If Git is installed in the Jupyter terminal, the repository host is reachable and
you have repository access, clone through HTTPS into a new/empty folder under
`privado` and run the same setup commands.
For updates, finish active runs, check local changes, then use `git pull --ff-only`.
Do not embed access tokens in clone URLs, scripts or notebooks.

Create a source bundle with:

```powershell
python -B src/Scripts/bundle_project.py
```

Use `--output dist/another-name.zip` for another snapshot. Each ZIP includes a
`source_manifest.json` with source hashes and Git state, even before an initial
commit. The supplied training provenance records Git when available; retain this
bundle manifest alongside archived results when deploying by upload.

## Testing

Run the synthetic tests for annotations, multiview sections, legacy temporal
exclusions, shared assignments, manifest validation and result exports:

```bash
python -B -m unittest discover -s tests -v
```

Small CPU train/resume/evaluation integration tests are opt-in and use synthetic
data only, without pretrained downloads:

```bash
JIUJITSU_RUN_TORCH_TESTS=1 python -B -m unittest discover -s tests -v
```

Split allocation and its tests require NumPy. The integration tests also require
PyTorch, torchvision, tqdm and Transformers. They cover training, resume and evaluation,
augmentation, configurable weight decay, independent dropout
branches and checkpoint reconstruction with nondefault attention settings.
They verify that resumed CPU weights match uninterrupted training and that
progress reporting preserves weights, metrics and RNG state.
Whole-model compilation checks exercise real TorchDynamo graph capture with
the CPU `eager` backend for all five models, plus checkpoint portability and
runtime-option overrides. These do not validate native CUDA/Inductor compilation
or establish a speedup; use the compiled DGX smoke command for that execution path.
DINOv3 tests cover a real randomly initialized ViT-B/16 forward/backward pass,
whole-model CPU graph capture, head/input-head gradients and offline checkpoint reconstruction.
Input-head checks verify that the input projection and classifier update while
all intermediate weights stay unchanged, including ResNet BatchNorm statistics.
Warm-start checks cover copied weights, a fresh optimizer/history, source
checkpoint identity, independent continuation and unchanged source artifacts.
Small substitute-backbone fixtures exercise its shared training/resume/evaluation
integration and pretrained-loading error paths without downloading gated weights.
Synthetic execution checks do not measure model quality on the real dataset.

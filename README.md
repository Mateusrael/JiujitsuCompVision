# Jiu-jitsu position classification

Compare a classifier using supplied athlete poses with an image classifier.
The target environment is Linux with Python 3.11, PyTorch 2.11.0,
torchvision 0.26.0 and CUDA 12.8. Setup creates an isolated project virtual
environment. Training uses one GPU per process.

Code is organized into thin launchers and scripts, reusable feature modules,
and separate per-experiment outputs.

```text
install/        environment probe and Linux dependency setup
launchers/      editable paths and commands to run common operations
src/
  Dataset/      annotation validation, audit, download, recording splits
  Loader/       paired-pose features and image loading
  Modules/      pose MLP and ResNet-18
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

## First upload to JupyterLab

Open `privado/Jiujitsu` in JupyterLab's file browser and upload
`dist/VisaoComp-source.zip` using the upload arrow. Open a **Terminal** from the
Launcher. Opening a terminal does not guarantee it starts inside the selected
file-browser folder. Run these commands on separate lines:

```bash
cd /home/jovyan/privado/Jiujitsu
python3 -m zipfile -e VisaoComp-source.zip .
bash bootstrap.sh --check-only
```

The ZIP has a flat project root, so extraction inside `Jiujitsu` puts `src/`,
`install/` and `launchers/` there. Extract an updated bundle into a fresh folder
or review source differences before replacing edited files. Datasets and runs
are never included in this source bundle.

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
`torch==2.11.0` and `torchvision==0.26.0` from the official CUDA 12.8 wheel index.
It checks package versions, their imports and the PyTorch CUDA runtime build.
No system packages, NVIDIA drivers or system CUDA toolkit are installed.
The corresponding explicit installer command is:

```bash
bash install/install.sh --torch cu128
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

Start with annotations, which serve both baselines:

```bash
bash launchers/DOWNLOAD_DATA.sh
bash launchers/AUDIT_DATASET.sh
```

For the image baseline, download and extract the image archive on the DGX:

```bash
bash launchers/DOWNLOAD_DATA.sh --images
bash launchers/AUDIT_DATASET.sh --images-dir "$PWD/Data/images"
```

Use your configured data path in the last command if it differs from `Data/`.
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

## Create one split for both models

**The recording/camera mapping is not yet verified.** The image prefix is a video
identifier; it is not proof of an independent recording. Do not assign random
frames to train/validation/test.

The audit reports observed prefixes and class counts. Use that report together
with authoritative recording metadata to fill a copy of
`docs/recording_groups.example.json` at `Data/recording_groups.json`. Each observed
prefix needs an entry with `recording` and `split` fields. The allowed split names
are `train`, `val`, and `test`. In `description`, record the evidence used to group
cameras. The template intentionally contains no invented assignments.

```bash
bash launchers/PREPARE_SPLITS.sh --groups "$PWD/Data/recording_groups.json"
```

The split tool rejects missing/extra prefixes, a recording appearing in multiple
splits, duplicate images, and categories absent from any split. It saves the
annotation hash, explicit class mapping, exact assignments and coverage. Reuse
that same manifest for both baselines. Existing manifests are not overwritten.

If there are too few independent recordings per category to satisfy coverage,
this evaluation design cannot support the desired comparison yet. Inspect that
constraint and agree on a clearly labeled alternative or collect additional
recordings; this project will not silently fall back to frame-random evaluation.

## Smoke tests and training

The default comparison has **10 identity-independent position categories**. The
18 raw labels are merged with an explicit mapping. This avoids requiring an RGB
classifier to infer an unobservable athlete numbering convention.

Run a synthetic forward/backward smoke check inside the approved allocation:

```bash
.venv/bin/python -m src.Scripts.smoke_test --device cuda
```

For a small CPU check when a GPU is not allocated, use `--device cpu` explicitly.
The smoke check uses synthetic inputs and untrained image weights. Full image
training uses pretrained ResNet-18 weights, requiring a download/cache on first use.

After the shared recording split is verified and generated:

```bash
RUN_NAME=pose-baseline bash launchers/START_TRAINING.sh --model pose --epochs 30
RUN_NAME=image-baseline bash launchers/START_TRAINING.sh --model image --epochs 30
```

The MLP takes 102 paired-pose features. Both athletes are centered and scaled
together, missing joints remain zero, and confidence is clipped only during
featurization. The image model preserves both athletes using resize-and-pad to
224 × 224, starts with a pretrained ResNet-18 backbone and trains its new head.
Use `--fine-tune` for a separate experiment that updates the backbone. Use
`--no-pretrained` only for an explicitly untrained image baseline or smoke check.

CUDA is required by default; unavailable CUDA fails clearly. Data-loading workers,
batch size, seed and learning rate can be set with CLI flags; see `--help`. Run
directories are separate and must be new. Keep the seed and split fixed when
comparing models. Missing poses are represented explicitly instead of excluding
those samples from one side of the comparison.

Resume from the last checkpoint and extend the total epoch count:

```bash
bash launchers/CONTINUE_TRAINING.sh --resume trainings/pose-baseline/checkpoints/last.pt --epochs 60
```

Checkpoints include model configuration, optimizer and random states. Saved
configuration is reused on resume, with incompatible changes rejected. Best
checkpoints are selected on validation data; test data is used by the explicit
evaluation command:

```bash
bash launchers/EVALUATE.sh --checkpoint trainings/pose-baseline/checkpoints/best.pt
bash launchers/EVALUATE.sh --checkpoint trainings/image-baseline/checkpoints/best.pt
```

Results include accuracy, macro-F1, per-class precision/recall/support and a
confusion matrix. Supplied-pose MLP results assume pose annotations already exist;
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

The parser and annotation conventions were checked against a bounded sample of
the official annotation endpoint. Only a small ZIP header sample was inspected;
the complete dataset was not downloaded to this PC. Local tests use synthetic
fixtures; no real-data training or DGX/H100 execution has been performed here.
The remote environment report, full dataset audit, recording mapping and actual
GPU smoke check remain the first deployment checks.

```bash
python -B -m unittest discover -s tests -v
```

Small CPU train/resume/evaluation integration tests are opt-in and use synthetic
data only, without pretrained downloads:

```bash
JIUJITSU_RUN_TORCH_TESTS=1 python -B -m unittest discover -s tests -v
```

These require the project environment; other checks use Python's standard
library. Both baseline integration checks have passed locally, including exact
pose-resume equivalence to uninterrupted CPU execution. The PyTorch 2.11 / CUDA
12.8 installation and GPU smoke check must still be run on the target DGX.

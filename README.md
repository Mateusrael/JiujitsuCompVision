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
  Dataset/      annotation validation, audit, download, shared evaluation splits
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

## Create one split for both models

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
is excluded. Both classifiers use exactly the same retained examples.

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
Both loaders reconstruct and verify the manifest before using it. Existing
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
The smoke check uses synthetic inputs and untrained image weights. Full image
training uses pretrained ResNet-18 weights, requiring a download/cache on first use.

The following are training examples to run after preparing the data and shared
split and choosing the model experiment. Setup and split preparation do not
start them automatically:

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

The user-reported DGX environment check confirmed PyTorch 2.11.0 + CUDA 12.8,
torchvision 0.26.0, and GPU access. Both pose and image GPU smoke tests passed.
The full remote annotation audit found 120,279 records, all ten normalized
classes, no malformed records, and no duplicate image IDs. That audit did not
check image files; the full image download and image audit are not yet confirmed.
No real-data training has been reported.

The default temporal split also passed validation against the complete official
annotation content on 2026-10-07, with the same SHA-256 as the DGX audit and the
counts reported above. The 208,848,730 annotation bytes were read into memory;
the annotation file and image archive were not saved on this development PC.
Only the derived report was saved to `diagnostics/temporal_split_validation.json`.
Generate the actual shared manifest on the DGX with `PREPARE_SPLITS.sh`.

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

These require the project environment; other checks use Python's standard
library. Both baseline integration checks have passed locally, including exact
pose-resume equivalence to uninterrupted CPU execution. Synthetic execution
checks do not measure model quality on the real dataset.

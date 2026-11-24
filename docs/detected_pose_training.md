# Train classifiers with YOLO26 and ViTPose inputs

This queue compares frozen **YOLO26x-pose** and **RT-DETR + ViTPose++ Huge** inputs
using the existing ten class labels and the same verified split. Only the pose
classifiers are trained. Detector inference has already happened in the
[full-dataset extraction queue](pose_extraction_queue.md); training reads its
completed caches and does not load or fine-tune either extractor.

## Prepare after extraction finishes

Run from the project root in the server environment:

```bash
.venv/bin/python -m src.Scripts.prepare_detected_pose_queue
```

Defaults are `Data/pose_detections/full_dataset`, `Data/annotations.json`,
`Data/splits/multiview_sections_shifted.json`, and
`launchers/detected-poses.local.json`. Override them with `--cache-dir`,
`--annotations`, `--split`, or `--output`. Add `--dry-run` to perform the same
checks without saving the queue and preflight report.

Preparation refuses absent or unfinished final caches, changed SHA-256 identities,
different detector configurations, missing/duplicate cache images, and invalid
split manifests. It verifies both final metadata files and every cache row,
including frames with zero or one detected athlete. It does not repair or resume
an extraction. The extractor's final merge verifies the image files; preparation
checks those serialized image identities and the cache/annotation identity without
opening all images again.

The preflight prints eligible counts for each split and class, and saves the
detailed report as `launchers/detected-poses.preflight.local.json`. Both default
outputs are ignored by Git. The report records full-dataset two-athlete detection
coverage, recovered images whose supplied poses were incomplete, and their original
split assignments (including excluded frames). Per-split additions to the supplied-pose
baseline separate recovered missing poses from views that regain a valid training
counterpart. Losses are counted independently, so simultaneous gains and losses are
visible. Every cohort includes class counts. Evaluation-group counts, cache identities
and common-image counts between detectors are also saved. Test counts describe
coverage only; no test predictions or scores are computed during preparation or
training selection.

The report compares effective train/validation/test shares with **80%/10%/10%**,
overall and within each class, and shows deviations in percentage points. The
denominator is the retained train + validation + test population, excluding buffers,
missing detected pairs and unpaired held-out views. Quality flags identify absent
classes, overall deviations greater than 5 percentage points, and per-class
deviations greater than 10 points. These flags are for review and do not change
assignments or prevent queue preparation; empty partitions or evaluation groups
still fail because the training/evaluation pipeline cannot use them. Review the
actual coverage and balance after both caches finish.

Preparation validates all 24 commands with the existing queue parser. It does not
start training. Inspect the resolved commands, then start explicitly:

```bash
bash launchers/QUEUE.sh start launchers/detected-poses.local.json --dry-run
bash launchers/QUEUE.sh start launchers/detected-poses.local.json
bash launchers/QUEUE.sh status --watch
```

The queue allows up to six concurrent jobs, subject to available GPU memory.
Detector pairs are adjacent within each model and seed; all three model families
appear among the first six jobs. Each job requires 4,096 MiB free before launch.
This is a scheduling threshold, not a memory reservation. Preserve completed caches
after preparation, and rerun preflight if the dataset, split, or caches change.

## Matched recipe

Each row below runs on both detectors with seeds **46, 47, and 48**, giving
**24 fresh jobs**. All use batch size 128, LR 3e-4, two data-loader workers,
compilation, athlete swapping, and horizontal flip probability 0.5.

| Classifier | Epochs | Weight decay | Dropout | Attention configuration |
|---|---:|---:|---|---|
| `pose` | 300 | 1e-4 | 0.3 | — |
| `pose-wide` | 150 | 1e-4 | 0.3 | — |
| `pose-attention`, width 32 | 100 | 0.05 | attention 0; MLP 0.4 | 4 heads, 2 layers, MLP 128, mean pooling |
| `pose-attention`, width 64 | 100 | 0.05 | attention 0; MLP 0.4 | 4 heads, 2 layers, MLP 256, mean pooling |

The initial detector comparison keeps joint noise, joint dropout and athlete
dropout at zero. No supplied/detected mixture or image classifier is included.
The queue uses `--pose-detections-scope all --no-pose-keep-annotated`: the same
detector supplies training, epoch-validation and final-test inputs. Omitting the
scope option preserves the older `train` behavior, where only training inputs
are replaced and validation/test use supplied poses.

## Interpret the detector comparison

Extraction covers every annotated image, including excluded/buffer frames and
frames missing supplied poses. In scope `all`, every image assigned to train,
validation or test can qualify when the detector supplies two observed athlete
poses; supplied-pose completeness does not gate eligibility. A held-out view also
needs a matching same-class training-camera witness with two detected poses.
Previously unusable supplied poses and previously unpaired views may therefore
be recovered. Frames are never moved between partitions, original buffers remain
excluded, and lost witnesses are excluded rather than reclassified as unseen
moments. Class labels remain annotation-derived.

YOLO26 and ViTPose may retain different images and class distributions. Matching
seeds, hyperparameters and split assignments does **not** make their effective
evaluation cohorts identical. Read the preflight coverage and common-image counts
alongside validation scores. A detector-specific score measures performance on
that detector's retained cohort; a direct pose-quality comparison also needs
evaluation on the same image intersection, with coverage/failures reported
separately. The generator reports that intersection but does not silently restrict
either detector's training data to it.

Select checkpoints and compare settings using pooled validation macro F1, with
held-out-view and unseen-moment metrics at the same selected checkpoint. Keep
final-test evaluation for the selected configurations. Seeds provide repeated
runs; three-seed means and spread are descriptive, not a significance claim.

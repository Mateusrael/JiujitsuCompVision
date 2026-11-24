# Loader

`splits.read_manifest(path, annotations_path, records)` verifies the annotation
hash, exact image membership, 10-class taxonomy, and class coverage in all three
partitions. The default schema-3 multi-view manifest rebuilds section allocation,
paired-view eligibility, population counts and buffers from its saved plan and
the annotations, rejecting edited derived fields. Schema-2 temporal and schema-1
recording-group manifests remain supported for historical results.

All models share temporal sections. The image model uses the main assignments;
pose models additionally require two observed poses in every partition, and a
complete-pose training counterpart for held-out-view evaluation. `pose_assignments`
can only exclude main-assignment records, never move them to another partition.
`split_records` optionally selects an evaluation type; `filter_pose_records`
applies the pose eligibility rules. Training/validation/test remain disjoint.

`pose.pose_features(pose1, pose2, swap=False)` returns 102 floats: 34 joints with
x, y, confidence. One bounding box over valid joints from both athletes supplies
the center and scale, preserving their relative geometry. Confidence must be
positive; its feature is clamped to [0, 1]. Missing joints/athletes are zeros.
The annotation parser preserves original confidence values, including > 1.

`datasets.PoseDataset` optionally swaps athlete slots with probability 0.5 during
training only; training enables this by default. Optional horizontal mirroring
is controlled by `horizontal_flip_prob`, default 0.0 (disabled). It negates
centered x and exchanges each athlete's COCO left/right joint slots. The y and
confidence values move with their joint unchanged; athlete slots keep their
order. The pure-Python `pose.horizontal_flip_pose(features)` implements the same
mapping for a 102-element feature vector. Both augmentations preserve labels
under the supported 10-class mapping and make independent random choices.

To imitate detector errors, training can also apply `pose_joint_noise` (x/y noise
over the pair scale), `pose_joint_dropout` and `pose_athlete_dropout`, all 0.0 by
default. `perturb_pose` normalizes the degraded pose again; evaluation never uses it.

`--pose-detections` replaces the annotated poses of the train split with a detector's
JSONL cache (`with_detected_poses`); `--pose-keep-annotated` trains on both. Validation
and test stay annotated, so checkpoint selection does not change.
Add `--pose-detections-scope all` for detector-only train/validation/test inputs.
This mode requires two observed detected poses and starts from the original image
assignments, independently of supplied-pose completeness. Held-out views must have
a same-class training-camera counterpart with two detected poses. The policy can
recover missing supplied poses and previously unpaired views; it can also exclude
new detector failures. Main split assignments and buffers remain unchanged.
It cannot be combined with `--pose-keep-annotated`. Evaluation and notebook input
inspection use the saved source and verify the cache checksum.
Cache rows contain a unique `image` string and `poses`, a list of zero to two
17-joint poses with finite `[x, y, confidence]` values. Additional detector metadata
is allowed. In the legacy `train` scope, every supplied-pose-eligible training frame
must have a row; empty detections omit the detected copy, while one detected athlete
leaves the second slot empty. That scope applies the supplied-pose eligibility filter
before replacement. Scope `all` requires rows for every original image, preserves
empty results for coverage auditing, then excludes images without two observed
detected athletes and held-out views without detected training witnesses. The cache SHA-256
is saved with the run and checked again on resume and when consuming its rows;
changed caches and cache-trained checkpoints without a saved hash cannot resume.
`sample_counts` retains `original` image assignment counts and the selected source's
`eligible` and `removed` counts;
`samples` records the actual population after detection replacement or mixing.
Legacy train-only replacement additionally reports `annotated_samples`, `detected_samples`
and `frames_without_detection`.
`pose`, `pose-wide` and `pose-attention` all use this dataset and its `(B, 102)`
batch contract. The attention model reshapes the features into 34 joint tokens
inside the model; it does not introduce a separate feature pipeline. All three
pose models use the same source policy: supplied-pose runs require both supplied
athletes; scope `all` requires both detected athletes regardless of supplied poses;
legacy train-only detector copies may contain only one detected athlete.
Confidence <= 0 still supplies the attention model's joint mask.

The [queue preflight](../../docs/detected_pose_training.md) reports full-cache
recovery, independent gains/losses against the supplied-pose baseline, per-split
class counts, evaluation groups, and effective 80/10/10 proportions/deviations.
Its absent-class and imbalance flags are descriptive; it never reallocates images.

`datasets.ImageDataset` requires exactly one `<image>.jpg`, `.jpeg`, or `.png`
directly under the supplied image directory. It resizes the entire frame with
aspect ratio preserved, pads to 224 by 224, and applies ImageNet normalization.
Optional horizontal mirroring happens before resize-and-pad and uses the same
`horizontal_flip_prob` setting, default 0.0. It never crops or vertically flips
the frame. Image paths are checked before training starts.

The training entry point applies mirroring only to training datasets, across all
four models. Validation and test use probability zero and never swap athlete
slots. Disabled mirroring consumes no random-number draw, preserving earlier
zero-mirroring run behavior. Dataset augmentation does not alter cached pose
features or image files.

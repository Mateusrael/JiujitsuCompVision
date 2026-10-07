# Loader

`splits.read_manifest(path, annotations_path, records)` verifies the annotation
hash, exact image membership, 10-class taxonomy, and class coverage in all three
partitions. For the default schema-2 temporal manifest, it rebuilds assignments,
exclusions, boundaries and separation checks from saved source selections and
parameters, rejecting edited derived fields. Excluded examples enter none of the
models. The schema-1 recording-group method remains supported and rebuilds
assignments from its saved grouping evidence. All four classifiers consume
exactly the same retained train/validation/test examples.

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
`pose`, `pose-wide` and `pose-attention` all use this dataset and its `(B, 102)`
batch contract. The attention model reshapes the features into 34 joint tokens
inside the model; it does not introduce a separate feature pipeline or exclude
missing-pose examples. Confidence <= 0 supplies its missing-joint mask.

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

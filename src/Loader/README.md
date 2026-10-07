# Loader

`splits.read_manifest(path, annotations_path, records)` verifies the annotation
hash, exact image membership, 10-class taxonomy, and class coverage in all three
partitions. For the default schema-2 temporal manifest, it rebuilds assignments,
exclusions, boundaries and separation checks from saved source selections and
parameters, rejecting edited derived fields. Excluded examples enter neither
classifier. The schema-1 recording-group method remains supported and rebuilds
assignments from its saved grouping evidence. Both classifiers consume exactly
the same retained train/validation/test examples.

`pose.pose_features(pose1, pose2, swap=False)` returns 102 floats: 34 joints with
x, y, confidence. One bounding box over valid joints from both athletes supplies
the center and scale, preserving their relative geometry. Confidence must be
positive; its feature is clamped to [0, 1]. Missing joints/athletes are zeros.
The annotation parser preserves original confidence values, including > 1.

`datasets.PoseDataset` optionally swaps athlete slots with probability 0.5 during
training only. This is label preserving for the supported 10-class mapping.
`datasets.ImageDataset` requires exactly one `<image>.jpg`, `.jpeg`, or `.png`
directly under the supplied image directory. It resizes the entire frame with
aspect ratio preserved, pads to 224 by 224, and applies ImageNet normalization.
It never crops the frame. Image paths are checked before training starts.

# Loader

`splits.read_manifest(path, annotations_path, records)` verifies the annotation
hash, exact image membership, 10-class taxonomy, and class coverage in all three
partitions. It rebuilds the assignments from the saved recording-group evidence,
rejecting edited assignments and recordings/camera views that cross partitions.
Both classifiers consume this same manifest.

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

"""Pure-Python pose features, also usable before installing PyTorch."""

import math


def pose_features(pose1, pose2, *, swap=False):
    """Return 102 pair-normalized features; invalid joints become three zeros.

    Both athletes share one bounding-box center and scale, preserving their
    relative geometry. Confidence > 0 marks an observed joint. Input scores
    are allowed above one; the confidence feature alone is clamped to [0, 1].
    Missing athletes and nonfinite coordinates/confidences produce zero joints.
    """
    poses = (pose2, pose1) if swap else (pose1, pose2)
    joints = []
    for pose in poses:
        if pose is None:
            joints.extend([None] * 17)
            continue
        if len(pose) != 17:
            raise ValueError("Each pose must contain exactly 17 joints")
        for joint in pose:
            if len(joint) != 3:
                raise ValueError("Each joint must contain x, y, confidence")
            x, y, confidence = (float(value) for value in joint)
            valid = all(math.isfinite(value) for value in (x, y, confidence))
            joints.append((x, y, confidence) if valid and confidence > 0 else None)
    observed = [joint for joint in joints if joint is not None]
    if not observed:
        return [0.0] * 102
    xmin, xmax = min(j[0] for j in observed), max(j[0] for j in observed)
    ymin, ymax = min(j[1] for j in observed), max(j[1] for j in observed)
    center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
    scale = max(xmax - xmin, ymax - ymin, 1e-8)
    features = []
    for joint in joints:
        if joint is None:
            features.extend((0.0, 0.0, 0.0))
        else:
            x, y, confidence = joint
            features.extend(((x - center_x) / scale, (y - center_y) / scale,
                             min(confidence, 1.0)))
    return features

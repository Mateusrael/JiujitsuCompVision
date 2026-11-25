"""Conservative filtering of the zero, one or two poses already in a cache.

The policy favors reliable skeletons over coverage. It uses no images, labels,
reference annotations, extra candidates or detector inference. The fixed
heuristics need a coverage/quality audit; they do not establish keypoint ground
truth or promise better classifier metrics. No upright-body assumption is used.
"""

from copy import deepcopy
from itertools import combinations
import math
from statistics import median


BODY = tuple(range(5, 17))
TORSO = (5, 6, 11, 12)
LOWER_LIMBS = (13, 14, 15, 16)
BODY_EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
              (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
              (12, 14), (14, 16))

QUALITY_PROFILES = ("cached-pose-quality-v1", "cached-pose-quality-v2")
DEFAULT_PROFILE = QUALITY_PROFILES[-1]

# Keep the original settings immutable so previously filtered caches can be
# validated against the policy that produced them.
_SETTINGS = {
    "profile": "cached-pose-quality-v1",
    "min_body_joints": 8,
    "min_torso_joints": 3,
    "min_lower_limb_joints": 2,
    "min_body_scale_pixels": 4.0,
    "min_torso_scale_pixels": 2.0,
    "max_center_outlier_ratio": 3.5,
    "max_head_center_distance_ratio": 4.5,
    "max_bone_length_ratio": 6.0,
    "duplicate_min_common_body_joints": 6,
    "duplicate_mean_distance_ratio": 0.08,
    "duplicate_joint_distance_ratio": 0.10,
    "duplicate_min_close_fraction": 0.75,
}


def quality_config(detector, profile=DEFAULT_PROFILE):
    """Return a fresh JSON-serializable copy of the immutable named policy."""
    if detector not in ("yolo", "vitpose"):
        raise ValueError(f"Unknown pose detector: {detector!r}")
    if profile not in QUALITY_PROFILES:
        raise ValueError(f"Unknown cached-pose quality profile: {profile!r}")
    config = {**deepcopy(_SETTINGS), "detector": detector,
              "joint_confidence": 0.5 if detector == "yolo" else 0.3}
    if profile == "cached-pose-quality-v2":
        # Hidden knees/ankles need not invalidate a reliable upper-body estimate.
        # The eight-body-joint floor still prevents very sparse poses passing.
        config.update(profile=profile, min_torso_joints=2, min_lower_limb_joints=0)
    return config


def _numeric(value):
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _distance(first, second):
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _assess_pose(pose, index, config):
    if not isinstance(pose, (list, tuple)) or len(pose) != 17:
        raise ValueError(f"Pose {index} must contain 17 COCO joints")
    clean, discarded = [], []
    for joint_index, joint in enumerate(pose):
        if (not isinstance(joint, (list, tuple)) or len(joint) != 3
                or any(not _numeric(v) for v in joint)):
            raise ValueError(f"Pose {index}, joint {joint_index}: expected numeric [x,y,c]")
        x, y, confidence = joint
        valid = (all(math.isfinite(v) for v in joint)
                 and confidence >= config["joint_confidence"] and x >= 0 and y >= 0)
        if valid:
            clean.append([float(x), float(y), min(float(confidence), 1.0)])
        else:
            clean.append([0.0, 0.0, 0.0])
            discarded.append(joint_index)

    visible = [i for i in BODY if clean[i][2] > 0]
    torso = sum(clean[i][2] > 0 for i in TORSO)
    lower = sum(clean[i][2] > 0 for i in LOWER_LIMBS)
    reasons = []
    if len(visible) < config["min_body_joints"]:
        reasons.append("insufficient_body_joints")
    if torso < config["min_torso_joints"]:
        reasons.append("insufficient_torso_joints")
    if lower < config["min_lower_limb_joints"]:
        reasons.append("insufficient_lower_limb_joints")

    center, scale = None, 0.0
    if visible:
        points = [clean[i] for i in visible]
        center = [median(p[axis] for p in points) for axis in (0, 1)]
        distances = [_distance(a, b) for a, b in combinations(points, 2)]
        # Twice the median pair distance is robust to one distant joint. It
        # remains rotation-independent for athletes lying or rolling on the mat.
        scale = 2 * median(distances) if distances else 0.0
        if not math.isfinite(scale) or not all(math.isfinite(v) for v in center):
            reasons.append("nonfinite_body_geometry")
            center, scale = None, 0.0
        elif scale < config["min_body_scale_pixels"]:
            reasons.append("degenerate_body_geometry")
        elif max(_distance(p, center) for p in points) > config["max_center_outlier_ratio"] * scale / 2:
            reasons.append("body_joint_outlier")
        if scale >= config["min_body_scale_pixels"]:
            # Face joints also contribute to pair_bbox_v1 normalization. A
            # confidently misplaced nose can otherwise distort the whole pair.
            # No head joints are required; only gross visible outliers matter.
            if any(clean[i][2] > 0 and _distance(clean[i], center)
                   > config["max_head_center_distance_ratio"] * scale / 2
                   for i in range(5)):
                reasons.append("head_joint_outlier")
        torso_points = [clean[i] for i in TORSO if clean[i][2] > 0]
        if len(torso_points) >= config["min_torso_joints"]:
            torso_span = max(_distance(a, b) for a, b in combinations(torso_points, 2))
            if torso_span < config["min_torso_scale_pixels"]:
                reasons.append("degenerate_torso_geometry")
        edges = [_distance(clean[a], clean[b]) for a, b in BODY_EDGES
                 if clean[a][2] > 0 and clean[b][2] > 0]
        positive_edges = [length for length in edges if length > 1e-8]
        # Reject only gross segment outliers; do not demand symmetry, a standing
        # pose or equal projected limb lengths under occlusion/foreshortening.
        if (len(positive_edges) >= 4
                and max(positive_edges) > config["max_bone_length_ratio"] * median(positive_edges)):
            reasons.append("body_segment_outlier")
    report = {"index": index, "accepted": not reasons, "reasons": reasons,
              "body_joints": len(visible), "torso_joints": torso,
              "lower_limb_joints": lower, "reliable_joints": 17 - len(discarded),
              "discarded_joint_indices": discarded,
              "mean_body_confidence": (sum(clean[i][2] for i in visible) / len(visible)
                                       if visible else 0.0),
              "body_center": center, "body_scale": scale}
    return {"pose": clean, "report": report}


def _duplicates(first, second, config):
    common = [i for i in BODY if first["pose"][i][2] > 0 and second["pose"][i][2] > 0]
    if len(common) < config["duplicate_min_common_body_joints"]:
        return False
    scale = max(first["report"]["body_scale"], second["report"]["body_scale"])
    if scale <= 0:
        return False
    distances = [_distance(first["pose"][i], second["pose"][i]) / scale for i in common]
    close = sum(d <= config["duplicate_joint_distance_ratio"] for d in distances) / len(distances)
    return (sum(distances) / len(distances) <= config["duplicate_mean_distance_ratio"]
            and close >= config["duplicate_min_close_fraction"])


def filter_pose_pair(poses, *, detector, profile=DEFAULT_PROFILE):
    """Filter a cached pair without reordering or introducing any new poses.

    Returns zero or two sanitized poses and a detailed ``quality_filter`` audit.
    Coordinates are retained except that invalid or low-confidence joints become
    three zeros; confidence is clamped to [0,1]. The original image dimensions
    are unavailable, so there is deliberately no invented upper coordinate bound.
    A weak/missing/duplicate pair is entirely excluded. Input is not modified.
    v2 allows two reliable torso joints and imposes no separate lower-limb
    minimum. v1 retains the original three-torso/two-lower-limb requirements.
    """
    config = quality_config(detector, profile)
    if not isinstance(poses, (list, tuple)) or len(poses) > 2:
        raise ValueError("Cached input must contain zero, one or two poses")
    assessed = [_assess_pose(pose, i, config) for i, pose in enumerate(poses)]
    report = {"profile": config["profile"], "detector": detector,
              "accepted": False, "reason": "missing_pair", "input_pose_count": len(poses),
              "pose_quality": [item["report"] for item in assessed]}
    if len(assessed) != 2:
        return {"poses": [], "quality_filter": report}
    if any(not item["report"]["accepted"] for item in assessed):
        report["reason"] = "unreliable_pose"
        return {"poses": [], "quality_filter": report}
    if _duplicates(*assessed, config):
        report["reason"] = "duplicate_skeletons"
        for item in assessed:
            item["report"].update(accepted=False, reasons=["duplicate_skeleton"])
        return {"poses": [], "quality_filter": report}
    report.update(accepted=True, reason="accepted")
    return {"poses": [item["pose"] for item in assessed], "quality_filter": report}

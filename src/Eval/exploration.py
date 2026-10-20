"""Read-only dataset browsing and checkpoint inference for interactive notebooks.

Importing this module does not import PyTorch, Matplotlib, Pillow or torchvision.
Checkpoint inference uses the training pipeline's validation and preprocessing;
it never downloads pretrained weights, compiles a model or enables augmentation.
"""

import gc
import math
from pathlib import Path
import random

from src.Dataset.annotations import both_poses_present, load_annotations, sha256_file
from src.Loader.pose import pose_features
from src.Loader.splits import filter_pose_records, read_manifest, split_records
from src.Modules.registry import MODEL_NAMES, POSE_MODEL_NAMES


# COCO's zero-based skeleton; each athlete is drawn independently.
POSE_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
)
POSE_COLORS = ("#0B9D9A", "#E77B27")


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def resolve_checkpoint(run_or_checkpoint, checkpoint="best"):
    """Resolve a run directory or an explicit trusted local checkpoint file."""
    path = Path(run_or_checkpoint).expanduser().resolve()
    if path.is_dir():
        if checkpoint not in ("best", "last"):
            raise ValueError("checkpoint must be best or last when selecting a run directory")
        path = path / "checkpoints" / f"{checkpoint}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint is not available locally: {path}")
    return path


class ExplorationData:
    """Verified canonical annotations, split membership and model eligibility.

    Frame filters are inclusive and use each source's local frame numbers;
    either endpoint may be None for an open-ended range.
    Returned rows are canonical annotations; do not edit them in place.
    """

    def __init__(self, annotations_path, split_path, images_dir=None):
        self.annotations_path = Path(annotations_path).expanduser().resolve()
        self.split_path = Path(split_path).expanduser().resolve()
        self.images_dir = (Path(images_dir).expanduser().resolve() if images_dir is not None
                           else self.annotations_path.parent / "images")
        self.records = load_annotations(self.annotations_path)
        self.manifest = read_manifest(self.split_path, self.annotations_path, self.records)
        self.annotation_sha256 = sha256_file(self.annotations_path)
        self.split_sha256 = sha256_file(self.split_path)
        self.by_id = {record["image"]: record for record in self.records}
        self.class_names = list(self.manifest["class_names"])
        self.sources = sorted({record["image"][:2] for record in self.records})

    def verify_unchanged(self):
        """Reject files replaced since this notebook loaded its dataset snapshot."""
        if (sha256_file(self.annotations_path) != self.annotation_sha256
                or sha256_file(self.split_path) != self.split_sha256):
            raise ValueError("Dataset files changed; recreate ExplorationData before inference")

    def canonical(self, record_or_id):
        image_id = record_or_id if isinstance(record_or_id, str) else record_or_id["image"]
        if image_id not in self.by_id:
            raise ValueError(f"Unknown image ID: {image_id}")
        record = self.by_id[image_id]
        if not isinstance(record_or_id, str) and record_or_id != record:
            raise ValueError(f"Annotation differs from the verified dataset: {image_id}")
        return record

    def select(self, model, *, split="val", class_name=None, source=None,
               evaluation_type=None, frame_range=None):
        """Select one split, applying the same eligibility as the chosen model."""
        if model not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {model!r}")
        if class_name is not None and class_name not in self.class_names:
            raise ValueError(f"Unknown class: {class_name!r}")
        if source is not None:
            source = str(source).zfill(2)
            if source not in self.sources:
                raise ValueError(f"Unknown source: {source!r}")
        if frame_range is not None:
            if (not isinstance(frame_range, (tuple, list)) or len(frame_range) != 2
                    or any(v is not None and
                           (isinstance(v, bool) or not isinstance(v, int) or v < 0)
                           for v in frame_range)
                    or (all(v is not None for v in frame_range)
                        and frame_range[0] > frame_range[1])):
                raise ValueError("frame_range must be an inclusive (first, last) pair")
        selected = split_records(self.records, self.manifest, split, evaluation_type)
        if model in POSE_MODEL_NAMES:
            selected = filter_pose_records(selected, manifest=self.manifest)
        label_map = self.manifest["label_map"]
        return [record for record in selected
                if (class_name is None or label_map[record["position"]] == class_name)
                and (source is None or record["image"][:2] == source)
                and (frame_range is None
                     or ((frame_range[0] is None or record["frame"] >= frame_range[0])
                         and (frame_range[1] is None or record["frame"] <= frame_range[1])))]

    def sample(self, model, *, count=6, seed=None, **filters):
        """Sample up to count rows; seed=None draws a new sample on each call.

        Small filtered populations return all available rows without replacement.
        An empty selection raises a descriptive error.
        """
        _positive_integer(count, "count")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or None")
        selected = self.select(model, **filters)
        if not selected:
            raise ValueError("No eligible samples match the current filters")
        return random.Random(seed).sample(selected, min(count, len(selected)))

    def describe(self, record_or_id):
        record = self.canonical(record_or_id)
        image_id = record["image"]
        return {
            "image": image_id, "actual": self.manifest["label_map"][record["position"]],
            "raw_label": record["position"], "source": image_id[:2], "frame": record["frame"],
            "split": self.manifest["assignments"][image_id],
            "evaluation_type": self.manifest.get("evaluation_types", {}).get(image_id),
            "both_poses_present": both_poses_present(record),
            "pose_split": self.manifest.get("pose_assignments", {}).get(image_id),
        }

    def features(self, record_or_id):
        """The exact 102 pair-normalized input values used by the pose models."""
        record = self.canonical(record_or_id)
        return pose_features(record.get("pose1"), record.get("pose2"))

    def image_path(self, record_or_id):
        # Resolve image files without importing torch-backed dataset adapters.
        record = self.canonical(record_or_id)
        found = [self.images_dir / f"{record['image']}{suffix}"
                 for suffix in (".jpg", ".jpeg", ".png")
                 if (self.images_dir / f"{record['image']}{suffix}").is_file()]
        if len(found) != 1:
            raise FileNotFoundError(
                f"Expected exactly one image for {record['image']} in {self.images_dir}; "
                f"found {len(found)}")
        return found[0]

    def open_image(self, record_or_id):
        """Return an independent RGB Pillow image after closing the source file."""
        from PIL import Image
        with Image.open(self.image_path(record_or_id)) as image:
            return image.convert("RGB")

    def draw_pose(self, ax, record_or_id, *, min_confidence=0.0):
        """Draw supplied raw pixel coordinates over an existing image axes."""
        if (isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float))
                or not math.isfinite(min_confidence) or min_confidence < 0):
            raise ValueError("min_confidence must be finite and nonnegative")
        record = self.canonical(record_or_id)
        for slot, color in zip(("pose1", "pose2"), POSE_COLORS):
            pose = record.get(slot)
            if not pose:
                continue
            valid = [joint[2] > min_confidence for joint in pose]
            for start, end in POSE_EDGES:
                if valid[start] and valid[end]:
                    ax.plot([pose[start][0], pose[end][0]], [pose[start][1], pose[end][1]],
                            color=color, linewidth=1.6, alpha=0.9)
            ax.scatter([joint[0] for joint, observed in zip(pose, valid) if observed],
                       [joint[1] for joint, observed in zip(pose, valid) if observed],
                       c=color, s=18, edgecolors="white", linewidths=0.4, label=slot)
        return ax


class ModelExplorer:
    """One resident model loaded from an existing trusted project checkpoint.

    ``from_run`` uses torch.load with Python state, like the project's evaluator:
    supply only your own checkpoints. Metrics ZIPs do not contain these weights.
    Use ``close`` before loading another large model to release device memory.
    """

    @classmethod
    def from_run(cls, run_or_checkpoint, data, *, checkpoint="best", device="cpu"):
        import torch
        from src.Modules.models import build_model
        from src.Train.checkpoints import load_checkpoint
        from src.Train.engine import validate_checkpoint_data

        if not isinstance(data, ExplorationData):
            raise TypeError("data must be an ExplorationData instance")
        path = resolve_checkpoint(run_or_checkpoint, checkpoint)
        target = torch.device(device)
        if target.type not in ("cpu", "cuda"):
            raise ValueError("Choose cpu or an explicit CUDA device such as cuda:0")
        if target.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; choose device='cpu' or check allocation")
            index = target.index if target.index is not None else torch.cuda.current_device()
            if not 0 <= index < torch.cuda.device_count():
                raise ValueError(f"CUDA device index {index} is not visible to this process")
            target = torch.device("cuda", index)
        data.verify_unchanged()
        payload = load_checkpoint(path)
        config = dict(payload["config"])
        validate_checkpoint_data(config, config["model"], data.annotations_path,
                                 data.split_path, data.manifest)
        # Relocated files are allowed only after the hashes and label map agree.
        config.update(annotations=str(data.annotations_path), split=str(data.split_path),
                      images_dir=str(data.images_dir))
        model = build_model(config["model_config"], pretrained=False)
        model.load_state_dict(payload["model_state"], strict=True)
        instance = cls()
        instance.checkpoint_path = path
        instance.checkpoint_epoch = payload["epoch"]
        instance.run_dir = path.parent.parent if path.parent.name == "checkpoints" else path.parent
        instance.class_names = list(config["class_names"])
        instance.model_name = config["model"]
        instance.config = config
        instance.data = data
        instance.device = target
        trainable = sum(parameter.numel() for parameter in model.parameters()
                        if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        instance.parameter_counts = {"total": total, "trainable": trainable,
                                     "frozen": total - trainable}
        # Optimizer/RNG payload is unnecessary for inference and can be large.
        del payload
        instance.model = model.to(target).eval()
        return instance

    def predict(self, records, *, top_k=3, batch_size=32):
        """Return CPU-only prediction dictionaries in exactly the requested order.

        Rows may be annotations or image IDs, and may span explicit train/val/test
        selections. Excluded rows and pose-ineligible rows are rejected. Sampling
        in a notebook is exploration, not a complete held-out evaluation.
        """
        import torch
        from src.Train.engine import make_dataset

        if self.model is None:
            raise RuntimeError("This model has been closed; load a checkpoint again")
        _positive_integer(top_k, "top_k")
        _positive_integer(batch_size, "batch_size")
        if top_k > len(self.class_names):
            raise ValueError(f"top_k cannot exceed {len(self.class_names)} classes")
        if isinstance(records, (str, dict)):
            raise TypeError("Pass a list of records or image IDs")
        rows = [self.data.canonical(record) for record in records]
        for record in rows:
            image_id = record["image"]
            if self.data.manifest["assignments"][image_id] not in ("train", "val", "test"):
                raise ValueError(f"Sample {image_id} is excluded from the split")
        if (self.model_name in POSE_MODEL_NAMES
                and len(filter_pose_records(rows, manifest=self.data.manifest)) != len(rows)):
            raise ValueError("Selected samples include pose-ineligible rows; select a pose model first")
        if not rows:
            return []
        dataset = make_dataset(self.model_name, rows, self.config, training=False)
        result = []
        self.model.eval()
        with torch.inference_mode():
            for first in range(0, len(rows), batch_size):
                features = torch.stack([dataset[index][0]
                                        for index in range(first, min(first + batch_size, len(rows)))])
                logits = self.model(features.to(self.device))
                if logits.ndim != 2 or logits.shape[1] != len(self.class_names):
                    raise ValueError("Model output does not match the checkpoint class names")
                if not torch.isfinite(logits).all():
                    raise ValueError("Model produced non-finite logits")
                probabilities = logits.softmax(dim=1).cpu()
                for offset, values in enumerate(probabilities.tolist()):
                    metadata = self.data.describe(rows[first + offset])
                    # Stable ties match argmax in the project's evaluator.
                    ranked = sorted(range(len(values)), key=lambda index: (-values[index], index))
                    predicted = self.class_names[ranked[0]]
                    result.append({
                        **metadata, "predicted": predicted,
                        "correct": predicted == metadata["actual"],
                        "probabilities": values,
                        "top_k": [{"class_name": self.class_names[index],
                                   "probability": values[index]}
                                  for index in ranked[:top_k]],
                    })
        return result

    def close(self):
        """Release this model; no files, weights or training runs are modified."""
        import torch
        self.model = None
        gc.collect()
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

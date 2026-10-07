"""Torch dataset adapters over canonical annotations and a shared split."""

import math
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.Loader.pose import COCO_FLIP_INDICES, pose_features


def validate_flip_probability(probability):
    if (isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not math.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError("horizontal_flip_prob must be finite and in [0, 1]")


def should_flip(probability):
    # Disabled augmentation does not consume random draws.
    return probability > 0 and (probability == 1 or torch.rand(()) < probability)


class PoseDataset(Dataset):
    def __init__(self, records, class_names, label_map, *, swap_athletes=False,
                 horizontal_flip_prob=0.0):
        validate_flip_probability(horizontal_flip_prob)
        class_index = {name: index for index, name in enumerate(class_names)}
        self.features = torch.tensor(
            [pose_features(record.get("pose1"), record.get("pose2"))
             for record in records], dtype=torch.float32)
        self.labels = torch.tensor(
            [class_index[label_map[record["position"]]] for record in records],
            dtype=torch.long)
        self.swap_athletes = swap_athletes
        self.horizontal_flip_prob = horizontal_flip_prob

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        features = self.features[index]
        if self.swap_athletes and torch.rand(()) < 0.5:
            features = torch.cat((features[51:], features[:51]))
        if should_flip(self.horizontal_flip_prob):
            joints = features.reshape(2, 17, 3)[:, list(COCO_FLIP_INDICES), :].clone()
            joints[..., 0].neg_()
            features = joints.reshape(102)
        return features, self.labels[index]


def resolve_image(images_dir, image_id):
    candidates = [Path(images_dir) / f"{image_id}{suffix}"
                  for suffix in (".jpg", ".jpeg", ".png")]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(
            f"Expected exactly one image for {image_id} in {images_dir}; found {len(found)}")
    return found[0]


def letterbox(image, size=224):
    """Resize the whole frame into a square without cropping either athlete."""
    from PIL import Image
    width, height = image.size
    ratio = size / max(width, height)
    resized = image.resize((max(1, round(width * ratio)), max(1, round(height * ratio))),
                           Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (124, 116, 104))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


class ImageDataset(Dataset):
    def __init__(self, records, class_names, label_map, images_dir, *, horizontal_flip_prob=0.0):
        validate_flip_probability(horizontal_flip_prob)
        from torchvision.transforms import Normalize, PILToTensor
        class_index = {name: index for index, name in enumerate(class_names)}
        # Resolve once, so absent/ambiguous files fail before a GPU run starts.
        self.paths = [resolve_image(images_dir, record["image"]) for record in records]
        self.labels = [class_index[label_map[record["position"]]] for record in records]
        self.horizontal_flip_prob = horizontal_flip_prob
        self.to_tensor = PILToTensor()
        self.normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        from PIL import Image, ImageOps
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            if should_flip(self.horizontal_flip_prob):
                image = ImageOps.mirror(image)
            image = letterbox(image)
        pixels = self.to_tensor(image).float().div_(255)
        return self.normalize(pixels), self.labels[index]

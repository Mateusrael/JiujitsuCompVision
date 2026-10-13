"""Frozen DINOv3 ViT-B/16 image features with a trainable linear classifier."""

import math
from numbers import Real

import torch
from torch import nn

from src.Modules.dinov3_config import (
    DINOV3_BACKBONE_CONFIG, DINOV3_TRANSFORMERS_VERSION, DINOV3_WEIGHTS_ID,
)


def _transformers_classes():
    try:
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
    except ModuleNotFoundError as error:
        if error.name != "transformers":
            raise
        raise ModuleNotFoundError(
            "DINOv3 requires its optional dependencies. Install them with "
            f"python -m pip install transformers=={DINOV3_TRANSFORMERS_VERSION}"
        ) from error
    return DINOv3ViTConfig, DINOv3ViTModel


def _validate_backbone_config(config):
    differences = {
        key: (expected, getattr(config, key, None))
        for key, expected in DINOV3_BACKBONE_CONFIG.items()
        if getattr(config, key, None) != expected
    }
    if differences:
        raise ValueError(
            "DINOv3 weights do not match the supported ViT-B/16 configuration "
            f"(expected, loaded): {differences}"
        )


class FrozenDINOv3Classifier(nn.Module):
    def __init__(self, num_classes=10, *, pretrained=True, dropout=0.0):
        super().__init__()
        if (isinstance(num_classes, bool) or not isinstance(num_classes, int)
                or num_classes <= 0):
            raise ValueError("num_classes must be a positive integer")
        if (isinstance(dropout, bool) or not isinstance(dropout, Real)
                or not math.isfinite(dropout) or not 0 <= dropout < 1):
            raise ValueError("dropout must be a finite number in [0, 1)")
        if not isinstance(pretrained, bool):
            raise ValueError("pretrained must be a boolean")
        config_class, model_class = _transformers_classes()
        if pretrained:
            try:
                self.backbone = model_class.from_pretrained(
                    DINOV3_WEIGHTS_ID, use_safetensors=True)
            except OSError as error:
                raise RuntimeError(
                    "Could not load DINOv3 pretrained weights. Request access at "
                    f"https://huggingface.co/{DINOV3_WEIGHTS_ID}, then run "
                    "hf auth login in the training environment with an account "
                    "that has access. The first load also requires network "
                    "access unless the weights are already cached."
                ) from error
            _validate_backbone_config(self.backbone.config)
        else:
            # Full project checkpoints include the frozen backbone weights.
            # This path also supports random-weight smoke tests without network.
            self.backbone = model_class(config_class(**DINOV3_BACKBONE_CONFIG))
        self.backbone.requires_grad_(False)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(DINOV3_BACKBONE_CONFIG["hidden_size"], num_classes)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen features deterministic, including RoPE coordinate
        # augmentation, while allowing the classification head's dropout.
        self.backbone.eval()
        return self

    def forward(self, images):
        # Do not use inference_mode: the head needs to save these features for
        # its weight gradients even though the backbone itself remains frozen.
        with torch.no_grad():
            features = self.backbone(pixel_values=images).pooler_output
        return self.classifier(self.dropout(features))

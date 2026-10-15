"""DINOv3 ViT-B/16 with head-only or input-projection-and-head training."""

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
    def __init__(self, num_classes=10, *, pretrained=True, dropout=0.0,
                 image_training="head"):
        super().__init__()
        if (isinstance(num_classes, bool) or not isinstance(num_classes, int)
                or num_classes <= 0):
            raise ValueError("num_classes must be a positive integer")
        if (isinstance(dropout, bool) or not isinstance(dropout, Real)
                or not math.isfinite(dropout) or not 0 <= dropout < 1):
            raise ValueError("dropout must be a finite number in [0, 1)")
        if not isinstance(pretrained, bool):
            raise ValueError("pretrained must be a boolean")
        if image_training not in ("head", "input-head"):
            raise ValueError("DINOv3 image_training must be head or input-head")
        self.image_training = image_training
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
            # Full project checkpoints include all backbone weights.
            # This path also supports random-weight smoke tests without network.
            self.backbone = model_class(config_class(**DINOV3_BACKBONE_CONFIG))
        self.backbone.requires_grad_(False)
        if self.image_training == "input-head":
            # The patch projection is the input layer. Keep the CLS, register,
            # and mask tokens frozen along with every transformer block.
            self.backbone.get_input_embeddings().requires_grad_(True)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(DINOV3_BACKBONE_CONFIG["hidden_size"], num_classes)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Keep backbone execution deterministic, including RoPE coordinate
        # augmentation, while allowing the classification head's dropout.
        self.backbone.eval()
        return self

    def forward(self, images):
        if self.image_training == "head":
            # The head saves these frozen features for its own weight gradients.
            with torch.no_grad():
                features = self.backbone(pixel_values=images).pooler_output
        else:
            # Frozen middle weights still need to propagate gradients back to
            # the trainable input projection. Respect any outer inference mode.
            features = self.backbone(pixel_values=images).pooler_output
        return self.classifier(self.dropout(features))

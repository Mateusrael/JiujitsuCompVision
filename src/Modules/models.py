"""Self-describing pose and image comparison architectures."""

import math
from numbers import Real

import torch.nn as nn

from src.Modules.pose_attention import (
    PoseAttentionClassifier, resolve_dropout_rates, validate_attention_parameters,
)


def _validate_dropout(dropout):
    if (isinstance(dropout, bool) or not isinstance(dropout, Real)
            or not math.isfinite(dropout) or not 0 <= dropout < 1):
        raise ValueError("dropout must be a finite number in [0, 1)")


class PoseMLP(nn.Module):
    def __init__(self, num_classes=10, *, dropout=0.0):
        super().__init__()
        _validate_dropout(dropout)
        self.network = nn.Sequential(nn.Linear(102, 102), nn.ReLU(),
                                     nn.Linear(102, 34), nn.ReLU(),
                                     nn.Linear(34, num_classes))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, features):
        for layer in self.network:
            features = layer(features)
            if isinstance(layer, nn.ReLU):
                features = self.dropout(features)
        return features


class PoseWideMLP(nn.Module):
    def __init__(self, num_classes=10, *, dropout=0.0):
        super().__init__()
        _validate_dropout(dropout)
        self.network = nn.Sequential(
            nn.Linear(102, 512), nn.ReLU(), nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, num_classes))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, features):
        for layer in self.network:
            features = layer(features)
            if isinstance(layer, nn.ReLU):
                features = self.dropout(features)
        return features


class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10, *, pretrained=True, fine_tune=False,
                 dropout=0.0):
        super().__init__()
        _validate_dropout(dropout)
        from torchvision.models import ResNet18_Weights, resnet18
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(512, num_classes)
        self.fine_tune = fine_tune
        if not fine_tune:
            self.backbone.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if not self.fine_tune:
            # Freezing weights also keeps pretrained BatchNorm statistics fixed.
            self.backbone.eval()
        return self

    def forward(self, images):
        return self.classifier(self.dropout(self.backbone(images)))


def architecture_config(model, num_classes=10, *, fine_tune=False, dropout=0.0,
                        attention_dim=128, attention_heads=4, attention_layers=4,
                        attention_mlp_dim=512, attention_dropout=None,
                        attention_mlp_dropout=None, attention_pooling="mean"):
    _validate_dropout(dropout)
    if model == "pose":
        return {"type": "pose_mlp", "input_features": 102, "hidden": [102, 34],
                "num_classes": num_classes, "pose_preprocessing": "pair_bbox_v1",
                "dropout": float(dropout)}
    if model == "image":
        return {"type": "resnet18", "num_classes": num_classes,
                "fine_tune": fine_tune, "image_size": 224,
                "image_preprocessing": "letterbox_imagenet_v1",
                "dropout": float(dropout)}
    if model == "pose-wide":
        if (isinstance(num_classes, bool) or not isinstance(num_classes, int)
                or num_classes <= 0):
            raise ValueError("num_classes must be a positive integer")
        return {"type": "pose_wide_mlp", "input_features": 102,
                "hidden": [512, 128, 32], "num_classes": num_classes,
                "activation": "relu", "pose_preprocessing": "pair_bbox_v1",
                "dropout": float(dropout)}
    if model == "pose-attention":
        attention_rate, mlp_rate = resolve_dropout_rates(
            dropout, attention_dropout=attention_dropout,
            mlp_dropout=attention_mlp_dropout)
        validate_attention_parameters(
            num_classes=num_classes, embedding_dim=attention_dim,
            num_heads=attention_heads, num_layers=attention_layers,
            mlp_dim=attention_mlp_dim, dropout=dropout,
            pooling=attention_pooling,
            attention_dropout=attention_rate, mlp_dropout=mlp_rate)
        return {"type": "pose_attention", "input_features": 102,
                "num_joints": 34, "joint_features": 3,
                "embedding_dim": attention_dim, "num_heads": attention_heads,
                "num_layers": attention_layers, "mlp_dim": attention_mlp_dim,
                "attention_dropout": attention_rate, "mlp_dropout": mlp_rate,
                "activation": "gelu",
                "normalization": "pre_layer_norm",
                "position_encoding": "learned_joint_slots",
                "pooling": attention_pooling,
                "missing_joint_mask": "confidence_le_zero",
                "joint_order": "athlete1_coco17_then_athlete2_coco17",
                "num_classes": num_classes,
                "pose_preprocessing": "pair_bbox_v1"}
    raise ValueError(f"Unknown model: {model!r}")


def build_model(config, *, pretrained=False):
    if not isinstance(config, dict):
        raise ValueError("Model configuration must be a dictionary")
    model_type = config.get("type")
    if model_type == "pose_mlp":
        expected = architecture_config("pose", config["num_classes"],
                                       dropout=config.get("dropout"))
        if config != expected:
            raise ValueError(f"Unsupported model configuration: {config!r}")
        model = PoseMLP(config["num_classes"], dropout=config["dropout"])
    elif model_type == "pose_wide_mlp":
        expected = architecture_config("pose-wide", config["num_classes"],
                                       dropout=config.get("dropout"))
        if config != expected:
            raise ValueError(f"Unsupported model configuration: {config!r}")
        model = PoseWideMLP(config["num_classes"], dropout=config["dropout"])
    elif model_type == "pose_attention":
        if "attention_dropout" not in config or "mlp_dropout" not in config:
            raise ValueError(f"Unsupported model configuration: {config!r}")
        expected = architecture_config(
            "pose-attention", config["num_classes"],
            attention_dim=config.get("embedding_dim"),
            attention_heads=config.get("num_heads"),
            attention_layers=config.get("num_layers"),
            attention_mlp_dim=config.get("mlp_dim"),
            attention_dropout=config["attention_dropout"],
            attention_mlp_dropout=config["mlp_dropout"],
            attention_pooling=config.get("pooling"))
        if config != expected:
            raise ValueError(f"Unsupported model configuration: {config!r}")
        model = PoseAttentionClassifier(
            config["num_classes"], embedding_dim=config["embedding_dim"],
            num_heads=config["num_heads"], num_layers=config["num_layers"],
            mlp_dim=config["mlp_dim"], attention_dropout=config["attention_dropout"],
            mlp_dropout=config["mlp_dropout"],
            pooling=config["pooling"])
    elif model_type == "resnet18":
        expected = architecture_config("image", config["num_classes"],
                                       fine_tune=config.get("fine_tune", False),
                                       dropout=config.get("dropout"))
        # Checkpoints supply all weights; loading never downloads them again.
        if config != expected:
            raise ValueError(f"Unsupported model configuration: {config!r}")
        model = ImageClassifier(config["num_classes"], pretrained=pretrained,
                                fine_tune=config["fine_tune"],
                                dropout=config["dropout"])
    else:
        raise ValueError(f"Unsupported model configuration: {config!r}")
    if config != expected:
        raise ValueError(f"Unsupported model configuration: {config!r}")
    return model

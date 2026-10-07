"""Fixed, self-describing baseline architectures."""

import torch.nn as nn


class PoseMLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(102, 102), nn.ReLU(),
                                     nn.Linear(102, 34), nn.ReLU(),
                                     nn.Linear(34, num_classes))

    def forward(self, features):
        return self.network(features)


class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10, *, pretrained=True, fine_tune=False):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone.fc = nn.Identity()
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
        return self.classifier(self.backbone(images))


def architecture_config(model, num_classes=10, *, fine_tune=False):
    if model == "pose":
        return {"type": "pose_mlp", "input_features": 102, "hidden": [102, 34],
                "num_classes": num_classes, "pose_preprocessing": "pair_bbox_v1"}
    if model == "image":
        return {"type": "resnet18", "num_classes": num_classes,
                "fine_tune": fine_tune, "image_size": 224,
                "image_preprocessing": "letterbox_imagenet_v1"}
    raise ValueError(f"Unknown model: {model!r}")


def build_model(config, *, pretrained=False):
    model_type = config.get("type")
    if model_type == "pose_mlp":
        expected = architecture_config("pose", config["num_classes"])
        model = PoseMLP(config["num_classes"])
    elif model_type == "resnet18":
        expected = architecture_config("image", config["num_classes"],
                                       fine_tune=config.get("fine_tune", False))
        # Checkpoints supply all weights; loading never downloads them again.
        if config != expected:
            raise ValueError(f"Unsupported model configuration: {config!r}")
        model = ImageClassifier(config["num_classes"], pretrained=pretrained,
                                fine_tune=config["fine_tune"])
    else:
        raise ValueError(f"Unsupported model configuration: {config!r}")
    if config != expected:
        raise ValueError(f"Unsupported model configuration: {config!r}")
    return model

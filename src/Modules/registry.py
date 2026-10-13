"""Model names usable by command-line tools without importing PyTorch."""

POSE_MODEL_NAMES = ("pose", "pose-wide", "pose-attention")
IMAGE_MODEL_NAMES = ("image", "image-dinov3")
MODEL_NAMES = (*POSE_MODEL_NAMES, *IMAGE_MODEL_NAMES)

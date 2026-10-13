"""Image training modes, shared by CLI and model construction without importing Torch."""

IMAGE_TRAINING_MODES = ("head", "layer4", "full")


def resolve_image_training(image_training=None, fine_tune=None):
    if fine_tune is not None and not isinstance(fine_tune, bool):
        raise ValueError("fine_tune must be a boolean")
    if image_training is None:
        return "full" if fine_tune else "head"
    if image_training not in IMAGE_TRAINING_MODES:
        raise ValueError(f"image_training must be one of {IMAGE_TRAINING_MODES}")
    if fine_tune is not None and fine_tune != (image_training != "head"):
        raise ValueError("image_training and fine_tune disagree about backbone training")
    return image_training

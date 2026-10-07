"""CLI configuration resolution, independent of PyTorch."""

from pathlib import Path


DEFAULTS = {"epochs": 20, "batch_size": 128, "lr": 0.001, "workers": 2,
            "seed": 42, "pretrained": True, "fine_tune": False,
            "swap_athletes": True}
RESUME_FIXED = ("batch_size", "lr", "workers", "seed", "pretrained",
                "fine_tune", "swap_athletes")


def resolve_settings(args, previous=None):
    resolved = dict(previous or DEFAULTS)
    for name, default in DEFAULTS.items():
        supplied = getattr(args, name, None)
        if previous is not None and name in RESUME_FIXED:
            if supplied is not None and supplied != previous[name]:
                raise ValueError(f"Cannot change --{name.replace('_', '-')} when resuming")
        resolved[name] = supplied if supplied is not None else resolved.get(name, default)
    if resolved["epochs"] < 1 or resolved["batch_size"] < 1:
        raise ValueError("epochs and batch-size must be positive")
    if resolved["workers"] < 0 or resolved["seed"] < 0:
        raise ValueError("workers and seed must be nonnegative")
    if not 0 < resolved["lr"] < float("inf"):
        raise ValueError("lr must be positive and finite")
    if args.model == "pose" and resolved["fine_tune"]:
        raise ValueError("--fine-tune applies only to the image model")
    return resolved


def resolve_run_dir(run_dir, resume=None):
    if resume:
        checkpoint = Path(resume).expanduser().resolve()
        if checkpoint.name != "last.pt" or checkpoint.parent.name != "checkpoints":
            raise ValueError("Resume must use <run-dir>/checkpoints/last.pt")
        actual = checkpoint.parent.parent
        if run_dir and Path(run_dir).expanduser().resolve() != actual:
            raise ValueError("--run-dir must be the directory owning the resume checkpoint")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        return actual
    if not run_dir:
        raise ValueError("A fresh run requires --run-dir")
    actual = Path(run_dir).expanduser().resolve()
    if actual.exists():
        raise FileExistsError(f"Fresh run directory already exists: {actual}")
    return actual

"""CLI configuration resolution, independent of PyTorch."""

import math
from pathlib import Path

from src.Modules.registry import POSE_MODEL_NAMES
from src.Modules.execution import validate_compile_settings


DEFAULTS = {"epochs": 20, "batch_size": 128, "lr": 0.001, "weight_decay": 0.0001,
            "workers": 2,
            "seed": 42, "pretrained": True, "fine_tune": False,
            "swap_athletes": True, "horizontal_flip_prob": 0.0, "dropout": 0.0,
            "compile": False, "compile_mode": "default", "progress": True}
RESUME_FIXED = ("batch_size", "lr", "weight_decay", "workers", "seed", "pretrained",
                "fine_tune", "swap_athletes", "horizontal_flip_prob", "dropout")
ATTENTION_DEFAULTS = {"attention_dim": 128, "attention_heads": 4, "attention_layers": 4,
                      "attention_mlp_dim": 512, "attention_dropout": 0.0,
                      "attention_mlp_dropout": 0.0,
                      "attention_pooling": "mean"}
ATTENTION_DROPOUTS = ("attention_dropout", "attention_mlp_dropout")


def resolve_settings(args, previous=None):
    resolved = dict(previous or DEFAULTS)
    for name, default in DEFAULTS.items():
        supplied = getattr(args, name, None)
        saved = (previous or {}).get(name, default)
        if previous is not None and name in RESUME_FIXED:
            if supplied is not None and supplied != saved:
                raise ValueError(f"Cannot change --{name.replace('_', '-')} when resuming")
        resolved[name] = supplied if supplied is not None else saved
    validate_compile_settings(resolved["compile"], resolved["compile_mode"])
    if (not resolved["compile"]
            and getattr(args, "compile_mode", None) not in (None, "default")):
        raise ValueError("A nondefault --compile-mode requires --compile")
    if not isinstance(resolved["progress"], bool):
        raise ValueError("progress must be a boolean")
    if resolved["epochs"] < 1 or resolved["batch_size"] < 1:
        raise ValueError("epochs and batch-size must be positive")
    if resolved["workers"] < 0 or resolved["seed"] < 0:
        raise ValueError("workers and seed must be nonnegative")
    if not 0 < resolved["lr"] < float("inf"):
        raise ValueError("lr must be positive and finite")
    weight_decay = resolved["weight_decay"]
    if (isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float))
            or not math.isfinite(weight_decay) or weight_decay < 0):
        raise ValueError("--weight-decay must be finite and nonnegative")
    if args.model in POSE_MODEL_NAMES and resolved["fine_tune"]:
        raise ValueError("--fine-tune applies only to the image model")
    for name, default in ATTENTION_DEFAULTS.items():
        supplied = getattr(args, name, None)
        if args.model != "pose-attention":
            if supplied is not None:
                raise ValueError(f"--{name.replace('_', '-')} requires --model pose-attention")
            continue
        if name in ATTENTION_DROPOUTS:
            # The shared rate is only a fallback when starting a fresh run.
            # Resuming restores the two effective rates independently.
            if previous is not None and name not in previous:
                raise ValueError(f"Checkpoint is missing the required setting {name}")
            saved = previous[name] if previous is not None else resolved["dropout"]
        else:
            saved = (previous or {}).get(name, default)
        if previous is not None and supplied is not None and supplied != saved:
            raise ValueError(f"Cannot change --{name.replace('_', '-')} when resuming")
        resolved[name] = supplied if supplied is not None else saved
    for name, inclusive_upper in (("dropout", False), ("horizontal_flip_prob", True)):
        value = resolved[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (value > 1 if inclusive_upper else value >= 1)):
            interval = "[0, 1]" if inclusive_upper else "[0, 1)"
            raise ValueError(f"--{name.replace('_', '-')} must be finite and in {interval}")
    if args.model == "pose-attention":
        for name in ("attention_dim", "attention_heads", "attention_layers", "attention_mlp_dim"):
            value = resolved[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be a positive integer")
        if resolved["attention_dim"] % resolved["attention_heads"]:
            raise ValueError("attention-dim must be divisible by attention-heads")
        for name in ATTENTION_DROPOUTS:
            dropout = resolved[name]
            if (isinstance(dropout, bool) or not isinstance(dropout, (int, float))
                    or not math.isfinite(dropout) or not 0 <= dropout < 1):
                raise ValueError(f"--{name.replace('_', '-')} must be finite and in [0, 1)")
        if resolved["attention_pooling"] not in ("mean", "cls"):
            raise ValueError("attention-pooling must be mean or cls")
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

"""Atomic checkpoint writes and explicit random-state restoration."""

import random
from pathlib import Path

import torch


def random_state(generators):
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loaders": {name: generator.get_state() for name, generator in generators.items()}}


def restore_random_state(state, generators):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed since checkpoint; exact resume is unavailable")
        torch.cuda.set_rng_state_all(state["cuda"])
    for name, generator in generators.items():
        generator.set_state(state["loaders"][name])


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    # Own, trusted training artifacts include Python RNG state as well as tensors.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"schema_version", "config", "model_state", "optimizer_state",
                "epoch", "best_macro_f1", "random_state"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Not a complete project training checkpoint")
    if checkpoint["schema_version"] != 1:
        raise ValueError("Unsupported checkpoint schema version")
    return checkpoint

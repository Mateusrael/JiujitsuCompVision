"""Evaluate one saved classifier on the manifest's held-out test partition."""

from pathlib import Path

import torch

from src.Dataset.annotations import load_annotations
from src.Loader.splits import read_manifest, split_records
from src.Modules.models import build_model
from src.Modules.execution import compile_model, validate_compile_settings
from src.Train.checkpoints import load_checkpoint
from src.Train.engine import (make_dataset, make_loader, resolve_device, score_model,
                              validate_checkpoint_data)
from src.Train.run_metadata import provenance, write_json


def evaluate(args):
    compile_enabled = getattr(args, "compile", False)
    compile_mode = getattr(args, "compile_mode", "default")
    progress = getattr(args, "progress", True)
    validate_compile_settings(compile_enabled, compile_mode)
    if not compile_enabled and compile_mode != "default":
        raise ValueError("A nondefault --compile-mode requires --compile")
    if not isinstance(progress, bool):
        raise ValueError("progress must be a boolean")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    config = dict(checkpoint["config"])
    annotations_path = Path(args.annotations or config["annotations"]).expanduser().resolve()
    split_path = Path(args.split or config["split"]).expanduser().resolve()
    records = load_annotations(annotations_path)
    manifest = read_manifest(split_path, annotations_path, records)
    validate_checkpoint_data(config, config["model"], annotations_path, split_path, manifest)
    if args.images_dir:
        config["images_dir"] = str(Path(args.images_dir).expanduser().resolve())
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.workers is not None:
        config["workers"] = args.workers
    if config["batch_size"] < 1 or config["workers"] < 0:
        raise ValueError("batch-size must be positive and workers nonnegative")
    device = resolve_device(args.device)
    dataset = make_dataset(config["model"], split_records(records, manifest, "test"), config)
    loader = make_loader(dataset, config, device,
                         generator=torch.Generator().manual_seed(config["seed"]))
    model = build_model(config["model_config"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    runtime_model = compile_model(model, enabled=compile_enabled, mode=compile_mode)
    metrics = score_model(runtime_model, loader, device, config["class_names"],
                          progress=progress, description="test")
    result = {"split": "test", "checkpoint": str(checkpoint_path),
              "checkpoint_epoch": checkpoint["epoch"], "model": config["model"],
              "model_config": config["model_config"],
              "execution": {"compile": compile_enabled, "compile_mode": compile_mode,
                            "progress": progress},
              "annotation_sha256": config["annotation_sha256"],
              "split_sha256": config["split_sha256"], "label_map": config["label_map"],
              "split_method": manifest["split_method"],
              "evaluation_scope": manifest.get("evaluation_scope", "Held-out recording groups."),
              "metrics": metrics, "provenance": provenance(torch, device)}
    output = Path(args.output).expanduser().resolve() if args.output else (
        checkpoint_path.parent.parent / "diagnostics" / "test_metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(f"test accuracy {metrics['accuracy']:.4f} | macro F1 {metrics['macro_f1']:.4f}")
    print(f"Saved held-out test metrics: {output}")
    return result

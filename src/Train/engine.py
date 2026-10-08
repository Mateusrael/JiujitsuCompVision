"""Training and validation over the same verified split manifest for all models."""

import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.Dataset.annotations import load_annotations, sha256_file
from src.Eval.metrics import classification_metrics
from src.Loader.datasets import ImageDataset, PoseDataset
from src.Loader.splits import read_manifest, split_records
from src.Modules.execution import compile_model
from src.Modules.models import architecture_config, build_model
from src.Modules.registry import POSE_MODEL_NAMES
from src.Train.checkpoints import (load_checkpoint, random_state, restore_random_state,
                                   save_checkpoint)
from src.Train.config import ATTENTION_DEFAULTS, resolve_run_dir, resolve_settings
from src.Train.run_metadata import provenance, write_json


def resolve_device(requested):
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Check the VLab GPU allocation, "
                           "or pass --device cpu explicitly for a small test.")
    return device


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    random.seed(torch.initial_seed() % (2 ** 32))


def make_dataset(model, records, config, *, training=False):
    flip_probability = config.get("horizontal_flip_prob", 0.0) if training else 0.0
    if model in POSE_MODEL_NAMES:
        return PoseDataset(records, config["class_names"], config["label_map"],
                           swap_athletes=training and config["swap_athletes"],
                           horizontal_flip_prob=flip_probability)
    if model != "image":
        raise ValueError(f"Unknown model: {model!r}")
    return ImageDataset(records, config["class_names"], config["label_map"],
                        config["images_dir"], horizontal_flip_prob=flip_probability)


def make_loader(dataset, config, device, *, training=False, generator=None):
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=training,
                      num_workers=config["workers"], pin_memory=device.type == "cuda",
                      worker_init_fn=seed_worker, generator=generator,
                      persistent_workers=False)


def score_model(model, loader, device, class_names, *, progress=True,
                description="validation", position=0):
    model.eval()
    loss_sum, correct, targets, predictions = 0.0, 0, [], []
    with torch.inference_mode(), tqdm(
            total=len(loader), desc=description, unit="batch", leave=False,
            dynamic_ncols=True, mininterval=0.2, miniters=1,
            disable=not progress, position=position) as batches:
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
            loss_sum += loss.item()
            batch_targets = labels.cpu().tolist()
            batch_predictions = logits.argmax(dim=1).cpu().tolist()
            targets.extend(batch_targets)
            predictions.extend(batch_predictions)
            if progress:
                correct += sum(target == prediction
                               for target, prediction in zip(batch_targets, batch_predictions))
                batches.set_postfix(loss=f"{loss_sum / len(targets):.4f}",
                                    accuracy=f"{correct / len(targets):.4f}", refresh=False)
            batches.update(1)
    metrics = classification_metrics(targets, predictions, class_names)
    metrics["loss"] = loss_sum / len(targets)
    return metrics


def train_epoch(model, loader, optimizer, device, *, progress=True,
                description="train", position=0):
    model.train()
    loss_sum, correct, samples = 0.0, 0, 0
    with tqdm(total=len(loader), desc=description, unit="batch", leave=False,
              dynamic_ncols=True, mininterval=0.2, miniters=1,
              disable=not progress, position=position) as batches:
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; inspect the inputs and learning rate")
            loss.backward()
            optimizer.step()
            samples += labels.numel()
            loss_sum += loss.item() * labels.numel()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            if progress:
                batches.set_postfix(loss=f"{loss_sum / samples:.4f}",
                                    accuracy=f"{correct / samples:.4f}", refresh=False)
            batches.update(1)
    if not samples:
        raise ValueError("The training split is empty")
    return {"loss": loss_sum / samples, "accuracy": correct / samples, "samples": samples}


def configured_architecture(model, num_classes, config):
    options = ({name: config[name] for name in ATTENTION_DEFAULTS}
               if model == "pose-attention" else {})
    return architecture_config(model, num_classes, fine_tune=config["fine_tune"],
                               dropout=config["dropout"], **options)


def validate_checkpoint_data(config, model, annotations_path, split_path, manifest):
    expected = {"model": model, "annotation_sha256": sha256_file(annotations_path),
                "split_sha256": sha256_file(split_path),
                "class_names": manifest["class_names"], "label_map": manifest["label_map"]}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Checkpoint {key} does not match this run's data/model/split")
    model_config = configured_architecture(model, len(manifest["class_names"]), config)
    if config.get("model_config") != model_config:
        raise ValueError("Checkpoint architecture/preprocessing configuration is unsupported")


def train(args):
    run_dir = resolve_run_dir(args.run_dir, args.resume)
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    previous = checkpoint["config"] if checkpoint else None
    for name in ("model", "annotations", "split"):
        if getattr(args, name) is None:
            saved = (previous or {}).get(name)
            if saved is None:
                raise ValueError(f"A fresh run requires --{name}")
            setattr(args, name, saved)
    config = resolve_settings(args, previous)
    device = resolve_device(args.device or (previous or {}).get("device", "cuda"))
    print(f"Preparing {args.model} data and model on {device}...", flush=True)
    annotations_path = Path(args.annotations).expanduser().resolve()
    split_path = Path(args.split).expanduser().resolve()
    records = load_annotations(annotations_path)
    manifest = read_manifest(split_path, annotations_path, records)
    if checkpoint:
        validate_checkpoint_data(previous, args.model, annotations_path, split_path, manifest)
    images_dir = args.images_dir or (previous or {}).get("images_dir")
    if args.model == "image" and not images_dir:
        raise ValueError("--images-dir is required for the image model")
    images_dir = str(Path(images_dir).expanduser().resolve()) if images_dir else None
    if checkpoint and images_dir != previous.get("images_dir"):
        raise ValueError("Cannot change the image directory while resuming")
    config.update({"schema_version": 1, "model": args.model,
                   "model_config": configured_architecture(args.model, len(manifest["class_names"]), config),
                   "annotations": str(annotations_path), "split": str(split_path),
                   "annotation_sha256": sha256_file(annotations_path),
                   "split_sha256": sha256_file(split_path),
                   "split_method": manifest["split_method"],
                   "evaluation_scope": manifest.get("evaluation_scope", "Held-out recording groups."),
                   "class_names": manifest["class_names"], "label_map": manifest["label_map"],
                   "images_dir": images_dir, "device": str(device), "run_dir": str(run_dir)})
    start_epoch = checkpoint["epoch"] if checkpoint else 0
    if config["epochs"] <= start_epoch:
        raise ValueError(f"--epochs is the total target; choose a value above {start_epoch}")
    seed_everything(config["seed"])
    generators = {part: torch.Generator().manual_seed(config["seed"] + index)
                  for index, part in enumerate(("train", "val"))}
    loaders = {}
    for part in ("train", "val"):
        dataset = make_dataset(args.model, split_records(records, manifest, part), config,
                               training=part == "train")
        loaders[part] = make_loader(dataset, config, device, training=part == "train",
                                    generator=generators[part])
    del records
    model = build_model(config["model_config"],
                        pretrained=config["pretrained"] and checkpoint is None).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=config["lr"], weight_decay=0.0001)
    best_macro_f1 = -1.0
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_macro_f1 = checkpoint["best_macro_f1"]
        restore_random_state(checkpoint["random_state"], generators)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    # Keep the original module for optimizer ownership and portable state dicts.
    # The execution wrapper compiles the complete model, including its classifier.
    runtime_model = compile_model(model, enabled=config["compile"], mode=config["compile_mode"])
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "diagnostics").mkdir(exist_ok=True)
    event_file = "config_start.json" if not checkpoint else (
        f"config_resume_epoch{start_epoch}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json")
    write_json(run_dir / event_file, {"config": config, "provenance": provenance(torch, device),
                                     "resumed_from": str(args.resume) if checkpoint else None})
    metrics_path = run_dir / "diagnostics" / "metrics.jsonl"
    if checkpoint and metrics_path.exists():
        # A crash after a log write but before last.pt must not duplicate epochs.
        kept = []
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if row["epoch"] <= start_epoch:
                kept.append(json.dumps(row, allow_nan=False))
        metrics_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    with tqdm(total=config["epochs"], initial=start_epoch, desc=f"{args.model} epochs",
              unit="epoch", dynamic_ncols=True, disable=not config["progress"],
              position=0) as epochs:
        for epoch in range(start_epoch + 1, config["epochs"] + 1):
            started = time.monotonic()
            training = train_epoch(
                runtime_model, loaders["train"], optimizer, device,
                progress=config["progress"], description=f"train {epoch}/{config['epochs']}",
                position=1)
            validation = score_model(
                runtime_model, loaders["val"], device, config["class_names"],
                progress=config["progress"], description=f"val {epoch}/{config['epochs']}",
                position=1)
            improved = validation["macro_f1"] > best_macro_f1
            best_macro_f1 = max(best_macro_f1, validation["macro_f1"])
            row = {"epoch": epoch, "train": training, "val": validation,
                   "seconds": time.monotonic() - started, "best_macro_f1": best_macro_f1}
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            payload = {"schema_version": 1, "config": config, "epoch": epoch,
                       "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                       "best_macro_f1": best_macro_f1, "random_state": random_state(generators)}
            if improved:
                save_checkpoint(run_dir / "checkpoints" / "best.pt", payload)
            save_checkpoint(run_dir / "checkpoints" / "last.pt", payload)
            if config["progress"]:
                epochs.set_postfix(train_loss=f"{training['loss']:.4f}",
                                   train_acc=f"{training['accuracy']:.4f}",
                                   val_loss=f"{validation['loss']:.4f}",
                                   val_acc=f"{validation['accuracy']:.4f}",
                                   val_f1=f"{validation['macro_f1']:.4f}", refresh=False)
            epochs.update(1)
            tqdm.write(f"epoch {epoch}/{config['epochs']} | train loss {training['loss']:.4f} | "
                       f"train accuracy {training['accuracy']:.4f} | "
                       f"val loss {validation['loss']:.4f} | "
                       f"val accuracy {validation['accuracy']:.4f} | val macro F1 "
                       f"{validation['macro_f1']:.4f}")
            sys.stdout.flush()
    return run_dir

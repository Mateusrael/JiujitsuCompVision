"""Training and validation over the same verified split manifest for all models."""

import hashlib
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm

from src.Dataset.annotations import load_annotations, sha256_file
from src.Eval.metrics import classification_metrics
from src.Loader.datasets import ImageDataset, PoseDataset
from src.Loader.splits import filter_pose_records, read_manifest, split_records
from src.Modules.execution import compile_model
from src.Modules.models import architecture_config, build_model
from src.Modules.registry import IMAGE_MODEL_NAMES, POSE_MODEL_NAMES
from src.Train.checkpoints import (load_checkpoint, random_state, restore_random_state,
                                   save_checkpoint)
from src.Train.config import ATTENTION_DEFAULTS, resolve_run_dir, resolve_settings
from src.Train.run_metadata import provenance, write_json


EVALUATION_TYPES = ("held_out_view", "unseen_moment")
CHECKPOINT_SELECTION = "pooled_validation_macro_f1"


def evaluation_types_for_records(records, manifest):
    """Match evaluation labels to exactly the order consumed by the dataset."""
    if "evaluation_types" not in manifest:
        return None
    mapping = manifest["evaluation_types"]
    if not isinstance(mapping, dict):
        raise ValueError("evaluation_types must map image IDs to evaluation types")
    result = []
    for record in records:
        kind = mapping.get(record["image"])
        if kind not in EVALUATION_TYPES:
            raise ValueError(f"Missing or unsupported evaluation type for {record['image']}")
        result.append(kind)
    return result


def select_model_records(model, records, manifest, partition):
    """Apply model eligibility after the common temporal partition assignment."""
    original = split_records(records, manifest, partition)
    eligible = filter_pose_records(original, manifest=manifest) if model in POSE_MODEL_NAMES else original
    counts = {"original": len(original), "eligible": len(eligible),
              "removed": len(original) - len(eligible)}
    if model in POSE_MODEL_NAMES and "pose_exclusion_reasons" in manifest:
        eligible_ids = {record["image"] for record in eligible}
        counts["removed_by_reason"] = dict(Counter(
            manifest["pose_exclusion_reasons"].get(record["image"], "unspecified")
            for record in original if record["image"] not in eligible_ids))
    if partition != "train" and "evaluation_types" in manifest:
        original_types = evaluation_types_for_records(original, manifest)
        eligible_types = evaluation_types_for_records(eligible, manifest)
        counts["by_evaluation_type"] = {
            kind: {"original": original_types.count(kind), "eligible": eligible_types.count(kind),
                   "removed": original_types.count(kind) - eligible_types.count(kind)}
            for kind in EVALUATION_TYPES}
        if any(not values["eligible"] for values in counts["by_evaluation_type"].values()):
            raise ValueError(f"The {partition} partition has an empty evaluation type after model filtering")
    if not eligible:
        raise ValueError(f"The {partition} partition is empty after model filtering")
    return eligible, counts


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
    if model not in IMAGE_MODEL_NAMES:
        raise ValueError(f"Unknown model: {model!r}")
    return ImageDataset(records, config["class_names"], config["label_map"],
                        config["images_dir"], horizontal_flip_prob=flip_probability)


def make_loader(dataset, config, device, *, training=False, generator=None):
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=training,
                      num_workers=config["workers"], pin_memory=device.type == "cuda",
                      worker_init_fn=seed_worker, generator=generator,
                      persistent_workers=False)


def score_model(model, loader, device, class_names, *, progress=True,
                description="validation", position=0, evaluation_types=None):
    if evaluation_types is not None:
        evaluation_types = list(evaluation_types)
        if len(evaluation_types) != len(loader.dataset):
            raise ValueError("Evaluation types must match the number of evaluation samples")
        if any(kind not in EVALUATION_TYPES for kind in evaluation_types):
            raise ValueError("Unsupported evaluation type")
        if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
            raise ValueError("Grouped evaluation requires a sequential loader with drop_last=False")
    model.eval()
    loss_sum, correct, targets, predictions = 0.0, 0, [], []
    sample_losses = []
    with torch.inference_mode(), tqdm(
            total=len(loader), desc=description, unit="batch", leave=False,
            dynamic_ncols=True, mininterval=0.2, miniters=1,
            disable=not progress, position=position) as batches:
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            losses = torch.nn.functional.cross_entropy(
                logits, labels, reduction="none" if evaluation_types is not None else "sum")
            loss = losses.sum()
            loss_sum += loss.item()
            if evaluation_types is not None:
                sample_losses.extend(losses.cpu().tolist())
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
    if evaluation_types is not None:
        if len(targets) != len(evaluation_types):
            raise ValueError("Evaluation loader did not yield all labeled samples exactly once")
        grouped = {}
        for kind in EVALUATION_TYPES:
            indices = [index for index, value in enumerate(evaluation_types) if value == kind]
            if not indices:
                continue
            group = classification_metrics([targets[index] for index in indices],
                                           [predictions[index] for index in indices], class_names)
            group["loss"] = sum(sample_losses[index] for index in indices) / len(indices)
            grouped[kind] = group
        metrics["by_evaluation_type"] = grouped
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
    if model in IMAGE_MODEL_NAMES:
        options["image_training"] = config.get("image_training")
    return architecture_config(model, num_classes, fine_tune=config["fine_tune"],
                               dropout=config["dropout"], **options)


def validate_checkpoint_data(config, model, annotations_path, split_path, manifest):
    if model in POSE_MODEL_NAMES and config.get("pose_filter") != "both_present":
        raise ValueError("This pose checkpoint predates the both-present pose filter; "
                         "start a fresh training run with the current data policy")
    expected = {"model": model, "annotation_sha256": sha256_file(annotations_path),
                "split_sha256": sha256_file(split_path),
                "class_names": manifest["class_names"], "label_map": manifest["label_map"]}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Checkpoint {key} does not match this run's data/model/split")
    model_config = configured_architecture(model, len(manifest["class_names"]), config)
    if config.get("model_config") != model_config:
        raise ValueError("Checkpoint architecture/preprocessing configuration is unsupported")


def validate_initialization_architecture(source, target):
    """A new image-training stage may change trainability and head dropout only."""
    allowed = {"fine_tune", "image_training", "dropout"}
    source_architecture = {key: value for key, value in source.items() if key not in allowed}
    target_architecture = {key: value for key, value in target.items() if key not in allowed}
    if source_architecture != target_architecture:
        raise ValueError("--init-from requires matching architecture and preprocessing")


def train(args):
    init_from = getattr(args, "init_from", None)
    if args.resume and init_from:
        raise ValueError("--resume and --init-from are mutually exclusive")
    run_dir = resolve_run_dir(args.run_dir, args.resume)
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    previous = checkpoint["config"] if checkpoint else None
    initialization = None
    if init_from:
        init_from = Path(init_from).expanduser().resolve()
        if init_from.name not in ("best.pt", "last.pt") or init_from.parent.name != "checkpoints":
            raise ValueError("--init-from must use <run-dir>/checkpoints/best.pt or last.pt")
        # Training replaces checkpoint files atomically. Keep one open handle so
        # the recorded digest identifies exactly the weights loaded into memory.
        with init_from.open("rb") as handle:
            initialization = load_checkpoint(handle)
            handle.seek(0)
            initialization_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        if initialization["config"].get("model") not in IMAGE_MODEL_NAMES:
            raise ValueError("--init-from supports image checkpoints only")
    source_config = previous if checkpoint else (initialization or {}).get("config")
    for name in ("model", "annotations", "split"):
        if getattr(args, name) is None:
            saved = (source_config or {}).get(name)
            if saved is None:
                raise ValueError(f"A fresh run requires --{name}")
            setattr(args, name, saved)
    if initialization and args.model not in IMAGE_MODEL_NAMES:
        raise ValueError("--init-from supports image models only")
    if initialization:
        # Weight origin describes the copied artifact, not a new optimization choice.
        pretrained = source_config.get("pretrained")
        if not isinstance(pretrained, bool):
            raise ValueError("Initialization checkpoint has no valid pretrained weight origin")
        if getattr(args, "pretrained", None) is not None and args.pretrained != pretrained:
            raise ValueError("Cannot change --pretrained/--no-pretrained when using --init-from")
        args.pretrained = pretrained
    # Warm starts intentionally use new-run defaults and a fresh random seed.
    # Only --resume restores the source run's optimization settings and history.
    config = resolve_settings(args, previous)
    device = resolve_device(args.device or (previous or {}).get("device", "cuda"))
    print(f"Preparing {args.model} data and model on {device}...", flush=True)
    annotations_path = Path(args.annotations).expanduser().resolve()
    split_path = Path(args.split).expanduser().resolve()
    records = load_annotations(annotations_path)
    manifest = read_manifest(split_path, annotations_path, records)
    if source_config:
        validate_checkpoint_data(source_config, args.model, annotations_path, split_path, manifest)
    images_dir = args.images_dir or (source_config or {}).get("images_dir")
    if args.model in IMAGE_MODEL_NAMES and not images_dir:
        raise ValueError("--images-dir is required for image models")
    images_dir = str(Path(images_dir).expanduser().resolve()) if images_dir else None
    if source_config and images_dir != source_config.get("images_dir"):
        raise ValueError("Cannot change the image directory when resuming or using --init-from")
    config.update({"schema_version": 1, "model": args.model,
                   "model_config": configured_architecture(args.model, len(manifest["class_names"]), config),
                   "annotations": str(annotations_path), "split": str(split_path),
                   "annotation_sha256": sha256_file(annotations_path),
                   "split_sha256": sha256_file(split_path),
                   "split_method": manifest["split_method"],
                   "checkpoint_selection": CHECKPOINT_SELECTION,
                   "pose_filter": "both_present" if args.model in POSE_MODEL_NAMES else "not_applicable",
                   "evaluation_scope": manifest.get("evaluation_scope", "Held-out recording groups."),
                   "class_names": manifest["class_names"], "label_map": manifest["label_map"],
                   "images_dir": images_dir, "device": str(device), "run_dir": str(run_dir)})
    if initialization:
        validate_initialization_architecture(source_config["model_config"], config["model_config"])
        config["initialized_from"] = {"checkpoint": str(init_from),
                                      "checkpoint_sha256": initialization_sha256,
                                      "source_epoch": initialization["epoch"]}
    start_epoch = checkpoint["epoch"] if checkpoint else 0
    if config["epochs"] <= start_epoch:
        raise ValueError(f"--epochs is the total target; choose a value above {start_epoch}")
    seed_everything(config["seed"])
    generators = {part: torch.Generator().manual_seed(config["seed"] + index)
                  for index, part in enumerate(("train", "val"))}
    loaders = {}
    validation_types = None
    config["sample_counts"] = {}
    for part in ("train", "val", "test"):
        selected_records, counts = select_model_records(args.model, records, manifest, part)
        config["sample_counts"][part] = counts
        print(f"{part}: {counts['eligible']:,} eligible / {counts['original']:,} assigned "
              f"({counts['removed']:,} removed by model filter)", flush=True)
        if part == "test":
            continue
        if part == "val":
            validation_types = evaluation_types_for_records(selected_records, manifest)
        dataset = make_dataset(args.model, selected_records, config, training=part == "train")
        loaders[part] = make_loader(dataset, config, device, training=part == "train",
                                    generator=generators[part])
    del records, selected_records
    model = build_model(config["model_config"],
                        pretrained=config["pretrained"] and checkpoint is None
                        and initialization is None).to(device)
    if args.model in IMAGE_MODEL_NAMES:
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        config["parameter_counts"] = {"trainable": trainable, "frozen": total - trainable, "total": total}
        print(f"Image training: {config['image_training']} | "
              f"{trainable:,} trainable / {total:,} total parameters", flush=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=config["lr"], weight_decay=config["weight_decay"])
    best_macro_f1 = -1.0
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_macro_f1 = checkpoint["best_macro_f1"]
        restore_random_state(checkpoint["random_state"], generators)
    else:
        if initialization:
            model.load_state_dict(initialization["model_state"], strict=True)
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
                                     "resumed_from": str(args.resume) if checkpoint else None,
                                     "initialized_from": config.get("initialized_from")})
    metrics_path = run_dir / "diagnostics" / "metrics.jsonl"
    print("Best checkpoint selection: pooled validation macro F1.", flush=True)
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
                position=1, evaluation_types=validation_types)
            improved = validation["macro_f1"] > best_macro_f1
            best_macro_f1 = max(best_macro_f1, validation["macro_f1"])
            row = {"epoch": epoch, "train": training, "val": validation,
                   "seconds": time.monotonic() - started, "best_macro_f1": best_macro_f1,
                   "checkpoint_selection": CHECKPOINT_SELECTION}
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            payload = {"schema_version": 1, "config": config, "epoch": epoch,
                       "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                       "best_macro_f1": best_macro_f1, "random_state": random_state(generators)}
            if improved:
                save_checkpoint(run_dir / "checkpoints" / "best.pt", payload)
            save_checkpoint(run_dir / "checkpoints" / "last.pt", payload)
            if config["progress"]:
                group_scores = {f"val_{kind}_f1": f"{scores['macro_f1']:.4f}"
                                for kind, scores in validation.get("by_evaluation_type", {}).items()}
                epochs.set_postfix(train_loss=f"{training['loss']:.4f}",
                                   train_acc=f"{training['accuracy']:.4f}",
                                   val_loss=f"{validation['loss']:.4f}",
                                   val_acc=f"{validation['accuracy']:.4f}",
                                   val_f1=f"{validation['macro_f1']:.4f}",
                                   **group_scores, refresh=False)
            epochs.update(1)
            group_text = "".join(f" | val {kind} macro F1 {scores['macro_f1']:.4f}"
                                 for kind, scores in validation.get("by_evaluation_type", {}).items())
            tqdm.write(f"epoch {epoch}/{config['epochs']} | train loss {training['loss']:.4f} | "
                       f"train accuracy {training['accuracy']:.4f} | "
                       f"val loss {validation['loss']:.4f} | "
                       f"val accuracy {validation['accuracy']:.4f} | val macro F1 "
                       f"{validation['macro_f1']:.4f}{group_text}")
            sys.stdout.flush()
    return run_dir

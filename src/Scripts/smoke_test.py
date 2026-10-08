"""Small synthetic forward/backward checks; no dataset or weight downloads."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.Modules.registry import MODEL_NAMES, POSE_MODEL_NAMES
from src.Modules.execution import COMPILE_MODES, compile_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODEL_NAMES, "both", "all"), default="all",
                        help="Default: all four models; both retains the original pose/image check")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                        help="Exercise whole-model compilation; default: disabled")
    parser.add_argument("--compile-mode", choices=COMPILE_MODES, default="default")
    args = parser.parse_args(argv)
    if not args.compile and args.compile_mode != "default":
        parser.error("A nondefault --compile-mode requires --compile")
    import torch
    from src.Modules.models import architecture_config, build_model
    from src.Train.engine import resolve_device, seed_everything
    torch.set_num_threads(1)
    seed_everything(42)
    device = resolve_device(args.device)
    names = MODEL_NAMES if args.model == "all" else (
        ("pose", "image") if args.model == "both" else (args.model,))
    for name in names:
        model = build_model(architecture_config(name), pretrained=False).to(device).train()
        if name in POSE_MODEL_NAMES:
            joints = torch.randn((2, 34, 3), device=device)
            joints[..., 2] = torch.rand((2, 34), device=device)
            joints[0, 17:] = 0  # Exercise a missing athlete without changing the common input contract.
            features = joints.flatten(1)
        else:
            features = torch.randn((2, 3, 224, 224), device=device)
        labels = torch.tensor([0, 1], device=device)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.001)
        runtime_model = compile_model(model, enabled=args.compile, mode=args.compile_mode)
        logits = runtime_model(features)
        assert logits.shape == (2, 10)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss for {name}")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        if not gradients or any(g is None or not torch.isfinite(g).all() for g in gradients):
            raise RuntimeError(f"Missing/nonfinite gradients for {name}")
        optimizer.step()
        runtime_model.eval()
        with torch.inference_mode():
            evaluation_logits = runtime_model(features)
        if evaluation_logits.shape != (2, 10) or not torch.isfinite(evaluation_logits).all():
            raise RuntimeError(f"Invalid evaluation logits for {name}")
        print(f"PASS {name}: device={device}, logits={tuple(logits.shape)}, loss={loss.item():.4f}")


if __name__ == "__main__":
    main()

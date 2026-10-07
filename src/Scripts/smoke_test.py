"""Small synthetic forward/backward checks; no dataset or weight downloads."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pose", "image", "both"), default="both")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    import torch
    from src.Modules.models import architecture_config, build_model
    from src.Train.engine import resolve_device, seed_everything
    torch.set_num_threads(1)
    seed_everything(42)
    device = resolve_device(args.device)
    for name in (("pose", "image") if args.model == "both" else (args.model,)):
        model = build_model(architecture_config(name), pretrained=False).to(device).train()
        features = torch.randn((2, 102) if name == "pose" else (2, 3, 224, 224), device=device)
        labels = torch.tensor([0, 1], device=device)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.001)
        logits = model(features)
        assert logits.shape == (2, 10)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss for {name}")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        if not gradients or any(g is None or not torch.isfinite(g).all() for g in gradients):
            raise RuntimeError(f"Missing/nonfinite gradients for {name}")
        optimizer.step()
        print(f"PASS {name}: device={device}, logits={tuple(logits.shape)}, loss={loss.item():.4f}")


if __name__ == "__main__":
    main()

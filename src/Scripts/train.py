"""Train a pose or image classifier; --help works before PyTorch is installed."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.Modules.registry import MODEL_NAMES
from src.Modules.execution import COMPILE_MODES


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, help="Required for a fresh run")
    parser.add_argument("--annotations", help="Required for a fresh run; restored on resume")
    parser.add_argument("--split", help="Shared split manifest; restored on resume")
    parser.add_argument("--images-dir", help="Directory containing seven-digit image filenames")
    parser.add_argument("--run-dir", help="New run directory; required unless resuming")
    parser.add_argument("--resume", help="Resume this run's checkpoints/last.pt (trusted files only)")
    parser.add_argument("--epochs", type=int, help="Total target epochs; fresh default: 20")
    parser.add_argument("--batch-size", type=int, help="Fresh default: 128")
    parser.add_argument("--lr", type=float, help="Fresh default: 0.001")
    parser.add_argument("--weight-decay", type=float,
                        help="AdamW weight decay; finite and nonnegative; fresh default: 0.0001")
    parser.add_argument("--workers", type=int, help="DataLoader workers; fresh default: 2")
    parser.add_argument("--seed", type=int, help="Fresh default: 42; preserved on resume")
    parser.add_argument("--device", choices=("cuda", "cpu"),
                        help="Fresh default: cuda; use cpu explicitly for small tests")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                        help="Compile the whole model with torch.compile; fresh default: disabled")
    parser.add_argument("--compile-mode", choices=COMPILE_MODES,
                        help="torch.compile mode; fresh default: default; requires compilation")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=None,
                        help="Show epoch and batch progress bars; fresh default: enabled")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None,
                        help="ImageNet ResNet-18 weights (default); --no-pretrained permits offline smoke")
    parser.add_argument("--fine-tune", action=argparse.BooleanOptionalAction, default=None,
                        help="Train image backbone too; by default train only its classifier head")
    parser.add_argument("--swap-athletes", action=argparse.BooleanOptionalAction, default=None,
                        help="Randomly swap pose slots during training (default: enabled for 10 classes)")
    parser.add_argument("--horizontal-flip-prob", type=float,
                        help="Training-only horizontal mirror probability for poses/images; default: 0 (off)")
    parser.add_argument("--dropout", type=float,
                        help="MLP hidden/image head dropout, or fallback for both attention branches; default: 0")
    attention = parser.add_argument_group("pose-attention architecture (saved and restored on resume)")
    attention.add_argument("--attention-dim", type=int, help="Token embedding width; default: 128")
    attention.add_argument("--attention-heads", type=int, help="Attention heads per block; default: 4")
    attention.add_argument("--attention-layers", type=int, help="Residual attention/MLP blocks; default: 4")
    attention.add_argument("--attention-mlp-dim", type=int, help="Hidden width inside each block; default: 512")
    attention.add_argument("--attention-dropout", type=float,
                           help="Attention weight and attention residual-output dropout; overrides --dropout")
    attention.add_argument("--attention-mlp-dropout", type=float,
                           help="Hidden and output dropout inside each attention block's MLP; overrides --dropout")
    attention.add_argument("--attention-pooling", choices=("mean", "cls"),
                           help="Pool observed joints, or use an extra learned classification token; default: mean")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from src.Train.engine import train
    run_dir = train(args)
    print(f"Run saved in {run_dir}")


if __name__ == "__main__":
    main()

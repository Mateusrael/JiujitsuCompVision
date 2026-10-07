"""Train either baseline; --help works before PyTorch is installed."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pose", "image"), help="Required for a fresh run")
    parser.add_argument("--annotations", help="Required for a fresh run; restored on resume")
    parser.add_argument("--split", help="Shared recording-group manifest; restored on resume")
    parser.add_argument("--images-dir", help="Directory containing seven-digit image filenames")
    parser.add_argument("--run-dir", help="New run directory; required unless resuming")
    parser.add_argument("--resume", help="Resume this run's checkpoints/last.pt (trusted files only)")
    parser.add_argument("--epochs", type=int, help="Total target epochs; fresh default: 20")
    parser.add_argument("--batch-size", type=int, help="Fresh default: 128")
    parser.add_argument("--lr", type=float, help="Fresh default: 0.001")
    parser.add_argument("--workers", type=int, help="DataLoader workers; fresh default: 2")
    parser.add_argument("--seed", type=int, help="Fresh default: 42; preserved on resume")
    parser.add_argument("--device", choices=("cuda", "cpu"),
                        help="Fresh default: cuda; use cpu explicitly for small tests")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None,
                        help="ImageNet ResNet-18 weights (default); --no-pretrained permits offline smoke")
    parser.add_argument("--fine-tune", action=argparse.BooleanOptionalAction, default=None,
                        help="Train image backbone too; by default train only its classifier head")
    parser.add_argument("--swap-athletes", action=argparse.BooleanOptionalAction, default=None,
                        help="Randomly swap pose slots during training (default: enabled for 10 classes)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from src.Train.engine import train
    run_dir = train(args)
    print(f"Run saved in {run_dir}")


if __name__ == "__main__":
    main()

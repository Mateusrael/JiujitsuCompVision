"""Evaluate a checkpoint on the shared manifest's held-out test partition."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.Modules.execution import COMPILE_MODES


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Use best.pt selected on validation")
    parser.add_argument("--annotations", help="Defaults to the checkpoint's annotation path")
    parser.add_argument("--split", help="Defaults to the checkpoint's split manifest path")
    parser.add_argument("--images-dir", help="Image directory; defaults to checkpoint setting")
    parser.add_argument("--output", help="Default: <run>/diagnostics/test_metrics.json")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                        help="Compile the whole model for evaluation; default: disabled")
    parser.add_argument("--compile-mode", choices=COMPILE_MODES, default="default",
                        help="torch.compile mode; a nondefault mode requires --compile")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                        help="Show evaluation batch progress; default: enabled")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from src.Eval.evaluate import evaluate
    evaluate(args)


if __name__ == "__main__":
    main()

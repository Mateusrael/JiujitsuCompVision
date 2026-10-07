"""Audit annotations without importing PyTorch or downloading images."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.Dataset.audit import audit_annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.resolve() == args.annotations.resolve():
            raise ValueError("Report output must not overwrite annotations.")
        report = audit_annotations(args.annotations, args.images_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return int(bool(report["invalid_records"] or report["duplicate_image_ids"]
                        or report["missing_image_files"] or report["ambiguous_image_files"]))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Audit failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

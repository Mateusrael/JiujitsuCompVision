"""Prepare one immutable image assignment manifest for both baselines."""

import argparse
import json
from pathlib import Path

from src.Dataset.annotations import load_annotations, sha256_file
from src.Dataset.splits import build_manifest, write_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Split by verified recordings, keeping synchronized cameras together."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True,
                        help="JSON with mapping evidence and sequence recording/split assignments")
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON manifest path; existing files are never overwritten")
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError(f"Manifest already exists: {args.output}. Choose a new path.")
        digest = sha256_file(args.annotations)
        records = load_annotations(args.annotations)
        if sha256_file(args.annotations) != digest:
            raise ValueError("Annotations changed while reading; retry with an unchanged file.")
        groups = json.loads(args.groups.read_text(encoding="utf-8"))
        manifest = build_manifest(records, groups, digest)
        write_manifest(manifest, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Saved {args.output}: {manifest['counts']}")


if __name__ == "__main__":
    main()

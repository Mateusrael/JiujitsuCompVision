"""Prepare ViCoS data on the remote compute environment; annotations by default."""

import argparse
import http.client
import json
import sys
from pathlib import Path
import zipfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.Dataset.download import prepare_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "Data")
    parser.add_argument("--images", action="store_true", help="Also download images.zip and extract it under DATA/images.")
    parser.add_argument("--extract-only", action="store_true", help="Extract an existing DATA/images.zip without network access.")
    parser.add_argument("--timeout", type=float, default=60, help="Network socket timeout in seconds (default: 60).")
    args = parser.parse_args()
    if not 0 < args.timeout <= 600:
        parser.error("--timeout must be greater than 0 and at most 600 seconds")
    try:
        report = prepare_data(args.data_dir, args.images, args.extract_only, args.timeout)
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError, http.client.HTTPException) as exc:
        parser.exit(1, f"Data preparation failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

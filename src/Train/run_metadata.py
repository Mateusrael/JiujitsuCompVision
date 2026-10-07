"""Readable run configuration and runtime provenance."""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def provenance(torch, device):
    repository = Path(__file__).resolve().parents[2]
    git = {"commit": None, "dirty": None}
    try:
        git["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True,
            text=True, check=True, timeout=5).stdout.strip()
        git["dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=repository, capture_output=True,
            text=True, check=True, timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "torch": torch.__version__,
            "platform": platform.platform(), "git": git,
            "device": str(device), "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}

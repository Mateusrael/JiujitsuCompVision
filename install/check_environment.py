#!/usr/bin/env python3
"""Standalone, standard-library-only probe; no installs, downloads, or training."""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request


PROJECT_DIR = Path(__file__).resolve().parent
if PROJECT_DIR.name == "install":
    PROJECT_DIR = PROJECT_DIR.parent
NETWORK_URLS = {
    "github": "https://github.com/",
    "annotations": "https://data.vicos.si/datasets/JuiJuitsu/annotations.json",
}
TORCH_CODE = r'''
import importlib, json, sys
result = {"imports": {}}
for name in ("torch", "torchvision"):
    try:
        module = importlib.import_module(name)
        result["imports"][name] = {"ok": True, "version": str(module.__version__)}
    except Exception as exc:
        result["imports"][name] = {"ok": False, "error": str(exc)}
result["ok"] = all(item["ok"] for item in result["imports"].values())
result["cuda"] = {"available": False}
if result["imports"]["torch"]["ok"]:
    import torch
    try:
        available = torch.cuda.is_available()
        result["cuda"] = {"available": available, "build": torch.version.cuda,
                          "devices": [torch.cuda.get_device_name(i) for i in
                                      range(torch.cuda.device_count())] if available else []}
    except Exception as exc:
        result["cuda"]["error"] = str(exc)
if sys.argv[1]:
    device = sys.argv[1]
    try:
        import torch
        if device == "cuda" and not result["cuda"]["available"]:
            raise RuntimeError("CUDA requested but unavailable; obtain a GPU allocation and check the environment.")
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2).to(device)
        loss = model(torch.ones(2, 4, device=device)).square().mean()
        loss.backward()
        if not torch.isfinite(loss).item() or not all(
            p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()
        ):
            raise RuntimeError("Non-finite loss or gradients.")
        result["smoke_test"] = {"ok": True, "device": device, "loss": loss.item()}
    except Exception as exc:
        result["smoke_test"] = {"ok": False, "device": device, "error": str(exc)}
print("VISAOCOMP_JSON=" + json.dumps(result))
'''


def run_command(command, timeout):
    """Contain missing executables, import crashes, and commands that hang."""
    try:
        process = subprocess.run(command, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=timeout)
        return {"ok": process.returncode == 0, "returncode": process.returncode,
                "stdout": process.stdout[-12000:], "stderr": process.stderr[-6000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}


def probe_torch(timeout, smoke_test=None):
    result = run_command([sys.executable, "-B", "-c", TORCH_CODE, smoke_test or ""], timeout)
    if not result["ok"]:
        return result
    for line in reversed(result["stdout"].splitlines()):
        if line.startswith("VISAOCOMP_JSON="):
            try:
                parsed = json.loads(line.partition("=")[2])
                if result["stderr"]:
                    parsed["stderr"] = result["stderr"]
                return parsed
            except json.JSONDecodeError:
                break
    return {**result, "ok": False, "error": "PyTorch subprocess returned no valid report."}


def disk_report(path):
    requested = Path(path).expanduser().resolve()
    measured = requested
    while not measured.exists() and measured != measured.parent:
        measured = measured.parent
    if measured.is_file():
        measured = measured.parent
    result = {"requested_path": str(requested), "exists": requested.exists(),
              "measured_path": str(measured),
              "note": "Filesystem space only; user quota and storage persistence are unknown."}
    try:
        result.update(shutil.disk_usage(measured)._asdict())
    except OSError as exc:
        result["error"] = str(exc)
    return result


def network_report(url, timeout):
    try:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "VisaoComp-environment-probe"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {"ok": True, "url": url, "status": response.status,
                    "content_length": response.headers.get("Content-Length")}
    except (OSError, urllib.error.URLError) as exc:
        return {"ok": False, "url": url, "error": str(exc)}


def write_report(path, text):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "Data")
    parser.add_argument("--output", type=Path, help="Atomically save the JSON report (also printed).")
    parser.add_argument("--check-network", action="store_true", help="Send HEAD requests to GitHub and ViCoS; no dataset download.")
    parser.add_argument("--require-torch", action="store_true", help="Fail if torch or torchvision cannot import.")
    parser.add_argument("--require-cuda", action="store_true", help="Fail if PyTorch cannot access CUDA.")
    parser.add_argument("--smoke-test", choices=("cpu", "cuda"), help="Explicitly run a tiny forward/backward test on this device.")
    parser.add_argument("--timeout", type=float, default=30, help="Timeout in seconds per subprocess/network request (default: 30).")
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 600:
        parser.error("--timeout must be greater than 0 and at most 600 seconds")
    report = {
        "python": {"version": platform.python_version(), "executable": sys.executable,
                   "supported": sys.version_info >= (3, 10)},
        "platform": platform.platform(),
        "git": run_command(["git", "--version"], args.timeout),
        "nvidia_smi": run_command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], args.timeout),
        "storage": {"project": disk_report(PROJECT_DIR), "data": disk_report(args.data_dir)},
        "pytorch": probe_torch(args.timeout, args.smoke_test),
        "network": {name: network_report(url, args.timeout) for name, url in NETWORK_URLS.items()} if args.check_network else {"checked": False},
        "errors": [],
    }
    if not report["python"]["supported"]:
        report["errors"].append("Python 3.10 or newer is required.")
    if args.require_torch and not report["pytorch"]["ok"]:
        report["errors"].append("torch and torchvision must both import successfully; inspect pytorch in this report.")
    if args.require_cuda and not report["pytorch"].get("cuda", {}).get("available"):
        report["errors"].append("CUDA is required but unavailable to PyTorch; check your GPU allocation and environment.")
    if args.smoke_test and not report["pytorch"].get("smoke_test", {}).get("ok"):
        report["errors"].append("The requested forward/backward smoke test failed; inspect pytorch in this report.")
    report["ok"] = not report["errors"]
    encoded = json.dumps(report, indent=2) + "\n"
    print(encoded, end="")
    if args.output:
        try:
            write_report(args.output, encoded)
        except OSError as exc:
            print(f"Could not save report: {exc}", file=sys.stderr)
            return 2
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

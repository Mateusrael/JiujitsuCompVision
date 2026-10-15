"""GPU discovery and selection without initializing CUDA in the queue process."""

import csv
import io
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    name: str
    total_mib: int
    free_mib: int
    utilization: int


_UUID = re.compile(r"(?:GPU-)?([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})\Z")
_PROBE_MARKER = "JIUJITSU_VISIBLE_GPU_UUIDS="
_PROBE = """
import json
import torch

devices = []
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    devices.append(str(getattr(properties, 'uuid', '')))
print('JIUJITSU_VISIBLE_GPU_UUIDS=' + json.dumps(devices))
"""


def _normalize_uuid(value: str) -> str:
    match = _UUID.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"Unsupported or missing GPU UUID: {value!r}")
    return "GPU-" + match.group(1).lower()


def _run(command: list[str], *, cwd: Path | None = None,
         timeout: int = 30) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                                timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"GPU discovery command failed: {command[0]}: {error}") from error
    if result.returncode:
        detail = (result.stderr or result.stdout or "no error output").strip()
        raise RuntimeError(f"GPU discovery command failed ({result.returncode}): "
                           f"{command[0]}: {detail}")
    return result


def discover_allowed_gpus(python: str, project_root: Path) -> set[str]:
    """Read actual PyTorch-visible UUIDs in an isolated child process.

    The child's environment is inherited untouched. This respects CUDA masks,
    including reordered device indices, and container device restrictions. The
    parent never imports torch or establishes its own CUDA context. In
    particular, NVIDIA_VISIBLE_DEVICES is not treated as an additional mask:
    the devices actually exposed to PyTorch are authoritative.
    """
    result = _run([str(python), "-c", _PROBE], cwd=project_root, timeout=60)
    records = [line[len(_PROBE_MARKER):] for line in result.stdout.splitlines()
               if line.startswith(_PROBE_MARKER)]
    if len(records) != 1:
        raise RuntimeError("PyTorch GPU probe did not return one GPU UUID list")
    try:
        values = json.loads(records[0])
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError("Expected a list of GPU UUID strings")
        normalized = {_normalize_uuid(value) for value in values}
        if len(normalized) != len(values):
            raise ValueError("PyTorch GPU probe returned duplicate GPU UUIDs")
    except (ValueError, TypeError) as error:
        raise RuntimeError(f"Cannot determine permitted CUDA devices: {error}") from error
    return normalized


def _integer(value: str, field: str, line: int) -> int:
    value = value.strip()
    if not value.isascii() or not value.isdecimal():
        raise RuntimeError(f"Invalid nvidia-smi {field} on row {line}: {value!r}")
    return int(value)


def query_gpus() -> list[GPU]:
    """Read current telemetry; skip GPUs with prohibited or unavailable data.

    Other malformed telemetry is an error, rather than being interpreted as
    free memory or idle compute. The returned indices are physical nvidia-smi
    indices; the scheduler must launch a child by UUID, not by these indices.
    """
    result = _run([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,compute_mode",
        "--format=csv,noheader,nounits",
    ])
    gpus = []
    seen_indices = set()
    seen_uuids = set()
    try:
        rows = csv.reader(io.StringIO(result.stdout), skipinitialspace=True, strict=True)
        for number, row in enumerate(rows, 1):
            if not row:
                continue
            if len(row) != 7:
                raise RuntimeError(f"Expected 7 nvidia-smi fields on row {number}, got {len(row)}")
            values = [value.strip() for value in row]
            if any(value.upper() in {"N/A", "[N/A]", "NOT SUPPORTED", "[NOT SUPPORTED]"}
                   for value in values):
                continue
            index, raw_uuid, name, total, free, utilization, mode = values
            if mode.lower() == "prohibited":
                continue
            if mode.lower() not in {"default", "exclusive_process", "exclusive process",
                                    "exclusive_thread", "exclusive thread"}:
                raise RuntimeError(f"Unknown nvidia-smi compute mode on row {number}: {mode!r}")
            try:
                uuid = _normalize_uuid(raw_uuid)
            except ValueError as error:
                raise RuntimeError(f"Invalid nvidia-smi UUID on row {number}: {error}") from error
            gpu = GPU(index=_integer(index, "index", number), uuid=uuid, name=name,
                      total_mib=_integer(total, "memory.total", number),
                      free_mib=_integer(free, "memory.free", number),
                      utilization=_integer(utilization, "utilization.gpu", number))
            if not name or gpu.total_mib <= 0 or gpu.free_mib > gpu.total_mib or gpu.utilization > 100:
                raise RuntimeError(f"Out-of-range nvidia-smi telemetry on row {number}")
            if gpu.index in seen_indices or gpu.uuid in seen_uuids:
                raise RuntimeError(f"Duplicate nvidia-smi GPU on row {number}")
            seen_indices.add(gpu.index)
            seen_uuids.add(gpu.uuid)
            gpus.append(gpu)
    except csv.Error as error:
        raise RuntimeError(f"Malformed nvidia-smi CSV: {error}") from error
    return gpus


def choose_gpu(gpus: list[GPU], allowed: set[str], busy: set[str],
               min_free_mib: int, max_utilization: int = 100) -> GPU | None:
    """Select an allowed, unreserved GPU by utilization, free memory, and index."""
    if isinstance(min_free_mib, bool) or not isinstance(min_free_mib, int) or min_free_mib < 0:
        raise ValueError("min_free_mib must be a nonnegative integer")
    if (isinstance(max_utilization, bool) or not isinstance(max_utilization, int)
            or not 0 <= max_utilization <= 100):
        raise ValueError("max_utilization must be an integer between 0 and 100")
    eligible = [gpu for gpu in gpus if gpu.uuid in allowed and gpu.uuid not in busy
                and gpu.free_mib >= min_free_mib and gpu.utilization <= max_utilization]
    return min(eligible, key=lambda gpu: (gpu.utilization, -gpu.free_mib, gpu.index), default=None)

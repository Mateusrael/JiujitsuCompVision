#!/usr/bin/env bash
# Run on the Linux compute environment. Never installs drivers or runs training.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname -- "$script_dir")"
venv="$project_dir/.venv"
torch_mode=cu128
python_cmd="${PYTHON:-python3}"
die() { printf '%s\n' "$*" >&2; exit 1; }
usage() {
    cat <<'EOF'
Usage: bash install/install.sh [--torch existing|cpu|cu126|cu128] [--venv PATH]

Default: create an isolated .venv and install PyTorch 2.11.0 + torchvision 0.26.0
from the CUDA 12.8 wheel index. Python 3.10+ is required; Python 3.11 is supported.
Set PYTHON to the Python executable used to create this environment if needed.

Optional cpu/cu126 modes install the same pinned package versions for that build.
Explicit --torch existing reuses lab packages via --system-site-packages instead
of installing the pinned stack. A nested venv cannot inherit another ordinary
virtualenv's packages; use that interpreter directly through PYTHON if necessary.

Examples (run remotely):
  bash install/install.sh
  PYTHON=/path/to/python bash install/install.sh --venv /my/storage/visao-env
  bash install/install.sh --torch cu128 --venv .venv-cu128

Reruns preserve the selected mode. Use a fresh --venv path to change modes.
No data, model weights, CUDA toolkit, drivers, or training are included.
EOF
}
while (($#)); do
    case "$1" in
        --torch) (($# >= 2)) || die "--torch needs a mode"; torch_mode="$2"; shift 2 ;;
        --venv) (($# >= 2)) || die "--venv needs a path"; venv="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) die "Unknown option: $1 (use --help)" ;;
    esac
done
case "$torch_mode" in existing|cpu|cu126|cu128) ;; *) die "Unsupported --torch mode: $torch_mode" ;; esac
[[ "$(uname -s)" == Linux ]] || die "Run this installer on the Linux DGX environment."
command -v "$python_cmd" >/dev/null || die "Python not found: $python_cmd"
"$python_cmd" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Python 3.10 or newer is required.")'
[[ -n "$venv" ]] || die "--venv cannot be empty"
venv="$("$python_cmd" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$venv")"
marker="$venv/.visaocomp-torch-mode"
[[ ! -L "$venv" ]] || die "Refusing a symlink as --venv; use a fresh directory."
if [[ -e "$venv" ]]; then
    [[ -d "$venv" && -O "$venv" && -f "$marker" && ! -L "$marker" ]] || die "Existing path is not a user-owned VisaoComp environment; choose a fresh --venv."
    [[ "$(cat -- "$marker")" == "$torch_mode" ]] || die "Environment mode differs; choose a fresh --venv instead of mixing builds."
    [[ -x "$venv/bin/python" && -f "$venv/pyvenv.cfg" ]] || die "Incomplete virtual environment; choose a fresh --venv."
else
    venv_options=()
    if [[ "$torch_mode" == existing ]]; then
        "$python_cmd" - <<'PY'
import sys
if sys.prefix != sys.base_prefix:
    raise SystemExit(
        "The selected Python is already inside an ordinary virtualenv. A nested "
        "--system-site-packages environment would not inherit its installed torch. "
        "Use this approved interpreter directly: set PYTHON=" + sys.executable +
        " in launchers/local_config.sh and run the launchers without install.sh. "
        "Alternatively, choose a fresh --venv path with an explicit --torch build."
    )
PY
        "$python_cmd" "$script_dir/check_environment.py" --require-torch
        venv_options+=(--system-site-packages)
    fi
    "$python_cmd" -m venv "${venv_options[@]}" "$venv"
    printf '%s\n' "$torch_mode" > "$marker"
fi
if [[ "$torch_mode" != existing ]]; then
    "$venv/bin/python" -m pip --isolated install --require-virtualenv --no-cache-dir \
        --index-url "https://download.pytorch.org/whl/$torch_mode" \
        -r "$script_dir/requirements-torch.txt"
    "$venv/bin/python" - "$torch_mode" "$script_dir/requirements-torch.txt" <<'PY'
import sys
from pathlib import Path
import torch
import torchvision

expected = dict(line.strip().split("==", 1) for line in
                Path(sys.argv[2]).read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#"))
for package in (torch, torchvision):
    actual = package.__version__.split("+", 1)[0]
    if actual != expected[package.__name__]:
        raise SystemExit(f"Unexpected {package.__name__} version: {package.__version__}")
expected_cuda = {"cpu": None, "cu126": "12.6", "cu128": "12.8"}[sys.argv[1]]
if torch.version.cuda != expected_cuda:
    raise SystemExit(f"Expected CUDA runtime {expected_cuda}, found {torch.version.cuda}")
print(f"Verified torch {torch.__version__}, torchvision {torchvision.__version__}, CUDA {torch.version.cuda}")
PY
fi
"$venv/bin/python" -m pip --isolated install --require-virtualenv --no-cache-dir \
    --index-url "https://pypi.org/simple" -r "$script_dir/requirements-runtime.txt"
"$venv/bin/python" "$script_dir/check_environment.py" --require-torch
printf '\nSetup complete. Activate with:\n  source %q/bin/activate\n' "$venv"
printf 'Run the environment probe inside your approved GPU allocation before using CUDA.\n'

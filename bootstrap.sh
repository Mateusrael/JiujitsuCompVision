#!/usr/bin/env bash
# Prepare the uploaded Linux project. Training is a separate command.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_ROOT/launchers/project_config.sh"
torch_mode=cu128
download=none
check_only=false
while (($#)); do
    case "$1" in
        --check-only) check_only=true; shift ;;
        --torch) [[ $# -ge 2 ]] || { echo '--torch needs a value' >&2; exit 2; }; torch_mode="$2"; shift 2 ;;
        --data-dir) [[ $# -ge 2 ]] || { echo '--data-dir needs a path' >&2; exit 2; }; DATA_DIR="$2"; shift 2 ;;
        --download) [[ $# -ge 2 ]] || { echo '--download needs annotations or images' >&2; exit 2; }; download="$2"; shift 2 ;;
        --help|-h)
            echo 'Usage: bash bootstrap.sh [--check-only] [--torch existing|cpu|cu126|cu128] [--data-dir PATH] [--download annotations|images]'
            echo 'Default: install PyTorch 2.11.0 and torchvision 0.26.0 (CUDA 12.8) in isolated .venv.'
            echo 'Data downloads and training are separate opt-in steps.'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
case "$download" in none|annotations|images) ;; *) echo 'Invalid --download value' >&2; exit 2 ;; esac
export DATA_DIR DIAGNOSTICS_DIR PYTHON
if [[ "$check_only" == true ]]; then
    exec bash "$PROJECT_ROOT/launchers/CHECK_ENVIRONMENT.sh" --check-network
fi
bash "$PROJECT_ROOT/install/install.sh" --torch "$torch_mode"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
export PYTHON
bash "$PROJECT_ROOT/launchers/CHECK_ENVIRONMENT.sh"
if [[ "$download" != none ]]; then
    options=()
    [[ "$download" != images ]] || options+=(--images)
    bash "$PROJECT_ROOT/launchers/DOWNLOAD_DATA.sh" "${options[@]}"
    bash "$PROJECT_ROOT/launchers/AUDIT_DATASET.sh"
fi
echo 'Project setup complete. Prepare a verified recording split before training.'

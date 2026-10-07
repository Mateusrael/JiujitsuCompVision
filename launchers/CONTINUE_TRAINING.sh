#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/launchers/project_config.sh"
# The checkpoint retains model, paths and configuration; forward overrides explicitly.
exec "$PYTHON" "$PROJECT_ROOT/src/Scripts/train.py" "$@"

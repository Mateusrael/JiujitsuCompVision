#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/launchers/project_config.sh"
exec "$PYTHON" "$PROJECT_ROOT/install/check_environment.py" \
    --data-dir "$DATA_DIR" --output "$DIAGNOSTICS_DIR/environment.json" "$@"

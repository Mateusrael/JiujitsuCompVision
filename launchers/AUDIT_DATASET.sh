#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/launchers/project_config.sh"
exec "$PYTHON" "$PROJECT_ROOT/src/Scripts/audit_dataset.py" \
    --annotations "$DATA_DIR/annotations.json" \
    --output "$DIAGNOSTICS_DIR/dataset_audit.json" "$@"

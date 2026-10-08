#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/launchers/project_config.sh"
exec "$PYTHON" "$PROJECT_ROOT/src/Scripts/export_metrics.py" \
    --trainings-dir "$TRAININGS_DIR" \
    --output "$DIAGNOSTICS_DIR/metrics-$(date -u +%Y%m%dT%H%M%SZ)-$$.zip" "$@"

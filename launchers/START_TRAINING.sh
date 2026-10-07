#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/launchers/project_config.sh"
RUN_NAME="${RUN_NAME:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
exec "$PYTHON" "$PROJECT_ROOT/src/Scripts/train.py" \
    --model pose --annotations "$DATA_DIR/annotations.json" --split "$SPLIT_FILE" \
    --images-dir "$DATA_DIR/images" --run-dir "$TRAININGS_DIR/$RUN_NAME" "$@"

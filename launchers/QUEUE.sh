#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/launchers/project_config.sh"
export TRAININGS_DIR
cd "$PROJECT_ROOT"
exec "$PYTHON" -m src.Scripts.training_queue --queue-root "$DIAGNOSTICS_DIR/queues" "$@"

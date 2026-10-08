#!/usr/bin/env bash
# Precedence: caller environment > local_config.sh > versioned defaults.
# The launchers set PROJECT_ROOT before sourcing this file.
declare -A project_overrides=()
for config_key in DATA_DIR DIAGNOSTICS_DIR TRAININGS_DIR SPLIT_FILE PYTHON; do
    if [[ -v "$config_key" ]]; then
        project_overrides["$config_key"]="${!config_key}"
    fi
done
if [[ -f "$PROJECT_ROOT/launchers/local_config.sh" ]]; then
    source "$PROJECT_ROOT/launchers/local_config.sh"
fi
for config_key in "${!project_overrides[@]}"; do
    printf -v "$config_key" '%s' "${project_overrides[$config_key]}"
done
unset project_overrides config_key
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/Data}"
DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-$PROJECT_ROOT/diagnostics}"
TRAININGS_DIR="${TRAININGS_DIR:-$PROJECT_ROOT/trainings}"
SPLIT_FILE="${SPLIT_FILE:-$DATA_DIR/splits/multiview_sections.json}"
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
else
    PYTHON="${PYTHON:-python3}"
fi

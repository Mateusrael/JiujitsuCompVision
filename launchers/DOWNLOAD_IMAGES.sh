#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/launchers/project_config.sh"

usage() {
    cat <<'USAGE'
Usage: bash launchers/DOWNLOAD_IMAGES.sh [--timeout SECONDS] [--extract-only]

Download annotations and images, extract/verify the archive, then audit image coverage.
  --timeout SECONDS   Network socket timeout (also accepts --timeout=SECONDS).
  --extract-only      Use existing annotations.json and images.zip without downloading.
  -h, --help          Show this help.

Set DATA_DIR in the environment or launchers/local_config.sh to change the data folder.
Example: DATA_DIR="/path/to/data" bash launchers/DOWNLOAD_IMAGES.sh
USAGE
}

download_args=()
while (($#)); do
    case "$1" in
        --timeout)
            if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
                printf 'Missing value for --timeout. Use --timeout SECONDS.\n' >&2
                exit 2
            fi
            download_args+=(--timeout "$2")
            shift 2
            ;;
        --timeout=*)
            if [[ -z "${1#--timeout=}" ]]; then
                printf 'Missing value for --timeout. Use --timeout SECONDS.\n' >&2
                exit 2
            fi
            download_args+=("$1")
            shift
            ;;
        --extract-only)
            download_args+=("$1")
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unsupported argument: %s. Use --help for supported options.\n' "$1" >&2
            printf 'Set DATA_DIR in the environment or launchers/local_config.sh to change the data folder.\n' >&2
            exit 2
            ;;
    esac
done

"$PYTHON" "$PROJECT_ROOT/src/Scripts/download_data.py" --data-dir "$DATA_DIR" --images "${download_args[@]}"
"$PYTHON" "$PROJECT_ROOT/src/Scripts/audit_dataset.py" \
    --annotations "$DATA_DIR/annotations.json" \
    --images-dir "$DATA_DIR/images" \
    --output "$DIAGNOSTICS_DIR/dataset_audit.json"
printf 'Image preparation complete: download/extraction succeeded and all annotated image files passed the audit.\n'
printf 'Audit report: %s\n' "$DIAGNOSTICS_DIR/dataset_audit.json"

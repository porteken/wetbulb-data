#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${USE_UV:-1}" == "1" ]]; then
    exec "${UV_BIN:-uv}" run python "${HERE}/pipeline.py" "$@"
else
    exec "${PYTHON_BIN:-python}" "${HERE}/pipeline.py" "$@"
fi

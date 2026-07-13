#!/usr/bin/env bash
# Thin wrapper around the Python orchestrator (pipeline.py).
#
# The historical environment-variable interface (MODE, YEARS, PRODUCTS,
# USE_CLOUD_RUN, SKIP_DB_LOAD, ...) is still honored: pipeline.py reads the
# same variables as defaults, and any CLI flags passed here override them.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${USE_UV:-1}" == "1" ]]; then
    exec "${UV_BIN:-uv}" run python "${HERE}/pipeline.py" "$@"
else
    exec "${PYTHON_BIN:-python}" "${HERE}/pipeline.py" "$@"
fi

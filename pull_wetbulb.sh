#!/usr/bin/env bash
# Convenience wrapper: run the full pipeline (fetch -> process -> load -> views)
# for the wetbulb product only. The pet table is left untouched.
# Equivalent to: ./pull_all.sh --products wetbulb
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/pull_all.sh" --products wetbulb "$@"

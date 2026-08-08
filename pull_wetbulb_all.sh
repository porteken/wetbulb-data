#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

DRY_RUN=0
CONFIRM_DB=0
SHARD_COUNT="${COMBINED_CITY_SHARDS:-${CITY_SHARD_COUNT:-5}}"
SHARD_WORKERS="${COMBINED_SHARD_WORKERS:-1}"
STEPS=()

usage() {
  echo "usage: $0 [--dry-run] [--confirm-db-write] [--city-shard-count N] [--shard-workers N] [steps...]"
  echo "steps: cities crosswalk trial backfill gapfill validate bootstrap load cleanup views all"
}

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --confirm-db-write|--yes)
      CONFIRM_DB=1
      ;;
    --city-shard-count)
      shift
      SHARD_COUNT="${1:?missing shard count}"
      ;;
    --shard-workers)
      shift
      SHARD_WORKERS="${1:?missing shard worker count}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    cities|crosswalk|trial|backfill|gapfill|validate|bootstrap|load|cleanup|views|all)
      STEPS+=("$1")
      ;;
    *)
      echo "unknown option or step: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if ((${#STEPS[@]} == 0)); then
  STEPS=(backfill gapfill validate)
fi

NA_FLAGS=()
EU_FLAGS=()
((DRY_RUN)) && NA_FLAGS+=(--dry-run) && EU_FLAGS+=(--dry-run)
((CONFIRM_DB)) && NA_FLAGS+=(--confirm-db-write) && EU_FLAGS+=(--yes)
[[ ${SHARD_COUNT} =~ ^[1-9][0-9]*$ ]] || {
  echo "city shard count must be a positive integer" >&2
  exit 2
}
[[ ${SHARD_WORKERS} =~ ^[1-9][0-9]*$ ]] || {
  echo "shard worker count must be a positive integer" >&2
  exit 2
}
NA_FLAGS+=(--city-shard-count "${SHARD_COUNT}")

run_na() {
  NA_SHARD_WORKERS="${SHARD_WORKERS}" "${HERE}/pull_wetbulb.sh" "${NA_FLAGS[@]}" "$@"
}

run_eu() {
  EU_CITY_SHARDS="${SHARD_COUNT}" EU_SHARD_WORKERS="${SHARD_WORKERS}" \
    "${HERE}/pull_wetbulb_eu.sh" "${EU_FLAGS[@]}" "$@"
}

run_both() {
  local step=$1
  local na_pid eu_pid rc=0
  run_na "${step}" &
  na_pid=$!
  run_eu "${step}" &
  eu_pid=$!
  wait "${na_pid}" || rc=1
  wait "${eu_pid}" || rc=1
  ((rc == 0))
}

for step in "${STEPS[@]}"; do
  case "${step}" in
    all)
      run_both cities
      run_both crosswalk
      run_na backfill
      run_eu backfill
      run_na gapfill
      run_eu gapfill
      run_na validate
      run_na bootstrap
      run_eu load
      run_na views
      ;;
    validate|bootstrap|cleanup|views)
      run_na "${step}"
      ;;
    backfill|gapfill)
      run_na "${step}"
      run_eu "${step}"
      ;;
    *)
      run_both "${step}"
      ;;
  esac
done

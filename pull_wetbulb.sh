#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

CALLER_START_YEAR="${START_YEAR-}"
CALLER_END_YEAR="${END_YEAR-}"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . .env
  set +a
fi
[[ -z "${CALLER_START_YEAR}" ]] || START_YEAR="${CALLER_START_YEAR}"
[[ -z "${CALLER_END_YEAR}" ]] || END_YEAR="${CALLER_END_YEAR}"
unset CALLER_START_YEAR CALLER_END_YEAR

CATALOG_VERSION="na-msa-principal-cities-2023-ca-cma-ca-2021-deduplicated-v1"
OUTPUT_ROOT="${WETBULB_NA_ROOT:-na/${CATALOG_VERSION}}"
ALL_STEPS=(cities crosswalk backfill gapfill validate bootstrap views)
STEPS=()
DRY_RUN=0
CONFIRM_DB=0
SHARD_COUNT="${CITY_SHARD_COUNT:-5}"
START_YEAR="${START_YEAR:-1990}"
END_YEAR="${END_YEAR:-$((10#$(date -u +%Y) - 1))}"
FORCE_STATION_BACKFILL="${FORCE_STATION_BACKFILL:-0}"
LOAD_WORKERS="${LOAD_WORKERS:-1}"
NA_SHARD_WORKERS="${NA_SHARD_WORKERS:-1}"
PYTHON_RUN="${PYTHON_RUN:-uv run python}"

read -ra PY <<<"${PYTHON_RUN}"

usage() {
  echo "usage: $0 [--dry-run] [--confirm-db-write] [--city-shard-count N] [steps...]"
  echo "steps: cities crosswalk trial backfill gapfill validate bootstrap load cleanup views all"
}

run() {
  if ((DRY_RUN)); then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --confirm-db-write) CONFIRM_DB=1 ;;
    --city-shard-count)
      shift
      SHARD_COUNT="${1:?missing shard count}"
      ;;
    -h|--help) usage; exit 0 ;;
    all) STEPS+=("${ALL_STEPS[@]}") ;;
    cities|crosswalk|trial|backfill|gapfill|validate|bootstrap|load|cleanup|views) STEPS+=("$1") ;;
    *) echo "unknown option or step: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
if ((${#STEPS[@]} == 0)); then
  STEPS=(backfill gapfill validate)
fi
[[ "${SHARD_COUNT}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid shard count" >&2; exit 2; }
[[ "${FORCE_STATION_BACKFILL}" =~ ^[01]$ ]] || {
  echo "FORCE_STATION_BACKFILL must be 0 or 1" >&2
  exit 2
}
[[ "${LOAD_WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "LOAD_WORKERS must be a positive integer" >&2
  exit 2
}
[[ "${NA_SHARD_WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "NA_SHARD_WORKERS must be a positive integer" >&2
  exit 2
}
STATION_FORCE_ARGS=()
((FORCE_STATION_BACKFILL)) && STATION_FORCE_ARGS=(--force)

MANIFEST="${HERE}/cities_na.catalog.json"
mkdir -p "${OUTPUT_ROOT}"
check_catalog() {
  [[ -f "${MANIFEST}" ]] || { echo "missing committed ${MANIFEST}" >&2; exit 1; }
  local expected recorded
  expected="$("${PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["catalog_sha256"])' "${MANIFEST}")"
  recorded="${OUTPUT_ROOT}/catalog.sha256"
  if [[ -f "${recorded}" ]] && [[ "$(<"${recorded}")" != "${expected}" ]]; then
    echo "catalog hash mismatch in ${OUTPUT_ROOT}; refusing to mix shards" >&2
    exit 1
  fi
  if ((!DRY_RUN)); then
    printf '%s\n' "${expected}" >"${recorded}"
  fi
}

run_backfill_shard() {
  local shard=$1
  local year
  local years=()
  for ((year=10#${START_YEAR}; year<=10#${END_YEAR}; year++)); do
    years+=("${year}")
  done
  run "${PY[@]}" "${HERE}/pipeline.py" --years "${years[@]}" \
    --city-shard-count "${SHARD_COUNT}" --city-shard-index "${shard}" \
    --out-dir "${OUTPUT_ROOT}" "${STATION_FORCE_ARGS[@]}"
  run "${PY[@]}" "${HERE}/eccc.py" --start-year "${START_YEAR}" \
    --end-year "${END_YEAR}" --city-shard-count "${SHARD_COUNT}" \
    --city-shard-index "${shard}" --out-dir "${OUTPUT_ROOT}" \
    "${STATION_FORCE_ARGS[@]}"
}

run_gapfill_shard() {
  local shard=$1
  run "${PY[@]}" "${HERE}/gapfill.py" --start-year "${START_YEAR}" \
    --end-year "${END_YEAR}" --city-shard-count "${SHARD_COUNT}" \
    --city-shard-index "${shard}" --out-dir "${OUTPUT_ROOT}" \
    "${STATION_FORCE_ARGS[@]}"
}

run_city_shards() {
  local worker=$1
  local pids=() rc=0 shard pid
  for ((shard=0; shard<SHARD_COUNT; shard++)); do
    "${worker}" "${shard}" &
    pids+=("$!")
    if ((${#pids[@]} >= NA_SHARD_WORKERS)); then
      for pid in "${pids[@]}"; do
        wait "${pid}" || rc=1
      done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done
  ((rc == 0))
}

for step in "${STEPS[@]}"; do
  case "${step}" in
    cities)
      run "${PY[@]}" "${HERE}/cities_na.py"
      ;;
    crosswalk)
      run "${PY[@]}" "${HERE}/make_nldas_cell_map.py"
      run "${PY[@]}" "${HERE}/make_isd_station_map_na.py"
      run "${PY[@]}" "${HERE}/make_eccc_station_map.py"
      run "${PY[@]}" "${HERE}/make_city_center_map_na.py"
      run "${PY[@]}" "${HERE}/generate_forecast_inputs.py"
      ;;
    trial)
      check_catalog
      run "${PY[@]}" "${HERE}/pipeline.py" --years "${END_YEAR}" --city-shard-count 1 \
        --city-shard-index 0 --out-dir "${OUTPUT_ROOT}"
      ;;
    backfill)
      check_catalog
      run_city_shards run_backfill_shard
      ;;
    gapfill)
      check_catalog
      ((DRY_RUN)) || [[ -n "${EARTHDATA_USERNAME:-}" && -n "${EARTHDATA_PASSWORD:-}" ]] || {
        echo "gapfill requires EARTHDATA_USERNAME and EARTHDATA_PASSWORD" >&2
        exit 1
      }
      run_city_shards run_gapfill_shard
      ;;
    validate)
      check_catalog
      run "${PY[@]}" "${HERE}/validate_na_catalog.py" --root "${HERE}"
      ;;
    bootstrap)
      ((CONFIRM_DB)) || { echo "bootstrap requires --confirm-db-write" >&2; exit 1; }
      run "${PY[@]}" "${HERE}/make_city_center_map_na.py"
      run "${PY[@]}" "${HERE}/locations.py"
      run "${PY[@]}" "${HERE}/load.py" \
        --wetbulb-root "${OUTPUT_ROOT}/wetbulb_data_csv"
      ;;
    load)
      ((CONFIRM_DB)) || { echo "load requires --confirm-db-write" >&2; exit 1; }
      run "${PY[@]}" "${HERE}/make_city_center_map_na.py"
      run "${PY[@]}" "${HERE}/locations.py"
      run "${PY[@]}" "${HERE}/validate_location_catalog_db.py" \
        --locations-csv "${HERE}/locations.csv"
      run "${PY[@]}" "${HERE}/load.py" \
        --wetbulb-root "${OUTPUT_ROOT}/wetbulb_data_csv" \
        --append-only --skip-drop-views --skip-create-views --ensure-schema \
        --skip-table wetbulb
      run "${PY[@]}" "${HERE}/load_wetbulb.py" \
        --wetbulb-root "${OUTPUT_ROOT}/wetbulb_data_csv" \
        --load-workers "${LOAD_WORKERS}"
      ;;
    cleanup)
      ((CONFIRM_DB)) || { echo "cleanup requires --confirm-db-write" >&2; exit 1; }
      run "${PY[@]}" "${HERE}/cleanup_legacy_pet.py" --confirm-db-write
      ;;
    views)
      ((CONFIRM_DB)) || { echo "views requires --confirm-db-write" >&2; exit 1; }
      run "${PY[@]}" "${HERE}/refresh_views.py" --non-concurrent
      ;;
    *)
      echo "unknown step: ${step}" >&2
      exit 2
      ;;
  esac
done

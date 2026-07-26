#!/usr/bin/env bash
# Pull and optionally load the top-100 Canadian ECCC wet-bulb dataset.
#
# With no step arguments this generates inputs and performs the resumable
# 1991-2025 ECCC pull. Database writes are explicit:
#
#   ./pull_wetbulb_ca.sh --yes load views
#
# ERA5-Land gap filling is also explicit because it requires CDS credentials:
#
#   ./pull_wetbulb_ca.sh gapfill
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

if [[ -f .env ]]; then
  set -a
  . .env
  set +a
fi

CA_OUT_DIR=${CA_OUT_DIR:-ca}
CA_CITIES_CSV=${CA_CITIES_CSV:-cities_ca.csv}
CA_LOCATIONS_CSV=${CA_LOCATIONS_CSV:-locations_ca.csv}
CA_STATION_MAP_CSV=${CA_STATION_MAP_CSV:-cities_ca_eccc_stations.csv}
CA_START_YEAR=${CA_START_YEAR:-1991}
CA_END_YEAR=${CA_END_YEAR:-2025}
CA_ECCC_CONCURRENCY=${CA_ECCC_CONCURRENCY:-4}
CA_CDS_CONCURRENCY=${CA_CDS_CONCURRENCY:-2}
CA_CITY_SHARDS=${CA_CITY_SHARDS:-1}
CA_MIN_MISSING_DAYS=${CA_MIN_MISSING_DAYS:-19}
PYTHON_RUN=${PYTHON_RUN:-uv run python}

DEFAULT_STEPS=(cities crosswalk pull)
ALL_STEPS=(cities crosswalk pull gapfill load views)
DRY_RUN=0
ASSUME_YES=0

read -ra PY <<<"${PYTHON_RUN}"

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
die() {
  log "ERROR: $*"
  exit 1
}

run() {
  if ((DRY_RUN)); then
    printf '  [dry-run] %s\n' "$*" >&2
    return 0
  fi
  "$@"
}

usage() {
  cat <<EOF
Usage: ${0##*/} [options] [step ...]

Steps: ${ALL_STEPS[*]}
       all    -> ${ALL_STEPS[*]}
       (none) -> ${DEFAULT_STEPS[*]}

Options:
  -n, --dry-run   Print commands without running them.
  -y, --yes       Skip confirmation for database-writing steps.
  -h, --help      Show this help.

Environment overrides:
  CA_OUT_DIR=${CA_OUT_DIR}
  CA_START_YEAR=${CA_START_YEAR}
  CA_END_YEAR=${CA_END_YEAR}
  CA_CITY_SHARDS=${CA_CITY_SHARDS}
  CA_ECCC_CONCURRENCY=${CA_ECCC_CONCURRENCY}
  CA_CDS_CONCURRENCY=${CA_CDS_CONCURRENCY}
  CA_MIN_MISSING_DAYS=${CA_MIN_MISSING_DAYS}
  PYTHON_RUN='${PYTHON_RUN}'

Examples:
  ${0##*/}                         # generate inputs and pull ECCC data
  ${0##*/} --dry-run all           # show the complete workflow
  CA_CITY_SHARDS=4 ${0##*/} pull
  ${0##*/} gapfill                 # requires CDSAPI_URL and CDSAPI_KEY
  ${0##*/} --yes load views        # append Canada to Postgres
EOF
}

confirm_db_step() {
  local step=$1 reply
  ((ASSUME_YES || DRY_RUN)) && return 0
  if [[ ! -t 0 ]]; then
    die "step '${step}' writes to the database; use --yes in a non-interactive shell"
  fi
  read -rp "Step '${step}' writes to the database. Continue? [y/N] " reply
  [[ ${reply} == [yY]* ]] || die "aborted by user"
}

require_file() {
  [[ -f $1 ]] && return 0
  if ((DRY_RUN)); then
    log "  [dry-run] would require $1 (from the '$2' step)"
    return 0
  fi
  die "$1 not found -- run the '$2' step first"
}

run_sharded() {
  local script=$1
  shift
  local pids=() rc=0 index

  if ((CA_CITY_SHARDS <= 1)); then
    run "${PY[@]}" "${script}" \
      --city-shard-index 0 --city-shard-count 1 "$@"
    return
  fi

  for ((index = 0; index < CA_CITY_SHARDS; index++)); do
    if ((DRY_RUN)); then
      printf '  [dry-run] %s %s --city-shard-index %s --city-shard-count %s %s\n' \
        "${PY[*]}" "${script}" "${index}" "${CA_CITY_SHARDS}" "$*" >&2
      continue
    fi
    "${PY[@]}" "${script}" \
      --city-shard-index "${index}" --city-shard-count "${CA_CITY_SHARDS}" "$@" &
    pids+=("$!")
  done
  for pid in "${pids[@]:-}"; do
    [[ -n ${pid} ]] || continue
    wait "${pid}" || rc=1
  done
  ((rc == 0)) || die "${script}: at least one city shard failed"
}

step_cities() {
  log "[cities] generating Statistics Canada top-100 population centres"
  run "${PY[@]}" cities_ca.py --out "${CA_CITIES_CSV}"
  run "${PY[@]}" locations.py \
    --cities-csv "${CA_CITIES_CSV}" --out "${CA_LOCATIONS_CSV}"
}

step_crosswalk() {
  require_file "${CA_CITIES_CSV}" cities
  log "[crosswalk] matching up to three ECCC stations per city"
  run "${PY[@]}" make_eccc_station_map.py \
    --cities-csv "${CA_CITIES_CSV}" \
    --out "${CA_STATION_MAP_CSV}" \
    --start-year "${CA_START_YEAR}" \
    --end-year "${CA_END_YEAR}"
}

step_pull() {
  require_file "${CA_STATION_MAP_CSV}" crosswalk
  log "[pull] ECCC ${CA_START_YEAR}-${CA_END_YEAR} into ${CA_OUT_DIR}" \
    "(${CA_CITY_SHARDS} shard(s), ${CA_ECCC_CONCURRENCY} request threads each)"
  run_sharded eccc.py \
    --cities-csv "${CA_CITIES_CSV}" \
    --station-map-csv "${CA_STATION_MAP_CSV}" \
    --start-year "${CA_START_YEAR}" \
    --end-year "${CA_END_YEAR}" \
    --out-dir "${CA_OUT_DIR}" \
    --concurrency "${CA_ECCC_CONCURRENCY}"
}

step_gapfill() {
  require_file "${CA_STATION_MAP_CSV}" crosswalk
  if ((DRY_RUN)); then
    log "  [dry-run] would require cdsapi plus CDSAPI_URL/CDSAPI_KEY"
  else
    "${PY[@]}" -c \
      'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("cdsapi") else 1)' ||
      die "cdsapi is not installed -- run: uv sync --extra era5"
    : "${CDSAPI_URL:?CDSAPI_URL must be set and the ERA5-Land licence accepted}"
    : "${CDSAPI_KEY:?CDSAPI_KEY must be set}"
  fi
  log "[gapfill] filling material ECCC gaps with ERA5-Land"
  run "${PY[@]}" era5land.py \
    --cities-csv "${CA_CITIES_CSV}" \
    --station-map-csv "${CA_STATION_MAP_CSV}" \
    --start-year "${CA_START_YEAR}" \
    --end-year "${CA_END_YEAR}" \
    --out-dir "${CA_OUT_DIR}" \
    --city-shard-index 0 --city-shard-count 1 \
    --concurrency "${CA_CDS_CONCURRENCY}" \
    --min-missing-days "${CA_MIN_MISSING_DAYS}"
}

step_load() {
  require_file "${CA_LOCATIONS_CSV}" cities
  require_file "${CA_STATION_MAP_CSV}" crosswalk
  confirm_db_step load

  log "[load] enabling ECCC provenance in the database"
  run "${PY[@]}" -c "
import psycopg
from load import execute_sql_file
from shared_config import resolve_database_uri
uri = resolve_database_uri()
if not uri:
    raise SystemExit('database is not configured; set POSTGRES_DB_URI')
conn = psycopg.connect(uri)
conn.autocommit = True
try:
    execute_sql_file(conn, 'migrate_wetbulb_source_eccc.sql')
finally:
    conn.close()
"

  # Append-only is essential: Canada must not truncate existing US locations.
  log "[load] appending Canadian locations"
  run "${PY[@]}" load.py \
    --locations-csv "${CA_LOCATIONS_CSV}" \
    --append-only --skip-drop-views --skip-create-views --ensure-schema \
    --skip-table wetbulb \
    --skip-table gmst_observations \
    --skip-table gmst_scenarios \
    --skip-table forecast_station_groups \
    --skip-table forecast_interval_calibration

  log "[load] upserting Canadian daily wet-bulb rows"
  run "${PY[@]}" load_wetbulb.py \
    --wetbulb-root "${CA_OUT_DIR}/wetbulb_data_csv"
}

step_views() {
  confirm_db_step views
  log "[views] rebuilding and refreshing database views"
  run "${PY[@]}" load.py \
    --skip-table locations \
    --skip-table wetbulb \
    --skip-table gmst_observations \
    --skip-table gmst_scenarios \
    --skip-table forecast_station_groups \
    --skip-table forecast_interval_calibration
  run "${PY[@]}" refresh_views.py
}

main() {
  local steps=()
  while (($#)); do
    case $1 in
    -n | --dry-run) DRY_RUN=1 ;;
    -y | --yes) ASSUME_YES=1 ;;
    -h | --help)
      usage
      return
      ;;
    all) steps+=("${ALL_STEPS[@]}") ;;
    cities | crosswalk | pull | gapfill | load | views) steps+=("$1") ;;
    -*) die "unknown option: $1 (try --help)" ;;
    *) die "unknown step: $1 (valid: ${ALL_STEPS[*]}, all)" ;;
    esac
    shift
  done
  ((${#steps[@]})) || steps=("${DEFAULT_STEPS[@]}")

  [[ ${CA_OUT_DIR} != "." ]] ||
    die "CA_OUT_DIR must not be '.': Canadian parquet would collide with US batches"
  [[ ${CA_CITY_SHARDS} =~ ^[1-9][0-9]*$ ]] ||
    die "CA_CITY_SHARDS must be a positive integer"

  log "steps: ${steps[*]} (out-dir=${CA_OUT_DIR}, years=${CA_START_YEAR}-${CA_END_YEAR})"
  local step
  for step in "${steps[@]}"; do
    "step_${step}"
  done
  log "finished: ${steps[*]}"
}

main "$@"

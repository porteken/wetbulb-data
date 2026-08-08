#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

CALLER_EU_START_YEAR="${EU_START_YEAR-}"
CALLER_EU_END_YEAR="${EU_END_YEAR-}"
CALLER_EU_CITY_SHARDS="${EU_CITY_SHARDS-}"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . .env
  set +a
fi
[[ -z "${CALLER_EU_START_YEAR}" ]] || EU_START_YEAR="${CALLER_EU_START_YEAR}"
[[ -z "${CALLER_EU_END_YEAR}" ]] || EU_END_YEAR="${CALLER_EU_END_YEAR}"
[[ -z "${CALLER_EU_CITY_SHARDS}" ]] || EU_CITY_SHARDS="${CALLER_EU_CITY_SHARDS}"
unset CALLER_EU_START_YEAR CALLER_EU_END_YEAR CALLER_EU_CITY_SHARDS

EU_OUT_DIR=${EU_OUT_DIR:-eu}
EU_CITIES_CSV=${EU_CITIES_CSV:-cities_eu.csv}
EU_LOCATIONS_CSV=${EU_LOCATIONS_CSV:-locations_eu.csv}
EU_STATION_MAP_CSV=${EU_STATION_MAP_CSV:-cities_eu_isd_stations.csv}
EU_GHCNH_STATION_MAP_CSV=${EU_GHCNH_STATION_MAP_CSV:-cities_eu_ghcnh_stations.csv}
EU_START_YEAR=${EU_START_YEAR:-1990}
EU_END_YEAR=${EU_END_YEAR:-2025}
EU_CROSSWALK_START_YEAR=${EU_CROSSWALK_START_YEAR:-1991}
EU_TRIAL_YEARS=${EU_TRIAL_YEARS:-1990 2012 2025}
EU_ISD_CONCURRENCY=${EU_ISD_CONCURRENCY:-8}
EU_EARTH_ENGINE_CONCURRENCY=${EU_EARTH_ENGINE_CONCURRENCY:-8}
EU_CITY_SHARDS=${EU_CITY_SHARDS:-1}
EU_MIN_MISSING_DAYS=${EU_MIN_MISSING_DAYS:-19}
EU_CELL_MAP_CSV=${EU_CELL_MAP_CSV:-cities_eu_era5land_cells.csv}
EU_FORCE_STATION_BACKFILL=${EU_FORCE_STATION_BACKFILL:-0}
EU_LOAD_WORKERS=${EU_LOAD_WORKERS:-1}
PYTHON_RUN=${PYTHON_RUN:-uv run python}

DEFAULT_STEPS=(cities crosswalk backfill gapfill)
ALL_STEPS=(cities crosswalk trial backfill gapfill load views)
VALID_STEPS=("${ALL_STEPS[@]}" cleanup)

DRY_RUN=0
ASSUME_YES=0

read -ra PY <<<"${PYTHON_RUN}"
read -ra TRIAL_YEARS <<<"${EU_TRIAL_YEARS}"
[[ ${EU_FORCE_STATION_BACKFILL} =~ ^[01]$ ]] || {
  printf 'EU_FORCE_STATION_BACKFILL must be 0 or 1\n' >&2
  exit 2
}
[[ ${EU_LOAD_WORKERS} =~ ^[1-9][0-9]*$ ]] || {
  printf 'EU_LOAD_WORKERS must be a positive integer\n' >&2
  exit 2
}
STATION_FORCE_ARGS=()
((EU_FORCE_STATION_BACKFILL)) && STATION_FORCE_ARGS=(--force)

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S' || true)" "$*" >&2; }
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

confirm_db_step() {
  local step=$1
  ((ASSUME_YES)) && return 0
  ((DRY_RUN)) && return 0
  if [[ ! -t 0 ]]; then
    die "step '${step}' writes to the database; re-run with --yes in a non-interactive shell"
  fi
  local reply
  read -rp "Step '${step}' writes to the database. Continue? [y/N] " reply
  [[ ${reply} == [yY]* ]] || die "aborted by user"
}

require_file() {
  local path=$1 step=$2
  [[ -f ${path} ]] && return 0
  ((DRY_RUN)) && {
    log "  [dry-run] would require ${path} (from the '${step}' step)"
    return 0
  }
  die "${path} not found -- run the '${step}' step first"
}

run_sharded() {
  local script=$1
  shift
  local shards=${EU_CITY_SHARDS}

  if ((shards <= 1)); then
    run "${PY[@]}" "${script}" \
      --city-shard-index 0 --city-shard-count 1 "$@"
    return
  fi

  local pids=() rc=0 index
  for ((index = 0; index < shards; index++)); do
    if ((DRY_RUN)); then
      printf '  [dry-run] %s %s --city-shard-index %s --city-shard-count %s %s\n' \
        "${PY[*]}" "${script}" "${index}" "${shards}" "$*" >&2
      continue
    fi
    "${PY[@]}" "${script}" \
      --city-shard-index "${index}" --city-shard-count "${shards}" "$@" &
    pids+=($!)
  done
  for pid in "${pids[@]:-}"; do
    [[ -n ${pid} ]] || continue
    wait "${pid}" || rc=1
  done
  ((rc == 0)) || die "${script}: at least one shard failed"
}

step_cities() {
  log "[cities] jointly selecting EU cities, unique ERA5-Land cells, and unique ISD stations"
  run "${PY[@]}" cities_eu.py
  run "${PY[@]}" locations.py \
    --cities-csv "${EU_CITIES_CSV}" --out "${EU_LOCATIONS_CSV}"
  log "[cities] wrote ${EU_CITIES_CSV}, ${EU_STATION_MAP_CSV}, and ${EU_LOCATIONS_CSV}"
}

step_crosswalk() {
  require_file "${EU_CITIES_CSV}" cities
  require_file "${EU_STATION_MAP_CSV}" cities
  log "[crosswalk] unique station map was produced atomically with city selection"
}

step_trial() {
  require_file "${EU_STATION_MAP_CSV}" crosswalk
  local year
  for year in "${TRIAL_YEARS[@]}"; do
    local worker=isd.py station_map=${EU_STATION_MAP_CSV} source=ISD
    if ((year >= 1990)); then
      worker=ghcnh.py
      station_map=${EU_GHCNH_STATION_MAP_CSV}
      source=GHCNh
      require_file "${station_map}" crosswalk
    fi
    log "[trial] ${source} year ${year}"
    run "${PY[@]}" "${worker}" \
      --cities-csv "${EU_CITIES_CSV}" \
      --station-map-csv "${station_map}" \
      --start-year "${year}" --end-year "${year}" \
      --out-dir "${EU_OUT_DIR}" \
      --concurrency "${EU_ISD_CONCURRENCY}" \
      "${STATION_FORCE_ARGS[@]}"
  done
  log "[trial] inspect ${EU_OUT_DIR}/wetbulb_data_csv before running the backfill"
}

step_backfill() {
  require_file "${EU_STATION_MAP_CSV}" crosswalk
  require_file "${EU_GHCNH_STATION_MAP_CSV}" crosswalk
  log "[backfill] station observations ${EU_START_YEAR}-${EU_END_YEAR} into ${EU_OUT_DIR}" \
    "(${EU_CITY_SHARDS} shard(s) x ${EU_ISD_CONCURRENCY} threads);" \
    "processing one year at a time to bound memory; resumable"
  local year
  for ((year = EU_START_YEAR; year <= EU_END_YEAR; year++)); do
    local worker=isd.py station_map=${EU_STATION_MAP_CSV} source=ISD
    if ((year >= 1990)); then
      worker=ghcnh.py
      station_map=${EU_GHCNH_STATION_MAP_CSV}
      source=GHCNh
    fi
    log "[backfill] ${source} year ${year}"
    run_sharded "${worker}" \
      --cities-csv "${EU_CITIES_CSV}" \
      --station-map-csv "${station_map}" \
      --start-year "${year}" --end-year "${year}" \
      --out-dir "${EU_OUT_DIR}" \
      --concurrency "${EU_ISD_CONCURRENCY}" \
      "${STATION_FORCE_ARGS[@]}"
  done
  log "[backfill] done"
}

step_gapfill() {
  require_file "${EU_STATION_MAP_CSV}" crosswalk
  if ((DRY_RUN)); then
    log "  [dry-run] would require earthengine-api and Google Earth Engine credentials"
  else
    "${PY[@]}" -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("ee") else 1)' ||
      die "earthengine-api is not installed -- run: uv sync --extra earth-engine"
  fi

  local cell_map_args=()
  if [[ -n ${EU_CELL_MAP_CSV} && -f ${EU_CELL_MAP_CSV} ]]; then
    cell_map_args=(--cell-map-csv "${EU_CELL_MAP_CSV}")
    log "[gapfill] applying ERA5-Land land-cell overrides from ${EU_CELL_MAP_CSV}"
  fi

  log "[gapfill] ERA5-Land ${EU_START_YEAR}-${EU_END_YEAR}," \
    "min-missing-days=${EU_MIN_MISSING_DAYS}, ${EU_EARTH_ENGINE_CONCURRENCY} concurrent Earth Engine request(s)"
  run "${PY[@]}" earth_engine_era5land.py \
    --cities-csv "${EU_CITIES_CSV}" \
    --station-map-csv "${EU_STATION_MAP_CSV}" \
    --start-year "${EU_START_YEAR}" --end-year "${EU_END_YEAR}" \
    --out-dir "${EU_OUT_DIR}" \
    --city-shard-index 0 --city-shard-count 1 \
    --concurrency "${EU_EARTH_ENGINE_CONCURRENCY}" \
    --min-missing-days "${EU_MIN_MISSING_DAYS}" \
    "${STATION_FORCE_ARGS[@]}" \
    "${cell_map_args[@]}"
  log "[gapfill] done"
}

step_load() {
  require_file "${EU_LOCATIONS_CSV}" cities
  confirm_db_step load

  log "[load] appending EU rows to locations"
  run "${PY[@]}" load.py \
    --locations-csv "${EU_LOCATIONS_CSV}" \
    --append-only --skip-drop-views --skip-create-views --ensure-schema \
    --skip-table wetbulb \
    --skip-table gmst_observations \
    --skip-table gmst_scenarios \
    --skip-table forecast_station_groups \
    --skip-table forecast_interval_calibration

  log "[load] upserting daily wet-bulb rows from ${EU_OUT_DIR}/wetbulb_data_csv"
  run "${PY[@]}" load_wetbulb.py \
    --wetbulb-root "${EU_OUT_DIR}/wetbulb_data_csv" \
    --wetbulb-start-year "${EU_START_YEAR}" \
    --wetbulb-end-year "${EU_END_YEAR}" \
    --load-workers "${EU_LOAD_WORKERS}"
  log "[load] done"
}

step_cleanup() {
  confirm_db_step cleanup
  log "[cleanup] removing obsolete PET database objects"
  run "${PY[@]}" cleanup_legacy_pet.py --confirm-db-write
  log "[cleanup] done"
}

step_views() {
  confirm_db_step views
  log "[views] refreshing existing materialized views"
  run "${PY[@]}" refresh_views.py --non-concurrent
  log "[views] done"
}

main() {
  local steps=() argument
  while (($#)); do
    argument=$1
    case ${argument} in
    -n | --dry-run) DRY_RUN=1 ;;
    -y | --yes) ASSUME_YES=1 ;;
    -h | --help)
      usage
      return 0
      ;;
    all) steps+=("${ALL_STEPS[@]}") ;;
    cities | crosswalk | trial | backfill | gapfill | load | cleanup | views)
      steps+=("${argument}")
      ;;
    -*) die "unknown option: ${argument} (try --help)" ;;
    *) die "unknown step: ${argument} (valid: ${VALID_STEPS[*]}, all)" ;;
    esac
    shift
  done

  ((${#steps[@]})) || steps=("${DEFAULT_STEPS[@]}")

  [[ ${EU_OUT_DIR} != "." ]] ||
    die "EU_OUT_DIR must not be '.': parquet batch filenames encode only year/batch/shard, so EU output would collide with the US tree"

  log "steps: ${steps[*]}  (out-dir=${EU_OUT_DIR}, years=${EU_START_YEAR}-${EU_END_YEAR})"
  local step
  for step in "${steps[@]}"; do
    "step_${step}"
  done
  log "finished: ${steps[*]}"

  if [[ " ${steps[*]} " != *" load "* &&
        " ${steps[*]} " != *" cleanup "* &&
        " ${steps[*]} " != *" views "* ]]; then
    log "note: nothing was written to the database; run '${0##*/} load views' when ready"
  fi
}

main "$@"

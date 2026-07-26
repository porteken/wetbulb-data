#!/usr/bin/env bash
# Orchestrate the EU (EU-27 + UK + Switzerland) daily wet-bulb pull.
#
# Steps run in order and are individually resumable, because the backfill
# alone takes 6-12h:
#
#   cities     cities_eu.csv + locations_eu.csv          (~1 min, network)
#   crosswalk  cities_eu_isd_stations.csv                (~minutes, network)
#   trial      a few sample years, to eyeball coverage   (~minutes)
#   backfill   full ISD station backfill                 (6-12h)
#   gapfill    ERA5-Land fill for missing city-days      (hours, CDS queue)
#   load       locations + wetbulb into Postgres         (DB write)
#   views      rebuild views and refresh matviews        (DB write)
#
# With no arguments the data steps run (cities crosswalk backfill gapfill);
# the DB steps are opt-in because they mutate a live database. Name steps
# explicitly to run a subset, e.g. `./pull_wetbulb_eu.sh gapfill load`.
#
# Deliberately drives isd.py/era5land.py directly rather than pipeline.py:
# pipeline.py forwards --city-shard-count but never --city-shard-index, so
# a sharded run through it would silently process only shard 0.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . .env
  set +a
fi

EU_OUT_DIR=${EU_OUT_DIR:-eu}
EU_CITIES_CSV=${EU_CITIES_CSV:-cities_eu.csv}
EU_LOCATIONS_CSV=${EU_LOCATIONS_CSV:-locations_eu.csv}
EU_STATION_MAP_CSV=${EU_STATION_MAP_CSV:-cities_eu_isd_stations.csv}
EU_START_YEAR=${EU_START_YEAR:-2000}
EU_END_YEAR=${EU_END_YEAR:-2025}
EU_CROSSWALK_START_YEAR=${EU_CROSSWALK_START_YEAR:-1991}
EU_TRIAL_YEARS=${EU_TRIAL_YEARS:-2000 2012 2025}
EU_ISD_CONCURRENCY=${EU_ISD_CONCURRENCY:-8}
EU_CDS_CONCURRENCY=${EU_CDS_CONCURRENCY:-2}
EU_CITY_SHARDS=${EU_CITY_SHARDS:-1}
EU_MIN_MISSING_DAYS=${EU_MIN_MISSING_DAYS:-19}
PYTHON_RUN=${PYTHON_RUN:-uv run python}

DEFAULT_STEPS=(cities crosswalk backfill gapfill)
ALL_STEPS=(cities crosswalk trial backfill gapfill load views)

DRY_RUN=0
ASSUME_YES=0

read -ra PY <<<"${PYTHON_RUN}"
read -ra TRIAL_YEARS <<<"${EU_TRIAL_YEARS}"

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
       all   -> ${ALL_STEPS[*]}
       (none) -> ${DEFAULT_STEPS[*]}

Options:
  -n, --dry-run   Print the commands that would run, then exit.
  -y, --yes       Skip the confirmation prompt on DB-mutating steps.
  -h, --help      Show this help.

Key environment overrides (current value shown):
  EU_OUT_DIR=${EU_OUT_DIR}                 parquet root; must differ from the US tree
  EU_START_YEAR=${EU_START_YEAR}              backfill start (set 1991 for the extension)
  EU_END_YEAR=${EU_END_YEAR}
  EU_CITY_SHARDS=${EU_CITY_SHARDS}                parallel local ISD processes
  EU_ISD_CONCURRENCY=${EU_ISD_CONCURRENCY}            threads per ISD process
  EU_CDS_CONCURRENCY=${EU_CDS_CONCURRENCY}            concurrent CDS requests (keep low)
  EU_MIN_MISSING_DAYS=${EU_MIN_MISSING_DAYS}          gap-fill threshold; 1 fills everything
  EU_TRIAL_YEARS='${EU_TRIAL_YEARS}'

Examples:
  ${0##*/}                          # cities -> crosswalk -> backfill -> gapfill
  ${0##*/} --dry-run all            # show every command without running
  ${0##*/} trial                    # sample years only, before committing
  EU_CITY_SHARDS=4 ${0##*/} backfill
  EU_START_YEAR=1991 EU_END_YEAR=1999 ${0##*/} backfill gapfill
  ${0##*/} load views               # after a completed backfill
EOF
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
  [[ -f $1 ]] && return 0
  # Under --dry-run the earlier steps never actually produced their outputs,
  # so a missing input is expected rather than fatal.
  ((DRY_RUN)) && {
    log "  [dry-run] would require $1 (from the '$2' step)"
    return 0
  }
  die "$1 not found -- run the '$2' step first"
}

# Run one command per city shard, in parallel, failing if any shard fails.
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
  log "[cities] building the EU city list"
  run "${PY[@]}" cities_eu.py
  run "${PY[@]}" locations.py \
    --cities-csv "${EU_CITIES_CSV}" --out "${EU_LOCATIONS_CSV}"
  log "[cities] wrote ${EU_CITIES_CSV} and ${EU_LOCATIONS_CSV}"
}

step_crosswalk() {
  require_file "${EU_CITIES_CSV}" cities
  log "[crosswalk] resolving one ISD station per city (probing NCEI)"
  # Exits non-zero when some city has no verified station; those cities are
  # still written with an empty isd_ids and fall through to ERA5-Land.
  if ! run "${PY[@]}" make_isd_station_map_eu.py \
    --cities-csv "${EU_CITIES_CSV}" \
    --out "${EU_STATION_MAP_CSV}" \
    --start-year "${EU_CROSSWALK_START_YEAR}" \
    --end-year "${EU_END_YEAR}"; then
    log "[crosswalk] some cities have no verified station (see warnings above);" \
      "they will be covered entirely by ERA5-Land"
  fi
  require_file "${EU_STATION_MAP_CSV}" crosswalk
}

step_trial() {
  require_file "${EU_STATION_MAP_CSV}" crosswalk
  local year
  for year in "${TRIAL_YEARS[@]}"; do
    log "[trial] ISD year ${year}"
    run "${PY[@]}" isd.py \
      --cities-csv "${EU_CITIES_CSV}" \
      --station-map-csv "${EU_STATION_MAP_CSV}" \
      --start-year "${year}" --end-year "${year}" \
      --out-dir "${EU_OUT_DIR}" \
      --concurrency "${EU_ISD_CONCURRENCY}"
  done
  log "[trial] inspect ${EU_OUT_DIR}/wetbulb_data_csv before running the backfill"
}

step_backfill() {
  require_file "${EU_STATION_MAP_CSV}" crosswalk
  log "[backfill] ISD ${EU_START_YEAR}-${EU_END_YEAR} into ${EU_OUT_DIR}" \
    "(${EU_CITY_SHARDS} shard(s) x ${EU_ISD_CONCURRENCY} threads); resumable"
  run_sharded isd.py \
    --cities-csv "${EU_CITIES_CSV}" \
    --station-map-csv "${EU_STATION_MAP_CSV}" \
    --start-year "${EU_START_YEAR}" --end-year "${EU_END_YEAR}" \
    --out-dir "${EU_OUT_DIR}" \
    --concurrency "${EU_ISD_CONCURRENCY}"
  log "[backfill] done"
}

step_gapfill() {
  require_file "${EU_STATION_MAP_CSV}" crosswalk
  if ((DRY_RUN)); then
    log "  [dry-run] would require cdsapi installed and CDSAPI_URL/CDSAPI_KEY set"
  else
    "${PY[@]}" -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("cdsapi") else 1)' ||
      die "cdsapi is not installed -- run: uv sync --extra era5"
    : "${CDSAPI_URL:?CDSAPI_URL must be set (see .env); the ERA5-Land licence must also be accepted on your CDS account}"
    : "${CDSAPI_KEY:?CDSAPI_KEY must be set (see .env)}"
  fi

  # Deliberately single-process: gap detection scans every wetbulb_batch_*
  # file in each year partition regardless of how the backfill was sharded,
  # and CDS enforces low per-user concurrency.
  log "[gapfill] ERA5-Land ${EU_START_YEAR}-${EU_END_YEAR}," \
    "min-missing-days=${EU_MIN_MISSING_DAYS}, ${EU_CDS_CONCURRENCY} concurrent CDS request(s)"
  run "${PY[@]}" era5land.py \
    --cities-csv "${EU_CITIES_CSV}" \
    --station-map-csv "${EU_STATION_MAP_CSV}" \
    --start-year "${EU_START_YEAR}" --end-year "${EU_END_YEAR}" \
    --out-dir "${EU_OUT_DIR}" \
    --city-shard-index 0 --city-shard-count 1 \
    --concurrency "${EU_CDS_CONCURRENCY}" \
    --min-missing-days "${EU_MIN_MISSING_DAYS}"
  log "[gapfill] done"
}

step_load() {
  require_file "${EU_LOCATIONS_CSV}" cities
  confirm_db_step load

  log "[load] applying the era5land source-constraint migration (idempotent)"
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
    execute_sql_file(conn, 'migrate_wetbulb_source_era5land.sql')
finally:
    conn.close()
"

  # --skip-drop-views is load-bearing: without it load.py truncates
  # locations, which would delete the US cities.
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
  run "${PY[@]}" load_wetbulb.py --wetbulb-root "${EU_OUT_DIR}/wetbulb_data_csv"
  log "[load] done"
}

step_views() {
  confirm_db_step views
  log "[views] rebuilding views (drop -> create -> gmst -> eu, atomically)"
  run "${PY[@]}" load.py \
    --skip-table locations \
    --skip-table wetbulb \
    --skip-table gmst_observations \
    --skip-table gmst_scenarios \
    --skip-table forecast_station_groups \
    --skip-table forecast_interval_calibration
  log "[views] refreshing materialized views"
  run "${PY[@]}" refresh_views.py
  log "[views] done"
}

main() {
  local steps=()
  while (($#)); do
    case $1 in
    -n | --dry-run) DRY_RUN=1 ;;
    -y | --yes) ASSUME_YES=1 ;;
    -h | --help)
      usage
      return 0
      ;;
    all) steps+=("${ALL_STEPS[@]}") ;;
    cities | crosswalk | trial | backfill | gapfill | load | views) steps+=("$1") ;;
    -*) die "unknown option: $1 (try --help)" ;;
    *) die "unknown step: $1 (valid: ${ALL_STEPS[*]}, all)" ;;
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

  if [[ " ${steps[*]} " != *" load "* ]]; then
    log "note: nothing was written to the database; run '${0##*/} load views' when ready"
  fi
}

main "$@"

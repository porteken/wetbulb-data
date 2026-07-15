# Wetbulb Data Pipeline

This is the automated pipeline for processing wet-bulb temperature data and loading it into the web application. It was split out of the `pet-data` pipeline repo, so it still shares the underlying orchestration (`pipeline.py`, `load.py`, `historical_pet_update.py`) and the Postgres/S3 infrastructure with that project's PET pipeline.

1. **Compute**: PET (Physiological Equivalent Temperature) and wet-bulb temperature are computed daily per US city in parallel. PET comes from ARCO-ERA5 on Google Cloud Run (`google_era5.py`, `era5-worker`). Wet-bulb comes from NOAA's Local Climatological Data (LCD) v2 station observations via the Davies-Jones (2008) method (`wetbulb.py`), which stores both the daily average and daily max; it's fetched from NCEI's bulk archive directly on GitHub Actions runners (`lcd.py`, the default, no auth required), with NLDAS-2 gridded reanalysis kept as a fallback source (`giovanni.py` via the Giovanni Time Series API, or `nldas.py`'s original per-hour granule-download path on Cloud Run).
2. **Store**: Workers write parquet shards directly to an AWS S3 bucket, partitioned by year.
3. **Load**: Parquet shards are COPYed into a staging table and upserted into Northflank-hosted Postgres on the `(location_id, date)` unique key, so every load is idempotent and safe to re-run.
4. **Analyze**: Materialized views in Postgres generate summary statistics, historical trend comparisons, and long-term forecasts. Yearly appends refresh the views in place (`refresh_views.py`); full rebuilds recreate them from `create_views.sql`.

## Entry points

- `pipeline.py` orchestrates the full pipeline locally (fetch → compute → load → views). `pull_all.sh`, `pull_pet.sh`, `pull_wetbulb.sh`, and `pull_both.sh` are thin wrappers around it, and the historical environment variables (`MODE`, `YEARS`, `PRODUCTS`, `USE_CLOUD_RUN`, ...) are still honored as defaults. Run `pipeline.py --help` for the full flag list; `--mode smoke` processes a single week for quick validation.
- The `yearly_update` GitHub workflow appends the previous year every early April (once ARCO's final data covers it — a pre-flight check in `build_year_range.py --check-arco` fails fast otherwise) and opens a GitHub issue if any job fails.
- The `wetbulb_backfill` GitHub workflow (`workflow_dispatch`) reruns any historical wet-bulb range via NOAA LCD, city-sharded across parallel jobs, then truncates and reloads the `wetbulb` table.

## Wet-bulb data source: NOAA LCD vs. NLDAS-2 (Giovanni / granules)

`pipeline.py --wetbulb-source` selects how station/grid data is fetched:

- **`lcd`** (default): `lcd.py` fetches each city's nearest NOAA weather station's daily CSV files from [NCEI's Local Climatological Data v2 bulk archive](https://www.ncei.noaa.gov/products/land-based-station/local-climatological-data) — plain HTTPS, **no authentication**, and no rate limiting observed under sustained load. Nearby cities that share an airport station are deduplicated to one fetch. Always runs as local subprocesses — even when `--use-cloud-run` is set for the PET/ERA5 path — writing straight to S3. This is what the `yearly_update` and `wetbulb_backfill` workflows use, for free on GitHub Actions instead of billed Cloud Run compute.
  - Each city is mapped to its nearest station with a verified hourly-observation record using the checked-in `cities_lcd_stations.csv` (regenerate it with `make_lcd_station_map.py` if `cities.csv`'s city list changes; this hits the IEM Mesonet and NCEI over the network, no credentials needed).
  - Wet-bulb is computed uniformly with `wetbulb_davies_jones` from station temperature/dewpoint/pressure for every year, rather than trusting NOAA's own `HourlyWetBulbTemperature` column — that field is only populated in recent years, so relying on it would create a methodological seam mid-record.
  - A station-year fetch failure (network/5xx, exhausted retries) blocks only that specific year's write for the affected cities — unlike the Giovanni path's whole-shard skip — since LCD fetches one file per station-year and the gap is known precisely. A 404 or a file with zero hourly rows is treated as a legitimate permanent absence (the station just didn't report that year), not a gap, and the year is still written.
- **`giovanni`** (fallback): fetches each city's whole multi-year hourly NLDAS-2 series from the [Giovanni Time Series API](https://api.giovanni.earthdata.nasa.gov/timeseries) in one request per variable. Requires a free NASA Earthdata account (see Credentials below). Each city is asked for the same land-snapped NLDAS-2 grid cell the granule path resolves via its static land mask, using the checked-in `cities_nldas_cells.csv` (regenerate with `make_nldas_cell_map.py`; needs the `nldas` optional dependency group and a live granule download).
- **`granules`** (fallback): the original `nldas.py` path — downloads full hourly NLDAS-2 CONUS granules over HTTPS and extracts each city's nearest grid cell. Requires Earthdata credentials; runs on Cloud Run (`nldas-worker`) when `--use-cloud-run` is set. `nldas-worker` isn't deployed by `cloudrun_provision.sh` by default — set `DEPLOY_NLDAS_WORKER=1` to deploy it.

### LCD vs. NLDAS-2: known differences

LCD is a point station observation; NLDAS-2 is a 0.125° grid-cell average. A validation comparison across 15 cities spanning 2000–2025 found daily-max wet-bulb MAE ≈ 1.35 °C and mean bias ≈ −0.66 °C between the two sources, stable across decades — an expected spatial-representativeness gap, not a data-quality problem. Also note: LCD hourly timestamps are station local standard time, while NLDAS-2 is UTC, so daily aggregation windows (and therefore daily max/average values on any given calendar date) shift by the station's UTC offset relative to the old NLDAS-derived dataset.

## Running locally (no cloud)

The LCD wet-bulb path needs no cloud compute and no third-party credentials at all — `lcd.py` runs as local subprocesses even in the default (Cloud Run/S3) configuration, and `--local` just changes where it reads/writes and skips the ERA5 Cloud Run dispatch. A full local backfill or yearly update needs only Postgres credentials (`POSTGRES_DB_URI` or `PG*`) in `.env` — no AWS, GCP, or Earthdata credentials required (those are only needed for the `giovanni`/`granules` fallback sources).

- **Full historical backfill**: `./pull_wetbulb.sh --local --truncate`. With the defaults (years 2000 → last year, 10 city shards × 2 concurrent `lcd.py` processes × 8 threads each), a full ~26-year backfill for 500 cities takes roughly 30–45 minutes end to end, including the Postgres load and view rebuild.
- **Yearly/incremental update**: `./pull_wetbulb.sh --local --years 2025` (no `--truncate` — loads upsert on `(location_id, date)`, so this is safe to re-run).
- **Tuning parallelism**: total concurrent LCD requests ≈ `--lcd-job-limit` × `--lcd-concurrency`. NCEI's bulk archive has shown no rate limiting under sustained load (unlike the Giovanni API), so the modest defaults here are a courtesy, not a known necessity — raise them cautiously if you need a faster backfill. `--lcd-city-shard-count` controls how many `lcd.py` processes the city list is split across (each `--lcd-job-limit` bounds how many run at once).
- **Resuming an interrupted run**: re-run the same command with `--resume-local` added. This skips wiping the local `wetbulb_data_csv/year=YYYY` output before the run, so `lcd.py`'s per-shard/per-year `batch_exists` check picks up only the work that's still missing (including any individual years that hit a transient fetch gap). Only reuse this across reruns that keep `--lcd-city-shard-count` the same — partition filenames encode the shard index, so changing the shard count invalidates the resume check.
- The `wetbulb_backfill` and `yearly_update` GitHub workflows remain available as a free, hands-off alternative to running locally.

## Operational notes (NCEI LCD bulk archive)

- **File GETs are fast and unthrottled; bucket LIST/HEAD requests are pathologically slow.** Never probe station-file existence with HEAD or a directory listing — `make_lcd_station_map.py` learned this the hard way. Use a small ranged GET instead.
- **A 200 response doesn't mean hourly data exists.** Some stations (e.g. Buckley SFB near Denver, `USW00023062`) publish an LCD file every year that contains only daily/monthly summary rows, zero hourly METAR. `make_lcd_station_map.py` verifies actual hourly-report-type markers are present before matching a city to a station, not just that the file downloads.
- **Station coverage is tiered.** Most cities match a station whose archive covers the full pipeline window (`tier` 1 in `cities_lcd_stations.csv`); a handful may fall back to a shorter-record station (`tier` 2) if no full-window candidate verifies among the nearest 15 — those cities simply have fewer years of data, same as any other station gap.

## Credentials

- **LCD** (default wet-bulb source): no credentials needed beyond Postgres.
- **Giovanni / granules** (fallback wet-bulb sources): need a free [NASA Earthdata](https://urs.earthdata.nasa.gov/) account with the "NASA GESDISC DATA ARCHIVE" application authorized. Set `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` in `.env` for local runs, or as GitHub Actions / Secret Manager secrets (`cloudrun_provision.sh` wires them into the `nldas-worker` job when `DEPLOY_NLDAS_WORKER=1`).
- PET/ERA5 and all database loads still need the usual Postgres/AWS/GCP credentials regardless of wet-bulb source.

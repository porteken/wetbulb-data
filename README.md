# Wetbulb Data Pipeline

This is the automated pipeline for processing wet-bulb temperature data and loading it into the web application. It was split out of the `pet-data` pipeline repo, so it still shares the underlying orchestration (`pipeline.py`, `load.py`, `historical_pet_update.py`) and the Postgres/S3 infrastructure with that project's PET pipeline.

1. **Compute**: PET (Physiological Equivalent Temperature) and wet-bulb temperature are computed daily per US city grid point in parallel. PET comes from ARCO-ERA5 on Google Cloud Run (`google_era5.py`, `era5-worker`). Wet-bulb comes from NLDAS-2 forcing data via the Davies-Jones (2008) method (`wetbulb.py`), which stores both the daily average and daily max; it's fetched via the Giovanni Time Series API directly on GitHub Actions runners (`giovanni.py`, the default), with the original per-hour granule-download path (`nldas.py`, `nldas-worker` on Cloud Run) kept as a fallback.
2. **Store**: Workers write parquet shards directly to an AWS S3 bucket, partitioned by year.
3. **Load**: Parquet shards are COPYed into a staging table and upserted into Northflank-hosted Postgres on the `(location_id, date)` unique key, so every load is idempotent and safe to re-run.
4. **Analyze**: Materialized views in Postgres generate summary statistics, historical trend comparisons, and long-term forecasts. Yearly appends refresh the views in place (`refresh_views.py`); full rebuilds recreate them from `create_views.sql`.

## Entry points

- `pipeline.py` orchestrates the full pipeline locally (fetch → compute → load → views). `pull_all.sh`, `pull_pet.sh`, `pull_wetbulb.sh`, and `pull_both.sh` are thin wrappers around it, and the historical environment variables (`MODE`, `YEARS`, `PRODUCTS`, `USE_CLOUD_RUN`, ...) are still honored as defaults. Run `pipeline.py --help` for the full flag list; `--mode smoke` processes a single week for quick validation.
- The `yearly_update` GitHub workflow appends the previous year every early April (once ARCO's final data covers it — a pre-flight check in `build_year_range.py --check-arco` fails fast otherwise) and opens a GitHub issue if any job fails.
- The `wetbulb_backfill` GitHub workflow (`workflow_dispatch`) reruns any historical wet-bulb range via Giovanni, city-sharded across parallel jobs, then truncates and reloads the `wetbulb` table.

## Wet-bulb data source: Giovanni vs. NLDAS-2 granules

`pipeline.py --wetbulb-source` selects how NLDAS-2 data is fetched:

- **`giovanni`** (default): `giovanni.py` fetches each city's whole multi-year hourly series from the [Giovanni Time Series API](https://api.giovanni.earthdata.nasa.gov/timeseries) in one request per variable (~1,500 requests total for 500 cities × 3 variables, versus ~219,000 hourly granule downloads for the old path) and always runs as local subprocesses — even when `--use-cloud-run` is set for the PET/ERA5 path — writing straight to S3. This is what the `yearly_update` and `wetbulb_backfill` workflows use, for free on GitHub Actions instead of billed Cloud Run compute.
  - Each city is asked for the same land-snapped NLDAS-2 grid cell that the granule path resolves via its static land mask, using the checked-in `cities_nldas_cells.csv` (regenerate it with `make_nldas_cell_map.py` if `cities.csv`'s city list changes; it needs the `nldas` optional dependency group and a live granule download).
- **`granules`**: the original `nldas.py` path — downloads full hourly NLDAS-2 CONUS granules over HTTPS and extracts each city's nearest grid cell. Kept as a fallback in case the Giovanni API regresses; runs on Cloud Run (`nldas-worker`) when `--use-cloud-run` is set. `nldas-worker` isn't deployed by `cloudrun_provision.sh` by default — set `DEPLOY_NLDAS_WORKER=1` to deploy it.

## Credentials

Besides the Postgres/AWS/GCP credentials, both wet-bulb paths need a free [NASA Earthdata](https://urs.earthdata.nasa.gov/) account with the "NASA GESDISC DATA ARCHIVE" application authorized. Set `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` in `.env` for local runs, or as GitHub Actions / Secret Manager secrets (`cloudrun_provision.sh` wires them into the `nldas-worker` job when `DEPLOY_NLDAS_WORKER=1`).

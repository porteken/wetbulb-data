# Wetbulb Data Pipeline

This is the automated pipeline for processing wet-bulb temperature data and loading it into the web application. It was split out of the `pet-data` pipeline repo, so it still shares the underlying orchestration (`pipeline.py`, `load.py`, `historical_pet_update.py`) and the Postgres/S3 infrastructure with that project's PET pipeline.

1. **Compute**: Google Cloud Run job tasks pull weather data and compute daily PET (Physiological Equivalent Temperature) and wet-bulb temperature per US city grid point in parallel. PET comes from ARCO-ERA5 (`google_era5.py`, `era5-worker`); wet-bulb comes from NLDAS-2 hourly forcing data via the Davies-Jones (2008) method (`nldas.py`, `nldas-worker`), which stores both the daily average and daily max.
2. **Store**: Workers write parquet shards directly to an AWS S3 bucket, partitioned by year.
3. **Load**: Parquet shards are COPYed into a staging table and upserted into Northflank-hosted Postgres on the `(location_id, date)` unique key, so every load is idempotent and safe to re-run.
4. **Analyze**: Materialized views in Postgres generate summary statistics, historical trend comparisons, and long-term forecasts. Yearly appends refresh the views in place (`refresh_views.py`); full rebuilds recreate them from `create_views.sql`.

## Entry points

- `pipeline.py` orchestrates the full pipeline locally (fetch → compute → load → views). `pull_all.sh`, `pull_pet.sh`, `pull_wetbulb.sh`, and `pull_both.sh` are thin wrappers around it, and the historical environment variables (`MODE`, `YEARS`, `PRODUCTS`, `USE_CLOUD_RUN`, ...) are still honored as defaults. Run `pipeline.py --help` for the full flag list; `--mode smoke` processes a single week for quick validation.
- The `yearly_update` GitHub workflow appends the previous year every early April (once ARCO's final data covers it — a pre-flight check in `build_year_range.py --check-arco` fails fast otherwise) and opens a GitHub issue if any job fails.

## Credentials

Besides the Postgres/AWS/GCP credentials, the NLDAS-2 worker needs a free [NASA Earthdata](https://urs.earthdata.nasa.gov/) account with the "NASA GESDISC DATA ARCHIVE" application authorized. Set `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` in `.env` for local runs, or as GitHub Actions / Secret Manager secrets for Cloud Run (`cloudrun_provision.sh` wires them into the `nldas-worker` job).

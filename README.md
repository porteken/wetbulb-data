# Wetbulb Data Pipeline

This is the automated pipeline for processing North American and European
wet-bulb data and loading it into the
[Web Application](https://wetbulb-app.vercel.app/). The repository for the web
application itself is
[porteken/wetbulb-app](https://github.com/porteken/wetbulb-app).

1. **Compute**: US, Canadian, and European observations use NOAA GHCNh with year-specific, multi-station candidate maps throughout the 2000-forward analysis period. This also avoids the upstream ISD archive's August 2025 cutoff. Canadian cities are supplemented from Environment and Climate Change Canada's official hourly archive. Station days require at least 20 distinct hourly observations; ERA5-Land or NLDAS fills remaining gaps.
2. **Store**: Workers write parquet shards directly to an AWS S3 bucket, partitioned by year.
3. **Load**: Parquet shards are copied into a staging table and upserted into Northflank-hosted Postgres, so every load is idempotent and safe to re-run.
4. **Analyze**: Materialized views in Postgres generate summary statistics, historical trend comparisons, and long-term forecasts. Yearly appends refresh the views in place.

The `Daily Wetbulb Update` GitHub Actions workflow refreshes the current year
for North America and Europe every day at 20:27 UTC.

Station parquet rows include the selected station ID, station distance,
elevation difference, distinct observed-hour count, and quality classification.
When duplicate city-days are loaded, deterministic precedence is ECCC, GHCNh,
ISD, NLDAS, then ERA5-Land. Run `audit_station_coverage.py` against one or more
`wetbulb_data_csv` roots to measure the remaining station-versus-grid share by
year and city.

```bash
uv run python audit_station_coverage.py \
  --root na=na/na-census-2025-csd-2021/wetbulb_data_csv \
  --root eu=eu/wetbulb_data_csv \
  --cities-csv cities_na.csv --cities-csv cities_eu.csv
```

To replace existing legacy station shards during the historical migration, run
the regional backfills with `FORCE_STATION_BACKFILL=1` for North America and
`EU_FORCE_STATION_BACKFILL=1` for Europe. Without those flags, completed shard
files remain resumable and are intentionally skipped.

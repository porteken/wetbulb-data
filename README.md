# Wetbulb Data Pipeline

This is the automated pipeline for processing wetbulb data and loading it into the [Web Application](https://wetbulb-app.vercel.app/). The repository for the web application itself is [porteken/wetbulb-app](https://github.com/porteken/wetbulb-app).

1. **Compute**: US wet-bulb comes from NOAA ISD and Canadian wet-bulb from ECCC Historical Climate Data. Both use the pressure-aware Davies-Jones (2008) method and store daily average and maximum values. NOAA LCD/NLDAS-2 remain US fallbacks; ERA5-Land fills material Canadian station gaps.
2. **Store**: Workers write parquet shards directly to an AWS S3 bucket, partitioned by year.
3. **Load**: Parquet shards are Copied into a staging table and upserted into Northflank-hosted Postgres, so every load is idempotent and safe to re-run.
4. **Analyze**: Materialized views in Postgres generate summary statistics, historical trend comparisons, and long-term forecasts. Yearly appends refresh the views in place.

## Canada

Canadian locations are the 100 largest 2021 Statistics Canada population
centres. Their stable location IDs are 500–599, following the US 0–499
range, and province abbreviations occupy the existing `state` column.
`cities_ca.py` uses GeoSuite population counts and official representative
place coordinates; cross-provincial centres such as Ottawa–Gatineau use a
combined value such as `ON-QC`.

Generate the inputs:

```bash
uv run python cities_ca.py
uv run python make_eccc_station_map.py
uv run python locations.py \
  --cities-csv cities.csv cities_ca.csv \
  --out locations.csv
```

Pull ECCC observations into a Canada-specific shard tree (separate output
directories prevent Canadian and US Parquet filenames from colliding):

```bash
./pull_wetbulb_ca.sh
./pull_wetbulb_ca.sh --yes load views
```

The wrapper is resumable and supports individual `cities`, `crosswalk`,
`pull`, `gapfill`, `load`, and `views` steps. Run
`./pull_wetbulb_ca.sh --help` for concurrency, year, sharding, and dry-run
options.

ECCC timestamps marked LST are aggregated as local-standard calendar days.
The worker rejects missing/erroneous temperature, dew-point, and pressure
flags, derives station pressure from sea-level pressure and elevation when
necessary, and tries up to three nearby stations in order for each year.
After station shards are present, the existing ERA5-Land gap filler can be
pointed at the Canadian files:

```bash
uv run python era5land.py \
  --cities-csv cities_ca.csv \
  --station-map-csv cities_ca_eccc_stations.csv \
  --start-year 1991 --end-year 2025 \
  --out-dir ca
```

Existing databases must apply `migrate_wetbulb_source_eccc.sql` before
loading rows whose provenance is `eccc`; fresh databases get the same
constraint from `create_tables.sql`.

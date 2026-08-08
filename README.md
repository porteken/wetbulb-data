# Wetbulb Data Pipeline

This is the automated pipeline for processing North American and European
wet-bulb data and loading it into the
[Web Application](https://wetbulb-app.vercel.app/). The repository for the web
application itself is
[porteken/wetbulb-app](https://github.com/porteken/wetbulb-app).

1. **Compute**: US, Canadian, and European observations use NOAA GHCNh with year-specific, multi-station candidate maps throughout the 1990-forward analysis period. Each city has one stable reference station selected from the latest decade of the complete candidate catalog. Secondary stations are adjusted to that baseline using robust, same-day monthly overlap before they may fill a missing reference day; an uncalibrated secondary is rejected. Historical Airways and Synoptic report families are retained alongside modern METAR reports. Canadian cities are supplemented from Environment and Climate Change Canada's official hourly archive. Station days require at least 20 distinct hourly observations; ERA5-Land or NLDAS fills remaining gaps.
2. **Store**: Workers write parquet shards directly to an AWS S3 bucket, partitioned by year.
3. **Load**: Parquet shards are copied into a staging table and upserted into Northflank-hosted Postgres, so every load is idempotent and safe to re-run.
4. **Analyze**: Materialized views in Postgres generate summary statistics, historical trend comparisons, and long-term forecasts. Yearly appends refresh the views in place.

The `Daily Wetbulb Update` GitHub Actions workflow refreshes the current year
for North America and Europe every day at 20:27 UTC.



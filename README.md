# Wetbulb Data Pipeline

This is the automated pipeline for processing wetbulb data and loading it into the [Web Application](https://wetbulb-app.vercel.app/). The repository for the web application itself is [porteken/wetbulb-app](https://github.com/porteken/wetbulb-app).

1. **Compute**: Wet-bulb comes from NOAA's ISD Global Hourly station observations via the Davies-Jones (2008) method, which stores both the daily average and daily max; it's fetched from NCEI's bulk archive directly on GitHub Actions runners, with NOAA LCD v2 and NLDAS-2 gridded reanalysis kept as fallback sources.
2. **Store**: Workers write parquet shards directly to an AWS S3 bucket, partitioned by year.
3. **Load**: Parquet shards are Copied into a staging table and upserted into Northflank-hosted Postgres, so every load is idempotent and safe to re-run.
4. **Analyze**: Materialized views in Postgres generate summary statistics, historical trend comparisons, and long-term forecasts. Yearly appends refresh the views in place.
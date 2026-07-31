# Wetbulb Data Pipeline

This is the automated pipeline for processing North American and European
wet-bulb data and loading it into the
[Web Application](https://wetbulb-app.vercel.app/). The repository for the web
application itself is
[porteken/wetbulb-app](https://github.com/porteken/wetbulb-app).

1. **Compute**: US, Canadian, and European observations use NOAA GHCNh from 2026 onward, while earlier station history retains its ISD provenance. The pressure-aware Davies-Jones (2008) method stores daily average and maximum values; Google Earth Engine's ERA5-Land fills every station gap using only complete 24-hour local-standard days.
2. **Store**: Workers write parquet shards directly to an AWS S3 bucket, partitioned by year.
3. **Load**: Parquet shards are copied into a staging table and upserted into Northflank-hosted Postgres, so every load is idempotent and safe to re-run.
4. **Analyze**: Materialized views in Postgres generate summary statistics, historical trend comparisons, and long-term forecasts. Yearly appends refresh the views in place.

The `Daily Wetbulb Update` GitHub Actions workflow refreshes the current year
for North America and Europe every day at 20:27 UTC.

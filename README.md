# Wetbulb Data Pipeline

This is the automated pipeline for processing wetbulb data and loading it into the [Web Application](https://wetbulb-app.vercel.app/). The repository for the web application itself is [porteken/wetbulb-app](https://github.com/porteken/wetbulb-app).

1. **Compute**: US wet-bulb comes from NOAA ISD and Canadian wet-bulb from ECCC Historical Climate Data. Both use the pressure-aware Davies-Jones (2008) method and store daily average and maximum values. NOAA LCD/NLDAS-2 remain US fallbacks; ERA5-Land fills material Canadian station gaps.
2. **Store**: Workers write parquet shards directly to an AWS S3 bucket, partitioned by year.
3. **Load**: Parquet shards are Copied into a staging table and upserted into Northflank-hosted Postgres, so every load is idempotent and safe to re-run.
4. **Analyze**: Materialized views in Postgres generate summary statistics, historical trend comparisons, and long-term forecasts. Yearly appends refresh the views in place.
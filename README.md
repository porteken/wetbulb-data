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

## Database maintenance

Older deployments may still contain the superseded `pet` table and PET views.
They are not used by the wet-bulb application and occupy about 1.1 GB in the
current production database. Cleanup is deliberately opt-in and avoids
`CASCADE`, so an unexpected new dependency stops the migration:

```bash
# North American wrapper
./pull_wetbulb.sh --confirm-db-write cleanup views

# European wrapper (the cleanup is idempotent and only needs to run once)
./pull_wetbulb_eu.sh --yes cleanup views
```

Normal `views` runs refresh only `wetbulb_*` materialized views. This prevents
unrelated legacy materialized views from consuming maintenance time and disk.

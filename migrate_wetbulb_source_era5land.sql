-- One-off migration for an already-deployed database: `create_tables.sql`'s
-- `CREATE TABLE IF NOT EXISTS` won't touch an existing `public.wetbulb`
-- table, so its `wetbulb_source_check` constraint needs updating here to
-- allow `era5land` (the EU ERA5-Land gap-fill provenance stamp; see
-- `era5land.py`). `NOT VALID` + a separate `VALIDATE CONSTRAINT` keeps the
-- `ADD CONSTRAINT` step to a brief lock; the validation scan runs without
-- blocking concurrent reads/writes.
ALTER TABLE public.wetbulb DROP constraint IF EXISTS wetbulb_source_check ;

ALTER TABLE public.wetbulb ADD CONSTRAINT wetbulb_source_check
CHECK (source IN ('eccc',
'ghcnh',
'isd',
'nldas',
'lcd',
'giovanni',
'era5land')) NOT VALID ;

ALTER TABLE public.wetbulb VALIDATE CONSTRAINT wetbulb_source_check ;

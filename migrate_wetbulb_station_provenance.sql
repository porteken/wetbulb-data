-- Add auditable station provenance without rewriting existing grid-filled rows.
ALTER TABLE public.wetbulb ADD column IF NOT EXISTS station_id text ;
ALTER TABLE public.wetbulb ADD COLUMN IF NOT EXISTS station_distance_km real ;
ALTER TABLE public.wetbulb
ADD COLUMN IF NOT EXISTS station_elevation_difference_m real ;
ALTER TABLE public.wetbulb ADD COLUMN IF NOT EXISTS observed_hours smallint ;
ALTER TABLE public.wetbulb ADD COLUMN IF NOT EXISTS station_quality text ;

ALTER TABLE public.wetbulb DROP CONSTRAINT IF EXISTS wetbulb_source_check ;
ALTER TABLE public.wetbulb ADD CONSTRAINT wetbulb_source_check CHECK (
source IN ('eccc', 'ghcnh', 'isd', 'nldas', 'lcd', 'giovanni', 'era5land')
) NOT VALID ;
ALTER TABLE public.wetbulb VALIDATE CONSTRAINT wetbulb_source_check ;

ALTER TABLE public.wetbulb
DROP CONSTRAINT IF EXISTS wetbulb_station_quality_check ;
ALTER TABLE public.wetbulb ADD CONSTRAINT wetbulb_station_quality_check CHECK (
station_quality IS NULL OR station_quality IN ('complete', 'sparse')
) NOT VALID ;
ALTER TABLE public.wetbulb VALIDATE CONSTRAINT wetbulb_station_quality_check ;

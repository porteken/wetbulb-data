-- Allow NOAA GHCNh provenance in an already-deployed wetbulb table.
ALTER TABLE public.wetbulb DROP constraint IF EXISTS wetbulb_source_check ;

ALTER TABLE public.wetbulb ADD CONSTRAINT wetbulb_source_check
CHECK (
source IN ('ghcnh', 'isd', 'nldas', 'lcd', 'giovanni', 'era5land')
) NOT VALID ;

ALTER TABLE public.wetbulb VALIDATE CONSTRAINT wetbulb_source_check ;

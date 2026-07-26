BEGIN;

ALTER TABLE public.wetbulb
DROP constraint IF EXISTS wetbulb_source_check ;

ALTER TABLE public.wetbulb
ADD CONSTRAINT wetbulb_source_check
CHECK (source IN ('isd', 'eccc', 'nldas', 'lcd', 'giovanni', 'era5land')) ;

COMMIT ;

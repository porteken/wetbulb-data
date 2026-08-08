DROP MATERIALIZED VIEW IF EXISTS public.wetbulb_forecast_scenarios CASCADE ;

CREATE MATERIALIZED VIEW public.wetbulb_forecast_scenarios AS
SELECT * FROM public.wetbulb_gmst_forecast ('mean') ;

CREATE UNIQUE INDEX wetbulb_forecast_scenarios_uidx
ON public.wetbulb_forecast_scenarios (location_id, year, season, scenario) ;

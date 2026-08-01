-- One-time cleanup for the PET schema superseded by public.wetbulb.
--
-- Deliberately avoid CASCADE: if a new object has acquired a dependency on one
-- of these legacy objects, this migration must stop instead of deleting it.
SET LOCAL lock_timeout = '5s' ;

DROP VIEW IF EXISTS public.city_rankings_view ;

DROP MATERIALIZED VIEW IF EXISTS public.pet_forecast_max ;
DROP MATERIALIZED VIEW IF EXISTS public.pet_forecast ;
DROP MATERIALIZED VIEW IF EXISTS public.pet_year_stats ;

DROP FUNCTION IF EXISTS public.pet_forecast_for_location (
bigint,
integer [],
double precision []
) ;
DROP FUNCTION IF EXISTS public.pet_forecast_for_location (
integer,
integer [],
double precision []
) ;
DROP FUNCTION IF EXISTS public.pet_select_best_model (
integer [],
double precision []
) ;
DROP FUNCTION IF EXISTS public.pet_rolling_origin_rmse (
integer [],
double precision [],
integer,
integer
) ;
DROP FUNCTION IF EXISTS public.pet_fit_polynomial_model (
integer [],
double precision [],
integer
) ;
DROP FUNCTION IF EXISTS public.pet_model_predict (
integer,
double precision,
double precision,
double precision,
double precision,
double precision
) ;
DROP FUNCTION IF EXISTS public.pet_model_acceleration (
integer,
double precision
) ;
DROP FUNCTION IF EXISTS public.pet_model_slope (
integer,
double precision,
double precision,
double precision,
double precision
) ;
DROP FUNCTION IF EXISTS public.pet_array_median (integer []) ;
DROP FUNCTION IF EXISTS public.pet_is_finite (double precision) ;
DROP FUNCTION IF EXISTS public.pet_infinity () ;
DROP FUNCTION IF EXISTS public.pet_annual_season () ;
DROP FUNCTION IF EXISTS public.pet_season (date) ;
DROP FUNCTION IF EXISTS public.pet_fall () ;
DROP FUNCTION IF EXISTS public.pet_spring () ;
DROP FUNCTION IF EXISTS public.pet_summer () ;
DROP FUNCTION IF EXISTS public.pet_winter () ;

DROP TYPE IF EXISTS public.pet_trend_model ;
DROP TABLE IF EXISTS public.pet ;

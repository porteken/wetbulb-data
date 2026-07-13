-- squawk-ignore-file require-concurrent-index-creation

-- SET LOCAL keeps these scoped to this file's transaction.
SET LOCAL statement_timeout = '0' ;
SET LOCAL lock_timeout = '0' ;

CREATE OR REPLACE FUNCTION public.pet_array_median (years integer [])
RETURNS double precision
LANGUAGE SQL
IMMUTABLE
AS $$
WITH sorted AS (
SELECT ARRAY (
SELECT unnest ($1)
ORDER BY 1
) AS sorted_years
)
SELECT CASE
WHEN array_length (sorted_years, 1) IS NULL OR array_length (sorted_years, 1) = 0 THEN NULL
WHEN MOD (array_length (sorted_years, 1), 2) = 1 THEN sorted_years[(array_length (sorted_years, 1) + 1) / 2]::double precision
ELSE (sorted_years[array_length (sorted_years, 1) / 2] + sorted_years[(array_length (sorted_years, 1) / 2) + 1]) / 2.0
END
FROM sorted
$$ ;

CREATE OR REPLACE FUNCTION public.pet_infinity ()
RETURNS double precision
LANGUAGE SQL
IMMUTABLE
AS $$
SELECT 'Infinity'::double precision
$$ ;

CREATE OR REPLACE FUNCTION public.pet_annual_season ()
RETURNS text
LANGUAGE SQL
IMMUTABLE
AS $$
SELECT 'Annual'::text
$$ ;

CREATE OR REPLACE FUNCTION public.pet_winter ()
RETURNS text
LANGUAGE SQL
IMMUTABLE
AS $$
SELECT 'Winter'::text
$$ ;

CREATE OR REPLACE FUNCTION public.pet_spring ()
RETURNS text
LANGUAGE SQL
IMMUTABLE
AS $$
SELECT 'Spring'::text
$$ ;

CREATE OR REPLACE FUNCTION public.pet_summer ()
RETURNS text
LANGUAGE SQL
IMMUTABLE
AS $$
SELECT 'Summer'::text
$$ ;

CREATE OR REPLACE FUNCTION public.pet_fall ()
RETURNS text
LANGUAGE SQL
IMMUTABLE
AS $$
SELECT 'Fall'::text
$$ ;

CREATE OR REPLACE FUNCTION public.pet_season (input_date date)
RETURNS text
LANGUAGE SQL
IMMUTABLE
AS $$
SELECT CASE
WHEN EXTRACT (MONTH FROM input_date)::int IN (12, 1, 2) THEN public.pet_winter ()
WHEN EXTRACT (MONTH FROM input_date)::int IN (3, 4, 5) THEN public.pet_spring ()
WHEN EXTRACT (MONTH FROM input_date)::int IN (6, 7, 8) THEN public.pet_summer ()
ELSE public.pet_fall ()
END
$$ ;

CREATE OR REPLACE FUNCTION public.pet_is_finite (value double precision)
RETURNS boolean
LANGUAGE SQL
IMMUTABLE
AS $$
SELECT $1 IS NOT NULL AND $1 = $1 AND abs ($1) < 'Infinity'::double precision
$$ ;

DO $$
BEGIN
IF NOT EXISTS (
SELECT
1
FROM pg_catalog.pg_type AS t
JOIN pg_catalog.pg_namespace AS n ON n.oid = t.typnamespace
WHERE n.nspname = 'public'
AND t.typname = 'pet_trend_model'
) THEN
CREATE TYPE public.pet_trend_model AS (
degree integer,
coef2 double precision,
coef1 double precision,
coef0 double precision,
reference_year double precision,
rmse double precision,
cv_rmse double precision
) ;
END IF ;
END $$ ;

CREATE OR REPLACE FUNCTION public.pet_model_predict (
degree integer,
coef2 double precision,
coef1 double precision,
coef0 double precision,
reference_year double precision,
target_year double precision
)
RETURNS double precision
LANGUAGE SQL
IMMUTABLE
AS $$
SELECT CASE
WHEN $1 = 2 THEN ($2 * POWER($6 - $5, 2)) + ($3 * ($6 - $5)) + $4
ELSE ($3 * ($6 - $5)) + $4
END
$$ ;

CREATE OR REPLACE FUNCTION public.pet_model_slope (
degree integer,
coef2 double precision,
coef1 double precision,
reference_year double precision,
target_year double precision
)
RETURNS double precision
LANGUAGE SQL
IMMUTABLE
AS $$
SELECT CASE
WHEN $1 = 2 THEN (2.0 * $2 * ($5 - $4)) + $3
ELSE $3
END
$$ ;

CREATE OR REPLACE FUNCTION public.pet_model_acceleration (
degree integer,
coef2 double precision
)
RETURNS double precision
LANGUAGE SQL
IMMUTABLE
AS $$
SELECT CASE
WHEN $1 = 2 THEN 2.0 * $2
ELSE 0.0
END
$$ ;

CREATE OR REPLACE FUNCTION public.pet_fit_polynomial_model (
years integer [],
pet_values double precision [],
degree integer
)
RETURNS public.pet_trend_model
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
result public.pet_trend_model ;
n integer := COALESCE(array_length(years, 1), 0) ;
idx integer ;
x double precision ;
y double precision ;
sx double precision := 0.0 ;
sy double precision := 0.0 ;
sxx double precision := 0.0 ;
sxy double precision := 0.0 ;
sxxx double precision := 0.0 ;
sxxxx double precision := 0.0 ;
sxxy double precision := 0.0 ;
reference_year double precision ;
denom double precision ;
sum_sq double precision := 0.0 ;
fitted double precision ;
m11 double precision ;
m12 double precision ;
m13 double precision ;
m21 double precision ;
m22 double precision ;
m23 double precision ;
m31 double precision ;
m32 double precision ;
m33 double precision ;
b1 double precision ;
b2 double precision ;
b3 double precision ;
det double precision ;
det_coef2 double precision ;
det_coef1 double precision ;
det_coef0 double precision ;
BEGIN
IF n = 0 OR n <> COALESCE(array_length(pet_values, 1), 0) OR n <= degree THEN
RETURN NULL ;
END IF ;

reference_year := public.pet_array_median(years) ;

FOR idx IN 1..n LOOP
x := years[idx] - reference_year ;
y := pet_values[idx] ;

sx := sx + x ;
sy := sy + y ;
sxx := sxx + (x * x) ;
sxy := sxy + (x * y) ;
IF degree = 2 THEN
sxxx := sxxx + (x * x * x) ;
sxxxx := sxxxx + (x * x * x * x) ;
sxxy := sxxy + (x * x * y) ;
END IF ;
END LOOP ;

result.degree := degree ;
result.reference_year := reference_year ;

IF degree = 1 THEN
denom := (n::double precision * sxx) - (sx * sx) ;
IF abs(denom) < 1e-12 THEN
RETURN NULL ;
END IF ;

result.coef2 := 0.0 ;
result.coef1 := ((n::double precision * sxy) - (sx * sy)) / denom ;
result.coef0 := (sy - (result.coef1 * sx)) / n::double precision ;
ELSIF degree = 2 THEN
m11 := sxxxx ;
m12 := sxxx ;
m13 := sxx ;
m21 := sxxx ;
m22 := sxx ;
m23 := sx ;
m31 := sxx ;
m32 := sx ;
m33 := n::double precision ;

b1 := sxxy ;
b2 := sxy ;
b3 := sy ;

det :=
(m11 * ((m22 * m33) - (m23 * m32)))
- (m12 * ((m21 * m33) - (m23 * m31)))
+ (m13 * ((m21 * m32) - (m22 * m31))) ;

IF abs(det) < 1e-12 THEN
RETURN NULL ;
END IF ;

det_coef2 :=
(b1 * ((m22 * m33) - (m23 * m32)))
- (m12 * ((b2 * m33) - (m23 * b3)))
+ (m13 * ((b2 * m32) - (m22 * b3))) ;

det_coef1 :=
(m11 * ((b2 * m33) - (m23 * b3)))
- (b1 * ((m21 * m33) - (m23 * m31)))
+ (m13 * ((m21 * b3) - (b2 * m31))) ;

det_coef0 :=
(m11 * ((m22 * b3) - (b2 * m32)))
- (m12 * ((m21 * b3) - (b2 * m31)))
+ (b1 * ((m21 * m32) - (m22 * m31))) ;

result.coef2 := det_coef2 / det ;
result.coef1 := det_coef1 / det ;
result.coef0 := det_coef0 / det ;
ELSE
RAISE EXCEPTION 'Unsupported polynomial degree %', degree ;
END IF ;

FOR idx IN 1..n LOOP
fitted := public.pet_model_predict(
result.degree,
result.coef2,
result.coef1,
result.coef0,
result.reference_year,
years[idx]::double precision
) ;
sum_sq := sum_sq + POWER(pet_values[idx] - fitted, 2) ;
END LOOP ;

result.rmse := SQRT(sum_sq / n::double precision) ;
result.cv_rmse := public.pet_infinity() ;
RETURN result ;
END
$$ ;

CREATE OR REPLACE FUNCTION public.pet_rolling_origin_rmse (
years integer [],
pet_values double precision [],
degree integer,
min_train_years integer
)
RETURNS double precision
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
n integer := COALESCE(array_length(years, 1), 0) ;
pred_idx integer ;
error_count integer := 0 ;
sum_sq double precision := 0.0 ;
model public.pet_trend_model ;
predicted double precision ;
err double precision ;
BEGIN
IF n < (min_train_years + 3) THEN
RETURN public.pet_infinity() ;
END IF ;

FOR pred_idx IN (min_train_years + 1)..n LOOP
model := public.pet_fit_polynomial_model(
years[1:(pred_idx - 1)],
pet_values[1:(pred_idx - 1)],
degree
) ;

IF model IS NULL THEN
CONTINUE ;
END IF ;

predicted := public.pet_model_predict(
model.degree,
model.coef2,
model.coef1,
model.coef0,
model.reference_year,
years[pred_idx]::double precision
) ;
err := pet_values[pred_idx] - predicted ;
sum_sq := sum_sq + (err * err) ;
error_count := error_count + 1 ;
END LOOP ;

IF error_count < 5 THEN
RETURN public.pet_infinity() ;
END IF ;

RETURN SQRT(sum_sq / error_count::double precision) ;
END
$$ ;

CREATE OR REPLACE FUNCTION public.pet_select_best_model (
years integer [],
pet_values double precision []
)
RETURNS public.pet_trend_model
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
linear_model public.pet_trend_model ;
BEGIN
linear_model := public.pet_fit_polynomial_model(years, pet_values, 1) ;
IF linear_model IS NULL THEN
RETURN NULL ;
END IF ;

linear_model.cv_rmse := public.pet_rolling_origin_rmse(years, pet_values, 1, 12) ;
-- Option C from forecast_analysis.md: keep the public forecast linear-only.
RETURN linear_model ;
END
$$ ;

CREATE OR REPLACE FUNCTION public.pet_forecast_for_location (
input_location_id integer,
years integer [],
pet_values double precision []
)
RETURNS TABLE (
location_id integer,
year smallint,
pet numeric (5, 1),
lower numeric (5, 1),
upper numeric (5, 1),
model_type text,
full_years_used smallint,
warming_rate numeric (5, 2),
acceleration numeric (6, 3)
)
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
model public.pet_trend_model ;
n integer := COALESCE(array_length(years, 1), 0) ;
last_year integer ;
future_year integer ;
forecast_end_year constant integer := 2100 ;
acceleration_damping_start_years constant integer := 15 ;
acceleration_damping_end_years constant integer := 35 ;
prediction_interval_z constant double precision := 1.2816 ;
uncertainty_growth_denominator constant double precision := 10.0 ;
uncertainty_growth_cap_years constant double precision := 25.0 ;
quadratic_uncertainty_denominator constant double precision := 30.0 ;
quadratic_degree constant integer := 2 ;
start_year integer ;
end_year integer ;
raw_pet double precision ;
future_pet double precision ;
y_start double precision ;
slope_start double precision ;
linear_tail double precision ;
weight double precision ;
base_rmse double precision ;
horizon double precision ;
capped_horizon double precision ;
quadratic_term double precision ;
sigma double precision ;
out_model_type text ;
out_full_years_used smallint ;
out_acceleration numeric (6, 3) ;
BEGIN
IF n = 0 OR n <> COALESCE(array_length(pet_values, 1), 0) OR n < 10 THEN
RETURN ;
END IF ;

model := public.pet_select_best_model(years, pet_values) ;
IF model IS NULL THEN
RETURN ;
END IF ;

last_year := years[n] ;
IF last_year >= forecast_end_year THEN
RETURN ;
END IF ;

base_rmse := GREATEST(
model.rmse,
CASE
WHEN public.pet_is_finite(model.cv_rmse) THEN model.cv_rmse
ELSE 0.0
END
) ;
out_model_type := CASE
WHEN model.degree = quadratic_degree THEN 'quadratic'
ELSE 'linear'
END ;
out_full_years_used := n::smallint ;
out_acceleration := ROUND(
public.pet_model_acceleration(model.degree, model.coef2)::numeric,
3
)::numeric(6, 3) ;

FOR future_year IN (last_year + 1)..forecast_end_year LOOP
raw_pet := public.pet_model_predict(
model.degree,
model.coef2,
model.coef1,
model.coef0,
model.reference_year,
future_year::double precision
) ;
future_pet := raw_pet ;

IF model.degree = quadratic_degree THEN
start_year := LEAST(last_year + acceleration_damping_start_years, forecast_end_year) ;
end_year := LEAST(last_year + acceleration_damping_end_years, forecast_end_year) ;

IF start_year < end_year THEN
y_start := public.pet_model_predict(
model.degree,
model.coef2,
model.coef1,
model.coef0,
model.reference_year,
start_year::double precision
) ;
slope_start := public.pet_model_slope(
model.degree,
model.coef2,
model.coef1,
model.reference_year,
start_year::double precision
) ;
linear_tail := y_start + ((future_year - start_year)::double precision * slope_start) ;
weight := LEAST(
1.0,
GREATEST(
0.0,
(future_year - start_year)::double precision / (end_year - start_year)::double precision
)
) ;
future_pet := (raw_pet * (1.0 - weight)) + (linear_tail * weight) ;
END IF ;
END IF ;

horizon := (future_year - last_year)::double precision ;
capped_horizon := LEAST(horizon, uncertainty_growth_cap_years) ;
quadratic_term := CASE
WHEN model.degree = quadratic_degree THEN POWER(capped_horizon / quadratic_uncertainty_denominator, 2)
ELSE 0.0
END ;
sigma := base_rmse * SQRT(1.0 + (capped_horizon / uncertainty_growth_denominator) + quadratic_term) ;

location_id := input_location_id ;
year := future_year::smallint ;
pet := ROUND(future_pet::numeric, 1)::numeric(5, 1) ;
lower := ROUND((future_pet - (prediction_interval_z * sigma))::numeric, 1)::numeric(5, 1) ;
upper := ROUND((future_pet + (prediction_interval_z * sigma))::numeric, 1)::numeric(5, 1) ;
model_type := out_model_type ;
full_years_used := out_full_years_used ;
warming_rate := ROUND(
public.pet_model_slope(
model.degree,
model.coef2,
model.coef1,
model.reference_year,
future_year::double precision
)::numeric,
2
)::numeric(5, 2) ;
acceleration := out_acceleration ;
RETURN NEXT ;
END LOOP ;
END
$$ ;

CREATE OR REPLACE FUNCTION public.pet_forecast_for_location (
input_location_id bigint,
years integer [],
pet_values double precision []
)
RETURNS TABLE (
location_id integer,
year smallint,
pet numeric (5, 1),
lower numeric (5, 1),
upper numeric (5, 1),
model_type text,
full_years_used smallint,
warming_rate numeric (5, 2),
acceleration numeric (6, 3)
)
LANGUAGE SQL
IMMUTABLE
AS $$
SELECT
forecast.location_id,
forecast.year,
forecast.pet,
forecast.lower,
forecast.upper,
forecast.model_type,
forecast.full_years_used,
forecast.warming_rate,
forecast.acceleration
FROM public.pet_forecast_for_location(
input_location_id::integer,
years,
pet_values
) AS forecast
$$ ;

CREATE MATERIALIZED VIEW public.pet_year_stats AS
WITH pet_with_seasons AS (
SELECT
p.location_id::smallint AS location_id,
EXTRACT (YEAR FROM p.date)::smallint AS year,
p_seasons.season,
p.pet::real AS pet
FROM public.pet AS p
CROSS JOIN LATERAL (
VALUES
(public.pet_annual_season ()),
(public.pet_season (p.date))
) AS p_seasons (season)
)
SELECT
location_id::integer AS location_id,
year::smallint AS year,
season,
ROUND (AVG (pet)::numeric, 1)::real AS avg_pet,
ROUND (MAX (pet)::numeric, 1)::real AS max_pet,
ROUND ((PERCENTILE_CONT (0.1) WITHIN GROUP (ORDER BY pet))::numeric,
1)::real AS p10,
ROUND ((PERCENTILE_CONT (0.9) WITHIN GROUP (ORDER BY pet))::numeric,
1)::real AS p90
FROM pet_with_seasons
GROUP BY
location_id,
year,
season ;

CREATE UNIQUE INDEX if not exists pet_year_stats_location_year_season_uidx
ON public.pet_year_stats (location_id, year, season) ;

CREATE INDEX if not exists pet_year_stats_year_idx
ON public.pet_year_stats (year) ;

CREATE INDEX if not exists pet_year_stats_season_idx
ON public.pet_year_stats (season) ;


CREATE MATERIALIZED VIEW public.pet_forecast AS
WITH pet_with_seasons AS (SELECT p.location_id::smallint AS location_id,
p.date,
p_seasons.season,
p.pet::real AS pet
FROM public.pet AS p
CROSS JOIN LATERAL (
VALUES (public.pet_annual_season ()),
(public.pet_season (p.date))
) AS p_seasons (season)),
daily_pet AS (SELECT p.location_id::smallint AS location_id,
p.date,
p.season,
AVG (p.pet)::real AS pet
FROM pet_with_seasons AS p
GROUP BY p.location_id,
p.date,
p.season),
yearly_pet AS (SELECT d.location_id::smallint AS location_id,
EXTRACT (YEAR FROM d.date)::smallint AS year,
d.season,
AVG (d.pet)::real AS pet,
COUNT (*)::int AS days_present
FROM daily_pet AS d
GROUP BY d.location_id,
EXTRACT (YEAR FROM d.date)::smallint,
d.season),
complete_yearly_pet AS (SELECT y.location_id::smallint AS location_id,
y.year::smallint AS year,
y.season,
y.pet::real AS pet
FROM yearly_pet AS y
-- noqa: disable=LT01
WHERE y.days_present = CASE
WHEN y.season = public.pet_annual_season () THEN CASE
WHEN
MOD (y.year, 4) =
0
AND
(MOD (y.year, 100) != 0 OR MOD (y.year, 400) = 0)
THEN 366
ELSE 365
END
WHEN y.season = public.pet_winter () THEN CASE
WHEN
MOD (y.year, 4) =
0
AND
(MOD (y.year, 100) != 0 OR MOD (y.year, 400) = 0)
THEN 91
ELSE 90
END
WHEN y.season IN (public.pet_spring (), public.pet_summer ())
THEN 92
ELSE 91
END
-- noqa: enable=LT01
),
forecast_inputs AS (SELECT location_id::smallint AS location_id,
season,
array_agg (year ORDER BY year) AS years,
array_agg (pet ORDER BY year) AS pet_values
FROM complete_yearly_pet
GROUP BY location_id,
season)
SELECT forecast.location_id::smallint AS location_id,
forecast.year::smallint AS year,
inputs.season,
forecast.pet::real AS pet,
forecast.lower::real AS lower,
forecast.upper::real AS upper,
forecast.model_type,
forecast.full_years_used,
forecast.warming_rate::real AS warming_rate,
forecast.acceleration::real AS acceleration
FROM forecast_inputs AS inputs
CROSS JOIN LATERAL public.pet_forecast_for_location (
inputs.location_id::integer,
inputs.years,
inputs.pet_values
) AS forecast ; ;

CREATE UNIQUE INDEX if not exists pet_forecast_location_year_season_uidx
ON public.pet_forecast (location_id, year, season) ;

CREATE INDEX if not exists pet_forecast_year_idx
ON public.pet_forecast (year) ;

CREATE INDEX if not exists pet_forecast_season_idx
ON public.pet_forecast (season) ;

CREATE MATERIALIZED VIEW public.pet_forecast_max AS
WITH pet_with_seasons AS (SELECT p.location_id::smallint AS location_id,
p.date,
p_seasons.season,
p.pet::real AS pet
FROM public.pet AS p
CROSS JOIN LATERAL (
VALUES (public.pet_annual_season ()),
(public.pet_season (p.date))
) AS p_seasons (season)),
daily_pet AS (SELECT p.location_id::smallint AS location_id,
p.date,
p.season,
MAX (p.pet)::real AS pet
FROM pet_with_seasons AS p
GROUP BY p.location_id,
p.date,
p.season),
yearly_pet AS (SELECT d.location_id::smallint AS location_id,
EXTRACT (YEAR FROM d.date)::smallint AS year,
d.season,
MAX (d.pet)::real AS pet,
COUNT (*)::int AS days_present
FROM daily_pet AS d
GROUP BY d.location_id,
EXTRACT (YEAR FROM d.date)::smallint,
d.season),
complete_yearly_pet AS (SELECT y.location_id::smallint AS location_id,
y.year::smallint AS year,
y.season,
y.pet::real AS pet
FROM yearly_pet AS y
-- noqa: disable=LT01
WHERE y.days_present = CASE
WHEN y.season = public.pet_annual_season () THEN CASE
WHEN
MOD (y.year, 4) =
0
AND
(MOD (y.year, 100) != 0 OR MOD (y.year, 400) = 0)
THEN 366
ELSE 365
END
WHEN y.season = public.pet_winter () THEN CASE
WHEN
MOD (y.year, 4) =
0
AND
(MOD (y.year, 100) != 0 OR MOD (y.year, 400) = 0)
THEN 91
ELSE 90
END
WHEN y.season IN (public.pet_spring (), public.pet_summer ())
THEN 92
ELSE 91
END
-- noqa: enable=LT01
),
forecast_inputs AS (SELECT location_id::smallint AS location_id,
season,
array_agg (year ORDER BY year) AS years,
array_agg (pet ORDER BY year) AS pet_values
FROM complete_yearly_pet
GROUP BY location_id,
season)
SELECT forecast.location_id::smallint AS location_id,
forecast.year::smallint AS year,
inputs.season,
forecast.pet::real AS pet,
forecast.lower::real AS lower,
forecast.upper::real AS upper,
forecast.model_type,
forecast.full_years_used,
forecast.warming_rate::real AS warming_rate,
forecast.acceleration::real AS acceleration
FROM forecast_inputs AS inputs
CROSS JOIN LATERAL public.pet_forecast_for_location (
inputs.location_id::integer,
inputs.years,
inputs.pet_values
) AS forecast ; ;

CREATE UNIQUE INDEX if not exists pet_forecast_max_location_year_season_uidx
ON public.pet_forecast_max (location_id, year, season) ;

CREATE INDEX if not exists pet_forecast_max_year_idx
ON public.pet_forecast_max (year) ;

CREATE INDEX if not exists pet_forecast_max_season_idx
ON public.pet_forecast_max (season) ;

CREATE VIEW public.city_rankings_view AS
WITH combined_yearly_avg AS (
SELECT
a.location_id::smallint AS location_id,
a.year::smallint AS year,
a.season,
a.avg_pet::real AS pet,
0 AS source_order
FROM public.pet_year_stats AS a
UNION ALL
SELECT
f.location_id::smallint AS location_id,
f.year::smallint AS year,
f.season,
f.pet::real AS pet,
1 AS source_order
FROM public.pet_forecast AS f
), deduplicated_yearly_avg AS (
SELECT DISTINCT ON (location_id, year, season)
location_id::smallint AS location_id,
year::smallint AS year,
season,
pet::real AS pet
FROM combined_yearly_avg
ORDER BY
location_id,
year,
season,
source_order
), year_2000_value AS (
SELECT
location_id::smallint AS location_id,
season,
pet::real AS pet
FROM deduplicated_yearly_avg
WHERE year = 2000
)
SELECT
s.location_id::smallint,
s.year::smallint,
s.season,
s.avg_pet::real AS avg_pet,
s.max_pet::real AS max_pet,
l.city,
l.state,
s.p10::real,
s.p90::real,
f.lower::real AS future_lower,
f.upper::real AS future_upper,
ROUND ((s.avg_pet - y2k.pet)::numeric, 2)::real AS change_from_2000
FROM public.pet_year_stats AS s
JOIN public.locations AS l ON l.id = s.location_id
LEFT JOIN public.pet_forecast AS f ON f.location_id = s.location_id
AND f.year = 2100::smallint
AND f.season = s.season
LEFT JOIN year_2000_value AS y2k ON y2k.location_id = s.location_id
AND y2k.season = s.season
WHERE
s.location_id > 0 ;

CREATE MATERIALIZED VIEW public.wetbulb_year_stats AS
WITH wetbulb_with_seasons AS (
SELECT
w.location_id::smallint AS location_id,
EXTRACT (YEAR FROM w.date)::smallint AS year,
w_seasons.season,
w.wetbulb::real AS wetbulb,
w.wetbulb_avg::real AS wetbulb_avg
FROM public.wetbulb AS w
CROSS JOIN LATERAL (
VALUES
(public.pet_annual_season ()),
(public.pet_season (w.date))
) AS w_seasons (season)
)
SELECT
location_id::integer AS location_id,
year::smallint AS year,
season,
ROUND (AVG (wetbulb)::numeric, 1)::real AS avg_wetbulb,
ROUND (MAX (wetbulb)::numeric, 1)::real AS max_wetbulb,
ROUND ((PERCENTILE_CONT (0.1) WITHIN GROUP (ORDER BY wetbulb))::numeric,
1)::real AS p10,
ROUND ((PERCENTILE_CONT (0.9) WITHIN GROUP (ORDER BY wetbulb))::numeric,
1)::real AS p90,
ROUND (AVG (wetbulb_avg)::numeric, 1)::real AS avg_wetbulb_avg,
ROUND (MAX (wetbulb_avg)::numeric, 1)::real AS max_wetbulb_avg,
ROUND ((PERCENTILE_CONT (0.1) WITHIN GROUP (ORDER BY wetbulb_avg))::numeric,
1)::real AS p10_avg,
ROUND ((PERCENTILE_CONT (0.9) WITHIN GROUP (ORDER BY wetbulb_avg))::numeric,
1)::real AS p90_avg
FROM wetbulb_with_seasons
GROUP BY
location_id,
year,
season ;

CREATE UNIQUE INDEX if not exists wetbulb_year_stats_location_year_season_uidx
ON public.wetbulb_year_stats (location_id, year, season) ;

CREATE INDEX if not exists wetbulb_year_stats_year_idx
ON public.wetbulb_year_stats (year) ;

CREATE INDEX if not exists wetbulb_year_stats_season_idx
ON public.wetbulb_year_stats (season) ;


CREATE MATERIALIZED VIEW public.wetbulb_forecast AS
WITH wetbulb_with_seasons AS (SELECT w.location_id::smallint AS location_id,
w.date,
w_seasons.season,
w.wetbulb_avg::real AS wetbulb
FROM public.wetbulb AS w
CROSS JOIN LATERAL (
VALUES (public.pet_annual_season ()),
(public.pet_season (w.date))
) AS w_seasons (season)
WHERE w.wetbulb_avg IS NOT NULL),
daily_wetbulb AS (SELECT w.location_id::smallint AS location_id,
w.date,
w.season,
AVG (w.wetbulb)::real AS wetbulb
FROM wetbulb_with_seasons AS w
GROUP BY w.location_id,
w.date,
w.season),
yearly_wetbulb AS (SELECT d.location_id::smallint AS location_id,
EXTRACT (YEAR FROM d.date)::smallint AS year,
d.season,
AVG (d.wetbulb)::real AS wetbulb,
COUNT (*)::int AS days_present
FROM daily_wetbulb AS d
GROUP BY d.location_id,
EXTRACT (YEAR FROM d.date)::smallint,
d.season),
complete_yearly_wetbulb AS (SELECT y.location_id::smallint AS location_id,
y.year::smallint AS year,
y.season,
y.wetbulb::real AS wetbulb
FROM yearly_wetbulb AS y
-- noqa: disable=LT01
WHERE y.days_present = CASE
WHEN y.season = public.pet_annual_season () THEN CASE
WHEN
MOD (y.year, 4) =
0
AND
(MOD (y.year, 100) != 0 OR MOD (y.year, 400) = 0)
THEN 366
ELSE 365
END
WHEN y.season = public.pet_winter () THEN CASE
WHEN
MOD (y.year, 4) =
0
AND
(MOD (y.year, 100) != 0 OR MOD (y.year, 400) = 0)
THEN 91
ELSE 90
END
WHEN y.season IN (public.pet_spring (), public.pet_summer ())
THEN 92
ELSE 91
END
-- noqa: enable=LT01
),
forecast_inputs AS (SELECT location_id::smallint AS location_id,
season,
array_agg (year ORDER BY year) AS years,
array_agg (wetbulb ORDER BY year) AS wetbulb_values
FROM complete_yearly_wetbulb
GROUP BY location_id,
season)
SELECT forecast.location_id::smallint AS location_id,
forecast.year::smallint AS year,
inputs.season,
forecast.pet::real AS wetbulb,
forecast.lower::real AS lower,
forecast.upper::real AS upper,
forecast.model_type,
forecast.full_years_used,
forecast.warming_rate::real AS warming_rate,
forecast.acceleration::real AS acceleration
FROM forecast_inputs AS inputs
CROSS JOIN LATERAL public.pet_forecast_for_location (
inputs.location_id::integer,
inputs.years,
inputs.wetbulb_values
) AS forecast ;

CREATE UNIQUE INDEX if not exists wetbulb_forecast_location_year_season_uidx
ON public.wetbulb_forecast (location_id, year, season) ;

CREATE INDEX if not exists wetbulb_forecast_year_idx
ON public.wetbulb_forecast (year) ;

CREATE INDEX if not exists wetbulb_forecast_season_idx
ON public.wetbulb_forecast (season) ;

CREATE MATERIALIZED VIEW public.wetbulb_forecast_max AS
WITH wetbulb_with_seasons AS (SELECT w.location_id::smallint AS location_id,
w.date,
w_seasons.season,
w.wetbulb::real AS wetbulb
FROM public.wetbulb AS w
CROSS JOIN LATERAL (
VALUES (public.pet_annual_season ()),
(public.pet_season (w.date))
) AS w_seasons (season)),
daily_wetbulb AS (SELECT w.location_id::smallint AS location_id,
w.date,
w.season,
MAX (w.wetbulb)::real AS wetbulb
FROM wetbulb_with_seasons AS w
GROUP BY w.location_id,
w.date,
w.season),
yearly_wetbulb AS (SELECT d.location_id::smallint AS location_id,
EXTRACT (YEAR FROM d.date)::smallint AS year,
d.season,
MAX (d.wetbulb)::real AS wetbulb,
COUNT (*)::int AS days_present
FROM daily_wetbulb AS d
GROUP BY d.location_id,
EXTRACT (YEAR FROM d.date)::smallint,
d.season),
complete_yearly_wetbulb AS (SELECT y.location_id::smallint AS location_id,
y.year::smallint AS year,
y.season,
y.wetbulb::real AS wetbulb
FROM yearly_wetbulb AS y
-- noqa: disable=LT01
WHERE y.days_present = CASE
WHEN y.season = public.pet_annual_season () THEN CASE
WHEN
MOD (y.year, 4) =
0
AND
(MOD (y.year, 100) != 0 OR MOD (y.year, 400) = 0)
THEN 366
ELSE 365
END
WHEN y.season = public.pet_winter () THEN CASE
WHEN
MOD (y.year, 4) =
0
AND
(MOD (y.year, 100) != 0 OR MOD (y.year, 400) = 0)
THEN 91
ELSE 90
END
WHEN y.season IN (public.pet_spring (), public.pet_summer ())
THEN 92
ELSE 91
END
-- noqa: enable=LT01
),
forecast_inputs AS (SELECT location_id::smallint AS location_id,
season,
array_agg (year ORDER BY year) AS years,
array_agg (wetbulb ORDER BY year) AS wetbulb_values
FROM complete_yearly_wetbulb
GROUP BY location_id,
season)
SELECT forecast.location_id::smallint AS location_id,
forecast.year::smallint AS year,
inputs.season,
forecast.pet::real AS wetbulb,
forecast.lower::real AS lower,
forecast.upper::real AS upper,
forecast.model_type,
forecast.full_years_used,
forecast.warming_rate::real AS warming_rate,
forecast.acceleration::real AS acceleration
FROM forecast_inputs AS inputs
CROSS JOIN LATERAL public.pet_forecast_for_location (
inputs.location_id::integer,
inputs.years,
inputs.wetbulb_values
) AS forecast ;

CREATE UNIQUE INDEX if not exists wetbulb_forecast_max_location_year_season_uidx
ON public.wetbulb_forecast_max (location_id, year, season) ;

CREATE INDEX if not exists wetbulb_forecast_max_year_idx
ON public.wetbulb_forecast_max (year) ;

CREATE INDEX if not exists wetbulb_forecast_max_season_idx
ON public.wetbulb_forecast_max (season) ;

CREATE VIEW public.wetbulb_city_rankings_view AS
WITH combined_yearly_avg AS (
SELECT
a.location_id::smallint AS location_id,
a.year::smallint AS year,
a.season,
a.avg_wetbulb::real AS wetbulb,
0 AS source_order
FROM public.wetbulb_year_stats AS a
UNION ALL
SELECT
f.location_id::smallint AS location_id,
f.year::smallint AS year,
f.season,
f.wetbulb::real AS wetbulb,
1 AS source_order
FROM public.wetbulb_forecast AS f
), deduplicated_yearly_avg AS (
SELECT DISTINCT ON (location_id, year, season)
location_id::smallint AS location_id,
year::smallint AS year,
season,
wetbulb::real AS wetbulb
FROM combined_yearly_avg
ORDER BY
location_id,
year,
season,
source_order
), year_2000_value AS (
SELECT
location_id::smallint AS location_id,
season,
wetbulb::real AS wetbulb
FROM deduplicated_yearly_avg
WHERE year = 2000
), combined_yearly_avg_of_avg AS (
SELECT
a.location_id::smallint AS location_id,
a.year::smallint AS year,
a.season,
a.avg_wetbulb_avg::real AS wetbulb_avg,
0 AS source_order
FROM public.wetbulb_year_stats AS a
UNION ALL
SELECT
f.location_id::smallint AS location_id,
f.year::smallint AS year,
f.season,
f.wetbulb::real AS wetbulb_avg,
1 AS source_order
FROM public.wetbulb_forecast AS f
), deduplicated_yearly_avg_of_avg AS (
SELECT DISTINCT ON (location_id, year, season)
location_id::smallint AS location_id,
year::smallint AS year,
season,
wetbulb_avg::real AS wetbulb_avg
FROM combined_yearly_avg_of_avg
ORDER BY
location_id,
year,
season,
source_order
), year_2000_value_avg AS (
SELECT
location_id::smallint AS location_id,
season,
wetbulb_avg::real AS wetbulb_avg
FROM deduplicated_yearly_avg_of_avg
WHERE year = 2000
)
SELECT
s.location_id::smallint,
s.year::smallint,
s.season,
s.avg_wetbulb::real AS avg_wetbulb,
s.max_wetbulb::real AS max_wetbulb,
s.avg_wetbulb_avg::real AS avg_wetbulb_avg,
l.city,
l.state,
s.p10::real,
s.p90::real,
f.lower::real AS future_lower,
f.upper::real AS future_upper,
ROUND ((s.avg_wetbulb - y2k.wetbulb)::numeric, 2)::real AS change_from_2000,
ROUND ((s.avg_wetbulb_avg - y2k_avg.wetbulb_avg)::numeric,
2)::real AS change_from_2000_avg
FROM public.wetbulb_year_stats AS s
JOIN public.locations AS l ON l.id = s.location_id
LEFT JOIN public.wetbulb_forecast AS f ON f.location_id = s.location_id
AND f.year = 2100::smallint
AND f.season = s.season
LEFT JOIN year_2000_value AS y2k ON y2k.location_id = s.location_id
AND y2k.season = s.season
LEFT JOIN year_2000_value_avg AS y2k_avg ON y2k_avg.location_id = s.location_id
AND y2k_avg.season = s.season
WHERE
s.location_id > 0 ;

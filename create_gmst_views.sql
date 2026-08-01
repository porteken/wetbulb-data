CREATE OR REPLACE FUNCTION public.wetbulb_gmst_forecast (input_statistic text)
RETURNS TABLE (
location_id smallint,
year smallint,
season text,
wetbulb real,
lower real,
upper real,
wetbulb_avg real,
lower_avg real,
upper_avg real,
model_type text,
full_years_used smallint,
warming_rate real,
acceleration real,
scenario text
)
LANGUAGE SQL
STABLE
AS $$
WITH daily AS (
SELECT
w.location_id,
w.date,
s.season,
MAX(w.wetbulb)::double precision AS wetbulb,
MAX(w.wetbulb_avg)::double precision AS wetbulb_avg
FROM public.wetbulb AS w
CROSS JOIN LATERAL (
VALUES ('Annual'::text), (public.wetbulb_season(w.date))
) AS s (season)
GROUP BY w.location_id, w.date, s.season
), yearly AS (
SELECT
d.location_id,
EXTRACT(YEAR FROM d.date)::integer AS year,
d.season,
CASE WHEN input_statistic = 'max' THEN MAX(d.wetbulb) ELSE AVG(d.wetbulb) END AS primary_value,
CASE WHEN input_statistic = 'max' THEN MAX(d.wetbulb_avg) ELSE AVG(d.wetbulb_avg) END AS average_value,
COUNT(d.wetbulb)::integer AS days_present,
COUNT(d.wetbulb_avg)::integer AS days_present_avg
FROM daily AS d
GROUP BY d.location_id, EXTRACT(YEAR FROM d.date)::integer, d.season
), qualified AS (
SELECT
y.*,
0.95 * CASE
WHEN y.season = 'Annual' THEN CASE
WHEN MOD(y.year, 4) = 0 AND (MOD(y.year, 100) <> 0 OR MOD(y.year, 400) = 0) THEN 366
ELSE 365 END
WHEN y.season = 'Winter' THEN CASE
WHEN MOD(y.year, 4) = 0 AND (MOD(y.year, 100) <> 0 OR MOD(y.year, 400) = 0) THEN 91
ELSE 90 END
WHEN y.season IN ('Spring', 'Summer') THEN 92
ELSE 91 END AS required_days
FROM yearly AS y
), long_values AS (
SELECT
q.location_id,
q.year,
q.season,
v.metric,
v.value
FROM qualified AS q
CROSS JOIN LATERAL (
VALUES
(CASE WHEN input_statistic = 'max' THEN 'max_wetbulb' ELSE 'avg_wetbulb' END,
 CASE WHEN q.days_present >= q.required_days THEN q.primary_value END),
(CASE WHEN input_statistic = 'max' THEN 'max_wetbulb_avg' ELSE 'avg_wetbulb_avg' END,
 CASE WHEN q.days_present_avg >= q.required_days THEN q.average_value END)
) AS v (metric, value)
WHERE q.year >= 2000 AND v.value IS NOT NULL
), fit_base AS (
SELECT
v.location_id,
v.season,
v.metric,
COUNT(*)::integer AS n,
MAX(v.year)::integer AS last_year,
AVG(v.value) AS wbar,
AVG(g.anomaly) AS gbar,
regr_slope(v.value, g.anomaly) AS slope,
regr_sxx(v.value, g.anomaly) AS unused_sxx,
regr_sxx(g.anomaly, g.anomaly) AS g_sxx,
regr_syy(v.value, g.anomaly) AS w_syy
FROM long_values AS v
JOIN public.gmst_observations AS g ON g.year = v.year
GROUP BY v.location_id, v.season, v.metric
HAVING COUNT(*) >= 10
), fits AS (
SELECT
f.*,
GREATEST(0.0, (f.w_syy - (POWER(f.slope, 2) * f.g_sxx)) / (f.n - 2)) AS residual_variance,
GREATEST(
1e-8,
GREATEST(0.0, (f.w_syy - (POWER(f.slope, 2) * f.g_sxx)) / (f.n - 2)) / NULLIF(f.g_sxx, 0.0)
) AS slope_variance
FROM fit_base AS f
WHERE f.slope IS NOT NULL AND f.g_sxx > 0.0
), representatives AS (
SELECT ranked.*
FROM (
SELECT
f.*,
sg.station_group,
ROW_NUMBER() OVER (
PARTITION BY sg.station_group, f.season, f.metric
ORDER BY f.n DESC, f.slope_variance ASC, f.location_id ASC
) AS representative_rank
FROM fits AS f
JOIN public.forecast_station_groups AS sg ON sg.location_id = f.location_id
) AS ranked
WHERE ranked.representative_rank = 1
), fixed AS (
SELECT
r.season,
r.metric,
COUNT(*)::integer AS groups,
SUM(1.0 / r.slope_variance) AS sum_w,
SUM(1.0 / POWER(r.slope_variance, 2)) AS sum_w2,
SUM(r.slope / r.slope_variance) / SUM(1.0 / r.slope_variance) AS fixed_mean
FROM representatives AS r
GROUP BY r.season, r.metric
), heterogeneity AS (
SELECT
f.*,
SUM((1.0 / r.slope_variance) * POWER(r.slope - f.fixed_mean, 2)) AS q
FROM fixed AS f
JOIN representatives AS r ON r.season = f.season AND r.metric = f.metric
GROUP BY f.season, f.metric, f.groups, f.sum_w, f.sum_w2, f.fixed_mean
), tau AS (
SELECT
h.*,
CASE
WHEN h.groups <= 1 OR (h.sum_w - (h.sum_w2 / h.sum_w)) <= 0.0 THEN 0.0
ELSE GREATEST(0.0, (h.q - (h.groups - 1)) / (h.sum_w - (h.sum_w2 / h.sum_w)))
END AS tau_squared
FROM heterogeneity AS h
), pooled AS (
SELECT
t.season,
t.metric,
t.tau_squared,
SUM(r.slope / (r.slope_variance + t.tau_squared))
/ SUM(1.0 / (r.slope_variance + t.tau_squared)) AS pooled_mean,
GREATEST(1e-8, 1.0 / SUM(1.0 / (r.slope_variance + t.tau_squared))) AS pooled_variance
FROM tau AS t
JOIN representatives AS r ON r.season = t.season AND r.metric = t.metric
GROUP BY t.season, t.metric, t.tau_squared
), posterior AS (
SELECT
f.*,
CASE WHEN p.tau_squared <= 0.0 THEN p.pooled_mean
ELSE (
(f.slope / f.slope_variance) + (p.pooled_mean / p.tau_squared)
) / ((1.0 / f.slope_variance) + (1.0 / p.tau_squared)) END AS slope_star,
CASE WHEN p.tau_squared <= 0.0 THEN p.pooled_variance
ELSE GREATEST(1e-8, 1.0 / ((1.0 / f.slope_variance) + (1.0 / p.tau_squared))) END AS slope_star_variance
FROM fits AS f
JOIN pooled AS p ON p.season = f.season AND p.metric = f.metric
), forecast_long AS (
SELECT
p.location_id,
s.year,
p.season,
p.metric,
s.scenario,
p.n,
(p.wbar + (p.slope_star * (s.anomaly - p.gbar))) AS point,
cal.calibration_factor * 1.2816 * SQRT(GREATEST(0.0,
(p.residual_variance * (1.0 + (1.0 / p.n)))
+ (p.slope_star_variance * POWER(s.anomaly - p.gbar, 2))
+ (POWER(p.slope_star, 2) * POWER(s.sigma_g, 2))
)) AS margin,
(s.anomaly - previous.anomaly) AS warming_rate
FROM posterior AS p
JOIN public.gmst_scenarios AS s ON s.year > p.last_year
LEFT JOIN public.gmst_scenarios AS previous
ON previous.scenario = s.scenario AND previous.year = s.year - 1
JOIN public.forecast_interval_calibration AS cal ON cal.metric = p.metric
)
SELECT
f.location_id::smallint,
f.year::smallint,
f.season,
ROUND(MAX(f.point) FILTER (WHERE f.metric NOT LIKE '%_avg')::numeric, 1)::real,
ROUND(MAX(f.point - f.margin) FILTER (WHERE f.metric NOT LIKE '%_avg')::numeric, 1)::real,
ROUND(MAX(f.point + f.margin) FILTER (WHERE f.metric NOT LIKE '%_avg')::numeric, 1)::real,
ROUND(MAX(f.point) FILTER (WHERE f.metric LIKE '%_avg')::numeric, 1)::real,
ROUND(MAX(f.point - f.margin) FILTER (WHERE f.metric LIKE '%_avg')::numeric, 1)::real,
ROUND(MAX(f.point + f.margin) FILTER (WHERE f.metric LIKE '%_avg')::numeric, 1)::real,
'gmst_linear'::text,
MAX(f.n) FILTER (WHERE f.metric NOT LIKE '%_avg')::smallint,
MAX(f.warming_rate)::real,
0.0::real,
f.scenario
FROM forecast_long AS f
GROUP BY f.location_id, f.year, f.season, f.scenario
$$ ;

DROP VIEW IF EXISTS public.wetbulb_city_rankings_view CASCADE ;
DROP MATERIALIZED VIEW IF EXISTS public.wetbulb_forecast_max CASCADE ;
DROP MATERIALIZED VIEW IF EXISTS public.wetbulb_forecast CASCADE ;

CREATE MATERIALIZED VIEW public.wetbulb_forecast_scenarios AS
SELECT * FROM public.wetbulb_gmst_forecast ('mean') ;

CREATE UNIQUE INDEX wetbulb_forecast_scenarios_uidx
ON public.wetbulb_forecast_scenarios (location_id, year, season, scenario) ;

CREATE MATERIALIZED VIEW public.wetbulb_forecast_max_scenarios AS
SELECT * FROM public.wetbulb_gmst_forecast ('max') ;

CREATE UNIQUE INDEX wetbulb_forecast_max_scenarios_uidx
ON public.wetbulb_forecast_max_scenarios (location_id, year, season, scenario) ;

CREATE MATERIALIZED VIEW public.wetbulb_forecast AS
SELECT
location_id, year, season, wetbulb, lower, upper, wetbulb_avg, lower_avg,
upper_avg, model_type, full_years_used, warming_rate, acceleration
FROM public.wetbulb_forecast_scenarios
WHERE scenario = 'ssp245' ;

CREATE UNIQUE INDEX wetbulb_forecast_location_year_season_uidx
ON public.wetbulb_forecast (location_id, year, season) ;

CREATE MATERIALIZED VIEW public.wetbulb_forecast_max AS
SELECT
location_id, year, season, wetbulb, lower, upper, wetbulb_avg, lower_avg,
upper_avg, model_type, full_years_used, warming_rate, acceleration
FROM public.wetbulb_forecast_max_scenarios
WHERE scenario = 'ssp245' ;

CREATE UNIQUE INDEX wetbulb_forecast_max_location_year_season_uidx
ON public.wetbulb_forecast_max (location_id, year, season) ;

CREATE VIEW public.wetbulb_city_rankings_view AS
WITH year_2000 AS (
SELECT location_id, season, avg_wetbulb, avg_wetbulb_avg
FROM public.wetbulb_year_stats WHERE year = 2000
)
SELECT
s.location_id,
s.year,
s.season,
s.avg_wetbulb,
s.max_wetbulb,
s.avg_wetbulb_avg,
s.max_wetbulb_avg,
l.city,
l.state,
s.p10,
s.p90,
s.p10_avg,
s.p90_avg,
s.p5,
s.p95,
s.p5_avg,
s.p95_avg,
f.lower AS future_lower,
f.upper AS future_upper,
f.lower_avg AS future_lower_avg,
f.upper_avg AS future_upper_avg,
ROUND ((s.avg_wetbulb - y.avg_wetbulb)::numeric, 2)::real AS change_from_2000,
ROUND ((s.avg_wetbulb_avg - y.avg_wetbulb_avg)::numeric,
2)::real AS change_from_2000_avg
FROM public.wetbulb_year_stats AS s
JOIN public.locations AS l ON l.id = s.location_id
LEFT JOIN public.wetbulb_forecast AS f
ON f.location_id = s.location_id AND f.year = 2100 AND f.season = s.season
LEFT JOIN year_2000 AS y
ON y.location_id = s.location_id AND y.season = s.season
WHERE s.location_id BETWEEN 0 AND 999 ;

CREATE MATERIALIZED VIEW public.wetbulb_year_stats AS
WITH seasonal AS (
SELECT location_id, EXTRACT (YEAR FROM date)::integer AS year,
'Annual'::text AS season, wetbulb
FROM public.wetbulb
UNION ALL
SELECT location_id, EXTRACT (YEAR FROM date)::integer AS year,
CASE
WHEN EXTRACT (MONTH FROM date)::integer IN (12, 1, 2) THEN 'Winter'
WHEN EXTRACT (MONTH FROM date)::integer IN (3, 4, 5) THEN 'Spring'
WHEN EXTRACT (MONTH FROM date)::integer IN (6, 7, 8) THEN 'Summer'
ELSE 'Fall'
END,
wetbulb
FROM public.wetbulb
)
SELECT location_id, year, season,
ROUND (AVG (wetbulb)::numeric, 1)::real AS wetbulb_avg,
ROUND (MAX (wetbulb)::numeric, 1)::real AS wetbulb_max,
ROUND ((PERCENTILE_CONT (0.1) WITHIN GROUP (ORDER BY wetbulb))::numeric,
1)::real AS p10,
ROUND ((PERCENTILE_CONT (0.9) WITHIN GROUP (ORDER BY wetbulb))::numeric,
1)::real AS p90
FROM seasonal
GROUP BY location_id, year, season ;

CREATE UNIQUE INDEX wetbulb_year_stats_location_year_season_uidx
ON public.wetbulb_year_stats (location_id, year, season) ;

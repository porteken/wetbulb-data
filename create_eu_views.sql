CREATE VIEW public.wetbulb_eu_city_rankings_view AS
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
    ROUND(
        (s.avg_wetbulb - y.avg_wetbulb)::numeric,
        2
    )::real AS change_from_2000,
    ROUND(
        (s.avg_wetbulb_avg - y.avg_wetbulb_avg)::numeric,
        2
    )::real AS change_from_2000_avg
FROM public.wetbulb_year_stats AS s
JOIN public.locations AS l ON l.id = s.location_id
LEFT JOIN public.wetbulb_forecast AS f
    ON f.location_id = s.location_id AND f.year = 2100 AND f.season = s.season
LEFT JOIN year_2000 AS y
    ON y.location_id = s.location_id AND y.season = s.season
WHERE s.location_id >= 1000;

-- squawk-ignore-file require-concurrent-index-deletion

-- SET LOCAL keeps these timeouts scoped to the executing transaction so they
-- never leak into later statements on the same connection.
SET LOCAL statement_timeout = '5s' ;
SET LOCAL lock_timeout = '1s' ;
DROP INDEX IF EXISTS pet_year_stats_location_year_season_uidx ;
DROP INDEX IF EXISTS pet_year_stats_year_idx ;
DROP INDEX IF EXISTS pet_year_stats_season_idx ;
DROP INDEX IF EXISTS pet_forecast_location_year_season_uidx ;
DROP INDEX IF EXISTS pet_forecast_location_year_uidx ;
DROP INDEX IF EXISTS pet_forecast_year_idx ;
DROP INDEX IF EXISTS pet_forecast_season_idx ;
DROP INDEX IF EXISTS pet_forecast_max_location_year_season_uidx ;
DROP INDEX IF EXISTS pet_forecast_max_year_idx ;
DROP INDEX IF EXISTS pet_forecast_max_season_idx ;
DROP INDEX IF EXISTS pet_change_location_year_uidx ;
DROP INDEX IF EXISTS pet_change_year_idx ;
DROP INDEX IF EXISTS wetbulb_year_stats_location_year_season_uidx ;
DROP INDEX IF EXISTS wetbulb_year_stats_year_idx ;
DROP INDEX IF EXISTS wetbulb_year_stats_season_idx ;
DROP INDEX IF EXISTS wetbulb_forecast_location_year_season_uidx ;
DROP INDEX IF EXISTS wetbulb_forecast_year_idx ;
DROP INDEX IF EXISTS wetbulb_forecast_season_idx ;
DROP INDEX IF EXISTS wetbulb_forecast_max_location_year_season_uidx ;
DROP INDEX IF EXISTS wetbulb_forecast_max_year_idx ;
DROP INDEX IF EXISTS wetbulb_forecast_max_season_idx ;

DO $$
DECLARE
	relation_name text;
	relation_kind "char";
BEGIN
	FOREACH relation_name IN ARRAY ARRAY['pet_year_stats', 'city_rankings_view', 'pet_forecast', 'pet_forecast_max', 'pet_change', 'wetbulb_year_stats', 'wetbulb_city_rankings_view', 'wetbulb_forecast', 'wetbulb_forecast_max'] LOOP
		SELECT c.relkind
		INTO relation_kind
		FROM pg_catalog.pg_class AS c
		JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
		WHERE n.nspname = 'public'
		AND c.relname = relation_name ;

		IF relation_kind = 'm' THEN
			EXECUTE format('DROP MATERIALIZED VIEW public.%I CASCADE', relation_name);
		ELSIF relation_kind = 'v' THEN
			EXECUTE format('DROP VIEW public.%I CASCADE', relation_name);
		ELSIF relation_name NOT IN ('pet_forecast', 'pet_forecast_max') AND relation_kind IN ('r', 'p') THEN
			EXECUTE format('DROP TABLE public.%I CASCADE', relation_name);
		END IF;
	END LOOP;
END $$ ;

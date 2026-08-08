# Copyright (C) 2026 Kenneth Porter

"""Generate `cities_eu_isd_stations.csv`: each EU city's ISD station candidate."""

from __future__ import annotations

import argparse
import importlib
import logging
from typing import Any, cast

import requests

import nldas
from isd_history import (
    EARTH_RADIUS_KM,
    MAX_ELEV_DELTA_M,
    MAX_STATION_DISTANCE_KM,
    candidate_ids_for_station,
    haversine_km,
    prepare_history,
    rank_station_rows,
)
from make_isd_station_map import (
    _candidate_verified_at,
    fetch_isd_history,
    write_station_map,
)

pd = cast("Any", importlib.import_module("pandas"))
np = cast("Any", importlib.import_module("numpy"))

type DataFrame = Any

__all__ = [
    "EARTH_RADIUS_KM",
    "MAX_ELEV_DELTA_M",
    "MAX_STATION_DISTANCE_KM",
    "candidate_ids_for_station",
    "haversine_km",
    "rank_station_rows",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

CITIES_EU_CSV = "cities_eu.csv"
EU_START_YEAR = 1991

EUROPE_HISTORY_MIN_LAT = 30.0
EUROPE_HISTORY_MAX_LAT = 75.0
EUROPE_HISTORY_MIN_LON = -30.0
EUROPE_HISTORY_MAX_LON = 65.0


def _prepare_history(history: DataFrame) -> DataFrame:
    """Numeric-parse and pre-filter the global inventory to a loose Europe bbox."""
    return prepare_history(
        history,
        min_lat=EUROPE_HISTORY_MIN_LAT,
        max_lat=EUROPE_HISTORY_MAX_LAT,
        min_lon=EUROPE_HISTORY_MIN_LON,
        max_lon=EUROPE_HISTORY_MAX_LON,
    )


def _select_city_station(
    ranked: DataFrame,
    *,
    start_year: int,
    end_year: int,
    session: requests.Session,
    verified_cache: dict[tuple[str, int], bool],
) -> tuple[Any, list[str], bool] | None:
    """Return (chosen station row, candidate ids, begin_verified) for the first verified match."""
    for candidate_row in ranked.itertuples():
        candidate_ids = candidate_ids_for_station(
            candidate_row.USAF, candidate_row.WBAN
        )
        if not candidate_ids:
            continue
        if not _candidate_verified_at(
            candidate_ids, end_year, session=session, cache=verified_cache
        ):
            continue
        begin_verified = _candidate_verified_at(
            candidate_ids, start_year, session=session, cache=verified_cache
        )
        return candidate_row, candidate_ids, begin_verified
    return None


def build_station_map_eu(
    cities_csv: str = CITIES_EU_CSV,
    *,
    session: requests.Session | None = None,
    start_year: int = EU_START_YEAR,
    end_year: int = nldas.NLDAS_END_YEAR,
) -> DataFrame:
    """Return a DataFrame of location_id, usaf, wban, isd_ids, ... per EU city."""
    return build_station_map_region(
        cities_csv,
        min_lat=EUROPE_HISTORY_MIN_LAT,
        max_lat=EUROPE_HISTORY_MAX_LAT,
        min_lon=EUROPE_HISTORY_MIN_LON,
        max_lon=EUROPE_HISTORY_MAX_LON,
        session=session,
        start_year=start_year,
        end_year=end_year,
    )


def build_station_map_region(
    cities_csv: str,
    *,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    session: requests.Session | None = None,
    start_year: int,
    end_year: int,
) -> DataFrame:
    """Return a global-ISD station map for cities inside a regional bounding box."""
    http = session or requests.Session()
    history = prepare_history(
        fetch_isd_history(session=http),
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
    )
    cities = pd.read_csv(
        cities_csv,
        usecols=["location_id", "lat", "lng", "dem_m", "utc_offset_hours"],
    )

    verified_cache: dict[tuple[str, int], bool] = {}
    rows: list[dict[str, Any]] = []
    unmatched: list[str] = []

    for city in cities.itertuples():
        ranked = rank_station_rows(
            history,
            city.lat,
            city.lng,
            city.dem_m,
            start_year=start_year,
            end_year=end_year,
        )
        selected = (
            _select_city_station(
                ranked,
                start_year=start_year,
                end_year=end_year,
                session=http,
                verified_cache=verified_cache,
            )
            if not ranked.empty
            else None
        )

        if selected is None:
            unmatched.append(str(city.location_id))
            rows.append(
                {
                    "location_id": city.location_id,
                    "usaf": None,
                    "wban": None,
                    "isd_ids": None,
                    "lon": None,
                    "dist_km": None,
                    "elev_m": None,
                    "utc_offset_hours": city.utc_offset_hours,
                    "begin_verified": False,
                },
            )
            continue

        station_row, candidate_ids, begin_verified = selected
        rows.append(
            {
                "location_id": city.location_id,
                "usaf": station_row.USAF,
                "wban": station_row.WBAN,
                "isd_ids": "|".join(candidate_ids),
                "lon": float(station_row.LON_NUM),
                "dist_km": round(float(station_row.dist_km), 2),
                "elev_m": float(station_row.ELEV_NUM),
                "utc_offset_hours": city.utc_offset_hours,
                "begin_verified": begin_verified,
            },
        )

    if unmatched:
        LOGGER.warning(
            "%d city/cities have no verified ISD station: %s",
            len(unmatched),
            ", ".join(unmatched),
        )

    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities-csv", default=CITIES_EU_CSV)
    parser.add_argument("--out", default="cities_eu_isd_stations.csv")
    parser.add_argument("--start-year", type=int, default=EU_START_YEAR)
    parser.add_argument("--end-year", type=int, default=nldas.NLDAS_END_YEAR)
    return parser.parse_args()


def main() -> None:
    """Build and write the EU city-to-ISD-station map."""
    args = _parse_args()
    write_station_map(
        build_station_map_eu(
            args.cities_csv,
            start_year=args.start_year,
            end_year=args.end_year,
        ),
        args.out,
        logger=LOGGER,
    )


if __name__ == "__main__":
    main()

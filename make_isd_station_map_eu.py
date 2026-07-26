"""Generate `cities_eu_isd_stations.csv`: each EU city's ISD station candidate.

Unlike `make_isd_station_map.py` (which reuses the already-verified US LCD
station assignment because IEM ASOS -- the source `make_lcd_station_map.py`
matches against -- is US-only), there is no equivalent verified assignment
to start from for Europe. Instead this module selects directly from NCEI's
global `isd-history.csv` inventory: for each city, rank every station within
`MAX_STATION_DISTANCE_KM` and `MAX_ELEV_DELTA_M` by (coverage bucket,
distance), then probe candidates with a ranged GET (reusing
`make_isd_station_map._isd_hourly_data_present`) until one verifies at both
`start_year` and `end_year`.

Defaulting `--start-year` to 1991 (rather than the 2000 the initial EU
backfill starts at) means a station is only chosen if it likely also covers
the pipeline's eventual 1991 extension, so that extension needs no
re-crosswalk.

Only one physical station is kept per city (unlike the US crosswalk's
WBAN-history fallthrough, which spans USAF-id transitions for a *single*
known-correct physical station): chaining two different physical stations
here could silently produce an inconsistent record, whereas a year that
station can't cover becomes an ERA5-Land gap-fill instead (`era5land.py`),
keeping provenance clean. The three-candidate-id fallthrough
(`candidate_ids_for_station`) mirrors `make_isd_station_map.py`'s
USAF+WBAN / USAF+placeholder-WBAN / placeholder-USAF+WBAN order so `isd.py`
can try past NCEI's own placeholder-filing quirks for this one station.

Network calls: one `isd-history.csv` download plus up to a few small ranged
GETs per candidate id per probed year (start_year and end_year), per ranked
candidate station, per city.
"""

from __future__ import annotations

import argparse
import importlib
import logging
from typing import Any, cast

import requests

import nldas
from make_isd_station_map import (
    PLACEHOLDER_USAF,
    PLACEHOLDER_WBAN,
    _candidate_verified_at,
    fetch_isd_history,
    write_station_map,
)

pd = cast("Any", importlib.import_module("pandas"))
np = cast("Any", importlib.import_module("numpy"))

type DataFrame = Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

CITIES_EU_CSV = "cities_eu.csv"
EU_START_YEAR = 1991
EARTH_RADIUS_KM = 6371.0088
MAX_STATION_DISTANCE_KM = 60.0
MAX_ELEV_DELTA_M = 300.0

EUROPE_HISTORY_MIN_LAT = 30.0
EUROPE_HISTORY_MAX_LAT = 75.0
EUROPE_HISTORY_MIN_LON = -30.0
EUROPE_HISTORY_MAX_LON = 65.0


def haversine_km(
    lat1: DataFrame | float,
    lon1: DataFrame | float,
    lat2: DataFrame,
    lon2: DataFrame,
) -> DataFrame:
    """Return the great-circle distance in km between one point and a series of points."""
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat_rad = np.radians(lat2 - lat1)
    dlon_rad = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat_rad / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon_rad / 2.0) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def rank_station_rows(
    history: DataFrame,
    city_lat: float,
    city_lng: float,
    city_dem_m: float,
    *,
    start_year: int,
    end_year: int,
) -> DataFrame:
    """Return candidate station rows within range, ranked by (coverage, distance)."""
    dist_km = haversine_km(city_lat, city_lng, history["LAT_NUM"], history["LON_NUM"])
    elev_delta = (history["ELEV_NUM"] - city_dem_m).abs()
    within_range = (dist_km <= MAX_STATION_DISTANCE_KM) & (
        elev_delta <= MAX_ELEV_DELTA_M
    )
    candidates = history[within_range].copy()
    if candidates.empty:
        return candidates
    candidates["dist_km"] = dist_km[within_range]

    begin_year = pd.to_numeric(candidates["BEGIN"].str[:4], errors="coerce")
    end_year_col = pd.to_numeric(candidates["END"].str[:4], errors="coerce")
    full_coverage = (begin_year <= start_year) & (end_year_col >= end_year - 1)
    recent_only = end_year_col >= end_year - 1
    candidates["coverage_bucket"] = np.where(
        full_coverage, 0, np.where(recent_only, 1, 2)
    )

    return candidates.sort_values(["coverage_bucket", "dist_km"])


def candidate_ids_for_station(usaf: str, wban: str) -> list[str]:
    """Return the ordered id fallthrough for one physical station.

    Order: the station's real USAF+WBAN id, then USAF paired with the
    `99999` WBAN placeholder (some station-years are filed that way
    instead), then the `999999` USAF placeholder paired with the real WBAN,
    last. Placeholder-only combinations aren't real stations and are
    skipped.
    """
    ids: list[str] = []
    if usaf != PLACEHOLDER_USAF and wban != PLACEHOLDER_WBAN:
        ids.append(usaf + wban)
    if usaf != PLACEHOLDER_USAF:
        ids.append(usaf + PLACEHOLDER_WBAN)
    if wban != PLACEHOLDER_WBAN:
        ids.append(PLACEHOLDER_USAF + wban)

    seen: set[str] = set()
    ordered: list[str] = []
    for station_id in ids:
        if station_id not in seen:
            seen.add(station_id)
            ordered.append(station_id)
    return ordered


def _prepare_history(history: DataFrame) -> DataFrame:
    """Numeric-parse and pre-filter the global inventory to a loose Europe bbox."""
    history = history.copy()
    history["LAT_NUM"] = pd.to_numeric(history["LAT"], errors="coerce")
    history["LON_NUM"] = pd.to_numeric(history["LON"], errors="coerce")
    history["ELEV_NUM"] = pd.to_numeric(history["ELEV(M)"], errors="coerce")
    history = history.dropna(subset=["LAT_NUM", "LON_NUM", "ELEV_NUM"])
    return history[
        (history["LAT_NUM"] >= EUROPE_HISTORY_MIN_LAT)
        & (history["LAT_NUM"] <= EUROPE_HISTORY_MAX_LAT)
        & (history["LON_NUM"] >= EUROPE_HISTORY_MIN_LON)
        & (history["LON_NUM"] <= EUROPE_HISTORY_MAX_LON)
    ]


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
    http = session or requests.Session()
    history = _prepare_history(fetch_isd_history(session=http))
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

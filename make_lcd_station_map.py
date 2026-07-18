"""Generate `cities_lcd_stations.csv`: each city's nearest usable LCD station.

Builds the city -> LCD station crosswalk that `lcd.py` reads to fetch NOAA
LCD v2 wet-bulb station data, used as the default wetbulb pipeline source:

1. Fetch IEM's per-state ASOS network metadata (includes each station's
   `ncei91` field, the exact station id LCD v2 bulk filenames use), keeping
   only stations that are still active and have a known archive start date.
2. For each city in `cities.csv`, rank the `LCD_MAX_CANDIDATES_PER_CITY`
   nearest stations by great-circle distance and pick one whose LCD bulk
   files actually contain hourly-cadence observations
   (`HOURLY_REPORT_TYPES` in `lcd.py`), in two tiers:

   - **Tier 1** (preferred): the station's archive covers the full
     `start_year`-`end_year` pipeline window; verified by probing both
     endpoints.
   - **Tier 2** (fallback, only tried once every candidate fails tier 1):
     any candidate regardless of archive start, verified by probing its
     first full archive year and `end_year`. These stations have fewer
     years of history -- the missing early years simply produce no daily
     wet-bulb rows for that city, same as any other station gap.

   Existence alone is not enough to verify a candidate: some IEM-listed
   ASOS stations (e.g. Buckley SFB near Denver, `USW00023062`) have an LCD
   file that exists for every year but contains only daily/monthly summary
   rows (`SOD`/`SOM`), no hourly METAR at all -- verified empirically when
   a first pilot run silently produced zero daily wet-bulb rows for
   Denver. Checked with a ranged GET over the first ~200 KB (not a full
   download, and not HEAD/LIST -- see module docstring in `lcd.py` for
   why), which is enough bytes to observe a month+ of hourly rows if the
   station reports them at all near the start of the probed year.

Network calls: ~49 IEM geojson requests (one per CONUS state, throttled to
1/sec) plus up to a couple dozen ~200 KB ranged GETs per city against NCEI.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import logging
import math
import sys
import time
from typing import Any, cast

import requests

import nldas
from lcd import HOURLY_REPORT_TYPES

pd = cast("Any", importlib.import_module("pandas"))

type DataFrame = Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

CONUS_STATES: tuple[str, ...] = (
    "AL",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
)

IEM_NETWORK_GEOJSON_URL = (
    "https://mesonet.agron.iastate.edu/geojson/network/{state}_ASOS.geojson"
)
IEM_REQUEST_DELAY_SECONDS = 1.0
IEM_REQUEST_TIMEOUT_SECONDS = 60

LCD_URL_TEMPLATE = (
    "https://www.ncei.noaa.gov/oa/local-climatological-data/"
    "v2/access/{year}/LCD_{station_id}_{year}.csv"
)
LCD_EXISTENCE_TIMEOUT_SECONDS = 20
LCD_MAX_CANDIDATES_PER_CITY = 15
LCD_MAX_REASONABLE_DIST_KM = 100
TIER_FULL_WINDOW = 1
TIER_SHORT_ARCHIVE = 2
LCD_PROBE_RANGE_BYTES = 200_000
_HOURLY_REPORT_TYPE_MARKERS = tuple(f'"{rt}"' for rt in HOURLY_REPORT_TYPES)

EARTH_DIAMETER_KM = 12742.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p = math.pi / 180.0
    a = (
        0.5
        - math.cos((lat2 - lat1) * p) / 2
        + math.cos(lat1 * p)
        * math.cos(lat2 * p)
        * (1 - math.cos((lon2 - lon1) * p))
        / 2
    )
    return EARTH_DIAMETER_KM * math.asin(math.sqrt(a))


def fetch_asos_stations(
    states: tuple[str, ...] = CONUS_STATES,
    *,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Return active CONUS ASOS stations with a known LCD (`ncei91`) id and archive start date.

    No `archive_begin` cutoff is applied here -- candidates with a short
    archive are still returned so a city can fall back to one in tier 2 if
    every full-window candidate fails verification.
    """
    http = session or requests.Session()
    stations: list[dict[str, Any]] = []
    for state in states:
        url = IEM_NETWORK_GEOJSON_URL.format(state=state)
        response = http.get(url, timeout=IEM_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        for feature in response.json().get("features", []):
            props = feature["properties"]
            lcd_id = props.get("ncei91")
            archive_begin = props.get("archive_begin")
            if not lcd_id or not archive_begin:
                continue
            if props.get("archive_end"):
                continue
            lon, lat = feature["geometry"]["coordinates"][:2]
            stations.append(
                {
                    "icao": feature.get("id") or props.get("sid"),
                    "lat": lat,
                    "lng": lon,
                    "name": props.get("sname", ""),
                    "lcd_id": lcd_id,
                    "elev_m": props.get("elevation"),
                    "archive_begin": archive_begin,
                },
            )
        time.sleep(IEM_REQUEST_DELAY_SECONDS)
    LOGGER.info(
        "Fetched %d active CONUS ASOS station(s) with a known archive start date.",
        len(stations),
    )
    return stations


def _lcd_hourly_data_present(
    station_id: str,
    year: int,
    *,
    session: requests.Session,
) -> bool:
    """Return whether a station's LCD file exists and actually contains hourly rows.

    A 200/206 response alone isn't sufficient: some stations (e.g. Buckley
    SFB near Denver) publish an LCD file every year that contains only
    daily/monthly summaries and zero hourly METAR, which would otherwise
    silently produce zero daily wet-bulb rows downstream.
    """
    url = LCD_URL_TEMPLATE.format(year=year, station_id=station_id)
    try:
        response = session.get(
            url,
            headers={"Range": f"bytes=0-{LCD_PROBE_RANGE_BYTES}"},
            timeout=LCD_EXISTENCE_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        LOGGER.warning("Existence check failed for %s (year %d).", station_id, year)
        return False
    if response.status_code not in (200, 206):
        return False
    return any(marker in response.text for marker in _HOURLY_REPORT_TYPE_MARKERS)


def _first_full_archive_year(archive_begin: str, start_year: int) -> int:
    """Return the first full year to probe for a tier-2 (short-archive) candidate.

    Probing the partial first calendar year of a station's archive risks a
    false negative if hourly reporting ramped up partway through it, so
    tier 2 probes the year after `archive_begin` instead (still clamped to
    `start_year` if the archive already covers it).
    """
    first_archive_year = int(archive_begin[:4])
    return max(start_year, first_archive_year + 1)


def _station_verified_at(
    station_id: str,
    year: int,
    *,
    session: requests.Session,
    cache: dict[tuple[str, int], bool],
) -> bool:
    key = (station_id, year)
    if key not in cache:
        cache[key] = _lcd_hourly_data_present(station_id, year, session=session)
    return cache[key]


def _pick_station(
    ranked: list[dict[str, Any]],
    *,
    start_year: int,
    end_year: int,
    session: requests.Session,
    cache: dict[tuple[str, int], bool],
) -> tuple[dict[str, Any], int] | None:
    """Return the best (station, tier) pick among ranked candidates, or None."""
    tier1_cutoff = f"{start_year}-01-01"
    for station in ranked:
        if station["archive_begin"] > tier1_cutoff:
            continue
        if _station_verified_at(
            station["lcd_id"], start_year, session=session, cache=cache
        ) and _station_verified_at(
            station["lcd_id"], end_year, session=session, cache=cache
        ):
            return station, TIER_FULL_WINDOW

    for station in ranked:
        probe_year = _first_full_archive_year(station["archive_begin"], start_year)
        if probe_year > end_year:
            continue
        if _station_verified_at(
            station["lcd_id"], probe_year, session=session, cache=cache
        ) and _station_verified_at(
            station["lcd_id"], end_year, session=session, cache=cache
        ):
            return station, TIER_SHORT_ARCHIVE

    return None


def build_station_map(
    cities_csv: str = "cities.csv",
    *,
    session: requests.Session | None = None,
    start_year: int = nldas.NLDAS_START_YEAR,
    end_year: int = nldas.NLDAS_END_YEAR,
) -> DataFrame:
    """Return a DataFrame of location_id, icao, lcd_id, dist_km, elev_m, archive_begin, tier per city."""
    http = session or requests.Session()
    stations = fetch_asos_stations(session=http)
    cities_df = pd.read_csv(
        cities_csv, usecols=["location_id", "city", "state", "lat", "lng"]
    )
    cities_df = cities_df.sort_values("location_id").reset_index(drop=True)

    verified_cache: dict[tuple[str, int], bool] = {}

    rows: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for city in cities_df.itertuples():
        ranked = sorted(
            stations,
            key=lambda s: _haversine_km(city.lat, city.lng, s["lat"], s["lng"]),
        )[:LCD_MAX_CANDIDATES_PER_CITY]

        result = _pick_station(
            ranked,
            start_year=start_year,
            end_year=end_year,
            session=http,
            cache=verified_cache,
        )

        if result is None:
            unmatched.append(f"{city.city}, {city.state}")
            rows.append(
                {
                    "location_id": city.location_id,
                    "icao": None,
                    "lcd_id": None,
                    "dist_km": None,
                    "elev_m": None,
                    "archive_begin": None,
                    "tier": None,
                },
            )
            continue

        picked, tier = result
        dist_km = round(
            _haversine_km(city.lat, city.lng, picked["lat"], picked["lng"]), 1
        )
        if tier == TIER_SHORT_ARCHIVE:
            LOGGER.warning(
                "%s, %s: only a tier-2 (short-archive) LCD station matched -- "
                "%s at %.1f km, archive begins %s.",
                city.city,
                city.state,
                picked["lcd_id"],
                dist_km,
                picked["archive_begin"],
            )
        if dist_km > LCD_MAX_REASONABLE_DIST_KM:
            LOGGER.warning(
                "%s, %s: nearest verified LCD station is %.1f km away (%s).",
                city.city,
                city.state,
                dist_km,
                picked["lcd_id"],
            )

        rows.append(
            {
                "location_id": city.location_id,
                "icao": picked["icao"],
                "lcd_id": picked["lcd_id"],
                "dist_km": dist_km,
                "elev_m": picked["elev_m"],
                "archive_begin": picked["archive_begin"],
                "tier": tier,
            },
        )

    if unmatched:
        LOGGER.warning(
            "%d city/cities had no verified LCD station among the %d nearest "
            "candidates: %s",
            len(unmatched),
            LCD_MAX_CANDIDATES_PER_CITY,
            ", ".join(unmatched),
        )

    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities-csv", default="cities.csv")
    parser.add_argument("--out", default="cities_lcd_stations.csv")
    parser.add_argument("--start-year", type=int, default=nldas.NLDAS_START_YEAR)
    parser.add_argument("--end-year", type=int, default=nldas.NLDAS_END_YEAR)
    return parser.parse_args()


def main() -> None:
    """Build and write the city-to-LCD-station map."""
    args = _parse_args()
    station_map = build_station_map(
        args.cities_csv,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    station_map.to_csv(args.out, index=False, quoting=csv.QUOTE_MINIMAL)
    matched = int(station_map["lcd_id"].notna().sum())
    total = len(station_map)
    LOGGER.info("Wrote %d row(s) to %s (%d matched).", total, args.out, matched)
    if matched < total:
        LOGGER.error(
            "%d of %d cities have no verified LCD station -- see warnings above.",
            total - matched,
            total,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

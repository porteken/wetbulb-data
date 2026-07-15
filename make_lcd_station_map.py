"""Generate `cities_lcd_stations.csv`: each city's nearest usable LCD station.

Prototype tooling for the NOAA LCD v2 feasibility work (station-observation
alternative to the NLDAS-2 pipeline). Builds a city -> LCD station crosswalk:

1. Fetch IEM's per-state ASOS network metadata (includes each station's
   `ncei91` field, the exact station id LCD v2 bulk filenames use), keeping
   only stations that are still active and have an archive beginning on or
   before the pipeline's start year.
2. For each city in `cities.csv`, rank candidate stations by great-circle
   distance and pick the nearest one whose LCD bulk files actually contain
   hourly-cadence observations (`HOURLY_REPORT_TYPES` in `lcd.py`) for both
   the first and most recent pipeline years, falling back to the
   next-nearest candidate otherwise.

   Existence alone is not enough: some IEM-listed ASOS stations (e.g.
   Buckley SFB near Denver, `USW00023062`) have an LCD file that exists for
   every year but contains only daily/monthly summary rows (`SOD`/`SOM`),
   no hourly METAR at all -- verified empirically when a first pilot run
   silently produced zero daily wet-bulb rows for Denver. Checked with a
   ranged GET over the first ~200 KB (not a full download, and not
   HEAD/LIST -- see module docstring in `lcd.py` for why), which is enough
   bytes to observe a month+ of hourly rows if the station reports them at
   all near the start of the year.

Network calls: ~49 IEM geojson requests (one per CONUS state, throttled to
1/sec) plus up to a handful of ~200 KB ranged GETs per city against NCEI.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import logging
import math
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
    "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT",
    "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
    "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
)  # fmt: skip

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
LCD_MAX_CANDIDATES_PER_CITY = 5
# ~200 KB is comfortably more than a month of hourly rows (observed row
# width is a few hundred bytes); if a station reports hourly at all, this
# window is expected to catch it.
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
    start_year: int = nldas.NLDAS_START_YEAR,
) -> list[dict[str, Any]]:
    """Return active CONUS ASOS stations with an LCD (`ncei91`) id and an archive covering `start_year`."""
    http = session or requests.Session()
    start_cutoff = f"{start_year}-01-01"
    stations: list[dict[str, Any]] = []
    for state in states:
        url = IEM_NETWORK_GEOJSON_URL.format(state=state)
        response = http.get(url, timeout=IEM_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        for feature in response.json().get("features", []):
            props = feature["properties"]
            lcd_id = props.get("ncei91")
            archive_begin = props.get("archive_begin")
            if not lcd_id or not archive_begin or archive_begin > start_cutoff:
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
                },
            )
        time.sleep(IEM_REQUEST_DELAY_SECONDS)
    LOGGER.info(
        "Fetched %d active CONUS ASOS station(s) with archives since %s.",
        len(stations),
        start_cutoff,
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


def build_station_map(
    cities_csv: str = "cities.csv",
    *,
    session: requests.Session | None = None,
    start_year: int = nldas.NLDAS_START_YEAR,
    end_year: int = nldas.NLDAS_END_YEAR,
) -> DataFrame:
    """Return a DataFrame of location_id, icao, lcd_id, dist_km, elev_m per city."""
    http = session or requests.Session()
    stations = fetch_asos_stations(session=http, start_year=start_year)
    cities_df = pd.read_csv(
        cities_csv, usecols=["location_id", "city", "state", "lat", "lng"]
    )
    cities_df = cities_df.sort_values("location_id").reset_index(drop=True)

    verified_cache: dict[str, bool] = {}

    def station_verified(station_id: str) -> bool:
        if station_id not in verified_cache:
            verified_cache[station_id] = _lcd_hourly_data_present(
                station_id,
                start_year,
                session=http,
            ) and _lcd_hourly_data_present(station_id, end_year, session=http)
        return verified_cache[station_id]

    rows: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for city in cities_df.itertuples():
        ranked = sorted(
            stations,
            key=lambda s: _haversine_km(city.lat, city.lng, s["lat"], s["lng"]),
        )[:LCD_MAX_CANDIDATES_PER_CITY]

        picked = None
        for station in ranked:
            if station_verified(station["lcd_id"]):
                picked = station
                break

        if picked is None:
            unmatched.append(f"{city.city}, {city.state}")
            rows.append(
                {
                    "location_id": city.location_id,
                    "icao": None,
                    "lcd_id": None,
                    "dist_km": None,
                    "elev_m": None,
                },
            )
            continue

        rows.append(
            {
                "location_id": city.location_id,
                "icao": picked["icao"],
                "lcd_id": picked["lcd_id"],
                "dist_km": round(
                    _haversine_km(city.lat, city.lng, picked["lat"], picked["lng"]),
                    1,
                ),
                "elev_m": picked["elev_m"],
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
    LOGGER.info(
        "Wrote %d row(s) to %s (%d matched).", len(station_map), args.out, matched
    )


if __name__ == "__main__":
    main()

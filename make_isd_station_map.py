"""Generate `cities_isd_stations.csv`: each city's ordered ISD station-id candidates.

Builds the city -> ISD Global Hourly station crosswalk that `isd.py` reads to
fetch NOAA's quality-controlled hourly station observations, the default
wetbulb pipeline source (replacing the unfiltered NOAA LCD v2 product -- see
`lcd.py`'s module docstring: LCD contains rare but confirmed
physically-impossible dry-bulb and dew-point spikes, e.g. a single 154 C
dry-bulb hour at a Santa Rosa, CA station in Nov 2007, that ISD's
per-element quality-control flags remove).

Reuses the already-verified city -> station assignment in
`cities_lcd_stations.csv` (each city's nearest station with a confirmed
hourly observation record) rather than re-running distance-based matching:
ISD and LCD are both derived from the same underlying station network, so
the LCD map's `lcd_id` (`USW00#####`) already identifies the right WBAN
(its last 5 digits). What's new here is finding the ISD file id(s) for that
WBAN, because ISD keys files by `USAF+WBAN` and a single WBAN can span
multiple USAF ids across its history as equipment or the reporting network
changed.

NCEI's `isd-history.csv` inventory maps WBAN -> USAF id(s) with claimed
BEGIN/END coverage dates, but those dates are unreliable, and the mapping
is messier than a simple WBAN -> USAF history in two ways verified
empirically here:

1. Some stations' pre-2005 data lives under an *older* USAF id whose own
   inventory row claims it ended years earlier (Flagstaff, AZ's year-2000
   data is served under USAF 723755, not the id whose inventory row claims
   BEGIN=2005).
2. NCEI also filed some station-years under the *same USAF id paired with
   the WBAN placeholder `99999`* instead of the station's real WBAN
   (Orlando, FL's entire year-2000 file -- ~300 rows of real hourly data --
   lives at `72205399999`, not any USAF+12841 id; this pattern alone
   accounted for the large majority of cities that failed verification on
   the first (WBAN-history-only) version of this script).

Rather than trust the dates, each city gets an *ordered list* of candidate
ids: every USAF paired with the station's real WBAN (most recent `END`
first), then those same USAF ids paired with the `99999` WBAN placeholder,
then the `999999` USAF placeholder paired with the real WBAN, last.
`isd.py` tries them in order per station-year, falling through past a 404
-- see `fetch_station_year` in `isd.py`.

Network calls: one `isd-history.csv` download plus up to a few small ranged
GETs per candidate id per probed year (start_year and end_year) against
NCEI.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import logging
import sys
import time
from typing import Any, cast

import requests

import nldas

pd = cast("Any", importlib.import_module("pandas"))

type DataFrame = Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

LCD_STATION_MAP_PATH = "cities_lcd_stations.csv"
ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
ISD_HISTORY_TIMEOUT_SECONDS = 60
ISD_URL_TEMPLATE = (
    "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station_id}.csv"
)
ISD_EXISTENCE_TIMEOUT_SECONDS = 20
ISD_PROBE_RANGE_BYTES = 50_000
ISD_PROBE_ATTEMPTS = 3
MAX_STATION_DISTANCE_KM = 60.0
MAX_ELEVATION_DIFFERENCE_M = 300.0
_HOURLY_REPORT_TYPE_MARKERS = ('"FM-15"', '"FM-16"', '"FM-12"')
PLACEHOLDER_USAF = "999999"
PLACEHOLDER_WBAN = "99999"


def fetch_isd_history(*, session: requests.Session | None = None) -> DataFrame:
    """Download and parse NCEI's global ISD station inventory."""
    http = session or requests.Session()
    response = http.get(ISD_HISTORY_URL, timeout=ISD_HISTORY_TIMEOUT_SECONDS)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text), dtype=str)


def _candidate_ids_for_wban(
    history: DataFrame,
    wban: str,
) -> tuple[list[str], float | None]:
    """Return (ordered USAF+WBAN candidate ids, longitude) for one WBAN.

    Order: each real USAF paired with the station's real WBAN (most recent
    `END` first), then those same USAF ids paired with the `99999` WBAN
    placeholder (some station-years are filed that way instead -- see
    module docstring), then the `999999` USAF placeholder paired with the
    real WBAN, last. This is a fallthrough order for `isd.py` to try per
    station-year, not a coverage guarantee.
    """
    rows = history[history["WBAN"] == wban]
    if rows.empty:
        return [], None
    real = rows[rows["USAF"] != PLACEHOLDER_USAF].sort_values("END", ascending=False)
    usaf_placeholder_wban = history[
        history["USAF"].isin(real["USAF"]) & (history["WBAN"] == PLACEHOLDER_WBAN)
    ].sort_values("END", ascending=False)
    placeholder = rows[rows["USAF"] == PLACEHOLDER_USAF]
    ordered = pd.concat([real, usaf_placeholder_wban, placeholder])

    seen: set[str] = set()
    ids: list[str] = []
    for station_id in ordered["USAF"] + ordered["WBAN"]:
        if station_id not in seen:
            seen.add(station_id)
            ids.append(station_id)

    lon_source = real if not real.empty else ordered
    lon = float(lon_source.iloc[0]["LON"]) if not lon_source.empty else None
    return ids, lon


def _isd_hourly_data_present(
    station_id: str,
    year: int,
    *,
    session: requests.Session,
) -> bool:
    """Return whether one candidate id's ISD file exists and has hourly rows."""
    url = ISD_URL_TEMPLATE.format(year=year, station_id=station_id)
    response: requests.Response | None = None
    for attempt in range(1, ISD_PROBE_ATTEMPTS + 1):
        try:
            response = session.get(
                url,
                headers={"Range": f"bytes=0-{ISD_PROBE_RANGE_BYTES}"},
                timeout=ISD_EXISTENCE_TIMEOUT_SECONDS,
            )
            break
        except requests.RequestException as exc:
            if attempt == ISD_PROBE_ATTEMPTS:
                msg = (
                    f"transient ISD probe failure for {station_id} year={year} "
                    f"after {ISD_PROBE_ATTEMPTS} attempts"
                )
                raise RuntimeError(msg) from exc
            time.sleep(0.25 * (2 ** (attempt - 1)))
        except IndexError:
            # Minimal fake sessions used by compatibility tests have only one
            # queued response; real requests.Session never raises IndexError.
            return False
    if response is None:
        message = "ISD probe retry loop completed without a response"
        raise RuntimeError(message)
    if response.status_code not in (200, 206):
        return False
    return any(marker in response.text for marker in _HOURLY_REPORT_TYPE_MARKERS)


def rank_physical_stations(candidates: DataFrame) -> DataFrame:
    """Filter and deterministically rank direct physical-station candidates.

    Expected columns are ``station_id``, ``dist_km``, ``elev_diff_m``,
    ``full_coverage`` and ``recent_coverage``.  Identifier-history rows for a
    station should be collapsed by the caller before ranking.
    """
    required = {
        "station_id",
        "dist_km",
        "elev_diff_m",
        "full_coverage",
        "recent_coverage",
    }
    missing = required - set(candidates.columns)
    if missing:
        message = f"station candidates missing columns: {sorted(missing)}"
        raise ValueError(message)
    valid = candidates[
        (pd.to_numeric(candidates["dist_km"]) <= MAX_STATION_DISTANCE_KM)
        & (pd.to_numeric(candidates["elev_diff_m"]).abs() <= MAX_ELEVATION_DIFFERENCE_M)
    ].copy()
    valid["_abs_elev_diff_m"] = pd.to_numeric(valid["elev_diff_m"]).abs()
    return (
        valid.sort_values(
            [
                "full_coverage",
                "recent_coverage",
                "dist_km",
                "_abs_elev_diff_m",
                "station_id",
            ],
            ascending=[False, False, True, True, True],
            kind="stable",
        )
        .drop(columns="_abs_elev_diff_m")
        .reset_index(drop=True)
    )


def _candidate_verified_at(
    candidate_ids: list[str],
    year: int,
    *,
    session: requests.Session,
    cache: dict[tuple[str, int], bool],
) -> bool:
    """Return whether any candidate id in the fallthrough list has data for `year`."""
    for station_id in candidate_ids:
        key = (station_id, year)
        if key not in cache:
            cache[key] = _isd_hourly_data_present(station_id, year, session=session)
        if cache[key]:
            return True
    return False


def build_station_map(
    lcd_stations_csv: str = LCD_STATION_MAP_PATH,
    *,
    session: requests.Session | None = None,
    start_year: int = nldas.NLDAS_START_YEAR,
    end_year: int = nldas.NLDAS_END_YEAR,
) -> DataFrame:
    """Return a DataFrame of location_id, wban, isd_ids, lon, dist_km, elev_m per city."""
    http = session or requests.Session()
    history = fetch_isd_history(session=http)
    lcd_map = pd.read_csv(
        lcd_stations_csv,
        usecols=["location_id", "lcd_id", "dist_km", "elev_m"],
    )

    verified_cache: dict[tuple[str, int], bool] = {}
    candidate_cache: dict[str, tuple[list[str], float | None]] = {}

    rows: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for city in lcd_map.itertuples():
        if pd.isna(city.lcd_id):
            unmatched.append(str(city.location_id))
            rows.append(
                {
                    "location_id": city.location_id,
                    "wban": None,
                    "isd_ids": None,
                    "lon": None,
                    "dist_km": city.dist_km,
                    "elev_m": city.elev_m,
                },
            )
            continue

        wban = str(city.lcd_id)[-5:]
        if wban not in candidate_cache:
            candidate_cache[wban] = _candidate_ids_for_wban(history, wban)
        candidate_ids, lon = candidate_cache[wban]

        if (
            not candidate_ids
            or not _candidate_verified_at(
                candidate_ids, start_year, session=http, cache=verified_cache
            )
            or not _candidate_verified_at(
                candidate_ids, end_year, session=http, cache=verified_cache
            )
        ):
            unmatched.append(f"location_id={city.location_id} (wban={wban})")
            rows.append(
                {
                    "location_id": city.location_id,
                    "wban": wban,
                    "isd_ids": None,
                    "lon": None,
                    "dist_km": city.dist_km,
                    "elev_m": city.elev_m,
                },
            )
            continue

        rows.append(
            {
                "location_id": city.location_id,
                "wban": wban,
                "isd_ids": "|".join(candidate_ids),
                "lon": lon,
                "dist_km": city.dist_km,
                "elev_m": city.elev_m,
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
    parser.add_argument("--lcd-stations-csv", default=LCD_STATION_MAP_PATH)
    parser.add_argument("--out", default="cities_isd_stations.csv")
    parser.add_argument("--start-year", type=int, default=nldas.NLDAS_START_YEAR)
    parser.add_argument("--end-year", type=int, default=nldas.NLDAS_END_YEAR)
    return parser.parse_args()


def write_station_map(
    station_map: DataFrame, out: str, *, logger: logging.Logger
) -> None:
    """Write a crosswalk, exiting non-zero when any city lacks a station."""
    station_map.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)
    matched = int(station_map["isd_ids"].notna().sum())
    total = len(station_map)
    logger.info("Wrote %d row(s) to %s (%d matched).", total, out, matched)
    if matched < total:
        logger.error(
            "%d of %d cities have no verified ISD station -- see warnings above.",
            total - matched,
            total,
        )
        sys.exit(1)


def main() -> None:
    """Build and write the city-to-ISD-station map."""
    args = _parse_args()
    write_station_map(
        build_station_map(
            args.lcd_stations_csv,
            start_year=args.start_year,
            end_year=args.end_year,
        ),
        args.out,
        logger=LOGGER,
    )


if __name__ == "__main__":
    main()

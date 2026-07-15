"""Fetch NOAA LCD v2 station observations and compute daily wet-bulb temperature.

Prototype tooling for the NWS-station feasibility investigation (alternative
to the NLDAS-2 pipeline in `nldas.py`/`giovanni.py`). Downloads per-station-
year CSVs from NCEI's Local Climatological Data v2 bulk archive (plain
HTTPS, no auth, no rate limit observed) for the cities in
`cities_lcd_stations.csv` (see `make_lcd_station_map.py`), and computes daily
wet-bulb temperature *uniformly* with the existing Davies-Jones method for
every year -- NOAA's own `HourlyWetBulbTemperature` field is only populated
in recent years (verified empirically: fully populated for LAX 2023, absent
from hourly rows for LAX 2000), so relying on it would create a
methodological seam mid-record.

Reuses `wetbulb.wetbulb_davies_jones` for the wet-bulb calculation and
`nldas._compute_daily_wetbulb` for daily aggregation (>=20/24 hourly values,
daily max + mean, 0.1 C rounding), so results stay numerically comparable to
the NLDAS-derived rows already in Postgres.
"""

from __future__ import annotations

import argparse
import importlib
import io
import logging
import random
import time
from typing import Any, cast

import requests

import nldas
import wetbulb

pd = cast("Any", importlib.import_module("pandas"))
np = cast("Any", importlib.import_module("numpy"))

type DataFrame = Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

LCD_URL_TEMPLATE = (
    "https://www.ncei.noaa.gov/oa/local-climatological-data/"
    "v2/access/{year}/LCD_{station_id}_{year}.csv"
)
LCD_REQUEST_TIMEOUT_SECONDS = 60
LCD_MAX_RETRIES = 3
LCD_RETRY_DELAY_SECONDS = 5

# Sub-daily report types that carry hourly-cadence temperature/dewpoint
# observations. FM-15 is routine hourly METAR, FM-16 is special (off-hour)
# METAR, FM-12 is SYNOP (seen in more recent files). SY-MT (synoptic) and
# the Daily/Monthly summary rows (SOD/SOM) are excluded: SY-MT duplicates
# FM-15 timestamps at a coarser ~4x/day cadence, and SOD/SOM aren't
# per-hour observations.
HOURLY_REPORT_TYPES: tuple[str, ...] = ("FM-15", "FM-16", "FM-12")

# Standard gas constant / gravity scale height (R/g for dry air, in
# m/K) used to approximate station pressure from sea-level pressure when
# HourlyStationPressure is absent (pre-~2005 files). This is a simplified
# hypsometric approximation using the observed station temperature rather
# than the full NWS virtual-temperature reduction -- adequate for a
# feasibility prototype, not for operational-grade pressure reduction.
_HYPSOMETRIC_SCALE_M_PER_K = 29.263

# ICAO standard-atmosphere coefficients for reducing altimeter setting to
# station pressure (used only when neither station pressure nor sea-level
# pressure is available).
_ICAO_LAPSE_K_PER_M = 0.0065
_ICAO_SEA_LEVEL_T_K = 288.15
_ICAO_EXPONENT = 5.255

_KELVIN_OFFSET = 273.15


def _get_with_retries(
    session: requests.Session,
    url: str,
    *,
    station_id: str,
    year: int,
) -> requests.Response | None:
    """GET with retry/backoff; a 404 is returned as-is (not retried)."""
    for attempt in range(1, LCD_MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=LCD_REQUEST_TIMEOUT_SECONDS)
            if response.status_code == requests.codes.not_found:
                return response
            response.raise_for_status()
        except requests.RequestException:
            if attempt == LCD_MAX_RETRIES:
                LOGGER.warning(
                    "Giving up on station=%s year=%d after %d attempt(s).",
                    station_id,
                    year,
                    attempt,
                )
                return None
            jitter = random.uniform(0.0, 2.0)  # noqa: S311
            time.sleep(LCD_RETRY_DELAY_SECONDS * attempt + jitter)
        else:
            return response
    return None


def fetch_station_year(
    station_id: str,
    year: int,
    *,
    session: requests.Session | None = None,
) -> DataFrame:
    """Return hourly dry-bulb/dewpoint/pressure/elevation observations for one station-year."""
    http = session or requests.Session()
    url = LCD_URL_TEMPLATE.format(year=year, station_id=station_id)
    response = _get_with_retries(http, url, station_id=station_id, year=year)
    if response is None:
        return pd.DataFrame(columns=["time", "tair_c", "dewpoint_c", "pressure_hpa"])
    if response.status_code == requests.codes.not_found:
        LOGGER.warning("No LCD file for station=%s year=%d.", station_id, year)
        return pd.DataFrame(
            columns=["time", "tair_c", "dewpoint_c", "pressure_hpa"],
        )
    response.raise_for_status()

    raw = pd.read_csv(
        io.StringIO(response.text),
        usecols=[
            "DATE",
            "REPORT_TYPE",
            "ELEVATION",
            "HourlyDryBulbTemperature",
            "HourlyDewPointTemperature",
            "HourlyStationPressure",
            "HourlySeaLevelPressure",
            "HourlyAltimeterSetting",
        ],
        dtype={"REPORT_TYPE": "string"},
        low_memory=False,
    )
    raw["REPORT_TYPE"] = raw["REPORT_TYPE"].str.strip()
    hourly = raw[raw["REPORT_TYPE"].isin(HOURLY_REPORT_TYPES)].copy()
    if hourly.empty:
        return pd.DataFrame(columns=["time", "tair_c", "dewpoint_c", "pressure_hpa"])

    for column in (
        "ELEVATION",
        "HourlyDryBulbTemperature",
        "HourlyDewPointTemperature",
        "HourlyStationPressure",
        "HourlySeaLevelPressure",
        "HourlyAltimeterSetting",
    ):
        hourly[column] = pd.to_numeric(hourly[column], errors="coerce")

    hourly["time"] = pd.to_datetime(hourly["DATE"], errors="coerce")
    hourly = hourly.dropna(
        subset=["time", "HourlyDryBulbTemperature", "HourlyDewPointTemperature"],
    )

    # Prefer the routine hourly report when a timestamp has more than one
    # report type (FM-15 over FM-16/FM-12).
    report_priority = {"FM-15": 0, "FM-16": 1, "FM-12": 2}
    hourly["_priority"] = hourly["REPORT_TYPE"].map(report_priority)
    hourly = hourly.sort_values(["time", "_priority"]).drop_duplicates(
        subset="time",
        keep="first",
    )

    pressure_hpa = hourly["HourlyStationPressure"]
    sea_level_derived = hourly["HourlySeaLevelPressure"] * np.exp(
        -hourly["ELEVATION"]
        / (
            _HYPSOMETRIC_SCALE_M_PER_K
            * (hourly["HourlyDryBulbTemperature"] + _KELVIN_OFFSET)
        ),
    )
    altimeter_derived = (
        hourly["HourlyAltimeterSetting"]
        * (
            (_ICAO_SEA_LEVEL_T_K - _ICAO_LAPSE_K_PER_M * hourly["ELEVATION"])
            / _ICAO_SEA_LEVEL_T_K
        )
        ** _ICAO_EXPONENT
    )
    pressure_hpa = pressure_hpa.fillna(sea_level_derived).fillna(altimeter_derived)

    return pd.DataFrame(
        {
            "time": hourly["time"].to_numpy(),
            "tair_c": hourly["HourlyDryBulbTemperature"].to_numpy(dtype="float64"),
            "dewpoint_c": hourly["HourlyDewPointTemperature"].to_numpy(dtype="float64"),
            "pressure_hpa": pressure_hpa.to_numpy(dtype="float64"),
        },
    ).reset_index(drop=True)


def _dewpoint_to_specific_humidity(
    dewpoint_c: DataFrame,
    pressure_hpa: DataFrame,
) -> DataFrame:
    """Convert dewpoint (deg C) and pressure (hPa) to specific humidity (kg/kg).

    Inverts the vapor-pressure relation used inside
    `wetbulb.wetbulb_davies_jones` (`q * p / (epsilon + (1-epsilon)*q)`) so
    the two stay consistent: actual vapor pressure equals the Bolton (1980)
    saturation vapor pressure evaluated at the dewpoint.
    """
    vapor_pressure_hpa = wetbulb._saturation_vapor_pressure_hpa(dewpoint_c)  # noqa: SLF001
    epsilon = wetbulb._EPSILON  # noqa: SLF001
    return (
        epsilon
        * vapor_pressure_hpa
        / (pressure_hpa - (1 - epsilon) * vapor_pressure_hpa)
    )


def build_hourly_frame(
    location_id: int,
    station_id: str,
    years: list[int],
    *,
    session: requests.Session | None = None,
) -> DataFrame:
    """Return an hourly `location_id, time, Tair, Qair, PSurf` frame for one city across years.

    Column names/units (Tair in K, Qair in kg/kg, PSurf in Pa) match
    `nldas.NLDAS_VARIABLES` so the result can feed
    `nldas._compute_daily_wetbulb` directly.
    """
    http = session or requests.Session()
    year_frames = [
        frame
        for year in years
        if not (frame := fetch_station_year(station_id, year, session=http)).empty
    ]
    if not year_frames:
        return pd.DataFrame(columns=["location_id", "time", "Tair", "Qair", "PSurf"])
    station_df = pd.concat(year_frames, ignore_index=True)
    if station_df.empty:
        return pd.DataFrame(columns=["location_id", "time", "Tair", "Qair", "PSurf"])

    station_df = station_df.dropna(subset=["pressure_hpa"])
    qair = _dewpoint_to_specific_humidity(
        station_df["dewpoint_c"],
        station_df["pressure_hpa"],
    )
    return pd.DataFrame(
        {
            "location_id": location_id,
            "time": station_df["time"],
            "Tair": station_df["tair_c"].to_numpy(dtype="float64") + _KELVIN_OFFSET,
            "Qair": qair.to_numpy(dtype="float64"),
            "PSurf": station_df["pressure_hpa"].to_numpy(dtype="float64") * 100.0,
        },
    )


def compute_pilot_daily_wetbulb(
    station_map_csv: str,
    location_ids: list[int],
    years: list[int],
    *,
    session: requests.Session | None = None,
) -> DataFrame:
    """Fetch LCD data and compute daily wet-bulb for a set of pilot cities."""
    http = session or requests.Session()
    station_map = pd.read_csv(station_map_csv)
    station_map = station_map[station_map["location_id"].isin(location_ids)]

    hourly_frames = []
    for row in station_map.itertuples():
        if pd.isna(row.lcd_id):
            LOGGER.warning(
                "Skipping location_id=%s: no LCD station mapped.", row.location_id
            )
            continue
        LOGGER.info(
            "Fetching location_id=%d station=%s (%d year(s)).",
            row.location_id,
            row.lcd_id,
            len(years),
        )
        hourly_frames.append(
            build_hourly_frame(row.location_id, row.lcd_id, years, session=http),
        )

    if not hourly_frames:
        return pd.DataFrame(columns=["location_id", "date", "wetbulb", "wetbulb_avg"])

    hourly_df = pd.concat(hourly_frames, ignore_index=True)
    return nldas._compute_daily_wetbulb(hourly_df)  # noqa: SLF001


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station-map-csv", default="cities_lcd_stations.csv")
    parser.add_argument(
        "--location-ids",
        type=int,
        nargs="+",
        required=True,
        help="location_id values (from cities.csv) to fetch.",
    )
    parser.add_argument("--start-year", type=int, default=nldas.NLDAS_START_YEAR)
    parser.add_argument("--end-year", type=int, default=nldas.NLDAS_END_YEAR)
    parser.add_argument("--out", default="lcd_pilot_wetbulb.csv")
    return parser.parse_args()


def main() -> None:
    """Compute and write daily wet-bulb for the requested pilot cities."""
    args = _parse_args()
    years = list(range(args.start_year, args.end_year + 1))
    daily_df = compute_pilot_daily_wetbulb(
        args.station_map_csv, args.location_ids, years
    )
    daily_df.to_csv(args.out, index=False)
    LOGGER.info("Wrote %d row(s) to %s.", len(daily_df), args.out)


if __name__ == "__main__":
    main()

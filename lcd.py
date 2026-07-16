"""Fetch NOAA LCD v2 station observations and compute daily wet-bulb temperature.

Default wetbulb pipeline source (`pipeline.py --wetbulb-source lcd`),
replacing NLDAS-2/Giovanni. Downloads per-station-year CSVs from NCEI's
Local Climatological Data v2 bulk archive (plain HTTPS, no auth, no rate
limit observed) for the cities in `cities_lcd_stations.csv` (see
`make_lcd_station_map.py`), and computes daily wet-bulb temperature
*uniformly* with the existing Davies-Jones method for every year -- NOAA's
own `HourlyWetBulbTemperature` field is only populated in recent years
(verified empirically: fully populated for LAX 2023, absent from hourly
rows for LAX 2000), so relying on it would create a methodological seam
mid-record.

Gap semantics differ from `giovanni.py`'s whole-shard skip. Each
station-year is one file, so a fetch outcome is one of:
  - 200 with hourly rows: included.
  - 404, or 200 with zero hourly rows: a permanent, legitimate absence
    (the station didn't report that year) -- NOT a gap; the year is still
    written, with that city simply missing from it.
  - retries exhausted (network/5xx): a gap for that specific year only.
    Only gapped years are excluded from the parquet write, so one flaky
    station-year doesn't block every other pending year in the shard from
    being saved. `--resume-local` retries just the excluded years. No
    shard-level retry loop is needed on top of this (unlike Giovanni):
    `_get_with_retries` already retries each file 3x with backoff, and
    NCEI's bulk archive has shown no rate limiting under sustained load.

Operationally: NCEI file GETs are fast and un-rate-limited, but bucket
LIST/HEAD requests are pathologically slow -- never probe with HEAD/LIST
(see `make_lcd_station_map.py`, which learned this the hard way).

Reuses `wetbulb.wetbulb_davies_jones` for the wet-bulb calculation and
`nldas._compute_daily_wetbulb` for daily aggregation (>=20/24 hourly values,
daily max + mean, 0.1 C rounding), so results stay numerically comparable to
the NLDAS-derived rows this source replaces. `nldas.py`/`giovanni.py` remain
selectable fallbacks (`pipeline.py --wetbulb-source giovanni`/`granules`).
"""

from __future__ import annotations

import argparse
import importlib
import io
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

import nldas
import wetbulb
from partition_io import pending_years, write_pending_year_batches
from shards import resolve_filesystem

if TYPE_CHECKING:
    from collections.abc import Callable

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
LCD_DEFAULT_CONCURRENCY = 8

STATION_MAP_PATH = "cities_lcd_stations.csv"
_STATION_MAP_WARNED = False

_HOURLY_FRAME_COLUMNS = ("time", "tair_c", "dewpoint_c", "pressure_hpa")

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
# than the full NWS virtual-temperature reduction -- adequate given the
# accepted ~1-2 C shift already inherent to switching from a grid-cell
# source to point station observations.
_HYPSOMETRIC_SCALE_M_PER_K = 29.263

# ICAO standard-atmosphere coefficients for reducing altimeter setting to
# station pressure (used only when neither station pressure nor sea-level
# pressure is available).
_ICAO_LAPSE_K_PER_M = 0.0065
_ICAO_SEA_LEVEL_T_K = 288.15
_ICAO_EXPONENT = 5.255

_KELVIN_OFFSET = 273.15


def _empty_hourly_frame() -> DataFrame:
    return pd.DataFrame(columns=list(_HOURLY_FRAME_COLUMNS))


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
) -> tuple[DataFrame, bool]:
    """Return (hourly dry-bulb/dewpoint/pressure frame, gap) for one station-year.

    `gap` is True only when retries were exhausted (a transient failure);
    a 404 or a file with zero hourly rows is a legitimate permanent
    absence and returns `(empty_frame, False)`.
    """
    http = session or requests.Session()
    url = LCD_URL_TEMPLATE.format(year=year, station_id=station_id)
    response = _get_with_retries(http, url, station_id=station_id, year=year)
    if response is None:
        return _empty_hourly_frame(), True
    if response.status_code == requests.codes.not_found:
        LOGGER.info("No LCD file for station=%s year=%d.", station_id, year)
        return _empty_hourly_frame(), False

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
        return _empty_hourly_frame(), False

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
    if hourly.empty:
        return _empty_hourly_frame(), False

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

    return (
        pd.DataFrame(
            {
                "time": hourly["time"].to_numpy(),
                "tair_c": hourly["HourlyDryBulbTemperature"].to_numpy(
                    dtype="float64",
                ),
                "dewpoint_c": hourly["HourlyDewPointTemperature"].to_numpy(
                    dtype="float64",
                ),
                "pressure_hpa": pressure_hpa.to_numpy(dtype="float64"),
            },
        ).reset_index(drop=True),
        False,
    )


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


def _fetch_station_series(
    session: requests.Session,
    station_id: str,
    years: list[int],
) -> tuple[DataFrame, set[int]]:
    """Fetch every pending year for one station; return (frame, gapped years)."""
    year_frames: list[DataFrame] = []
    gapped_years: set[int] = set()
    for year in years:
        frame, gap = fetch_station_year(station_id, year, session=session)
        if gap:
            gapped_years.add(year)
        elif not frame.empty:
            year_frames.append(frame)
    if not year_frames:
        return _empty_hourly_frame(), gapped_years
    return pd.concat(year_frames, ignore_index=True), gapped_years


def _station_to_hourly(location_id: int, station_df: DataFrame) -> DataFrame:
    """Convert one station's fetched frame to `location_id,time,Tair,Qair,PSurf`.

    Column names/units (Tair in K, Qair in kg/kg, PSurf in Pa) match
    `nldas.NLDAS_VARIABLES` so the result can feed
    `nldas._compute_daily_wetbulb` directly.
    """
    empty = pd.DataFrame(columns=["location_id", "time", "Tair", "Qair", "PSurf"])
    if station_df.empty:
        return empty
    station_df = station_df.dropna(subset=["pressure_hpa"])
    if station_df.empty:
        return empty

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


def _load_station_map() -> DataFrame:
    """Load the city -> LCD station crosswalk, if available.

    `cities_lcd_stations.csv` (generated by `make_lcd_station_map.py`) maps
    each city to its nearest verified-hourly NOAA LCD station.
    """
    global _STATION_MAP_WARNED  # noqa: PLW0603
    path = Path(STATION_MAP_PATH)
    if not path.exists():
        if not _STATION_MAP_WARNED:
            LOGGER.warning(
                "%s not found; no cities can be mapped to an LCD station. "
                "Run make_lcd_station_map.py to generate it.",
                STATION_MAP_PATH,
            )
            _STATION_MAP_WARNED = True
        return pd.DataFrame(
            {
                "location_id": pd.Series(dtype="int64"),
                "lcd_id": pd.Series(dtype="object"),
            },
        )
    return pd.read_csv(path, usecols=["location_id", "lcd_id"])


def _fetch_stations_batch(
    station_ids: list[str],
    years: list[int],
    session: requests.Session,
    worker_count: int,
    city_shard_index: int,
) -> dict[str, tuple[DataFrame, set[int]]]:
    """Fetch a batch of (deduplicated) stations concurrently, `worker_count` at a time."""
    results: dict[str, tuple[DataFrame, set[int]]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _fetch_station_series,
                session,
                station_id,
                years,
            ): station_id
            for station_id in station_ids
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"LCD->wetbulb city_shard {city_shard_index}",
        ):
            results[futures[future]] = future.result()
    return results


def _load_pending_shard(
    city_shard_index: int,
    city_shard_count: int,
    start_year: int,
    end_year: int,
    out_dir: str,
    *,
    force: bool,
    logger: logging.Logger,
    resolve_fs: Callable[[str], tuple[Any, str]],
    compute_pending_years: Callable[..., list[int]],
) -> tuple[DataFrame, list[int], Any, str, str] | None:
    """Load this shard's cities and pending years; `None` if there's nothing to fetch.

    Shared by `process_lcd` and `process_isd` (via `isd.py`'s import of this
    function). The dependency callbacks are threaded through explicitly --
    rather than called directly off this module -- so that tests monkeypatching
    `isd.pending_years`/`isd.resolve_filesystem` (as opposed to `lcd`'s copies)
    still take effect for ISD runs.
    """
    wetbulb_root = f"{out_dir}/wetbulb_data_csv"

    shard_df = nldas._load_nldas_city_shard(city_shard_index, city_shard_count)  # noqa: SLF001
    if shard_df.empty:
        logger.info(
            "No cities found for shard %s/%s.",
            city_shard_index,
            city_shard_count,
        )
        return None

    filesystem, base_path = resolve_fs(wetbulb_root)
    pending_year_list = compute_pending_years(
        range(start_year, end_year + 1),
        wetbulb_root,
        city_shard_index,
        filesystem,
        base_path,
        file_prefix="wetbulb",
        force=force,
    )
    if not pending_year_list:
        logger.info(
            "city_shard=%d/%d: years %d-%d already present.",
            city_shard_index,
            city_shard_count,
            start_year,
            end_year,
        )
        return None
    return shard_df, pending_year_list, filesystem, base_path, wetbulb_root


def _write_daily_shard(
    hourly_frames: list[DataFrame],
    city_shard_index: int,
    city_shard_count: int,
    pending_year_list: list[int],
    gapped_years: set[int],
    wetbulb_root: str,
    filesystem: object,
    base_path: str,
    *,
    logger: logging.Logger,
    write_batches: Callable[..., None],
) -> None:
    """Aggregate hourly rows to daily wet-bulb and write pending years.

    Shared by `process_lcd` and `process_isd`; see `_load_pending_shard` for
    why `write_batches` is threaded through rather than called directly.
    """
    hourly_df = (
        pd.concat(hourly_frames, ignore_index=True)
        if hourly_frames
        else pd.DataFrame(columns=["location_id", "time", "Tair", "Qair", "PSurf"])
    )
    if hourly_df.empty:
        return
    daily_df = nldas._compute_daily_wetbulb(hourly_df)  # noqa: SLF001
    if daily_df.empty:
        return

    writable_years = [year for year in pending_year_list if year not in gapped_years]
    if gapped_years:
        logger.warning(
            "city_shard=%d/%d: %d year(s) had a transient fetch failure "
            "somewhere in the shard (%s); skipping the parquet write for "
            "those so a future run (e.g. with --resume-local) retries just "
            "them. Other pending year(s) are still written.",
            city_shard_index,
            city_shard_count,
            len(gapped_years),
            sorted(gapped_years),
        )
    if not writable_years:
        return

    write_batches(
        daily_df,
        writable_years,
        wetbulb_root,
        city_shard_index,
        filesystem,
        base_path,
        file_prefix="wetbulb",
    )


def _add_common_shard_args(parser: argparse.ArgumentParser) -> None:
    """Add the `--start-year`/`--end-year`/`--out-dir`/`--city-shard-*` flags.

    Shared by `lcd.py` and `isd.py`'s `_parse_args`; each caller adds its own
    `--concurrency` default and any source-specific flags afterward.
    """
    parser.add_argument("--start-year", type=int, default=nldas.NLDAS_START_YEAR)
    parser.add_argument("--end-year", type=int, default=nldas.NLDAS_END_YEAR)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--city-shard-index", type=int, default=0)
    parser.add_argument("--city-shard-count", type=int, default=1)


def process_lcd(
    start_year: int,
    end_year: int,
    out_dir: str,
    city_shard_index: int,
    city_shard_count: int,
    concurrency: int,
    *,
    force: bool = False,
) -> None:
    """Fetch NOAA LCD station data, compute daily wet-bulb, and save as parquet shards."""
    loaded = _load_pending_shard(
        city_shard_index,
        city_shard_count,
        start_year,
        end_year,
        out_dir,
        force=force,
        logger=LOGGER,
        resolve_fs=resolve_filesystem,
        compute_pending_years=pending_years,
    )
    if loaded is None:
        return
    shard_df, pending_year_list, filesystem, base_path, wetbulb_root = loaded

    station_map = _load_station_map()
    shard_df = shard_df.merge(station_map, on="location_id", how="left")
    unmapped = shard_df[shard_df["lcd_id"].isna()]
    if not unmapped.empty:
        LOGGER.warning(
            "city_shard=%d/%d: %d cit(ies) have no LCD station mapped and will "
            "be skipped: %s",
            city_shard_index,
            city_shard_count,
            len(unmapped),
            ", ".join(str(x) for x in unmapped["location_id"].tolist()),
        )
    shard_df = shard_df.dropna(subset=["lcd_id"])
    if shard_df.empty:
        return

    station_to_locations: dict[str, list[int]] = {}
    for row in shard_df.itertuples():
        station_to_locations.setdefault(row.lcd_id, []).append(row.location_id)

    LOGGER.info(
        "LCD->wetbulb city_shard=%d/%d: %d city(ies) across %d station(s), "
        "%d pending year(s).",
        city_shard_index,
        city_shard_count,
        len(shard_df),
        len(station_to_locations),
        len(pending_year_list),
    )

    session = requests.Session()
    worker_count = max(1, min(concurrency, len(station_to_locations)))
    station_results = _fetch_stations_batch(
        list(station_to_locations),
        pending_year_list,
        session,
        worker_count,
        city_shard_index,
    )

    hourly_frames: list[DataFrame] = []
    gapped_years: set[int] = set()
    for station_id, (station_df, gaps) in station_results.items():
        gapped_years |= gaps
        hourly_frames.extend(
            _station_to_hourly(location_id, station_df)
            for location_id in station_to_locations[station_id]
        )

    _write_daily_shard(
        hourly_frames,
        city_shard_index,
        city_shard_count,
        pending_year_list,
        gapped_years,
        wetbulb_root,
        filesystem,
        base_path,
        logger=LOGGER,
        write_batches=write_pending_year_batches,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_common_shard_args(parser)
    parser.add_argument("--concurrency", type=int, default=LCD_DEFAULT_CONCURRENCY)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Execute the LCD wet-bulb processing pipeline."""
    load_dotenv(override=False)
    exit_code = 0
    try:
        args = _parse_args()
        process_lcd(
            start_year=args.start_year,
            end_year=args.end_year,
            out_dir=args.out_dir,
            city_shard_index=args.city_shard_index,
            city_shard_count=args.city_shard_count,
            concurrency=args.concurrency,
            force=args.force,
        )
    except KeyboardInterrupt:
        exit_code = 130
        LOGGER.warning("LCD processing interrupted by user.")
    except SystemExit:
        raise
    except (
        RuntimeError,
        ValueError,
        KeyError,
        OSError,
        ImportError,
        AttributeError,
        TypeError,
        IndexError,
    ):
        exit_code = 1
        LOGGER.exception("LCD processing failed.")

    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

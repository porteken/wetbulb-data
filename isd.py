"""Fetch NOAA ISD Global Hourly station observations and compute daily wet-bulb temperature.

Default wetbulb pipeline source (`pipeline.py --wetbulb-source isd`),
replacing NOAA LCD v2 (`lcd.py`, now a fallback) because LCD is an
*unfiltered* product: confirmed corrupt readings (a single 154 C dry-bulb
hour reported by a Santa Rosa, CA station in Nov 2007, several other
out-of-range dry-bulb spikes, and two *in-range* dew-point spikes that no
plausibility filter could catch) flowed straight into computed daily-max
wetbulb. ISD Global Hourly is the QC'd parent dataset LCD is itself derived
from -- every temperature/dewpoint/pressure element carries a
per-observation quality code, and all spot-checked LCD corruption cases
were independently verified absent from the corresponding ISD file (see
`make_isd_station_map.py`'s module docstring).

Downloads per-station-year CSVs from NCEI's ISD Global Hourly bulk archive
(`https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station_id}.csv`)
-- plain HTTPS, no auth, no rate limit observed, same operational profile
as LCD. Station ids are `USAF+WBAN` and, unlike LCD's single id per
station, each city maps to an *ordered list* of candidate ids (see
`make_isd_station_map.py`) because NCEI's own inventory both misdates
USAF-id transitions and sometimes files a station-year under the same USAF
paired with a `99999` WBAN placeholder. `fetch_station_year` tries each
candidate in order and falls through past a 404.

QC semantics: each relevant ISD element is a comma-packed
`"value,quality_code"` pair (`TMP`, `DEW`, `SLP`), or a compound field split
further (`MA1`: altimeter value+quality, station pressure value+quality). A
value is treated as missing when the raw value itself is a NOAA sentinel
(9999/99999) *or* when its quality code is in the reject set `{2,3,6,7}`
(NCEI's "suspect"/"erroneous" outcomes, both the standard and
from-an-NCEI-source variants) -- see `_parse_isd_field`.

ISD timestamps are UTC; LCD's (and this pipeline's daily-aggregation
convention) were station local standard time. This module approximates
local standard time by shifting each station's UTC timestamps by
`round(longitude / 15)` hours, so day boundaries -- and therefore daily
max/avg values -- stay consistent with the currently loaded dataset rather
than introducing a second, independent shift.

Gap semantics mirror `lcd.py`'s exactly: each station-year fetch outcome is
one of (a) data present -> included, (b) every candidate id 404s for that
year -> a permanent, legitimate absence (not a gap; the year is still
written, with that city simply missing from it), or (c) retries exhausted
on a non-404 response -> a gap for that specific year only (excluded from
the parquet write so `--resume-local` retries just it). A transient failure
on one candidate does not fall through to the next candidate id -- a
different physical station-year id could otherwise produce a partial,
inconsistent year -- so retry exhaustion on the first non-404-capable
candidate ends the attempt for that station-year.

Reuses `wetbulb.wetbulb_davies_jones` (via `nldas._compute_daily_wetbulb`)
for daily aggregation (>=20/24 hourly values, daily max + mean, 0.1 C
rounding) and `lcd.py`'s dewpoint->specific-humidity conversion plus
pressure-derivation fallback chain, so results stay numerically comparable
to the LCD-derived rows this source replaces. `lcd.py`/`giovanni.py`/
`nldas.py` remain selectable fallbacks (`pipeline.py --wetbulb-source
lcd`/`giovanni`/`granules`).
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
from typing import Any, cast

import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

import nldas
from lcd import (
    _HYPSOMETRIC_SCALE_M_PER_K,
    _ICAO_EXPONENT,
    _ICAO_LAPSE_K_PER_M,
    _ICAO_SEA_LEVEL_T_K,
    _KELVIN_OFFSET,
    _dewpoint_to_specific_humidity,
)
from partition_io import pending_years, write_pending_year_batches
from shards import resolve_filesystem

pd = cast("Any", importlib.import_module("pandas"))
np = cast("Any", importlib.import_module("numpy"))

type DataFrame = Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

ISD_URL_TEMPLATE = (
    "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station_id}.csv"
)
ISD_REQUEST_TIMEOUT_SECONDS = 60
ISD_MAX_RETRIES = 3
ISD_RETRY_DELAY_SECONDS = 5
ISD_DEFAULT_CONCURRENCY = 8

STATION_MAP_PATH = "cities_isd_stations.csv"
_STATION_MAP_WARNED = False

_HOURLY_FRAME_COLUMNS = ("time", "tair_c", "dewpoint_c", "pressure_hpa")

# Sub-daily report types that carry hourly-cadence temperature/dewpoint
# observations -- same vocabulary as lcd.py, since ISD is that product's
# source dataset. See lcd.py's HOURLY_REPORT_TYPES for the rationale.
HOURLY_REPORT_TYPES: tuple[str, ...] = ("FM-15", "FM-16", "FM-12")

# ISD quality codes (the second half of each comma-packed element) that
# mark a present value as failing NCEI's automated/manual QC -- reject
# these alongside the NOAA "missing" sentinel. 1/5 = passed; 9 accompanies
# supplementary fields without a definitive check (not a missing marker by
# itself -- the sentinel already covers true absence); 2/3/6/7 =
# suspect/erroneous (standard and NCEI-source variants).
_ISD_REJECT_QC_CODES = frozenset({"2", "3", "6", "7"})
_ISD_MISSING_TMP_DEW = "9999"
_ISD_MISSING_PRESSURE = "99999"


def _empty_hourly_frame() -> DataFrame:
    return pd.DataFrame(columns=list(_HOURLY_FRAME_COLUMNS))


def _column_or_missing(df: DataFrame, name: str) -> DataFrame:
    """Return `df[name]`, or an all-missing string Series if the column is absent.

    NCEI's ISD CSV writer omits a column entirely (not just blank cells)
    when a station-year file has no non-null values for it, so a fixed
    `usecols` read can raise -- verified empirically on a station-year
    whose only report types were daily/monthly summaries with no `MA1`
    pressure element at all.
    """
    if name in df.columns:
        return df[name]
    return pd.Series([None] * len(df), index=df.index, dtype="object")


def _parse_isd_field(raw: DataFrame, *, missing: str) -> DataFrame:
    """Split a comma-packed `"value,quality_code"` ISD element into a scaled float.

    Returns NaN wherever the raw value equals NOAA's missing-data sentinel
    or the quality code is in `_ISD_REJECT_QC_CODES`. ISD packs these
    elements at 10x their physical unit (tenths of C or hPa). The sentinel
    comparison is numeric (not string) because TMP/DEW sentinels carry a
    sign prefix (`"+9999"`) that a literal string match would miss.
    """
    parts = raw.str.split(",")
    value_str = parts.str[0]
    qc = parts.str[1]
    value_raw = pd.to_numeric(value_str, errors="coerce")
    is_bad = (
        value_raw.isna()
        | (value_raw.abs() == float(missing))
        | qc.isin(_ISD_REJECT_QC_CODES)
    )
    return (value_raw / 10.0).where(~is_bad)


def _get_with_retries(
    session: requests.Session,
    url: str,
    *,
    station_id: str,
    year: int,
) -> requests.Response | None:
    """GET with retry/backoff; a 404 is returned as-is (not retried)."""
    for attempt in range(1, ISD_MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=ISD_REQUEST_TIMEOUT_SECONDS)
            if response.status_code == requests.codes.not_found:
                return response
            response.raise_for_status()
        except requests.RequestException:
            if attempt == ISD_MAX_RETRIES:
                LOGGER.warning(
                    "Giving up on station=%s year=%d after %d attempt(s).",
                    station_id,
                    year,
                    attempt,
                )
                return None
            jitter = random.uniform(0.0, 2.0)  # noqa: S311
            time.sleep(ISD_RETRY_DELAY_SECONDS * attempt + jitter)
        else:
            return response
    return None


def _parse_isd_response(response_text: str, *, lon: float | None) -> DataFrame:
    """Parse one candidate id's ISD station-year CSV into the hourly frame.

    Returns an empty frame (not an error) when the file has zero rows of a
    report type in `HOURLY_REPORT_TYPES` -- some station-years exist but
    carry only daily/monthly summaries, the same "file exists but has no
    hourly data" case `lcd.py`/`make_isd_station_map.py` guard against for
    LCD. The caller falls through to the next candidate id in that case.
    """
    raw = pd.read_csv(io.StringIO(response_text), dtype=str, low_memory=False)
    raw["REPORT_TYPE"] = _column_or_missing(raw, "REPORT_TYPE").str.strip()
    hourly = raw[raw["REPORT_TYPE"].isin(HOURLY_REPORT_TYPES)].copy()
    if hourly.empty:
        return _empty_hourly_frame()

    hourly["ELEVATION"] = pd.to_numeric(
        _column_or_missing(hourly, "ELEVATION"), errors="coerce"
    )
    hourly["tair_c"] = _parse_isd_field(
        _column_or_missing(hourly, "TMP"), missing=_ISD_MISSING_TMP_DEW
    )
    hourly["dewpoint_c"] = _parse_isd_field(
        _column_or_missing(hourly, "DEW"), missing=_ISD_MISSING_TMP_DEW
    )
    hourly["slp_hpa"] = _parse_isd_field(
        _column_or_missing(hourly, "SLP"), missing=_ISD_MISSING_PRESSURE
    )

    ma1 = _column_or_missing(hourly, "MA1").str.split(",")
    altimeter_value_str = ma1.str[0]
    altimeter_qc = ma1.str[1]
    station_pressure_str = ma1.str[2]
    station_pressure_qc = ma1.str[3]
    altimeter_value = pd.to_numeric(altimeter_value_str, errors="coerce") / 10.0
    altimeter_bad = (
        altimeter_value_str.isna()
        | (altimeter_value_str == _ISD_MISSING_PRESSURE)
        | altimeter_qc.isin(_ISD_REJECT_QC_CODES)
    )
    hourly["altimeter_hpa"] = altimeter_value.where(~altimeter_bad)
    station_pressure_value = pd.to_numeric(station_pressure_str, errors="coerce") / 10.0
    station_pressure_bad = (
        station_pressure_str.isna()
        | (station_pressure_str == _ISD_MISSING_PRESSURE)
        | station_pressure_qc.isin(_ISD_REJECT_QC_CODES)
    )
    hourly["station_pressure_hpa"] = station_pressure_value.where(~station_pressure_bad)

    hourly["time_utc"] = pd.to_datetime(hourly["DATE"], errors="coerce")
    hourly = hourly.dropna(subset=["time_utc", "tair_c", "dewpoint_c"])
    if hourly.empty:
        return _empty_hourly_frame()

    # Prefer the routine hourly report when a timestamp has more than one
    # report type (FM-15 over FM-16/FM-12), same tie-break as lcd.py.
    report_priority = {"FM-15": 0, "FM-16": 1, "FM-12": 2}
    hourly["_priority"] = hourly["REPORT_TYPE"].map(report_priority)
    hourly = hourly.sort_values(["time_utc", "_priority"]).drop_duplicates(
        subset="time_utc",
        keep="first",
    )

    pressure_hpa = hourly["station_pressure_hpa"]
    sea_level_derived = hourly["slp_hpa"] * np.exp(
        -hourly["ELEVATION"]
        / (_HYPSOMETRIC_SCALE_M_PER_K * (hourly["tair_c"] + _KELVIN_OFFSET)),
    )
    altimeter_derived = (
        hourly["altimeter_hpa"]
        * (
            (_ICAO_SEA_LEVEL_T_K - _ICAO_LAPSE_K_PER_M * hourly["ELEVATION"])
            / _ICAO_SEA_LEVEL_T_K
        )
        ** _ICAO_EXPONENT
    )
    pressure_hpa = pressure_hpa.fillna(sea_level_derived).fillna(altimeter_derived)

    station_lon = float(lon) if lon is not None else None
    if station_lon is None:
        observed_lon = pd.to_numeric(
            _column_or_missing(hourly, "LONGITUDE"), errors="coerce"
        ).dropna()
        station_lon = float(observed_lon.iloc[0]) if not observed_lon.empty else 0.0
    # station_lon is a plain Python float here (not a numpy scalar), so
    # round() returns a plain int -- required by pd.to_timedelta below,
    # which warns on numpy "generic" integer units otherwise.
    utc_offset_hours = round(station_lon / 15.0)
    local_time = hourly["time_utc"] + pd.to_timedelta(utc_offset_hours, unit="h")

    return pd.DataFrame(
        {
            "time": local_time.to_numpy(),
            "tair_c": hourly["tair_c"].to_numpy(dtype="float64"),
            "dewpoint_c": hourly["dewpoint_c"].to_numpy(dtype="float64"),
            "pressure_hpa": pressure_hpa.to_numpy(dtype="float64"),
        },
    ).reset_index(drop=True)


def fetch_station_year(
    candidate_ids: list[str],
    year: int,
    *,
    lon: float | None = None,
    session: requests.Session | None = None,
) -> tuple[DataFrame, bool]:
    """Return (hourly dry-bulb/dewpoint/pressure frame, gap) for one station-year.

    Tries each id in `candidate_ids` in order. A 404 or a response that
    parses to zero hourly rows (see `_parse_isd_response`) falls through to
    the next candidate. `gap` is True only when retries were exhausted on a
    non-404 response for whichever candidate hit that failure (a transient
    failure ends the attempt for this station-year rather than falling
    through to a different physical id); if every candidate either 404s or
    has no hourly data, that's a legitimate permanent absence and returns
    `(empty_frame, False)`.
    """
    http = session or requests.Session()
    for station_id in candidate_ids:
        url = ISD_URL_TEMPLATE.format(year=year, station_id=station_id)
        response = _get_with_retries(http, url, station_id=station_id, year=year)
        if response is None:
            return _empty_hourly_frame(), True
        if response.status_code == requests.codes.not_found:
            continue
        frame = _parse_isd_response(response.text, lon=lon)
        if not frame.empty:
            return frame, False
        LOGGER.info(
            "Candidate %s year=%d exists but has no hourly rows; trying "
            "the next candidate.",
            station_id,
            year,
        )

    LOGGER.info(
        "No ISD file with hourly rows among %d candidate(s) for year=%d.",
        len(candidate_ids),
        year,
    )
    return _empty_hourly_frame(), False


def _fetch_station_series(
    session: requests.Session,
    candidate_ids: list[str],
    years: list[int],
    lon: float | None,
) -> tuple[DataFrame, set[int]]:
    """Fetch every pending year for one station; return (frame, gapped years)."""
    year_frames: list[DataFrame] = []
    gapped_years: set[int] = set()
    for year in years:
        frame, gap = fetch_station_year(candidate_ids, year, lon=lon, session=session)
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
    """Load the city -> ISD candidate-id crosswalk, if available.

    `cities_isd_stations.csv` (generated by `make_isd_station_map.py`) maps
    each city to its `|`-separated ordered ISD station-id candidate list.
    """
    global _STATION_MAP_WARNED  # noqa: PLW0603
    path = Path(STATION_MAP_PATH)
    if not path.exists():
        if not _STATION_MAP_WARNED:
            LOGGER.warning(
                "%s not found; no cities can be mapped to an ISD station. "
                "Run make_isd_station_map.py to generate it.",
                STATION_MAP_PATH,
            )
            _STATION_MAP_WARNED = True
        return pd.DataFrame(
            {
                "location_id": pd.Series(dtype="int64"),
                "isd_ids": pd.Series(dtype="object"),
                "lon": pd.Series(dtype="float64"),
            },
        )
    return pd.read_csv(path, usecols=["location_id", "isd_ids", "lon"])


def _fetch_stations_batch(
    station_keys: list[str],
    lon_by_key: dict[str, float | None],
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
                station_key.split("|"),
                years,
                lon_by_key[station_key],
            ): station_key
            for station_key in station_keys
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"ISD->wetbulb city_shard {city_shard_index}",
        ):
            results[futures[future]] = future.result()
    return results


def process_isd(
    start_year: int,
    end_year: int,
    out_dir: str,
    city_shard_index: int,
    city_shard_count: int,
    concurrency: int,
    *,
    force: bool = False,
) -> None:
    """Fetch NOAA ISD station data, compute daily wet-bulb, and save as parquet shards."""
    wetbulb_root = f"{out_dir}/wetbulb_data_csv"

    shard_df = nldas._load_nldas_city_shard(city_shard_index, city_shard_count)  # noqa: SLF001
    if shard_df.empty:
        LOGGER.info(
            "No cities found for shard %s/%s.",
            city_shard_index,
            city_shard_count,
        )
        return

    filesystem, base_path = resolve_filesystem(wetbulb_root)
    pending_year_list = pending_years(
        range(start_year, end_year + 1),
        wetbulb_root,
        city_shard_index,
        filesystem,
        base_path,
        file_prefix="wetbulb",
        force=force,
    )
    if not pending_year_list:
        LOGGER.info(
            "city_shard=%d/%d: years %d-%d already present.",
            city_shard_index,
            city_shard_count,
            start_year,
            end_year,
        )
        return

    station_map = _load_station_map()
    shard_df = shard_df.merge(station_map, on="location_id", how="left")
    unmapped = shard_df[shard_df["isd_ids"].isna()]
    if not unmapped.empty:
        LOGGER.warning(
            "city_shard=%d/%d: %d cit(ies) have no ISD station mapped and will "
            "be skipped: %s",
            city_shard_index,
            city_shard_count,
            len(unmapped),
            ", ".join(str(x) for x in unmapped["location_id"].tolist()),
        )
    shard_df = shard_df.dropna(subset=["isd_ids"])
    if shard_df.empty:
        return

    station_to_locations: dict[str, list[int]] = {}
    lon_by_key: dict[str, float | None] = {}
    for row in shard_df.itertuples():
        station_to_locations.setdefault(row.isd_ids, []).append(row.location_id)
        lon_by_key[row.isd_ids] = row.lon

    LOGGER.info(
        "ISD->wetbulb city_shard=%d/%d: %d city(ies) across %d station(s), "
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
        lon_by_key,
        pending_year_list,
        session,
        worker_count,
        city_shard_index,
    )

    hourly_frames: list[DataFrame] = []
    gapped_years: set[int] = set()
    for station_key, (station_df, gaps) in station_results.items():
        gapped_years |= gaps
        hourly_frames.extend(
            _station_to_hourly(location_id, station_df)
            for location_id in station_to_locations[station_key]
        )

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
        LOGGER.warning(
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

    write_pending_year_batches(
        daily_df,
        writable_years,
        wetbulb_root,
        city_shard_index,
        filesystem,
        base_path,
        file_prefix="wetbulb",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=nldas.NLDAS_START_YEAR)
    parser.add_argument("--end-year", type=int, default=nldas.NLDAS_END_YEAR)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--city-shard-index", type=int, default=0)
    parser.add_argument("--city-shard-count", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=ISD_DEFAULT_CONCURRENCY)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Execute the ISD wet-bulb processing pipeline."""
    load_dotenv(override=False)
    exit_code = 0
    try:
        args = _parse_args()
        process_isd(
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
        LOGGER.warning("ISD processing interrupted by user.")
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
        LOGGER.exception("ISD processing failed.")

    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

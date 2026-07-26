"""Fetch ECCC hourly observations and compute Canadian daily wet-bulb.

The worker consumes ``cities_ca.csv`` and ``cities_ca_eccc_stations.csv``.
ECCC's historical files publish local-standard timestamps; they are kept as
such so every archived day has a stable 24-hour boundary.  Temperature,
dew-point and station pressure quality flags are honoured before the shared
pressure-aware Davies-Jones calculation is applied.
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

from lcd import (
    _HYPSOMETRIC_SCALE_M_PER_K,
    _KELVIN_OFFSET,
    _load_pending_shard,
    _station_to_hourly,
    _write_daily_shard,
    add_common_shard_args,
)
from partition_io import pending_years, write_pending_year_batches
from shards import resolve_filesystem

pd = cast("Any", importlib.import_module("pandas"))
np = cast("Any", importlib.import_module("numpy"))
type DataFrame = Any

LOGGER = logging.getLogger(__name__)
ECCC_BULK_URL = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
ECCC_STATION_MAP = "cities_ca_eccc_stations.csv"
ECCC_DEFAULT_CONCURRENCY = 4
ECCC_TIMEOUT_SECONDS = 60
ECCC_MAX_RETRIES = 3
_RNG = random.SystemRandom()

_ALIASES = {
    "Date/Time (LST)": "time",
    "Date/Time": "time",
    "Temp (°C)": "tair_c",
    "Temp (°C) Flag": "tair_flag",
    "Temp Flag": "tair_flag",
    "Dew Point Temp (°C)": "dewpoint_c",
    "Dew Point Temp (°C) Flag": "dewpoint_flag",
    "Dew Point Temp Flag": "dewpoint_flag",
    "Stn Press (kPa)": "station_pressure_kpa",
    "Stn Press (kPa) Flag": "pressure_flag",
    "Stn Press Flag": "pressure_flag",
    "Sea Level Press (kPa)": "sea_level_pressure_kpa",
    "Elevation (m)": "elevation_m",
}
# M = missing.  ECCC also uses single-letter estimates; those remain valid
# observations, while clearly erroneous/rejected values are discarded.
_REJECT_FLAGS = frozenset({"M", "X"})
_OUTPUT_COLUMNS = ("time", "tair_c", "dewpoint_c", "pressure_hpa")


def _empty_hourly() -> DataFrame:
    return pd.DataFrame(columns=list(_OUTPUT_COLUMNS))


def _flagged_numeric(frame: DataFrame, value: str, flag: str) -> DataFrame:
    values = (
        pd.to_numeric(frame[value], errors="coerce")
        if value in frame
        else pd.Series(np.nan, index=frame.index, dtype="float64")
    )
    if flag not in frame:
        return values
    flags = frame[flag].fillna("").astype(str).str.strip().str.upper()
    return values.where(~flags.isin(_REJECT_FLAGS))


def parse_eccc_hourly(text: str, *, elevation_m: float | None = None) -> DataFrame:
    """Normalize one ECCC monthly CSV into the shared station-hourly schema."""
    if not text.strip():
        return _empty_hourly()
    raw = pd.read_csv(io.StringIO(text.lstrip("\ufeff")), low_memory=False)
    raw = raw.rename(
        columns={key: value for key, value in _ALIASES.items() if key in raw}
    )
    required = {"time", "tair_c", "dewpoint_c"}
    if not required.issubset(raw.columns):
        return _empty_hourly()

    raw["time"] = pd.to_datetime(raw["time"], errors="coerce")
    raw["tair_c"] = _flagged_numeric(raw, "tair_c", "tair_flag")
    raw["dewpoint_c"] = _flagged_numeric(raw, "dewpoint_c", "dewpoint_flag")
    pressure_kpa = _flagged_numeric(raw, "station_pressure_kpa", "pressure_flag")
    if elevation_m is None and "elevation_m" in raw:
        observed = pd.to_numeric(raw["elevation_m"], errors="coerce").dropna()
        elevation_m = float(observed.iloc[0]) if not observed.empty else None
    if "sea_level_pressure_kpa" in raw and elevation_m is not None:
        sea_level_hpa = (
            pd.to_numeric(raw["sea_level_pressure_kpa"], errors="coerce") * 10.0
        )
        derived_hpa = sea_level_hpa * np.exp(
            -float(elevation_m)
            / (_HYPSOMETRIC_SCALE_M_PER_K * (raw["tair_c"] + _KELVIN_OFFSET))
        )
        pressure_kpa = pressure_kpa.fillna(derived_hpa / 10.0)

    result = pd.DataFrame(
        {
            "time": raw["time"],
            "tair_c": raw["tair_c"],
            "dewpoint_c": raw["dewpoint_c"],
            "pressure_hpa": pressure_kpa * 10.0,
        }
    )
    return (
        result.dropna(subset=list(_OUTPUT_COLUMNS))
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )


def download_eccc_month(
    station_id: int,
    year: int,
    month: int,
    *,
    session: requests.Session | None = None,
) -> tuple[str | None, bool]:
    """Return ``(CSV text, transient_gap)`` for one station-month."""
    http = session or requests.Session()
    params = {
        "format": "csv",
        "stationID": station_id,
        "Year": year,
        "Month": month,
        "Day": 1,
        "timeframe": 1,
        "submit": "Download Data",
    }
    for attempt in range(1, ECCC_MAX_RETRIES + 1):
        try:
            response = http.get(
                ECCC_BULK_URL, params=params, timeout=ECCC_TIMEOUT_SECONDS
            )
            if response.status_code == requests.codes.not_found:
                return "", False
            response.raise_for_status()
            return response.content.decode("utf-8-sig", errors="replace"), False
        except requests.RequestException:
            if attempt == ECCC_MAX_RETRIES:
                return None, True
            time.sleep(2**attempt + _RNG.uniform(0, 1))
    return None, True


def fetch_station_year(
    station_ids: list[int],
    year: int,
    *,
    elevation_m: float | None = None,
    session: requests.Session | None = None,
) -> tuple[DataFrame, bool]:
    """Fetch one year, using the first candidate with usable hourly data."""
    http = session or requests.Session()
    for station_id in station_ids:
        frames: list[DataFrame] = []
        for month in range(1, 13):
            text, gap = download_eccc_month(station_id, year, month, session=http)
            if gap:
                return _empty_hourly(), True
            if text:
                frame = parse_eccc_hourly(text, elevation_m=elevation_m)
                if not frame.empty:
                    frames.append(frame)
        if frames:
            return pd.concat(frames, ignore_index=True), False
    return _empty_hourly(), False


def _fetch_station_series(
    station_ids: list[int],
    years: list[int],
    elevation_m: float | None,
    session: requests.Session,
) -> tuple[DataFrame, set[int]]:
    frames: list[DataFrame] = []
    gaps: set[int] = set()
    for year in years:
        frame, gap = fetch_station_year(
            station_ids, year, elevation_m=elevation_m, session=session
        )
        if gap:
            gaps.add(year)
        elif not frame.empty:
            frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True) if frames else _empty_hourly(),
        gaps,
    )


def _load_station_map(path: str) -> DataFrame:
    if not Path(path).exists():
        message = f"{path} not found; run make_eccc_station_map.py first"
        raise FileNotFoundError(message)
    return pd.read_csv(path, usecols=["location_id", "eccc_station_ids", "elevation_m"])


def process_eccc(
    start_year: int,
    end_year: int,
    out_dir: str,
    city_shard_index: int,
    city_shard_count: int,
    concurrency: int,
    *,
    force: bool = False,
    cities_csv: str = "cities_ca.csv",
    station_map_csv: str = ECCC_STATION_MAP,
) -> None:
    """Fetch ECCC observations and write database-compatible daily shards."""
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
        cities_csv=cities_csv,
    )
    if loaded is None:
        return
    shard, years, filesystem, base_path, root = loaded
    shard = shard.merge(
        _load_station_map(station_map_csv), on="location_id", how="left"
    )
    shard = shard.dropna(subset=["eccc_station_ids"])
    if shard.empty:
        return

    station_to_locations: dict[str, list[int]] = {}
    elevation_by_key: dict[str, float | None] = {}
    for row in shard.itertuples():
        station_to_locations.setdefault(row.eccc_station_ids, []).append(
            row.location_id
        )
        elevation_by_key[row.eccc_station_ids] = row.elevation_m

    session = requests.Session()
    results: dict[str, tuple[DataFrame, set[int]]] = {}
    workers = max(1, min(concurrency, len(station_to_locations)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _fetch_station_series,
                [int(value) for value in key.split("|")],
                years,
                elevation_by_key[key],
                session,
            ): key
            for key in station_to_locations
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"ECCC->wetbulb city_shard {city_shard_index}",
        ):
            results[futures[future]] = future.result()

    hourly: list[DataFrame] = []
    gaps: set[int] = set()
    for key, (frame, station_gaps) in results.items():
        gaps |= station_gaps
        hourly.extend(
            _station_to_hourly(location_id, frame)
            for location_id in station_to_locations[key]
        )
    _write_daily_shard(
        hourly,
        city_shard_index,
        city_shard_count,
        years,
        gaps,
        root,
        filesystem,
        base_path,
        logger=LOGGER,
        write_batches=write_pending_year_batches,
        source="eccc",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_shard_args(parser)
    parser.add_argument("--concurrency", type=int, default=ECCC_DEFAULT_CONCURRENCY)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--station-map-csv", default=ECCC_STATION_MAP)
    return parser.parse_args()


def main() -> None:
    """Run the ECCC daily wet-bulb worker."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    load_dotenv(override=False)
    args = _parse_args()
    try:
        process_eccc(
            args.start_year,
            args.end_year,
            args.out_dir,
            args.city_shard_index,
            args.city_shard_count,
            args.concurrency,
            force=args.force,
            cities_csv=args.cities_csv,
            station_map_csv=args.station_map_csv,
        )
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()

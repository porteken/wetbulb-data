"""Fetch NLDAS-2 point time series from the Giovanni Time Series API.

Also computes daily wet-bulb temperature from the fetched series.

This is the successor to `nldas.py`'s per-hour granule downloads. NASA GES
DISC retired the old "Data Rods" point-time-series service and replaced it
with the Giovanni Time Series Service
(`https://api.giovanni.earthdata.nasa.gov/timeseries`), which returns a full
multi-year hourly series for one point/variable in a single request. A live
probe against this endpoint (2026-07) confirmed a 25-year hourly Tair series
for one point returns in ~11s, so the whole 2000-2024 backfill needs only
`len(cities) * 3` requests (~1,500) instead of ~219,000 hourly granule
downloads. Output parquet lands in the same `wetbulb_data_csv/year=YYYY/
wetbulb_batch_*` tree that `nldas.py` writes, so `load.py`/`load_wetbulb.py`
and the pipeline's S3 sync/resume logic need no changes.

`nldas.py` is kept as a fallback (`pipeline.py --wetbulb-source granules`)
in case this newer API regresses; this module intentionally reuses its
daily wet-bulb computation, city sharding, and fill-value handling so the
two paths stay numerically consistent.
"""

from __future__ import annotations

import argparse
import importlib
import io
import logging
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import urllib3.util.connection
from tqdm.auto import tqdm

import nldas
from partition_io import batch_exists, write_batch_partition
from shards import resolve_filesystem

pd = cast("Any", importlib.import_module("pandas"))
np = cast("Any", importlib.import_module("numpy"))
requests = cast("Any", importlib.import_module("requests"))

# Same IPv6 blackhole as hydro1.gesdisc.eosdis.nasa.gov (see nldas.py); set
# explicitly here too rather than relying only on the `nldas` import so this
# module stays correct even if that import path ever changes.
urllib3.util.connection.HAS_IPV6 = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

type DataFrame = Any
type Session = Any
type CityRow = Any

GIOVANNI_TIMESERIES_URL = "https://api.giovanni.earthdata.nasa.gov/timeseries"
GIOVANNI_TOKEN_URL = (
    "https://urs.earthdata.nasa.gov/api/users/find_or_create_token"  # noqa: S105
)

# Verified live against the Giovanni Time Series API (2026-07): these are the
# only `data=` values that returned 200 for NLDAS_FORA0125_H v2.0; several
# plausible alternates (..._2.0_Tair, ..._002_Tair, NLDAS2:...:Tair) all 403'd
# with "Data parameter is invalid".
GIOVANNI_VARIABLE_IDS: dict[str, str] = {
    "Tair": "NLDAS_FORA0125_H_2_0_Tair",
    "Qair": "NLDAS_FORA0125_H_2_0_Qair",
    "PSurf": "NLDAS_FORA0125_H_2_0_PSurf",
}

GIOVANNI_REQUEST_TIMEOUT_SECONDS = 120
GIOVANNI_MAX_RETRIES = 5
GIOVANNI_RETRY_DELAY_SECONDS = 5
DEFAULT_CONCURRENCY = 8

CELL_MAP_PATH = "cities_nldas_cells.csv"
# If more than this fraction of a city's hourly values come back fill/NaN,
# the sampled grid cell is probably water, not the intended land cell; log
# an alarm so it can be added to cities_nldas_cells.csv.
FILL_FRACTION_ALARM_THRESHOLD = 0.5

_CELL_MAP_WARNED = False


class TokenManager:
    """Fetches and caches an Earthdata Login bearer token, refreshing on 401."""

    def __init__(self, username: str, password: str) -> None:
        """Store Basic Auth credentials used to mint EDL tokens on demand."""
        self._username = username
        self._password = password
        self._token: str | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> TokenManager:
        """Build a TokenManager from EARTHDATA_USERNAME/EARTHDATA_PASSWORD."""
        username = os.environ.get("EARTHDATA_USERNAME")
        password = os.environ.get("EARTHDATA_PASSWORD")
        if not username or not password:
            msg = (
                "EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set to use "
                "the Giovanni Time Series API (a free NASA Earthdata account "
                "with the 'NASA GESDISC DATA ARCHIVE' application authorized "
                "is required)."
            )
            raise RuntimeError(msg)
        return cls(username, password)

    def get(self) -> str:
        """Return the cached token, minting one on first use."""
        with self._lock:
            if self._token is None:
                self._token = self._fetch()
            return self._token

    def refresh(self) -> str:
        """Force-fetch a new token, e.g. after a 401."""
        with self._lock:
            self._token = self._fetch()
            return self._token

    def _fetch(self) -> str:
        response = requests.post(
            GIOVANNI_TOKEN_URL,
            auth=(self._username, self._password),
            timeout=GIOVANNI_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        token = response.json().get("access_token")
        if not token:
            msg = "Earthdata token response did not contain access_token."
            raise RuntimeError(msg)
        return cast("str", token)


def _sleep_with_backoff(attempt: int, *, retry_after: str | None = None) -> None:
    if retry_after:
        try:
            delay = float(retry_after)
        except ValueError:
            delay = GIOVANNI_RETRY_DELAY_SECONDS * attempt
    else:
        delay = GIOVANNI_RETRY_DELAY_SECONDS * attempt
    time.sleep(delay + random.uniform(0.0, 2.0))  # noqa: S311


def _get_timeseries_csv(
    session: Session,
    token_manager: TokenManager,
    var_id: str,
    lat: float,
    lon: float,
    start_iso: str,
    end_iso: str,
) -> str | None:
    """Fetch one variable's time series as raw CSV text, or None on failure."""
    params = {
        "data": var_id,
        "location": f"[{lat},{lon}]",
        "time": f"{start_iso}/{end_iso}",
    }
    for attempt in range(1, GIOVANNI_MAX_RETRIES + 1):
        headers = {"Authorization": f"Bearer {token_manager.get()}"}
        try:
            response = session.get(
                GIOVANNI_TIMESERIES_URL,
                params=params,
                headers=headers,
                timeout=GIOVANNI_REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException:
            if attempt == GIOVANNI_MAX_RETRIES:
                LOGGER.warning(
                    "Giving up on %s [%s,%s] %s after %d attempt(s) (network error).",
                    var_id,
                    lat,
                    lon,
                    params["time"],
                    attempt,
                )
                return None
            _sleep_with_backoff(attempt)
            continue

        if response.status_code == requests.codes.unauthorized:
            token_manager.refresh()
            continue
        if (
            response.status_code == requests.codes.too_many_requests
            or response.status_code >= requests.codes.internal_server_error
        ):
            if attempt == GIOVANNI_MAX_RETRIES:
                LOGGER.warning(
                    "Giving up on %s [%s,%s] %s after %d attempt(s) (HTTP %d).",
                    var_id,
                    lat,
                    lon,
                    params["time"],
                    attempt,
                    response.status_code,
                )
                return None
            _sleep_with_backoff(attempt, retry_after=response.headers.get("Retry-After"))
            continue
        if response.status_code >= requests.codes.bad_request:
            LOGGER.warning(
                "Giovanni request failed for %s [%s,%s] %s: HTTP %d %s",
                var_id,
                lat,
                lon,
                params["time"],
                response.status_code,
                response.text[:200],
            )
            return None
        return cast("str", response.text)
    return None


def _parse_timeseries_csv(text: str) -> tuple[dict[str, str], DataFrame]:
    """Split the Giovanni CSV into its `key,value` header block and data rows."""
    lines = text.splitlines()
    headers: dict[str, str] = {}
    i = 0
    while i < len(lines) and lines[i].strip() and not lines[i].startswith("Timestamp"):
        key, _, value = lines[i].partition(",")
        headers[key.strip()] = value.strip()
        i += 1
    while i < len(lines) and (not lines[i].strip() or lines[i].startswith("Timestamp")):
        i += 1

    if "param_short_name" not in headers:
        msg = f"Giovanni response missing expected header fields; got: {text[:300]!r}"
        raise ValueError(msg)

    data_text = "\n".join(lines[i:])
    df = pd.read_csv(
        io.StringIO(data_text),
        header=None,
        names=["time", "value"],
    )
    df["time"] = pd.to_datetime(df["time"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return headers, df


def _empty_series_df() -> DataFrame:
    return pd.DataFrame(
        {
            "time": pd.Series(dtype="datetime64[ns]"),
            "value": pd.Series(dtype="float64"),
        },
    )


def _fetch_variable_series(
    session: Session,
    token_manager: TokenManager,
    var_id: str,
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
) -> DataFrame:
    """Fetch one variable over [start_year, end_year], halving the range on failure."""
    start_iso = f"{start_year}-01-01T00:00:00"
    end_iso = f"{end_year + 1}-01-01T00:00:00"
    text = _get_timeseries_csv(
        session, token_manager, var_id, lat, lon, start_iso, end_iso
    )
    if text is not None:
        try:
            df = _parse_timeseries_csv(text)[1]
        except ValueError:
            LOGGER.exception(
                "Failed to parse Giovanni response for %s [%s,%s] %s/%s.",
                var_id,
                lat,
                lon,
                start_iso,
                end_iso,
            )
        else:
            return df

    if start_year == end_year:
        LOGGER.error(
            "Giving up on %s [%s,%s] year=%d after exhausting retries.",
            var_id,
            lat,
            lon,
            start_year,
        )
        return _empty_series_df()

    mid = (start_year + end_year) // 2
    left = _fetch_variable_series(session, token_manager, var_id, lat, lon, start_year, mid)
    right = _fetch_variable_series(
        session, token_manager, var_id, lat, lon, mid + 1, end_year
    )
    return pd.concat([left, right], ignore_index=True)


def fetch_city_hourly(
    session: Session,
    token_manager: TokenManager,
    location_id: int,
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
) -> DataFrame:
    """Fetch Tair/Qair/PSurf for one point and join them into an hourly frame."""
    series: dict[str, DataFrame] = {}
    for canonical, var_id in GIOVANNI_VARIABLE_IDS.items():
        df = _fetch_variable_series(
            session, token_manager, var_id, lat, lon, start_year, end_year
        )
        series[canonical] = df.set_index("time")["value"].rename(canonical)

    combined = pd.concat(series.values(), axis=1, join="outer")
    for canonical in GIOVANNI_VARIABLE_IDS:
        if canonical not in combined:
            combined[canonical] = np.nan
        combined.loc[
            combined[canonical] <= nldas.NLDAS_FILL_THRESHOLD, canonical
        ] = np.nan

    combined = combined.reset_index().rename(columns={"index": "time"})
    combined.insert(0, "location_id", location_id)

    total = len(combined)
    if total:
        fill_fraction = combined["Tair"].isna().mean()
        if fill_fraction > FILL_FRACTION_ALARM_THRESHOLD:
            LOGGER.error(
                "location_id=%s at (%.4f, %.4f): %.0f%% missing/fill Tair "
                "values; the sampled NLDAS cell is likely water, not land. "
                "Consider adding a snapped cell for it to %s.",
                location_id,
                lat,
                lon,
                fill_fraction * 100,
                CELL_MAP_PATH,
            )
    return combined


def _contiguous_year_ranges(years: list[int]) -> list[tuple[int, int]]:
    """Group a sorted-or-not list of years into contiguous (start, end) runs."""
    ordered = sorted(set(years))
    if not ordered:
        return []
    ranges: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for year in ordered[1:]:
        if year == prev + 1:
            prev = year
            continue
        ranges.append((start, prev))
        start = prev = year
    ranges.append((start, prev))
    return ranges


def _load_cell_map() -> DataFrame:
    """Load pre-snapped land-cell centers for cities, if available.

    `cities_nldas_cells.csv` (generated by `make_nldas_cell_map.py`) maps
    each city to the same land-adjusted NLDAS grid cell that `nldas.py`'s
    `_resolve_valid_indices` would pick, so Giovanni is asked for the same
    cell instead of the nearest raw-coordinate cell, which may be water.
    """
    global _CELL_MAP_WARNED  # noqa: PLW0603
    path = Path(CELL_MAP_PATH)
    if not path.exists():
        if not _CELL_MAP_WARNED:
            LOGGER.warning(
                "%s not found; requesting raw city coordinates instead of "
                "land-snapped NLDAS cells. Run make_nldas_cell_map.py to "
                "generate it. Coastal/water-adjacent cities may come back "
                "mostly fill.",
                CELL_MAP_PATH,
            )
            _CELL_MAP_WARNED = True
        return pd.DataFrame(
            {
                "location_id": pd.Series(dtype="int64"),
                "cell_lat": pd.Series(dtype="float64"),
                "cell_lon": pd.Series(dtype="float64"),
            },
        )
    return pd.read_csv(path, usecols=["location_id", "cell_lat", "cell_lon"])


def process_giovanni(
    start_year: int,
    end_year: int,
    out_dir: str,
    city_shard_index: int,
    city_shard_count: int,
    concurrency: int,
    *,
    force: bool = False,
) -> None:
    """Fetch NLDAS-2 point series via Giovanni, compute daily wet-bulb, and save."""
    wetbulb_root = f"{out_dir}/wetbulb_data_csv"

    shard_df = nldas._load_nldas_city_shard(city_shard_index, city_shard_count)  # noqa: SLF001
    if shard_df.empty:
        LOGGER.info("No cities found for shard %s/%s.", city_shard_index, city_shard_count)
        return

    filesystem, base_path = resolve_filesystem(wetbulb_root)
    years = list(range(start_year, end_year + 1))
    pending_years = [
        year
        for year in years
        if force
        or not batch_exists(
            wetbulb_root,
            year,
            city_shard_index,
            0,
            file_prefix="wetbulb",
            filesystem=filesystem,
            base_path=base_path,
        )
    ]
    if not pending_years:
        LOGGER.info(
            "city_shard=%d/%d: years %d-%d already present.",
            city_shard_index,
            city_shard_count,
            start_year,
            end_year,
        )
        return

    fetch_ranges = _contiguous_year_ranges(pending_years)
    LOGGER.info(
        "Giovanni->wetbulb city_shard=%d/%d: %d city(ies), %d pending year(s) "
        "as %d contiguous range(s).",
        city_shard_index,
        city_shard_count,
        len(shard_df),
        len(pending_years),
        len(fetch_ranges),
    )

    cell_map = _load_cell_map()
    shard_df = shard_df.merge(cell_map, on="location_id", how="left")
    shard_df["fetch_lat"] = shard_df["cell_lat"].fillna(shard_df["lat"])
    shard_df["fetch_lon"] = shard_df["cell_lon"].fillna(shard_df["lng"])

    token_manager = TokenManager.from_env()
    session = requests.Session()

    def _fetch_city(row: CityRow) -> DataFrame:
        frames = [
            fetch_city_hourly(
                session,
                token_manager,
                row.location_id,
                row.fetch_lat,
                row.fetch_lon,
                range_start,
                range_end,
            )
            for range_start, range_end in fetch_ranges
        ]
        return pd.concat(frames, ignore_index=True)

    worker_count = max(1, min(concurrency, len(shard_df)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_fetch_city, row): row.location_id
            for row in shard_df.itertuples()
        }
        hourly_frames = [
            future.result()
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Giovanni->wetbulb city_shard {city_shard_index}",
            )
        ]

    if not hourly_frames:
        return
    hourly_df = pd.concat(hourly_frames, ignore_index=True)
    daily_df = nldas._compute_daily_wetbulb(hourly_df)  # noqa: SLF001
    if daily_df.empty:
        return

    daily_df["year"] = pd.to_datetime(daily_df["date"]).dt.year
    pending_year_set = set(pending_years)
    for year, year_df in daily_df.groupby("year"):
        if year not in pending_year_set:
            continue
        write_batch_partition(
            wetbulb_root,
            int(year),
            city_shard_index,
            year_df.drop(columns="year"),
            0,
            file_prefix="wetbulb",
            filesystem=filesystem,
            base_path=base_path,
        )


def _run_smoke(args: argparse.Namespace) -> None:
    """Issue one authenticated request per variable and print diagnostics."""
    token_manager = TokenManager.from_env()
    session = requests.Session()
    for canonical, var_id in GIOVANNI_VARIABLE_IDS.items():
        text = _get_timeseries_csv(
            session, token_manager, var_id, args.lat, args.lon, args.start, args.end
        )
        if text is None:
            print(f"{canonical} ({var_id}): request failed")  # noqa: T201
            continue
        headers, df = _parse_timeseries_csv(text)
        print(f"--- {canonical} ({var_id}) ---")  # noqa: T201
        print(  # noqa: T201
            f"resolved cell: ({headers.get('lat')}, {headers.get('lon')}) "
            f"unit={headers.get('unit')} fill={headers.get('undef')}",
        )
        print(df.head())  # noqa: T201


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=nldas.NLDAS_START_YEAR)
    parser.add_argument("--end-year", type=int, default=nldas.NLDAS_END_YEAR)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--city-shard-index", type=int, default=0)
    parser.add_argument("--city-shard-count", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Issue one authenticated request per variable and print diagnostics, then exit.",
    )
    parser.add_argument("--lat", type=float, default=40.713)
    parser.add_argument("--lon", type=float, default=-74.006)
    parser.add_argument("--start", type=str, default="2024-01-01T00:00:00")
    parser.add_argument("--end", type=str, default="2024-01-02T00:00:00")
    return parser.parse_args()


def main() -> None:
    """Execute the Giovanni NLDAS-2 wet-bulb processing pipeline."""
    exit_code = 0
    try:
        args = _parse_args()
        if args.smoke:
            _run_smoke(args)
        else:
            process_giovanni(
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
        LOGGER.warning("Giovanni processing interrupted by user.")
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
        LOGGER.exception("Giovanni processing failed.")

    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

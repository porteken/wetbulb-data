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
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol, cast

import urllib3.util.connection
from dotenv import load_dotenv
from tqdm.auto import tqdm

import nldas
from partition_io import pending_years, write_pending_year_batches
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
type Response = Any
type Filesystem = Any

GIOVANNI_TIMESERIES_URL = "https://api.giovanni.earthdata.nasa.gov/timeseries"
GIOVANNI_TOKEN_URL = "https://urs.earthdata.nasa.gov/api/users/find_or_create_token"  # noqa: S105

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
# A full local backfill run (2026-07) drew sustained HTTP 500s from Giovanni
# after several minutes of high concurrency, while isolated requests (even
# 32 concurrent full 26-year ranges in a burst) always succeeded -- pointing
# at a server-side quota tied to sustained request volume rather than raw
# concurrency. These retries lean toward "wait out a rate-limit window"
# rather than "fail fast": exponential backoff capped at
# GIOVANNI_MAX_RETRY_DELAY_SECONDS, more attempts than the earlier linear
# schedule.
GIOVANNI_MAX_RETRIES = 8
GIOVANNI_RETRY_DELAY_SECONDS = 5
GIOVANNI_MAX_RETRY_DELAY_SECONDS = 60
DEFAULT_CONCURRENCY = 8

# A shard's parquet write is skipped entirely if any city has a fetch gap
# (see process_giovanni), so a single stubborn city would otherwise force
# discarding an entire shard's worth of good data. Re-fetching just the
# stragglers a few times, with a pause to let a transient overload window
# pass, converts most single-city failures into eventual successes instead
# of wasted work.
GIOVANNI_SHARD_RETRY_ATTEMPTS = 3
GIOVANNI_SHARD_RETRY_DELAY_SECONDS = 30

CELL_MAP_PATH = "cities_nldas_cells.csv"
# If more than this fraction of a city's hourly values come back fill/NaN,
# the sampled grid cell is probably water, not the intended land cell; log
# an alarm so it can be added to cities_nldas_cells.csv.
FILL_FRACTION_ALARM_THRESHOLD = 0.5

_CELL_MAP_WARNED = False


class TokenManagerProtocol(Protocol):
    """Structural interface for minting/refreshing an EDL bearer token."""

    def get(self) -> str:
        """Return the cached token, minting one on first use."""
        ...

    def refresh(self) -> str:
        """Force-fetch a new token, e.g. after a 401."""
        ...


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
            delay = GIOVANNI_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
    else:
        delay = GIOVANNI_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
    delay = min(delay, GIOVANNI_MAX_RETRY_DELAY_SECONDS)
    time.sleep(delay + random.uniform(0.0, 2.0))  # noqa: S311


class _RangeTooLargeError(RuntimeError):
    """Giovanni rejected the request as too large (HTTP 413).

    Unlike a 5xx/429 (server overloaded, retry the same request later), 413
    means this exact request will never succeed no matter how many times
    it's retried -- the caller needs to split the date range into smaller
    pieces instead. A live backfill (2026-07) showed Qair responses hitting
    this limit at the full 26-year range while Tair/PSurf did not, so the
    threshold is response-size-dependent, not a fixed year count.
    """


class _ResponseOutcome(Enum):
    """What `_get_timeseries_csv`'s retry loop should do with a response."""

    RETRY = auto()
    GIVE_UP = auto()


def _classify_response(
    response: Response,
    attempt: int,
    token_manager: TokenManagerProtocol,
    var_id: str,
    lat: float,
    lon: float,
    time_range: str,
) -> _ResponseOutcome | None:
    """Decide what to do with a non-exception Giovanni response.

    Returns None if `response` is a success the caller should return as-is,
    RETRY if the caller should retry immediately, or GIVE_UP if the caller
    should return None. Raises `_RangeTooLargeError` on HTTP 413 (see that
    class's docstring).
    """
    if response.status_code == requests.codes.unauthorized:
        token_manager.refresh()
        return _ResponseOutcome.RETRY
    if response.status_code == requests.codes.request_entity_too_large:
        LOGGER.info(
            "%s [%s,%s] %s: range too large (HTTP 413); will split it.",
            var_id,
            lat,
            lon,
            time_range,
        )
        raise _RangeTooLargeError(response.text[:200])
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
                time_range,
                attempt,
                response.status_code,
            )
            return _ResponseOutcome.GIVE_UP
        _sleep_with_backoff(attempt, retry_after=response.headers.get("Retry-After"))
        return _ResponseOutcome.RETRY
    if response.status_code >= requests.codes.bad_request:
        LOGGER.warning(
            "Giovanni request failed for %s [%s,%s] %s: HTTP %d %s",
            var_id,
            lat,
            lon,
            time_range,
            response.status_code,
            response.text[:200],
        )
        return _ResponseOutcome.GIVE_UP
    return None


def _get_timeseries_csv(
    session: Session,
    token_manager: TokenManagerProtocol,
    var_id: str,
    lat: float,
    lon: float,
    start_iso: str,
    end_iso: str,
) -> str | None:
    """Fetch one variable's time series as raw CSV text, or None on failure.

    Raises `_RangeTooLargeError` on HTTP 413 (see that class's docstring).
    """
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

        outcome = _classify_response(
            response, attempt, token_manager, var_id, lat, lon, params["time"]
        )
        if outcome is _ResponseOutcome.RETRY:
            continue
        if outcome is _ResponseOutcome.GIVE_UP:
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
    token_manager: TokenManagerProtocol,
    var_id: str,
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
) -> tuple[DataFrame, bool]:
    """Fetch one variable over [start_year, end_year].

    Returns (df, had_gap). Three failure modes are handled differently:

    - HTTP 413 (`_RangeTooLargeError`): a deterministic "this exact request
      will never fit" signal (seen live for Qair at the full 26-year range
      while Tair/PSurf fit fine), so this always halves and retries the
      smaller pieces.
    - Any other HTTP/network failure (`_get_timeseries_csv` returns None
      after exhausting its own retries with backoff): the server is
      already struggling, so this gives up on the whole requested range at
      once instead of halving into more requests -- halving here just
      resubmits the same total demand as more requests, which turned one
      slow window into a ~50-minute outage during a real backfill
      (2026-07).
    - A malformed/unparseable response body: ambiguous cause, so this
      halves as a diagnostic fallback (existing behavior).
    """
    start_iso = f"{start_year}-01-01T00:00:00"
    end_iso = f"{end_year + 1}-01-01T00:00:00"
    try:
        text = _get_timeseries_csv(
            session, token_manager, var_id, lat, lon, start_iso, end_iso
        )
    except _RangeTooLargeError:
        text = None
        range_too_large = True
    else:
        range_too_large = False

    if text is None and not range_too_large:
        LOGGER.error(
            "Giving up on %s [%s,%s] %d-%d after exhausting retries.",
            var_id,
            lat,
            lon,
            start_year,
            end_year,
        )
        return _empty_series_df(), True

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
            return df, False

    if start_year == end_year:
        LOGGER.error(
            "Giving up on %s [%s,%s] year=%d: %s.",
            var_id,
            lat,
            lon,
            start_year,
            "range too large even at one year"
            if range_too_large
            else "response could not be parsed",
        )
        return _empty_series_df(), True

    mid = (start_year + end_year) // 2
    left, left_gap = _fetch_variable_series(
        session, token_manager, var_id, lat, lon, start_year, mid
    )
    right, right_gap = _fetch_variable_series(
        session, token_manager, var_id, lat, lon, mid + 1, end_year
    )
    return pd.concat([left, right], ignore_index=True), left_gap or right_gap


def fetch_city_hourly(
    session: Session,
    token_manager: TokenManagerProtocol,
    location_id: int,
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
) -> tuple[DataFrame, bool]:
    """Fetch Tair/Qair/PSurf for one point and join them into an hourly frame.

    Returns (df, had_gap); had_gap is True if any variable in [start_year,
    end_year] could not be fetched (see `_fetch_variable_series`).
    """
    series: dict[str, DataFrame] = {}
    had_gap = False
    for canonical, var_id in GIOVANNI_VARIABLE_IDS.items():
        df, gap = _fetch_variable_series(
            session, token_manager, var_id, lat, lon, start_year, end_year
        )
        had_gap = had_gap or gap
        series[canonical] = df.set_index("time")["value"].rename(canonical)

    combined = pd.concat(series.values(), axis=1, join="outer")
    for canonical in GIOVANNI_VARIABLE_IDS:
        if canonical not in combined:
            combined[canonical] = np.nan
        combined.loc[combined[canonical] <= nldas.NLDAS_FILL_THRESHOLD, canonical] = (
            np.nan
        )

    combined = combined.reset_index().rename(columns={"index": "time"})
    combined.insert(0, "location_id", location_id)

    total = len(combined)
    if total:
        fill_fraction = combined["Tair"].isna().mean()
        if fill_fraction > FILL_FRACTION_ALARM_THRESHOLD:
            if had_gap:
                LOGGER.error(
                    "location_id=%s at (%.4f, %.4f): %.0f%% missing/fill "
                    "Tair values, but a Giovanni request failed for this "
                    "point -- likely an API gap, not a water cell.",
                    location_id,
                    lat,
                    lon,
                    fill_fraction * 100,
                )
            else:
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
    return combined, had_gap


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


def _fetch_city_series(
    session: Session,
    token_manager: TokenManagerProtocol,
    row: CityRow,
    fetch_ranges: list[tuple[int, int]],
) -> tuple[DataFrame, bool]:
    """Fetch one city's hourly data across all pending contiguous year ranges."""
    frames: list[DataFrame] = []
    city_had_gap = False
    for range_start, range_end in fetch_ranges:
        frame, gap = fetch_city_hourly(
            session,
            token_manager,
            row.location_id,
            row.fetch_lat,
            row.fetch_lon,
            range_start,
            range_end,
        )
        frames.append(frame)
        city_had_gap = city_had_gap or gap
    return pd.concat(frames, ignore_index=True), city_had_gap


def _fetch_cities_batch(
    rows: list[CityRow],
    session: Session,
    token_manager: TokenManagerProtocol,
    fetch_ranges: list[tuple[int, int]],
    worker_count: int,
    city_shard_index: int,
) -> dict[int, tuple[DataFrame, bool]]:
    """Fetch a batch of cities concurrently, `worker_count` at a time."""
    results: dict[int, tuple[DataFrame, bool]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _fetch_city_series, session, token_manager, row, fetch_ranges
            ): row.location_id
            for row in rows
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Giovanni->wetbulb city_shard {city_shard_index}",
        ):
            results[futures[future]] = future.result()
    return results


def _fetch_shard_with_retries(
    rows_by_location: dict[int, CityRow],
    session: Session,
    token_manager: TokenManagerProtocol,
    fetch_ranges: list[tuple[int, int]],
    worker_count: int,
    city_shard_index: int,
    city_shard_count: int,
) -> dict[int, tuple[DataFrame, bool]]:
    """Fetch every city in the shard, retrying just the ones with a fetch gap."""
    pending_rows = list(rows_by_location.values())
    city_results: dict[int, tuple[DataFrame, bool]] = {}
    for attempt in range(1, GIOVANNI_SHARD_RETRY_ATTEMPTS + 1):
        city_results.update(
            _fetch_cities_batch(
                pending_rows,
                session,
                token_manager,
                fetch_ranges,
                worker_count,
                city_shard_index,
            )
        )
        pending_rows = [
            rows_by_location[location_id]
            for location_id, (_, gap) in city_results.items()
            if gap
        ]
        if not pending_rows or attempt == GIOVANNI_SHARD_RETRY_ATTEMPTS:
            break
        LOGGER.warning(
            "city_shard=%d/%d: %d/%d city(ies) still had a fetch gap after "
            "attempt %d/%d; retrying just those after a %ds pause.",
            city_shard_index,
            city_shard_count,
            len(pending_rows),
            len(rows_by_location),
            attempt,
            GIOVANNI_SHARD_RETRY_ATTEMPTS,
            GIOVANNI_SHARD_RETRY_DELAY_SECONDS,
        )
        time.sleep(GIOVANNI_SHARD_RETRY_DELAY_SECONDS)
    return city_results


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
        LOGGER.info(
            "No cities found for shard %s/%s.", city_shard_index, city_shard_count
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

    fetch_ranges = _contiguous_year_ranges(pending_year_list)
    LOGGER.info(
        "Giovanni->wetbulb city_shard=%d/%d: %d city(ies), %d pending year(s) "
        "as %d contiguous range(s).",
        city_shard_index,
        city_shard_count,
        len(shard_df),
        len(pending_year_list),
        len(fetch_ranges),
    )

    cell_map = _load_cell_map()
    shard_df = shard_df.merge(cell_map, on="location_id", how="left")
    shard_df["fetch_lat"] = shard_df["cell_lat"].fillna(shard_df["lat"])
    shard_df["fetch_lon"] = shard_df["cell_lon"].fillna(shard_df["lng"])

    token_manager = TokenManager.from_env()
    session = requests.Session()
    worker_count = max(1, min(concurrency, len(shard_df)))
    rows_by_location = {row.location_id: row for row in shard_df.itertuples()}
    city_results = _fetch_shard_with_retries(
        rows_by_location,
        session,
        token_manager,
        fetch_ranges,
        worker_count,
        city_shard_index,
        city_shard_count,
    )

    hourly_frames = [df for df, _ in city_results.values()]
    shard_had_gap = any(gap for _, gap in city_results.values())

    if not hourly_frames:
        return
    hourly_df = pd.concat(hourly_frames, ignore_index=True)
    daily_df = nldas._compute_daily_wetbulb(hourly_df)  # noqa: SLF001
    if daily_df.empty:
        return

    if shard_had_gap:
        LOGGER.warning(
            "city_shard=%d/%d: one or more cities had a Giovanni fetch gap "
            "somewhere in %d-%d (see 'Giving up' errors above); skipping "
            "the parquet write for these pending year(s) so a future run "
            "(e.g. with --resume-local, once the API recovers) retries "
            "them instead of treating incomplete data as done.",
            city_shard_index,
            city_shard_count,
            start_year,
            end_year,
        )
        return

    write_pending_year_batches(
        daily_df,
        pending_year_list,
        wetbulb_root,
        city_shard_index,
        filesystem,
        base_path,
        file_prefix="wetbulb",
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
    load_dotenv(override=False)
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

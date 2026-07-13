"""Download NLDAS-2 hourly forcing data and compute daily wet-bulb temperature.

Fetches `NLDAS_FORA0125_H` v2.0 hourly netCDF granules directly over HTTPS
from NASA GES DISC (deterministic URLs, no CMR search needed), extracts the
nearest 0.125-degree grid cell for each city, and computes wet-bulb
temperature with the Davies-Jones (2008) method (see `wetbulb.py`). Output
parquet lands in the same `wetbulb_data_csv/year=YYYY/wetbulb_batch_*`
tree that `google_era5.py` used to write, so `load.py`/`load_wetbulb.py`
and the pipeline's S3 sync/resume logic need no changes.
"""

from __future__ import annotations

import argparse
import importlib
import io
import logging
import multiprocessing
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any, cast

from tqdm.auto import tqdm

from partition_io import batch_exists, write_batch_partition
from shards import resolve_filesystem
from wetbulb import wetbulb_davies_jones

pd = cast("Any", importlib.import_module("pandas"))
np = cast("Any", importlib.import_module("numpy"))
requests = cast("Any", importlib.import_module("requests"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

type DataFrame = Any
type Dataset = Any
type ArrayLike = Any
type Session = Any
type HttpMessage = Any

NLDAS_BASE_URL = (
    "https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.2.0"
)
NLDAS_GRID_LAT0 = 25.0625
NLDAS_GRID_LON0 = -124.9375
NLDAS_GRID_STEP = 0.125
NLDAS_GRID_NLAT = 224
NLDAS_GRID_NLON = 464
NLDAS_FILL_THRESHOLD = -9000.0
NLDAS_LAND_SEARCH_RADIUS = 2

# Candidate names cover both the netCDF short names documented for
# NLDAS_FORA0125_H v2.0 and the GRIB-derived short names some conversions
# retain; verify against a real downloaded granule before first production
# use and trim/extend this list if needed.
NLDAS_VARIABLE_CANDIDATES: dict[str, list[str]] = {
    "Tair": ["Tair", "Tair_f_inst", "TMP"],
    "Qair": ["Qair", "Qair_f_inst", "SPFH"],
    "PSurf": ["PSurf", "PSurf_f_inst", "PRES"],
}
NLDAS_VARIABLES = tuple(NLDAS_VARIABLE_CANDIDATES)

EARTHDATA_AUTH_HOST = "urs.earthdata.nasa.gov"

DEFAULT_BATCH_HOURS = 24 * 30
NLDAS_MAX_RETRIES = 3
NLDAS_RETRY_DELAY_SECONDS = 10
NLDAS_REQUEST_TIMEOUT_SECONDS = 60
NLDAS_STABLE_DATA_LATENCY_DAYS = 5  # NLDAS-2 near-real-time latency is ~4 days

WETBULB_ROUNDING_FACTOR = 2.0  # round to nearest 0.5 C, matching the ERA5 max product
WETBULB_AVG_ROUNDING_FACTOR = 10.0  # round to nearest 0.1 C
MIN_REASONABLE_WETBULB_C = -50.0
MAX_REASONABLE_WETBULB_C = 40.0
MIN_DAILY_HOURS = 20  # require >=20/24 hours before trusting a daily average

NLDAS_THREAD_LOCAL = threading.local()


def _arco_style_end_year() -> int:
    now = datetime.now(tz=UTC)
    if now.month == 1 and now.day <= NLDAS_STABLE_DATA_LATENCY_DAYS:
        return now.year - 2
    return now.year - 1


NLDAS_START_YEAR = 2000
NLDAS_END_YEAR = _arco_style_end_year()


class EarthdataSession(requests.Session):
    """`requests.Session` that keeps Basic Auth across the URS redirect chain.

    NASA Earthdata Login redirects an unauthenticated request to
    urs.earthdata.nasa.gov and back to the data host; `requests` strips the
    Authorization header on any redirect that changes host, which breaks
    that round trip. This is the standard NASA-documented workaround.
    """

    def __init__(self, username: str, password: str) -> None:
        """Store Basic Auth credentials to reapply across URS redirects."""
        super().__init__()
        self.auth = (username, password)

    def rebuild_auth(
        self, prepared_request: HttpMessage, response: HttpMessage
    ) -> None:
        """Keep the Authorization header only when redirecting to/from URS."""
        headers = prepared_request.headers
        if "Authorization" not in headers:
            return
        original_host = requests.utils.urlparse(response.request.url).hostname
        redirect_host = requests.utils.urlparse(prepared_request.url).hostname
        auth_hosts = {EARTHDATA_AUTH_HOST}
        if original_host != redirect_host and not auth_hosts & {
            original_host,
            redirect_host,
        }:
            del headers["Authorization"]


def _build_earthdata_session() -> Session:
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if username and password:
        return EarthdataSession(username, password)
    LOGGER.info(
        "EARTHDATA_USERNAME/EARTHDATA_PASSWORD not set; relying on ~/.netrc "
        "(machine %s) for Earthdata Login. A free account with the 'NASA "
        "GESDISC DATA ARCHIVE' application authorized is required either way.",
        EARTHDATA_AUTH_HOST,
    )
    return requests.Session()


def _thread_session() -> Session:
    session = getattr(NLDAS_THREAD_LOCAL, "session", None)
    if session is None:
        session = NLDAS_THREAD_LOCAL.session = _build_earthdata_session()
    return session


def granule_url(timestamp: object) -> str:
    """Return the deterministic GES DISC URL for one hourly NLDAS-2 granule."""
    ts = pd.Timestamp(timestamp)
    return (
        f"{NLDAS_BASE_URL}/{ts.year:04d}/{ts.dayofyear:03d}/"
        f"NLDAS_FORA0125_H.A{ts.year:04d}{ts.month:02d}{ts.day:02d}"
        f".{ts.hour:02d}00.020.nc"
    )


def _nearest_grid_indices(
    lats: ArrayLike, lons: ArrayLike
) -> tuple[ArrayLike, ArrayLike]:
    iy = np.clip(
        np.round(
            (np.asarray(lats, dtype="float64") - NLDAS_GRID_LAT0) / NLDAS_GRID_STEP,
        ).astype(int),
        0,
        NLDAS_GRID_NLAT - 1,
    )
    ix = np.clip(
        np.round(
            (np.asarray(lons, dtype="float64") - NLDAS_GRID_LON0) / NLDAS_GRID_STEP,
        ).astype(int),
        0,
        NLDAS_GRID_NLON - 1,
    )
    return iy, ix


def _resolve_valid_indices(
    iy: ArrayLike,
    ix: ArrayLike,
    lats: ArrayLike,
    lons: ArrayLike,
    reference_grid: ArrayLike,
    *,
    fill_threshold: float = NLDAS_FILL_THRESHOLD,
    max_radius: int = NLDAS_LAND_SEARCH_RADIUS,
) -> tuple[ArrayLike, ArrayLike]:
    """Snap any location whose nearest cell is fill/water to the nearest land cell.

    `reference_grid` is a 2D (lat, lon) field (e.g. Tair) from one granule,
    used only to detect fill vs. land cells; the NLDAS-2 land mask is static
    across the archive so one granule is representative for the whole run.
    Locations with no valid cell within `max_radius` get index -1 (sentinel
    for "no data"), which downstream extraction turns into NaN.
    """
    resolved_iy = np.array(iy, dtype="int64", copy=True)
    resolved_ix = np.array(ix, dtype="int64", copy=True)
    n_lat, n_lon = reference_grid.shape

    for i in range(len(iy)):
        if reference_grid[iy[i], ix[i]] > fill_threshold:
            continue

        best_cell: tuple[int, int] | None = None
        for radius in range(1, max_radius + 1):
            candidates = [
                (iy[i] + dy, ix[i] + dx)
                for dy in range(-radius, radius + 1)
                for dx in range(-radius, radius + 1)
                if max(abs(dy), abs(dx)) == radius
            ]
            candidates = [
                (cy, cx)
                for cy, cx in candidates
                if 0 <= cy < n_lat
                and 0 <= cx < n_lon
                and reference_grid[cy, cx] > fill_threshold
            ]
            if not candidates:
                continue
            cell_lats = (
                NLDAS_GRID_LAT0 + np.array([c[0] for c in candidates]) * NLDAS_GRID_STEP
            )
            cell_lons = (
                NLDAS_GRID_LON0 + np.array([c[1] for c in candidates]) * NLDAS_GRID_STEP
            )
            dists = (cell_lats - lats[i]) ** 2 + (cell_lons - lons[i]) ** 2
            best_cell = candidates[int(np.argmin(dists))]
            break

        if best_cell is None:
            LOGGER.error(
                "No valid NLDAS land cell within radius %d of index (%d, %d) "
                "for location at (%.4f, %.4f); values will be NaN.",
                max_radius,
                iy[i],
                ix[i],
                lats[i],
                lons[i],
            )
            resolved_iy[i], resolved_ix[i] = -1, -1
            continue

        resolved_iy[i], resolved_ix[i] = best_cell

    return resolved_iy, resolved_ix


def _resolve_variable_name(ds: Dataset, canonical: str) -> str:
    candidates = NLDAS_VARIABLE_CANDIDATES[canonical]
    available = set(ds.data_vars) if hasattr(ds, "data_vars") else set(ds.variables)
    for name in candidates:
        if name in available:
            return name
    msg = f"None of the candidate NLDAS variable names {candidates} found in granule."
    raise KeyError(msg)


def _extract_grid(ds: Dataset, canonical_variable: str) -> ArrayLike:
    variable_name = _resolve_variable_name(ds, canonical_variable)
    return np.asarray(ds[variable_name].values, dtype="float64").squeeze()


def _extract_point_values(
    ds: Dataset,
    iy: ArrayLike,
    ix: ArrayLike,
) -> dict[str, ArrayLike]:
    valid = iy >= 0
    values: dict[str, ArrayLike] = {}
    for canonical_variable in NLDAS_VARIABLES:
        grid = _extract_grid(ds, canonical_variable)
        result = np.full(len(iy), np.nan, dtype="float64")
        result[valid] = grid[iy[valid], ix[valid]]
        result[valid & (result <= NLDAS_FILL_THRESHOLD)] = np.nan
        values[canonical_variable] = result
    return values


def _open_granule(content: bytes) -> Dataset:
    xr = cast("Any", importlib.import_module("xarray"))
    return xr.open_dataset(io.BytesIO(content), engine="h5netcdf")


def _download_granule(session: Session, url: str) -> bytes | None:
    for attempt in range(1, NLDAS_MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=NLDAS_REQUEST_TIMEOUT_SECONDS)
            if response.status_code == requests.codes.not_found:
                return None
            response.raise_for_status()
        except requests.RequestException:
            if attempt == NLDAS_MAX_RETRIES:
                LOGGER.warning(
                    "Giving up on granule %s after %d attempt(s).",
                    url,
                    attempt,
                )
                return None
            jitter = random.uniform(0.0, 2.0)  # noqa: S311
            time.sleep(NLDAS_RETRY_DELAY_SECONDS * attempt + jitter)
        else:
            return response.content
    return None


def _fetch_hour(
    session: Session,
    ts: object,
    iy: ArrayLike,
    ix: ArrayLike,
) -> dict[str, ArrayLike] | None:
    content = _download_granule(session, granule_url(ts))
    if content is None:
        return None
    try:
        ds = _open_granule(content)
        return _extract_point_values(ds, iy, ix)
    except OSError, ValueError, KeyError:
        LOGGER.exception("Failed to parse NLDAS granule for %s.", ts)
        return None


def _resolve_location_indices(
    shard_df: DataFrame,
    candidate_hours: list[Any],
) -> tuple[ArrayLike, ArrayLike]:
    """Resolve land-adjusted grid indices using one representative granule."""
    lats = shard_df["lat"].to_numpy(dtype="float64")
    lons = shard_df["lng"].to_numpy(dtype="float64")
    iy, ix = _nearest_grid_indices(lats, lons)

    session = _build_earthdata_session()
    for ts in candidate_hours:
        content = _download_granule(session, granule_url(ts))
        if content is None:
            continue
        ds = _open_granule(content)
        tair_grid = _extract_grid(ds, "Tair")
        return _resolve_valid_indices(iy, ix, lats, lons, tair_grid)

    LOGGER.error(
        "Could not download any of %d candidate granule(s) to resolve "
        "land-mask indices; proceeding with un-adjusted nearest-cell indices.",
        len(candidate_hours),
    )
    return iy, ix


def _load_nldas_city_shard(city_shard_index: int, city_shard_count: int) -> DataFrame:
    cities_df = pd.read_csv("cities.csv", usecols=["location_id", "lat", "lng"])
    cities_df = cities_df.sort_values("location_id").reset_index(drop=True)

    shard_size = (len(cities_df) + city_shard_count - 1) // city_shard_count
    start = city_shard_index * shard_size
    end = min(start + shard_size, len(cities_df))
    return cities_df.iloc[start:end].copy()


def _year_hour_range(year: int, allowed_months: list[int] | None = None) -> list[Any]:
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year + 1}-01-01")
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    if allowed_months:
        allowed = set(allowed_months)
        hours = hours[hours.month.isin(allowed)]
    return list(hours)


def _iter_time_batches(
    hours: list[Any], batch_hours: int
) -> list[tuple[int, list[Any]]]:
    if batch_hours <= 0:
        msg = f"batch_hours must be positive, got {batch_hours}"
        raise ValueError(msg)
    return [
        (batch_index, hours[start : start + batch_hours])
        for batch_index, start in enumerate(range(0, len(hours), batch_hours))
    ]


def _select_time_shard_batches(
    batches: list[tuple[int, list[Any]]],
    time_shard_index: int,
    time_shard_count: int,
) -> list[tuple[int, list[Any]]]:
    if time_shard_index >= time_shard_count:
        msg = f"time_shard_index {time_shard_index} must be less than time_shard_count {time_shard_count}"
        raise ValueError(msg)
    return [b for b in batches if b[0] % time_shard_count == time_shard_index]


def _process_time_batch(
    shard_df: DataFrame,
    iy: ArrayLike,
    ix: ArrayLike,
    hours: list[Any],
    download_workers: int,
) -> DataFrame:
    loc_ids = shard_df["location_id"].to_numpy()
    columns = ["location_id", "time", *NLDAS_VARIABLES]

    def _fetch(ts: object) -> tuple[object, dict[str, ArrayLike] | None]:
        return ts, _fetch_hour(_thread_session(), ts, iy, ix)

    rows: list[DataFrame] = []
    worker_count = max(1, min(download_workers, len(hours)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_fetch, ts) for ts in hours]
        for future in as_completed(futures):
            ts, values = future.result()
            if values is None:
                continue
            rows.append(
                pd.DataFrame(
                    {
                        "location_id": loc_ids,
                        "time": ts,
                        **{name: values[name] for name in NLDAS_VARIABLES},
                    },
                ),
            )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True)


def _compute_daily_wetbulb(hourly_df: DataFrame) -> DataFrame:
    output_columns = ["location_id", "date", "wetbulb", "wetbulb_avg"]
    df = hourly_df.dropna(subset=list(NLDAS_VARIABLES)).copy()
    if df.empty:
        return pd.DataFrame(columns=output_columns)

    tw = np.asarray(
        wetbulb_davies_jones(
            df["Tair"].to_numpy(dtype="float64"),
            df["Qair"].to_numpy(dtype="float64"),
            df["PSurf"].to_numpy(dtype="float64"),
        ),
        dtype="float64",
    )
    invalid = (
        ~np.isfinite(tw)
        | (tw < MIN_REASONABLE_WETBULB_C)
        | (tw > MAX_REASONABLE_WETBULB_C)
    )
    if np.any(invalid):
        LOGGER.warning(
            "Discarding %s wetbulb value(s) outside sanity range [%s, %s] C.",
            int(np.count_nonzero(invalid)),
            MIN_REASONABLE_WETBULB_C,
            MAX_REASONABLE_WETBULB_C,
        )
    df["tw"] = tw
    df = df[~invalid].copy()
    if df.empty:
        return pd.DataFrame(columns=output_columns)

    df["date"] = pd.to_datetime(df["time"]).dt.date
    daily = df.groupby(["location_id", "date"], as_index=False).agg(
        wetbulb=("tw", "max"),
        wetbulb_avg=("tw", "mean"),
        n_hours=("tw", "count"),
    )
    daily = daily[daily["n_hours"] >= MIN_DAILY_HOURS].drop(columns="n_hours")
    daily["wetbulb"] = (
        daily["wetbulb"] * WETBULB_ROUNDING_FACTOR
    ).round() / WETBULB_ROUNDING_FACTOR
    daily["wetbulb_avg"] = (
        daily["wetbulb_avg"] * WETBULB_AVG_ROUNDING_FACTOR
    ).round() / WETBULB_AVG_ROUNDING_FACTOR
    return daily.reset_index(drop=True)


def process_nldas(
    year: int,
    out_dir: str,
    city_shard_index: int,
    city_shard_count: int,
    time_shard_index: int,
    time_shard_count: int,
    download_workers: int,
    batch_hours: int,
    *,
    months: list[int] | None = None,
    force: bool = False,
) -> None:
    """Download NLDAS-2 data, compute daily wet-bulb, and save as parquet shards."""
    wetbulb_root = f"{out_dir}/wetbulb_data_csv"

    shard_df = _load_nldas_city_shard(city_shard_index, city_shard_count)
    if shard_df.empty:
        LOGGER.info(
            "No cities found for shard %s/%s.",
            city_shard_index,
            city_shard_count,
        )
        return

    batches = _select_time_shard_batches(
        _iter_time_batches(_year_hour_range(year, months), batch_hours),
        time_shard_index,
        time_shard_count,
    )
    if not batches:
        return

    filesystem, base_path = resolve_filesystem(wetbulb_root)
    pending_batches = [
        (batch_index, hours)
        for batch_index, hours in batches
        if force
        or not batch_exists(
            wetbulb_root,
            year,
            city_shard_index,
            batch_index,
            file_prefix="wetbulb",
            filesystem=filesystem,
            base_path=base_path,
        )
    ]
    if not pending_batches:
        return

    LOGGER.info(
        "NLDAS->wetbulb year=%d city_shard=%d/%d time_shard=%d/%d: %d batch(es)",
        year,
        city_shard_index,
        city_shard_count,
        time_shard_index,
        time_shard_count,
        len(pending_batches),
    )

    iy, ix = _resolve_location_indices(shard_df, pending_batches[0][1])

    for batch_index, hours in tqdm(pending_batches, desc=f"NLDAS->wetbulb {year}"):
        hourly_df = _process_time_batch(shard_df, iy, ix, hours, download_workers)
        daily_df = _compute_daily_wetbulb(hourly_df)
        if daily_df.empty:
            continue
        write_batch_partition(
            wetbulb_root,
            year,
            city_shard_index,
            daily_df,
            batch_index,
            file_prefix="wetbulb",
            filesystem=filesystem,
            base_path=base_path,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        help="Only process specific months (1-12)",
    )
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--city-shard-index", type=int, default=0)
    parser.add_argument("--city-shard-count", type=int, default=1)
    parser.add_argument(
        "--time-shard-index",
        type=int,
        default=None,
        help=(
            "Time shard to process. Defaults to the CLOUD_RUN_TASK_INDEX "
            "environment variable (0 outside Cloud Run), so a single Cloud "
            "Run execution with --tasks=N fans out across time shards."
        ),
    )
    parser.add_argument("--time-shard-count", type=int, default=1)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--batch-hours", type=int, default=DEFAULT_BATCH_HOURS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.time_shard_index is None:
        args.time_shard_index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))
    return args


def main() -> None:
    """Execute the NLDAS-2 wet-bulb processing pipeline."""
    exit_code = 0
    try:
        args = _parse_args()
        process_nldas(
            year=args.year,
            out_dir=args.out_dir,
            city_shard_index=args.city_shard_index,
            city_shard_count=args.city_shard_count,
            time_shard_index=args.time_shard_index,
            time_shard_count=args.time_shard_count,
            download_workers=args.download_workers,
            batch_hours=args.batch_hours,
            months=args.months,
            force=args.force,
        )
    except KeyboardInterrupt:
        exit_code = 130
        LOGGER.warning("NLDAS processing interrupted by user.")
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
        LOGGER.exception("NLDAS processing failed.")

    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(exit_code)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

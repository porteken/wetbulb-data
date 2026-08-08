# Copyright (C) 2026 Kenneth Porter

"""Map coastal EU cities onto a nearby ERA5-Land grid cell that actually has land.

ERA5-Land is a land-only reanalysis, so a city whose centre falls in a cell the
model treats as sea returns a complete time series of NaN -- no error, no empty
response, just nothing usable. This finds the nearest unclaimed cell with real
data for each such city and records it for `era5land.py --cell-map-csv`.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

import era5land
from cities_na import ERA5_LAND_GRID_DEG

pd = cast("Any", importlib.import_module("pandas"))

type DataFrame = Any
type CdsClient = Any
type CityRow = Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

EU_CITIES_CSV = "cities_eu.csv"
CELL_MAP_CSV = "cities_eu_era5land_cells.csv"
PROBE_DATE = "2020-07-01"
PROBE_DIR_DEFAULT = "era5land_probes"
MAX_SEARCH_CELLS = 8
EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def grid_cell(lat: float, lng: float) -> tuple[int, int]:
    """Return the ERA5-Land grid point a coordinate falls in."""
    return (round(lat / ERA5_LAND_GRID_DEG), round(lng / ERA5_LAND_GRID_DEG))


def _cell_centre(cell: tuple[int, int]) -> tuple[float, float]:
    return (
        round(cell[0] * ERA5_LAND_GRID_DEG, 4),
        round(cell[1] * ERA5_LAND_GRID_DEG, 4),
    )


def candidate_cells(
    lat: float,
    lng: float,
    claimed: set[tuple[int, int]],
    max_rings: int = MAX_SEARCH_CELLS,
) -> list[tuple[int, int]]:
    """Unclaimed grid cells near a point, nearest first."""
    origin = grid_cell(lat, lng)
    seen: set[tuple[int, int]] = set()
    scored: list[tuple[float, tuple[int, int]]] = []
    for d_lat in range(-max_rings, max_rings + 1):
        for d_lng in range(-max_rings, max_rings + 1):
            cell = (origin[0] + d_lat, origin[1] + d_lng)
            if cell in seen or cell in claimed:
                continue
            seen.add(cell)
            centre_lat, centre_lng = _cell_centre(cell)
            scored.append((_haversine_km(lat, lng, centre_lat, centre_lng), cell))
    scored.sort()
    return [cell for _, cell in scored]


def probe_has_land(client: CdsClient, lat: float, lng: float, probe_dir: Path) -> bool:
    """Fetch a single day at a cell and report whether the model has land there."""
    target = probe_dir / f"probe_{lat}_{lng}.csv"
    if not target.exists():
        request = {
            "variable": list(era5land.ERA5LAND_VARIABLES),
            "location": {"latitude": lat, "longitude": lng},
            "date": [f"{PROBE_DATE}/{PROBE_DATE}"],
            "data_format": "csv",
        }
        client.retrieve(era5land.ERA5LAND_DATASET, request, str(target))
    frame = era5land.read_era5land_download(str(target))
    return bool(frame[["t2m", "d2m", "sp"]].notna().all(axis=1).any())


def _snap_one(
    client: CdsClient,
    row: CityRow,
    candidates: list[tuple[int, int]],
    probe_dir: Path,
) -> dict[str, Any] | None:
    """Probe outward from one city until a cell with land is found."""
    for cell in candidates:
        centre_lat, centre_lng = _cell_centre(cell)
        try:
            has_land = probe_has_land(client, centre_lat, centre_lng, probe_dir)
        except era5land.RETRYABLE_FETCH_ERRORS:
            LOGGER.warning(
                "probe failed for %s at (%s, %s); trying the next cell",
                row.city,
                centre_lat,
                centre_lng,
            )
            continue
        if not has_land:
            continue
        moved = _haversine_km(row.lat, row.lng, centre_lat, centre_lng)
        LOGGER.info(
            "%-16s (%.4f, %.4f) -> (%.1f, %.1f)  %.1f km",
            row.city,
            row.lat,
            row.lng,
            centre_lat,
            centre_lng,
            moved,
        )
        return {
            "location_id": int(row.location_id),
            "city": row.city,
            "orig_lat": row.lat,
            "orig_lng": row.lng,
            "lat": centre_lat,
            "lng": centre_lng,
            "moved_km": round(moved, 2),
            "cell": cell,
        }
    LOGGER.error("%s: no land cell found within %d cells", row.city, MAX_SEARCH_CELLS)
    return None


def _existing_map(out_csv: str, requested: set[int]) -> DataFrame:
    """Rows of an earlier mapping for cities not being re-snapped now.

    Keeping these lets the script run incrementally: a city that lost a
    collision can be re-snapped without stealing a cell already handed out.
    """
    path = Path(out_csv)
    if not path.exists():
        return pd.DataFrame(columns=["location_id", "lat", "lng"])
    prior = pd.read_csv(path)
    return prior[~prior["location_id"].isin(requested)]


def _resolve_probe_dir(probe_dir: str) -> Path | None:
    """Resolve the probe directory, rejecting a path that escapes the working tree."""
    root = Path.cwd().resolve()
    resolved = Path(probe_dir).expanduser()
    resolved = (root / resolved).resolve() if not resolved.is_absolute() else resolved
    if not resolved.is_relative_to(root):
        LOGGER.error(
            "Refusing --probe-dir %s: it resolves outside the working directory %s.",
            probe_dir,
            root,
        )
        return None
    return resolved


def build_cell_map(
    location_ids: list[int],
    cities_csv: str,
    out_csv: str,
    probe_dir: str,
    concurrency: int,
) -> int:
    """Find a land cell for each requested city and write the override map."""
    cities = pd.read_csv(cities_csv)
    targets = cities[cities["location_id"].isin(location_ids)]
    if targets.empty:
        LOGGER.error("None of the requested location_ids are in %s", cities_csv)
        return 1

    requested = set(location_ids)
    claimed = {
        grid_cell(float(r.lat), float(r.lng))
        for r in cities.itertuples()
        if int(r.location_id) not in requested
    }

    existing = _existing_map(out_csv, requested)
    claimed |= {grid_cell(float(r.lat), float(r.lng)) for r in existing.itertuples()}
    LOGGER.info(
        "Snapping %d city(ies); %d cell(s) already claimed "
        "(%d by other cities' centres, %d by a previous mapping).",
        len(targets),
        len(claimed),
        len(claimed) - len(existing),
        len(existing),
    )

    probe_path = _resolve_probe_dir(probe_dir)
    if probe_path is None:
        return 1
    probe_path.mkdir(parents=True, exist_ok=True)
    client = era5land.cds_client()

    results: list[dict[str, Any]] = []
    rows = list(targets.itertuples())
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(
                _snap_one,
                client,
                row,
                candidate_cells(float(row.lat), float(row.lng), claimed),
                probe_path,
            ): row
            for row in rows
        }
        for future in as_completed(futures):
            found = future.result()
            if found is not None:
                results.append(found)

    if not results:
        LOGGER.error("No cities could be snapped.")
        return 1

    resolved: list[dict[str, Any]] = []
    taken: set[tuple[int, int]] = set()
    for found in sorted(results, key=lambda f: f["moved_km"]):
        if found["cell"] in taken:
            LOGGER.warning(
                "%s resolved to a cell another snapped city already took; "
                "leaving it unmapped so it is not silently duplicated.",
                found["city"],
            )
            continue
        taken.add(found["cell"])
        resolved.append({k: v for k, v in found.items() if k != "cell"})

    snapped = pd.DataFrame(resolved)
    frame = (
        pd.concat([existing, snapped], ignore_index=True)
        if not existing.empty
        else snapped
    )
    frame = frame.sort_values("location_id")
    frame.to_csv(out_csv, index=False)
    LOGGER.info(
        "Wrote %s with %d row(s): %d newly snapped of %d requested, %d kept "
        "from a previous run. Max move %.1f km.",
        out_csv,
        len(frame),
        len(resolved),
        len(targets),
        len(existing),
        frame["moved_km"].max(),
    )
    return 0 if len(resolved) == len(targets) else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities-csv", default=EU_CITIES_CSV)
    parser.add_argument("--out-csv", default=CELL_MAP_CSV)
    parser.add_argument("--probe-dir", default=PROBE_DIR_DEFAULT)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--location-ids", type=int, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    """Build the ERA5-Land land-cell override map."""
    load_dotenv(override=False)
    args = _parse_args()
    sys.exit(
        build_cell_map(
            args.location_ids,
            args.cities_csv,
            args.out_csv,
            args.probe_dir,
            args.concurrency,
        )
    )


if __name__ == "__main__":
    main()

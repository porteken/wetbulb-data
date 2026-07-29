"""Map city catalogs to nearby GHCNh stations with current-year observations."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, cast

import pandas as pd

EARTH_RADIUS_KM = 6371.0
MAX_DISTANCE_KM = 60.0
MAX_ELEVATION_DIFFERENCE_M = 300.0
REQUIRED_VARIABLES = frozenset({"temperature", "dew_point_temperature"})
PRESSURE_VARIABLES = frozenset(
    {"station_level_pressure", "sea_level_pressure", "altimeter"},
)
LOGGER = logging.getLogger(__name__)


def _haversine_km(
    lat1: float,
    lon1: float,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    phi1 = math.radians(lat1)
    phi2 = lon2 * 0 + lat2.map(math.radians)
    dphi = phi2 - phi1
    dlambda = lon2.map(math.radians) - math.radians(lon1)
    a = (dphi / 2).map(math.sin) ** 2 + (
        math.cos(phi1) * phi2.map(math.cos) * (dlambda / 2).map(math.sin) ** 2
    )
    return 2 * EARTH_RADIUS_KM * a.map(math.sqrt).map(math.asin)


def _observed_variables(metadata: dict[str, Any]) -> set[str]:
    observed: set[str] = set()
    for station in metadata.get("stations", []):
        for variable in station.get("dataTypes", []):
            if float(variable.get("coverage", 0.0)) > 0:
                observed.add(str(variable["id"]))
    return observed


def active_station_ids(inventory_dir: Path) -> set[str]:
    """Return stations with temperature, dew point, and at least one pressure."""
    station_ids: set[str] = set()
    for path in inventory_dir.glob("GHCNh_*.json"):
        if path.stat().st_size == 0:
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        variables = _observed_variables(metadata)
        if variables >= REQUIRED_VARIABLES and variables & PRESSURE_VARIABLES:
            station_ids.add(path.stem.removeprefix("GHCNh_").rsplit("_", 1)[0])
    return station_ids


def build_station_map(
    cities: pd.DataFrame,
    stations: pd.DataFrame,
    eligible_ids: set[str],
    *,
    max_distance_km: float = MAX_DISTANCE_KM,
    max_elevation_difference_m: float = MAX_ELEVATION_DIFFERENCE_M,
) -> pd.DataFrame:
    """Return the nearest eligible station satisfying distance/elevation limits."""
    eligible_mask = cast("pd.Series", stations["GHCN_ID"]).isin(list(eligible_ids))
    candidates = cast("pd.DataFrame", stations.loc[eligible_mask]).copy()
    rows: list[dict[str, Any]] = []
    for city in cities.to_dict("records"):
        nearby = candidates.copy()
        latitudes = cast("pd.Series", nearby["LATITUDE"])
        longitudes = cast("pd.Series", nearby["LONGITUDE"])
        distances = _haversine_km(
            float(city["lat"]),
            float(city["lng"]),
            latitudes,
            longitudes,
        )
        elevations = cast("pd.Series", nearby["ELEVATION"])
        elevation_differences = (elevations - float(city["dem_m"])).abs()
        nearby["dist_km"] = distances
        nearby["elevation_difference_m"] = elevation_differences
        within_limits = (distances <= max_distance_km) & (
            elevation_differences <= max_elevation_difference_m
        )
        nearby = cast("pd.DataFrame", nearby.loc[within_limits]).sort_values(
            by=["dist_km", "elevation_difference_m", "GHCN_ID"],
        )
        if nearby.empty:
            continue
        station = cast("pd.Series", nearby.iloc[0])
        rows.append(
            {
                "location_id": int(city["location_id"]),
                "ghcn_id": station["GHCN_ID"],
                "lon": round(float(station["LONGITUDE"]), 4),
                "dist_km": round(float(station["dist_km"]), 2),
                "elev_m": round(float(station["ELEVATION"]), 1),
                "utc_offset_hours": float(city["utc_offset_hours"]),
            },
        )
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", required=True)
    parser.add_argument("--station-list", required=True)
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    """Generate one committed city-to-GHCNh map."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    cities = pd.read_csv(args.cities)
    stations = pd.read_csv(args.station_list)
    station_map = build_station_map(
        cities,
        stations,
        active_station_ids(args.inventory_dir),
    )
    station_map.to_csv(args.out, index=False)
    missing = len(cities) - len(station_map)
    LOGGER.info(
        "Wrote %d mappings to %s; %d use gap-fill.",
        len(station_map),
        args.out,
        missing,
    )


if __name__ == "__main__":
    main()

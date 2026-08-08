# Copyright (C) 2026 Kenneth Porter

"""Map city catalogs to nearby GHCNh stations with current-year observations."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from station_exclusions import DISALLOWED_GHCNH_STATION_IDS

if TYPE_CHECKING:
    from collections.abc import Iterable

EARTH_RADIUS_KM = 6371.0
MAX_DISTANCE_KM = 60.0
MAX_ELEVATION_DIFFERENCE_M = 300.0
MAX_CANDIDATES_PER_CITY = 5
REQUIRED_VARIABLES = frozenset({"temperature", "dew_point_temperature"})
PRESSURE_VARIABLES = frozenset(
    {"station_level_pressure", "sea_level_pressure", "altimeter"},
)
DISALLOWED_STATION_PREFIXES = ("USL",)
type PandasSeries = pd.Series
LOGGER = logging.getLogger(__name__)
_YEAR_SUFFIX_ERRORS = (IndexError, ValueError)


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


def _station_coverage_scores(paths: Iterable[Path]) -> dict[str, float]:
    """Return limiting required-variable coverage for the supplied inventories.

    GHCNh's inventory coverage values are materially more useful than a binary
    "variable appeared at least once" check.  The limiting value ensures a
    station with abundant temperature but almost no humidity or pressure ranks
    behind a genuinely complete station.
    """
    scores: dict[str, float] = {}
    for path in paths:
        if path.stat().st_size == 0:
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        coverage: dict[str, float] = {}
        for station in metadata.get("stations", []):
            for variable in station.get("dataTypes", []):
                variable_id = str(variable["id"])
                coverage[variable_id] = max(
                    coverage.get(variable_id, 0.0),
                    float(variable.get("coverage", 0.0)),
                )
        pressure_coverage = max(
            (coverage.get(variable, 0.0) for variable in PRESSURE_VARIABLES),
            default=0.0,
        )
        required_coverages = [
            coverage.get("temperature", 0.0),
            coverage.get("dew_point_temperature", 0.0),
            pressure_coverage,
        ]
        score = min(required_coverages)
        if score <= 0:
            continue
        station_id = path.stem.removeprefix("GHCNh_").rsplit("_", 1)[0]
        scores[station_id] = max(scores.get(station_id, 0.0), score)
    return scores


def station_coverage_scores(inventory_dir: Path) -> dict[str, float]:
    """Return each usable station's limiting required-variable coverage."""
    return _station_coverage_scores(inventory_dir.glob("GHCNh_*.json"))


def station_coverage_scores_by_year(
    inventory_dir: Path,
) -> dict[int, dict[str, float]]:
    """Return usable-station coverage separately for each inventory year."""
    paths_by_year: dict[int, list[Path]] = {}
    for path in inventory_dir.glob("GHCNh_*.json"):
        try:
            year = int(path.stem.rsplit("_", 1)[1])
        except _YEAR_SUFFIX_ERRORS:
            continue
        paths_by_year.setdefault(year, []).append(path)

    scores_by_year: dict[int, dict[str, float]] = {}
    for year, paths in paths_by_year.items():
        scores_by_year[year] = _station_coverage_scores(paths)
    return scores_by_year


def build_year_specific_station_map(
    cities: pd.DataFrame,
    stations: pd.DataFrame,
    scores_by_year: dict[int, dict[str, float]],
    *,
    max_candidates_per_city: int = MAX_CANDIDATES_PER_CITY,
) -> pd.DataFrame:
    """Return independent ranked candidates for every inventory year."""
    frames: list[pd.DataFrame] = []
    for year, scores in sorted(scores_by_year.items()):
        year_map = build_station_map(
            cities,
            stations,
            scores,
            max_candidates_per_city=max_candidates_per_city,
        )
        if year_map.empty:
            continue
        year_map.insert(2, "year", year)
        frames.append(year_map)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def active_station_ids(inventory_dir: Path) -> set[str]:
    """Return stations with temperature, dew point, and at least one pressure."""
    return set(station_coverage_scores(inventory_dir))


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a DataFrame column with pandas' Series type for static checking."""
    return cast("PandasSeries", frame[column])


def build_station_map(
    cities: pd.DataFrame,
    stations: pd.DataFrame,
    eligible_ids: set[str] | dict[str, float],
    *,
    max_distance_km: float = MAX_DISTANCE_KM,
    max_elevation_difference_m: float = MAX_ELEVATION_DIFFERENCE_M,
    max_candidates_per_city: int = MAX_CANDIDATES_PER_CITY,
) -> pd.DataFrame:
    """Return ranked eligible station candidates satisfying physical limits."""
    if max_candidates_per_city < 1:
        msg = "max_candidates_per_city must be positive"
        raise ValueError(msg)
    coverage_by_id = (
        eligible_ids
        if isinstance(eligible_ids, dict)
        else dict.fromkeys(eligible_ids, 0.0)
    )
    station_ids = _series(stations, "GHCN_ID").astype("string")
    eligible_mask = station_ids.isin(list(coverage_by_id)) & (
        ~station_ids.str.startswith(DISALLOWED_STATION_PREFIXES, na=False)
        & ~station_ids.isin(DISALLOWED_GHCNH_STATION_IDS)
    )
    candidates = cast("pd.DataFrame", stations.loc[eligible_mask]).copy()
    candidates["variable_coverage"] = station_ids.loc[candidates.index].map(
        coverage_by_id,
    )
    rows: list[dict[str, Any]] = []
    for city in cities.to_dict("records"):
        nearby = candidates.copy()
        latitudes = _series(nearby, "LATITUDE")
        longitudes = _series(nearby, "LONGITUDE")
        distances = _haversine_km(
            float(city["lat"]),
            float(city["lng"]),
            latitudes,
            longitudes,
        )
        elevations = _series(nearby, "ELEVATION")
        elevation_differences = (elevations - float(city["dem_m"])).abs()
        nearby["dist_km"] = distances
        nearby["elevation_difference_m"] = elevation_differences
        within_limits = (distances <= max_distance_km) & (
            elevation_differences <= max_elevation_difference_m
        )
        nearby = cast("pd.DataFrame", nearby.loc[within_limits]).sort_values(
            by=[
                "variable_coverage",
                "dist_km",
                "elevation_difference_m",
                "GHCN_ID",
            ],
            ascending=[False, True, True, True],
        )
        if nearby.empty:
            continue
        for candidate_rank, (_, station) in enumerate(
            nearby.head(max_candidates_per_city).iterrows(),
            start=1,
        ):
            rows.append(
                {
                    "location_id": int(city["location_id"]),
                    "ghcn_id": station["GHCN_ID"],
                    "candidate_rank": candidate_rank,
                    "variable_coverage": float(station["variable_coverage"]),
                    "lon": round(float(station["LONGITUDE"]), 4),
                    "dist_km": round(float(station["dist_km"]), 2),
                    "elev_m": round(float(station["ELEVATION"]), 1),
                    "elevation_difference_m": round(
                        float(station["elevation_difference_m"]),
                        1,
                    ),
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
    parser.add_argument(
        "--max-candidates-per-city",
        type=int,
        default=MAX_CANDIDATES_PER_CITY,
    )
    parser.add_argument(
        "--year-specific",
        action="store_true",
        help="Write separate candidate rows for each inventory year.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate one committed city-to-GHCNh map."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    cities = pd.read_csv(args.cities)
    stations = pd.read_csv(args.station_list)
    if args.year_specific:
        station_map = build_year_specific_station_map(
            cities,
            stations,
            station_coverage_scores_by_year(args.inventory_dir),
            max_candidates_per_city=args.max_candidates_per_city,
        )
    else:
        station_map = build_station_map(
            cities,
            stations,
            station_coverage_scores(args.inventory_dir),
            max_candidates_per_city=args.max_candidates_per_city,
        )
    station_map.to_csv(args.out, index=False)
    mapped_cities = station_map["location_id"].nunique() if not station_map.empty else 0
    missing = len(cities) - mapped_cities
    LOGGER.info(
        "Wrote %d candidate mappings for %d cities to %s; %d use gap-fill.",
        len(station_map),
        mapped_cities,
        args.out,
        missing,
    )


if __name__ == "__main__":
    main()

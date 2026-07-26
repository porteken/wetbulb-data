"""Match Canadian population centres to nearby ECCC hourly stations."""

from __future__ import annotations

import argparse
import importlib
import io
import logging
from typing import Any, cast

import requests

pd = cast("Any", importlib.import_module("pandas"))
np = cast("Any", importlib.import_module("numpy"))
type DataFrame = Any

LOGGER = logging.getLogger(__name__)
STATION_INVENTORY_URL = (
    "https://collaboration.cmc.ec.gc.ca/cmc/climate/"
    "Get_More_Data_Plus_de_donnees/Station Inventory EN.csv"
)
EARTH_RADIUS_KM = 6371.0088
MAX_DISTANCE_KM = 100.0
PREFERRED_DISTANCE_KM = 30.0
EXTENDED_DISTANCE_KM = 60.0
MAX_CANDIDATES = 3
CANADA_START_YEAR = 2000

_INVENTORY_ALIASES = {
    "Station ID": "station_id",
    "Name": "station_name",
    "Station Name": "station_name",
    "Province": "province",
    "Latitude (Decimal Degrees)": "lat",
    "Longitude (Decimal Degrees)": "lng",
    "Elevation (m)": "elevation_m",
    "HLY First Year": "hourly_first_year",
    "HLY Last Year": "hourly_last_year",
    "Hourly First Year": "hourly_first_year",
    "Hourly Last Year": "hourly_last_year",
    "Climate ID": "climate_id",
}


def load_station_inventory(
    source: str = STATION_INVENTORY_URL,
    *,
    session: requests.Session | None = None,
) -> DataFrame:
    """Load and normalize ECCC's station inventory CSV."""
    csv_source: Any = source
    if source.startswith(("http://", "https://")):
        response = (session or requests.Session()).get(source, timeout=60)
        response.raise_for_status()
        csv_source = io.BytesIO(response.content)
    frame = pd.read_csv(csv_source, skiprows=3, encoding="latin1")
    frame = frame.rename(
        columns={
            key: value for key, value in _INVENTORY_ALIASES.items() if key in frame
        }
    )
    required = {
        "station_id",
        "lat",
        "lng",
        "elevation_m",
        "hourly_first_year",
        "hourly_last_year",
    }
    missing = required - set(frame.columns)
    if missing:
        message = f"ECCC station inventory is missing columns: {sorted(missing)}"
        raise ValueError(message)
    for column in required - {"station_id"}:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["station_id"] = pd.to_numeric(frame["station_id"], errors="coerce")
    return frame.dropna(subset=list(required)).copy()


def haversine_km(
    lat1: float, lng1: float, lat2: DataFrame, lng2: DataFrame
) -> DataFrame:
    """Return great-circle distance from one city to every station."""
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    value = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlng / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(value))


def rank_stations(
    inventory: DataFrame,
    *,
    city_lat: float,
    city_lng: float,
    start_year: int,
    end_year: int,
) -> DataFrame:
    """Rank nearby stations by record coverage, radius tier, then distance."""
    distance = haversine_km(city_lat, city_lng, inventory["lat"], inventory["lng"])
    candidates = inventory[distance <= MAX_DISTANCE_KM].copy()
    if candidates.empty:
        return candidates
    candidates["distance_km"] = distance.loc[candidates.index]
    candidates["coverage_years"] = (
        np.minimum(candidates["hourly_last_year"], end_year)
        - np.maximum(candidates["hourly_first_year"], start_year)
        + 1
    ).clip(lower=0)
    candidates["full_coverage"] = (candidates["hourly_first_year"] <= start_year) & (
        candidates["hourly_last_year"] >= end_year
    )
    candidates["radius_tier"] = np.select(
        [
            candidates["distance_km"] <= PREFERRED_DISTANCE_KM,
            candidates["distance_km"] <= EXTENDED_DISTANCE_KM,
        ],
        [0, 1],
        default=2,
    )
    return candidates.sort_values(
        ["radius_tier", "full_coverage", "coverage_years", "distance_km"],
        ascending=[True, False, False, True],
    )


def build_station_map(
    cities_csv: str,
    inventory: DataFrame,
    *,
    start_year: int = CANADA_START_YEAR,
    end_year: int = 2025,
) -> DataFrame:
    """Return an ordered three-station ECCC crosswalk for each Canadian city."""
    cities = pd.read_csv(cities_csv, usecols=["location_id", "city", "lat", "lng"])
    rows: list[dict[str, Any]] = []
    for city in cities.itertuples():
        ranked = rank_stations(
            inventory,
            city_lat=float(city.lat),
            city_lng=float(city.lng),
            start_year=start_year,
            end_year=end_year,
        ).head(MAX_CANDIDATES)
        if ranked.empty:
            LOGGER.warning("No ECCC hourly station within 100 km of %s", city.city)
            rows.append(
                {
                    "location_id": city.location_id,
                    "eccc_station_ids": None,
                    "primary_station_name": None,
                    "distance_km": None,
                    "elevation_m": None,
                    "utc_offset_hours": round(float(city.lng) / 15.0),
                }
            )
            continue
        primary = ranked.iloc[0]
        rows.append(
            {
                "location_id": city.location_id,
                "eccc_station_ids": "|".join(
                    str(int(value)) for value in ranked["station_id"]
                ),
                "primary_station_name": primary.get("station_name"),
                "distance_km": round(float(primary["distance_km"]), 2),
                "elevation_m": float(primary["elevation_m"]),
                "utc_offset_hours": round(float(city.lng) / 15.0),
            }
        )
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities-csv", default="cities_ca.csv")
    parser.add_argument("--inventory", default=STATION_INVENTORY_URL)
    parser.add_argument("--out", default="cities_ca_eccc_stations.csv")
    parser.add_argument("--start-year", type=int, default=CANADA_START_YEAR)
    parser.add_argument("--end-year", type=int, default=2025)
    return parser.parse_args()


def main() -> None:
    """Generate the Canadian city-to-ECCC-station crosswalk."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    args = _parse_args()
    result = build_station_map(
        args.cities_csv,
        load_station_inventory(args.inventory),
        start_year=args.start_year,
        end_year=args.end_year,
    )
    result.to_csv(args.out, index=False)
    LOGGER.info("Saved %d station mappings to %s", len(result), args.out)


if __name__ == "__main__":
    main()

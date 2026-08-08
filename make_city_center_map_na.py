# Copyright (C) 2026 Kenneth Porter

"""Generate `cities_na_centers.csv`: each NA city's downtown display coordinate.

The catalog stores the Census/StatCan representative point of a municipal polygon,
which lands wherever the polygon balances rather than on the city center -- Los
Angeles resolves to the west side and New York to Brooklyn. Map markers instead use
the GeoNames populated-place coordinate for the same named city, matched on name,
country, and admin1 division -- or on the seat town for amalgamated municipalities that
have no settlement of their own name. Cities with no confident match keep their point.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import unicodedata
from typing import TYPE_CHECKING, Any, Protocol, cast

import requests

from cities_na import GEONAMES_CITIES_URL, _download, read_geonames_table
from isd_history import haversine_km

if TYPE_CHECKING:
    from pathlib import Path

pd = cast("Any", importlib.import_module("pandas"))

type DataFrame = Any


class CityRow(Protocol):
    """Catalog fields consumed while resolving a city center."""

    city: str
    place_id: str
    country: str
    state: str
    lat: float
    lng: float


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

CITIES_NA_CSV = "cities_na.csv"
OUTPUT_FILE = "cities_na_centers.csv"

MAX_CENTER_OFFSET_KM = 60.0
CENTER_COORD_DECIMALS = 4

_GEONAMES_COLUMNS = {
    1: "name",
    2: "asciiname",
    3: "alternatenames",
    4: "lat",
    5: "lng",
    6: "feature_class",
    8: "country_code",
    10: "admin1_code",
}

_SAINT_PREFIXES = ("st ", "ste ")

# Amalgamated municipalities that carry no settlement of their own name; markers
# belong on the seat town rather than on the centroid of the merged region.
SEAT_SETTLEMENTS = {
    "CA3536020": "Chatham",
    "CA3518017": "Bowmanville",
    "CA4811052": "Sherwood Park",
    "CA1217030": "Sydney",
}

_GEONAMES_ADMIN1_CA = {
    "AB": "01",
    "BC": "02",
    "MB": "03",
    "NB": "04",
    "NL": "05",
    "NS": "07",
    "ON": "08",
    "PE": "09",
    "QC": "10",
    "SK": "11",
    "YT": "12",
    "NT": "13",
    "NU": "14",
}


def geonames_admin1_code(country: str, state: str) -> str:
    """Return the GeoNames admin1 code for a catalog state or province abbreviation."""
    if country == "CA":
        return _GEONAMES_ADMIN1_CA.get(state, state)
    return state


def normalize_place_name(name: str) -> str:
    """Return a name folded to a form comparable across Census and GeoNames spellings."""
    decomposed = unicodedata.normalize("NFKD", str(name))
    stripped = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character) and character not in ".'`"
    )
    folded = " ".join(stripped.replace("-", " ").casefold().split())
    for prefix in _SAINT_PREFIXES:
        if folded.startswith(prefix):
            return "saint " + folded[len(prefix) :]
    return folded


def name_variants(city_name: str, place_id: str = "") -> set[str]:
    """Return the normalized names a city may carry, splitting bilingual CSD names."""
    seat = SEAT_SETTLEMENTS.get(place_id)
    if seat is not None:
        return {normalize_place_name(seat)}
    parts = [city_name, *str(city_name).split(" / ")]
    return {normalize_place_name(part) for part in parts}


def load_geonames_populated_places(payload: bytes) -> DataFrame:
    """Return GeoNames populated places with their normalized name variants."""
    frame = read_geonames_table(payload, _GEONAMES_COLUMNS)
    for column in ("lat", "lng"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["lat", "lng", "country_code", "admin1_code"])
    frame["primary_names"] = [
        {normalize_place_name(name), normalize_place_name(ascii_name)}
        for name, ascii_name in zip(frame["name"], frame["asciiname"], strict=True)
    ]
    frame["alternate_names"] = [
        {normalize_place_name(part) for part in str(alternates).split(",")}
        if isinstance(alternates, str)
        else set()
        for alternates in frame["alternatenames"]
    ]
    return frame.reset_index(drop=True)


def _nearest_match(
    matches: DataFrame, lat: float, lng: float
) -> tuple[Any, float] | None:
    if matches.empty:
        return None
    distance = haversine_km(
        lat, lng, matches["lat"].to_numpy(), matches["lng"].to_numpy()
    )
    nearest = int(distance.argmin())
    if float(distance[nearest]) > MAX_CENTER_OFFSET_KM:
        return None
    return matches.iloc[nearest], float(distance[nearest])


def resolve_city_center(
    places: DataFrame,
    city: object,
) -> tuple[Any, float] | None:
    """Return the GeoNames place and its offset for a city, preferring primary names."""
    row = cast("CityRow", city)
    wanted = name_variants(row.city, str(row.place_id))
    admin1 = geonames_admin1_code(str(row.country), str(row.state))
    in_division = places[
        (places["country_code"] == row.country) & (places["admin1_code"] == admin1)
    ]
    for column in ("primary_names", "alternate_names"):
        candidates = in_division[
            in_division[column].map(lambda names: bool(names & wanted))
        ]
        match = _nearest_match(candidates, float(row.lat), float(row.lng))
        if match is not None:
            return match
    return None


def build_center_map(
    cities_csv: str | Path = CITIES_NA_CSV,
    *,
    session: requests.Session | None = None,
    geonames_url: str = GEONAMES_CITIES_URL,
) -> DataFrame:
    """Return a DataFrame of location_id, center_lat, center_lng, offset_km per city."""
    http = session or requests.Session()
    places = load_geonames_populated_places(_download(geonames_url, session=http))
    cities = pd.read_csv(
        cities_csv,
        usecols=["location_id", "place_id", "city", "state", "country", "lat", "lng"],
        dtype={"place_id": str},
    )

    rows: list[dict[str, Any]] = []
    unmatched: list[str] = []

    for city in cities.itertuples():
        match = resolve_city_center(places, city)
        if match is None:
            unmatched.append(f"{city.city}, {city.state}")
            rows.append(
                {
                    "location_id": city.location_id,
                    "center_lat": round(float(city.lat), CENTER_COORD_DECIMALS),
                    "center_lng": round(float(city.lng), CENTER_COORD_DECIMALS),
                    "offset_km": 0.0,
                    "matched": False,
                },
            )
            continue
        place, offset_km = match
        rows.append(
            {
                "location_id": city.location_id,
                "center_lat": round(float(place.lat), CENTER_COORD_DECIMALS),
                "center_lng": round(float(place.lng), CENTER_COORD_DECIMALS),
                "offset_km": round(offset_km, 2),
                "matched": True,
            },
        )

    if unmatched:
        LOGGER.warning(
            "%d city/cities kept their catalog point (no GeoNames match): %s",
            len(unmatched),
            "; ".join(unmatched),
        )

    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities-csv", default=CITIES_NA_CSV)
    parser.add_argument("--out", default=OUTPUT_FILE)
    parser.add_argument("--geonames-url", default=GEONAMES_CITIES_URL)
    return parser.parse_args()


def main() -> None:
    """Build and write the North America city-to-center map."""
    args = _parse_args()
    center_map = build_center_map(args.cities_csv, geonames_url=args.geonames_url)
    center_map.to_csv(args.out, index=False)
    matched = center_map["matched"]
    LOGGER.info(
        "Wrote %d row(s) to %s (%d recentered, median shift %.1f km, max %.1f km).",
        len(center_map),
        args.out,
        int(matched.sum()),
        float(center_map.loc[matched, "offset_km"].median()),
        float(center_map["offset_km"].max()),
    )


if __name__ == "__main__":
    main()

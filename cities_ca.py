"""Generate the top-100 Canadian population-centre city table."""

from __future__ import annotations

import argparse
import io
import logging
import zipfile
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import requests

pd: Any = cast("Any", import_module("pandas"))
type DataFrame = Any

LOGGER = logging.getLogger(__name__)
GEOSUITE_URL = (
    "https://www12.statcan.gc.ca/census-recensement/2021/geo/aip-pia/"
    "geosuite/files-fichiers/2021_92-150-X_eng.zip"
)
MAX_CANADIAN_CITIES = 100
CANADA_LOCATION_ID_START = 500
CITY_COORD_DECIMALS = 3
PROVINCE_ABBREVIATIONS = {
    "10": "NL",
    "11": "PE",
    "12": "NS",
    "13": "NB",
    "24": "QC",
    "35": "ON",
    "46": "MB",
    "47": "SK",
    "48": "AB",
    "59": "BC",
    "60": "YT",
    "61": "NT",
    "62": "NU",
}

_ALIASES = {
    "PCNAME": "city",
    "POPCTRNAME": "city",
    "PRUID": "province_code",
    "POP_2021": "population",
    "POP2021": "population",
    "Population, 2021": "population",
    "LAT": "lat",
    "LATITUDE": "lat",
    "LONG": "lng",
    "LON": "lng",
    "LONGITUDE": "lng",
    "POPCTRRAname": "city",
    "POPCTRRAuid": "city_id",
    "PRuid": "province_code",
    "XPRuid": "secondary_province_code",
    "POPCTRRApop_2021": "population",
    "PNrplat": "lat",
    "PNrplong": "lng",
}


def normalize_population_centres(frame: DataFrame) -> DataFrame:
    """Normalize GeoSuite POPCTR rows into the application city schema."""
    frame = frame.rename(
        columns={key: value for key, value in _ALIASES.items() if key in frame}
    ).copy()
    required = {"city", "province_code", "population", "lat", "lng"}
    missing = required - set(frame.columns)
    if missing:
        message = f"GeoSuite POPCTR data is missing columns: {sorted(missing)}"
        raise ValueError(message)
    for column in ("population", "lat", "lng"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["province_code"] = (
        frame["province_code"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )
    frame["state"] = frame["province_code"].map(PROVINCE_ABBREVIATIONS)
    if "secondary_province_code" in frame:
        secondary = (
            frame["secondary_province_code"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(2)
            .map(PROVINCE_ABBREVIATIONS)
        )
        frame["state"] = [
            "-".join(sorted({primary, other})) if pd.notna(other) else primary
            for primary, other in zip(frame["state"], secondary, strict=True)
        ]
    # Ottawa-Gatineau is represented once per provincial part in GeoSuite.
    # Collapse any such rows into one urban entity and expose both provinces.
    grouped_rows: list[dict[str, Any]] = []
    valid = frame.dropna(subset=["city", "population", "lat", "lng", "state"]).copy()
    if "city_id" not in valid:
        valid["city_id"] = valid["city"]
    for _, rows in valid.groupby("city_id", sort=False):
        population = rows["population"].sum()
        grouped_rows.append(
            {
                "city": rows["city"].iloc[0],
                "state": "-".join(sorted(rows["state"].unique())),
                "population": population,
                "lat": (rows["lat"] * rows["population"]).sum() / population,
                "lng": (rows["lng"] * rows["population"]).sum() / population,
            }
        )
    grouped = pd.DataFrame(grouped_rows)
    grouped = grouped.sort_values("population", ascending=False).head(
        MAX_CANADIAN_CITIES
    )
    grouped.insert(
        0,
        "location_id",
        range(CANADA_LOCATION_ID_START, CANADA_LOCATION_ID_START + len(grouped)),
    )
    grouped["lat"] = grouped["lat"].round(CITY_COORD_DECIMALS)
    grouped["lng"] = grouped["lng"].round(CITY_COORD_DECIMALS)
    return grouped[["location_id", "city", "state", "lat", "lng"]]


def load_geosuite_popctr(
    source: str = GEOSUITE_URL,
    *,
    session: requests.Session | None = None,
) -> DataFrame:
    """Download GeoSuite and return its population-centre reference table."""
    if source.startswith(("http://", "https://")):
        response = (session or requests.Session()).get(source, timeout=300)
        response.raise_for_status()
        archive_bytes = response.content
    else:
        with Path(source).open("rb") as stream:
            archive_bytes = stream.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        population_tables = [
            name for name in archive.namelist() if name.upper().endswith("/POPCTR.CSV")
        ]
        place_tables = [
            name for name in archive.namelist() if name.upper().endswith("/PN.CSV")
        ]
        if not population_tables or not place_tables:
            message = "GeoSuite archive has no POPCTR.csv table"
            raise ValueError(message)
        with archive.open(population_tables[0]) as stream:
            population = pd.read_csv(stream, encoding="windows-1252", dtype=str)
        with archive.open(place_tables[0]) as stream:
            places = pd.read_csv(
                stream,
                encoding="windows-1252",
                dtype=str,
                usecols=["POPCTRRAuid", "PNrplat", "PNrplong"],
            )
    places["PNrplat"] = pd.to_numeric(places["PNrplat"], errors="coerce")
    places["PNrplong"] = pd.to_numeric(places["PNrplong"], errors="coerce")
    representative = places.groupby("POPCTRRAuid", as_index=False).agg(
        PNrplat=("PNrplat", "mean"),
        PNrplong=("PNrplong", "mean"),
    )
    return population.merge(representative, on="POPCTRRAuid", how="left")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geosuite", default=GEOSUITE_URL)
    parser.add_argument("--out", default="cities_ca.csv")
    return parser.parse_args()


def main() -> None:
    """Generate ``cities_ca.csv`` from the official GeoSuite archive."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    args = _parse_args()
    cities = normalize_population_centres(load_geosuite_popctr(args.geosuite))
    cities.to_csv(args.out, index=False, float_format=f"%.{CITY_COORD_DECIMALS}f")
    LOGGER.info("Saved %d Canadian population centres to %s", len(cities), args.out)


if __name__ == "__main__":
    main()

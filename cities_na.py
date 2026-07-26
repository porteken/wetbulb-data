"""Build the combined North America catalog of the 500 largest US and Canadian cities.

Cities come from authoritative municipal registers -- Census incorporated places for the
US and StatCan census subdivisions for Canada -- so census-designated places, townships,
and municipal boroughs never enter the ranking. Selection then walks those cities in
descending population order and keeps one only when it claims both an unused ERA5-Land
grid cell and an unused ISD station, so no two entries share a reanalysis cell or a
weather station.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import logging
import os
import re
import tempfile
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests

import nldas
from isd_history import (
    candidate_ids_for_station,
    haversine_km,
    prepare_history,
    rank_station_rows,
)
from make_isd_station_map import _candidate_verified_at, fetch_isd_history

pd = cast("Any", importlib.import_module("pandas"))

type DataFrame = Any
type StationVerifier = Callable[[list[str], int], bool]

LOGGER = logging.getLogger(__name__)

CATALOG_VERSION = "na-census-2025-csd-2021"

CENSUS_ESTIMATES_URL = (
    "https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/"
    "cities/totals/sub-est2025.csv"
)
CENSUS_GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2025_Gazetteer/"
    "2025_Gaz_place_national.zip"
)
GEOSUITE_URL = (
    "https://www12.statcan.gc.ca/census-recensement/2021/geo/aip-pia/"
    "geosuite/files-fichiers/2021_92-150-X_eng.zip"
)
GEONAMES_CITIES_URL = "https://download.geonames.org/export/dump/cities15000.zip"
ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"

MAX_CITIES = 500
CANDIDATE_POOL_SIZE = 3000
ERA5_LAND_GRID_DEG = 0.1
CITY_COORD_DECIMALS = 4
NA_START_YEAR = 2000
CENSUS_POPULATION_COLUMN = "POPESTIMATE2025"
INCORPORATED_PLACE_SUMLEV = 162
EXCLUDED_STATE_FIPS = frozenset({"02", "15"})
MAX_ATTRIBUTE_DISTANCE_KM = 60.0
NOTABLE_CSD_POPULATION = 50_000

NA_HISTORY_MIN_LAT = 24.0
NA_HISTORY_MAX_LAT = 72.0
NA_HISTORY_MIN_LON = -142.0
NA_HISTORY_MAX_LON = -52.0

STATE_ABBR = {
    "01": "AL",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}

PROVINCE_ABBR = {
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

MUNICIPAL_CSD_TYPES = frozenset(
    {
        "C",
        "CC",
        "CG",
        "CT",
        "CU",
        "CV",
        "CY",
        "DM",
        "HAM",
        "ID",
        "IM",
        "LGD",
        "M",
        "MD",
        "MRM",
        "MU",
        "MÉ",
        "NH",
        "NV",
        "P",
        "PE",
        "RCR",
        "RGM",
        "RM",
        "RMU",
        "RV",
        "SM",
        "SV",
        "T",
        "TP",
        "TV",
        "V",
        "VL",
    }
)

_PLACE_SUFFIX_PATTERN = re.compile(
    r"\s+(city|town|village|borough|municipality|CDP|"
    r"urban county|unified government|consolidated government|"
    r"metro government|metropolitan government)$",
    re.IGNORECASE,
)

CONSOLIDATED_NAMES = {
    "1836003": "Indianapolis",
    "4752006": "Nashville",
    "2148006": "Louisville",
    "2146027": "Lexington",
    "1304204": "Augusta",
    "1349008": "Macon",
    "1303440": "Athens",
    "1608830": "Boise",
    "0665042": "Ventura",
}

_GEONAMES_COLUMNS = {
    4: "lat",
    5: "lng",
    6: "feature_class",
    16: "dem_m",
    17: "timezone",
}

CATALOG_COLUMNS = [
    "location_id",
    "place_id",
    "city",
    "state",
    "country",
    "population",
    "lat",
    "lng",
    "dem_m",
    "timezone",
    "utc_offset_hours",
]

STATION_COLUMNS = [
    "location_id",
    "usaf",
    "wban",
    "isd_ids",
    "lon",
    "dist_km",
    "elev_m",
    "utc_offset_hours",
    "begin_verified",
]

_PLACE_COLUMNS = ["place_id", "city", "state", "country", "population", "lat", "lng"]
_STANDARD_OFFSET_YEAR = 2025


def _download(url: str, *, session: requests.Session) -> bytes:
    response = session.get(url, timeout=300)
    response.raise_for_status()
    if not response.content:
        message = f"empty download: {url}"
        raise ValueError(message)
    return response.content


def _column(frame: DataFrame, *names: str) -> str:
    lookup = {str(name).strip().upper(): name for name in frame.columns}
    for name in names:
        if name.upper() in lookup:
            return lookup[name.upper()]
    message = f"missing required column (one of {', '.join(names)})"
    raise ValueError(message)


def clean_place_name(name: str, census_geoid: str) -> str:
    """Return a display name without Census legal-entity suffixes."""
    override = CONSOLIDATED_NAMES.get(census_geoid)
    if override is not None:
        return override
    cleaned = re.sub(r"\s*\([^)]*\)", "", str(name)).strip()
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = _PLACE_SUFFIX_PATTERN.sub("", cleaned).strip()
    return cleaned or str(name).strip()


def load_census_places(payload: bytes) -> DataFrame:
    """Return incorporated CONUS places from the Census subcounty estimates file."""
    frame = pd.read_csv(io.BytesIO(payload), dtype=str, encoding="latin-1")
    sumlev = _column(frame, "SUMLEV")
    state = _column(frame, "STATE")
    place = _column(frame, "PLACE")
    name = _column(frame, "NAME")
    population = _column(frame, CENSUS_POPULATION_COLUMN)

    frame = frame.copy()
    frame[state] = frame[state].astype(str).str.zfill(2)
    frame[place] = frame[place].astype(str).str.zfill(5)
    frame = frame[
        (pd.to_numeric(frame[sumlev], errors="coerce") == INCORPORATED_PLACE_SUMLEV)
        & ~frame[state].isin(EXCLUDED_STATE_FIPS)
    ].copy()

    frame["census_geoid"] = frame[state] + frame[place]
    frame["state"] = frame[state].map(STATE_ABBR)
    if frame["state"].isna().any():
        unknown = sorted(frame.loc[frame["state"].isna(), state].unique())
        message = f"Census input contains unknown state FIPS: {unknown}"
        raise ValueError(message)
    frame["population"] = pd.to_numeric(frame[population], errors="coerce")
    frame = frame.dropna(subset=["population"])
    frame["population"] = frame["population"].astype("int64")
    frame["city"] = [
        clean_place_name(row_name, geoid)
        for row_name, geoid in zip(frame[name], frame["census_geoid"], strict=True)
    ]
    return frame.drop_duplicates("census_geoid", keep="last")


def load_gazetteer_points(payload: bytes, url: str) -> DataFrame:
    """Return each Census place GEOID with its internal representative point."""
    raw = payload
    if urlparse(url).path.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [
                name
                for name in archive.namelist()
                if not name.endswith("/")
                and Path(name).suffix.lower() in {".txt", ".csv"}
            ]
            if len(members) != 1:
                message = f"expected one data file in {url}"
                raise ValueError(message)
            raw = archive.read(members[0])
    header = raw.splitlines()[0]
    separator = next(
        (candidate for candidate in ("|", "\t") if candidate.encode() in header), ","
    )
    frame = pd.read_csv(io.BytesIO(raw), sep=separator, dtype=str, encoding="latin-1")
    frame.columns = [str(column).strip() for column in frame.columns]
    geoid = _column(frame, "GEOID", "GEOID20")
    lat = _column(frame, "INTPTLAT", "INTPTLAT20")
    lng = _column(frame, "INTPTLONG", "INTPTLON", "INTPTLONG20")
    points = frame[[geoid, lat, lng]].copy()
    points.columns = ["census_geoid", "lat", "lng"]
    points["census_geoid"] = points["census_geoid"].astype(str).str.zfill(7)
    points["lat"] = pd.to_numeric(points["lat"], errors="coerce")
    points["lng"] = pd.to_numeric(points["lng"], errors="coerce")
    return points.dropna(subset=["lat", "lng"]).drop_duplicates(
        "census_geoid", keep="first"
    )


def build_us_places(estimates: DataFrame, gazetteer: DataFrame) -> DataFrame:
    """Return normalized US city rows with authoritative coordinates."""
    merged = estimates.merge(gazetteer, on="census_geoid", how="inner")
    merged["place_id"] = "US" + merged["census_geoid"]
    merged["country"] = "US"
    return merged[_PLACE_COLUMNS].reset_index(drop=True)


def load_geosuite_csd(payload: bytes) -> DataFrame:
    """Return the StatCan census-subdivision table from a GeoSuite archive."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [
            name for name in archive.namelist() if name.upper().endswith("/CSD.CSV")
        ]
        if not members:
            message = "GeoSuite archive has no CSD.csv table"
            raise ValueError(message)
        with archive.open(members[0]) as stream:
            return pd.read_csv(stream, encoding="windows-1252", dtype=str)


def build_ca_places(csd: DataFrame) -> DataFrame:
    """Return normalized Canadian municipality rows from the CSD table."""
    frame = csd.copy()
    for column in ("CSDuid", "CSDname", "CSDtype", "CSDpop_2021", "PRuid"):
        if column not in frame:
            message = f"GeoSuite CSD table is missing column: {column}"
            raise ValueError(message)
    frame["population"] = pd.to_numeric(frame["CSDpop_2021"], errors="coerce")
    frame["lat"] = pd.to_numeric(frame["CSDrplat"], errors="coerce")
    frame["lng"] = pd.to_numeric(frame["CSDrplong"], errors="coerce")
    frame["state"] = frame["PRuid"].astype(str).str.zfill(2).map(PROVINCE_ABBR)
    frame = frame.dropna(subset=["population", "lat", "lng", "state"])

    municipal = frame[frame["CSDtype"].isin(MUNICIPAL_CSD_TYPES)]
    dropped = frame[~frame["CSDtype"].isin(MUNICIPAL_CSD_TYPES)]
    notable = dropped[dropped["population"] >= NOTABLE_CSD_POPULATION]
    if not notable.empty:
        LOGGER.warning(
            "Excluded %d non-municipal CSD(s) above 50k: %s",
            len(notable),
            ", ".join(f"{row.CSDname} ({row.CSDtype})" for row in notable.itertuples()),
        )

    result = municipal.copy()
    result["place_id"] = "CA" + result["CSDuid"].astype(str)
    result["city"] = result["CSDname"].astype(str).str.strip()
    result["country"] = "CA"
    result["population"] = result["population"].astype("int64")
    return result[_PLACE_COLUMNS].reset_index(drop=True)


def load_geonames_places(payload: bytes) -> DataFrame:
    """Return GeoNames populated places used only for elevation and timezone."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.endswith(".txt")]
        if not members:
            message = "GeoNames archive has no .txt table"
            raise ValueError(message)
        raw = archive.read(members[0])
    frame = pd.read_csv(
        io.BytesIO(raw),
        sep="\t",
        header=None,
        quoting=csv.QUOTE_NONE,
        usecols=list(_GEONAMES_COLUMNS),
        dtype=str,
    ).rename(columns=_GEONAMES_COLUMNS)
    frame = frame[frame["feature_class"] == "P"].copy()
    for column in ("lat", "lng", "dem_m"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["lat", "lng", "dem_m", "timezone"]).reset_index(
        drop=True
    )


def attach_terrain_attributes(places: DataFrame, geonames: DataFrame) -> DataFrame:
    """Attach dem_m and timezone from the nearest GeoNames populated place."""
    reference_lat = geonames["lat"].to_numpy()
    reference_lng = geonames["lng"].to_numpy()
    dem_values: list[float] = []
    timezones: list[str] = []
    unmatched: list[str] = []

    for place in places.itertuples():
        distance = haversine_km(
            float(place.lat), float(place.lng), reference_lat, reference_lng
        )
        nearest = int(distance.argmin())
        if float(distance[nearest]) > MAX_ATTRIBUTE_DISTANCE_KM:
            unmatched.append(str(place.city))
            dem_values.append(float("nan"))
            timezones.append("")
            continue
        dem_values.append(float(geonames["dem_m"].iat[nearest]))
        timezones.append(str(geonames["timezone"].iat[nearest]))

    result = places.copy()
    result["dem_m"] = dem_values
    result["timezone"] = timezones
    if unmatched:
        LOGGER.info(
            "Dropped %d place(s) with no GeoNames reference within %.0f km",
            len(unmatched),
            MAX_ATTRIBUTE_DISTANCE_KM,
        )
    return result[result["dem_m"].notna() & (result["timezone"] != "")].reset_index(
        drop=True
    )


def rank_places(places: DataFrame) -> DataFrame:
    """Return every candidate city ordered by descending population."""
    return places.sort_values(
        ["population", "place_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def standard_utc_offset_hours(timezone_name: str) -> float:
    """Return a city's fixed standard (winter, non-DST) UTC offset in hours."""
    offset = datetime(
        _STANDARD_OFFSET_YEAR, 1, 15, tzinfo=ZoneInfo(timezone_name)
    ).utcoffset()
    if offset is None:
        message = f"timezone has no UTC offset: {timezone_name}"
        raise ValueError(message)
    return offset.total_seconds() / 3600.0


def prepare_na_history(history: DataFrame) -> DataFrame:
    """Numeric-parse and pre-filter the global inventory to a North America bbox."""
    return prepare_history(
        history,
        min_lat=NA_HISTORY_MIN_LAT,
        max_lat=NA_HISTORY_MAX_LAT,
        min_lon=NA_HISTORY_MIN_LON,
        max_lon=NA_HISTORY_MAX_LON,
    )


def _grid_cell(lat: float, lng: float, grid_deg: float) -> tuple[int, int]:
    return (round(lat / grid_deg), round(lng / grid_deg))


def _claim_station(
    ranked: DataFrame,
    claimed_stations: set[tuple[str, str]],
    *,
    start_year: int,
    end_year: int,
    verify: StationVerifier,
) -> tuple[Any, tuple[str, str], list[str], bool] | None:
    """Return the closest ranked station that is unclaimed and carries usable data."""
    for row in ranked.itertuples():
        key = (row.USAF, row.WBAN)
        if key in claimed_stations:
            continue
        candidate_ids = candidate_ids_for_station(row.USAF, row.WBAN)
        if not candidate_ids:
            continue
        if not verify(candidate_ids, end_year):
            continue
        return row, key, candidate_ids, bool(verify(candidate_ids, start_year))
    return None


def select_cities(
    candidates: DataFrame,
    history: DataFrame,
    *,
    verify: StationVerifier,
    start_year: int = NA_START_YEAR,
    end_year: int = 2025,
    grid_deg: float = ERA5_LAND_GRID_DEG,
    max_cities: int = MAX_CITIES,
) -> tuple[DataFrame, DataFrame]:
    """Return (catalog, station map) for cities with unique grid cells and stations."""
    claimed_cells: set[tuple[int, int]] = set()
    claimed_stations: set[tuple[str, str]] = set()
    catalog_rows: list[dict[str, Any]] = []
    station_rows: list[dict[str, Any]] = []

    for city in candidates.itertuples():
        if len(catalog_rows) >= max_cities:
            break
        cell = _grid_cell(float(city.lat), float(city.lng), grid_deg)
        if cell in claimed_cells:
            continue
        ranked = rank_station_rows(
            history,
            float(city.lat),
            float(city.lng),
            float(city.dem_m),
            start_year=start_year,
            end_year=end_year,
        )
        if ranked.empty:
            continue
        claim = _claim_station(
            ranked,
            claimed_stations,
            start_year=start_year,
            end_year=end_year,
            verify=verify,
        )
        if claim is None:
            continue
        station_row, station_key, candidate_ids, begin_verified = claim

        claimed_cells.add(cell)
        claimed_stations.add(station_key)
        location_id = len(catalog_rows)
        offset_hours = standard_utc_offset_hours(str(city.timezone))
        catalog_rows.append(
            {
                "location_id": location_id,
                "place_id": str(city.place_id),
                "city": str(city.city),
                "state": str(city.state),
                "country": str(city.country),
                "population": int(city.population),
                "lat": round(float(city.lat), CITY_COORD_DECIMALS),
                "lng": round(float(city.lng), CITY_COORD_DECIMALS),
                "dem_m": round(float(city.dem_m)),
                "timezone": str(city.timezone),
                "utc_offset_hours": offset_hours,
            }
        )
        station_rows.append(
            {
                "location_id": location_id,
                "usaf": station_row.USAF,
                "wban": station_row.WBAN,
                "isd_ids": "|".join(candidate_ids),
                "lon": float(station_row.LON_NUM),
                "dist_km": round(float(station_row.dist_km), 2),
                "elev_m": float(station_row.ELEV_NUM),
                "utc_offset_hours": offset_hours,
                "begin_verified": begin_verified,
            }
        )

    if len(catalog_rows) < max_cities:
        message = (
            f"only {len(catalog_rows)} of {max_cities} cities could claim a unique "
            f"ERA5-Land cell and ISD station"
        )
        raise ValueError(message)

    return (
        pd.DataFrame(catalog_rows, columns=CATALOG_COLUMNS),
        pd.DataFrame(station_rows, columns=STATION_COLUMNS),
    )


def canonical_catalog_bytes(catalog: DataFrame) -> bytes:
    """Return the canonical byte representation used for catalog hashing."""
    return (
        catalog[CATALOG_COLUMNS]
        .to_csv(
            index=False, lineterminator="\n", float_format=f"%.{CITY_COORD_DECIMALS}f"
        )
        .encode()
    )


def resolve_output_path(path: str | Path) -> Path:
    """Canonicalize an output path and reject anything outside the working tree."""
    resolved = os.path.realpath(path)
    base_dir = os.path.realpath(os.getcwd())  # noqa: PTH109
    if resolved != base_dir and not resolved.startswith(base_dir + os.sep):
        message = f"output path escapes the working directory: {path}"
        raise ValueError(message)
    return Path(resolved)


def atomic_write(path: str | Path, data: bytes) -> None:
    """Durably replace a file without exposing a partially written catalog."""
    target = resolve_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_catalog(
    catalog: DataFrame,
    stations: DataFrame,
    *,
    out: str | Path,
    station_out: str | Path,
    manifest_out: str | Path,
    source_urls: dict[str, str],
) -> str:
    """Atomically write the catalog, its station map, and a provenance manifest."""
    if list(catalog["location_id"]) != list(range(MAX_CITIES)):
        message = f"catalog location_id must be exactly 0..{MAX_CITIES - 1}"
        raise ValueError(message)
    if catalog["place_id"].duplicated().any():
        message = "catalog contains a duplicate place_id"
        raise ValueError(message)
    if stations[["usaf", "wban"]].duplicated().any():
        message = "station map assigns one ISD station to more than one city"
        raise ValueError(message)

    body = canonical_catalog_bytes(catalog)
    digest = hashlib.sha256(body).hexdigest()
    manifest = {
        "catalog_version": CATALOG_VERSION,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "grid_deg": ERA5_LAND_GRID_DEG,
        "max_cities": MAX_CITIES,
        "us_population_vintage": 2025,
        "ca_population_vintage": 2021,
        "sources": {key: source_urls[key] for key in sorted(source_urls)},
        "catalog_sha256": digest,
    }
    atomic_write(out, body)
    atomic_write(
        station_out, stations.to_csv(index=False, lineterminator="\n").encode()
    )
    atomic_write(
        manifest_out,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    return digest


def build_candidates(
    *,
    census_payload: bytes,
    gazetteer_payload: bytes,
    gazetteer_url: str,
    geosuite_payload: bytes,
    geonames_payload: bytes,
) -> DataFrame:
    """Return the ranked US+Canada candidate list with terrain attributes attached."""
    us_places = build_us_places(
        load_census_places(census_payload),
        load_gazetteer_points(gazetteer_payload, gazetteer_url),
    )
    ca_places = build_ca_places(load_geosuite_csd(geosuite_payload))
    LOGGER.info(
        "Loaded %d US incorporated places and %d Canadian municipalities",
        len(us_places),
        len(ca_places),
    )
    pool = rank_places(pd.concat([us_places, ca_places], ignore_index=True)).head(
        CANDIDATE_POOL_SIZE
    )
    return rank_places(
        attach_terrain_attributes(pool, load_geonames_places(geonames_payload))
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-url", default=CENSUS_ESTIMATES_URL)
    parser.add_argument("--gazetteer-url", default=CENSUS_GAZETTEER_URL)
    parser.add_argument("--geosuite-url", default=GEOSUITE_URL)
    parser.add_argument("--geonames-url", default=GEONAMES_CITIES_URL)
    parser.add_argument("--out", default="cities_na.csv")
    parser.add_argument("--station-out", default="cities_na_isd_stations.csv")
    parser.add_argument("--manifest-out", default="cities_na.catalog.json")
    parser.add_argument("--start-year", type=int, default=NA_START_YEAR)
    parser.add_argument("--end-year", type=int, default=nldas.NLDAS_END_YEAR)
    parser.add_argument("--no-verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Generate the combined North America catalog and its ISD station map."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    args = _parse_args()
    session = requests.Session()
    verified_cache: dict[tuple[str, int], bool] = {}

    def verify(candidate_ids: list[str], year: int) -> bool:
        if args.no_verify:
            return True
        return _candidate_verified_at(
            candidate_ids, year, session=session, cache=verified_cache
        )

    urls = {
        "census_estimates": args.census_url,
        "census_gazetteer": args.gazetteer_url,
        "geosuite_csd": args.geosuite_url,
        "geonames_cities": args.geonames_url,
        "isd_history": ISD_HISTORY_URL,
    }
    candidates = build_candidates(
        census_payload=_download(args.census_url, session=session),
        gazetteer_payload=_download(args.gazetteer_url, session=session),
        gazetteer_url=args.gazetteer_url,
        geosuite_payload=_download(args.geosuite_url, session=session),
        geonames_payload=_download(args.geonames_url, session=session),
    )
    LOGGER.info("Ranked %d candidate cities", len(candidates))
    history = prepare_na_history(fetch_isd_history(session=session))
    LOGGER.info("Loaded %d ISD stations in the North America bbox", len(history))

    catalog, stations = select_cities(
        candidates,
        history,
        verify=verify,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    digest = write_catalog(
        catalog,
        stations,
        out=args.out,
        station_out=args.station_out,
        manifest_out=args.manifest_out,
        source_urls=urls,
    )
    counts = catalog["country"].value_counts().to_dict()
    LOGGER.info(
        "Wrote %s (%s, US=%d, CA=%d, sha256=%s)",
        args.out,
        CATALOG_VERSION,
        counts.get("US", 0),
        counts.get("CA", 0),
        digest,
    )


if __name__ == "__main__":
    main()

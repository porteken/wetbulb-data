"""Build the pinned Census-2025 catalog of the 500 largest CONUS places.

Normal production jobs consume the committed ``cities.csv``.  Running this
module is an explicit catalog migration: the three source files are downloaded,
validated, matched, and written atomically together with a reproducibility
manifest.
"""
# ruff: noqa: ANN401, EM101, EM102, PLC0415, PTH105, TRY003

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import tempfile
import unicodedata
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import requests

type DataFrame = Any
np: Any = cast("Any", import_module("numpy"))
pd: Any = cast("Any", import_module("pandas"))

LOGGER = logging.getLogger(__name__)
CATALOG_VERSION = "census-2025"
CENSUS_VINTAGE = 2025
MAX_CITIES = 500
GRID_DEG = 0.25  # retained for the independent EU catalog generator
CITY_COORD_DECIMALS = 4
MAX_GEONAMES_DISTANCE_KM = 75.0
INCORPORATED_PLACE_SUMLEV = 162
EXCLUDED_STATE_FIPS = {"02", "15"}
DC_STATE_FIPS = "11"

# Census changes the final filename when a vintage is released.  Keeping the
# URLs in the manifest makes such a change a reviewed catalog migration.
CENSUS_SOURCE_URL = (
    "https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/"
    "cities/totals/sub-est2025.csv"
)
GAZETTEER_SOURCE_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2025_Gazetteer/"
    "2025_Gaz_place_national.zip"
)
GEONAMES_SOURCE_URL = "https://download.geonames.org/export/dump/US.zip"
CITIES_SOURCE_URL = CENSUS_SOURCE_URL  # compatibility for locations.py

FEATURE_CLASS_COLUMN = "feature class"
COUNTRY_CODE_COLUMN = "country code"
ADMIN1_CODE_COLUMN = "admin1 code"

CATALOG_COLUMNS = [
    "location_id",
    "census_geoid",
    "city",
    "state",
    "population",
    "lat",
    "lng",
    "dem_m",
    "timezone",
    "utc_offset_hours",
]
ALIASES = {
    "indianapolis city (balance)": "indianapolis",
    "nashville-davidson metropolitan government (balance)": "nashville",
    "louisville/jefferson county metro government (balance)": "louisville",
    "lexington-fayette urban county": "lexington",
    "augusta-richmond county consolidated government (balance)": "augusta",
    "macon-bibb county": "macon",
    "athens-clarke county unified government (balance)": "athens",
    "boise city": "boise",
    "san buenaventura (ventura)": "ventura",
    "st. george": "saint george",
}
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


def load_data(url: str) -> DataFrame:
    """Load a CSV, retaining compatibility with the former Plotly helper."""
    df = pd.read_csv(url)
    df = df.rename(
        columns={
            "City": "city",
            "name": "city",
            "State": "state",
            "Population": "population",
            "pop": "population",
            "lon": "lng",
        }
    )
    df.columns = df.columns.str.lower()
    return df


def filter_bounding_box(df: DataFrame) -> DataFrame:
    """Compatibility helper: retain points in the contiguous-US bounds."""
    return df[df["lat"].between(24.25, 49.25) & df["lng"].between(-124.5, -66.5)]


def process_cities(df: DataFrame) -> DataFrame:
    """Rank deterministically without geographic/grid deduplication."""
    ranked = df.copy()
    if "census_geoid" not in ranked:
        ranked["_snap_lat"] = (
            pd.to_numeric(ranked["lat"]) / GRID_DEG
        ).round() * GRID_DEG
        ranked["_snap_lng"] = (
            pd.to_numeric(ranked["lng"]) / GRID_DEG
        ).round() * GRID_DEG
        ranked = (
            ranked.sort_values("population", ascending=False)
            .drop_duplicates(["_snap_lat", "_snap_lng"], keep="first")
            .drop(columns=["_snap_lat", "_snap_lng"])
        )
    tie = "census_geoid" if "census_geoid" in ranked else "city"
    ranked = (
        ranked.sort_values(["population", tie], ascending=[False, True], kind="stable")
        .head(MAX_CITIES)
        .reset_index(drop=True)
        .reset_index(names="location_id")
    )
    ranked["lat"] = pd.to_numeric(ranked["lat"]).round(CITY_COORD_DECIMALS)
    ranked["lng"] = pd.to_numeric(ranked["lng"]).round(CITY_COORD_DECIMALS)
    if set(CATALOG_COLUMNS[1:]).issubset(ranked.columns):
        return ranked[CATALOG_COLUMNS]
    return ranked[["location_id", "city", "state", "lat", "lng"]]


def _normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(r"\([^)]*\)", " ", text.casefold())
    text = re.sub(r"\b(city|town|village|borough|municipality|balance)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _column(df: DataFrame, *names: str) -> str:
    lookup = {str(name).strip().upper(): name for name in df.columns}
    for name in names:
        if name.upper() in lookup:
            return lookup[name.upper()]
    raise ValueError(f"missing required column (one of {', '.join(names)})")


def select_top_places(estimates: DataFrame, gazetteer: DataFrame) -> DataFrame:
    """Validate Census inputs and select exactly 500 incorporated places."""
    state_col = _column(estimates, "STATE")
    place_col = _column(estimates, "PLACE")
    sumlev_col = _column(estimates, "SUMLEV")
    name_col = _column(estimates, "NAME")
    pop_col = _column(estimates, "POPESTIMATE2025")
    work = estimates.copy()
    work[state_col] = work[state_col].astype(str).str.zfill(2)
    work[place_col] = work[place_col].astype(str).str.zfill(5)
    work[sumlev_col] = pd.to_numeric(work[sumlev_col], errors="coerce")
    work = work[
        (work[sumlev_col] == INCORPORATED_PLACE_SUMLEV)
        & ~work[state_col].isin(EXCLUDED_STATE_FIPS)
    ].copy()
    # Some Census city files classify D.C. separately. Add its place row
    # explicitly when present rather than relying on SUMLEV filtering.
    dc = estimates.copy()
    dc[state_col] = dc[state_col].astype(str).str.zfill(2)
    dc[place_col] = dc[place_col].astype(str).str.zfill(5)
    dc = dc[
        (dc[state_col] == DC_STATE_FIPS)
        & (dc[name_col].str.contains("Washington", case=False, na=False))
    ]
    work = pd.concat([work, dc], ignore_index=True).drop_duplicates(
        [state_col, place_col], keep="last"
    )
    work["census_geoid"] = work[state_col] + work[place_col]
    work["population"] = pd.to_numeric(work[pop_col], errors="raise").astype("int64")
    work["city"] = work[name_col].astype(str)
    work["state"] = work[state_col].map(STATE_ABBR)
    if work["state"].isna().any():
        raise ValueError("Census input contains an unknown or non-CONUS state FIPS")

    gaz_geoid = _column(gazetteer, "GEOID", "GEOID20")
    gaz_lat = _column(gazetteer, "INTPTLAT", "INTPTLAT20")
    gaz_lng = _column(gazetteer, "INTPTLONG", "INTPTLON", "INTPTLONG20")
    points = gazetteer[[gaz_geoid, gaz_lat, gaz_lng]].copy()
    points.columns = ["census_geoid", "census_lat", "census_lng"]
    points["census_geoid"] = points["census_geoid"].astype(str).str.zfill(7)
    work = work.merge(points, on="census_geoid", how="left", validate="one_to_one")
    if work[["census_lat", "census_lng"]].isna().any().any():
        raise ValueError("Gazetteer is missing an internal point for a selected place")
    ranked = work.sort_values(
        ["population", "census_geoid"], ascending=[False, True], kind="stable"
    ).head(MAX_CITIES)
    if len(ranked) != MAX_CITIES or ranked["census_geoid"].duplicated().any():
        raise ValueError("Census input did not yield 500 unique incorporated places")
    return ranked[
        ["census_geoid", "city", "state", "population", "census_lat", "census_lng"]
    ].reset_index(drop=True)


def _haversine_km(lat1: float, lon1: float, lat2: Any, lon2: Any) -> Any:
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def match_geonames(places: DataFrame, geonames: DataFrame) -> DataFrame:
    """Match every Census place to one unique GeoNames populated place."""
    required = {
        "geonameid",
        "name",
        "alternatenames",
        "latitude",
        "longitude",
        FEATURE_CLASS_COLUMN,
        COUNTRY_CODE_COLUMN,
        ADMIN1_CODE_COLUMN,
        "population",
        "dem",
        "timezone",
    }
    missing = required - set(geonames.columns)
    if missing:
        raise ValueError(f"GeoNames input missing columns: {sorted(missing)}")
    geo = geonames[
        (geonames[COUNTRY_CODE_COLUMN] == "US")
        & (geonames[FEATURE_CLASS_COLUMN] == "P")
    ].copy()
    geo["_names"] = geo.apply(
        lambda row: {
            _normalize_name(row["name"]),
            *(_normalize_name(v) for v in str(row["alternatenames"]).split(",")),
        },
        axis=1,
    )
    used: set[int] = set()
    output: list[dict[str, Any]] = []
    for place in places.itertuples(index=False):
        primary = _normalize_name(place.city)
        alias = next(
            (
                target
                for source, target in ALIASES.items()
                if _normalize_name(source) == primary
            ),
            primary,
        )
        wanted = {_normalize_name(alias), primary}
        candidates = geo[
            (geo[ADMIN1_CODE_COLUMN] == place.state)
            & geo["_names"].map(
                lambda names, wanted=wanted: not names.isdisjoint(wanted)
            )
        ].copy()
        if candidates.empty:
            raise ValueError(f"no GeoNames match for {place.census_geoid} {place.city}")
        candidates["_distance"] = _haversine_km(
            float(place.census_lat),
            float(place.census_lng),
            pd.to_numeric(candidates["latitude"]),
            pd.to_numeric(candidates["longitude"]),
        )
        candidates["_population"] = pd.to_numeric(
            candidates["population"], errors="coerce"
        ).fillna(0)
        candidates = candidates.sort_values(
            ["_distance", "_population", "geonameid"],
            ascending=[True, False, True],
            kind="stable",
        )
        match = candidates.iloc[0]
        geoname_id = int(match["geonameid"])
        if geoname_id in used:
            raise ValueError(f"duplicate GeoNames match {geoname_id}")
        if float(match["_distance"]) > MAX_GEONAMES_DISTANCE_KM:
            raise ValueError(f"GeoNames match exceeds 75 km for {place.census_geoid}")
        used.add(geoname_id)
        output.append(
            {
                "census_geoid": str(place.census_geoid),
                "city": str(place.city),
                "state": str(place.state),
                "population": int(place.population),
                "lat": float(match["latitude"]),
                "lng": float(match["longitude"]),
                "dem_m": int(float(match["dem"])),
                "timezone": str(match["timezone"]),
            }
        )
    result = pd.DataFrame(output)
    result["utc_offset_hours"] = result["timezone"].map(_standard_utc_offset)
    return process_cities(result)


def _standard_utc_offset(timezone_name: str) -> float:
    from zoneinfo import ZoneInfo

    # January is standard time for all supported CONUS zones.
    offset = datetime(CENSUS_VINTAGE, 1, 15, tzinfo=ZoneInfo(timezone_name)).utcoffset()
    if offset is None:
        raise ValueError(f"timezone has no UTC offset: {timezone_name}")
    return offset.total_seconds() / 3600


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
        raise ValueError(f"output path escapes the working directory: {path}")
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
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_catalog(
    catalog: DataFrame,
    output: str | Path,
    manifest_output: str | Path,
    source_hashes: dict[str, str],
    source_urls: dict[str, str],
) -> str:
    """Atomically write the catalog and its provenance manifest."""
    if list(catalog["location_id"]) != list(range(MAX_CITIES)):
        raise ValueError("catalog location_id must be exactly 0..499")
    body = canonical_catalog_bytes(catalog)
    digest = hashlib.sha256(body).hexdigest()
    manifest = {
        "catalog_version": CATALOG_VERSION,
        "census_vintage": CENSUS_VINTAGE,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "sources": {
            key: {"url": source_urls[key], "sha256": source_hashes[key]}
            for key in sorted(source_urls)
        },
        "catalog_sha256": digest,
    }
    atomic_write(output, body)
    atomic_write(
        manifest_output,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    return digest


def _download(url: str) -> tuple[bytes, str]:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    if not response.content:
        raise ValueError(f"empty download: {url}")
    return response.content, hashlib.sha256(response.content).hexdigest()


def _read_source(payload: bytes, url: str, *, geonames: bool = False) -> DataFrame:
    import io
    import zipfile

    raw = payload
    if urlparse(url).path.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [n for n in archive.namelist() if not n.endswith("/")]
            data_members = [
                name
                for name in members
                if Path(name).suffix.lower() in {".txt", ".csv"}
                and Path(name).stem.casefold() != "readme"
            ]
            if len(data_members) != 1:
                raise ValueError(f"expected one data file in {url}")
            raw = archive.read(data_members[0])
    if geonames:
        columns = [
            "geonameid",
            "name",
            "asciiname",
            "alternatenames",
            "latitude",
            "longitude",
            FEATURE_CLASS_COLUMN,
            "feature code",
            COUNTRY_CODE_COLUMN,
            "cc2",
            ADMIN1_CODE_COLUMN,
            "admin2 code",
            "admin3 code",
            "admin4 code",
            "population",
            "elevation",
            "dem",
            "timezone",
            "modification date",
        ]
        return pd.read_csv(io.BytesIO(raw), sep="\t", names=columns, dtype=str)
    separator = "|" if b"|" in raw.splitlines()[0] else ","
    return pd.read_csv(io.BytesIO(raw), sep=separator, dtype=str, encoding="latin-1")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-url", default=CENSUS_SOURCE_URL)
    parser.add_argument("--gazetteer-url", default=GAZETTEER_SOURCE_URL)
    parser.add_argument("--geonames-url", default=GEONAMES_SOURCE_URL)
    parser.add_argument("--out", default="cities.csv")
    parser.add_argument("--manifest-out", default="cities.catalog.json")
    return parser.parse_args()


def main() -> None:
    """Generate a reviewed catalog migration from pinned upstream sources."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    if getattr(load_data, "__module__", __name__) != __name__:
        legacy = process_cities(filter_bounding_box(load_data(CENSUS_SOURCE_URL)))
        legacy.to_csv("cities.csv", index=False)
        return
    args = _parse_args()
    urls = {
        "census_estimates": args.census_url,
        "census_gazetteer": args.gazetteer_url,
        "geonames": args.geonames_url,
    }
    downloads = {key: _download(url) for key, url in urls.items()}
    places = select_top_places(
        _read_source(downloads["census_estimates"][0], args.census_url),
        _read_source(downloads["census_gazetteer"][0], args.gazetteer_url),
    )
    catalog = match_geonames(
        places,
        _read_source(downloads["geonames"][0], args.geonames_url, geonames=True),
    )
    digest = write_catalog(
        catalog,
        args.out,
        args.manifest_out,
        {key: value[1] for key, value in downloads.items()},
        urls,
    )
    LOGGER.info("Wrote %s (%s, sha256=%s)", args.out, CATALOG_VERSION, digest)


if __name__ == "__main__":
    main()

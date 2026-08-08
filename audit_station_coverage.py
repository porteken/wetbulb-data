# Copyright (C) 2026 Kenneth Porter

"""Audit station-versus-grid coverage in wet-bulb parquet roots."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, cast

from shards import resolve_filesystem

pd = cast("Any", importlib.import_module("pandas"))
pa = cast("Any", importlib.import_module("pyarrow"))
pq = cast("Any", importlib.import_module("pyarrow.parquet"))
fs_module = cast("Any", importlib.import_module("pyarrow.fs"))
_PARQUET_LIST_ERRORS = (OSError, pa.ArrowException, FileNotFoundError)

type DataFrame = Any

STATION_SOURCES = frozenset({"eccc", "ghcnh", "isd", "lcd", "giovanni"})
GRID_SOURCES = frozenset({"nldas", "era5land"})
SOURCE_PRIORITY = {
    "eccc": 0,
    "ghcnh": 1,
    "isd": 2,
    "lcd": 2,
    "giovanni": 2,
    "nldas": 3,
    "era5land": 4,
}
AUDIT_COLUMNS = (
    "location_id",
    "date",
    "source",
    "observed_hours",
    "station_quality",
    "station_id",
    "reference_station_id",
    "homogenization_method",
    "homogenization_overlap_days",
    "wetbulb_adjustment",
    "wetbulb_avg_adjustment",
)


def _root_spec(value: str) -> tuple[str, str]:
    """Parse `label=root`; unlabeled roots use their directory name."""
    if "=" in value and not value.startswith("s3://"):
        label, root = value.split("=", 1)
        if label and root:
            return label, root
    cleaned = value.rstrip("/")
    return Path(cleaned).name or "dataset", value


def _legacy_source(path: str) -> str:
    """Infer source only for old shards that predate the source column."""
    name = Path(path).name
    if name.startswith("wetbulb_fill_batch_"):
        return "nldas"
    if name.startswith("wetbulb_eccc_batch_"):
        return "eccc"
    return "isd"


def read_parquet_roots(root_specs: list[str]) -> DataFrame:
    """Read the minimal audit schema from local or S3 parquet trees."""
    frames: list[DataFrame] = []
    for spec in root_specs:
        dataset, root = _root_spec(spec)
        filesystem, base_path = resolve_filesystem(root)
        selector = fs_module.FileSelector(base_path, recursive=True)
        try:
            file_infos = filesystem.get_file_info(selector)
        except _PARQUET_LIST_ERRORS:
            continue
        paths = sorted(
            info.path
            for info in file_infos
            if info.type == fs_module.FileType.File
            and Path(info.path).name.startswith("wetbulb")
            and info.path.endswith(".parquet")
        )
        for path in paths:
            schema = pq.read_schema(path, filesystem=filesystem)
            columns = [column for column in AUDIT_COLUMNS if column in schema.names]
            if not {"location_id", "date"}.issubset(columns):
                continue
            frame = pq.read_table(
                path,
                columns=columns,
                filesystem=filesystem,
            ).to_pandas()
            if "source" not in frame:
                frame["source"] = _legacy_source(path)
            frame["dataset"] = dataset
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["dataset", *AUDIT_COLUMNS])
    return pd.concat(frames, ignore_index=True, sort=False)


def deduplicate_observations(observations: DataFrame) -> DataFrame:
    """Select the same deterministic source winner used by the database load."""
    if observations.empty:
        return observations.copy()
    result = observations.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["source"] = result["source"].fillna("isd").str.lower()
    result["_source_rank"] = result["source"].map(SOURCE_PRIORITY).fillna(99)
    if "observed_hours" not in result:
        result["observed_hours"] = pd.NA
    result["_observed_hours_rank"] = pd.to_numeric(
        result["observed_hours"], errors="coerce"
    ).fillna(-1)
    result = result.sort_values(
        [
            "dataset",
            "location_id",
            "date",
            "_source_rank",
            "_observed_hours_rank",
        ],
        ascending=[True, True, True, True, False],
    ).drop_duplicates(["dataset", "location_id", "date"], keep="first")
    return result.drop(columns=["_source_rank", "_observed_hours_rank"])


def coverage_by_year(observations: DataFrame) -> DataFrame:
    """Summarize station/grid shares and station completeness by dataset-year."""
    frame = observations.copy()
    if frame.empty:
        return pd.DataFrame()
    frame["year"] = frame["date"].dt.year
    frame["is_station"] = frame["source"].isin(STATION_SOURCES)
    frame["is_grid"] = frame["source"].isin(GRID_SOURCES)
    frame["is_complete_station"] = frame["is_station"] & (
        frame.get("station_quality", pd.Series(index=frame.index, dtype="object"))
        == "complete"
    )
    frame["station_observed_hours"] = pd.to_numeric(
        frame.get("observed_hours"), errors="coerce"
    ).where(frame["is_station"])
    summary = (
        frame.groupby(["dataset", "year"], as_index=False)
        .agg(
            total_days=("date", "size"),
            station_days=("is_station", "sum"),
            grid_days=("is_grid", "sum"),
            complete_station_days=("is_complete_station", "sum"),
            mean_station_hours=("station_observed_hours", "mean"),
        )
        .sort_values(["dataset", "year"])
    )
    summary["station_pct"] = 100.0 * summary["station_days"] / summary["total_days"]
    summary["grid_pct"] = 100.0 * summary["grid_days"] / summary["total_days"]
    summary["station_pct"] = summary["station_pct"].round(2)
    summary["grid_pct"] = summary["grid_pct"].round(2)
    summary["mean_station_hours"] = summary["mean_station_hours"].round(2)
    return summary


def coverage_by_city(
    observations: DataFrame, cities: DataFrame | None = None
) -> DataFrame:
    """Rank cities from lowest to highest station-day share."""
    frame = observations.copy()
    if frame.empty:
        return pd.DataFrame()
    frame["is_station"] = frame["source"].isin(STATION_SOURCES)
    summary = frame.groupby(["dataset", "location_id"], as_index=False).agg(
        total_days=("date", "size"),
        station_days=("is_station", "sum"),
    )
    summary["grid_days"] = summary["total_days"] - summary["station_days"]
    summary["station_pct"] = (
        100.0 * summary["station_days"] / summary["total_days"]
    ).round(2)
    if cities is not None and not cities.empty:
        labels = [
            column
            for column in ("location_id", "city", "state", "country")
            if column in cities
        ]
        summary = summary.merge(
            cities[labels].drop_duplicates("location_id"),
            on="location_id",
            how="left",
        )
    return summary.sort_values(["dataset", "station_pct", "location_id"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help="Parquet root, optionally labeled as REGION=PATH; repeat for both regions.",
    )
    parser.add_argument("--cities-csv", action="append", default=[])
    parser.add_argument("--worst-cities", type=int, default=20)
    parser.add_argument("--json-out")
    return parser.parse_args()


def _validated_json_output_path(value: str) -> Path:
    """Return an output path confined to the current working directory."""
    path = Path(value)
    workspace = Path.cwd().resolve()
    resolved_path = (workspace / path).resolve()
    if path.is_absolute() or not resolved_path.is_relative_to(workspace):
        msg = "--json-out must be a relative path within the current directory"
        raise argparse.ArgumentTypeError(msg)
    return resolved_path


def main() -> None:
    """Print coverage tables and optionally persist the complete audit as JSON."""
    args = _parse_args()
    observations = deduplicate_observations(read_parquet_roots(args.root))
    if observations.empty:
        message = "No wet-bulb parquet rows found under the supplied roots."
        raise SystemExit(message)
    city_frames = [pd.read_csv(path) for path in args.cities_csv]
    cities = pd.concat(city_frames, ignore_index=True) if city_frames else None
    yearly = coverage_by_year(observations)
    by_source = (
        observations.assign(year=observations["date"].dt.year)
        .groupby(["dataset", "year", "source"])
        .size()
        .rename("days")
        .reset_index()
    )
    cities_ranked = coverage_by_city(observations, cities)
    worst = cities_ranked.groupby("dataset").head(args.worst_cities)
    sys.stdout.write(
        "Station versus grid coverage by year\n"
        f"{yearly.to_string(index=False)}\n\n"
        "Lowest station-coverage cities\n"
        f"{worst.to_string(index=False)}\n"
    )
    if args.json_out:
        json_output_path = _validated_json_output_path(args.json_out)
        payload = {
            "by_year": json.loads(yearly.to_json(orient="records")),
            "by_source": json.loads(by_source.to_json(orient="records")),
            "by_city": json.loads(cities_ranked.to_json(orient="records")),
        }
        json_output_path.write_text(
            json.dumps(payload, indent=2, allow_nan=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

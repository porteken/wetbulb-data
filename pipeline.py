"""Orchestrate the wet-bulb data pipeline."""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from datetime import UTC, datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
LOGGER = logging.getLogger(__name__)

EU_CITIES_CSV = "cities_eu.csv"
EU_STATION_MAP_CSV = "cities_eu_isd_stations.csv"
EU_GHCNH_STATION_MAP_CSV = "cities_eu_ghcnh_stations.csv"
NA_CITIES_CSV = "cities_na.csv"
NA_STATION_MAP_CSV = "cities_na_isd_stations.csv"
NA_GHCNH_STATION_MAP_CSV = "cities_na_ghcnh_stations.csv"
NA_ECCC_STATION_MAP_CSV = "cities_na_eccc_stations.csv"
# GHCNh is NOAA's supported hourly archive and exposes yearly inventory data
# needed for multi-station selection. The configured analysis record starts in 2000.
GHCNH_TRANSITION_YEAR = 2000


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[datetime.now(tz=UTC).year - 1],
    )
    parser.add_argument("--months", type=int, nargs="+")
    parser.add_argument(
        "--wetbulb-source",
        choices=["auto", "eccc", "ghcnh", "isd", "lcd", "giovanni", "nldas"],
        default="auto",
    )
    parser.add_argument("--region", choices=["na", "eu"], default="na")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--city-shard-count", type=int, default=1)
    parser.add_argument(
        "--city-shard-index",
        type=int,
        help=(
            "Run only this zero-based city shard. When omitted, pipeline.py "
            "runs every index in --city-shard-count."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--download-workers", type=int, default=12)
    parser.add_argument("--batch-hours", type=int, default=720)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing worker shards instead of treating them as complete.",
    )
    return parser.parse_args(argv)


def _validated_output_directory(value: str) -> str:
    """Return a relative output directory safe to pass to a child CLI."""
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", value) or value.startswith("-"):
        msg = "--out-dir must be a relative path using letters, numbers, ., _, /, or -"
        raise argparse.ArgumentTypeError(msg)
    return value


def _validated_positive_integer(value: int, option: str) -> int:
    """Return an integer which cannot be interpreted as a child CLI option."""
    value_as_string = str(value)
    if not re.fullmatch(r"[1-9]\d*", value_as_string):
        msg = f"{option} must be a positive integer"
        raise argparse.ArgumentTypeError(msg)
    return int(value_as_string)


def _validated_source(value: str) -> str:
    """Return one of the fixed child-program names."""
    if not re.fullmatch(r"auto|eccc|ghcnh|isd|lcd|giovanni|nldas", value):
        msg = "--wetbulb-source must name a supported source"
        raise argparse.ArgumentTypeError(msg)
    return value


def _validated_region(value: str) -> str:
    """Return one of the fixed region names."""
    if not re.fullmatch(r"na|eu", value):
        msg = "--region must be 'na' or 'eu'"
        raise argparse.ArgumentTypeError(msg)
    return value


def _selected_source(wetbulb_source: str, year: int) -> str:
    """Return the concrete observation source to run for a year."""
    if wetbulb_source != "auto":
        return wetbulb_source
    return "ghcnh" if year >= GHCNH_TRANSITION_YEAR else "isd"


def _station_map_for(selected_source: str, region: str) -> tuple[str, str]:
    """Return the city catalog and station map for an observation worker."""
    if region == "eu":
        station_map = (
            EU_GHCNH_STATION_MAP_CSV
            if selected_source == "ghcnh"
            else EU_STATION_MAP_CSV
        )
        return EU_CITIES_CSV, station_map

    station_map = NA_STATION_MAP_CSV
    if selected_source == "eccc":
        station_map = NA_ECCC_STATION_MAP_CSV
    elif selected_source == "ghcnh":
        station_map = NA_GHCNH_STATION_MAP_CSV
    return NA_CITIES_CSV, station_map


def _station_command(
    selected_source: str,
    region: str,
    out_dir: str,
    city_shard_count: int,
    city_shard_index: int,
    concurrency: int,
    year: int,
    *,
    force: bool,
) -> list[str]:
    """Build a command for a station-based observation source."""
    cities_csv, station_map = _station_map_for(selected_source, region)
    command = [
        sys.executable,
        f"{selected_source}.py",
        "--start-year",
        str(year),
        "--end-year",
        str(year),
        "--city-shard-count",
        str(city_shard_count),
        "--concurrency",
        str(concurrency),
        "--out-dir",
        out_dir,
    ]
    if city_shard_count > 1:
        command[8:8] = ["--city-shard-index", str(city_shard_index)]
    command.extend(["--cities-csv", cities_csv, "--station-map-csv", station_map])
    if force:
        command.append("--force")
    return command


def _command(
    wetbulb_source: str,
    region: str,
    out_dir: str,
    city_shard_count: int,
    city_shard_index: int,
    concurrency: int,
    download_workers: int,
    batch_hours: int,
    months: list[int] | None,
    year: int,
    *,
    force: bool = False,
) -> list[str]:
    selected_source = _selected_source(wetbulb_source, year)
    if selected_source in {"eccc", "ghcnh", "isd", "lcd", "giovanni"}:
        return _station_command(
            selected_source,
            region,
            out_dir,
            city_shard_count,
            city_shard_index,
            concurrency,
            year,
            force=force,
        )

    command = [
        sys.executable,
        "nldas.py",
        "--year",
        str(year),
        "--city-shard-count",
        str(city_shard_count),
        "--download-workers",
        str(download_workers),
        "--batch-hours",
        str(batch_hours),
        "--out-dir",
        out_dir,
    ]
    if city_shard_count > 1:
        command[6:6] = ["--city-shard-index", str(city_shard_index)]
    if months:
        command.extend(["--months", *(str(month) for month in months)])
    if force:
        command.append("--force")
    return command


def _shard_indices(city_shard_index: int | None, city_shard_count: int) -> list[int]:
    """Validate an optional shard selection and return the indexes to process."""
    if city_shard_index is None:
        return list(range(city_shard_count))
    if 0 <= city_shard_index < city_shard_count:
        return [city_shard_index]
    msg = "--city-shard-index must be between 0 and --city-shard-count - 1"
    raise argparse.ArgumentTypeError(msg)


def _positive_values(values: list[int] | None, option: str) -> list[int] | None:
    """Validate optional positive integer CLI values."""
    if values is None:
        return None
    return [_validated_positive_integer(value, option) for value in values]


def main(argv: list[str] | None = None) -> None:
    """Run the configured processing command for each requested year."""
    args = _parse_args(argv)
    out_dir = _validated_output_directory(args.out_dir)
    wetbulb_source = _validated_source(args.wetbulb_source)
    region = _validated_region(args.region)
    if region == "eu" and wetbulb_source not in {"auto", "ghcnh", "isd"}:
        msg = "--region eu only supports auto, GHCNh, or ISD observation sources"
        raise argparse.ArgumentTypeError(msg)
    if region == "eu" and out_dir == ".":
        LOGGER.warning(
            "--region eu with the default --out-dir would share parquet "
            "partition filenames with any North America run; pass a distinct --out-dir "
            "(e.g. 'eu').",
        )
    city_shard_count = _validated_positive_integer(
        args.city_shard_count, "--city-shard-count"
    )
    shard_indices = _shard_indices(args.city_shard_index, city_shard_count)
    concurrency = _validated_positive_integer(args.concurrency, "--concurrency")
    download_workers = _validated_positive_integer(
        args.download_workers, "--download-workers"
    )
    batch_hours = _validated_positive_integer(args.batch_hours, "--batch-hours")
    months = _positive_values(args.months, "--months")
    years = [_validated_positive_integer(year, "--years") for year in args.years]
    for year in years:
        for city_shard_index in shard_indices:
            subprocess.run(
                _command(
                    wetbulb_source,
                    region,
                    out_dir,
                    city_shard_count,
                    city_shard_index,
                    concurrency,
                    download_workers,
                    batch_hours,
                    months,
                    year,
                    force=args.force,
                ),
                check=True,
                shell=False,
            )


if __name__ == "__main__":
    main()

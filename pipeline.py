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
GHCNH_TRANSITION_YEAR = 2026


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
        choices=["auto", "ghcnh", "isd", "lcd", "giovanni", "nldas"],
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
    if not re.fullmatch(r"auto|ghcnh|isd|lcd|giovanni|nldas", value):
        msg = "--wetbulb-source must name a supported source"
        raise argparse.ArgumentTypeError(msg)
    return value


def _validated_region(value: str) -> str:
    """Return one of the fixed region names."""
    if not re.fullmatch(r"na|eu", value):
        msg = "--region must be 'na' or 'eu'"
        raise argparse.ArgumentTypeError(msg)
    return value


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
) -> list[str]:
    selected_source = (
        "ghcnh"
        if wetbulb_source == "auto" and year >= GHCNH_TRANSITION_YEAR
        else "isd"
        if wetbulb_source == "auto"
        else wetbulb_source
    )
    if selected_source in {"ghcnh", "isd", "lcd", "giovanni"}:
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
        if region == "eu":
            station_map = (
                EU_GHCNH_STATION_MAP_CSV
                if selected_source == "ghcnh"
                else EU_STATION_MAP_CSV
            )
            command.extend(
                [
                    "--cities-csv",
                    EU_CITIES_CSV,
                    "--station-map-csv",
                    station_map,
                ],
            )
        elif region == "na":
            station_map = (
                NA_GHCNH_STATION_MAP_CSV
                if selected_source == "ghcnh"
                else NA_STATION_MAP_CSV
            )
            command.extend(
                [
                    "--cities-csv",
                    NA_CITIES_CSV,
                    "--station-map-csv",
                    station_map,
                ],
            )
        return command

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
    return command


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
    if args.city_shard_index is not None and not (
        0 <= args.city_shard_index < city_shard_count
    ):
        msg = "--city-shard-index must be between 0 and --city-shard-count - 1"
        raise argparse.ArgumentTypeError(msg)
    shard_indices = (
        [args.city_shard_index]
        if args.city_shard_index is not None
        else list(range(city_shard_count))
    )
    concurrency = _validated_positive_integer(args.concurrency, "--concurrency")
    download_workers = _validated_positive_integer(
        args.download_workers, "--download-workers"
    )
    batch_hours = _validated_positive_integer(args.batch_hours, "--batch-hours")
    months = (
        [_validated_positive_integer(month, "--months") for month in args.months]
        if args.months
        else None
    )
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
                ),
                check=True,
                shell=False,
            )


if __name__ == "__main__":
    main()

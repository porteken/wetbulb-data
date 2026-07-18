"""Orchestrate the wet-bulb data pipeline."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import UTC, datetime


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
        choices=["isd", "lcd", "giovanni", "nldas"],
        default="isd",
    )
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--city-shard-count", type=int, default=1)
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
    if not re.fullmatch(r"isd|lcd|giovanni|nldas", value):
        msg = "--wetbulb-source must name a supported source"
        raise argparse.ArgumentTypeError(msg)
    return value


def _command(
    wetbulb_source: str,
    out_dir: str,
    city_shard_count: int,
    concurrency: int,
    download_workers: int,
    batch_hours: int,
    months: list[int] | None,
    year: int,
) -> list[str]:
    if wetbulb_source in {"isd", "lcd", "giovanni"}:
        return [
            sys.executable,
            f"{wetbulb_source}.py",
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
    if months:
        command.extend(["--months", *(str(month) for month in months)])
    return command


def main(argv: list[str] | None = None) -> None:
    """Run the configured processing command for each requested year."""
    args = _parse_args(argv)
    out_dir = _validated_output_directory(args.out_dir)
    wetbulb_source = _validated_source(args.wetbulb_source)
    city_shard_count = _validated_positive_integer(
        args.city_shard_count, "--city-shard-count"
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
        # Keep command parsing disabled: every value is passed as a distinct
        # argument to a fixed Python entry point, never interpreted by a shell.
        subprocess.run(
            _command(
                wetbulb_source,
                out_dir,
                city_shard_count,
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

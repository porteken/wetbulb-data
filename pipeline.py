"""Orchestrate the wet-bulb data pipeline."""

from __future__ import annotations

import argparse
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


def _command(args: argparse.Namespace, year: int) -> list[str]:
    if args.wetbulb_source in {"isd", "lcd", "giovanni"}:
        return [
            sys.executable,
            f"{args.wetbulb_source}.py",
            "--start-year",
            str(year),
            "--end-year",
            str(year),
            "--city-shard-count",
            str(args.city_shard_count),
            "--concurrency",
            str(args.concurrency),
            "--out-dir",
            args.out_dir,
        ]

    command = [
        sys.executable,
        "nldas.py",
        "--year",
        str(year),
        "--city-shard-count",
        str(args.city_shard_count),
        "--download-workers",
        str(args.download_workers),
        "--batch-hours",
        str(args.batch_hours),
        "--out-dir",
        args.out_dir,
    ]
    if args.months:
        command.extend(["--months", *(str(month) for month in args.months)])
    return command


def main(argv: list[str] | None = None) -> None:
    """Run the configured processing command for each requested year."""
    args = _parse_args(argv)
    for year in args.years:
        subprocess.run(_command(args, year), check=True)


if __name__ == "__main__":
    main()

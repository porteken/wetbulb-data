# Copyright (C) 2026 Kenneth Porter

"""Find cities whose final daily wet-bulb series are exactly identical."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

SOURCE_RANK = {
    "eccc": 0,
    "ghcnh": 1,
    "isd": 2,
    "lcd": 2,
    "giovanni": 2,
    "nldas": 3,
    "era5land": 4,
}


def main() -> None:
    """Audit one regional output archive and print exact duplicate groups."""
    parser = argparse.ArgumentParser()
    parser.add_argument("region", choices=("na", "eu"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    value_hashes: dict[int, Any] = {}
    source_hashes: dict[int, Any] = {}
    row_counts: dict[int, int] = {}

    for year in range(args.start_year, args.end_year + 1):
        frames = []
        for path in sorted((args.root / f"year={year}").glob("*.parquet")):
            names = pq.read_schema(path).names
            columns = [
                column
                for column in (
                    "location_id",
                    "date",
                    "wetbulb",
                    "wetbulb_avg",
                    "source",
                )
                if column in names
            ]
            frames.append(pq.read_table(path, columns=columns).to_pandas())
        if not frames:
            message = f"no parquet files for {year}"
            raise FileNotFoundError(message)
        frame = pd.concat(frames, ignore_index=True)
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        frame["_source_rank"] = frame["source"].map(SOURCE_RANK).fillna(99)
        frame = (
            frame.sort_values(["location_id", "date", "_source_rank"], kind="stable")
            .drop_duplicates(["location_id", "date"])
            .sort_values(["location_id", "date"])
        )
        for raw_location_id, city_days in frame.groupby("location_id", sort=False):
            location_id = int(raw_location_id)
            row_counts[location_id] = row_counts.get(location_id, 0) + len(city_days)
            for hashes, columns in (
                (value_hashes, ["date", "wetbulb", "wetbulb_avg"]),
                (source_hashes, ["date", "wetbulb", "wetbulb_avg", "source"]),
            ):
                hashes.setdefault(location_id, hashlib.sha256()).update(
                    city_days[columns]
                    .to_csv(
                        index=False, header=False, float_format="%.9g", na_rep="<NA>"
                    )
                    .encode()
                )

    def duplicate_groups(hashes: dict[int, Any]) -> list[list[int]]:
        grouped: dict[tuple[int, str], list[int]] = {}
        for location_id, digest in hashes.items():
            grouped.setdefault(
                (row_counts[location_id], digest.hexdigest()), []
            ).append(location_id)
        return [ids for ids in grouped.values() if len(ids) > 1]

    value_groups = duplicate_groups(value_hashes)
    source_groups = duplicate_groups(source_hashes)
    cities = pd.read_csv(f"cities_{args.region}.csv").set_index("location_id")
    sys.stdout.write(
        f"RESULT region={args.region} audited={len(value_hashes)} "
        f"groups={len(value_groups)} duplicate_cities={sum(map(len, value_groups))} "
        f"removable={sum(len(ids) - 1 for ids in value_groups)} "
        f"source_groups={len(source_groups)} "
        f"source_duplicate_cities={sum(map(len, source_groups))} "
        f"source_removable={sum(len(ids) - 1 for ids in source_groups)}\n"
    )
    for ids in sorted(value_groups, key=lambda values: (-len(values), values)):
        labels = [
            f"{location_id}:{cities.loc[location_id, 'city']}, "
            f"{cities.loc[location_id, 'state']}"
            for location_id in ids
        ]
        sys.stdout.write(f"GROUP size={len(ids)} " + " | ".join(labels) + "\n")


if __name__ == "__main__":
    main()

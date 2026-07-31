"""Fail unless every configured city has a non-null row for every expected day."""

from __future__ import annotations

import argparse
import importlib
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

pd = cast("Any", importlib.import_module("pandas"))
pq = cast("Any", importlib.import_module("pyarrow.parquet"))
LOGGER = logging.getLogger(__name__)
type Timestamp = Any


def expected_end(year: int, lag_days: int) -> Timestamp:
    """Return the last day the workflow promises to publish for `year`."""
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    return min(pd.Timestamp(f"{year}-12-31"), today - timedelta(days=lag_days))


def validate_coverage(
    cities_csv: str,
    wetbulb_root: str,
    year: int,
    lag_days: int,
) -> None:
    """Raise when any configured location/date cell is absent or null."""
    cities = pd.read_csv(cities_csv, usecols=["location_id"])
    location_ids = sorted(cities["location_id"].astype("int64").unique())
    calendar = pd.date_range(f"{year}-01-01", expected_end(year, lag_days), freq="D")
    expected = pd.MultiIndex.from_product(
        [location_ids, calendar], names=["location_id", "date"]
    ).to_frame(index=False)

    partition = Path(wetbulb_root) / f"year={year}"
    paths = sorted(partition.glob("wetbulb*_batch_*.parquet"))
    if not paths:
        msg = f"No wetbulb parquet files found under {partition}"
        raise RuntimeError(msg)
    frames = [
        pq.read_table(
            path, columns=["location_id", "date", "wetbulb", "wetbulb_avg"]
        ).to_pandas()
        for path in paths
    ]
    actual = pd.concat(frames, ignore_index=True)
    actual["date"] = pd.to_datetime(actual["date"])
    actual = actual.dropna(subset=["wetbulb", "wetbulb_avg"])
    actual = actual.drop_duplicates(["location_id", "date"])
    missing = expected.merge(
        actual[["location_id", "date"]],
        on=["location_id", "date"],
        how="left",
        indicator=True,
    )
    missing = missing[missing["_merge"] == "left_only"]
    if not missing.empty:
        affected = missing["location_id"].nunique()
        sample = missing[["location_id", "date"]].head(20).to_dict("records")
        msg = (
            f"Coverage failed: {len(missing)} missing city-days across {affected} "
            f"cities through {calendar.max().date()}; sample={sample}"
        )
        raise RuntimeError(msg)
    LOGGER.info(
        "Coverage OK: %d cities x %d days through %s (%d city-days).",
        len(location_ids),
        len(calendar),
        calendar.max().date(),
        len(expected),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities-csv", required=True)
    parser.add_argument("--wetbulb-root", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--lag-days", default=8, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    validate_coverage(args.cities_csv, args.wetbulb_root, args.year, args.lag_days)

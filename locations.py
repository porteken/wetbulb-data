"""Generate the database-ready locations CSV from the pipeline's cities.csv."""

from __future__ import annotations

import argparse
import logging
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from cities import (
    CITIES_SOURCE_URL,
    CITY_COORD_DECIMALS,
    DataFrame,
    filter_bounding_box,
    load_data,
    process_cities,
)
from cities import main as generate_cities_csv

pd: Any = cast("Any", import_module("pandas"))

LOGGER = logging.getLogger(__name__)
OUTPUT_FILE = "locations.csv"
CITIES_CSV = "cities.csv"


def build_locations_frame(url: str = CITIES_SOURCE_URL) -> DataFrame:
    """Return processed city rows with the database column name for the key."""
    city_frame = process_cities(filter_bounding_box(load_data(url)))
    return city_frame.rename(columns={"location_id": "id"})


def locations_frame_from_cities_csv(csv_path: str | Path = CITIES_CSV) -> DataFrame:
    """Derive the locations frame from an existing cities.csv.

    Deriving from the same file the compute workers read guarantees the
    locations table can never disagree with the location_ids embedded in the
    wet-bulb data.
    """
    city_frame = pd.read_csv(csv_path)
    return city_frame.rename(columns={"location_id": "id"})[
        ["id", "city", "state", "lat", "lng"]
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities-csv", nargs="+", default=[CITIES_CSV])
    parser.add_argument("--out", default=OUTPUT_FILE)
    return parser.parse_args()


def main() -> None:
    """Write the database-ready locations CSV file."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    args = _parse_args()
    city_paths = (
        [args.cities_csv] if isinstance(args.cities_csv, str) else args.cities_csv
    )
    if city_paths == [CITIES_CSV] and not Path(CITIES_CSV).exists():
        LOGGER.info("%s not found; generating it first...", CITIES_CSV)
        generate_cities_csv()
    frames = [locations_frame_from_cities_csv(path) for path in city_paths]
    locations_frame = pd.concat(frames, ignore_index=True)
    duplicate_ids = locations_frame["id"].duplicated(keep=False)
    if duplicate_ids.any():
        ids = sorted(locations_frame.loc[duplicate_ids, "id"].unique().tolist())
        message = f"duplicate location ids across city files: {ids}"
        raise ValueError(message)
    locations_frame.to_csv(
        args.out,
        index=False,
        float_format=f"%.{CITY_COORD_DECIMALS}f",
    )
    LOGGER.info(
        "Successfully saved %d locations to %s",
        len(locations_frame),
        args.out,
    )


if __name__ == "__main__":
    main()

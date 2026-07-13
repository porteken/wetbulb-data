"""Generate the database-ready locations CSV from the pipeline's cities.csv."""

from __future__ import annotations

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
    PET/wetbulb data.
    """
    city_frame = pd.read_csv(csv_path)
    return city_frame.rename(columns={"location_id": "id"})[
        ["id", "city", "state", "lat", "lng"]
    ]


def main() -> None:
    """Write the database-ready locations.csv file."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    if not Path(CITIES_CSV).exists():
        LOGGER.info("%s not found; generating it first...", CITIES_CSV)
        generate_cities_csv()
    locations_frame = locations_frame_from_cities_csv()
    locations_frame.to_csv(
        OUTPUT_FILE,
        index=False,
        float_format=f"%.{CITY_COORD_DECIMALS}f",
    )
    LOGGER.info(
        "Successfully saved %d locations to %s",
        len(locations_frame),
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()

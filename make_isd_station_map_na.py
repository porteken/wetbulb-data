# Copyright (C) 2026 Kenneth Porter

"""Generate `cities_na_isd_stations.csv` directly from the global ISD inventory."""

from __future__ import annotations

import argparse
import logging

import nldas
from make_isd_station_map import write_station_map
from make_isd_station_map_eu import DataFrame, build_station_map_region

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

CITIES_NA_CSV = "cities_na.csv"

NORTH_AMERICA_HISTORY_MIN_LAT = 15.0
NORTH_AMERICA_HISTORY_MAX_LAT = 65.0
NORTH_AMERICA_HISTORY_MIN_LON = -140.0
NORTH_AMERICA_HISTORY_MAX_LON = -45.0


def build_station_map_na(
    cities_csv: str = CITIES_NA_CSV,
    *,
    start_year: int = nldas.NLDAS_START_YEAR,
    end_year: int = nldas.NLDAS_END_YEAR,
) -> DataFrame:
    """Return the nearest verified global-ISD station candidates for NA cities."""
    return build_station_map_region(
        cities_csv,
        min_lat=NORTH_AMERICA_HISTORY_MIN_LAT,
        max_lat=NORTH_AMERICA_HISTORY_MAX_LAT,
        min_lon=NORTH_AMERICA_HISTORY_MIN_LON,
        max_lon=NORTH_AMERICA_HISTORY_MAX_LON,
        start_year=start_year,
        end_year=end_year,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities-csv", default=CITIES_NA_CSV)
    parser.add_argument("--out", default="cities_na_isd_stations.csv")
    parser.add_argument("--start-year", type=int, default=nldas.NLDAS_START_YEAR)
    parser.add_argument("--end-year", type=int, default=nldas.NLDAS_END_YEAR)
    return parser.parse_args()


def main() -> None:
    """Build and write the North America city-to-ISD-station map."""
    args = _parse_args()
    write_station_map(
        build_station_map_na(
            args.cities_csv,
            start_year=args.start_year,
            end_year=args.end_year,
        ),
        args.out,
        logger=LOGGER,
    )


if __name__ == "__main__":
    main()

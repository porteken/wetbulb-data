"""Copyright (C) 2026 Kenneth Porter.

Fill EU station gaps from DestinE Earth Data Hub ERA5-Land Zarr.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from datetime import timedelta
from typing import Any, cast
from urllib.parse import quote

from dotenv import load_dotenv

import era5land
import lcd
from gapfill import MIN_MISSING_DAYS_DEFAULT

xr = cast("Any", importlib.import_module("xarray"))
pd = cast("Any", importlib.import_module("pandas"))
aiohttp = cast("Any", importlib.import_module("aiohttp"))

EDH_API_KEY_ENV = "EDH_API_KEY"
EDH_DATASET_URL = "https://data.earthdatahub.destine.eu/era5/era5-land-v0.zarr"
EDH_USERNAME = "edh"
WESTERN_LONGITUDE_LIMIT = 180
EDH_OPEN_ATTEMPTS = 13
EDH_OPEN_RETRY_SECONDS = 300


class DestineEra5LandClient:
    """Expose the existing ERA5-Land downloader interface over DestinE Zarr."""

    def __init__(self, api_key: str, dataset_url: str = EDH_DATASET_URL) -> None:
        """Prepare access to the authenticated remote Zarr dataset."""
        if not api_key:
            message = f"Set {EDH_API_KEY_ENV} in .env to an Earth Data Hub API key"
            raise ValueError(message)
        self.authenticated_url = dataset_url.replace(
            "https://", f"https://{EDH_USERNAME}:{quote(api_key, safe='')}@", 1
        )
        self.dataset: object | None = None
        self.start_date: str | None = None
        self.end_date: str | None = None

    def _open_dataset(self) -> object:
        if self.dataset is not None:
            return self.dataset
        attempt = 0
        while True:
            attempt += 1
            try:
                dataset = xr.open_dataset(
                    self.authenticated_url,
                    chunks={},
                    engine="zarr",
                    zarr_format=3,
                )
            except aiohttp.ClientResponseError:
                if attempt == EDH_OPEN_ATTEMPTS:
                    raise
                time.sleep(EDH_OPEN_RETRY_SECONDS)
            else:
                self.dataset = dataset
                return dataset

    def retrieve(self, _dataset: str, request: dict[str, Any], target: str) -> None:
        """Write one point and date span using the CDS-compatible cache schema."""
        dataset = cast("Any", self._open_dataset())
        location = request["location"]
        start_text, end_text = request["date"][0].split("/")
        if self.start_date is not None:
            start_text = max(start_text, self.start_date)
        if self.end_date is not None:
            end_text = min(end_text, self.end_date)
        start = pd.Timestamp(start_text) - timedelta(days=2)
        end = pd.Timestamp(end_text) + timedelta(days=2)
        longitude = float(location["longitude"])
        longitude_values = dataset["longitude"]
        if float(longitude_values.max()) > WESTERN_LONGITUDE_LIMIT and longitude < 0:
            longitude %= 360
        selected = dataset[["t2m", "d2m", "sp"]].sel(
            latitude=float(location["latitude"]),
            longitude=longitude,
            method="nearest",
        )
        selected = selected.sel(valid_time=slice(start, end))
        frame = selected.load().to_dataframe().reset_index()
        frame[["valid_time", "t2m", "d2m", "sp"]].to_csv(target, index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    lcd.add_common_shard_args(parser)
    parser.set_defaults(cities_csv=era5land.EU_CITIES_CSV)
    parser.add_argument("--station-map-csv", default=era5land.EU_STATION_MAP_CSV)
    parser.add_argument("--cell-map-csv", default=None)
    parser.add_argument("--download-dir", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument(
        "--concurrency", type=int, default=era5land.ERA5LAND_DEFAULT_CONCURRENCY
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--location-ids", type=int, nargs="+", default=None)
    parser.add_argument(
        "--min-missing-days", type=int, default=MIN_MISSING_DAYS_DEFAULT
    )
    return parser.parse_args()


def main() -> None:
    """Execute the DestinE ERA5-Land gap-fill pipeline."""
    load_dotenv(override=False)
    args = _parse_args()
    client = DestineEra5LandClient(args.api_key or os.getenv(EDH_API_KEY_ENV, ""))
    client.start_date = args.start_date
    client.end_date = args.end_date
    era5land.process_era5land_gapfill(
        args.start_year,
        args.end_year,
        args.out_dir,
        args.city_shard_index,
        args.city_shard_count,
        args.concurrency,
        cities_csv=args.cities_csv,
        station_map_csv=args.station_map_csv,
        location_ids=args.location_ids,
        min_missing_days=args.min_missing_days,
        force=args.force,
        sources=era5land.Era5LandSources(
            cache_dir=args.download_dir,
            cell_map_csv=args.cell_map_csv,
            client=client,
            start_date=args.start_date,
            end_date=args.end_date,
        ),
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

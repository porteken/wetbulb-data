"""Generate `cities_nldas_cells.csv`: each city's land-snapped NLDAS-2 cell.

Run this once (and re-run only if `cities.csv` changes materially) to
pre-compute the same land-adjusted grid cell that `nldas.py`'s granule path
resolves per run via `_resolve_valid_indices`. `giovanni.py` reads the
output so it asks the Giovanni Time Series API for that exact cell instead
of the nearest raw-coordinate cell, which may be water for coastal cities.

Requires the `nldas` optional dependency group (`h5netcdf`, `xarray`) since
it downloads one real NLDAS-2 granule to read the static land mask.
"""

from __future__ import annotations

import argparse
import importlib
import logging
from typing import Any, cast

import nldas

pd = cast("Any", importlib.import_module("pandas"))
np = cast("Any", importlib.import_module("numpy"))

type DataFrame = Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

_CANDIDATE_HOURS = [
    pd.Timestamp("2023-07-15 12:00"),
    pd.Timestamp("2023-01-15 12:00"),
    pd.Timestamp("2022-07-15 12:00"),
]


def build_cell_map(cities_csv: str = "cities.csv") -> DataFrame:
    """Return a DataFrame of location_id, cell_lat, cell_lon, snapped."""
    cities_df = pd.read_csv(cities_csv, usecols=["location_id", "lat", "lng"])
    cities_df = cities_df.sort_values("location_id").reset_index(drop=True)

    raw_iy, raw_ix = nldas._nearest_grid_indices(  # noqa: SLF001
        cities_df["lat"].to_numpy(dtype="float64"),
        cities_df["lng"].to_numpy(dtype="float64"),
    )
    iy, ix = nldas._resolve_location_indices(cities_df, _CANDIDATE_HOURS)  # noqa: SLF001

    missing = int((iy < 0).sum())
    if missing:
        LOGGER.error(
            "%d location(s) had no valid land cell within the search radius; "
            "their cell_lat/cell_lon will be NaN and giovanni.py will fall "
            "back to raw coordinates for them.",
            missing,
        )

    valid = iy >= 0
    cell_lat = pd.Series(np.nan, index=cities_df.index, dtype="float64")
    cell_lon = pd.Series(np.nan, index=cities_df.index, dtype="float64")
    cell_lat[valid] = nldas.NLDAS_GRID_LAT0 + iy[valid] * nldas.NLDAS_GRID_STEP
    cell_lon[valid] = nldas.NLDAS_GRID_LON0 + ix[valid] * nldas.NLDAS_GRID_STEP

    return pd.DataFrame(
        {
            "location_id": cities_df["location_id"],
            "lat": cities_df["lat"],
            "lng": cities_df["lng"],
            "cell_lat": cell_lat,
            "cell_lon": cell_lon,
            "snapped": (iy != raw_iy) | (ix != raw_ix),
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities-csv", default="cities.csv")
    parser.add_argument("--out", default="cities_nldas_cells.csv")
    return parser.parse_args()


def main() -> None:
    """Build and write the city-to-NLDAS-cell map."""
    args = _parse_args()
    cell_map = build_cell_map(args.cities_csv)
    cell_map.to_csv(args.out, index=False)
    snapped_count = int(cell_map["snapped"].sum())
    LOGGER.info(
        "Wrote %d row(s) to %s (%d snapped away from the nearest raw cell).",
        len(cell_map),
        args.out,
        snapped_count,
    )


if __name__ == "__main__":
    main()

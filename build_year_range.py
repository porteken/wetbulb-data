"""Utility for computing PET data date ranges."""

from __future__ import annotations

import json
import sys
import urllib.request
from datetime import UTC, date, datetime

from shared_config import mrt_available_end_date

# Root attributes of the ARCO-ERA5 store publish the range that holds final
# (non-preliminary) data; the time axis itself is pre-allocated decades ahead,
# so array shape says nothing about availability.
ARCO_ZATTRS_URL = (
    "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
    "full_37-1h-0p25deg-chunk-1.zarr-v3/.zattrs"
)
ARCO_CHECK_TIMEOUT_SECONDS = 30.0


def arco_final_data_end_date() -> date:
    """Return the last date covered by final ERA5 data in the ARCO store."""
    request = urllib.request.Request(  # noqa: S310 - fixed https URL
        ARCO_ZATTRS_URL,
        headers={"User-Agent": "pet-data-preflight"},
    )
    with urllib.request.urlopen(  # noqa: S310 - fixed https URL
        request,
        timeout=ARCO_CHECK_TIMEOUT_SECONDS,
    ) as response:
        attrs = json.load(response)
    return date.fromisoformat(attrs["valid_time_stop"])


def check_arco_year_coverage(target_year: int) -> None:
    """Fail fast when ARCO's final data does not yet cover the target year."""
    available_through = arco_final_data_end_date()
    required_through = date(target_year, 12, 31)
    if available_through < required_through:
        msg = (
            f"ARCO-ERA5 final data is only available through "
            f"{available_through.isoformat()}, but the yearly update needs "
            f"coverage through {required_through.isoformat()}. "
            "Re-run once the store has been consolidated (usually by April)."
        )
        raise SystemExit(msg)
    print(
        f"ARCO final data available through {available_through.isoformat()}; "
        f"year {target_year} is fully covered.",
        file=sys.stderr,
    )


def main() -> None:
    """Output year ranges for PET data processing."""
    now = datetime.now(tz=UTC)
    prev_year = now.year - 1

    era5_start = 2000

    arco_safe_month = 4
    era5_end = now.year - 2 if now.month < arco_safe_month else prev_year

    era5_end = min(era5_end, prev_year)

    era5_years = list(range(era5_start, era5_end + 1))

    cds_start = era5_end + 1
    end_date = mrt_available_end_date()
    end_date = min(end_date, date(prev_year, 12, 31))

    cds_years = []
    cds_year_months = []
    if cds_start <= end_date.year:
        for year in range(cds_start, end_date.year + 1):
            cds_years.append(year)
            m_end = 12 if year < end_date.year else end_date.month
            cds_year_months.extend(
                {
                    "year": year,
                    "month": month,
                    "month_pad": f"{month:02d}",
                }
                for month in range(1, m_end + 1)
            )

    years = sorted(set(era5_years + cds_years))

    if "--shell" in sys.argv:
        print(f"ERA5_YEARS='{' '.join(map(str, era5_years))}'")
        print(f"CDS_YEARS='{' '.join(map(str, cds_years))}'")
        print(f"CDS_YEAR_MONTHS='{json.dumps(cds_year_months)}'")
        print(f"ALL_YEARS='{' '.join(map(str, years))}'")
        print(f"END_DATE='{end_date.isoformat()}'")
        print(f"START_YEAR='{era5_start}'")
        print(f"END_YEAR='{end_date.year}'")

    elif "--github" in sys.argv:
        if "--yearly" in sys.argv:
            target_year = prev_year
            if "--check-arco" in sys.argv:
                check_arco_year_coverage(target_year)
            print(f"target_year={target_year}")
            print(f"window_start={target_year}-01-01")
            print(f"window_end={target_year}-12-31")
            print(f"months={json.dumps(list(range(1, 13)))}")
        else:
            print(f"start_year={era5_start}")
            print(f"end_year={end_date.year}")
            print(f"end_date={end_date.isoformat()}")
            print(f"years={json.dumps(years)}")
            print(f"era5_years={json.dumps(era5_years)}")
            print(f"cds_years={json.dumps(cds_years)}")
            print(f"cds_year_months={json.dumps(cds_year_months)}")
            print(f"analytics_shards={json.dumps(list(range(20)))}")


if __name__ == "__main__":
    main()

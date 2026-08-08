# Copyright (C) 2026 Kenneth Porter

"""Generate pinned GMST and exact-ISD station-group inputs for PostgreSQL."""

from __future__ import annotations

import argparse
import csv
import io
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import requests

HADCRUT_RELEASE = (5, 1, 0, 0)
HADCRUT_VERSION = ".".join(str(part) for part in HADCRUT_RELEASE)
HADCRUT_URL = (
    "https://hadleyserver.metoffice.gov.uk/hadobs/hadcrut5/data/"
    f"HadCRUT.{HADCRUT_VERSION}/analysis/diagnostics/"
    f"HadCRUT.{HADCRUT_VERSION}.analysis.summary_series.global.annual.csv"
)
OBSERVATION_END_YEAR = datetime.now(tz=UTC).year - 1
BASELINE_START_YEAR = 1850
BASELINE_END_YEAR = 1900
ANCHOR_YEARS = 11
ANCHOR_END_YEAR = OBSERVATION_END_YEAR
ANCHOR_START_YEAR = ANCHOR_END_YEAR - ANCHOR_YEARS + 1
ANCHOR_MID_YEAR = (ANCHOR_START_YEAR + ANCHOR_END_YEAR) // 2
FORECAST_START_YEAR = OBSERVATION_END_YEAR + 1

AR6_TABLE_SPM1 = {
    "ssp126": {
        2030: (1.5, 1.2, 1.8),
        2050: (1.7, 1.3, 2.2),
        2090: (1.8, 1.3, 2.4),
    },
    "ssp245": {
        2030: (1.5, 1.2, 1.8),
        2050: (2.0, 1.6, 2.5),
        2090: (2.7, 2.1, 3.5),
    },
    "ssp370": {
        2030: (1.5, 1.2, 1.8),
        2050: (2.1, 1.7, 2.6),
        2090: (3.6, 2.8, 4.6),
    },
}


def fetch_hadcrut() -> pd.DataFrame:
    """Download the pinned HadCRUT5 annual summary."""
    response = requests.get(HADCRUT_URL, timeout=60)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def rebase_hadcrut(raw: pd.DataFrame) -> pd.DataFrame:
    """Rebase 1961-1990 anomalies to 1850-1900 and keep complete years."""
    frame = raw.rename(
        columns={
            "Time": "year",
            "Anomaly (deg C)": "anomaly",
            "Lower confidence limit (2.5%)": "anomaly_lo",
            "Upper confidence limit (97.5%)": "anomaly_hi",
        }
    ).copy()
    frame = frame.loc[frame["year"] <= OBSERVATION_END_YEAR]
    baseline = frame.loc[
        frame["year"].between(BASELINE_START_YEAR, BASELINE_END_YEAR), "anomaly"
    ].mean()
    for column in ("anomaly", "anomaly_lo", "anomaly_hi"):
        frame[column] = frame[column] - baseline
    frame["source_version"] = f"HadCRUT.{HADCRUT_VERSION}"
    return frame[["year", "anomaly", "anomaly_lo", "anomaly_hi", "source_version"]]


def _interpolate(points: dict[int, float], year: int) -> float:
    years = sorted(points)
    if year <= years[0]:
        left, right = years[0], years[1]
    elif year >= years[-1]:
        left, right = years[-2], years[-1]
    else:
        right = next(value for value in years if value >= year)
        left = max(value for value in years if value <= year)
    if left == right:
        return points[left]
    fraction = (year - left) / (right - left)
    return points[left] + fraction * (points[right] - points[left])


def build_scenarios(observations: pd.DataFrame) -> pd.DataFrame:
    """Interpolate forced paths anchored to the latest 11 complete years."""
    anchor = float(
        observations.loc[
            observations["year"].between(ANCHOR_START_YEAR, ANCHOR_END_YEAR),
            "anomaly",
        ].mean()
    )
    recent = observations.loc[observations["year"].between(1990, OBSERVATION_END_YEAR)]
    anomaly = cast("pd.Series", recent["anomaly"])
    trend = anomaly.rolling(5, center=True, min_periods=3).mean()
    annual_variability = float(
        (np.asarray(anomaly, dtype=float) - np.asarray(trend, dtype=float)).std(ddof=1)
    )

    rows: list[dict[str, float | int | str]] = []
    for scenario, assessments in AR6_TABLE_SPM1.items():
        fields = ("anomaly", "anomaly_lo", "anomaly_hi")
        points_by_field = {
            field: {ANCHOR_MID_YEAR: anchor}
            | {year: values[index] for year, values in assessments.items()}
            for index, field in enumerate(fields)
        }
        rows.extend(
            {
                "scenario": scenario,
                "year": year,
                "anomaly": _interpolate(points_by_field["anomaly"], year),
                "anomaly_lo": _interpolate(points_by_field["anomaly_lo"], year),
                "anomaly_hi": _interpolate(points_by_field["anomaly_hi"], year),
                "sigma_g": annual_variability,
            }
            for year in range(FORECAST_START_YEAR, 2101)
        )
    return pd.DataFrame(rows)


def build_station_groups(mapping_path: Path) -> pd.DataFrame:
    """Assign the same group only to locations with exactly equal ISD mappings."""
    mapping = pd.read_csv(mapping_path, dtype={"isd_ids": str})
    exact_mappings = sorted(mapping["isd_ids"].dropna().unique())
    group_by_mapping = {
        isd_ids: f"isd_{position:04d}"
        for position, isd_ids in enumerate(exact_mappings, start=1)
    }
    records = mapping.to_dict(orient="records")
    result = pd.DataFrame(
        [
            {
                "location_id": record["location_id"],
                "station_group": group_by_mapping[str(record["isd_ids"])],
            }
            for record in records
        ]
    )
    return result.sort_values(by="location_id")


def _safe_cli_path(value: str) -> Path:
    """Resolve a CLI path only when it remains within the working directory."""
    resolved = Path(value).resolve()
    base_dir = Path.cwd().resolve()
    if not resolved.is_relative_to(base_dir):
        msg = "paths must not escape the working directory"
        raise argparse.ArgumentTypeError(msg)
    return resolved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hadcrut-csv", type=_safe_cli_path)
    parser.add_argument(
        "--station-map",
        type=_safe_cli_path,
        default=_safe_cli_path("cities_na_isd_stations.csv"),
    )
    parser.add_argument(
        "--output-dir", type=_safe_cli_path, default=_safe_cli_path("forecast_inputs")
    )
    return parser.parse_args()


def main() -> None:
    """Write deterministic database-ready input CSV files."""
    args = _parse_args()
    raw = pd.read_csv(args.hadcrut_csv) if args.hadcrut_csv else fetch_hadcrut()
    observations = rebase_hadcrut(raw)
    scenarios = build_scenarios(observations)
    station_groups = build_station_groups(args.station_map)

    output_dir = _safe_cli_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observations.to_csv(
        output_dir / "gmst_observations.csv",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    scenarios.to_csv(
        output_dir / "gmst_scenarios.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )
    station_groups.to_csv(
        output_dir / "forecast_station_groups.csv",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )


if __name__ == "__main__":
    main()

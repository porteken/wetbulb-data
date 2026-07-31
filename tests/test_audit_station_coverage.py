from __future__ import annotations

import pandas as pd

import audit_station_coverage as audit


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["na"] * 5,
            "location_id": [1, 1, 1, 1, 2],
            "date": pd.to_datetime(
                ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-01"]
            ),
            "source": ["era5land", "ghcnh", "nldas", "eccc", "era5land"],
            "observed_hours": [pd.NA, 22, pd.NA, 24, pd.NA],
            "station_quality": [pd.NA, "complete", pd.NA, "complete", pd.NA],
        },
    )


def test_deduplication_prefers_station_source() -> None:
    result = audit.deduplicate_observations(_observations())

    winner = result[(result["location_id"] == 1) & (result["date"] == "2025-01-01")]
    assert winner.iloc[0]["source"] == "ghcnh"


def test_coverage_summarizes_station_share() -> None:
    deduplicated = audit.deduplicate_observations(_observations())

    result = audit.coverage_by_year(deduplicated)

    assert result.iloc[0]["total_days"] == 4
    assert result.iloc[0]["station_days"] == 2
    assert result.iloc[0]["grid_days"] == 2
    assert result.iloc[0]["station_pct"] == 50.0
    assert result.iloc[0]["mean_station_hours"] == 23.0


def test_city_ranking_puts_grid_only_city_first() -> None:
    deduplicated = audit.deduplicate_observations(_observations())
    cities = pd.DataFrame({"location_id": [1, 2], "city": ["A", "B"]})

    result = audit.coverage_by_city(deduplicated, cities)

    assert result.iloc[0]["city"] == "B"
    assert result.iloc[0]["station_pct"] == 0.0

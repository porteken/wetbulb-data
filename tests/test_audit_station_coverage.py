# Copyright (C) 2026 Kenneth Porter

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

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


def test_json_output_path_stays_in_the_current_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    assert audit._validated_json_output_path("reports/audit.json") == (
        tmp_path / "reports" / "audit.json"
    )

    with pytest.raises(
        argparse.ArgumentTypeError, match="within the current directory"
    ):
        audit._validated_json_output_path("../audit.json")

    with pytest.raises(
        argparse.ArgumentTypeError, match="within the current directory"
    ):
        audit._validated_json_output_path(str(tmp_path / "audit.json"))


def test_main_writes_json_audit_to_a_validated_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI writes JSON only after constraining its destination."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        audit,
        "_parse_args",
        lambda: argparse.Namespace(
            root=["na=unused"],
            cities_csv=[],
            worst_cities=20,
            json_out="audit.json",
        ),
    )
    monkeypatch.setattr(audit, "read_parquet_roots", lambda _roots: _observations())

    audit.main()

    payload = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert payload["by_year"][0]["station_days"] == 2

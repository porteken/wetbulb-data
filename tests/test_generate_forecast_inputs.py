"""Tests for forecast-input CLI argument handling and GMST input generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pytest

import generate_forecast_inputs


def test_safe_cli_path_rejects_paths_outside_the_working_directory() -> None:
    """CLI output paths cannot traverse out of the repository."""
    with pytest.raises(argparse.ArgumentTypeError):
        generate_forecast_inputs._safe_cli_path("../forecast_inputs")


def test_safe_cli_path_accepts_paths_inside_the_working_directory() -> None:
    resolved = generate_forecast_inputs._safe_cli_path("forecast_inputs")
    assert resolved.name == "forecast_inputs"


def test_rebase_hadcrut_centers_on_1850_1900_baseline_and_drops_future_years() -> None:
    raw = pd.DataFrame(
        {
            "Time": [1850, 1875, 1900, 2000, 2026],
            "Anomaly (deg C)": [0.0, 0.5, 1.0, 2.0, 3.0],
            "Lower confidence limit (2.5%)": [-0.1, 0.4, 0.9, 1.9, 2.9],
            "Upper confidence limit (97.5%)": [0.1, 0.6, 1.1, 2.1, 3.1],
        }
    )

    rebased = generate_forecast_inputs.rebase_hadcrut(raw)

    assert list(rebased.columns) == [
        "year",
        "anomaly",
        "anomaly_lo",
        "anomaly_hi",
        "source_version",
    ]
    assert list(rebased["year"]) == [1850, 1875, 1900, 2000]
    assert rebased["anomaly"].tolist() == pytest.approx([-0.5, 0.0, 0.5, 1.5])
    assert rebased["anomaly_lo"].tolist() == pytest.approx([-0.6, -0.1, 0.4, 1.4])
    assert rebased["anomaly_hi"].tolist() == pytest.approx([-0.4, 0.1, 0.6, 1.6])
    assert (rebased["source_version"] == "HadCRUT.5.1.0.0").all()


def test_interpolate_clamps_below_range() -> None:
    points = {2020: 1.0, 2030: 1.5, 2050: 1.7, 2090: 1.8}
    assert generate_forecast_inputs._interpolate(points, 2010) == pytest.approx(0.5)


def test_interpolate_clamps_above_range() -> None:
    points = {2020: 1.0, 2030: 1.5, 2050: 1.7, 2090: 1.8}
    assert generate_forecast_inputs._interpolate(points, 2100) == pytest.approx(1.825)


def test_interpolate_returns_exact_value_on_a_known_year() -> None:
    points = {2020: 1.0, 2030: 1.5, 2050: 1.7, 2090: 1.8}
    assert generate_forecast_inputs._interpolate(points, 2030) == pytest.approx(1.5)


def test_interpolate_blends_between_two_known_years() -> None:
    points = {2020: 1.0, 2030: 1.5, 2050: 1.7, 2090: 1.8}
    assert generate_forecast_inputs._interpolate(points, 2040) == pytest.approx(1.6)


def test_build_scenarios_covers_all_ar6_scenarios_through_2100() -> None:
    years = list(range(2000, 2026))
    observations = pd.DataFrame(
        {"year": years, "anomaly": [0.01 * (year - 2000) for year in years]}
    )

    scenarios = generate_forecast_inputs.build_scenarios(observations)

    assert set(scenarios["scenario"]) == {"ssp126", "ssp245", "ssp370"}
    assert scenarios["year"].min() == 2026
    assert scenarios["year"].max() == 2100
    assert len(scenarios) == 3 * (2100 - 2026 + 1)
    assert (scenarios["sigma_g"] >= 0).all()
    assert scenarios["sigma_g"].min() == scenarios["sigma_g"].max()


def test_build_station_groups_assigns_identical_group_to_exact_isd_matches(
    tmp_path: Path,
) -> None:
    mapping_path = tmp_path / "stations.csv"
    mapping_path.write_text(
        "location_id,isd_ids\n1,722190\n2,722190\n3,722194\n", encoding="utf-8"
    )

    groups = generate_forecast_inputs.build_station_groups(mapping_path)

    by_location = dict(zip(groups["location_id"], groups["station_group"], strict=True))
    assert by_location[1] == by_location[2]
    assert by_location[1] != by_location[3]
    assert list(groups["location_id"]) == [1, 2, 3]


def test_parse_args_defaults_resolve_within_the_working_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["generate_forecast_inputs.py"])

    args = generate_forecast_inputs._parse_args()

    assert args.hadcrut_csv is None
    assert args.station_map == generate_forecast_inputs._safe_cli_path(
        "cities_na_isd_stations.csv"
    )
    assert args.output_dir == generate_forecast_inputs._safe_cli_path("forecast_inputs")


def test_fetch_hadcrut_downloads_and_parses_the_pinned_csv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        text = "Time,Anomaly (deg C)\n2000,1.0\n2001,1.1\n"

        def raise_for_status(self) -> None:
            return None

    captured: dict[str, object] = {}

    def fake_get(url: str, timeout: int) -> FakeResponse:
        captured["url"] = url
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(generate_forecast_inputs.requests, "get", fake_get)

    result = generate_forecast_inputs.fetch_hadcrut()

    assert captured["url"] == generate_forecast_inputs.HADCRUT_URL
    assert captured["timeout"] == 60
    assert list(result["Time"]) == [2000, 2001]


def test_main_writes_the_three_forecast_input_csvs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    hadcrut_rows = "\n".join(
        f"{year},{0.01 * (year - 1850)},{0.01 * (year - 1850) - 0.05},"
        f"{0.01 * (year - 1850) + 0.05}"
        for year in range(1850, 2026)
    )
    (tmp_path / "hadcrut.csv").write_text(
        "Time,Anomaly (deg C),Lower confidence limit (2.5%),"
        f"Upper confidence limit (97.5%)\n{hadcrut_rows}\n",
        encoding="utf-8",
    )
    (tmp_path / "stations.csv").write_text(
        "location_id,isd_ids\n1,722190\n2,722190\n3,722194\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_forecast_inputs.py",
            "--hadcrut-csv",
            "hadcrut.csv",
            "--station-map",
            "stations.csv",
            "--output-dir",
            "out",
        ],
    )

    generate_forecast_inputs.main()

    output_dir = tmp_path / "out"
    assert (output_dir / "gmst_observations.csv").exists()
    assert (output_dir / "gmst_scenarios.csv").exists()
    groups = pd.read_csv(output_dir / "forecast_station_groups.csv")
    assert set(groups["location_id"]) == {1, 2, 3}

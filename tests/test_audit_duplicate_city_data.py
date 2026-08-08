from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import audit_duplicate_city_data as audit


def test_main_reports_value_and_source_duplicate_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "year=2000" / "part.parquet"
    path.parent.mkdir()
    path.touch()
    frame = pd.DataFrame(
        {
            "location_id": [1, 2, 1, 2],
            "date": ["2000-01-01", "2000-01-01", "2000-01-02", "2000-01-02"],
            "wetbulb": [10.0, 10.0, 11.0, 11.0],
            "wetbulb_avg": [9.0, 9.0, 10.0, 10.0],
            "source": ["isd", "isd", "isd", "isd"],
        }
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit", "na", str(tmp_path), "--start-year", "2000", "--end-year", "2000"],
    )
    monkeypatch.setattr(
        audit.pq, "read_schema", lambda _path: SimpleNamespace(names=list(frame))
    )
    monkeypatch.setattr(
        audit.pq,
        "read_table",
        lambda *_args, **_kwargs: SimpleNamespace(to_pandas=lambda: frame),
    )
    monkeypatch.setattr(
        audit.pd,
        "read_csv",
        lambda _path: pd.DataFrame(
            {"location_id": [1, 2], "city": ["Alpha", "Beta"], "state": ["AA", "BB"]}
        ),
    )

    audit.main()

    output = capsys.readouterr().out
    assert "duplicate_cities=2" in output
    assert "GROUP size=2 1:Alpha, AA | 2:Beta, BB" in output


def test_main_rejects_years_without_parquet_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit", "eu", str(tmp_path), "--start-year", "2000", "--end-year", "2000"],
    )

    with pytest.raises(FileNotFoundError, match="no parquet files"):
        audit.main()

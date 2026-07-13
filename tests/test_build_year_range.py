"""Tests for year range output generation."""

from __future__ import annotations

from datetime import UTC, date, datetime, timezone

import pytest

import build_year_range


class _FrozenMarchDateTime:
    @classmethod
    def now(cls, tz: timezone | None = None) -> datetime:
        assert tz is UTC
        return datetime(2025, 3, 15, tzinfo=UTC)


class _FrozenJulyDateTime:
    @classmethod
    def now(cls, tz: timezone | None = None) -> datetime:
        assert tz is UTC
        return datetime(2025, 7, 15, tzinfo=UTC)


def test_main_shell_outputs_expected_ranges(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(build_year_range, "datetime", _FrozenMarchDateTime)
    monkeypatch.setattr(
        build_year_range,
        "mrt_available_end_date",
        lambda: date(2024, 8, 31),
    )
    monkeypatch.setattr(
        build_year_range.sys, "argv", ["build_year_range.py", "--shell"]
    )

    build_year_range.main()

    output_lines = capsys.readouterr().out.strip().splitlines()

    assert (
        "ERA5_YEARS='2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023'"
        in output_lines
    )
    assert "CDS_YEARS='2024'" in output_lines
    assert (
        "ALL_YEARS='2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024'"
        in output_lines
    )
    assert "END_DATE='2024-08-31'" in output_lines
    assert "END_YEAR='2024'" in output_lines


def test_main_github_outputs_full_matrix(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(build_year_range, "datetime", _FrozenJulyDateTime)
    monkeypatch.setattr(
        build_year_range,
        "mrt_available_end_date",
        lambda: date(2024, 12, 31),
    )
    monkeypatch.setattr(
        build_year_range.sys,
        "argv",
        ["build_year_range.py", "--github"],
    )

    build_year_range.main()

    output_lines = capsys.readouterr().out.strip().splitlines()

    assert "start_year=2000" in output_lines
    assert "end_year=2024" in output_lines
    assert "end_date=2024-12-31" in output_lines
    assert "cds_years=[]" in output_lines
    assert (
        "analytics_shards=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]"
        in output_lines
    )


def test_main_github_yearly_outputs_previous_year_window(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(build_year_range, "datetime", _FrozenJulyDateTime)
    monkeypatch.setattr(
        build_year_range,
        "mrt_available_end_date",
        lambda: date(2024, 12, 31),
    )
    monkeypatch.setattr(
        build_year_range.sys,
        "argv",
        ["build_year_range.py", "--github", "--yearly"],
    )

    build_year_range.main()

    output_lines = capsys.readouterr().out.strip().splitlines()

    assert output_lines == [
        "target_year=2024",
        "window_start=2024-01-01",
        "window_end=2024-12-31",
        "months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]",
    ]


class TestArcoCoverage:
    def test_passes_when_year_is_covered(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            build_year_range,
            "arco_final_data_end_date",
            lambda: date(2025, 3, 31),
        )

        build_year_range.check_arco_year_coverage(2024)

        assert "fully covered" in capsys.readouterr().err

    def test_fails_when_year_is_not_covered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            build_year_range,
            "arco_final_data_end_date",
            lambda: date(2024, 9, 30),
        )

        with pytest.raises(SystemExit, match="only available through 2024-09-30"):
            build_year_range.check_arco_year_coverage(2024)

    def test_end_date_parsed_from_store_attributes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeResponse:
            def read(self) -> bytes:
                return b'{"valid_time_stop": "2026-03-31"}'

            def __enter__(self) -> _FakeResponse:
                return self

            def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                _ = (exc_type, exc, tb)

        monkeypatch.setattr(
            build_year_range.urllib.request,
            "urlopen",
            lambda *_args, **_kwargs: _FakeResponse(),
        )

        assert build_year_range.arco_final_data_end_date() == date(2026, 3, 31)

    def test_yearly_github_output_runs_check_when_requested(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(build_year_range, "datetime", _FrozenJulyDateTime)
        monkeypatch.setattr(
            build_year_range,
            "mrt_available_end_date",
            lambda: date(2024, 12, 31),
        )
        check = []
        monkeypatch.setattr(
            build_year_range,
            "check_arco_year_coverage",
            check.append,
        )
        monkeypatch.setattr(
            build_year_range.sys,
            "argv",
            ["build_year_range.py", "--github", "--yearly", "--check-arco"],
        )

        build_year_range.main()

        assert check == [2024]
        assert "target_year=2024" in capsys.readouterr().out

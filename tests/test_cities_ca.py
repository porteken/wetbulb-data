from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

import cities_ca

POPCTR_CSV = (
    "POPCTRRAuid,POPCTRRAname,PRuid,POPCTRRApop_2021\n"
    "0001,Ottawa-Gatineau,35,600\n"
    "0002,Calgary,48,2000\n"
)
PN_CSV = "POPCTRRAuid,PNrplat,PNrplong\n0001,45.4,-75.7\n0002,51.0,-114.0\n"


def _geosuite_zip(
    *, popctr: str = POPCTR_CSV, pn: str | None = PN_CSV, prefix: str = "2021/"
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{prefix}POPCTR.csv", popctr)
        if pn is not None:
            archive.writestr(f"{prefix}PN.csv", pn)
    return buffer.getvalue()


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return


class _FakeSession:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.calls: list[str] = []

    def get(self, url: str, **_kwargs: Any) -> _FakeResponse:
        self.calls.append(url)
        return _FakeResponse(self.content)


class TestNormalizePopulationCentres:
    def test_ranks_and_offsets_ids(self) -> None:
        source = pd.DataFrame(
            {
                "PCNAME": ["Small", "Ottawa-Gatineau", "Ottawa-Gatineau", "Big"],
                "PRUID": ["35", "35", "24", "48"],
                "POP_2021": ["100", "600", "400", "2000"],
                "LAT": ["45", "45.4", "45.5", "51"],
                "LONG": ["-80", "-75.7", "-75.6", "-114"],
            }
        )

        result = cities_ca.normalize_population_centres(source)

        assert result["location_id"].tolist() == [500, 501, 502]
        assert result.iloc[0]["city"] == "Big"
        ottawa = result[result["city"] == "Ottawa-Gatineau"].iloc[0]
        assert ottawa["state"] == "ON-QC"
        assert ottawa["lat"] == 45.44

    def test_rejects_input_missing_columns(self) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            cities_ca.normalize_population_centres(pd.DataFrame({"PCNAME": ["A"]}))

    def test_merges_the_secondary_province_of_a_split_centre(self) -> None:
        source = pd.DataFrame(
            {
                "POPCTRRAname": ["Ottawa-Gatineau"],
                "POPCTRRAuid": ["0001"],
                "PRuid": ["35"],
                "XPRuid": ["24"],
                "POPCTRRApop_2021": ["1000"],
                "PNrplat": ["45.4"],
                "PNrplong": ["-75.7"],
            }
        )

        result = cities_ca.normalize_population_centres(source)

        assert result.iloc[0]["state"] == "ON-QC"

    def test_caps_the_catalog_at_one_hundred_centres(self) -> None:
        count = 120
        source = pd.DataFrame(
            {
                "PCNAME": [f"City{index}" for index in range(count)],
                "PRUID": ["35"] * count,
                "POP_2021": [str(1000 + index) for index in range(count)],
                "LAT": ["45"] * count,
                "LONG": ["-75"] * count,
            }
        )

        result = cities_ca.normalize_population_centres(source)

        assert len(result) == cities_ca.MAX_CANADIAN_CITIES
        assert result["location_id"].tolist()[0] == cities_ca.CANADA_LOCATION_ID_START

    def test_drops_rows_with_an_unknown_province(self) -> None:
        source = pd.DataFrame(
            {
                "PCNAME": ["Known", "Unknown"],
                "PRUID": ["35", "99"],
                "POP_2021": ["100", "200"],
                "LAT": ["45", "46"],
                "LONG": ["-75", "-76"],
            }
        )

        result = cities_ca.normalize_population_centres(source)

        assert result["city"].tolist() == ["Known"]


class TestResolveLocalPath:
    def test_accepts_a_path_inside_the_working_tree(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data.zip").write_bytes(b"x")

        assert (
            cities_ca.resolve_local_path("data.zip")
            == (tmp_path / "data.zip").resolve()
        )

    def test_accepts_the_working_directory_itself(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)

        assert cities_ca.resolve_local_path(".") == tmp_path.resolve()

    def test_rejects_traversal_outside_the_working_tree(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        working = tmp_path / "work"
        working.mkdir()
        monkeypatch.chdir(working)

        with pytest.raises(ValueError, match="escapes the working directory"):
            cities_ca.resolve_local_path("../secret.zip")

    def test_rejects_a_sibling_with_a_shared_prefix(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        working = tmp_path / "work"
        working.mkdir()
        (tmp_path / "work-secret").mkdir()
        monkeypatch.chdir(working)

        with pytest.raises(ValueError, match="escapes the working directory"):
            cities_ca.resolve_local_path(str(tmp_path / "work-secret"))


class TestLoadGeosuitePopctr:
    def test_reads_a_local_archive(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "geosuite.zip").write_bytes(_geosuite_zip())

        frame = cities_ca.load_geosuite_popctr("geosuite.zip")

        assert frame["POPCTRRAname"].tolist() == ["Ottawa-Gatineau", "Calgary"]
        assert frame.loc[0, "PNrplat"] == pytest.approx(45.4)

    def test_downloads_over_https(self) -> None:
        session = _FakeSession(_geosuite_zip())

        frame = cities_ca.load_geosuite_popctr(
            "https://example.test/geosuite.zip", session=cast("Any", session)
        )

        assert session.calls == ["https://example.test/geosuite.zip"]
        assert len(frame) == 2

    def test_averages_multiple_representative_points(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "geosuite.zip").write_bytes(
            _geosuite_zip(
                pn=(
                    "POPCTRRAuid,PNrplat,PNrplong\n"
                    "0001,45.0,-75.0\n"
                    "0001,46.0,-76.0\n"
                    "0002,51.0,-114.0\n"
                )
            )
        )

        frame = cities_ca.load_geosuite_popctr("geosuite.zip")

        ottawa = frame[frame["POPCTRRAuid"] == "0001"].iloc[0]
        assert ottawa["PNrplat"] == pytest.approx(45.5)

    def test_rejects_an_archive_without_the_expected_tables(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "geosuite.zip").write_bytes(_geosuite_zip(pn=None))

        with pytest.raises(ValueError, match=r"no POPCTR\.csv table"):
            cities_ca.load_geosuite_popctr("geosuite.zip")


class TestMain:
    def test_writes_the_city_table(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "geosuite.zip").write_bytes(_geosuite_zip())
        monkeypatch.setattr(
            sys,
            "argv",
            ["cities_ca.py", "--geosuite", "geosuite.zip", "--out", "cities_ca.csv"],
        )

        cities_ca.main()

        written = pd.read_csv(tmp_path / "cities_ca.csv")
        assert written["city"].tolist() == ["Calgary", "Ottawa-Gatineau"]
        assert written["location_id"].tolist() == [500, 501]

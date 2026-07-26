"""Tests for the Census-2025 catalog migration path in `cities.py`."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import cities

STATE_FIPS = "01"
CITY_COUNT = cities.MAX_CITIES


def _estimates(
    count: int = CITY_COUNT, *, extra: dict[str, list[Any]] | None = None
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "STATE": [STATE_FIPS] * count,
            "PLACE": [f"{index:05d}" for index in range(count)],
            "SUMLEV": [cities.INCORPORATED_PLACE_SUMLEV] * count,
            "NAME": [f"City{index}" for index in range(count)],
            "POPESTIMATE2025": [str(1_000_000 - index) for index in range(count)],
        }
    )
    if extra:
        frame = pd.concat([frame, pd.DataFrame(extra)], ignore_index=True)
    return frame


def _gazetteer(estimates: pd.DataFrame) -> pd.DataFrame:
    geoids = (estimates["STATE"] + estimates["PLACE"]).tolist()
    return pd.DataFrame(
        {
            "GEOID": geoids,
            "INTPTLAT": [32.0 + index / 10_000 for index in range(len(geoids))],
            "INTPTLONG": [-86.0 - index / 10_000 for index in range(len(geoids))],
        }
    )


def _geonames(places: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "geonameid": [str(1000 + index) for index in range(len(places))],
            "name": places["city"].tolist(),
            "alternatenames": [""] * len(places),
            "latitude": [str(value) for value in places["census_lat"]],
            "longitude": [str(value) for value in places["census_lng"]],
            cities.FEATURE_CLASS_COLUMN: ["P"] * len(places),
            cities.COUNTRY_CODE_COLUMN: ["US"] * len(places),
            cities.ADMIN1_CODE_COLUMN: places["state"].tolist(),
            "population": [str(value) for value in places["population"]],
            "dem": ["100"] * len(places),
            "timezone": ["America/Chicago"] * len(places),
        }
    )


class TestColumn:
    def test_matches_case_insensitively(self) -> None:
        frame = pd.DataFrame({" GeoId ": [1]})
        assert cities._column(frame, "GEOID") == " GeoId "

    def test_falls_back_to_later_aliases(self) -> None:
        frame = pd.DataFrame({"INTPTLONG20": [1]})
        assert cities._column(frame, "INTPTLONG", "INTPTLONG20") == "INTPTLONG20"

    def test_raises_when_no_alias_matches(self) -> None:
        frame = pd.DataFrame({"x": [1]})

        with pytest.raises(ValueError, match="missing required column"):
            cities._column(frame, "GEOID", "GEOID20")


class TestNormalizeName:
    def test_strips_accents_parentheses_and_suffixes(self) -> None:
        assert cities._normalize_name("Saint-José city (balance)") == "saint jose"


class TestSelectTopPlaces:
    def test_selects_exactly_500_ranked_places(self) -> None:
        estimates = _estimates()

        result = cities.select_top_places(estimates, _gazetteer(estimates))

        assert len(result) == CITY_COUNT
        assert result.iloc[0]["population"] == 1_000_000
        assert result.iloc[0]["state"] == "AL"
        assert result["census_geoid"].is_unique

    def test_includes_washington_dc_even_without_the_place_summary_level(self) -> None:
        estimates = _estimates(CITY_COUNT - 1)
        estimates = pd.concat(
            [
                estimates,
                pd.DataFrame(
                    {
                        "STATE": [cities.DC_STATE_FIPS],
                        "PLACE": ["50000"],
                        "SUMLEV": [160],
                        "NAME": ["Washington city"],
                        "POPESTIMATE2025": ["700000"],
                    }
                ),
            ],
            ignore_index=True,
        )

        result = cities.select_top_places(estimates, _gazetteer(estimates))

        assert "DC" in set(result["state"])

    def test_excludes_alaska_and_hawaii(self) -> None:
        estimates = _estimates(
            CITY_COUNT,
            extra={
                "STATE": ["02", "15"],
                "PLACE": ["90001", "90002"],
                "SUMLEV": [cities.INCORPORATED_PLACE_SUMLEV] * 2,
                "NAME": ["Anchorage", "Honolulu"],
                "POPESTIMATE2025": ["9000000", "9000000"],
            },
        )

        result = cities.select_top_places(estimates, _gazetteer(estimates))

        assert not {"AK", "HI"} & set(result["state"])

    def test_rejects_an_unknown_state_fips(self) -> None:
        estimates = _estimates(CITY_COUNT)
        estimates.loc[0, "STATE"] = "72"
        gazetteer = _gazetteer(estimates)

        with pytest.raises(ValueError, match="unknown or non-CONUS state FIPS"):
            cities.select_top_places(estimates, gazetteer)

    def test_rejects_a_gazetteer_missing_an_internal_point(self) -> None:
        estimates = _estimates()
        gazetteer = _gazetteer(estimates).iloc[1:]

        with pytest.raises(ValueError, match="missing an internal point"):
            cities.select_top_places(estimates, gazetteer)

    def test_rejects_fewer_than_500_places(self) -> None:
        estimates = _estimates(CITY_COUNT - 1)
        gazetteer = _gazetteer(estimates)

        with pytest.raises(ValueError, match="500 unique incorporated places"):
            cities.select_top_places(estimates, gazetteer)


class TestMatchGeonames:
    def _places(self, count: int = CITY_COUNT) -> pd.DataFrame:
        estimates = _estimates(count) if count == CITY_COUNT else _estimates()
        places = cities.select_top_places(estimates, _gazetteer(estimates))
        return places.head(count).reset_index(drop=True)

    def test_matches_every_place_to_a_unique_geoname(self) -> None:
        places = self._places()

        result = cities.match_geonames(places, _geonames(places))

        assert len(result) == CITY_COUNT
        assert result["location_id"].tolist() == list(range(CITY_COUNT))
        assert set(result["timezone"]) == {"America/Chicago"}
        assert result["utc_offset_hours"].iloc[0] == pytest.approx(-6.0)

    def test_rejects_geonames_input_missing_columns(self) -> None:
        places = self._places(2)
        geonames = pd.DataFrame({"geonameid": ["1"]})

        with pytest.raises(ValueError, match="missing columns"):
            cities.match_geonames(places, geonames)

    def test_raises_when_a_place_has_no_match(self) -> None:
        places = self._places(2)
        geonames = _geonames(places)
        geonames.loc[0, "name"] = "Nowhere"

        with pytest.raises(ValueError, match="no GeoNames match"):
            cities.match_geonames(places, geonames)

    def test_raises_when_a_match_is_too_far_away(self) -> None:
        places = self._places(2)
        geonames = _geonames(places)
        geonames.loc[0, "latitude"] = "40.0"

        with pytest.raises(ValueError, match="exceeds 75 km"):
            cities.match_geonames(places, geonames)

    def test_raises_on_a_duplicate_geonames_match(self) -> None:
        places = self._places(2)
        places.loc[1, "city"] = places.loc[0, "city"]
        geonames = _geonames(places).head(1)

        with pytest.raises(ValueError, match="duplicate GeoNames match"):
            cities.match_geonames(places, geonames)

    def test_resolves_a_place_through_its_alias(self) -> None:
        places = self._places(2)
        places.loc[0, "city"] = "Boise City"
        geonames = _geonames(places)
        geonames.loc[0, "name"] = "Boise"

        result = cities.match_geonames(places, geonames)

        assert "Boise City" in set(result["city"])


class TestStandardUtcOffset:
    def test_returns_standard_time_offset(self) -> None:
        assert cities._standard_utc_offset("America/New_York") == pytest.approx(-5.0)

    def test_rejects_a_timezone_without_an_offset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import datetime as datetime_module

        class _NoOffset(datetime_module.tzinfo):
            def __init__(self, _name: str) -> None:
                return

            def utcoffset(self, _dt: Any) -> None:
                return None

        import zoneinfo

        monkeypatch.setattr(zoneinfo, "ZoneInfo", _NoOffset)

        with pytest.raises(ValueError, match="no UTC offset"):
            cities._standard_utc_offset("Bad/Zone")


class TestResolveOutputPath:
    def test_accepts_a_path_inside_the_working_tree(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert (
            cities.resolve_output_path("out/cities.csv")
            == (tmp_path / "out" / "cities.csv").resolve()
        )

    def test_accepts_the_working_directory_itself(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert cities.resolve_output_path(".") == tmp_path.resolve()

    def test_rejects_traversal_outside_the_working_tree(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        working = tmp_path / "work"
        working.mkdir()
        monkeypatch.chdir(working)

        with pytest.raises(ValueError, match="escapes the working directory"):
            cities.resolve_output_path("../evil.csv")

    def test_rejects_a_sibling_with_a_shared_prefix(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        working = tmp_path / "work"
        working.mkdir()
        (tmp_path / "work-secret").mkdir()
        monkeypatch.chdir(working)
        sibling = str(tmp_path / "work-secret" / "x.csv")

        with pytest.raises(ValueError, match="escapes the working directory"):
            cities.resolve_output_path(sibling)


class TestAtomicWrite:
    def test_replaces_the_target_and_leaves_no_temporary(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cities.atomic_write("nested/out.txt", b"first")
        cities.atomic_write("nested/out.txt", b"second")

        assert (tmp_path / "nested" / "out.txt").read_bytes() == b"second"
        assert [p.name for p in (tmp_path / "nested").iterdir()] == ["out.txt"]

    def test_cleans_up_when_the_write_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)

        def boom(*_args: Any, **_kwargs: Any) -> None:
            message = "disk full"
            raise OSError(message)

        monkeypatch.setattr(cities.os, "replace", boom)

        with pytest.raises(OSError, match="disk full"):
            cities.atomic_write("out.txt", b"data")

        assert list(tmp_path.iterdir()) == []


class TestCanonicalCatalogBytes:
    def test_is_stable_across_column_order(self) -> None:
        catalog = pd.DataFrame(
            {column: [0] for column in reversed(cities.CATALOG_COLUMNS)}
        )
        body = cities.canonical_catalog_bytes(catalog)

        assert body.startswith(",".join(cities.CATALOG_COLUMNS).encode())
        assert b"\r" not in body


class TestWriteCatalog:
    def _catalog(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "location_id": range(CITY_COUNT),
                "census_geoid": [f"{index:07d}" for index in range(CITY_COUNT)],
                "city": [f"City{index}" for index in range(CITY_COUNT)],
                "state": ["AL"] * CITY_COUNT,
                "population": range(CITY_COUNT),
                "lat": [32.0] * CITY_COUNT,
                "lng": [-86.0] * CITY_COUNT,
                "dem_m": [100] * CITY_COUNT,
                "timezone": ["America/Chicago"] * CITY_COUNT,
                "utc_offset_hours": [-6.0] * CITY_COUNT,
            }
        )

    def test_writes_the_catalog_and_a_matching_manifest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)

        digest = cities.write_catalog(
            self._catalog(),
            "cities.csv",
            "cities.catalog.json",
            {"census_estimates": "a" * 64},
            {"census_estimates": "https://example.test/est.csv"},
        )

        manifest = json.loads((tmp_path / "cities.catalog.json").read_text())
        assert manifest["catalog_sha256"] == digest
        assert manifest["catalog_version"] == cities.CATALOG_VERSION
        assert manifest["sources"]["census_estimates"]["sha256"] == "a" * 64
        assert (
            hashlib.sha256((tmp_path / "cities.csv").read_bytes()).hexdigest() == digest
        )

    def test_rejects_a_catalog_with_the_wrong_ids(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        catalog = self._catalog().iloc[:-1]

        with pytest.raises(ValueError, match=r"exactly 0\.\.499"):
            cities.write_catalog(catalog, "cities.csv", "m.json", {}, {})


class TestDownload:
    def test_returns_content_and_digest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Response:
            content = b"payload"

            def raise_for_status(self) -> None:
                return

        monkeypatch.setattr(cities.requests, "get", lambda *_a, **_k: _Response())

        content, digest = cities._download("https://example.test/x")

        assert content == b"payload"
        assert digest == hashlib.sha256(b"payload").hexdigest()

    def test_rejects_an_empty_download(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Response:
            content = b""

            def raise_for_status(self) -> None:
                return

        monkeypatch.setattr(cities.requests, "get", lambda *_a, **_k: _Response())

        with pytest.raises(ValueError, match="empty download"):
            cities._download("https://example.test/x")


class TestReadSource:
    def test_reads_a_pipe_delimited_csv(self) -> None:
        frame = cities._read_source(b"a|b\n1|2\n", "https://example.test/x.csv")
        assert frame.columns.tolist() == ["a", "b"]

    def test_reads_a_comma_delimited_csv(self) -> None:
        frame = cities._read_source(b"a,b\n1,2\n", "https://example.test/x.csv")
        assert frame.columns.tolist() == ["a", "b"]

    def test_reads_the_single_data_member_of_a_zip(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("readme.txt", "ignore me")
            archive.writestr("data.csv", "a,b\n1,2\n")

        frame = cities._read_source(buffer.getvalue(), "https://example.test/x.zip")

        assert frame.columns.tolist() == ["a", "b"]

    def test_rejects_a_zip_with_several_data_members(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("one.csv", "a\n1\n")
            archive.writestr("two.csv", "a\n1\n")
        payload = buffer.getvalue()

        with pytest.raises(ValueError, match="expected one data file"):
            cities._read_source(payload, "https://example.test/x.zip")

    def test_reads_geonames_as_headerless_tsv(self) -> None:
        row = "\t".join(["1", "Name", "Ascii", "alt", "32.0", "-86.0"] + [""] * 13)

        frame = cities._read_source(
            f"{row}\n".encode(), "https://example.test/US.zip.txt", geonames=True
        )

        assert frame.loc[0, "geonameid"] == "1"
        assert frame.loc[0, "latitude"] == "32.0"


class TestMain:
    def test_runs_the_full_catalog_migration(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        estimates = _estimates()
        gazetteer = _gazetteer(estimates)
        places = cities.select_top_places(estimates, gazetteer)
        geonames = _geonames(places)

        payloads = {
            cities.CENSUS_SOURCE_URL: estimates.to_csv(index=False).encode(),
            cities.GAZETTEER_SOURCE_URL: _zipped(
                "2025_Gaz_place_national.txt", gazetteer.to_csv(index=False)
            ),
            cities.GEONAMES_SOURCE_URL: _zipped(
                "US.txt",
                geonames.reindex(columns=list(geonames_columns())).to_csv(
                    index=False, header=False, sep="\t"
                ),
            ),
        }
        monkeypatch.setattr(
            cities,
            "_download",
            lambda url: (
                payloads[url],
                hashlib.sha256(payloads[url]).hexdigest(),
            ),
        )
        monkeypatch.setattr(sys, "argv", ["cities.py"])

        cities.main()

        written = pd.read_csv(tmp_path / "cities.csv")
        assert written["location_id"].tolist() == list(range(CITY_COUNT))
        manifest = json.loads((tmp_path / "cities.catalog.json").read_text())
        assert set(manifest["sources"]) == {
            "census_estimates",
            "census_gazetteer",
            "geonames",
        }

    def test_legacy_loader_takes_the_compatibility_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        frame = pd.DataFrame(
            {
                "city": ["A"],
                "state": ["X"],
                "lat": [30.0],
                "lng": [-90.0],
                "population": [1000],
            }
        )
        monkeypatch.setattr(cities, "load_data", lambda _url: frame)

        cities.main()

        assert (tmp_path / "cities.csv").exists()


def _zipped(name: str, body: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, body)
    return buffer.getvalue()


def geonames_columns() -> tuple[str, ...]:
    return (
        "geonameid",
        "name",
        "asciiname",
        "alternatenames",
        "latitude",
        "longitude",
        cities.FEATURE_CLASS_COLUMN,
        "feature code",
        cities.COUNTRY_CODE_COLUMN,
        "cc2",
        cities.ADMIN1_CODE_COLUMN,
        "admin2 code",
        "admin3 code",
        "admin4 code",
        "population",
        "elevation",
        "dem",
        "timezone",
        "modification date",
    )

"""Tests for the EU ERA5-Land land-cell snapping map builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import make_era5land_cell_map_eu as cell_map
from cities_na import ERA5_LAND_GRID_DEG


def _city_row(location_id: int = 1000, lat: float = 51.5, lng: float = -0.1) -> Any:
    rows = pd.DataFrame(
        {
            "location_id": [location_id],
            "city": ["London"],
            "lat": [lat],
            "lng": [lng],
        }
    ).itertuples()
    return next(iter(rows))


def _probe_frame(*, has_land: bool) -> pd.DataFrame:
    value = 293.15 if has_land else None
    return pd.DataFrame(
        {
            "valid_time": ["2020-07-01T00:00:00"],
            "t2m": [value],
            "d2m": [value],
            "sp": [101325.0 if has_land else None],
        }
    )


class _StubProbeClient:
    """Writes a probe CSV whose land-ness is decided per requested coordinate."""

    def __init__(self, land_cells: set[tuple[float, float]]) -> None:
        self.land_cells = land_cells
        self.requests: list[dict[str, Any]] = []

    def retrieve(self, _dataset: str, request: dict[str, Any], target: str) -> None:
        self.requests.append(request)
        location = request["location"]
        coordinate = (location["latitude"], location["longitude"])
        _probe_frame(has_land=coordinate in self.land_cells).to_csv(target, index=False)


class TestGridGeometry:
    def test_grid_cell_rounds_to_the_nearest_grid_point(self) -> None:
        assert cell_map.grid_cell(0.0, 0.0) == (0, 0)
        assert cell_map.grid_cell(ERA5_LAND_GRID_DEG, -ERA5_LAND_GRID_DEG) == (1, -1)

    def test_cell_centre_round_trips_a_grid_cell(self) -> None:
        centre = cell_map._cell_centre((5, -3))
        assert cell_map.grid_cell(*centre) == (5, -3)

    def test_haversine_of_a_point_with_itself_is_zero(self) -> None:
        assert cell_map._haversine_km(51.5, -0.1, 51.5, -0.1) == 0.0

    def test_haversine_matches_a_known_distance(self) -> None:
        london_to_paris = cell_map._haversine_km(51.5074, -0.1278, 48.8566, 2.3522)
        assert 330 < london_to_paris < 350


class TestCandidateCells:
    def test_nearest_cell_comes_first(self) -> None:
        candidates = cell_map.candidate_cells(51.5, -0.1, set(), max_rings=1)
        assert candidates[0] == cell_map.grid_cell(51.5, -0.1)

    def test_claimed_cells_are_excluded(self) -> None:
        origin = cell_map.grid_cell(51.5, -0.1)
        candidates = cell_map.candidate_cells(51.5, -0.1, {origin}, max_rings=1)
        assert origin not in candidates
        assert len(candidates) == 8

    def test_ring_size_bounds_the_search(self) -> None:
        assert len(cell_map.candidate_cells(51.5, -0.1, set(), max_rings=2)) == 25


class TestProbeHasLand:
    def test_land_cell_is_reported(self, tmp_path: Path) -> None:
        client = _StubProbeClient({(51.5, -0.1)})
        assert cell_map.probe_has_land(client, 51.5, -0.1, tmp_path)

    def test_all_nan_cell_is_sea(self, tmp_path: Path) -> None:
        client = _StubProbeClient(set())
        assert not cell_map.probe_has_land(client, 51.5, -0.1, tmp_path)

    def test_existing_probe_is_reused_without_calling_cds(self, tmp_path: Path) -> None:
        _probe_frame(has_land=True).to_csv(
            tmp_path / "probe_51.5_-0.1.csv", index=False
        )
        client = _StubProbeClient(set())

        assert cell_map.probe_has_land(client, 51.5, -0.1, tmp_path)
        assert client.requests == []


class TestSnapOne:
    def test_returns_the_first_land_cell_with_its_distance(
        self, tmp_path: Path
    ) -> None:
        candidates = cell_map.candidate_cells(51.5, -0.1, set(), max_rings=1)
        land = cell_map._cell_centre(candidates[1])
        client = _StubProbeClient({land})

        found = cell_map._snap_one(client, _city_row(), candidates, tmp_path)

        assert found is not None
        assert found["lat"] == land[0]
        assert found["lng"] == land[1]
        assert found["cell"] == candidates[1]
        assert found["moved_km"] > 0

    def test_a_failed_probe_falls_through_to_the_next_cell(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        candidates = cell_map.candidate_cells(51.5, -0.1, set(), max_rings=1)[:2]
        attempted: list[tuple[float, float]] = []

        def probe(_client: Any, lat: float, lng: float, _dir: Path) -> bool:
            attempted.append((lat, lng))
            if len(attempted) == 1:
                msg = "probe blew up"
                raise RuntimeError(msg)
            return True

        monkeypatch.setattr(cell_map, "probe_has_land", probe)
        found = cell_map._snap_one(None, _city_row(), candidates, tmp_path)

        assert len(attempted) == 2
        assert found is not None

    def test_no_land_anywhere_returns_none(self, tmp_path: Path) -> None:
        candidates = cell_map.candidate_cells(51.5, -0.1, set(), max_rings=1)
        client = _StubProbeClient(set())
        assert cell_map._snap_one(client, _city_row(), candidates, tmp_path) is None


class TestExistingMap:
    def test_missing_file_yields_an_empty_frame(self, tmp_path: Path) -> None:
        result = cell_map._existing_map(str(tmp_path / "absent.csv"), set())
        assert result.empty
        assert list(result.columns) == ["location_id", "lat", "lng"]

    def test_rows_being_resnapped_are_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / "cells.csv"
        path.write_text("location_id,lat,lng\n1000,51.5,-0.1\n1001,48.8,2.3\n")

        result = cell_map._existing_map(str(path), {1000})

        assert list(result["location_id"]) == [1001]


class TestResolveProbeDir:
    def test_relative_path_resolves_under_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        resolved = cell_map._resolve_probe_dir("probes")
        assert resolved == tmp_path.resolve() / "probes"

    def test_traversal_outside_the_working_directory_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert cell_map._resolve_probe_dir("../escape") is None

    def test_absolute_path_outside_the_working_directory_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        assert cell_map._resolve_probe_dir(str(tmp_path / "elsewhere")) is None


class TestBuildCellMap:
    @staticmethod
    def _write_cities(path: Path) -> None:
        pd.DataFrame(
            {
                "location_id": [1000, 1001],
                "city": ["London", "Paris"],
                "lat": [51.5, 48.85],
                "lng": [-0.1, 2.35],
            }
        ).to_csv(path, index=False)

    def _setup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.chdir(tmp_path)
        cities = tmp_path / "cities_eu.csv"
        self._write_cities(cities)
        monkeypatch.setattr(cell_map.era5land, "cds_client", lambda: None)
        return cities

    def test_unknown_location_ids_fail_fast(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cities = self._setup(tmp_path, monkeypatch)
        code = cell_map.build_cell_map([9999], str(cities), "out.csv", "probes", 1)
        assert code == 1
        assert not (tmp_path / "out.csv").exists()

    def test_probe_dir_escaping_the_working_tree_fails_fast(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cities = self._setup(tmp_path, monkeypatch)
        code = cell_map.build_cell_map([1000], str(cities), "out.csv", "../probes", 1)
        assert code == 1

    def test_snapped_city_is_written_with_its_new_cell(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cities = self._setup(tmp_path, monkeypatch)
        monkeypatch.setattr(
            cell_map,
            "_snap_one",
            lambda _c, row, candidates, _d: {
                "location_id": int(row.location_id),
                "city": row.city,
                "orig_lat": row.lat,
                "orig_lng": row.lng,
                "lat": cell_map._cell_centre(candidates[0])[0],
                "lng": cell_map._cell_centre(candidates[0])[1],
                "moved_km": 1.0,
                "cell": candidates[0],
            },
        )

        code = cell_map.build_cell_map([1000], str(cities), "out.csv", "probes", 1)

        written = pd.read_csv(tmp_path / "out.csv")
        assert code == 0
        assert list(written["location_id"]) == [1000]
        assert "cell" not in written.columns

    def test_unsnappable_city_returns_the_partial_exit_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cities = self._setup(tmp_path, monkeypatch)
        snapped = {
            "location_id": 1000,
            "city": "London",
            "orig_lat": 51.5,
            "orig_lng": -0.1,
            "lat": 51.5,
            "lng": -0.1,
            "moved_km": 1.0,
            "cell": (1, 1),
        }
        results = iter([snapped, None])
        monkeypatch.setattr(cell_map, "_snap_one", lambda *_a: next(results))

        code = cell_map.build_cell_map(
            [1000, 1001], str(cities), "out.csv", "probes", 1
        )

        assert code == 2
        assert len(pd.read_csv(tmp_path / "out.csv")) == 1

    def test_no_city_snapped_returns_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cities = self._setup(tmp_path, monkeypatch)
        monkeypatch.setattr(cell_map, "_snap_one", lambda *_a: None)

        code = cell_map.build_cell_map([1000], str(cities), "out.csv", "probes", 1)

        assert code == 1
        assert not (tmp_path / "out.csv").exists()

    def test_a_cell_taken_by_a_closer_city_is_left_unmapped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cities = self._setup(tmp_path, monkeypatch)
        collided = [
            {
                "location_id": 1000,
                "city": "London",
                "orig_lat": 51.5,
                "orig_lng": -0.1,
                "lat": 51.5,
                "lng": -0.1,
                "moved_km": 1.0,
                "cell": (1, 1),
            },
            {
                "location_id": 1001,
                "city": "Paris",
                "orig_lat": 48.85,
                "orig_lng": 2.35,
                "lat": 51.5,
                "lng": -0.1,
                "moved_km": 9.0,
                "cell": (1, 1),
            },
        ]
        results = iter(collided)
        monkeypatch.setattr(cell_map, "_snap_one", lambda *_a: next(results))

        code = cell_map.build_cell_map(
            [1000, 1001], str(cities), "out.csv", "probes", 1
        )

        written = pd.read_csv(tmp_path / "out.csv")
        assert code == 2
        assert list(written["location_id"]) == [1000]

    def test_previous_mapping_is_preserved_for_cities_not_resnapped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cities = self._setup(tmp_path, monkeypatch)
        pd.DataFrame(
            {
                "location_id": [1001],
                "city": ["Paris"],
                "orig_lat": [48.85],
                "orig_lng": [2.35],
                "lat": [48.9],
                "lng": [2.4],
                "moved_km": [5.0],
            }
        ).to_csv(tmp_path / "out.csv", index=False)
        monkeypatch.setattr(
            cell_map,
            "_snap_one",
            lambda _c, row, _cand, _d: {
                "location_id": int(row.location_id),
                "city": row.city,
                "orig_lat": row.lat,
                "orig_lng": row.lng,
                "lat": 51.6,
                "lng": -0.2,
                "moved_km": 2.0,
                "cell": (516, -2),
            },
        )

        code = cell_map.build_cell_map([1000], str(cities), "out.csv", "probes", 1)

        written = pd.read_csv(tmp_path / "out.csv")
        assert code == 0
        assert sorted(written["location_id"]) == [1000, 1001]


class TestParseArgs:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            cell_map.sys,
            "argv",
            ["make_era5land_cell_map_eu.py", "--location-ids", "1000", "1001"],
        )
        args = cell_map._parse_args()
        assert args.cities_csv == cell_map.EU_CITIES_CSV
        assert args.out_csv == cell_map.CELL_MAP_CSV
        assert args.probe_dir == cell_map.PROBE_DIR_DEFAULT
        assert args.location_ids == [1000, 1001]


class TestMain:
    def test_exits_with_the_build_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cell_map, "load_dotenv", lambda **_k: None)
        monkeypatch.setattr(
            cell_map.sys,
            "argv",
            ["make_era5land_cell_map_eu.py", "--location-ids", "1000"],
        )
        monkeypatch.setattr(cell_map, "build_cell_map", lambda *_a: 2)

        with pytest.raises(SystemExit) as exc_info:
            cell_map.main()

        assert exc_info.value.code == 2

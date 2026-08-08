# Copyright (C) 2026 Kenneth Porter

"""Tests for the database location-catalog identity guard."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

import validate_location_catalog_db as validator

identity_conflicts = validator.identity_conflicts


def test_identity_conflicts_finds_reassigned_ids() -> None:
    incoming = pd.DataFrame(
        {"id": [1, 2], "city": ["Same", "McAllen"], "state": ["TX", "TX"]}
    )
    existing = pd.DataFrame(
        {"id": [1, 2], "city": ["Same", "Everett"], "state": ["TX", "WA"]}
    )

    conflicts = identity_conflicts(incoming, existing)

    assert conflicts["id"].tolist() == [2]
    assert conflicts.loc[0, "city_old"] == "Everett"
    assert conflicts.loc[0, "city_new"] == "McAllen"


def test_identity_conflicts_allows_coordinate_only_catalog_updates() -> None:
    incoming = pd.DataFrame({"id": [1], "city": ["Same"], "state": ["TX"]})
    existing = pd.DataFrame({"id": [1], "city": ["Same"], "state": ["TX"]})

    assert identity_conflicts(incoming, existing).empty


def test_validate_database_catalog_skips_without_database_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    catalog = tmp_path / "locations.csv"
    pd.DataFrame({"id": [1], "city": ["Same"], "state": ["TX"]}).to_csv(
        catalog,
        index=False,
    )
    monkeypatch.setattr(validator, "resolve_database_uri", lambda: None)

    with caplog.at_level(logging.WARNING):
        validator.validate_database_catalog(catalog)

    assert "Skipping database catalog validation" in caplog.text


def _mock_database(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[tuple[int, str, str]],
) -> MagicMock:
    cursor = MagicMock()
    cursor.fetchall.return_value = rows
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor
    connect = MagicMock(return_value=connection)
    monkeypatch.setattr(validator, "resolve_database_uri", lambda: "postgresql://db")
    monkeypatch.setattr(validator.psycopg, "connect", connect)
    return cursor


def test_validate_database_catalog_accepts_unchanged_populated_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = tmp_path / "locations.csv"
    pd.DataFrame({"id": [1], "city": ["Same"], "state": ["TX"]}).to_csv(
        catalog,
        index=False,
    )
    cursor = _mock_database(monkeypatch, [(1, "Same", "TX")])

    validator.validate_database_catalog(catalog)

    cursor.execute.assert_called_once()
    assert cursor.execute.call_args.args[1] == ([1],)


def test_validate_database_catalog_rejects_reassigned_populated_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = tmp_path / "locations.csv"
    pd.DataFrame({"id": [2], "city": ["McAllen"], "state": ["TX"]}).to_csv(
        catalog,
        index=False,
    )
    _mock_database(monkeypatch, [(2, "Everett", "WA")])

    with pytest.raises(
        RuntimeError,
        match=r"2: Everett, WA -> McAllen, TX",
    ):
        validator.validate_database_catalog(catalog)


def test_main_uses_requested_catalog(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "locations.csv"
    validate = MagicMock()
    monkeypatch.setattr(validator, "validate_database_catalog", validate)
    monkeypatch.setattr(
        "sys.argv",
        ["validate_location_catalog_db.py", "--locations-csv", str(catalog)],
    )

    validator.main()

    validate.assert_called_once_with(str(catalog))

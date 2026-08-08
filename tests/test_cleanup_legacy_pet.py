# Copyright (C) 2026 Kenneth Porter

"""Tests for explicit legacy PET cleanup."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import cleanup_legacy_pet


def test_requires_explicit_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cleanup_legacy_pet.py"])

    with pytest.raises(SystemExit, match="permanently deletes"):
        cleanup_legacy_pet.main()


def test_applies_cleanup_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    execute = MagicMock()
    monkeypatch.setattr(sys, "argv", ["cleanup_legacy_pet.py", "--confirm-db-write"])
    monkeypatch.setattr(
        cleanup_legacy_pet,
        "resolve_database_uri",
        lambda: "postgresql://example",
    )
    monkeypatch.setattr(cleanup_legacy_pet, "execute_sql_files_with_retries", execute)

    cleanup_legacy_pet.main()

    execute.assert_called_once_with(
        "postgresql://example",
        (Path(cleanup_legacy_pet.__file__).with_name("drop_legacy_pet.sql"),),
    )


def test_cleanup_sql_refuses_cascade() -> None:
    sql = cleanup_legacy_pet.MIGRATION_PATH.read_text(encoding="utf-8")
    drop_statements = re.findall(r"\bDROP\b[^;]*;", sql, flags=re.IGNORECASE)

    assert re.search(
        r"\bDROP\s+TABLE\s+IF\s+EXISTS\s+public\.pet\s*;",
        sql,
        flags=re.IGNORECASE,
    )
    assert all("CASCADE" not in statement.upper() for statement in drop_statements)

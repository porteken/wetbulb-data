"""Tests for the database location-catalog identity guard."""

from __future__ import annotations

import pandas as pd

from validate_location_catalog_db import identity_conflicts


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

from __future__ import annotations

import pandas as pd

import cities_ca


def test_normalize_population_centres_ranks_and_offsets_ids() -> None:
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

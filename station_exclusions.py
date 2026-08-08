# Copyright (C) 2026 Kenneth Porter

"""Known observation stations excluded from wet-bulb candidate selection."""

# Denver Central Park (USW00023012) reports summer dew points roughly 9 C above
# nearby Denver-area stations from 2016 onward. Its internally QC-passing
# humidity creates a false 6-7 C wet-bulb step for Denver and Lakewood.
DISALLOWED_GHCNH_STATION_IDS = frozenset({"USW00023012"})

# extractors/

Specialized data extractors that pull specific categories of information from SC2 observations.

## Modules

- **`unit_extractor.py`** - Extracts combat unit attributes (position, health, shields, energy, build progress, flying/cloak status) using a declarative `UNIT_FIELD_CONFIG` list. Tracks unit lifecycle transitions (started, completed, destroyed).
- **`building_extractor.py`** - Extracts building attributes using the same declarative pattern (`BUILDING_FIELD_CONFIG`). Captures real data during construction and tracks building lifecycle (started, under construction, completed, destroyed, cancelled).
- **`economy_extractor.py`** - Parses replay tracker events via `mpyq` and `s2protocol` to extract economy snapshots (minerals, vespene, supply, collection rates). Uses binary search for efficient lookup at each game loop.
- **`upgrade_extractor.py`** - Extracts per-player technology upgrade levels (attack, armor, shields) across research tiers.

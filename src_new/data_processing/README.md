# data_processing/

Post-extraction feature engineering and data transformation modules.

## Modules

- **`create_unit_counts.py`** - Parses entity columns and adds derived count columns per entity type per player (e.g., `p1_marine_count`). Also adds `unit_types_present`, `production_building_count`, and `has_air_units` flags.
- **`engineer_army_features.py`** - Uses DBSCAN spatial clustering to compute army composition and movement features: main army direction (aggressive/defensive/neutral), army size, number of distinct armies, and army complexity ratio.
- **`discretize.py`** - Creates a simplified dataset containing only engineered features (army metrics + supply columns), dropping all raw entity columns. Useful for lightweight baseline model training.
- **`fetch_bot_replays.py`** - Downloads SC2 bot replays from the AI Arena platform by bot name.

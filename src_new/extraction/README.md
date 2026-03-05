# extraction/

Core extraction components that transform raw SC2 observations into structured parquet output.

## Modules

- **`replay_loader.py`** - Wraps pysc2's `PipelineReplayLoader` to load `.SC2Replay` files and start an SC2 instance in observer mode.
- **`state_extractor.py`** - Orchestrates all four specialized extractors (`UnitExtractor`, `BuildingExtractor`, `EconomyExtractor`, `UpgradeExtractor`) for each observation and returns a complete state dictionary.
- **`schema_manager.py`** - Defines and manages the dynamic parquet schema. Builds column definitions on-the-fly as new entity types are encountered.
- **`wide_table_builder.py`** - Flattens the hierarchical state dictionary into a single wide-format row dictionary. Missing entities are filled with NaN.
- **`parquet_writer.py`** - Writes accumulated rows to compressed parquet files with proper type coercion. Produces `game_state.parquet`, `messages.parquet`, and `schema.json`.
- **`metadata_writer.py`** - Writes column definitions and metadata as JSON for downstream consumers.

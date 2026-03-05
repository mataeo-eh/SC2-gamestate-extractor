# SC2 Gamestate Extractor

A standalone tool that extracts complete game state data from StarCraft II replay files (`.SC2Replay`) and outputs structured parquet datasets suitable for machine learning, data analysis, and strategy research.

The pipeline replays each game inside the SC2 engine in observer mode, captures every unit, building, economy metric, and upgrade at every game loop, and writes the result as a single wide-format parquet file with 1000-1600+ columns per replay.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Output Format](#output-format)
- [Feature Engineering Pipeline](#feature-engineering-pipeline)
- [EDA Notebooks](#eda-notebooks)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)

---

## Features

- **Full game state extraction** - units, buildings, economy, and upgrades at every game loop
- **Lifecycle tracking** - tracks entity birth, completion, destruction, and cancellation over time
- **Parallel batch processing** - process entire replay directories with configurable worker counts
- **Feature engineering** - army composition, movement direction, clustering, and complexity metrics
- **Discretization** - create lightweight datasets with only engineered features for quick modeling
- **Replay downloading** - fetch bot replays directly from AI Arena
- **Kaggle integration** - upload processed datasets to Kaggle
- **Validation & QA** - built-in output validators and auto-generated documentation

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9+ |
| StarCraft II | Installed ([download](https://starcraft2.com/)) |
| pandas | any recent |
| pyarrow | any recent |
| numpy | any recent |
| pysc2 | any recent |
| s2protocol | any recent (for economy extraction) |
| mpyq | any recent (for replay archive parsing) |
| scikit-learn | any recent (for DBSCAN in feature engineering) |
| python-dotenv | any recent (for .env config) |

Optional:
- `ipywidgets` - enables interactive entity exploration in the EDA notebook
- `kaggle` - required only for Kaggle dataset uploads

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd SC2-gamestate-extractor

# Install dependencies
pip install pandas pyarrow numpy pysc2 s2protocol mpyq scikit-learn python-dotenv

# Optional: for interactive notebooks and Kaggle uploads
pip install ipywidgets kaggle
```

The `quickstart.py` script will verify all prerequisites on launch and tell you what's missing.

---

## Quick Start

### Process a single replay

```bash
python quickstart.py --replay path/to/replay.SC2Replay --output data/output
```

Short form:

```bash
python quickstart.py -r path/to/replay.SC2Replay -o data/output
```

### Process a directory of replays (parallel)

```bash
python quickstart.py --process-replay-directory replays/ --output data/output --workers 4
```

Short form:

```bash
python quickstart.py -batch replays/ -o data/output -w 4
```

### Download bot replays from AI Arena and process them

```bash
python quickstart.py --bots really what why --download-replays --num-replays 50 \
  --process-replay-directory replays/ --output data/output --workers 4
```

### Full end-to-end pipeline

Download replays, extract game state, add unit counts, engineer army features, discretize, and upload to Kaggle:

```bash
python quickstart.py \
  --bots really what why \
  --download-replays --num-replays 100 \
  --process-replay-directory replays/ \
  --output data/quickstart \
  --workers 3 \
  --engineer-features \
  --discretize \
  --update-kaggle-dataset
```

### Auto-detect mode

If you have replays in a `replays/` directory, just run:

```bash
python quickstart.py
```

The script will auto-discover and process the first replay it finds.

---

## CLI Reference

| Flag | Short | Description | Default |
|---|---|---|---|
| `--replay PATH` | `-r` | Path to a single `.SC2Replay` file | auto-detect |
| `--output DIR` | `-o` | Output directory for parquet files | `data/quickstart` |
| `--process-replay-directory DIR` | `-batch` | Process all replays in a directory (parallel) | - |
| `--workers N` | `-w` | Number of CPU workers for parallel processing | half of CPU count |
| `--download-replays` | `-dr` | Download replays from AI Arena before processing | off |
| `--num-replays N` | `-nr` | Number of replays to download (used with `-dr`) | all available |
| `--bots NAME [NAME ...]` | `-b` | Bot names to download replays for | required with `-dr` |
| `--engineer-features` | `-e` | Run army feature engineering after extraction | off |
| `--discretize` | `-d` | Create simplified dataset with only engineered features | off |
| `--update-kaggle-dataset` | `-dataset` | Upload processed data to Kaggle | off |
| `--verbose` | `-v` | Enable verbose output | off |

---

## Output Format

Each processed replay produces three files in `<output>/parquet/`:

```
<replay_name>_game_state.parquet   # Main dataset (all columns)
<replay_name>_messages.parquet     # Event/message log
schema.json                        # Column definitions and metadata
```

### Column structure of `game_state.parquet`

**Meta columns (3):**

| Column | Type | Description |
|---|---|---|
| `game_loop` | int | SC2 simulation tick (0-indexed) |
| `timestamp_seconds` | float | Elapsed game time in seconds |
| `Messages` | string | Event log (if populated) |

**Economy columns (12+):**

Per player (`p1_`, `p2_`): `minerals`, `vespene`, `supply_used`, `supply_cap`, `collection_rate_minerals`, `collection_rate_vespene`, `workers`, `idle_workers`

**Entity columns (hundreds to thousands):**

Each entity (unit or building) gets a set of attribute columns following this naming convention:

```
{player}_{botname}_{entitytype}_{seqid}_{attribute}
```

Example: `p1_really_marine_003_health`, `p2_what_nexus_001_x`

Attributes include: `x`, `y`, `z`, `facing`, `health`, `health_max`, `shields` (Protoss), `energy` (casters), `build_progress`, `is_flying`, `is_cloaked`, and `lifecycle` state strings.

**Lifecycle states embedded in entity columns:**

When an entity transitions state, its attribute columns contain a lifecycle string instead of numeric data:

| State | Meaning |
|---|---|
| `unit_started` / `building_started` | Entity training/construction began |
| `building` / `under_construction` | Building is being constructed |
| `completed` | Entity finished building/training |
| `existing` | Entity is alive and present |
| `destroyed` | Entity was killed |
| `cancelled` | Building/unit was cancelled |

**Upgrade columns (6):**

Per player: `upgrade_attack_level`, `upgrade_armor_level`, `upgrade_shield_level`

### Typical dataset dimensions

- **Rows:** 1,000-10,000+ (one per game loop, depending on game length)
- **Columns:** 1,000-1,600+ (depends on number of unique entities in the game)
- **File size:** 5-50 MB per replay (compressed parquet)

---

## Feature Engineering Pipeline

After extraction, two optional processing steps add derived features:

### 1. Unit Count Columns (`--engineer-features` prerequisite)

Automatically added after batch processing. For each entity type per player:

- `{player}_{entitytype}_count` - number of alive entities of this type
- `{player}_unit_types_present` - count of unique unit types alive
- `{player}_production_building_count` - count of production buildings
- `{player}_has_air_units` - boolean flag

### 2. Army Feature Engineering (`-e`)

Uses DBSCAN spatial clustering to compute per-player army metrics at 10-game-loop intervals:

| Feature | Description |
|---|---|
| `{player}_main_army_direction` | `"aggressive"` (toward enemy base), `"defensive"` (toward own base), or `"neutral"` |
| `{player}_main_army_size` | Number of combat units in the largest spatial cluster |
| `{player}_army_count` | Number of distinct army clusters (min 3 units each) |
| `{player}_army_complexity_ratio` | Unique unit types / game loop (army diversity over time) |

### 3. Discretization (`-d`)

Creates a lightweight dataset keeping only:
- Army features: `_main_army_direction`, `_army_complexity_ratio`, `_main_army_size`, `_army_count`
- Economy features: `_supply_cap`, `_supply_used`

This is useful for quick baseline model training and proof-of-concept strategy prediction.

---

## EDA Notebooks

Two Jupyter notebooks in `EDA/` provide interactive visualization and verification of extracted data.

### `EDA/data_verification.ipynb` - Lifecycle Verification & Visualization

The primary data quality tool. Verifies the extraction pipeline's output is correct and complete.

**Sections:**

1. **Setup & Data Loading** - Auto-discovers parquet files, loads game state data, and builds an entity catalog by parsing column names with `ENTITY_COL_RE`.

2. **Lifecycle Event Table** - Scans all entity columns for lifecycle string values and produces a DataFrame of every lifecycle transition (started, completed, destroyed, cancelled) in the game. This is the most important verification: it confirms entities are being tracked correctly from birth to death.

3. **Interactive Entity Explorer** - Widget-based dropdown (or callable function fallback) to browse individual entities: see their lifecycle events, data values at key game loops, and raw attribute values at each transition.

4. **Unit/Building Timeline Chart** - Gantt-chart visualization where each entity gets a horizontal row. Green markers for started, blue for completed, red for destroyed, orange for cancelled, with gray lines showing lifespan. Filterable by player and entity type.

5. **Correlation Heatmap** - Converts entity columns to numeric, computes correlations, and flags unexpectedly high correlations. Warns about expected ones (health vs health_max, supply_used vs supply_cap).

6. **Data Sparsity Visualization** - Heatmap showing where real numeric data exists vs NaN or lifecycle strings across all entities over game time. Useful for identifying entities that appear late or die early.

### `EDA/raw_data_summary.ipynb` - Spatial Data Profiling

Profiles the raw spatial and structural properties of extracted data across multiple games.

**Sections:**

1. **PyArrow Schema Inspection** - Reads parquet schema metadata without loading data into memory. Lists every column name and its Arrow data type.

2. **Column Parsing & Categorization** - Categorizes all columns as entity, economy, upgrade, or meta. Builds an inventory of all entity types per player.

3. **Spatial Helpers** - Finds base positions for each player (from town hall buildings or worker fallback). Computes coordinate bounding boxes across all position data.

4. **Game Profiling** - For each parquet file: game duration, row/column counts, base positions, bounding box dimensions, entity type inventories, position data quality (non-NaN fraction).

5. **Summary Report** - Aggregated statistics across all games: duration ranges, coordinate ranges, base-to-base distances, entity type frequency tables, and a per-game detail table.

### Running the notebooks

```bash
# From the SC2-gamestate-extractor directory
jupyter notebook EDA/data_verification.ipynb
jupyter notebook EDA/raw_data_summary.ipynb
```

Both notebooks expect extracted parquet files to exist at `../data/quickstart/parquet/` relative to the `EDA/` directory. Adjust the `parquet_dir` path in the first code cell if your data is elsewhere.

---

## Architecture Overview

The pipeline follows a layered design:

```
quickstart.py (CLI entry point)
    |
    v
ParallelReplayProcessor          # Batch orchestration with multiprocessing
    |
    v (one per worker)
ReplayExtractionPipeline          # Single-replay end-to-end orchestrator
    |
    +-- ReplayLoader              # Loads replay, starts SC2 instance
    |
    +-- StateExtractor            # Coordinates all four extractors:
    |       +-- UnitExtractor     #   Combat units (marine, mutalisk, etc.)
    |       +-- BuildingExtractor #   Structures (barracks, nexus, etc.)
    |       +-- EconomyExtractor  #   Resources, supply, collection rates
    |       +-- UpgradeExtractor  #   Attack/armor/shield upgrade levels
    |
    +-- SchemaManager             # Dynamic column schema definition
    |
    +-- WideTableBuilder          # Flattens hierarchical state to wide row
    |
    +-- ParquetWriter             # Writes parquet files with compression
    |
    +-- MetadataWriter            # Writes schema.json metadata
```

**Processing flow for each replay:**

1. `ReplayLoader` opens the `.SC2Replay` file and launches an SC2 instance in observer mode
2. For each game loop: the SC2 controller steps forward, observes game state, and the `StateExtractor` runs all four extractors to capture the complete state
3. `WideTableBuilder` flattens each observation into a single row dictionary (one column per entity attribute)
4. After all game loops are processed, `ParquetWriter` converts accumulated rows to a DataFrame and writes compressed parquet output
5. For batch processing, `ParallelReplayProcessor` distributes replays across a worker pool with error isolation per replay

---

## Project Structure

```
SC2-gamestate-extractor/
|-- README.md                       # This file
|-- LICENSE
|-- quickstart.py                   # Main CLI entry point
|
|-- EDA/                            # Exploratory data analysis notebooks
|   |-- data_verification.ipynb     #   Lifecycle verification & visualization
|   +-- raw_data_summary.ipynb      #   Spatial data profiling & summary
|
+-- src_new/                        # Pipeline source code
    |-- __init__.py
    |-- shared_constants.py         # Central constants (building/unit type sets, regex, lifecycle states)
    |
    |-- extraction/                 # Core extraction components
    |   |-- replay_loader.py        #   Loads replay via pysc2
    |   |-- state_extractor.py      #   Orchestrates all extractors per observation
    |   |-- schema_manager.py       #   Defines & manages dynamic parquet schema
    |   |-- wide_table_builder.py   #   Flattens state dict to wide-format row
    |   |-- parquet_writer.py       #   Writes parquet files with proper types
    |   +-- metadata_writer.py      #   Writes schema/metadata JSON
    |
    |-- extractors/                 # Specialized data extractors
    |   |-- unit_extractor.py       #   Extracts combat unit attributes
    |   |-- building_extractor.py   #   Extracts building attributes
    |   |-- economy_extractor.py    #   Extracts economy from tracker events
    |   +-- upgrade_extractor.py    #   Extracts upgrade levels
    |
    |-- pipeline/                   # Pipeline orchestration & batch processing
    |   |-- extraction_pipeline.py  #   Single-replay end-to-end orchestrator
    |   |-- parallel_processor.py   #   Multi-worker batch processing
    |   |-- game_loop_iterator.py   #   Configurable game loop iteration
    |   |-- dataset_pipeline.py     #   Kaggle dataset upload
    |   |-- logging_config.py       #   Centralized logging setup
    |   +-- integration_check.py    #   System integration validation
    |
    |-- data_processing/            # Feature engineering & data transformation
    |   |-- create_unit_counts.py   #   Adds unit/building count columns
    |   |-- engineer_army_features.py  # Army composition & movement features
    |   |-- discretize.py           #   Reduces dataset to engineered features only
    |   +-- fetch_bot_replays.py    #   Downloads bot replays from AI Arena
    |
    |-- utils/                      # Validation & documentation utilities
    |   |-- validation.py           #   OutputValidator class
    |   |-- documentation.py        #   Auto-generated feature documentation
    |   +-- validation_check.py     #   Integration validation
    |
    +-- batch/                      # Batch processing utilities
        +-- __init__.py
```

---

## Environment Variables

The feature engineering and discretization steps use `.env` configuration for output directories:

| Variable | Used By | Description |
|---|---|---|
| `ENGINEER_FEATURES_OUTPUT_DIR` | `--engineer-features` | Output directory for engineered feature parquets |
| `DISCRETIZE_INPUT_DIR` | `--discretize` | Input directory for discretization |
| `DISCRETIZE_OUTPUT_DIR` | `--discretize` | Output directory for discretized parquets |

Create a `.env` file in the project root:

```env
ENGINEER_FEATURES_OUTPUT_DIR=data/engineered
DISCRETIZE_INPUT_DIR=data/engineered
DISCRETIZE_OUTPUT_DIR=data/discretized
```

---

## License

See [LICENSE](LICENSE) for details.

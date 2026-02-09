"""
Feature Engineering Script: Unit & Building Counts for SC2 Replay Data

Phase 3 (Army & Composition Features) of the feature engineering plan.
Reads raw game state parquet files and produces new parquet files with
unit count, building count, and derived army composition columns added.

Each input parquet file has per-entity columns following the pattern:
    p{n}_p{n}_{entity_type}_{id}_{attribute}

This script:
1. Parses all entity columns to discover (player, entity_type, entity_id) tuples
2. For each (player, entity_type), counts alive instances per row
3. Computes derived features: total_unit_types, production_building_count, has_air_units
4. Writes the augmented dataframe to the output directory
"""

import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

# Ensure the project root is on sys.path so src_new can be imported
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from tqdm import tqdm

from src_new.pipeline.logging_config import setup_logging

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Entity column pattern: p{player}_p{player}_{type}_{id}_{attribute}
ENTITY_COL_RE = re.compile(r"^(p[12])_p[12]_(.+?)_(\d+)_(.+)$")

# Known building types (structures) - reused from raw_data_summary.py
BUILDING_TYPES = {
    # Terran
    "commandcenter", "commandcenterflying", "orbitalcommand", "planetaryfortress",
    "supplydepot", "supplydepotlowered", "barracks", "barrackstechlab",
    "barracksreactor", "factory", "factorytechlab", "factoryreactor",
    "starport", "starporttechlab", "starportreactor", "engineeringbay",
    "armory", "ghostacademy", "fusioncore", "bunker", "missileturret",
    "sensortower", "refinery",
    # Protoss
    "nexus", "pylon", "gateway", "forge", "cyberneticscore",
    "assimilator", "twilightcouncil", "templararchive", "darkshrine",
    "roboticsfacility", "roboticsbay", "stargate", "fleetbeacon",
    "photoncannon", "shieldbattery",
    # Zerg
    "hatchery", "lair", "hive", "spawningpool", "evolutionchamber",
    "extractor", "roachwarren", "banelingnest", "hydraliskden",
    "lurkerden", "infestationpit", "spire", "greaterspire",
    "ultraliskcavern", "nydusnetwork", "nyduscanal",
    "spinecrawler", "sporecrawler",
}

# Known air unit types (SC2 domain knowledge)
AIR_UNIT_TYPES = {
    # Terran
    "banshee", "battlecruiser", "liberator", "medivac", "raven", "viking", "vikingfighter",
    # Protoss
    "carrier", "oracle", "phoenix", "tempest", "voidray", "warpprism", "mothership",
    # Zerg
    "broodlord", "corruptor", "mutalisk", "overlord", "overseer", "viper",
}

# Known production buildings (buildings that produce units)
PRODUCTION_BUILDING_TYPES = {
    # Terran
    "barracks", "factory", "starport", "commandcenter", "orbitalcommand", "planetaryfortress",
    # Protoss
    "gateway", "roboticsfacility", "stargate", "nexus",
    # Zerg
    "hatchery", "lair", "hive",
}

# Alive states for units with a _state column
ALIVE_STATES = {"built", "existing"}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def parse_entity_columns(columns):
    """Parse all entity columns and group by (player, entity_type, entity_id).

    Returns:
        dict: {(player, entity_type, entity_id): set of attribute names}
    """
    entities = defaultdict(set)
    for col in columns:
        m = ENTITY_COL_RE.match(col)
        if m:
            player, entity_type, entity_id, attribute = m.groups()
            entities[(player, entity_type, entity_id)].add(attribute)
    return dict(entities)


def group_entities_by_player_type(entities):
    """Group entity instances by (player, entity_type).

    Args:
        entities: dict from parse_entity_columns

    Returns:
        dict: {(player, entity_type): list of (entity_id, set of attributes)}
    """
    groups = defaultdict(list)
    for (player, entity_type, entity_id), attrs in entities.items():
        groups[(player, entity_type)].append((entity_id, attrs))
    return dict(groups)


def compute_alive_count_for_group(df, player, entity_type, instances):
    """Compute the alive count for all instances of one (player, entity_type) group.

    Uses the _state column if available (alive = 'built' or 'existing').
    Falls back to _completed_loop/_destroyed_loop lifecycle columns.

    Args:
        df: The full dataframe
        player: 'p1' or 'p2'
        entity_type: e.g. 'marine', 'barracks'
        instances: list of (entity_id, set of attributes)

    Returns:
        pd.Series: Integer series with alive count per row
    """
    n_rows = len(df)
    total_alive = np.zeros(n_rows, dtype=np.int64)

    for entity_id, attrs in instances:
        col_prefix = f"{player}_{player}_{entity_type}_{entity_id}"

        if "state" in attrs:
            # Use state column: alive = state in ('built', 'existing')
            state_col = f"{col_prefix}_state"
            if state_col in df.columns:
                alive = df[state_col].isin(ALIVE_STATES).to_numpy(dtype=np.int64, na_value=0)
                total_alive += alive
                continue

        # Fallback: use completed_loop / destroyed_loop lifecycle columns
        if "completed_loop" in attrs and "destroyed_loop" in attrs:
            completed_col = f"{col_prefix}_completed_loop"
            destroyed_col = f"{col_prefix}_destroyed_loop"

            if completed_col in df.columns and destroyed_col in df.columns:
                # Alive = completed_loop is not NaN AND destroyed_loop is NaN
                completed_notna = df[completed_col].notna().to_numpy(dtype=np.bool_)
                destroyed_isna = df[destroyed_col].isna().to_numpy(dtype=np.bool_)
                alive = (completed_notna & destroyed_isna).astype(np.int64)
                total_alive += alive
                continue

        # If we only have completed_loop (no destroyed info), entity is alive once built
        if "completed_loop" in attrs:
            completed_col = f"{col_prefix}_completed_loop"
            if completed_col in df.columns:
                alive = df[completed_col].notna().to_numpy(dtype=np.int64)
                total_alive += alive

    return pd.Series(total_alive, index=df.index, dtype="int64")


def process_single_file(filepath):
    """Process a single parquet file and add unit/building count columns.

    Args:
        filepath: Path to the input parquet file

    Returns:
        pd.DataFrame: The augmented dataframe with count columns added
    """
    df = pd.read_parquet(filepath)

    # Parse all entity columns
    entities = parse_entity_columns(df.columns)
    groups = group_entities_by_player_type(entities)

    # Track which entity types exist per player for derived features
    entity_type_count_cols = {}  # (player, entity_type) -> column_name

    # Step 1: Compute per-entity-type count columns
    for (player, entity_type), instances in groups.items():
        col_name = f"{player}_{entity_type}_count"
        count_series = compute_alive_count_for_group(df, player, entity_type, instances)
        df[col_name] = count_series
        entity_type_count_cols[(player, entity_type)] = col_name

    # Step 2: Compute derived features for each player
    for player in ["p1", "p2"]:
        # --- total_unit_types: count of distinct UNIT types with at least 1 alive ---
        unit_count_cols = [
            col_name for (p, etype), col_name in entity_type_count_cols.items()
            if p == player and etype not in BUILDING_TYPES
        ]
        if unit_count_cols:
            # For each row, count how many unit type count columns have value > 0
            unit_counts_matrix = df[unit_count_cols].to_numpy()
            df[f"{player}_total_unit_types"] = (unit_counts_matrix > 0).sum(axis=1).astype(np.int64)
        else:
            df[f"{player}_total_unit_types"] = 0

        # --- production_building_count: sum of alive production buildings ---
        prod_count_cols = [
            col_name for (p, etype), col_name in entity_type_count_cols.items()
            if p == player and etype in PRODUCTION_BUILDING_TYPES
        ]
        if prod_count_cols:
            df[f"{player}_production_building_count"] = (
                df[prod_count_cols].sum(axis=1).fillna(0).astype(np.int64)
            )
        else:
            df[f"{player}_production_building_count"] = 0

        # --- has_air_units: 1 if any air unit type has count > 0 ---
        air_count_cols = [
            col_name for (p, etype), col_name in entity_type_count_cols.items()
            if p == player and etype in AIR_UNIT_TYPES
        ]
        if air_count_cols:
            air_matrix = df[air_count_cols].to_numpy()
            df[f"{player}_has_air_units"] = (air_matrix.sum(axis=1) > 0).astype(np.int64)
        else:
            df[f"{player}_has_air_units"] = 0

    # Step 3: Ensure all new columns are integer with no NaN
    new_cols = [c for c in df.columns if c.endswith("_count") or c.endswith("_total_unit_types")
                or c.endswith("_production_building_count") or c.endswith("_has_air_units")]
    for col in new_cols:
        df[col] = df[col].fillna(0).astype(np.int64)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir, output_dir, naming_pattern="_game_state", print_output=False, file_extension=".parquet"):
    """Process all game state parquet files and add unit/building count columns.

    Args:
        input_dir: Directory containing input parquet files.
        output_dir: Directory to write output parquet files.
        naming_pattern: Pattern to match input files (glob: *{naming_pattern}{file_extension}).
        print_output: If True, print summary info to console.
        file_extension: File extension to match.
    """
    # Setup logging
    setup_logging()

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Validate input directory
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        if print_output:
            print(f"Error: Input directory does not exist: {input_path}")
        return

    # Create output directory if needed
    output_path.mkdir(parents=True, exist_ok=True)

    # Find matching files
    glob_pattern = f"*{naming_pattern}{file_extension}"
    parquet_files = sorted(input_path.glob(glob_pattern))

    if not parquet_files:
        logger.warning(f"No files matching '{glob_pattern}' found in {input_path}")
        if print_output:
            print(f"No files matching '{glob_pattern}' found in {input_path}")
        return

    logger.info(f"Starting unit/building count feature engineering")
    logger.info(f"Input directory: {input_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Found {len(parquet_files)} files to process")

    if print_output:
        print(f"Starting unit/building count feature engineering")
        print(f"Input directory: {input_path}")
        print(f"Output directory: {output_path}")
        print(f"Found {len(parquet_files)} files to process")

    # Process files with progress bar
    processed_files = []
    failed_files = []

    for filepath in tqdm(parquet_files, desc="Processing files", unit="file"):
        tqdm.write(f"  Processing: {filepath.name}")
        try:
            df_out = process_single_file(filepath)

            # Write to output directory with the same filename
            out_filepath = output_path / filepath.name
            df_out.to_parquet(out_filepath, index=False)

            processed_files.append(filepath.name)
        except Exception as e:
            logger.warning(f"Failed to process {filepath.name}: {e}")
            failed_files.append(filepath.name)
            if print_output:
                print(f"  WARNING: Failed to process {filepath.name}: {e}")

    # Summary logging
    logger.info(f"Processing complete. {len(processed_files)} succeeded, {len(failed_files)} failed.")
    if print_output:
        print(f"\nProcessing complete. {len(processed_files)} succeeded, {len(failed_files)} failed.")

    # DEBUG-level summary with filenames
    if processed_files:
        logger.debug("Successfully processed files:\n  " + "\n  ".join(processed_files))
    if failed_files:
        logger.debug("Failed files:\n  " + "\n  ".join(failed_files))

    logger.info("Unit/building count feature engineering finished.")
    if print_output:
        print("Unit/building count feature engineering finished.")


if __name__ == "__main__":
    main(
        input_dir="data/quickstart/parquet",
        output_dir="data/quickstart/features",
        print_output=True,
    )

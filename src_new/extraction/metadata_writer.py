"""
metadata_writer.py: Writes comprehensive metadata JSON for extracted SC2 replay datasets.

This module replaces the old schema-only JSON output (save_schema) with a rich
metadata file that makes each parquet file self-documenting for dataset consumers.

The metadata JSON contains:
  - dataset_info:    Row/column counts, file references, timestamps
  - game_info:       Map name, dimensions, game version, speed
  - players:         Player names, races, results, APM, MMR
  - unit_counts:     Per-player unit type counts (distinct entity instances)
  - building_counts: Per-player building type counts (distinct entity instances)
  - messages:        In-game chat messages

Unit and building counts are derived by parsing entity column names using the
ENTITY_COL_RE regex from shared_constants.py, then classifying each entity type
as a unit or building using the BUILDING_TYPES set.

Role in the pipeline:
  Called by extraction_pipeline.py after the game loop completes and parquet
  files are written. Receives the metadata dict (from replay_loader), column
  list (from schema_manager), messages, row count, and parquet filename.
"""

import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src_new.shared_constants import BUILDING_TYPES, ENTITY_COL_RE


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Game speed enum mapping (from s2protocol replay.details m_gameSpeed field)
# Standard competitive games are always Faster (4). The 22.4 game-loops-per-
# second constant applies specifically to Faster speed.
# ---------------------------------------------------------------------------
GAME_SPEED_NAMES: Dict[int, str] = {
    0: "Slower",
    1: "Slow",
    2: "Normal",
    3: "Fast",
    4: "Faster",
}


def _count_entities_from_columns(
    columns: List[str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Count distinct entity instances per player per entity type from column names.

    Parses each column name with ENTITY_COL_RE to extract (player, entity_type,
    entity_id). Groups by (player, entity_type) and counts distinct entity_ids
    to determine how many instances of each type appeared in the game.

    The entity type is extracted from the "middle" regex group by taking the
    last underscore-delimited token. This works because SC2 type names are
    single lowercase words (e.g., "marine", "commandcenter") while the middle
    group contains "{botname}_{entitytype}".

    Args:
        columns: Ordered list of all column names from the schema.

    Returns:
        Nested dict structured as:
        {
            "p1": {
                "marine": {"count": 5, "entity_ids": {"001", "002", ...}},
                "scv":    {"count": 12, "entity_ids": {"001", ...}},
                ...
            },
            "p2": { ... }
        }

    Depends on:
        - ENTITY_COL_RE from shared_constants.py for column parsing
    """
    # Map: player -> entity_type -> set of entity_ids
    entity_map: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    for col_name in columns:
        match = ENTITY_COL_RE.match(col_name)
        if not match:
            continue

        player, middle, entity_id, _attribute = match.groups()

        # Extract entity type name from the middle portion.
        # Middle = "{botname}_{entitytype}", e.g. "really_probe" -> "probe"
        # SC2 type names never contain underscores after sanitization.
        entity_type = middle.rsplit("_", 1)[-1]

        entity_map[player][entity_type].add(entity_id)

    # Convert sets to count dicts for structured output
    result: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for player in sorted(entity_map.keys()):
        result[player] = {}
        for entity_type in sorted(entity_map[player].keys()):
            ids = entity_map[player][entity_type]
            result[player][entity_type] = {
                "count": len(ids),
                "entity_ids": ids,  # kept as set; will be discarded before JSON
            }

    return result


def _split_units_and_buildings(
    entity_counts: Dict[str, Dict[str, Dict[str, Any]]],
) -> tuple:
    """
    Split entity counts into separate unit and building dicts.

    Uses the BUILDING_TYPES set from shared_constants.py to classify each
    entity type. Any entity type NOT in BUILDING_TYPES is considered a unit.

    Args:
        entity_counts: Output from _count_entities_from_columns().

    Returns:
        Tuple of (unit_counts_dict, building_counts_dict), each structured as:
        {
            "p1": {
                "marine": 5,
                "scv": 12,
                "total_unique_types": 2,
                "total_entities": 17
            },
            ...
        }

    Depends on:
        - BUILDING_TYPES from shared_constants.py
    """
    unit_counts: Dict[str, Dict[str, Any]] = {}
    building_counts: Dict[str, Dict[str, Any]] = {}

    for player, types_dict in entity_counts.items():
        units_for_player: Dict[str, int] = {}
        buildings_for_player: Dict[str, int] = {}

        for entity_type, info in types_dict.items():
            count = info["count"]
            if entity_type in BUILDING_TYPES:
                buildings_for_player[entity_type] = count
            else:
                units_for_player[entity_type] = count

        # Add summary stats for units
        unit_summary = dict(sorted(units_for_player.items()))
        unit_summary["total_unique_types"] = len(units_for_player)
        unit_summary["total_entities"] = sum(units_for_player.values())
        unit_counts[player] = unit_summary

        # Add summary stats for buildings
        building_summary = dict(sorted(buildings_for_player.items()))
        building_summary["total_unique_types"] = len(buildings_for_player)
        building_summary["total_entities"] = sum(buildings_for_player.values())
        building_counts[player] = building_summary

    return unit_counts, building_counts


def _build_dataset_info(
    parquet_filename: str,
    total_rows: int,
    total_columns: int,
    game_duration_loops: int,
    game_duration_seconds: float,
) -> Dict[str, Any]:
    """
    Build the dataset_info section of the metadata JSON.

    Contains statistical summary of the extracted dataset: file reference,
    row/column counts, game duration, and extraction timestamp.

    Args:
        parquet_filename: Name of the parquet file (e.g., "match_4184936_game_state.parquet")
        total_rows: Number of rows (game loops) in the dataset
        total_columns: Number of columns in the wide-format table
        game_duration_loops: Total game loops from replay metadata
        game_duration_seconds: Game duration in seconds (loops / 22.4)

    Returns:
        Dict with dataset_info fields.
    """
    return {
        "parquet_file": parquet_filename,
        "total_rows": total_rows,
        "total_columns": total_columns,
        "total_game_loops": game_duration_loops,
        "game_duration_seconds": round(game_duration_seconds, 1),
        "game_duration_minutes": round(game_duration_seconds / 60.0, 1),
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _build_game_info(
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the game_info section of the metadata JSON.

    Extracts map name, dimensions, game version, and speed from the metadata
    dict provided by the pipeline. Fields that are not available in the
    metadata dict are set to null in the output.

    Args:
        metadata: The metadata dict from replay_loader.get_replay_info(),
                  potentially enriched with additional fields (game_version,
                  map_width, map_height, game_speed, etc.).

    Returns:
        Dict with game_info fields.
    """
    game_info: Dict[str, Any] = {
        "map_name": metadata.get("map_name"),
    }

    # Map dimensions — available if enriched from s2protocol initdata
    map_width = metadata.get("map_width")
    map_height = metadata.get("map_height")
    if map_width is not None and map_height is not None:
        game_info["map_dimensions"] = {
            "width": map_width,
            "height": map_height,
        }
    else:
        game_info["map_dimensions"] = None

    # Game version — available from ResponseReplayInfo.game_version
    game_info["game_version"] = metadata.get("game_version")

    # Data build and base build numbers for precise version identification
    game_info["data_build"] = metadata.get("data_build")
    game_info["base_build"] = metadata.get("base_build")

    # Game speed — integer 0-4 mapped to human-readable string
    game_speed_int = metadata.get("game_speed")
    if game_speed_int is not None:
        game_info["game_speed"] = GAME_SPEED_NAMES.get(game_speed_int, str(game_speed_int))
    else:
        game_info["game_speed"] = None

    return game_info


def _build_players_section(
    metadata: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Build the players section of the metadata JSON.

    Transforms the list-based player data from the metadata dict into a
    dict keyed by player prefix (p1, p2) for easier downstream access.

    Args:
        metadata: The metadata dict from replay_loader.get_replay_info().
                  Must contain a 'players' list of player dicts with keys:
                  player_id, player_name, race, apm, mmr, result.

    Returns:
        Dict keyed by "p1", "p2", etc., each containing player details.
    """
    players_section: Dict[str, Dict[str, Any]] = {}

    for player_data in metadata.get("players", []):
        player_key = f"p{player_data['player_id']}"
        players_section[player_key] = {
            "name": player_data.get("player_name", "Unknown"),
            "race": player_data.get("race", "Unknown"),
            "result": player_data.get("result", "Unknown"),
            "apm": player_data.get("apm"),
            "mmr": player_data.get("mmr"),
        }

    return players_section


def _build_messages_section(
    all_messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build the messages section of the metadata JSON.

    Includes the total count and full list of in-game chat messages.
    Each message has game_loop, player_id, and message text.

    Args:
        all_messages: List of message dicts with keys:
                      game_loop (int), player_id (int), message (str).

    Returns:
        Dict with total_messages count and messages list.
    """
    return {
        "total_messages": len(all_messages),
        "messages": all_messages,
    }


def build_metadata(
    metadata: Dict[str, Any],
    columns: List[str],
    total_rows: int,
    parquet_filename: str,
    all_messages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Build the complete metadata dictionary for a processed replay.

    This is the main entry point for constructing the metadata JSON. It
    assembles all sections (dataset_info, game_info, players, unit_counts,
    building_counts, messages) into a single dict ready for JSON serialization.

    Args:
        metadata: The metadata dict from replay_loader.get_replay_info(),
                  containing map_name, game_duration_loops, game_duration_seconds,
                  num_players, players list, and optionally enriched fields
                  (game_version, map_width, map_height, etc.).
        columns: Ordered list of all column names from schema_manager.columns.
        total_rows: Number of data rows written to the parquet file.
        parquet_filename: Name of the parquet output file (e.g.,
                          "match_4184936_game_state.parquet").
        all_messages: List of in-game chat message dicts. Each dict has
                      keys: game_loop, player_id, message. Pass empty list
                      or None if no messages were captured.

    Returns:
        Complete metadata dict with keys: dataset_info, game_info, players,
        unit_counts, building_counts, messages.

    Depends on / calls:
        - _build_dataset_info()
        - _build_game_info()
        - _build_players_section()
        - _count_entities_from_columns()
        - _split_units_and_buildings()
        - _build_messages_section()
    """
    if all_messages is None:
        all_messages = []

    # Build each section of the metadata
    dataset_info = _build_dataset_info(
        parquet_filename=parquet_filename,
        total_rows=total_rows,
        total_columns=len(columns),
        game_duration_loops=metadata.get("game_duration_loops", 0),
        game_duration_seconds=metadata.get("game_duration_seconds", 0.0),
    )

    game_info = _build_game_info(metadata)
    players = _build_players_section(metadata)

    # Count entities from column names and split into units vs buildings
    entity_counts = _count_entities_from_columns(columns)
    unit_counts, building_counts = _split_units_and_buildings(entity_counts)

    messages_section = _build_messages_section(all_messages)

    return {
        "dataset_info": dataset_info,
        "game_info": game_info,
        "players": players,
        "unit_counts": unit_counts,
        "building_counts": building_counts,
        "messages": messages_section,
    }


def save_metadata(
    metadata_dict: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Write the metadata dictionary to a JSON file.

    Creates parent directories if they do not exist. Uses 2-space indentation
    for human readability.

    Args:
        metadata_dict: The complete metadata dict from build_metadata().
        output_path: File path for the output JSON (e.g.,
                     data/processed/json/match_4184936_metadata.json).

    Depends on:
        - build_metadata() to construct the metadata_dict
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2, default=str)

    logger.info(f"Metadata saved to {output_path}")

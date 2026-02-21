"""
SchemaManager: Manages wide-table column schema and documentation.

This component handles schema generation, column ordering, data types,
and documentation for the wide-format parquet output.

Schema is built in two stages:
- Static columns (game metadata, economy, upgrades) are created upfront from
  replay metadata via build_base_schema(), before extraction begins.
- Entity columns (units and buildings) are added dynamically during the game
  loop via ensure_unit_columns() and ensure_building_columns(), as each new
  entity is encountered for the first time.

Key behaviors:
- Only creates columns for units that actually complete (not cancelled/interrupted)
- Creates columns for ALL buildings (even cancelled ones, since position is meaningful)
- Conditionally includes shields (Protoss only) and energy (casters only) per entity
- No separate lifecycle columns (status, progress, started_loop, completed_loop,
  destroyed_loop) - lifecycle state is embedded in attribute columns
"""

from typing import List, Dict, Any, Set, Optional
from pathlib import Path
import json
import logging
import re

import numpy as np


logger = logging.getLogger(__name__)


def sanitize_name(name: str) -> str:
    """
    Sanitize a player/bot name for use in column names.

    Lowercases, replaces non-alphanumeric characters with underscores,
    strips leading/trailing underscores, and falls back to 'unknown' if empty.

    Args:
        name: Raw player name string

    Returns:
        Sanitized name safe for column naming
    """
    sanitized = re.sub(r'[^a-z0-9]', '_', name.lower())
    sanitized = sanitized.strip('_')
    return sanitized or 'unknown'


# Base attribute columns for units (always present).
# Suffixes must match the column_suffix values in UNIT_FIELD_CONFIG
# (src_new/extractors/unit_extractor.py).
UNIT_BASE_ATTRIBUTES = [
    ('pos_(X,Y,Z)', 'object', 'Position as (X, Y, Z) coordinate tuple'),
    ('health', 'object', 'Health as current/max fraction string'),
    ('facing', 'object', 'Facing direction'),
    ('radius', 'object', 'Unit radius'),
    ('build_progress', 'object', 'Build progress (0.0 to 1.0)'),
    ('is_flying', 'object', 'Whether unit is flying'),
    ('is_burrowed', 'object', 'Whether unit is burrowed'),
    ('is_hallucination', 'object', 'Whether unit is a hallucination'),
    ('weapon_cooldown', 'object', 'Weapon cooldown'),
    ('attack_upgrade_level', 'object', 'Attack upgrade level'),
    ('armor_upgrade_level', 'object', 'Armor upgrade level'),
    ('cargo_space_taken', 'object', 'Cargo space taken'),
    ('cargo_space_max', 'object', 'Maximum cargo space'),
    ('order_count', 'object', 'Number of queued orders'),
]

# Conditional attribute columns for units (added only when applicable).
# Suffixes must match the conditional column_suffix values in UNIT_FIELD_CONFIG.
UNIT_SHIELD_ATTRIBUTES = [
    ('shields', 'object', 'Shields as current/max fraction string (Protoss only)'),
    ('shield_upgrade_level', 'object', 'Shield upgrade level (Protoss only)'),
]

UNIT_ENERGY_ATTRIBUTES = [
    ('energy', 'object', 'Energy as current/max fraction string (casters only)'),
]

# Base attribute columns for buildings (always present).
# Suffixes must match the column_suffix values in BUILDING_FIELD_CONFIG
# (src_new/extractors/building_extractor.py).
BUILDING_BASE_ATTRIBUTES = [
    ('pos_(X,Y,Z)', 'object', 'Position as (X, Y, Z) coordinate tuple'),
    ('health', 'object', 'Health as current/max fraction string'),
    ('facing', 'object', 'Facing direction'),
    ('build_progress', 'object', 'Build progress (0.0 to 1.0)'),
    ('is_flying', 'object', 'Whether building is flying (lifted Terran)'),
    ('is_burrowed', 'object', 'Whether building is burrowed'),
    ('attack_upgrade_level', 'object', 'Attack upgrade level'),
    ('armor_upgrade_level', 'object', 'Armor upgrade level'),
    ('radius', 'object', 'Building radius'),
    ('order_count', 'object', 'Number of queued orders'),
]

# Conditional attribute columns for buildings.
# Suffixes must match the conditional column_suffix values in BUILDING_FIELD_CONFIG.
BUILDING_SHIELD_ATTRIBUTES = [
    ('shields', 'object', 'Shields as current/max fraction string (Protoss only)'),
    ('shield_upgrade_level', 'object', 'Shield upgrade level (Protoss only)'),
]

BUILDING_ENERGY_ATTRIBUTES = [
    ('energy', 'object', 'Energy as current/max fraction string (casters only)'),
]


class SchemaManager:
    """
    Manages wide-table column schema and documentation.

    This class manages the wide-table column schema by building static columns
    upfront from replay metadata and adding entity-specific columns dynamically
    during extraction. It provides column ordering, data types, missing value
    defaults, and documentation for the wide-format parquet output.

    Column dtype is 'object' for all unit/building attribute columns because
    they can contain either numeric data (float) or lifecycle state strings
    (e.g., "unit_started", "building", "completed", "destroyed").
    """

    def __init__(self):
        """Initialize the SchemaManager."""
        self.columns: List[str] = []
        self.column_docs: Dict[str, Dict[str, Any]] = {}
        self.dtypes: Dict[str, str] = {}

        # Track seen entities for schema building
        self._seen_units: Set[str] = set()
        self._seen_buildings: Set[str] = set()

        # Player name mapping: {player_num: sanitized_name} e.g. {1: "really", 2: "what"}
        self.player_names: Dict[int, str] = {}

        # Base columns that always exist
        self._add_base_columns()

        logger.info("SchemaManager initialized")

    def set_player_names(self, player_names: Dict[int, str]) -> None:
        """
        Store a mapping of player numbers to sanitized bot/player names.

        Args:
            player_names: Dict mapping player number to raw name,
                          e.g. {1: "Really", 2: "What!"}
        """
        self.player_names = {
            num: sanitize_name(name) for num, name in player_names.items()
        }
        logger.info(f"Player names set: {self.player_names}")

    def _add_base_columns(self):
        """Add base columns that exist in every row."""
        base_columns = [
            ('game_loop', 'int64', 'Current game loop (frame) number'),
            ('timestamp_seconds', 'float64', 'Time in seconds since game start (game_loop / 22.4)'),
            ('Messages', 'object', 'Ally chat messages sent at this game loop (NaN if none, string if one, list of strings if multiple)'),
        ]

        for col_name, dtype, description in base_columns:
            if col_name not in self.columns:
                self.columns.append(col_name)
                self.dtypes[col_name] = dtype
                self.column_docs[col_name] = {
                    'description': description,
                    'type': dtype,
                    'missing_value': 'N/A' if col_name != 'Messages' else 'NaN',
                }

    def build_base_schema(self, player_names: Dict[int, str]) -> None:
        """
        Build the static portion of the schema using player names from replay metadata.

        This replaces build_schema_from_replay(). Instead of launching a full SC2
        instance to pre-scan the replay, it builds only the columns whose structure is
        known before extraction begins: game metadata, economy, and upgrade columns.
        Unit and building columns are added dynamically during extraction via
        ensure_unit_columns() and ensure_building_columns().

        Call this AFTER get_replay_info() has returned player names, and BEFORE
        creating the WideTableBuilder or starting the game loop.

        Args:
            player_names: Dict mapping player number to raw player name,
                          e.g. {1: "Really", 2: "What!"}

        Calls:
            - self.set_player_names(player_names) — sanitizes and stores names
            - self._add_economy_columns() — adds p1_minerals, p1_vespene, etc.
            - self._add_upgrade_columns() — adds p1_upgrade_attack_level, etc.

        Note:
            _add_base_columns() is already called in __init__() and does not need
            to be called here. The static base columns (game_loop, timestamp_seconds,
            Messages) are present from object creation.
        """
        self.set_player_names(player_names)
        self._add_economy_columns()
        self._add_upgrade_columns()
        logger.info(
            f"Base schema built with {len(self.columns)} static columns"
        )

    def ensure_unit_columns(self, player: str, readable_id: str, extra_attrs: Set[str] = None) -> bool:
        """
        Add columns for a unit if they do not already exist in the schema.

        Called inside the game loop, once per unit per frame, BEFORE build_row().
        Prevents duplicate column creation when the same unit appears across multiple
        frames. Only call this when the unit's lifecycle == 'completed' (i.e., the
        unit has fully built). Units still in progress do not get columns yet.

        Args:
            player: Player prefix, e.g. 'p1' or 'p2'
            readable_id: Human-readable unit ID, e.g. 'p1_marine_001'
            extra_attrs: Set of conditional attribute names this unit has
                         (e.g. {'shields', 'energy'}). Pass None or empty set
                         if the unit has no conditional attributes.

        Returns:
            True if new columns were added (first time this unit was seen).
            False if columns already existed (unit was seen in a prior frame).

        Depends on:
            - self._seen_units: set of readable_ids already in schema
            - self.add_unit_columns(): creates the actual column entries
        """
        if extra_attrs is None:
            extra_attrs = set()
        if readable_id not in self._seen_units:
            self._seen_units.add(readable_id)
            self.add_unit_columns(player, readable_id, extra_attrs)
            return True  # New columns were added
        return False  # Already existed — no duplicate columns created

    def ensure_building_columns(self, player: str, readable_id: str, extra_attrs: Set[str] = None) -> bool:
        """
        Add columns for a building if they do not already exist in the schema.

        Called inside the game loop, once per building per frame, BEFORE build_row().
        Buildings always get columns on their first appearance — even cancelled ones —
        because building position data is meaningful regardless of whether construction
        completed.

        Args:
            player: Player prefix, e.g. 'p1' or 'p2'
            readable_id: Human-readable building ID, e.g. 'p1_barracks_001'
            extra_attrs: Set of conditional attribute names this building has
                         (e.g. {'shields', 'energy'}). Pass None or empty set
                         if the building has no conditional attributes.

        Returns:
            True if new columns were added (first time this building was seen).
            False if columns already existed (building was seen in a prior frame).

        Depends on:
            - self._seen_buildings: set of readable_ids already in schema
            - self.add_building_columns(): creates the actual column entries
        """
        if extra_attrs is None:
            extra_attrs = set()
        if readable_id not in self._seen_buildings:
            self._seen_buildings.add(readable_id)
            self.add_building_columns(player, readable_id, extra_attrs)
            return True  # New columns were added
        return False  # Already existed — no duplicate columns created

    def add_unit_columns(self, player: str, unit_id: str, extra_attrs: Set[str] = None) -> None:
        """
        Add columns for a specific unit.

        No separate lifecycle columns (state, etc.) - lifecycle is embedded
        in the attribute columns themselves.

        Args:
            player: Player prefix (e.g., 'p1', 'p2')
            unit_id: Unit identifier (e.g., 'p1_marine_001')
            extra_attrs: Set of conditional attribute names this unit has
                        (e.g., {'shields', 'shields_max', 'energy', 'energy_max'})
        """
        if extra_attrs is None:
            extra_attrs = set()

        # Build the list of attribute columns for this unit
        unit_columns = list(UNIT_BASE_ATTRIBUTES)

        # Conditionally add shields (extra_attrs contains column_suffix strings
        # from UNIT_FIELD_CONFIG, e.g. 'shields', 'shield_upgrade_level')
        if any(a in extra_attrs for a in ['shields', 'shield_upgrade_level']):
            unit_columns.extend(UNIT_SHIELD_ATTRIBUTES)

        # Conditionally add energy
        if 'energy' in extra_attrs:
            unit_columns.extend(UNIT_ENERGY_ATTRIBUTES)

        # Strip existing player prefix from unit_id (e.g., "p1_marine_001" -> "marine_001")
        stripped_id = '_'.join(unit_id.split('_')[1:]) if unit_id.startswith('p') else unit_id
        player_num = int(player[1:])  # "p1" -> 1
        bot_name = self.player_names.get(player_num, player)

        for col_suffix, dtype, description in unit_columns:
            col_name = f'{player}_{bot_name}_{stripped_id}_{col_suffix}'

            if col_name not in self.columns:
                self.columns.append(col_name)
                self.dtypes[col_name] = dtype
                self.column_docs[col_name] = {
                    'description': f'{description} for {player} {unit_id}',
                    'type': dtype,
                    'missing_value': 'NaN',
                }

    def add_building_columns(self, player: str, building_id: str, extra_attrs: Set[str] = None) -> None:
        """
        Add columns for a specific building.

        No separate lifecycle columns (status, progress, started_loop,
        completed_loop, destroyed_loop) - lifecycle is embedded in the
        attribute columns themselves.

        Args:
            player: Player prefix (e.g., 'p1', 'p2')
            building_id: Building identifier (e.g., 'p1_barracks_001')
            extra_attrs: Set of conditional attribute names this building has
        """
        if extra_attrs is None:
            extra_attrs = set()

        # Build the list of attribute columns for this building
        building_columns = list(BUILDING_BASE_ATTRIBUTES)

        # Conditionally add shields (extra_attrs contains column_suffix strings
        # from BUILDING_FIELD_CONFIG, e.g. 'shields', 'shield_upgrade_level')
        if any(a in extra_attrs for a in ['shields', 'shield_upgrade_level']):
            building_columns.extend(BUILDING_SHIELD_ATTRIBUTES)

        # Conditionally add energy
        if 'energy' in extra_attrs:
            building_columns.extend(BUILDING_ENERGY_ATTRIBUTES)

        # Strip existing player prefix from building_id (e.g., "p1_barracks_001" -> "barracks_001")
        stripped_id = '_'.join(building_id.split('_')[1:]) if building_id.startswith('p') else building_id
        player_num = int(player[1:])  # "p1" -> 1
        bot_name = self.player_names.get(player_num, player)

        for col_suffix, dtype, description in building_columns:
            col_name = f'{player}_{bot_name}_{stripped_id}_{col_suffix}'

            if col_name not in self.columns:
                self.columns.append(col_name)
                self.dtypes[col_name] = dtype
                self.column_docs[col_name] = {
                    'description': f'{description} for {player} {building_id}',
                    'type': dtype,
                    'missing_value': 'NaN',
                }

    def _add_economy_columns(self) -> None:
        """Add economy columns for both players."""
        for player_num in [1, 2]:
            economy_columns = [
                ('minerals', 'int64', 'Current minerals'),
                ('vespene', 'int64', 'Current vespene gas'),
                ('supply_used', 'int64', 'Supply used'),
                ('supply_cap', 'int64', 'Supply capacity'),
                ('workers', 'int64', 'Total worker count'),
                ('idle_workers', 'int64', 'Idle worker count'),
            ]

            for col_suffix, dtype, description in economy_columns:
                col_name = f'p{player_num}_{col_suffix}'

                if col_name not in self.columns:
                    self.columns.append(col_name)
                    self.dtypes[col_name] = dtype
                    self.column_docs[col_name] = {
                        'description': f'{description} for player {player_num}',
                        'type': dtype,
                        'missing_value': 'NaN',
                    }

    def _add_upgrade_columns(self) -> None:
        """Add upgrade columns for both players."""
        # Common upgrades across all races
        common_upgrades = [
            'attack_level',
            'armor_level',
            'shield_level',
        ]

        for player_num in [1, 2]:
            for upgrade in common_upgrades:
                col_name = f'p{player_num}_upgrade_{upgrade}'

                if col_name not in self.columns:
                    self.columns.append(col_name)
                    self.dtypes[col_name] = 'int64'
                    self.column_docs[col_name] = {
                        'description': f'{upgrade.replace("_", " ").title()} for player {player_num}',
                        'type': 'int64',
                        'missing_value': '0',
                    }

    def get_column_list(self) -> List[str]:
        """
        Return ordered list of all columns.

        Returns:
            List of column names in order
        """
        return self.columns.copy()

    def generate_documentation(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate data dictionary.

        Returns:
            Dictionary mapping column names to documentation.
        """
        return self.column_docs.copy()

    def save_schema(self, output_path: Path) -> None:
        """
        Save schema to JSON file.

        Args:
            output_path: Path to save schema JSON
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        schema_data = {
            'columns': self.columns,
            'dtypes': self.dtypes,
            'documentation': self.column_docs,
        }

        with open(output_path, 'w') as f:
            json.dump(schema_data, f, indent=2)

        logger.info(f"Schema saved to {output_path}")

    def load_schema(self, schema_path: Path) -> None:
        """
        Load schema from JSON file.

        Args:
            schema_path: Path to schema JSON file
        """
        schema_path = Path(schema_path)

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with open(schema_path, 'r') as f:
            schema_data = json.load(f)

        self.columns = schema_data['columns']
        self.dtypes = schema_data['dtypes']
        self.column_docs = schema_data['documentation']

        logger.info(f"Schema loaded from {schema_path} ({len(self.columns)} columns)")

    def get_dtype(self, column_name: str) -> str:
        """
        Get data type for a column.

        Args:
            column_name: Column name

        Returns:
            Data type string
        """
        return self.dtypes.get(column_name, 'object')

    def get_missing_value(self, column_name: str) -> Any:
        """
        Get appropriate missing value for a column.

        Args:
            column_name: Column name

        Returns:
            Missing value (NaN for numeric, None for object, etc.)
        """
        dtype = self.get_dtype(column_name)

        if dtype.startswith('float') or dtype.startswith('int'):
            return np.nan
        else:
            return np.nan  # Use NaN for object columns too (Parquet compatible)

    def add_unit_count_columns(self, unit_type: str) -> None:
        """
        Add unit count columns for a specific unit type.

        Args:
            unit_type: Unit type name (e.g., 'Marine')
        """
        for player_num in [1, 2]:
            col_name = f'p{player_num}_{unit_type.lower()}_count'

            if col_name not in self.columns:
                self.columns.append(col_name)
                self.dtypes[col_name] = 'int64'
                self.column_docs[col_name] = {
                    'description': f'Count of {unit_type} units for player {player_num}',
                    'type': 'int64',
                    'missing_value': '0',
                }

    def reset(self):
        """Reset the schema manager."""
        self.columns.clear()
        self.column_docs.clear()
        self.dtypes.clear()
        self._seen_units.clear()
        self._seen_buildings.clear()
        self._add_base_columns()
        logger.info("SchemaManager reset")

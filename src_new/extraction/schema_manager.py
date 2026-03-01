"""
SchemaManager: Manages wide-table column schema and documentation.

This component handles schema generation, column ordering, data types,
and documentation for the wide-format parquet output.

Schema is built in two stages:
- Static columns (game metadata, economy) are created upfront from replay
  metadata via build_base_schema(), before extraction begins.
- Entity columns (units and buildings) are added dynamically during the game
  loop via ensure_unit_columns() and ensure_building_columns(), as each new
  entity is encountered for the first time.
- Upgrade columns are registered dynamically via add_upgrade_column() as each
  new upgrade is discovered during extraction.

Key behaviors:
- Only creates columns for units that actually complete (not cancelled/interrupted)
- Creates columns for ALL buildings (even cancelled ones, since position is meaningful)
- Conditionally includes shields (Protoss only) and energy (casters only) per entity
- No separate lifecycle columns (status, progress, started_loop, completed_loop,
  destroyed_loop) - lifecycle state is embedded in attribute columns
- Upgrade columns use dtype 'object' to hold numeric values (0, 1) and lifecycle
  strings ('started', 'cancelled')
"""

from typing import List, Dict, Any, Set, Optional
from pathlib import Path
import json
import logging
import re

import numpy as np

# ---------------------------------------------------------------------------
# Auto-derive attribute lists from the canonical FIELD_CONFIG definitions in
# the extractor modules. This eliminates hardcoded duplication: if a field is
# added/removed/reordered in UNIT_FIELD_CONFIG or BUILDING_FIELD_CONFIG, the
# schema attribute lists update automatically.
# ---------------------------------------------------------------------------
from src_new.extractors.unit_extractor import UNIT_FIELD_CONFIG
from src_new.extractors.building_extractor import BUILDING_FIELD_CONFIG


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


def _derive_base_attributes(field_config: List[Dict[str, Any]]) -> List[tuple]:
    """
    Derive the "always present" attribute tuples from a FIELD_CONFIG list.

    Filters the config to entries where always=True and produces
    (column_suffix, 'object', description) tuples used by SchemaManager
    when creating entity columns.

    Args:
        field_config: A FIELD_CONFIG list (e.g. UNIT_FIELD_CONFIG or
                      BUILDING_FIELD_CONFIG) -- list of dicts with keys
                      'column_suffix', 'always', 'description', etc.

    Returns:
        List of (column_suffix, dtype_str, description) tuples for always-on fields.
    """
    return [
        (entry['column_suffix'], 'object', entry.get('description', entry['column_suffix']))
        for entry in field_config
        if entry.get('always', False)
    ]


def _derive_conditional_attributes(
    field_config: List[Dict[str, Any]],
    suffixes: Set[str],
) -> List[tuple]:
    """
    Derive conditional attribute tuples from a FIELD_CONFIG list.

    Filters the config to entries where always=False AND column_suffix is in
    the given set of suffixes, producing (column_suffix, 'object', description)
    tuples.

    Args:
        field_config: A FIELD_CONFIG list (e.g. UNIT_FIELD_CONFIG).
        suffixes: Set of column_suffix strings to include (e.g. {'shields',
                  'shield_upgrade_level'}).

    Returns:
        List of (column_suffix, dtype_str, description) tuples for matching
        conditional fields.
    """
    return [
        (entry['column_suffix'], 'object', entry.get('description', entry['column_suffix']))
        for entry in field_config
        if not entry.get('always', False) and entry['column_suffix'] in suffixes
    ]


# ---------------------------------------------------------------------------
# Unit attribute lists -- auto-derived from UNIT_FIELD_CONFIG
# ---------------------------------------------------------------------------

# Base attributes: always=True entries from UNIT_FIELD_CONFIG
# (e.g. pos_(X,Y,Z), health, facing, radius, build_progress, ...)
UNIT_BASE_ATTRIBUTES = _derive_base_attributes(UNIT_FIELD_CONFIG)

# Shield attributes: conditional entries with suffix 'shields' or 'shield_upgrade_level'
# These are added only for Protoss units that have shield_max > 0.
UNIT_SHIELD_ATTRIBUTES = _derive_conditional_attributes(
    UNIT_FIELD_CONFIG, {'shields', 'shield_upgrade_level'}
)

# Energy attributes: conditional entry with suffix 'energy'
# Added only for caster units that have energy_max > 0.
UNIT_ENERGY_ATTRIBUTES = _derive_conditional_attributes(
    UNIT_FIELD_CONFIG, {'energy'}
)

# ---------------------------------------------------------------------------
# Building attribute lists -- auto-derived from BUILDING_FIELD_CONFIG
# ---------------------------------------------------------------------------

# Base attributes: always=True entries from BUILDING_FIELD_CONFIG
# (e.g. pos_(X,Y,Z), health, facing, build_progress, ...)
BUILDING_BASE_ATTRIBUTES = _derive_base_attributes(BUILDING_FIELD_CONFIG)

# Shield attributes: conditional entries with suffix 'shields' or 'shield_upgrade_level'
BUILDING_SHIELD_ATTRIBUTES = _derive_conditional_attributes(
    BUILDING_FIELD_CONFIG, {'shields', 'shield_upgrade_level'}
)

# Energy attributes: conditional entry with suffix 'energy'
BUILDING_ENERGY_ATTRIBUTES = _derive_conditional_attributes(
    BUILDING_FIELD_CONFIG, {'energy'}
)


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
        known before extraction begins: game metadata and economy columns.
        Unit and building columns are added dynamically during extraction via
        ensure_unit_columns() and ensure_building_columns().
        Upgrade columns are registered dynamically via add_upgrade_column() as each
        new upgrade is discovered during extraction.

        Call this AFTER get_replay_info() has returned player names, and BEFORE
        creating the WideTableBuilder or starting the game loop.

        Args:
            player_names: Dict mapping player number to raw player name,
                          e.g. {1: "Really", 2: "What!"}

        Calls:
            - self.set_player_names(player_names) -- sanitizes and stores names
            - self._add_economy_columns() -- adds p1_minerals, p1_vespene, etc.

        Note:
            _add_base_columns() is already called in __init__() and does not need
            to be called here. The static base columns (game_loop, timestamp_seconds,
            Messages) are present from object creation.

            Upgrade columns are no longer pre-registered here. They are added
            dynamically via add_upgrade_column() when each upgrade is first
            encountered during extraction.
        """
        self.set_player_names(player_names)
        self._add_economy_columns()
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
        """
        Add economy columns for both players.

        Only registers columns that are available from s2protocol's
        SPlayerStatsEvent tracker events. Columns that were previously
        sourced from the engine's player_common (idle_workers, army_count,
        food_army, warp_gate_count, larva_count, workers) and score_details
        (collected_minerals, collected_vespene, spent_minerals, spent_vespene)
        have been removed because they are always zero in observer mode.
        """
        for player_num in [1, 2]:
            # These columns map 1:1 to SPlayerStatsEvent fields parsed by
            # economy_extractor.load_economy_snapshots().
            economy_columns = [
                ('minerals', 'int64', 'Current unspent minerals'),
                ('vespene', 'int64', 'Current unspent vespene gas'),
                ('supply_used', 'float64', 'Supply currently used (fixed-point / 4096)'),
                ('supply_cap', 'float64', 'Supply capacity (fixed-point / 4096)'),
                ('collection_rate_minerals', 'float64', 'Mineral collection rate per minute'),
                ('collection_rate_vespene', 'float64', 'Vespene collection rate per minute'),
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

    def add_upgrade_column(self, player: str, upgrade_name: str) -> None:
        """
        Dynamically register a single upgrade column for a player.

        Called during extraction whenever a new upgrade is first encountered.
        This replaces the old _add_upgrade_columns() which hardcoded three
        upgrade names. Upgrades are now discovered at runtime, so any upgrade
        in the game (race-specific or otherwise) gets a column automatically.

        The column dtype is 'object' because upgrade values can be:
        - 0           : upgrade not started
        - 1           : upgrade completed
        - 'started'   : upgrade research in progress
        - 'cancelled' : upgrade research was cancelled

        This method is idempotent -- calling it multiple times with the same
        player + upgrade_name is safe and will not create duplicate columns.

        Args:
            player: Player prefix string, e.g. 'p1' or 'p2'.
                    The player number is extracted from this string.
            upgrade_name: Human-readable upgrade name, e.g. 'TerranInfantryWeaponsLevel1'.
                          Will be lowercased for the column name.

        Depends on:
            - self.columns: master column list
            - self.dtypes: column dtype registry
            - self.column_docs: column documentation registry
        """
        # Extract player number from prefix (e.g. "p1" -> 1)
        player_num = int(player[1:])

        col_name = f'p{player_num}_upgrade_{upgrade_name.lower()}'

        # Idempotent: only add if not already registered
        if col_name not in self.columns:
            self.columns.append(col_name)
            self.dtypes[col_name] = 'object'
            self.column_docs[col_name] = {
                'description': (
                    f'Upgrade status for {upgrade_name}: '
                    f'0=not started, 1=completed, '
                    f"'started'=in progress, 'cancelled'=cancelled"
                ),
                'type': 'object',
                'missing_value': '0',
            }
            logger.debug(
                f"Registered upgrade column: {col_name} for player {player_num}"
            )

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

    def reset(self):
        """Reset the schema manager."""
        self.columns.clear()
        self.column_docs.clear()
        self.dtypes.clear()
        self._seen_units.clear()
        self._seen_buildings.clear()
        self._add_base_columns()
        logger.info("SchemaManager reset")

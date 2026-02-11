"""
SchemaManager: Manages wide-table column schema and documentation.

This component handles schema generation, column ordering, data types,
and documentation for the wide-format parquet output.

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


# Base attribute columns for units (always present)
UNIT_BASE_ATTRIBUTES = [
    ('x', 'object', 'X-coordinate'),
    ('y', 'object', 'Y-coordinate'),
    ('z', 'object', 'Z-coordinate (height)'),
    ('facing', 'object', 'Facing direction'),
    ('health', 'object', 'Current health'),
    ('health_max', 'object', 'Maximum health'),
    ('build_progress', 'object', 'Build progress (0.0 to 1.0)'),
    ('is_flying', 'object', 'Whether unit is flying'),
    ('is_burrowed', 'object', 'Whether unit is burrowed'),
    ('is_hallucination', 'object', 'Whether unit is a hallucination'),
    ('weapon_cooldown', 'object', 'Weapon cooldown'),
    ('attack_upgrade_level', 'object', 'Attack upgrade level'),
    ('armor_upgrade_level', 'object', 'Armor upgrade level'),
    ('radius', 'object', 'Unit radius'),
    ('cargo_space_taken', 'object', 'Cargo space taken'),
    ('cargo_space_max', 'object', 'Maximum cargo space'),
    ('order_count', 'object', 'Number of queued orders'),
]

# Conditional attribute columns for units (added only when applicable)
UNIT_SHIELD_ATTRIBUTES = [
    ('shields', 'object', 'Current shields (Protoss only)'),
    ('shields_max', 'object', 'Maximum shields (Protoss only)'),
    ('shield_upgrade_level', 'object', 'Shield upgrade level (Protoss only)'),
]

UNIT_ENERGY_ATTRIBUTES = [
    ('energy', 'object', 'Current energy (casters only)'),
    ('energy_max', 'object', 'Maximum energy (casters only)'),
]

# Base attribute columns for buildings (always present)
BUILDING_BASE_ATTRIBUTES = [
    ('x', 'object', 'X-coordinate'),
    ('y', 'object', 'Y-coordinate'),
    ('z', 'object', 'Z-coordinate'),
    ('facing', 'object', 'Facing direction'),
    ('health', 'object', 'Current health'),
    ('health_max', 'object', 'Maximum health'),
    ('build_progress', 'object', 'Build progress (0.0 to 1.0)'),
    ('is_flying', 'object', 'Whether building is flying (lifted Terran)'),
    ('is_burrowed', 'object', 'Whether building is burrowed'),
    ('attack_upgrade_level', 'object', 'Attack upgrade level'),
    ('armor_upgrade_level', 'object', 'Armor upgrade level'),
    ('radius', 'object', 'Building radius'),
    ('order_count', 'object', 'Number of queued orders'),
]

# Conditional attribute columns for buildings
BUILDING_SHIELD_ATTRIBUTES = [
    ('shields', 'object', 'Current shields (Protoss only)'),
    ('shields_max', 'object', 'Maximum shields (Protoss only)'),
    ('shield_upgrade_level', 'object', 'Shield upgrade level (Protoss only)'),
]

BUILDING_ENERGY_ATTRIBUTES = [
    ('energy', 'object', 'Current energy (casters only)'),
    ('energy_max', 'object', 'Maximum energy (casters only)'),
]


class SchemaManager:
    """
    Manages wide-table column schema and documentation.

    This class dynamically builds the schema by scanning replays and provides
    comprehensive documentation for all columns.

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

    def build_schema_from_replay(
        self,
        replay_path: Path,
        replay_loader: Any,
        state_extractor: Any
    ) -> None:
        """
        Pre-scan replay to determine all columns needed.

        This performs a full pass through the replay to discover all unit types,
        building types, and other dynamic columns that will be needed.

        After the scan, only units that actually completed (build_progress >= 1.0)
        will have columns created. Buildings always get columns (even if cancelled).

        Args:
            replay_path: Path to replay file
            replay_loader: ReplayLoader instance
            state_extractor: StateExtractor instance
        """
        logger.info(f"Pre-scanning replay to build schema: {replay_path}")

        # Load replay
        replay_loader.load_replay(replay_path)

        with replay_loader.start_sc2_instance() as controller:
            metadata = replay_loader.get_replay_info(controller)

            # Extract player names from metadata and set on schema manager
            player_names = {
                p['player_id']: p.get('player_name', '')
                for p in metadata.get('players', [])
            }
            self.set_player_names(player_names)

            replay_loader.start_replay(controller, observed_player_id=1, disable_fog=True)

            # Iterate through replay to discover entities
            game_loop = 0
            max_loops = metadata['game_duration_loops']

            while game_loop < max_loops:
                try:
                    # Step forward
                    controller.step(1)
                    obs = controller.observe()

                    # Check if replay has ended
                    if obs.player_result:
                        logger.info(f"Schema scan complete - replay ended at loop {game_loop} (expected {max_loops})")
                        break

                    # Extract state
                    state = state_extractor.extract_observation(obs, game_loop)

                    # We don't add columns yet - just let the extractors track everything
                    game_loop = obs.observation.game_loop

                except Exception as e:
                    logger.warning(f"Error during schema scan at loop {game_loop}: {e}")
                    break

        # Now build the schema based on what was discovered during the scan
        self._build_columns_from_extractors(state_extractor)

        logger.info(f"Schema built with {len(self.columns)} columns")
        logger.info(f"  Units discovered: {len(self._seen_units)}")
        logger.info(f"  Buildings discovered: {len(self._seen_buildings)}")

    def _build_columns_from_extractors(self, state_extractor: Any) -> None:
        """
        Build column schema from extractor state after pass 1 completes.

        Only creates columns for:
        - Units that actually completed (build_progress reached 1.0)
        - All buildings (even cancelled ones)

        Conditionally includes shields/energy based on what the extractor observed.

        Args:
            state_extractor: StateExtractor instance (after pass 1)
        """
        for player_num in [1, 2]:
            player = f'p{player_num}'

            # Get unit extractor for this player
            unit_extractor = state_extractor.unit_extractors[player_num]
            completed_ids = unit_extractor.get_completed_readable_ids()

            # Add columns only for completed units
            for readable_id in sorted(completed_ids):
                if readable_id not in self._seen_units:
                    self._seen_units.add(readable_id)
                    extra_attrs = unit_extractor.get_unit_attributes_for_id(readable_id)
                    self.add_unit_columns(player, readable_id, extra_attrs)

            # Get building extractor for this player - add ALL buildings
            building_extractor = state_extractor.building_extractors[player_num]
            for tag, readable_id in sorted(building_extractor.tag_to_readable_id.items(),
                                            key=lambda x: x[1]):
                if readable_id not in self._seen_buildings:
                    self._seen_buildings.add(readable_id)
                    extra_attrs = building_extractor.get_building_attributes_for_id(readable_id)
                    self.add_building_columns(player, readable_id, extra_attrs)

        # Add economy columns (fixed schema)
        self._add_economy_columns()

        # Add upgrade columns (dynamic but can be pre-defined)
        self._add_upgrade_columns()

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

        # Conditionally add shields
        if any(a in extra_attrs for a in ['shields', 'shields_max', 'shield_upgrade_level']):
            unit_columns.extend(UNIT_SHIELD_ATTRIBUTES)

        # Conditionally add energy
        if any(a in extra_attrs for a in ['energy', 'energy_max']):
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

        # Conditionally add shields
        if any(a in extra_attrs for a in ['shields', 'shields_max', 'shield_upgrade_level']):
            building_columns.extend(BUILDING_SHIELD_ATTRIBUTES)

        # Conditionally add energy
        if any(a in extra_attrs for a in ['energy', 'energy_max']):
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

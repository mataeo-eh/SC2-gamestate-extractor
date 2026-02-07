"""
SchemaManager: Manages wide-table column schema and documentation.

This component handles schema generation, column ordering, data types,
and documentation for the wide-format parquet output.
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


class SchemaManager:
    """
    Manages wide-table column schema and documentation.

    This class dynamically builds the schema by scanning replays and provides
    comprehensive documentation for all columns.
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

        Args:
            replay_path: Path to replay file
            replay_loader: ReplayLoader instance
            state_extractor: StateExtractor instance

        # TODO: Test case - Generate schema from sample replay
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

            replay_loader.start_replay(controller, observed_player_id=1)

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
                        # player_result only appears at the end of the match when one player has won
                        logger.info(f"Schema scan complete - replay ended at loop {game_loop} (expected {max_loops})")
                        break


                    # Extract state
                    state = state_extractor.extract_observation(obs, game_loop)

                    # Discover units and buildings
                    self._discover_entities_from_state(state)

                    game_loop = obs.observation.game_loop

                except Exception as e:
                    logger.warning(f"Error during schema scan at loop {game_loop}: {e}")
                    break

        logger.info(f"Schema built with {len(self.columns)} columns")
        logger.info(f"  Units discovered: {len(self._seen_units)}")
        logger.info(f"  Buildings discovered: {len(self._seen_buildings)}")

    def _discover_entities_from_state(self, state: Dict[str, Any]) -> None:
        """
        Discover entities from extracted state and add to schema.

        Args:
            state: Extracted state dictionary
        """
        # Discover units
        for player_num in [1, 2]:
            units_key = f'p{player_num}_units'
            if units_key in state:
                for unit_id, unit_data in state[units_key].items():
                    if unit_id not in self._seen_units:
                        self._seen_units.add(unit_id)
                        self.add_unit_columns(f'p{player_num}', unit_id, unit_data)

        # Discover buildings
        for player_num in [1, 2]:
            buildings_key = f'p{player_num}_buildings'
            if buildings_key in state:
                for building_id, building_data in state[buildings_key].items():
                    if building_id not in self._seen_buildings:
                        self._seen_buildings.add(building_id)
                        self.add_building_columns(f'p{player_num}', building_id, building_data)

        # Add economy columns (fixed schema)
        self._add_economy_columns()

        # Add upgrade columns (dynamic but can be pre-defined)
        self._add_upgrade_columns()

    def add_unit_columns(self, player: str, unit_id: str, unit_data: Dict) -> None:
        """
        Add columns for a specific unit.

        Args:
            player: Player prefix (e.g., 'p1', 'p2')
            unit_id: Unit identifier (e.g., 'marine_001')
            unit_data: Unit data dictionary for determining columns

        # TODO: Test case - Add unit columns dynamically
        """
        # Get unit type name for documentation
        unit_type_name = unit_data.get('unit_type_name', 'unknown')

        # Define unit columns
        unit_columns = [
            ('x', 'float64', 'X-coordinate'),
            ('y', 'float64', 'Y-coordinate'),
            ('z', 'float64', 'Z-coordinate (height)'),
            ('health', 'float64', 'Current health'),
            ('health_max', 'float64', 'Maximum health'),
            ('shields', 'float64', 'Current shields'),
            ('shields_max', 'float64', 'Maximum shields'),
            ('energy', 'float64', 'Current energy'),
            ('energy_max', 'float64', 'Maximum energy'),
            ('state', 'string', 'Unit state (built/existing/killed)'),
        ]

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
                    'description': f'{description} for {player} {unit_type_name} {unit_id}',
                    'type': dtype,
                    'missing_value': 'NaN' if dtype.startswith('float') or dtype.startswith('int') else 'null',
                }

    def add_building_columns(self, player: str, building_id: str, building_data: Dict) -> None:
        """
        Add columns for a specific building.

        Args:
            player: Player prefix (e.g., 'p1', 'p2')
            building_id: Building identifier
            building_data: Building data dictionary

        # TODO: Test case - Add building columns dynamically
        """
        building_type = building_data.get('building_type', 'unknown')

        # Define building columns
        building_columns = [
            ('x', 'float64', 'X-coordinate'),
            ('y', 'float64', 'Y-coordinate'),
            ('z', 'float64', 'Z-coordinate'),
            ('status', 'string', 'Building status (started/building/completed/destroyed)'),
            ('progress', 'int64', 'Construction progress (0-100)'),
            ('started_loop', 'int64', 'Game loop when construction started'),
            ('completed_loop', 'int64', 'Game loop when construction completed'),
            ('destroyed_loop', 'int64', 'Game loop when building destroyed'),
        ]

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
                    'description': f'{description} for {player} {building_type} {building_id}',
                    'type': dtype,
                    'missing_value': 'NaN' if dtype.startswith('float') or dtype.startswith('int') else 'null',
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
            Dictionary mapping column names to documentation:
            {
                'column_name': {
                    'description': str,
                    'type': str,
                    'example': Any,
                    'missing_value': str,
                },
                ...
            }

        # TODO: Test case - Generate documentation
        """
        return self.column_docs.copy()

    def save_schema(self, output_path: Path) -> None:
        """
        Save schema to JSON file.

        Args:
            output_path: Path to save schema JSON

        # TODO: Test case - Save/load schema
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
            return None

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

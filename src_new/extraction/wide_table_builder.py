"""
WideTableBuilder: Transforms extracted state into wide-format rows.

This component takes the extracted game state and converts it into wide-format
rows suitable for parquet storage and ML pipelines.

Lifecycle state embedding:
- Unit attribute columns contain lifecycle state strings at transition gameloops
  ("unit_started", "building", "completed", "destroyed") instead of separate
  lifecycle columns.
- Building attribute columns contain lifecycle state strings at transition gameloops
  ("building_started", "completed", "destroyed", "cancelled"). During construction,
  buildings capture real data (position, health) since this is strategically meaningful.
"""

from typing import Dict, Any, List, Set
import logging

import numpy as np

from .schema_manager import SchemaManager, UNIT_BASE_ATTRIBUTES, UNIT_SHIELD_ATTRIBUTES, \
    UNIT_ENERGY_ATTRIBUTES, BUILDING_BASE_ATTRIBUTES, BUILDING_SHIELD_ATTRIBUTES, \
    BUILDING_ENERGY_ATTRIBUTES


logger = logging.getLogger(__name__)


# Lifecycle states where ALL unit attribute columns should be filled with the state string
UNIT_LIFECYCLE_OVERRIDE_STATES = {'unit_started', 'building', 'completed', 'destroyed'}

# Lifecycle states where ALL building attribute columns should be filled with the state string
# (not 'under_construction' - buildings get real data during construction)
BUILDING_LIFECYCLE_OVERRIDE_STATES = {'building_started', 'completed', 'destroyed', 'cancelled'}


class WideTableBuilder:
    """
    Transforms extracted state into wide-format rows.

    This class takes the hierarchical state extracted by StateExtractor and
    flattens it into wide-format rows with one column per entity attribute.
    """

    def __init__(self, schema: SchemaManager):
        """
        Initialize with schema.

        Args:
            schema: SchemaManager instance defining columns
        """
        self.schema = schema
        # Player name mapping: {player_num: sanitized_name} e.g. {1: "really", 2: "what"}
        self.player_names: Dict[int, str] = {}
        logger.info("WideTableBuilder initialized")

    def set_player_names(self, player_names: Dict[int, str]) -> None:
        """
        Store a mapping of player numbers to sanitized bot/player names.

        Args:
            player_names: Dict mapping player number to raw name,
                          e.g. {1: "Really", 2: "What!"}
        """
        from .schema_manager import sanitize_name
        self.player_names = {
            num: sanitize_name(name) for num, name in player_names.items()
        }
        logger.info(f"Player names set: {self.player_names}")

    def build_row(self, extracted_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform extracted state to wide-format row.

        Args:
            extracted_state: State dictionary from StateExtractor.extract_observation()

        Returns:
            Dictionary with all columns, NaN for missing values.
        """
        # Initialize row with all columns as missing values
        row = {}
        for col in self.schema.get_column_list():
            row[col] = self.schema.get_missing_value(col)

        # Add base columns
        game_loop = extracted_state.get('game_loop', 0)
        row['game_loop'] = game_loop
        row['timestamp_seconds'] = game_loop / 22.4  # Convert to seconds

        # Add units for both players
        for player_num in [1, 2]:
            units_key = f'p{player_num}_units'
            if units_key in extracted_state:
                units = extracted_state[units_key]
                for unit_id, unit_data in units.items():
                    self.add_unit_to_row(row, f'p{player_num}', unit_id, unit_data)

        # Add buildings for both players
        for player_num in [1, 2]:
            buildings_key = f'p{player_num}_buildings'
            if buildings_key in extracted_state:
                buildings = extracted_state[buildings_key]
                for building_id, building_data in buildings.items():
                    self.add_building_to_row(row, f'p{player_num}', building_id, building_data)

        # Add economy for both players
        for player_num in [1, 2]:
            economy_key = f'p{player_num}_economy'
            if economy_key in extracted_state:
                economy = extracted_state[economy_key]
                self.add_economy_to_row(row, f'p{player_num}', economy)

        # Add upgrades for both players
        for player_num in [1, 2]:
            upgrades_key = f'p{player_num}_upgrades'
            if upgrades_key in extracted_state:
                upgrades = extracted_state[upgrades_key]
                self.add_upgrades_to_row(row, f'p{player_num}', upgrades)

        # Add unit counts
        for player_num in [1, 2]:
            units_key = f'p{player_num}_units'
            if units_key in extracted_state:
                units = extracted_state[units_key]
                unit_counts = self.calculate_unit_counts(units)
                self.add_unit_counts_to_row(row, f'p{player_num}', unit_counts)

        # Add messages
        messages = extracted_state.get('messages', [])
        if 'Messages' in row:
            row['Messages'] = self._format_messages(messages)

        return row

    def _get_column_prefix(self, player: str, entity_id: str) -> str:
        """
        Build the column name prefix for a unit or building.

        Args:
            player: Player prefix (e.g., 'p1', 'p2')
            entity_id: Entity identifier (e.g., 'p1_marine_001')

        Returns:
            Column prefix string (e.g., 'p1_botname_marine_001')
        """
        stripped_id = '_'.join(entity_id.split('_')[1:]) if entity_id.startswith('p') else entity_id
        player_num = int(player[1:])
        bot_name = self.player_names.get(player_num, player)
        return f'{player}_{bot_name}_{stripped_id}'

    def add_unit_to_row(
        self,
        row: Dict[str, Any],
        player: str,
        unit_id: str,
        unit_data: Dict[str, Any]
    ) -> None:
        """
        Add unit data to row with lifecycle state embedded in attribute columns.

        Lifecycle behavior:
        - 'unit_started': ALL attribute columns filled with "unit_started"
        - 'building': ALL attribute columns filled with "building"
        - 'completed': ALL attribute columns filled with "completed"
        - 'existing': Real data values written to attribute columns
        - 'destroyed': ALL attribute columns filled with "destroyed"
        - (no lifecycle / unit not in schema): columns remain NaN

        Args:
            row: Row dictionary to modify
            player: Player prefix (e.g., 'p1', 'p2')
            unit_id: Unit identifier
            unit_data: Unit data dictionary (with '_lifecycle' key)
        """
        prefix = self._get_column_prefix(player, unit_id)
        lifecycle = unit_data.get('_lifecycle', 'existing')

        # Determine which attribute suffixes exist for this unit in the schema
        attr_suffixes = self._get_unit_attr_suffixes_in_schema(prefix, row)

        if not attr_suffixes:
            # This unit has no columns in the schema (e.g., never completed)
            return

        if lifecycle in UNIT_LIFECYCLE_OVERRIDE_STATES:
            # Fill ALL attribute columns with the lifecycle state string
            for suffix in attr_suffixes:
                col_name = f'{prefix}_{suffix}'
                if col_name in row:
                    row[col_name] = lifecycle
        else:
            # 'existing' - write real data
            for suffix in attr_suffixes:
                col_name = f'{prefix}_{suffix}'
                if col_name in row and suffix in unit_data:
                    row[col_name] = unit_data[suffix]

    def add_building_to_row(
        self,
        row: Dict[str, Any],
        player: str,
        building_id: str,
        building_data: Dict[str, Any]
    ) -> None:
        """
        Add building data to row with lifecycle state embedded in attribute columns.

        Lifecycle behavior:
        - 'building_started': ALL attribute columns filled with "building_started"
        - 'under_construction': Real data written (position, health during construction)
        - 'completed': ALL attribute columns filled with "completed"
        - 'existing': Real data values written to attribute columns
        - 'destroyed': ALL attribute columns filled with "destroyed"
        - 'cancelled': ALL attribute columns filled with "cancelled"

        Args:
            row: Row dictionary to modify
            player: Player prefix
            building_id: Building identifier
            building_data: Building data dictionary (with '_lifecycle' key)
        """
        prefix = self._get_column_prefix(player, building_id)
        lifecycle = building_data.get('_lifecycle', 'existing')

        # Determine which attribute suffixes exist for this building in the schema
        attr_suffixes = self._get_building_attr_suffixes_in_schema(prefix, row)

        if not attr_suffixes:
            return

        if lifecycle in BUILDING_LIFECYCLE_OVERRIDE_STATES:
            # Fill ALL attribute columns with the lifecycle state string
            for suffix in attr_suffixes:
                col_name = f'{prefix}_{suffix}'
                if col_name in row:
                    row[col_name] = lifecycle
        else:
            # 'under_construction' or 'existing' - write real data
            for suffix in attr_suffixes:
                col_name = f'{prefix}_{suffix}'
                if col_name in row and suffix in building_data:
                    row[col_name] = building_data[suffix]

    def _get_unit_attr_suffixes_in_schema(self, prefix: str, row: Dict[str, Any]) -> List[str]:
        """
        Get the list of attribute suffixes for a unit that exist in the schema.

        Checks base attributes plus conditional shield/energy attributes.

        Args:
            prefix: Column name prefix (e.g., 'p1_botname_marine_001')
            row: Row dictionary (to check which columns exist)

        Returns:
            List of attribute suffix strings that have columns in the schema
        """
        suffixes = []
        all_possible = (
            UNIT_BASE_ATTRIBUTES
            + UNIT_SHIELD_ATTRIBUTES
            + UNIT_ENERGY_ATTRIBUTES
        )
        for suffix, _, _ in all_possible:
            col_name = f'{prefix}_{suffix}'
            if col_name in row:
                suffixes.append(suffix)
        return suffixes

    def _get_building_attr_suffixes_in_schema(self, prefix: str, row: Dict[str, Any]) -> List[str]:
        """
        Get the list of attribute suffixes for a building that exist in the schema.

        Args:
            prefix: Column name prefix
            row: Row dictionary

        Returns:
            List of attribute suffix strings that have columns in the schema
        """
        suffixes = []
        all_possible = (
            BUILDING_BASE_ATTRIBUTES
            + BUILDING_SHIELD_ATTRIBUTES
            + BUILDING_ENERGY_ATTRIBUTES
        )
        for suffix, _, _ in all_possible:
            col_name = f'{prefix}_{suffix}'
            if col_name in row:
                suffixes.append(suffix)
        return suffixes

    def add_economy_to_row(
        self,
        row: Dict[str, Any],
        player: str,
        economy_data: Dict[str, Any]
    ) -> None:
        """
        Add economy data to row.

        Args:
            row: Row dictionary to modify
            player: Player prefix
            economy_data: Economy data dictionary
        """
        # Must match the economy columns registered in schema_manager._add_economy_columns()
        # and the keys returned by economy_extractor.get_economy_at_loop().
        economy_columns = [
            'minerals', 'vespene',
            'supply_used', 'supply_cap',
            'collection_rate_minerals', 'collection_rate_vespene',
        ]

        for attr in economy_columns:
            col_name = f'{player}_{attr}'
            if col_name in row:
                row[col_name] = economy_data.get(attr, self.schema.get_missing_value(col_name))

    def add_upgrades_to_row(
        self,
        row: Dict[str, Any],
        player: str,
        upgrades_data: Dict[str, Any]
    ) -> None:
        """
        Add upgrades data to row.

        Args:
            row: Row dictionary to modify
            player: Player prefix
            upgrades_data: Upgrades data dictionary
        """
        # Map upgrade names to column names
        upgrade_mapping = {
            'attack_level': f'{player}_upgrade_attack_level',
            'armor_level': f'{player}_upgrade_armor_level',
            'shield_level': f'{player}_upgrade_shield_level',
        }

        for upgrade_name, col_name in upgrade_mapping.items():
            if col_name in row:
                row[col_name] = upgrades_data.get(upgrade_name, 0)

    def add_unit_counts_to_row(
        self,
        row: Dict[str, Any],
        player: str,
        unit_counts: Dict[str, int]
    ) -> None:
        """
        Add unit counts to row.

        Args:
            row: Row dictionary to modify
            player: Player prefix
            unit_counts: Dictionary mapping unit types to counts
        """
        for unit_type, count in unit_counts.items():
            col_name = f'{player}_{unit_type.lower()}_count'
            if col_name in row:
                row[col_name] = count

    def calculate_unit_counts(self, units: Dict[str, Dict]) -> Dict[str, int]:
        """
        Calculate unit counts by type.

        Args:
            units: Dictionary of units from extracted state

        Returns:
            Dictionary mapping unit type names to counts.
        """
        counts = {}

        for unit_id, unit_data in units.items():
            # Skip destroyed/building units for count
            lifecycle = unit_data.get('_lifecycle', 'existing')
            if lifecycle in ('destroyed', 'unit_started', 'building'):
                continue

            # Get unit type
            unit_type_name = unit_data.get('unit_type_name')
            if unit_type_name:
                counts[unit_type_name] = counts.get(unit_type_name, 0) + 1

        return counts

    def build_rows_batch(self, extracted_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build multiple rows from a batch of extracted states.

        Args:
            extracted_states: List of state dictionaries

        Returns:
            List of row dictionaries
        """
        rows = []
        for state in extracted_states:
            try:
                row = self.build_row(state)
                rows.append(row)
            except Exception as e:
                logger.error(f"Error building row for game_loop {state.get('game_loop', '?')}: {e}")

        return rows

    def validate_row(self, row: Dict[str, Any]) -> bool:
        """
        Validate that row has all required columns.

        Args:
            row: Row dictionary

        Returns:
            True if valid, False otherwise
        """
        schema_columns = set(self.schema.get_column_list())
        row_columns = set(row.keys())

        missing_columns = schema_columns - row_columns
        extra_columns = row_columns - schema_columns

        if missing_columns:
            logger.warning(f"Row missing {len(missing_columns)} columns: {list(missing_columns)[:10]}")
            return False

        if extra_columns:
            logger.warning(f"Row has {len(extra_columns)} extra columns: {list(extra_columns)[:10]}")
            return False

        return True

    def _format_messages(self, messages: List[Dict[str, Any]]) -> Any:
        """
        Format messages for storage in the Messages column.

        Args:
            messages: List of message dictionaries from state extractor

        Returns:
            - NaN if no messages
            - String if one message
            - JSON-serialized string if multiple messages
        """
        if not messages:
            return np.nan

        # Extract just the message text
        message_texts = [msg.get('message', '') for msg in messages]

        # Always return a string to avoid mixed types in parquet column.
        if len(message_texts) == 1:
            return message_texts[0]
        else:
            import json
            return json.dumps(message_texts)

    def get_row_summary(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get summary statistics for a row.

        Args:
            row: Row dictionary

        Returns:
            Summary dictionary with statistics
        """
        summary = {
            'game_loop': row.get('game_loop'),
            'timestamp_seconds': row.get('timestamp_seconds'),
            'total_columns': len(row),
            'missing_values': sum(1 for v in row.values() if v is None or (isinstance(v, float) and np.isnan(v))),
            'p1_minerals': row.get('p1_minerals'),
            'p2_minerals': row.get('p2_minerals'),
            'p1_supply_used': row.get('p1_supply_used'),
            'p2_supply_used': row.get('p2_supply_used'),
        }

        return summary

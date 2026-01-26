"""
WideTableBuilder: Transforms extracted state into wide-format rows.

This component takes the extracted game state and converts it into wide-format
rows suitable for parquet storage and ML pipelines.
"""

from typing import Dict, Any, List
import logging

import numpy as np

from .schema_manager import SchemaManager


logger = logging.getLogger(__name__)


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
        logger.info("WideTableBuilder initialized")

    def build_row(self, extracted_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform extracted state to wide-format row.

        Args:
            extracted_state: State dictionary from StateExtractor.extract_observation()

        Returns:
            Dictionary with all columns, NaN for missing values:
            {
                'game_loop': int,
                'timestamp_seconds': float,
                'p1_marine_001_x': float or NaN,
                'p1_marine_001_y': float or NaN,
                ...
            }

        # TODO: Test case - Build row from extracted state
        # TODO: Test case - Handle missing units (NaN values)
        # TODO: Test case - Validate row has all schema columns
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

    def add_unit_to_row(
        self,
        row: Dict[str, Any],
        player: str,
        unit_id: str,
        unit_data: Dict[str, Any]
    ) -> None:
        """
        Add unit data to row.

        Args:
            row: Row dictionary to modify
            player: Player prefix (e.g., 'p1', 'p2')
            unit_id: Unit identifier
            unit_data: Unit data dictionary

        # TODO: Test case - Add unit data to row
        # TODO: Test case - Handle dead units properly
        """
        # If unit is killed, only set state column
        if unit_data.get('state') == 'killed':
            state_col = f'{player}_{unit_id}_state'
            if state_col in row:
                row[state_col] = 'killed'
            return

        # Add all unit attributes
        unit_columns = [
            'x', 'y', 'z',
            'health', 'health_max',
            'shields', 'shields_max',
            'energy', 'energy_max',
            'state',
        ]

        for attr in unit_columns:
            col_name = f'{player}_{unit_id}_{attr}'
            if col_name in row:
                row[col_name] = unit_data.get(attr, self.schema.get_missing_value(col_name))

    def add_building_to_row(
        self,
        row: Dict[str, Any],
        player: str,
        building_id: str,
        building_data: Dict[str, Any]
    ) -> None:
        """
        Add building data to row.

        Args:
            row: Row dictionary to modify
            player: Player prefix
            building_id: Building identifier
            building_data: Building data dictionary

        # TODO: Test case - Add building lifecycle data
        """
        building_columns = [
            'x', 'y', 'z',
            'status', 'progress',
            'started_loop', 'completed_loop', 'destroyed_loop',
        ]

        for attr in building_columns:
            col_name = f'{player}_{building_id}_{attr}'
            if col_name in row:
                row[col_name] = building_data.get(attr, self.schema.get_missing_value(col_name))

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
        economy_columns = [
            'minerals', 'vespene',
            'supply_used', 'supply_cap',
            'workers', 'idle_workers',
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
            Dictionary mapping unit type names to counts:
            {'Marine': 25, 'Medivac': 3, ...}

        # TODO: Test case - Calculate unit counts correctly
        """
        counts = {}

        for unit_id, unit_data in units.items():
            # Skip killed units
            if unit_data.get('state') == 'killed':
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
                # Continue with next state

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
            - List of strings if multiple messages
        """
        if not messages:
            return np.nan

        # Extract just the message text
        message_texts = [msg.get('message', '') for msg in messages]

        # Return string if single message, list if multiple
        if len(message_texts) == 1:
            return message_texts[0]
        else:
            return message_texts

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

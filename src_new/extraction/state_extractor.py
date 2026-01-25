"""
StateExtractor: Extracts all required game state from pysc2 observations.

This component orchestrates the extraction of complete game state including
units, buildings, economy, upgrades, and messages from SC2 observations.
"""

from typing import Dict, Set, List, Optional, Any
from pathlib import Path
import logging

from ..extractors.unit_extractor import UnitExtractor
from ..extractors.building_extractor import BuildingExtractor
from ..extractors.economy_extractor import EconomyExtractor
from ..extractors.upgrade_extractor import UpgradeExtractor


logger = logging.getLogger(__name__)


class StateExtractor:
    """
    Extracts complete game state from pysc2 observations.

    This class orchestrates all the individual extractors (units, buildings,
    economy, upgrades) and provides a unified interface for extracting complete
    game state at each time step.
    """

    def __init__(self):
        """Initialize the StateExtractor with all component extractors."""
        # Create extractors for both players
        self.unit_extractors = {
            1: UnitExtractor(player_id=1),
            2: UnitExtractor(player_id=2),
        }

        self.building_extractors = {
            1: BuildingExtractor(player_id=1),
            2: BuildingExtractor(player_id=2),
        }

        self.economy_extractors = {
            1: EconomyExtractor(player_id=1),
            2: EconomyExtractor(player_id=2),
        }

        self.upgrade_extractors = {
            1: UpgradeExtractor(player_id=1),
            2: UpgradeExtractor(player_id=2),
        }

        # Initialize trackers
        self.unit_tracker = UnitTracker()
        self.building_tracker = BuildingTracker()

        logger.info("StateExtractor initialized")

    def extract_observation(self, obs, game_loop: int) -> Dict[str, Any]:
        """
        Extract complete state from single observation.

        Args:
            obs: SC2 observation from controller.observe()
            game_loop: Current game loop number

        Returns:
            Dictionary with extracted state:
            {
                'game_loop': int,
                'p1_units': dict,
                'p2_units': dict,
                'p1_buildings': dict,
                'p2_buildings': dict,
                'p1_economy': dict,
                'p2_economy': dict,
                'p1_upgrades': dict,
                'p2_upgrades': dict,
                'messages': list,
            }

        # TODO: Test case - Extract complete state from observation
        """
        state = {'game_loop': game_loop}

        # Extract units for both players
        state['p1_units'] = self.extract_units(obs, player_id=1)
        state['p2_units'] = self.extract_units(obs, player_id=2)

        # Extract buildings for both players
        state['p1_buildings'] = self.extract_buildings(obs, player_id=1)
        state['p2_buildings'] = self.extract_buildings(obs, player_id=2)

        # Extract economy for both players
        state['p1_economy'] = self.extract_economy(obs, player_id=1)
        state['p2_economy'] = self.extract_economy(obs, player_id=2)

        # Extract upgrades for both players
        state['p1_upgrades'] = self.extract_upgrades(obs, player_id=1)
        state['p2_upgrades'] = self.extract_upgrades(obs, player_id=2)

        # Extract messages
        state['messages'] = self.extract_messages(obs)

        return state

    def extract_units(self, obs, player_id: int) -> Dict[str, Dict]:
        """
        Extract all units for a player.

        Args:
            obs: SC2 observation
            player_id: Player ID (1 or 2)

        Returns:
            Dictionary mapping unit IDs to unit data

        # TODO: Test case - Extract units from observation
        # TODO: Test case - Track unit creation (state='built')
        # TODO: Test case - Track unit death (state='killed')
        # TODO: Test case - Track existing units (state='existing')
        """
        extractor = self.unit_extractors[player_id]
        units = extractor.extract(obs)
        return units

    def extract_buildings(self, obs, player_id: int) -> Dict[str, Dict]:
        """
        Extract all buildings for a player.

        Args:
            obs: SC2 observation
            player_id: Player ID (1 or 2)

        Returns:
            Dictionary mapping building IDs to building data including:
            - Position (x, y, z)
            - Status ('started', 'building', 'completed', 'destroyed')
            - Progress (0-100)
            - Construction timestamps

        # TODO: Test case - Extract building lifecycle
        # TODO: Test case - Track building construction progress
        # TODO: Test case - Handle cancelled buildings
        """
        extractor = self.building_extractors[player_id]
        buildings = extractor.extract(obs)
        return buildings

    def extract_economy(self, obs, player_id: int) -> Dict[str, Any]:
        """
        Extract economy metrics for a player.

        Args:
            obs: SC2 observation
            player_id: Player ID (1 or 2)

        Returns:
            Dictionary with economy data:
            {
                'minerals': int,
                'vespene': int,
                'supply_used': int,
                'supply_cap': int,
                'workers': int,
                'idle_workers': int,
                ...
            }

        # TODO: Test case - Extract economy metrics
        """
        extractor = self.economy_extractors[player_id]
        economy = extractor.extract(obs)
        return economy

    def extract_upgrades(self, obs, player_id: int) -> Dict[str, Any]:
        """
        Extract completed upgrades for a player.

        Args:
            obs: SC2 observation
            player_id: Player ID (1 or 2)

        Returns:
            Dictionary with upgrade data:
            {
                'upgrade_name': level,
                ...
            }

        # TODO: Test case - Extract upgrade levels
        """
        extractor = self.upgrade_extractors[player_id]
        upgrades = extractor.extract(obs)
        return upgrades

    def extract_messages(self, obs) -> List[Dict[str, Any]]:
        """
        Extract chat messages at this timestep.

        Args:
            obs: SC2 observation

        Returns:
            List of message dictionaries:
            [
                {
                    'game_loop': int,
                    'player_id': int,
                    'message': str,
                },
                ...
            ]

        # TODO: Test case - Extract messages
        """
        messages = []

        # Extract chat messages from observation
        # Note: Messages are in obs.observation.chat
        if hasattr(obs.observation, 'chat'):
            for msg in obs.observation.chat:
                messages.append({
                    'game_loop': obs.observation.game_loop,
                    'player_id': msg.player_id,
                    'message': msg.message,
                })

        return messages

    def reset(self):
        """Reset all extractors and trackers."""
        for extractor in self.unit_extractors.values():
            extractor.reset()
        for extractor in self.building_extractors.values():
            extractor.reset()
        for extractor in self.economy_extractors.values():
            extractor.reset()
        for extractor in self.upgrade_extractors.values():
            extractor.reset()

        self.unit_tracker.reset()
        self.building_tracker.reset()

        logger.info("StateExtractor reset")


class UnitTracker:
    """
    Tracks units across frames and assigns consistent IDs.

    This class maintains a registry of units seen across the replay and
    provides consistent ID assignment and state tracking.
    """

    def __init__(self):
        """Initialize the UnitTracker."""
        self.unit_registry: Dict[int, tuple] = {}  # tag -> (unit_type, id_num)
        self.unit_counters: Dict[int, int] = {}    # unit_type -> max_id
        self.previous_frame_tags: Set[int] = set()

        logger.debug("UnitTracker initialized")

    def process_units(self, raw_units, game_loop: int) -> Dict[str, Dict]:
        """
        Process raw units and return tracked units with states.

        Args:
            raw_units: Raw unit data from observation
            game_loop: Current game loop

        Returns:
            Dictionary mapping unit IDs to unit data:
            {
                'marine_001': {
                    'x': float, 'y': float, 'z': float,
                    'state': 'built'|'existing'|'killed',
                    'unit_type': str,
                    ...
                },
                ...
            }

        # TODO: Test case - Assign consistent IDs across frames
        # TODO: Test case - Detect state transitions
        """
        tracked_units = {}
        current_tags = set()

        # Process each unit
        for unit in raw_units:
            tag = unit.tag
            current_tags.add(tag)

            # Assign ID if new
            unit_id = self.assign_unit_id(tag, unit.unit_type)

            # Detect state
            state = self.detect_state(tag, current_tags)

            # Build tracked unit data
            tracked_units[unit_id] = {
                'tag': tag,
                'unit_type': unit.unit_type,
                'x': unit.pos.x,
                'y': unit.pos.y,
                'z': unit.pos.z,
                'state': state,
                'game_loop': game_loop,
            }

        # Detect killed units
        dead_tags = self.previous_frame_tags - current_tags
        for dead_tag in dead_tags:
            if dead_tag in self.unit_registry:
                unit_type, id_num = self.unit_registry[dead_tag]
                unit_id = f"unit_{unit_type}_{id_num:03d}"
                tracked_units[unit_id] = {
                    'tag': dead_tag,
                    'state': 'killed',
                    'game_loop': game_loop,
                }

        # Update for next frame
        self.previous_frame_tags = current_tags

        return tracked_units

    def assign_unit_id(self, tag: int, unit_type: int) -> str:
        """
        Assign or retrieve consistent ID for unit.

        Args:
            tag: SC2 unit tag
            unit_type: SC2 unit type ID

        Returns:
            Consistent unit ID string
        """
        if tag in self.unit_registry:
            unit_type, id_num = self.unit_registry[tag]
            return f"unit_{unit_type}_{id_num:03d}"

        # New unit - assign ID
        if unit_type not in self.unit_counters:
            self.unit_counters[unit_type] = 1

        id_num = self.unit_counters[unit_type]
        self.unit_counters[unit_type] += 1

        self.unit_registry[tag] = (unit_type, id_num)

        return f"unit_{unit_type}_{id_num:03d}"

    def detect_state(self, tag: int, current_tags: Set[int]) -> str:
        """
        Determine if unit is built/existing/killed.

        Args:
            tag: Unit tag
            current_tags: Set of all tags in current frame

        Returns:
            State string: 'built', 'existing', or 'killed'
        """
        if tag not in self.previous_frame_tags:
            return 'built'
        return 'existing'

    def reset(self):
        """Reset the tracker."""
        self.unit_registry.clear()
        self.unit_counters.clear()
        self.previous_frame_tags.clear()


class BuildingTracker:
    """
    Tracks buildings and their lifecycle.

    This class tracks building construction, completion, and destruction
    across the replay.
    """

    def __init__(self):
        """Initialize the BuildingTracker."""
        self.building_registry: Dict[int, Dict] = {}  # tag -> building_info
        self.previous_frame_tags: Set[int] = set()

        logger.debug("BuildingTracker initialized")

    def process_buildings(self, raw_buildings, game_loop: int) -> Dict[str, Dict]:
        """
        Process raw buildings and track lifecycle.

        Args:
            raw_buildings: Raw building data from observation
            game_loop: Current game loop

        Returns:
            Dictionary mapping building IDs to building data:
            {
                'barracks_001': {
                    'x': float, 'y': float, 'z': float,
                    'status': 'started'|'building'|'completed'|'destroyed',
                    'progress': int,  # 0-100
                    'started_loop': int,
                    'completed_loop': int or None,
                    'destroyed_loop': int or None,
                },
                ...
            }

        # TODO: Test case - Track building construction progress
        # TODO: Test case - Detect building completion
        # TODO: Test case - Detect building destruction
        """
        tracked_buildings = {}
        current_tags = set()

        # Process each building
        for building in raw_buildings:
            tag = building.tag
            current_tags.add(tag)

            # Initialize registry entry if new
            if tag not in self.building_registry:
                self.building_registry[tag] = {
                    'started_loop': game_loop,
                    'completed_loop': None,
                    'destroyed_loop': None,
                }

            building_info = self.building_registry[tag]

            # Determine status
            if building.build_progress >= 1.0:
                status = 'completed'
                if building_info['completed_loop'] is None:
                    building_info['completed_loop'] = game_loop
            elif building.build_progress > 0:
                status = 'building'
            else:
                status = 'started'

            # Build tracked building data
            building_id = f"building_{tag}"
            tracked_buildings[building_id] = {
                'tag': tag,
                'building_type': building.unit_type,
                'x': building.pos.x,
                'y': building.pos.y,
                'z': building.pos.z,
                'status': status,
                'progress': int(building.build_progress * 100),
                'started_loop': building_info['started_loop'],
                'completed_loop': building_info['completed_loop'],
                'destroyed_loop': building_info['destroyed_loop'],
                'game_loop': game_loop,
            }

        # Detect destroyed buildings
        dead_tags = self.previous_frame_tags - current_tags
        for dead_tag in dead_tags:
            if dead_tag in self.building_registry:
                building_info = self.building_registry[dead_tag]
                if building_info['destroyed_loop'] is None:
                    building_info['destroyed_loop'] = game_loop

                building_id = f"building_{dead_tag}"
                tracked_buildings[building_id] = {
                    'tag': dead_tag,
                    'status': 'destroyed',
                    'started_loop': building_info['started_loop'],
                    'completed_loop': building_info['completed_loop'],
                    'destroyed_loop': building_info['destroyed_loop'],
                    'game_loop': game_loop,
                }

        # Update for next frame
        self.previous_frame_tags = current_tags

        return tracked_buildings

    def reset(self):
        """Reset the tracker."""
        self.building_registry.clear()
        self.previous_frame_tags.clear()

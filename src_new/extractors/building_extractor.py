"""
BuildingExtractor: Extracts building data from SC2 observations.

This component handles:
- Extracting building information from raw observation data
- Tracking building tags (persistent IDs) across frames
- Assigning human-readable IDs to buildings
- Detecting building state changes (started, building, completed, destroyed)
- Managing building lifecycle tracking including construction timestamps
"""

from typing import Dict, Set, Optional
import logging

from pysc2.lib import units as pysc2_units


logger = logging.getLogger(__name__)


# Define which alliance values represent different player perspectives
ALLIANCE_SELF = 1
ALLIANCE_ALLY = 2
ALLIANCE_NEUTRAL = 3
ALLIANCE_ENEMY = 4


# Building unit type IDs
# This is a comprehensive list of all building types in SC2
BUILDING_TYPES = {
    # Terran buildings
    18,   # CommandCenter
    19,   # SupplyDepot
    20,   # Refinery
    21,   # Barracks
    22,   # OrbitalCommand
    27,   # EngineeringBay
    28,   # MissileTurret
    29,   # Bunker
    36,   # SupplyDepotLowered
    41,   # SensorTower
    43,   # GhostAcademy
    45,   # Factory
    46,   # Starport
    47,   # Armory
    48,   # FusionCore
    130,  # PlanetaryFortress
    132,  # Reactor
    133,  # TechLab
    134,  # BarracksReactor
    135,  # BarracksTechLab
    138,  # FactoryReactor
    139,  # FactoryTechLab
    142,  # StarportReactor
    143,  # StarportTechLab

    # Protoss buildings
    59,   # Nexus
    60,   # Pylon
    61,   # Assimilator
    62,   # Gateway
    63,   # Forge
    64,   # FleetBeacon
    66,   # PhotonCannon
    67,   # CyberneticsCore
    68,   # ShieldBattery
    69,   # RoboticsFacility
    70,   # Stargate
    71,   # TwilightCouncil
    72,   # RoboticsBay
    74,   # TemplarArchive
    75,   # DarkShrine
    133,  # WarpGate

    # Zerg buildings
    86,   # Hatchery
    88,   # Lair
    89,   # Hive
    90,   # SpawningPool
    91,   # EvolutionChamber
    92,   # HydraliskDen
    93,   # Spire
    94,   # UltraliskCavern
    95,   # InfestationPit
    96,   # NydusNetwork
    97,   # BanelingNest
    98,   # RoachWarren
    99,   # SpineCrawler
    100,  # SporeCrawler
    101,  # GreaterSpire
    104,  # Extractor
    142,  # LurkerDenMP

    # Creep tumors
    87,   # CreepTumor
    137,  # CreepTumorBurrowed
    138,  # CreepTumorQueen
}


def is_building(unit_type_id: int) -> bool:
    """
    Check if a unit type ID represents a building.

    Args:
        unit_type_id: SC2 unit type ID

    Returns:
        True if the unit type is a building, False otherwise
    """
    return unit_type_id in BUILDING_TYPES


def get_building_type_name(unit_type_id: int) -> str:
    """
    Convert building type ID to human-readable name.

    Args:
        unit_type_id: SC2 building type ID

    Returns:
        Building type name string
    """
    try:
        return pysc2_units.get_unit_type(unit_type_id).name
    except (KeyError, AttributeError):
        return f"Unknown({unit_type_id})"


class BuildingExtractor:
    """
    Extracts building data from SC2 observations.

    This class tracks buildings across frames, assigns readable IDs, and extracts
    comprehensive building state information including construction progress for
    ground truth data.

    Example usage:
        extractor = BuildingExtractor(player_id=1)

        for obs in game_loop_iterator:
            buildings_data = extractor.extract(obs)

            # Print building counts
            counts = extractor.get_building_counts(buildings_data)
            print(f"Buildings: {counts}")

            # Access specific building
            if 'p1_barracks_001' in buildings_data:
                barracks = buildings_data['p1_barracks_001']
                print(f"Barracks at ({barracks['x']}, {barracks['y']})")
                print(f"Construction: {barracks['build_progress']*100:.1f}%")
    """

    def __init__(self, player_id: int):
        """
        Initialize the BuildingExtractor.

        Args:
            player_id: Player ID this extractor is tracking (1 or 2)
        """
        self.player_id = player_id

        # Tag tracking: Maps SC2 tags (uint64) to readable IDs (e.g., "p1_barracks_001")
        self.tag_to_readable_id: Dict[int, str] = {}

        # Counter for generating sequential IDs per building type
        self.building_type_counters: Dict[int, int] = {}

        # Track previous frame's tags for state detection
        self.previous_tags: Set[int] = set()

        # Track buildings that have been seen (for "destroyed" detection)
        self.all_seen_tags: Set[int] = set()

        # Track construction completion timestamps
        self.completion_timestamps: Dict[int, int] = {}  # tag -> game_loop when completed

        # Track destruction timestamps
        self.destruction_timestamps: Dict[int, int] = {}  # tag -> game_loop when destroyed

        # Track previous build progress for completion detection
        self.previous_build_progress: Dict[int, float] = {}  # tag -> build_progress

    def extract(self, obs) -> Dict[str, Dict]:
        """
        Extract all building data from observation.

        Args:
            obs: SC2 observation from controller.observe()

        Returns:
            Dictionary mapping readable IDs to building data:
            {
                'p1_barracks_001': {
                    'tag': 12345,
                    'unit_type_id': 21,
                    'unit_type_name': 'Barracks',
                    'x': 100.5,
                    'y': 80.3,
                    'z': 8.0,
                    'health': 1000.0,
                    'health_max': 1000.0,
                    'shields': 0.0,
                    'shields_max': 0.0,
                    'build_progress': 0.75,
                    'state': 'building',  # 'started', 'building', 'completed', 'destroyed'
                    'started_loop': 500,
                    'completed_loop': None,  # or game_loop when completed
                    'destroyed_loop': None,  # or game_loop when destroyed
                },
                ...
            }
        """
        raw_data = obs.observation.raw_data
        game_loop = obs.observation.game_loop
        buildings_data = {}

        # Get current frame's building tags
        current_tags = set()

        # Process all units, filtering for buildings
        for unit in raw_data.units:
            # Filter: Only process units owned by this player
            if unit.owner != self.player_id:
                continue

            # Filter: Only process buildings
            if not is_building(unit.unit_type):
                continue

            # Track this tag
            tag = unit.tag
            current_tags.add(tag)
            self.all_seen_tags.add(tag)

            # Assign readable ID if new building
            if tag not in self.tag_to_readable_id:
                readable_id = self._assign_readable_id(unit.unit_type, tag)
                self.tag_to_readable_id[tag] = readable_id

            readable_id = self.tag_to_readable_id[tag]

            # Determine building state
            state = self._determine_state(tag, unit, game_loop)

            # Check for construction completion
            if tag not in self.completion_timestamps and unit.build_progress >= 1.0:
                prev_progress = self.previous_build_progress.get(tag, 0.0)
                if prev_progress < 1.0:
                    self.completion_timestamps[tag] = game_loop

            # Update build progress tracking
            self.previous_build_progress[tag] = unit.build_progress

            # Determine timestamps
            started_loop = None
            if tag in self.tag_to_readable_id:
                # Building was first seen in some previous frame
                # We don't track exact start time, but this could be enhanced
                started_loop = None  # Could track this if needed

            completed_loop = self.completion_timestamps.get(tag, None)
            destroyed_loop = self.destruction_timestamps.get(tag, None)

            # Extract building data
            building_data = {
                'tag': tag,
                'unit_type_id': unit.unit_type,
                'unit_type_name': get_building_type_name(unit.unit_type),

                # Position
                'x': unit.pos.x,
                'y': unit.pos.y,
                'z': unit.pos.z,
                'facing': unit.facing,

                # Vitals
                'health': unit.health,
                'health_max': unit.health_max,
                'shields': unit.shield,
                'shields_max': unit.shield_max,
                'energy': unit.energy,
                'energy_max': unit.energy_max,

                # Construction state
                'build_progress': unit.build_progress,
                'state': state,

                # Timestamps
                'started_loop': started_loop,
                'completed_loop': completed_loop,
                'destroyed_loop': destroyed_loop,

                # Additional state info
                'is_flying': unit.is_flying,  # For lifted Terran buildings
                'is_burrowed': unit.is_burrowed,  # For burrowed structures

                # Combat/Defense
                'attack_upgrade_level': unit.attack_upgrade_level,
                'armor_upgrade_level': unit.armor_upgrade_level,
                'shield_upgrade_level': unit.shield_upgrade_level,

                # Additional
                'radius': unit.radius,
                'order_count': len(unit.orders),
            }

            buildings_data[readable_id] = building_data

        # Detect destroyed buildings (in previous frame but not current)
        destroyed_tags = self.previous_tags - current_tags
        for destroyed_tag in destroyed_tags:
            if destroyed_tag in self.tag_to_readable_id:
                readable_id = self.tag_to_readable_id[destroyed_tag]

                # Record destruction timestamp if not already recorded
                if destroyed_tag not in self.destruction_timestamps:
                    self.destruction_timestamps[destroyed_tag] = game_loop

                # Add destroyed building with "destroyed" state
                buildings_data[readable_id] = {
                    'tag': destroyed_tag,
                    'state': 'destroyed',
                    'destroyed_loop': game_loop,
                }

        # Also check explicit dead units from event
        for dead_tag in raw_data.event.dead_units:
            if dead_tag in self.tag_to_readable_id and dead_tag in self.all_seen_tags:
                readable_id = self.tag_to_readable_id[dead_tag]

                # Record destruction timestamp if not already recorded
                if dead_tag not in self.destruction_timestamps:
                    self.destruction_timestamps[dead_tag] = game_loop

                if readable_id not in buildings_data:
                    buildings_data[readable_id] = {
                        'tag': dead_tag,
                        'state': 'destroyed',
                        'destroyed_loop': game_loop,
                    }

        # Update previous tags for next iteration
        self.previous_tags = current_tags

        return buildings_data

    def _assign_readable_id(self, building_type_id: int, tag: int) -> str:
        """
        Assign a human-readable ID to a building.

        Args:
            building_type_id: SC2 building type ID
            tag: SC2 building tag (persistent ID)

        Returns:
            Readable ID string like "p1_barracks_001"
        """
        # Get building type name
        building_type_name = get_building_type_name(building_type_id).lower()

        # Get next counter for this building type
        if building_type_id not in self.building_type_counters:
            self.building_type_counters[building_type_id] = 1

        counter = self.building_type_counters[building_type_id]
        self.building_type_counters[building_type_id] += 1

        # Create readable ID
        readable_id = f"p{self.player_id}_{building_type_name}_{counter:03d}"

        return readable_id

    def _determine_state(self, tag: int, unit, game_loop: int) -> str:
        """
        Determine the current state of a building.

        Args:
            tag: Building tag
            unit: Unit proto
            game_loop: Current game loop

        Returns:
            State string: 'started', 'building', 'completed', or 'destroyed'
        """
        # If build_progress is 0, it was just started
        if unit.build_progress == 0.0:
            return 'started'

        # If build_progress is between 0 and 1, it's being built
        if 0.0 < unit.build_progress < 1.0:
            return 'building'

        # If build_progress is 1.0, it's completed
        if unit.build_progress >= 1.0:
            return 'completed'

        # Default to building state
        return 'building'

    def get_building_counts(self, buildings_data: Dict[str, Dict]) -> Dict[str, int]:
        """
        Get count of buildings by type (excluding destroyed buildings).

        Args:
            buildings_data: Output from extract()

        Returns:
            Dictionary mapping building type names to counts:
            {'Barracks': 2, 'CommandCenter': 1, 'SupplyDepot': 5, ...}
        """
        counts = {}

        for building_data in buildings_data.values():
            # Skip destroyed buildings
            if building_data.get('state') == 'destroyed':
                continue

            building_type_name = building_data.get('unit_type_name')
            if building_type_name:
                counts[building_type_name] = counts.get(building_type_name, 0) + 1

        return counts

    def get_building_by_state(self, buildings_data: Dict[str, Dict]) -> Dict[str, list]:
        """
        Get buildings grouped by their state.

        Args:
            buildings_data: Output from extract()

        Returns:
            Dictionary mapping states to lists of building readable IDs:
            {
                'started': ['p1_barracks_003'],
                'building': ['p1_barracks_004', 'p1_supplydepot_010'],
                'completed': ['p1_commandcenter_001', ...],
                'destroyed': ['p1_barracks_001']
            }
        """
        by_state = {
            'started': [],
            'building': [],
            'completed': [],
            'destroyed': []
        }

        for readable_id, building_data in buildings_data.items():
            state = building_data.get('state', 'completed')
            if state in by_state:
                by_state[state].append(readable_id)

        return by_state

    def reset(self):
        """Reset all tracking state."""
        self.tag_to_readable_id.clear()
        self.building_type_counters.clear()
        self.previous_tags.clear()
        self.all_seen_tags.clear()
        self.completion_timestamps.clear()
        self.destruction_timestamps.clear()
        self.previous_build_progress.clear()

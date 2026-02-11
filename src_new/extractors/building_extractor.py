"""
BuildingExtractor: Extracts building data from SC2 observations.

This component handles:
- Extracting building information from raw observation data
- Tracking building tags (persistent IDs) across frames
- Assigning human-readable IDs to buildings
- Detecting building lifecycle transitions (building_started, under construction,
  completed, destroyed, cancelled)
- Managing building lifecycle tracking with embedded lifecycle state in attribute columns
- Conditional attribute extraction (shields only for Protoss, energy only for casters)

Building lifecycle differs from units:
- During construction, buildings capture REAL data (x, y, health) because position
  and health during construction is strategically meaningful.
- Buildings that start but never complete still appear in the dataset (unlike units).
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

    Lifecycle states are embedded into attribute columns:
    - "building_started": On the gameloop where construction starts, ALL attribute columns
    - Real data during construction: Position, health etc. are captured (strategically meaningful)
    - "completed": On the completion gameloop, ALL attribute columns
    - Real data after completion: Normal attribute data
    - "destroyed": On destruction gameloop, ALL attribute columns, then NaN permanently
    - "cancelled": On cancellation gameloop, ALL attribute columns, then NaN permanently

    Unlike units, buildings that start but never complete STILL appear in the dataset.
    Only applicable attributes are included (shields for Protoss, energy for casters).
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

        # Track which tags have ever reached build_progress >= 1.0 (completed)
        self.completed_tags: Set[int] = set()

        # Track previous build progress for completion detection
        self.previous_build_progress: Dict[int, float] = {}

        # Track which tags are confirmed dead/destroyed
        self.dead_tags: Set[int] = set()

        # Track which buildings were under construction when they disappeared
        # (for cancelled vs destroyed heuristic)
        self.was_under_construction: Dict[int, bool] = {}

        # Track which readable_ids have shields and/or energy (discovered during pass 1)
        self.building_attributes: Dict[str, Set[str]] = {}

    def extract(self, obs) -> Dict[str, Dict]:
        """
        Extract all building data from observation.

        Lifecycle is embedded in the data dict via the '_lifecycle' key:
        - 'building_started': First frame a building appears with build_progress near 0
        - 'under_construction': Building is being built (real data is still captured)
        - 'completed': The frame where build_progress transitions to >= 1.0
        - 'existing': Already completed in a previous frame
        - 'destroyed': Building was destroyed (was in dead_units or disappeared after completion)
        - 'cancelled': Building was cancelled (disappeared while under construction,
                       not in dead_units)

        Args:
            obs: SC2 observation from controller.observe()

        Returns:
            Dictionary mapping readable IDs to building data dicts.
        """
        raw_data = obs.observation.raw_data
        game_loop = obs.observation.game_loop
        buildings_data = {}

        # Collect dead unit tags for this frame for cancelled vs destroyed detection
        dead_unit_tags = set(raw_data.event.dead_units)

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

            # Determine lifecycle state
            lifecycle = self._determine_lifecycle(tag, unit)

            # Track completion
            if unit.build_progress >= 1.0:
                self.completed_tags.add(tag)

            # Track construction state for cancelled detection
            self.was_under_construction[tag] = unit.build_progress < 1.0

            # Update build progress tracking
            self.previous_build_progress[tag] = unit.build_progress

            # Extract building data
            building_data = {
                'tag': tag,
                'unit_type_id': unit.unit_type,
                'unit_type_name': get_building_type_name(unit.unit_type),
                '_lifecycle': lifecycle,

                # Position
                'x': unit.pos.x,
                'y': unit.pos.y,
                'z': unit.pos.z,
                'facing': unit.facing,

                # Vitals
                'health': unit.health,
                'health_max': unit.health_max,

                # Construction
                'build_progress': unit.build_progress,

                # Additional state info
                'is_flying': unit.is_flying,
                'is_burrowed': unit.is_burrowed,

                # Combat/Defense
                'attack_upgrade_level': unit.attack_upgrade_level,
                'armor_upgrade_level': unit.armor_upgrade_level,

                # Additional
                'radius': unit.radius,
                'order_count': len(unit.orders),
            }

            # Conditionally add shields (Protoss only - shield_max > 0)
            if unit.shield_max > 0:
                building_data['shields'] = unit.shield
                building_data['shields_max'] = unit.shield_max
                building_data['shield_upgrade_level'] = unit.shield_upgrade_level

            # Conditionally add energy (casters only - energy_max > 0)
            if unit.energy_max > 0:
                building_data['energy'] = unit.energy
                building_data['energy_max'] = unit.energy_max

            # Record which attributes this building has (for schema building)
            if readable_id not in self.building_attributes:
                attr_set = set()
                if unit.shield_max > 0:
                    attr_set.update(['shields', 'shields_max', 'shield_upgrade_level'])
                if unit.energy_max > 0:
                    attr_set.update(['energy', 'energy_max'])
                self.building_attributes[readable_id] = attr_set

            buildings_data[readable_id] = building_data

        # Detect destroyed/cancelled buildings (in previous frame but not current)
        disappeared_tags = self.previous_tags - current_tags
        for disappeared_tag in disappeared_tags:
            if disappeared_tag in self.tag_to_readable_id:
                readable_id = self.tag_to_readable_id[disappeared_tag]

                # Determine if destroyed or cancelled:
                # - If in dead_units event: destroyed
                # - If was under construction and NOT in dead_units: cancelled (heuristic)
                # - If was completed and NOT in dead_units: destroyed (disappeared)
                was_building = self.was_under_construction.get(disappeared_tag, False)
                in_dead_units = disappeared_tag in dead_unit_tags

                if was_building and not in_dead_units:
                    lifecycle = 'cancelled'
                else:
                    lifecycle = 'destroyed'

                self.dead_tags.add(disappeared_tag)
                buildings_data[readable_id] = {
                    'tag': disappeared_tag,
                    '_lifecycle': lifecycle,
                }

        # Also check explicit dead units from event for buildings we track
        for dead_tag in dead_unit_tags:
            if dead_tag in self.tag_to_readable_id and dead_tag in self.all_seen_tags:
                readable_id = self.tag_to_readable_id[dead_tag]
                self.dead_tags.add(dead_tag)
                if readable_id not in buildings_data:
                    buildings_data[readable_id] = {
                        'tag': dead_tag,
                        '_lifecycle': 'destroyed',
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

    def _determine_lifecycle(self, tag: int, unit) -> str:
        """
        Determine the lifecycle state of a building for the current frame.

        Lifecycle states:
        - 'building_started': First appearance with build_progress near 0
        - 'under_construction': Being built (build_progress < 1.0, not first frame)
        - 'completed': The frame where build_progress transitions to >= 1.0
        - 'existing': Already completed in a previous frame

        Args:
            tag: Building tag
            unit: Unit proto

        Returns:
            Lifecycle state string
        """
        is_new = tag not in self.previous_tags

        # If building is already known to be completed
        if tag in self.completed_tags:
            return 'existing'

        # Building just appeared
        if is_new:
            if unit.build_progress >= 1.0:
                # Building appeared already complete (e.g., game start buildings)
                return 'completed'
            else:
                return 'building_started'

        # Building existed in previous frame
        prev_progress = self.previous_build_progress.get(tag, 0.0)

        if unit.build_progress >= 1.0 and prev_progress < 1.0:
            # Just completed this frame
            return 'completed'
        elif unit.build_progress < 1.0:
            return 'under_construction'
        else:
            return 'existing'

    def get_building_attributes_for_id(self, readable_id: str) -> Set[str]:
        """
        Get the set of conditional attributes for a given building readable_id.

        Returns:
            Set of attribute keys like {'shields', 'shields_max', 'energy', ...}
        """
        return self.building_attributes.get(readable_id, set())

    def get_building_counts(self, buildings_data: Dict[str, Dict]) -> Dict[str, int]:
        """
        Get count of buildings by type (excluding destroyed/cancelled buildings).

        Args:
            buildings_data: Output from extract()

        Returns:
            Dictionary mapping building type names to counts:
            {'Barracks': 2, 'CommandCenter': 1, 'SupplyDepot': 5, ...}
        """
        counts = {}

        for building_data in buildings_data.values():
            # Skip destroyed/cancelled buildings
            lifecycle = building_data.get('_lifecycle', 'existing')
            if lifecycle in ('destroyed', 'cancelled'):
                continue

            building_type_name = building_data.get('unit_type_name')
            if building_type_name:
                counts[building_type_name] = counts.get(building_type_name, 0) + 1

        return counts

    def get_building_by_state(self, buildings_data: Dict[str, Dict]) -> Dict[str, list]:
        """
        Get buildings grouped by their lifecycle state.

        Args:
            buildings_data: Output from extract()

        Returns:
            Dictionary mapping lifecycle states to lists of building readable IDs.
        """
        by_state = {
            'building_started': [],
            'under_construction': [],
            'completed': [],
            'existing': [],
            'destroyed': [],
            'cancelled': [],
        }

        for readable_id, building_data in buildings_data.items():
            lifecycle = building_data.get('_lifecycle', 'existing')
            if lifecycle in by_state:
                by_state[lifecycle].append(readable_id)

        return by_state

    def reset(self):
        """Reset all tracking state."""
        self.tag_to_readable_id.clear()
        self.building_type_counters.clear()
        self.previous_tags.clear()
        self.all_seen_tags.clear()
        self.completed_tags.clear()
        self.previous_build_progress.clear()
        self.dead_tags.clear()
        self.was_under_construction.clear()
        self.building_attributes.clear()

    def reset_frame_state(self):
        """Reset only per-frame state, preserving tag-to-ID mappings and counters.

        Used between two-pass processing so pass 2 reuses the same readable IDs
        that were assigned during pass 1 (schema scan).
        """
        self.previous_tags.clear()
        self.previous_build_progress.clear()
        self.dead_tags.clear()
        self.was_under_construction.clear()

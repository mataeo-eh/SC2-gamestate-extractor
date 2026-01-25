"""
UnitExtractor: Extracts unit data from SC2 observations.

This component handles:
- Extracting unit information from raw observation data
- Tracking unit tags (persistent IDs) across frames
- Assigning human-readable IDs to units
- Detecting unit state changes (built, existing, killed)
- Managing unit lifecycle tracking
"""

from typing import Dict, Set, Tuple, Optional
import logging

from pysc2.lib import units as pysc2_units


logger = logging.getLogger(__name__)


# Define which alliance values represent different player perspectives
ALLIANCE_SELF = 1
ALLIANCE_ALLY = 2
ALLIANCE_NEUTRAL = 3
ALLIANCE_ENEMY = 4


# Building unit type IDs (these should be excluded from unit counts)
# This is a comprehensive list of common building types
BUILDING_TYPES = {
    # Terran buildings
    18,   # CommandCenter
    19,   # SupplyDepot
    20,   # Refinery
    21,   # Barracks
    27,   # EngineeringBay
    28,   # MissileTurret
    29,   # Bunker
    41,   # SensorTower
    43,   # GhostAcademy
    45,   # Factory
    46,   # Starport
    47,   # Armory
    48,   # FusionCore

    # Protoss buildings
    59,   # Nexus
    60,   # Pylon
    61,   # Assimilator
    62,   # Gateway
    63,   # Forge
    66,   # PhotonCannon
    67,   # CyberneticsCore
    68,   # ShieldBattery
    69,   # RoboticsFacility
    70,   # Stargate
    71,   # TwilightCouncil
    72,   # RoboticsBay
    73,   # FleetBeacon
    74,   # TemplarArchive
    75,   # DarkShrine

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
}


def is_building(unit_type_id: int) -> bool:
    """Check if a unit type ID represents a building."""
    return unit_type_id in BUILDING_TYPES


def get_unit_type_name(unit_type_id: int) -> str:
    """
    Convert unit type ID to human-readable name.

    Args:
        unit_type_id: SC2 unit type ID

    Returns:
        Unit type name string
    """
    try:
        return pysc2_units.get_unit_type(unit_type_id).name
    except (KeyError, AttributeError):
        return f"Unknown({unit_type_id})"


class UnitExtractor:
    """
    Extracts unit data from SC2 observations.

    This class tracks units across frames, assigns readable IDs, and extracts
    comprehensive unit state information for ground truth data.
    """

    def __init__(self, player_id: int):
        """
        Initialize the UnitExtractor.

        Args:
            player_id: Player ID this extractor is tracking (1 or 2)
        """
        self.player_id = player_id

        # Tag tracking: Maps SC2 tags (uint64) to readable IDs (e.g., "p1_marine_001")
        self.tag_to_readable_id: Dict[int, str] = {}

        # Counter for generating sequential IDs per unit type
        self.unit_type_counters: Dict[int, int] = {}

        # Track previous frame's tags for state detection
        self.previous_tags: Set[int] = set()

        # Track units that have been seen (for "killed" detection)
        self.all_seen_tags: Set[int] = set()

    def extract(self, obs) -> Dict[str, Dict]:
        """
        Extract all unit data from observation.

        Args:
            obs: SC2 observation from controller.observe()

        Returns:
            Dictionary mapping readable IDs to unit data:
            {
                'p1_marine_001': {
                    'tag': 12345,
                    'unit_type_id': 48,
                    'unit_type_name': 'Marine',
                    'x': 50.2,
                    'y': 30.1,
                    'z': 8.0,
                    'health': 45.0,
                    'health_max': 45.0,
                    'shields': 0.0,
                    'shields_max': 0.0,
                    'energy': 0.0,
                    'energy_max': 0.0,
                    'state': 'existing',  # 'built', 'existing', or 'killed'
                },
                ...
            }
        """
        raw_data = obs.observation.raw_data
        units_data = {}

        # Get current frame's unit tags
        current_tags = set()

        # Process all units
        for unit in raw_data.units:
            # Filter: Only process units owned by this player
            if unit.owner != self.player_id:
                continue

            # Filter: Skip buildings (handled by BuildingExtractor)
            if is_building(unit.unit_type):
                continue

            # Track this tag
            tag = unit.tag
            current_tags.add(tag)
            self.all_seen_tags.add(tag)

            # Assign readable ID if new unit
            if tag not in self.tag_to_readable_id:
                readable_id = self._assign_readable_id(unit.unit_type, tag)
                self.tag_to_readable_id[tag] = readable_id

            readable_id = self.tag_to_readable_id[tag]

            # Determine unit state
            state = self._determine_state(tag, unit)

            # Extract unit data
            unit_data = {
                'tag': tag,
                'unit_type_id': unit.unit_type,
                'unit_type_name': get_unit_type_name(unit.unit_type),

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

                # State
                'state': state,
                'build_progress': unit.build_progress,  # For morphing units
                'is_flying': unit.is_flying,
                'is_burrowed': unit.is_burrowed,
                'is_hallucination': unit.is_hallucination,

                # Combat
                'weapon_cooldown': unit.weapon_cooldown,
                'attack_upgrade_level': unit.attack_upgrade_level,
                'armor_upgrade_level': unit.armor_upgrade_level,
                'shield_upgrade_level': unit.shield_upgrade_level,

                # Additional
                'radius': unit.radius,
                'cargo_space_taken': unit.cargo_space_taken,
                'cargo_space_max': unit.cargo_space_max,
                'order_count': len(unit.orders),
            }

            units_data[readable_id] = unit_data

        # Detect dead units (in previous frame but not current, and not in dead_units event)
        dead_tags = self.previous_tags - current_tags
        for dead_tag in dead_tags:
            if dead_tag in self.tag_to_readable_id:
                readable_id = self.tag_to_readable_id[dead_tag]
                # Add dead unit with "killed" state
                units_data[readable_id] = {
                    'tag': dead_tag,
                    'state': 'killed',
                }

        # Also check explicit dead units from event
        for dead_tag in raw_data.event.dead_units:
            if dead_tag in self.tag_to_readable_id and dead_tag in self.all_seen_tags:
                readable_id = self.tag_to_readable_id[dead_tag]
                if readable_id not in units_data:
                    units_data[readable_id] = {
                        'tag': dead_tag,
                        'state': 'killed',
                    }

        # Update previous tags for next iteration
        self.previous_tags = current_tags

        return units_data

    def _assign_readable_id(self, unit_type_id: int, tag: int) -> str:
        """
        Assign a human-readable ID to a unit.

        Args:
            unit_type_id: SC2 unit type ID
            tag: SC2 unit tag (persistent ID)

        Returns:
            Readable ID string like "p1_marine_001"
        """
        # Get unit type name
        unit_type_name = get_unit_type_name(unit_type_id).lower()

        # Get next counter for this unit type
        if unit_type_id not in self.unit_type_counters:
            self.unit_type_counters[unit_type_id] = 1

        counter = self.unit_type_counters[unit_type_id]
        self.unit_type_counters[unit_type_id] += 1

        # Create readable ID
        readable_id = f"p{self.player_id}_{unit_type_name}_{counter:03d}"

        return readable_id

    def _determine_state(self, tag: int, unit) -> str:
        """
        Determine the current state of a unit.

        Args:
            tag: Unit tag
            unit: Unit proto

        Returns:
            State string: 'built', 'existing', or 'killed'
        """
        # If this is a new tag (not in previous frame), it was just built
        if tag not in self.previous_tags:
            return 'built'

        # If build_progress < 1.0, it's still being built/morphed
        if unit.build_progress < 1.0:
            return 'built'

        # Otherwise, it exists from a previous frame
        return 'existing'

    def get_unit_counts(self, units_data: Dict[str, Dict]) -> Dict[str, int]:
        """
        Get count of units by type (excluding killed units).

        Args:
            units_data: Output from extract()

        Returns:
            Dictionary mapping unit type names to counts:
            {'Marine': 10, 'Medivac': 2, 'SiegeTank': 3, ...}
        """
        counts = {}

        for unit_data in units_data.values():
            # Skip killed units
            if unit_data.get('state') == 'killed':
                continue

            unit_type_name = unit_data.get('unit_type_name')
            if unit_type_name:
                counts[unit_type_name] = counts.get(unit_type_name, 0) + 1

        return counts

    def reset(self):
        """Reset all tracking state."""
        self.tag_to_readable_id.clear()
        self.unit_type_counters.clear()
        self.previous_tags.clear()
        self.all_seen_tags.clear()

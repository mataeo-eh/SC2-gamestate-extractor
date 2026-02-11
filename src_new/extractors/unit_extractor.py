"""
UnitExtractor: Extracts unit data from SC2 observations.

This component handles:
- Extracting unit information from raw observation data
- Tracking unit tags (persistent IDs) across frames
- Assigning human-readable IDs to units
- Detecting unit lifecycle transitions (unit_started, building, completed, destroyed)
- Managing unit lifecycle tracking with embedded lifecycle state in attribute columns
- Conditional attribute extraction (shields only for Protoss, energy only for casters)
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

    Lifecycle states are embedded into attribute columns:
    - "unit_started": On the gameloop where production starts, ALL attribute columns
    - "building": While the unit is being produced, ALL attribute columns
    - "completed": On the gameloop where the unit completes, ALL attribute columns
    - Real data: After completion, columns capture real data (x, y, health, etc.)
    - "destroyed": On the gameloop where the unit is destroyed, ALL attribute columns
    - NaN: Before the unit exists and after it is destroyed

    Units that start but never complete are excluded from the dataset entirely.
    Only attributes applicable to a unit are included (shields for Protoss, energy for casters).
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

        # Track which tags have ever reached build_progress >= 1.0 (completed)
        self.completed_tags: Set[int] = set()

        # Track previous build_progress for detecting the completion transition frame
        self.previous_build_progress: Dict[int, float] = {}

        # Track which tags are confirmed dead (destroyed)
        self.dead_tags: Set[int] = set()

        # Track which readable_ids have shields and/or energy (discovered during pass 1)
        # Maps readable_id -> set of extra attribute keys present
        self.unit_attributes: Dict[str, Set[str]] = {}

    def extract(self, obs) -> Dict[str, Dict]:
        """
        Extract all unit data from observation.

        Lifecycle is embedded in the data dict via the '_lifecycle' key:
        - 'unit_started': First frame a unit appears with build_progress == 0.0
        - 'building': Unit is being produced (0.0 < build_progress < 1.0)
        - 'completed': The frame where build_progress transitions to >= 1.0
        - 'existing': Unit exists and is complete (normal data)
        - 'destroyed': Unit just died this frame

        Units that never complete will still appear in extract() output (needed
        for pass 1 tracking), but schema_manager will filter them out when
        building columns.

        Args:
            obs: SC2 observation from controller.observe()

        Returns:
            Dictionary mapping readable IDs to unit data dicts.
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

            # Determine lifecycle state
            lifecycle = self._determine_lifecycle(tag, unit)

            # Track completion
            if unit.build_progress >= 1.0:
                self.completed_tags.add(tag)

            # Update previous build progress
            self.previous_build_progress[tag] = unit.build_progress

            # Extract unit data
            unit_data = {
                'tag': tag,
                'unit_type_id': unit.unit_type,
                'unit_type_name': get_unit_type_name(unit.unit_type),
                '_lifecycle': lifecycle,

                # Position
                'x': unit.pos.x,
                'y': unit.pos.y,
                'z': unit.pos.z,
                'facing': unit.facing,

                # Vitals
                'health': unit.health,
                'health_max': unit.health_max,

                # Build progress (used internally, not written as separate column)
                'build_progress': unit.build_progress,
                'is_flying': unit.is_flying,
                'is_burrowed': unit.is_burrowed,
                'is_hallucination': unit.is_hallucination,

                # Combat
                'weapon_cooldown': unit.weapon_cooldown,
                'attack_upgrade_level': unit.attack_upgrade_level,
                'armor_upgrade_level': unit.armor_upgrade_level,
            }

            # Conditionally add shields (Protoss only - shield_max > 0)
            if unit.shield_max > 0:
                unit_data['shields'] = unit.shield
                unit_data['shields_max'] = unit.shield_max
                unit_data['shield_upgrade_level'] = unit.shield_upgrade_level

            # Conditionally add energy (casters only - energy_max > 0)
            if unit.energy_max > 0:
                unit_data['energy'] = unit.energy
                unit_data['energy_max'] = unit.energy_max

            # Additional attributes
            unit_data.update({
                'radius': unit.radius,
                'cargo_space_taken': unit.cargo_space_taken,
                'cargo_space_max': unit.cargo_space_max,
                'order_count': len(unit.orders),
            })

            # Record which attributes this unit has (for schema building)
            if readable_id not in self.unit_attributes:
                attr_set = set()
                if unit.shield_max > 0:
                    attr_set.update(['shields', 'shields_max', 'shield_upgrade_level'])
                if unit.energy_max > 0:
                    attr_set.update(['energy', 'energy_max'])
                self.unit_attributes[readable_id] = attr_set

            units_data[readable_id] = unit_data

        # Detect dead units (in previous frame but not current)
        disappeared_tags = self.previous_tags - current_tags
        for dead_tag in disappeared_tags:
            if dead_tag in self.tag_to_readable_id:
                readable_id = self.tag_to_readable_id[dead_tag]
                self.dead_tags.add(dead_tag)
                units_data[readable_id] = {
                    'tag': dead_tag,
                    '_lifecycle': 'destroyed',
                }

        # Also check explicit dead units from event
        for dead_tag in raw_data.event.dead_units:
            if dead_tag in self.tag_to_readable_id and dead_tag in self.all_seen_tags:
                readable_id = self.tag_to_readable_id[dead_tag]
                self.dead_tags.add(dead_tag)
                if readable_id not in units_data:
                    units_data[readable_id] = {
                        'tag': dead_tag,
                        '_lifecycle': 'destroyed',
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

    def _determine_lifecycle(self, tag: int, unit) -> str:
        """
        Determine the lifecycle state of a unit for the current frame.

        Lifecycle states:
        - 'unit_started': First appearance, build_progress == 0.0 (or very low)
        - 'building': Being produced (build_progress < 1.0)
        - 'completed': The frame where build_progress transitions to >= 1.0
        - 'existing': Already completed in a previous frame

        Args:
            tag: Unit tag
            unit: Unit proto

        Returns:
            Lifecycle state string
        """
        is_new = tag not in self.previous_tags

        # If unit is already known to be completed
        if tag in self.completed_tags:
            return 'existing'

        # Unit just appeared
        if is_new:
            if unit.build_progress >= 1.0:
                # Unit appeared already complete (e.g., game start units, or
                # we missed the build phase). Mark as completed on this frame.
                return 'completed'
            elif unit.build_progress == 0.0:
                return 'unit_started'
            else:
                # Appeared mid-build (possible if we start observing mid-game)
                return 'building'

        # Unit existed in previous frame
        prev_progress = self.previous_build_progress.get(tag, 0.0)

        if unit.build_progress >= 1.0 and prev_progress < 1.0:
            # Just completed this frame
            return 'completed'
        elif unit.build_progress < 1.0:
            return 'building'
        else:
            return 'existing'

    def has_completed(self, tag: int) -> bool:
        """Check if a unit tag has ever reached completion."""
        return tag in self.completed_tags

    def get_completed_readable_ids(self) -> Set[str]:
        """
        Get the set of readable IDs for units that completed during the replay.

        Used by schema_manager to determine which units should have columns.
        Units that started but never completed are excluded.

        Returns:
            Set of readable ID strings for completed units.
        """
        completed_ids = set()
        for tag in self.completed_tags:
            if tag in self.tag_to_readable_id:
                completed_ids.add(self.tag_to_readable_id[tag])
        return completed_ids

    def get_unit_attributes_for_id(self, readable_id: str) -> Set[str]:
        """
        Get the set of conditional attributes for a given unit readable_id.

        Returns:
            Set of attribute keys like {'shields', 'shields_max', 'energy', ...}
        """
        return self.unit_attributes.get(readable_id, set())

    def get_unit_counts(self, units_data: Dict[str, Dict]) -> Dict[str, int]:
        """
        Get count of units by type (excluding destroyed units).

        Args:
            units_data: Output from extract()

        Returns:
            Dictionary mapping unit type names to counts:
            {'Marine': 10, 'Medivac': 2, 'SiegeTank': 3, ...}
        """
        counts = {}

        for unit_data in units_data.values():
            # Skip destroyed units
            if unit_data.get('_lifecycle') == 'destroyed':
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
        self.completed_tags.clear()
        self.previous_build_progress.clear()
        self.dead_tags.clear()
        self.unit_attributes.clear()

    def reset_frame_state(self):
        """Reset only per-frame state, preserving tag-to-ID mappings and counters.

        Used between two-pass processing so pass 2 reuses the same readable IDs
        that were assigned during pass 1 (schema scan).
        """
        self.previous_tags.clear()
        self.previous_build_progress.clear()
        self.dead_tags.clear()

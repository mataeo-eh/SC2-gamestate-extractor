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
- During construction, buildings capture REAL data (position, health) because position
  and health during construction is strategically meaningful.
- Buildings that start but never complete still appear in the dataset (unlike units).

Field extraction uses a declarative BUILDING_FIELD_CONFIG list, mirroring the approach
in unit_extractor.py. Each field specifies a column suffix, extraction lambda, and
whether it is always present or conditional (e.g., shields only for Protoss buildings).
"""

from typing import Dict, Set, Optional
import logging

from pysc2.lib import units as pysc2_units

from src_new.shared_constants import BUILDING_TYPES


logger = logging.getLogger(__name__)


def is_building(unit_type_id: int) -> bool:
    """
    Check if a unit type ID represents a building.

    Converts the integer unit type ID to a lowercase string name via pysc2
    and checks membership in the shared BUILDING_TYPES frozenset (which
    stores lowercase string names, not integer IDs).

    Args:
        unit_type_id: SC2 unit type ID (integer from the protobuf)

    Returns:
        True if the unit type is a building, False otherwise

    Depends on / calls:
        - get_building_type_name() to resolve the integer ID to a string name
        - BUILDING_TYPES from shared_constants (frozenset of lowercase strings)
    """
    name = get_building_type_name(unit_type_id).lower()
    return name in BUILDING_TYPES


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


# ---------------------------------------------------------------------------
# Declarative field extraction config for buildings
# ---------------------------------------------------------------------------
# Each entry describes one output column. The extract() method iterates this
# list instead of hardcoding field accesses, making it easy to add or remove
# fields without touching extraction logic.
#
# Keys per entry:
#   column_suffix  – suffix appended to the readable ID to form the column name
#   extract        – callable(unit) -> value to store
#   always         – True if the field is present for every building; False if
#                    gated by `condition`
#   condition      – callable(unit) -> bool; evaluated only when always=False
#   description    – human-readable explanation (for maintainers, not runtime)
# ---------------------------------------------------------------------------
BUILDING_FIELD_CONFIG = [
    {
        'column_suffix': 'pos_(X,Y,Z)',
        'extract': lambda unit: f"({unit.pos.x}, {unit.pos.y}, {unit.pos.z})",
        'always': True,
        'description': 'Position as (X, Y, Z) coordinate tuple',
    },
    {
        'column_suffix': 'health',
        'extract': lambda unit: f"{unit.health}/{unit.health_max}",
        'always': True,
        'description': 'Health as current/max fraction string',
    },
    {
        'column_suffix': 'shields',
        'extract': lambda unit: f"{unit.shield}/{unit.shield_max}",
        'condition': lambda unit: unit.shield_max > 0,
        'always': False,
        'description': 'Shields as current/max fraction string (Protoss buildings)',
    },
    {
        'column_suffix': 'energy',
        'extract': lambda unit: f"{unit.energy}/{unit.energy_max}",
        'condition': lambda unit: unit.energy_max > 0,
        'always': False,
        'description': 'Energy as current/max fraction string (e.g., Nexus)',
    },
    {
        'column_suffix': 'facing',
        'extract': lambda unit: unit.facing,
        'always': True,
        'description': 'Facing direction in radians',
    },
    {
        'column_suffix': 'build_progress',
        'extract': lambda unit: unit.build_progress,
        'always': True,
        'description': 'Construction progress (0.0 to 1.0)',
    },
    {
        'column_suffix': 'is_flying',
        'extract': lambda unit: unit.is_flying,
        'always': True,
        'description': 'Whether building is flying (lifted Terran)',
    },
    {
        'column_suffix': 'is_burrowed',
        'extract': lambda unit: unit.is_burrowed,
        'always': True,
        'description': 'Whether building is burrowed',
    },
    {
        'column_suffix': 'attack_upgrade_level',
        'extract': lambda unit: unit.attack_upgrade_level,
        'always': True,
        'description': 'Attack upgrade level',
    },
    {
        'column_suffix': 'armor_upgrade_level',
        'extract': lambda unit: unit.armor_upgrade_level,
        'always': True,
        'description': 'Armor upgrade level',
    },
    {
        'column_suffix': 'shield_upgrade_level',
        'extract': lambda unit: unit.shield_upgrade_level,
        'condition': lambda unit: unit.shield_max > 0,
        'always': False,
        'description': 'Shield upgrade level (Protoss buildings)',
    },
    {
        'column_suffix': 'radius',
        'extract': lambda unit: unit.radius,
        'always': True,
        'description': 'Building radius',
    },
    {
        'column_suffix': 'order_count',
        'extract': lambda unit: len(unit.orders),
        'always': True,
        'description': 'Number of queued orders',
    },
]


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
            # Filter: Skip non-player units (mineral patches, vespene geysers,
            # destructible rocks, Xel'Naga towers, critters, etc.).
            # We filter by unit.owner instead of unit.alliance because the
            # SC2 engine misassigns alliance values in observer mode
            # (observed_player_id=0): player units get ALLIANCE_NEUTRAL and
            # neutral map entities get ALLIANCE_SELF. unit.owner is always
            # correct regardless of perspective mode.
            # Players are always owner 1 and 2; everything else (owner 16
            # for neutrals, etc.) is a map entity.
            if unit.owner not in {1, 2}:
                continue

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

            # Extract building data using field config
            # Identity and lifecycle fields are always present; the rest come
            # from BUILDING_FIELD_CONFIG so new fields can be added declaratively.
            building_data = {
                'tag': tag,
                'unit_type_id': unit.unit_type,
                'unit_type_name': get_building_type_name(unit.unit_type),
                '_lifecycle': lifecycle,
            }

            # Iterate the field config to populate remaining columns
            for field_config in BUILDING_FIELD_CONFIG:
                suffix = field_config['column_suffix']

                # For conditional fields, skip if the condition is not met
                if not field_config.get('always', True):
                    condition = field_config.get('condition')
                    if condition and not condition(unit):
                        continue

                # Extract the value using the config's lambda
                building_data[suffix] = field_config['extract'](unit)

            # Record which conditional attributes this building has (for schema
            # building). Only needs to happen once per readable_id since a
            # building's conditional capabilities don't change mid-game.
            if readable_id not in self.building_attributes:
                attr_set = set()
                for field_config in BUILDING_FIELD_CONFIG:
                    if not field_config.get('always', True):
                        condition = field_config.get('condition')
                        if condition and condition(unit):
                            attr_set.add(field_config['column_suffix'])
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

        These are the column_suffix values from BUILDING_FIELD_CONFIG entries
        where always=False and the condition was True for this building.

        Args:
            readable_id: Human-readable building ID (e.g., "p1_nexus_001")

        Returns:
            Set of column suffix strings like {'shields', 'energy', 'shield_upgrade_level'}

        Called by:
            schema_manager to determine which conditional columns to create
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


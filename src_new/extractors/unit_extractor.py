"""
UnitExtractor: Extracts unit data from SC2 observations.

This component handles:
- Extracting unit information from raw observation data
- Tracking unit tags (persistent IDs) across frames
- Assigning human-readable IDs to units
- Detecting unit lifecycle transitions (unit_started, building, completed, destroyed)
- Managing unit lifecycle tracking with embedded lifecycle state in attribute columns
- Conditional attribute extraction (shields only for Protoss, energy only for casters)

Field extraction is driven by UNIT_FIELD_CONFIG, a module-level list of dictionaries.
Each entry defines a column suffix, an extract function, an optional condition, and
whether the field is always present or conditional. Users can add, remove, or reorder
entries in UNIT_FIELD_CONFIG to control which protobuf fields appear in the output
without touching any extraction logic.
"""

from typing import Dict, Set, Tuple, Optional, List, Any, Callable, FrozenSet
import json
import logging

from pysc2.lib import units as pysc2_units

from src_new.shared_constants import BUILDING_TYPES, UNTRACKED_ENTITY_TYPES


logger = logging.getLogger(__name__)

# Module-level API-derived building type ID set. When populated by
# set_building_type_ids_global(), is_building() uses O(1) integer lookup
# instead of string name conversion + frozenset membership test.
# None means the fallback string-based path is used.
_BUILDING_TYPE_IDS: Optional[FrozenSet[int]] = None


def _parse_position(pos_string: str) -> tuple:
    """
    Parse a formatted position string back to a numeric tuple.

    Args:
        pos_string: Position string in the format '(x, y, z)' as produced by
                    the UNIT_FIELD_CONFIG / BUILDING_FIELD_CONFIG extract lambdas.

    Returns:
        Tuple of (x, y, z) as floats, e.g. (42.5, 63.0, 11.98).

    Raises:
        ValueError: If the string cannot be parsed as three float values.
    """
    parts = pos_string.strip('()').split(',')
    return tuple(float(p.strip()) for p in parts)


def is_building(unit_type_id: int) -> bool:
    """
    Check if a unit type ID represents a building.

    Uses a two-tier strategy:
    1. If the API-derived _BUILDING_TYPE_IDS set has been populated (via
       set_building_type_ids_global()), performs O(1) integer lookup directly
       against the set built from data_raw() Attribute.Structure entries.
       This is the preferred path -- it is faster and uses the SC2 engine's
       own type metadata rather than a manually curated name list.
    2. Falls back to converting the integer ID to a lowercase string name
       via pysc2 and checking membership in BUILDING_TYPES (frozenset of
       lowercase strings). This handles edge cases like name mismatches
       (e.g., LurkerDen vs LurkerDenMP) and is used when data_raw() is
       not available.

    Args:
        unit_type_id: SC2 unit type ID (integer from the protobuf)

    Returns:
        True if the unit type is a building, False otherwise

    Depends on / calls:
        - _BUILDING_TYPE_IDS (module-level, set by set_building_type_ids_global())
        - get_unit_type_name() (fallback path only)
        - BUILDING_TYPES from shared_constants (fallback path only)
    """
    if _BUILDING_TYPE_IDS is not None:
        return unit_type_id in _BUILDING_TYPE_IDS

    # Fallback: string-based lookup (pre-031 behavior)
    name = get_unit_type_name(unit_type_id).lower()
    return name in BUILDING_TYPES


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


# ---------------------------------------------------------------------------
# UNIT_FIELD_CONFIG: Configurable field extraction definitions
#
# Each entry controls one output column per unit. To add or remove a field,
# simply edit this list -- no other code changes are required.
#
# Keys per entry:
#   column_suffix  (str)            -- suffix appended to the readable ID to
#                                      form the output column name
#   extract        (Callable)       -- lambda/function receiving a unit proto,
#                                      returning the value to store
#   always         (bool)           -- True  = always extract this field
#                                      False = only extract when `condition`
#                                              returns True
#   condition      (Callable|None)  -- lambda/function receiving a unit proto,
#                                      returning True when the field applies.
#                                      Ignored when `always` is True.
#   description    (str)            -- human-readable explanation of the field
# ---------------------------------------------------------------------------
UNIT_FIELD_CONFIG: List[Dict[str, Any]] = [
    # -- Position (single column, tuple string) ----------------------------
    {
        'column_suffix': 'pos_(X,Y,Z)',
        'extract': lambda unit: f"({unit.pos.x}, {unit.pos.y}, {unit.pos.z})",
        'always': True,
        'condition': None,
        'description': 'Position as (X, Y, Z) coordinate tuple',
    },
    # -- Vitals ------------------------------------------------------------
    {
        'column_suffix': 'health',
        'extract': lambda unit: f"{unit.health}/{unit.health_max}",
        'always': True,
        'condition': None,
        'description': 'Health as current/max fraction string',
    },
    {
        'column_suffix': 'shields',
        'extract': lambda unit: f"{unit.shield}/{unit.shield_max}",
        'always': False,
        'condition': lambda unit: unit.shield_max > 0,
        'description': 'Shields as current/max fraction string (Protoss only)',
    },
    {
        'column_suffix': 'energy',
        'extract': lambda unit: f"{unit.energy}/{unit.energy_max}",
        'always': False,
        'condition': lambda unit: unit.energy_max > 0,
        'description': 'Energy as current/max fraction string (casters only)',
    },
    # -- Orientation / geometry --------------------------------------------
    {
        'column_suffix': 'facing',
        'extract': lambda unit: unit.facing,
        'always': True,
        'condition': None,
        'description': 'Facing direction in radians',
    },
    {
        'column_suffix': 'radius',
        'extract': lambda unit: unit.radius,
        'always': True,
        'condition': None,
        'description': 'Unit collision radius',
    },
    # -- Build state -------------------------------------------------------
    {
        'column_suffix': 'build_progress',
        'extract': lambda unit: unit.build_progress,
        'always': True,
        'condition': None,
        'description': 'Build progress (0.0 to 1.0)',
    },
    # -- Status flags ------------------------------------------------------
    {
        'column_suffix': 'is_flying',
        'extract': lambda unit: unit.is_flying,
        'always': True,
        'condition': None,
        'description': 'Whether the unit is currently flying',
    },
    {
        'column_suffix': 'is_burrowed',
        'extract': lambda unit: unit.is_burrowed,
        'always': True,
        'condition': None,
        'description': 'Whether the unit is currently burrowed',
    },
    {
        'column_suffix': 'is_hallucination',
        'extract': lambda unit: unit.is_hallucination,
        'always': True,
        'condition': None,
        'description': 'Whether the unit is a hallucination',
    },
    # -- Combat ------------------------------------------------------------
    {
        'column_suffix': 'weapon_cooldown',
        'extract': lambda unit: unit.weapon_cooldown,
        'always': True,
        'condition': None,
        'description': 'Weapon cooldown remaining',
    },
    {
        'column_suffix': 'attack_upgrade_level',
        'extract': lambda unit: unit.attack_upgrade_level,
        'always': True,
        'condition': None,
        'description': 'Attack upgrade level (0-3)',
    },
    {
        'column_suffix': 'armor_upgrade_level',
        'extract': lambda unit: unit.armor_upgrade_level,
        'always': True,
        'condition': None,
        'description': 'Armor upgrade level (0-3)',
    },
    {
        'column_suffix': 'shield_upgrade_level',
        'extract': lambda unit: unit.shield_upgrade_level,
        'always': False,
        'condition': lambda unit: unit.shield_max > 0,
        'description': 'Shield upgrade level (0-3, Protoss only)',
    },
    # -- Cargo -------------------------------------------------------------
    {
        'column_suffix': 'cargo_space_taken',
        'extract': lambda unit: unit.cargo_space_taken,
        'always': True,
        'condition': None,
        'description': 'Cargo space currently occupied',
    },
    {
        'column_suffix': 'cargo_space_max',
        'extract': lambda unit: unit.cargo_space_max,
        'always': True,
        'condition': None,
        'description': 'Maximum cargo space available',
    },
    # -- Orders (derived) --------------------------------------------------
    {
        'column_suffix': 'order_count',
        'extract': lambda unit: len(unit.orders),
        'always': True,
        'condition': None,
        'description': 'Number of queued orders',
    },
    # -- Enhancement fields (031) ------------------------------------------
    # These fields were confirmed populated in observer mode by research
    # script 030-verify-data-raw.py. See research/030-api-type-classification.md
    # Section 5 for empirical verification details.
    {
        'column_suffix': 'buff_ids',
        'extract': lambda unit: json.dumps(list(unit.buff_ids)) if unit.buff_ids else "[]",
        'always': True,
        'condition': None,
        'description': 'Active buff IDs as JSON array string (e.g., "[271, 5]")',
    },
    {
        'column_suffix': 'buff_duration_remain',
        'extract': lambda unit: unit.buff_duration_remain,
        'always': True,
        'condition': None,
        'description': 'Remaining buff duration (game loops)',
    },
    {
        'column_suffix': 'buff_duration_max',
        'extract': lambda unit: unit.buff_duration_max,
        'always': True,
        'condition': None,
        'description': 'Maximum buff duration (game loops)',
    },
    {
        'column_suffix': 'engaged_target_tag',
        'extract': lambda unit: unit.engaged_target_tag,
        'always': True,
        'condition': None,
        'description': 'Tag of unit currently being attacked (0 when not engaged)',
    },
    {
        'column_suffix': 'detect_range',
        'extract': lambda unit: unit.detect_range,
        'always': True,
        'condition': None,
        'description': 'Detection range (0.0 for non-detectors, e.g., Observer, Overseer)',
    },
    {
        'column_suffix': 'radar_range',
        'extract': lambda unit: unit.radar_range,
        'always': True,
        'condition': None,
        'description': 'Radar range (0.0 for most units)',
    },
    {
        'column_suffix': 'is_active',
        'extract': lambda unit: unit.is_active,
        'always': True,
        'condition': None,
        'description': 'Whether the unit is actively performing an action',
    },
    {
        'column_suffix': 'cloak',
        'extract': lambda unit: unit.cloak,
        'always': True,
        'condition': None,
        'description': 'CloakState enum: 0=Unknown, 1=Cloaked, 2=CloakedDetected, 3=NotCloaked, 4=CloakedAllied',
    },
    {
        'column_suffix': 'display_type',
        'extract': lambda unit: unit.display_type,
        'always': True,
        'condition': None,
        'description': 'DisplayType enum: 1=Visible, 2=Snapshot, 3=Hidden, 4=Placeholder',
    },
]


class UnitExtractor:
    """
    Extracts unit data from SC2 observations.

    This class tracks units across frames, assigns readable IDs, and extracts
    comprehensive unit state information for ground truth data.

    Field extraction is driven entirely by UNIT_FIELD_CONFIG (module-level).
    Adding or removing entries from that list changes which columns appear in
    the output without modifying any code in this class.

    Lifecycle states are embedded into attribute columns:
    - "unit_started": On the gameloop where production starts, ALL attribute columns
    - "building": While the unit is being produced, ALL attribute columns
    - "completed": On the gameloop where the unit completes, ALL attribute columns
    - Real data: After completion, columns capture real data (pos, health, etc.)
    - "inside <building_type>": Unit is inside a building (gas, bunker, CC);
      position column gets the building's coordinates, other columns get the string
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

        # Track which conditional column_suffix values apply to each unit.
        # Maps readable_id -> set of column_suffix strings from conditional
        # UNIT_FIELD_CONFIG entries (e.g. {'shields', 'shield_upgrade_level', 'energy'}).
        # The schema_manager uses these sets to decide which extra columns to create.
        self.unit_attributes: Dict[str, Set[str]] = {}

        # Track last known position per tag for hidden-unit resolution.
        # Stored as raw (x, y, z) float tuples from the protobuf, NOT as
        # formatted strings, so resolve_hidden_units() can compute distances.
        self.last_known_positions: Dict[int, tuple] = {}

        # Track unit type name (lowercase) per tag for building-compatibility
        # matching in resolve_hidden_units().
        self.last_known_unit_type: Dict[int, str] = {}

        # Tags visible in the most recent extract() call. Used by
        # resolve_hidden_units() to identify which tracked units are hidden.
        self._current_tags: Set[int] = set()

    def set_building_type_ids(self, building_type_ids: FrozenSet[int]) -> None:
        """
        Set the API-derived building type ID set for is_building() lookups.

        Updates the module-level _BUILDING_TYPE_IDS so that is_building() uses
        O(1) integer lookup instead of string name conversion. Called by
        StateExtractor.set_api_type_data() after data_raw() is available.

        Args:
            building_type_ids: frozenset of integer unit type IDs that have
                Attribute.Structure in data_raw()

        Depends on / calls:
            - Modifies module-level _BUILDING_TYPE_IDS
        """
        global _BUILDING_TYPE_IDS
        _BUILDING_TYPE_IDS = building_type_ids

    def extract(self, obs) -> Dict[str, Dict]:
        """
        Extract all unit data from observation.

        Iterates through UNIT_FIELD_CONFIG to extract each configured field
        instead of hardcoding individual protobuf field accesses. For each
        config entry, the extract function is called if the entry is marked
        as 'always' or if its 'condition' passes for the unit.

        Lifecycle is embedded in the data dict via the '_lifecycle' key:
        - 'unit_started': First frame a unit appears with build_progress == 0.0
        - 'building': Unit is being produced (0.0 < build_progress < 1.0)
        - 'completed': The frame where build_progress transitions to >= 1.0
        - 'existing': Unit exists and is complete (normal data)
        - 'destroyed': Unit confirmed dead via raw_data.event.dead_units

        Units temporarily absent from raw_data.units (e.g., workers inside gas
        buildings, units in transports) are NOT marked destroyed — their tag
        mapping is preserved so they resume their existing entity ID when they
        reappear. Their columns simply contain NaN for the frames they are hidden.

        Units that never complete will still appear in extract() output (needed
        for pass 1 tracking), but schema_manager will filter them out when
        building columns.

        Args:
            obs: SC2 observation from controller.observe()

        Returns:
            Dictionary mapping readable IDs to unit data dicts.

        Depends on / calls:
            - is_building() to filter out buildings
            - get_unit_type_name() to resolve unit type IDs to names
            - _assign_readable_id() for new units
            - _determine_lifecycle() for lifecycle state
            - _extract_fields() to run UNIT_FIELD_CONFIG against a unit
            - _discover_conditional_attributes() to record which conditional
              fields apply to a unit (for schema building)
        """
        raw_data = obs.observation.raw_data
        units_data = {}

        # Get current frame's unit tags
        current_tags = set()

        # Process all units
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

            # Filter: Skip buildings (handled by BuildingExtractor)
            if is_building(unit.unit_type):
                continue

            # Filter: Skip untracked entity types (projectiles, temporary
            # effects, etc.) that the engine tracks as units but add noise
            # to the dataset. Configurable via UNTRACKED_ENTITY_TYPES in
            # shared_constants.py.
            if get_unit_type_name(unit.unit_type).lower() in UNTRACKED_ENTITY_TYPES:
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

            # Track position and unit type for hidden-unit resolution.
            # These are used by resolve_hidden_units() to match disappeared
            # units to nearby buildings (gas mining, bunker loading, etc.).
            self.last_known_positions[tag] = (unit.pos.x, unit.pos.y, unit.pos.z)
            self.last_known_unit_type[tag] = get_unit_type_name(unit.unit_type).lower()

            # Build the unit data dict from the configurable field definitions
            unit_data = {
                'tag': tag,
                'unit_type_id': unit.unit_type,
                'unit_type_name': get_unit_type_name(unit.unit_type),
                '_lifecycle': lifecycle,
            }

            # Extract all configured fields from the unit proto
            unit_data.update(self._extract_fields(unit))

            # Record which conditional attributes this unit has (for schema building).
            # Only discover on first encounter so the set is stable across frames.
            if readable_id not in self.unit_attributes:
                self.unit_attributes[readable_id] = (
                    self._discover_conditional_attributes(unit)
                )

            units_data[readable_id] = unit_data

        # Detect dead units using the authoritative dead_units event list.
        #
        # IMPORTANT: We intentionally do NOT treat "disappeared from observation"
        # as death. Units can temporarily leave raw_data.units without dying:
        # - Workers enter gas buildings (Refinery/Assimilator/Extractor) to mine
        #   and are removed from the unit list for ~65 game loops per cycle
        # - Units loaded into transports (Medivac, Overlord) or bunkers
        #
        # Previously, any tag absent from current_tags but present in
        # previous_tags was marked "destroyed" and its tag mapping was deleted.
        # This caused gas-mining workers to get new entity IDs every time they
        # exited a refinery, inflating entity counts by 5-10x (e.g., 490 SCVs
        # instead of ~27). See diagnostics/023-lifecycle-timeline-rca.md.
        #
        # The raw_data.event.dead_units list is the engine's authoritative
        # death signal — it fires only for actual unit deaths (combat, spells,
        # timed life expiry), never for units entering buildings or transports.
        # By relying solely on dead_units, temporarily hidden units keep their
        # tag mapping intact and resume their existing entity ID when they
        # reappear. _determine_lifecycle() handles reappearance correctly
        # because it checks completed_tags before is_new.
        #
        # Tag cleanup after confirmed death ensures that if the SC2 engine
        # recycles this tag for a genuinely new unit later, the extractor
        # assigns a fresh readable_id instead of merging two physical units
        # into one entity's columns.
        for dead_tag in raw_data.event.dead_units:
            if dead_tag in self.tag_to_readable_id and dead_tag in self.all_seen_tags:
                readable_id = self.tag_to_readable_id[dead_tag]
                self.dead_tags.add(dead_tag)
                units_data[readable_id] = {
                    'tag': dead_tag,
                    '_lifecycle': 'destroyed',
                }
                # Clean up dead tag mappings so recycled tags get fresh IDs
                del self.tag_to_readable_id[dead_tag]
                self.completed_tags.discard(dead_tag)
                self.previous_build_progress.pop(dead_tag, None)
                self.last_known_positions.pop(dead_tag, None)
                self.last_known_unit_type.pop(dead_tag, None)

        # Store current_tags for resolve_hidden_units() to identify which
        # tracked units are no longer visible in the observation this frame.
        self._current_tags = current_tags

        # Update previous tags for next iteration
        self.previous_tags = current_tags

        return units_data

    def resolve_hidden_units(
        self,
        buildings_data: Dict[str, Dict],
        passenger_map: Optional[Dict[int, Dict]] = None,
    ) -> Dict[str, Dict]:
        """
        Resolve hidden units using a 4-tier status resolution system.

        After extract() runs, some units may be tracked (in tag_to_readable_id)
        but not visible in the current observation (not in _current_tags). These
        are units that temporarily entered a container -- gas miners inside
        refineries, marines inside bunkers, SCVs loaded into command centers,
        troops inside Medivacs, etc.

        Resolution tiers (checked in order for each hidden unit):

        Tier 1 -- PASSENGERS (primary for non-gas containers):
            If the unit's tag appears in passenger_map (built from containers
            with cargo_space_max > 0), the unit is authoritatively inside that
            container. Gas refineries have cargo_space_max=0 and never appear
            in passenger_map.

        Tier 2 -- GAS MINING (distance heuristic for gas refineries):
            Gas refineries are NOT containers in the engine's data model --
            workers literally despawn during gas harvesting. The distance
            heuristic against gas buildings is the primary method for gas
            mining detection.

        Tier 3 -- AUTHORITATIVE DEATH:
            dead_units is the sole source of truth for unit death. This is
            handled in extract(), not here. By the time resolve_hidden_units()
            runs, dead tags have already been removed from tag_to_readable_id.

        Tier 4 -- FALLBACK (distance heuristic for remaining buildings):
            For any remaining unresolved disappearances, the distance heuristic
            checks against all non-gas UNIT_CONTAINING_BUILDINGS. Units that
            still don't match stay NaN (unresolved hidden state).

        Call this AFTER extract() and AFTER extracting buildings for the same
        player and frame. The returned dict should be merged into the units_data
        via dict.update().

        Args:
            buildings_data: Building data dict from BuildingExtractor.extract()
                            for the same player and frame. Keys are readable IDs
                            (e.g., 'p1_refinery_001'), values are data dicts with
                            '_lifecycle', 'pos_(X,Y,Z)', 'unit_type_name', etc.
            passenger_map: Dict mapping passenger tags (int) to container info
                           dicts with 'building_type' (str) and 'building_pos'
                           (str). Built by StateExtractor._collect_passenger_maps()
                           from units with cargo_space_max > 0. Defaults to empty
                           dict for backward compatibility.

        Returns:
            Dict mapping readable_id to data dict for each hidden unit matched
            to a container. Example entry:
                'p1_scv_005': {
                    'tag': 12345,
                    '_lifecycle': 'inside refinery',
                    'pos_(X,Y,Z)': '(42.5, 63.0, 11.98)',
                }
            Units not matched to any container are omitted (columns stay NaN).

        Depends on / calls:
            - self._current_tags (set by extract())
            - self.tag_to_readable_id, last_known_positions, last_known_unit_type
            - UNIT_CONTAINING_BUILDINGS, INSIDE_BUILDING_DISTANCE_THRESHOLD
              from shared_constants
            - _parse_position() module-level helper
        """
        from src_new.shared_constants import (
            UNIT_CONTAINING_BUILDINGS,
            INSIDE_BUILDING_DISTANCE_THRESHOLD,
        )

        if passenger_map is None:
            passenger_map = {}

        resolved: Dict[str, Dict] = {}

        # Identify hidden tags: tracked (have readable IDs) but not currently
        # visible in the observation. After dead_units cleanup in extract(),
        # only genuinely hidden (inside-building) units remain in this set.
        hidden_tags = set(self.tag_to_readable_id.keys()) - self._current_tags

        if not hidden_tags:
            return resolved

        # --- Tier 1: Passenger-based resolution (non-gas containers) ---------
        # Check each hidden tag against the passenger_map built from units
        # with cargo_space_max > 0. This is the authoritative, API-driven
        # detection for Medivacs, Bunkers, Command Centers, etc.
        tier1_resolved: Set[int] = set()
        for tag in hidden_tags:
            if tag in passenger_map:
                readable_id = self.tag_to_readable_id[tag]
                container_info = passenger_map[tag]
                # building_type from passenger_map uses get_unit_type_name()
                # (e.g. "Bunker", "Medivac"). Lowercase it for lifecycle string
                # consistency with the distance heuristic path.
                container_type_lower = container_info['building_type'].lower()
                resolved[readable_id] = {
                    'tag': tag,
                    '_lifecycle': f"inside {container_type_lower}",
                    'pos_(X,Y,Z)': container_info['building_pos'],
                }
                tier1_resolved.add(tag)

        # Remove Tier 1 resolved tags from the set needing further resolution
        remaining_hidden = hidden_tags - tier1_resolved

        if not remaining_hidden:
            return resolved

        # --- Build candidate building lists for Tier 2 and Tier 4 ------------
        # Split buildings into gas refineries (Tier 2) and non-gas containers
        # (Tier 4) since they serve different resolution tiers.
        #
        # Gas building type names (lowercase) for identification:
        gas_building_types = frozenset({
            'refinery', 'refineryrich',
            'assimilator', 'assimilatorrich',
            'extractor', 'extractorrich',
        })

        gas_buildings: List[Dict] = []       # Tier 2 candidates
        non_gas_buildings: List[Dict] = []   # Tier 4 candidates

        for building_id, building_data in buildings_data.items():
            lifecycle = building_data.get('_lifecycle', '')
            # Skip destroyed/cancelled buildings (can't contain units) and
            # buildings still under construction (can't be entered yet).
            if lifecycle not in ('existing', 'completed'):
                continue

            pos_str = building_data.get('pos_(X,Y,Z)')
            building_type = building_data.get('unit_type_name', '').lower()
            if not pos_str or not building_type:
                continue

            # Only consider buildings that are known to contain units
            if building_type not in UNIT_CONTAINING_BUILDINGS:
                continue

            try:
                pos = _parse_position(pos_str)
            except (ValueError, IndexError):
                continue

            building_entry = {
                'building_type': building_type,
                'position': pos,
                'pos_string': pos_str,
                'compatible_units': UNIT_CONTAINING_BUILDINGS[building_type],
            }

            if building_type in gas_building_types:
                gas_buildings.append(building_entry)
            else:
                non_gas_buildings.append(building_entry)

        # --- Tier 2: Gas mining detection (distance heuristic) ---------------
        # Gas refineries have cargo_space_max=0 and never populate passengers.
        # Workers literally despawn during gas harvesting. The distance
        # heuristic is the primary detection method for gas mining.
        tier2_resolved: Set[int] = set()
        if gas_buildings:
            for tag in remaining_hidden:
                readable_id = self.tag_to_readable_id[tag]
                last_pos = self.last_known_positions.get(tag)
                unit_type = self.last_known_unit_type.get(tag)

                if last_pos is None or unit_type is None:
                    continue

                best_building = None
                best_distance = INSIDE_BUILDING_DISTANCE_THRESHOLD

                for building in gas_buildings:
                    # Check unit-type compatibility (e.g., only SCVs in Refineries)
                    compatible = building['compatible_units']
                    if compatible is not None and unit_type not in compatible:
                        continue

                    # 2D distance (x, y); z is elevation and irrelevant
                    dx = last_pos[0] - building['position'][0]
                    dy = last_pos[1] - building['position'][1]
                    dist = (dx * dx + dy * dy) ** 0.5

                    if dist < best_distance:
                        best_distance = dist
                        best_building = building

                if best_building is not None:
                    resolved[readable_id] = {
                        'tag': tag,
                        '_lifecycle': f"inside {best_building['building_type']}",
                        'pos_(X,Y,Z)': best_building['pos_string'],
                    }
                    tier2_resolved.add(tag)

        # Remove Tier 2 resolved tags
        remaining_hidden = remaining_hidden - tier2_resolved

        if not remaining_hidden:
            return resolved

        # --- Tier 4: Fallback distance heuristic (non-gas buildings) ---------
        # For any remaining unresolved disappearances, check against non-gas
        # UNIT_CONTAINING_BUILDINGS (Bunkers, Command Centers, Nydus, etc.).
        # This catches cases where the passengers field was empty even for
        # loaded transports (unconfirmed edge case) or Nydus transit.
        if non_gas_buildings:
            for tag in remaining_hidden:
                readable_id = self.tag_to_readable_id[tag]
                last_pos = self.last_known_positions.get(tag)
                unit_type = self.last_known_unit_type.get(tag)

                if last_pos is None or unit_type is None:
                    continue

                best_building = None
                best_distance = INSIDE_BUILDING_DISTANCE_THRESHOLD

                for building in non_gas_buildings:
                    # Check unit-type compatibility (None = any unit accepted)
                    compatible = building['compatible_units']
                    if compatible is not None and unit_type not in compatible:
                        continue

                    # 2D distance (x, y); z is elevation and irrelevant
                    dx = last_pos[0] - building['position'][0]
                    dy = last_pos[1] - building['position'][1]
                    dist = (dx * dx + dy * dy) ** 0.5

                    if dist < best_distance:
                        best_distance = dist
                        best_building = building

                if best_building is not None:
                    resolved[readable_id] = {
                        'tag': tag,
                        '_lifecycle': f"inside {best_building['building_type']}",
                        'pos_(X,Y,Z)': best_building['pos_string'],
                    }

        # Units that remain unresolved after all tiers are omitted --
        # their columns stay NaN (unresolved hidden state). They will
        # naturally reappear if temporarily hidden, or be marked destroyed
        # when they appear in dead_units (handled in extract()).
        return resolved

    @staticmethod
    def _extract_fields(unit) -> Dict[str, Any]:
        """
        Run every entry in UNIT_FIELD_CONFIG against a unit proto and return
        the extracted values as a dict keyed by column_suffix.

        For 'always' fields the extract function is called unconditionally.
        For conditional fields the extract function is called only when the
        condition lambda returns True; otherwise the field is omitted.

        Args:
            unit: A unit proto from raw_data.units

        Returns:
            Dict mapping column_suffix -> extracted value, e.g.
            {'pos_(X,Y,Z)': '(42.5, 63.0, 11.98)', 'health': '45.0/45.0', ...}

        Depends on / calls:
            - UNIT_FIELD_CONFIG (module-level list)
        """
        fields: Dict[str, Any] = {}

        for entry in UNIT_FIELD_CONFIG:
            # Determine whether this field should be included for this unit
            if entry['always']:
                # Always-present field -- extract unconditionally
                fields[entry['column_suffix']] = entry['extract'](unit)
            else:
                # Conditional field -- only extract when condition passes
                condition_fn = entry.get('condition')
                if condition_fn is not None and condition_fn(unit):
                    fields[entry['column_suffix']] = entry['extract'](unit)

        return fields

    @staticmethod
    def _discover_conditional_attributes(unit) -> Set[str]:
        """
        Determine which conditional UNIT_FIELD_CONFIG entries apply to a unit.

        Returns the set of column_suffix values for conditional fields whose
        condition passes. The schema_manager uses this set to decide which
        extra columns (shields, energy, etc.) to create for this unit.

        Args:
            unit: A unit proto from raw_data.units

        Returns:
            Set of column_suffix strings for applicable conditional fields,
            e.g. {'shields', 'shield_upgrade_level'} for a Protoss unit.

        Depends on / calls:
            - UNIT_FIELD_CONFIG (module-level list)
        """
        conditional_suffixes: Set[str] = set()

        for entry in UNIT_FIELD_CONFIG:
            # Only interested in conditional (non-always) entries
            if entry['always']:
                continue

            condition_fn = entry.get('condition')
            if condition_fn is not None and condition_fn(unit):
                conditional_suffixes.add(entry['column_suffix'])

        return conditional_suffixes

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
        Get the set of conditional column_suffix values for a given unit.

        The returned set contains column_suffix strings from conditional
        UNIT_FIELD_CONFIG entries that apply to this unit (e.g. 'shields',
        'shield_upgrade_level', 'energy').  The schema_manager uses this to
        decide which extra attribute columns to create.

        Args:
            readable_id: The human-readable unit ID (e.g. 'p1_marine_001')

        Returns:
            Set of column_suffix strings for applicable conditional fields.
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
        self.last_known_positions.clear()
        self.last_known_unit_type.clear()
        self._current_tags.clear()


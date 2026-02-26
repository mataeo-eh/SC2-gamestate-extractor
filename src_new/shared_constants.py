"""
shared_constants.py: Centralized constants for the SC2 replay extraction pipeline.

This module is the single source of truth for all shared constant sets used across
the extraction pipeline (building_extractor, unit_extractor, state_extractor,
wide_table_builder, schema_manager, etc.).

Design rationale:
  - frozenset is used for all type-membership sets: it is immutable (signals these
    are read-only constants) and provides O(1) membership testing.
  - All unit/building names are stored in LOWERCASE to match the sanitized key
    format already used throughout the pipeline (e.g., "commandcenter" not
    "CommandCenter").
  - Names come from pysc2.lib.units enum member names, lowercased.

Building ID cross-reference notes (building_extractor.py comment collisions):
  The original building_extractor.py contained three duplicate integer keys with
  conflicting comments. The CORRECT resolution per pysc2.lib.units is:
    ID 133 --> WarpGate (Protoss)   [NOT TechLab; TechLab = ID 5]
    ID 138 --> CreepTumorQueen (Zerg) [NOT FactoryReactor; FactoryReactor = ID 40]
    ID 142 --> NydusCanal (Zerg)    [NOT StarportReactor (ID 42) and NOT LurkerDenMP;
                                     LurkerDen = ID 504]
"""

import re


# ---------------------------------------------------------------------------
# BUILDING_TYPES
# ---------------------------------------------------------------------------

BUILDING_TYPES: frozenset = frozenset({

    # -----------------------------------------------------------------------
    # TERRAN BUILDINGS
    # Names sourced from pysc2.lib.units.Terran enum (lowercased).
    # -----------------------------------------------------------------------

    # Town-hall tier
    "commandcenter",          # ID 18  -- base tier
    "orbitalcommand",         # ID 132 -- morphed from CommandCenter
    "planetaryfortress",      # ID 130 -- morphed from CommandCenter

    # Lifted (flying) variants of liftable Terran buildings.
    # Terran buildings retain their role while airborne; we include them so
    # that a building does not disappear from the set when it lifts off.
    "commandcenterflying",    # ID 36
    "orbitalcommandflying",   # ID 134
    "barracksflying",         # ID 46
    "factoryflying",          # ID 43
    "starportflying",         # ID 44

    # Supply
    "supplydepot",            # ID 19
    "supplydepotlowered",     # ID 47 -- SupplyDepot burrowed below ground

    # Gas
    "refinery",               # ID 20
    "refineryrich",           # ID 1960 -- rich-vespene variant

    # Military production
    "barracks",               # ID 21
    "factory",                # ID 27
    "starport",               # ID 28

    # Tech / support
    "engineeringbay",         # ID 22
    "missileturret",          # ID 23
    "bunker",                 # ID 24
    "sensortower",            # ID 25
    "ghostacademy",           # ID 26
    "armory",                 # ID 29
    "fusioncore",             # ID 30

    # Add-on base types (generic; not attached to a specific host building)
    "techlab",                # ID 5
    "reactor",                # ID 6

    # Add-on variants attached to specific host buildings.
    # Each production building has its own named tech-lab and reactor variant.
    "barrackstechlab",        # ID 37
    "barracksreactor",        # ID 38
    "factorytechlab",         # ID 39
    "factoryreactor",         # ID 40
    "starporttechlab",        # ID 41
    "starportreactor",        # ID 42

    # -----------------------------------------------------------------------
    # PROTOSS BUILDINGS
    # Names sourced from pysc2.lib.units.Protoss enum (lowercased).
    # -----------------------------------------------------------------------

    # Town-hall
    "nexus",                  # ID 59

    # Power / supply
    "pylon",                  # ID 60

    # Gas
    "assimilator",            # ID 61
    "assimilatorrich",        # ID 1955 -- rich-vespene variant

    # Military production (and its warp-gate morph)
    "gateway",                # ID 62
    "warpgate",               # ID 133 -- morphed from Gateway (per user decision: included)

    # Tech / support
    "forge",                  # ID 63
    "fleetbeacon",            # ID 64
    "twilightcouncil",        # ID 65
    "photoncannon",           # ID 66
    "shieldbattery",          # ID 1910
    "stargate",               # ID 67
    "templararchive",         # ID 68
    "darkshrine",             # ID 69
    "roboticsbay",            # ID 70
    "roboticsfacility",       # ID 71
    "cyberneticscore",        # ID 72

    # -----------------------------------------------------------------------
    # ZERG BUILDINGS
    # Names sourced from pysc2.lib.units.Zerg enum (lowercased).
    # -----------------------------------------------------------------------

    # Town-hall tier (morph chain)
    "hatchery",               # ID 86
    "lair",                   # ID 100 -- morphed from Hatchery
    "hive",                   # ID 101 -- morphed from Lair

    # Gas
    "extractor",              # ID 88
    "extractorrich",          # ID 1956 -- rich-vespene variant

    # Military tech buildings
    "spawningpool",           # ID 89
    "evolutionchamber",       # ID 90
    "hydraliskden",           # ID 91
    "spire",                  # ID 92
    "greaterspire",           # ID 102 -- morphed from Spire
    "ultraliskcavern",        # ID 93
    "infestationpit",         # ID 94
    "banelingnest",           # ID 96
    "roachwarren",            # ID 97
    "lurkerden",              # ID 504 -- (was misidentified as ID 142 in old extractor)

    # Nydus
    "nydusnetwork",           # ID 95
    "nyduscanal",             # ID 142 -- worm exit point placed by NydusNetwork

    # Static defenses
    "spinecrawler",           # ID 98
    "spinecrawleruprooted",   # ID 139 -- SpineCrawler while moving
    "sporecrawler",           # ID 99
    "sporecrawleruprooted",   # ID 140 -- SporeCrawler while moving

    # Creep tumors -- included because they are placed structures that appear
    # in the raw unit list as building-type entries.
    "creeptumor",             # ID 87
    "creeptumorburrowed",     # ID 137 -- burrowed (active) state
    "creeptumorqueen",        # ID 138 -- placed directly by a Queen

})


# ---------------------------------------------------------------------------
# WORKER_TYPES
# ---------------------------------------------------------------------------

WORKER_TYPES: frozenset = frozenset({
    "scv",    # Terran worker
    "probe",  # Protoss worker
    "drone",  # Zerg worker
    # MULE is intentionally included: it is a resource-gathering economic unit
    # summoned by Orbital Command energy, and counts toward mineral collection
    # rate tracking in economy extraction even though it is a temporary unit.
    "mule",
})


# ---------------------------------------------------------------------------
# BASE_TYPES
# ---------------------------------------------------------------------------
# Town-hall buildings used to determine "base location" for army direction
# calculation (e.g., "which direction is the army moving relative to base?").
# Includes all morphed variants so the base remains detectable through the
# morph chain.

BASE_TYPES: frozenset = frozenset({
    # Terran
    "commandcenter",
    "orbitalcommand",
    "planetaryfortress",
    "commandcenterflying",    # lifted; still counts as a base location anchor
    "orbitalcommandflying",

    # Protoss
    "nexus",

    # Zerg (morph chain)
    "hatchery",
    "lair",
    "hive",
})


# ---------------------------------------------------------------------------
# AIR_UNIT_TYPES
# ---------------------------------------------------------------------------
# Units that are inherently airborne (cannot land or be grounded).
# Does NOT include ground units with temporary air modes (e.g., WarpPrism
# can switch to Phasing mode but is still fundamentally an air unit so it
# is included; VikingFighter lands as VikingAssault which is a separate type).

AIR_UNIT_TYPES: frozenset = frozenset({

    # --- Terran air units ---
    "banshee",            # cloakable air attack unit
    "battlecruiser",      # capital ship
    "liberator",          # air/ground siege unit (air mode = LiberatorAG excluded; same unit)
    "medivac",            # air transport / healer
    "raven",              # spellcaster
    "vikingfighter",      # anti-air form of Viking (VikingAssault is ground)

    # Lifted Terran buildings are NOT air combat units -- excluded here;
    # they remain in BUILDING_TYPES.

    # --- Protoss air units ---
    "carrier",            # capital ship with Interceptors
    # Colossus is NOT an air unit -- it's a ground cliff-walker; excluded.
    "oracle",             # harassment air unit
    "phoenix",            # air superiority fighter
    "tempest",            # heavy siege air unit
    "voidray",            # beam attack air unit
    "warpprism",          # air transport / pylon-equivalent
    "warpprismphasing",   # WarpPrism in stationary power field mode (still airborne)
    "mothership",         # capital unit (cloaks nearby units)
    "mothershipcore",     # early-game Nexus attachment (Legacy of the Void era)
    "observer",           # detector air unit (scouting)

    # --- Zerg air units ---
    "broodlord",          # air siege unit (spawns Broodlings)
    "corruptor",          # anti-air specialist / corrupts massive units
    "mutalisk",           # fast harassment air unit
    "overlord",           # supply provider / transport base form
    "overlordtransport",  # Overlord after Evolution Chamber research
    "overseer",           # morphed Overlord; detector
    "viper",              # spellcaster (Blinding Cloud, Parasitic Bomb, etc.)

})


# ---------------------------------------------------------------------------
# PRODUCTION_BUILDING_TYPES
# ---------------------------------------------------------------------------
# Buildings that produce units or conduct research. Used in army composition
# inference and build-order analysis. "warpgate" is explicitly included per
# user decision: Gateway morphs to WarpGate and retains production capability.

PRODUCTION_BUILDING_TYPES: frozenset = frozenset({

    # --- Terran production ---
    "barracks",           # trains Infantry
    "factory",            # trains Mechanical ground
    "starport",           # trains air units
    # Add-on variants are also production buildings (TechLab enables advanced units)
    "barrackstechlab",
    "barracksreactor",    # Reactor doubles production queue
    "factorytechlab",
    "factoryreactor",
    "starporttechlab",
    "starportreactor",
    # Orbital Command can call down MULEs and Scan (not unit production per se,
    # but it uses energy-based abilities on a production-tier building)
    "orbitalcommand",
    "orbitalcommandflying",

    # --- Terran research buildings ---
    "engineeringbay",     # upgrades Infantry armor/weapons
    "armory",             # upgrades vehicle/ship armor/weapons
    "ghostacademy",       # enables Ghost nuke and research
    "fusioncore",         # enables Battlecruiser research

    # --- Protoss production ---
    "gateway",            # trains ground units
    "warpgate",           # morphed Gateway; warps in units instantly at pylon range
    "roboticsfacility",   # trains Robotic units
    "stargate",           # trains air units

    # --- Protoss research buildings ---
    "forge",              # upgrades ground armor/weapons
    "cyberneticscore",    # enables air upgrades + Gateway upgrades
    "twilightcouncil",    # enables charge, blink, resonating glaives
    "templararchive",     # enables storm, preservation
    "darkshrine",         # enables shadow stride
    "roboticsbay",        # enables colossus, disruptor
    "fleetbeacon",        # enables void ray, carrier, tempest upgrades

    # --- Zerg production ---
    "hatchery",           # produces units from Larvae
    "lair",               # produces units from Larvae (morphed hatchery)
    "hive",               # produces units from Larvae (morphed lair)

    # --- Zerg research buildings ---
    "spawningpool",       # enables Zergling speed + upgrades
    "evolutionchamber",   # upgrades melee/missile/carapace
    "hydraliskden",       # enables Hydralisk upgrades
    "spire",              # enables air attack/armor upgrades
    "greaterspire",       # morphed Spire; enables Broodlord
    "ultraliskcavern",    # enables Ultralisk upgrades
    "infestationpit",     # enables Infestor upgrades
    "banelingnest",       # enables Baneling speed
    "roachwarren",        # enables Roach upgrades
    "lurkerden",          # enables Lurker + upgrades

})


# ---------------------------------------------------------------------------
# ALIVE_STATES
# ---------------------------------------------------------------------------
# Lifecycle state strings that indicate a unit or building is still present
# in the game world (not destroyed, not cancelled). Used for filtering active
# entities in wide-table queries and army direction calculations.
#
# "built"     -- used by building_extractor for a fully completed structure
# "existing"  -- used by unit_extractor for a live unit

ALIVE_STATES: frozenset = frozenset({
    "built",
    "existing",
})


# ---------------------------------------------------------------------------
# UNIT_LIFECYCLE_OVERRIDE_STATES
# ---------------------------------------------------------------------------
# The complete set of lifecycle state strings that a unit entity can hold.
# These strings appear as values in the '_lifecycle' field of unit data dicts.
# Used by schema_manager and wide_table_builder to validate or branch on state.
#
#   "unit_started"  -- unit has been queued/started training in a building
#   "building"      -- unit is being constructed (for units that take time, e.g., Zerg morphs)
#   "completed"     -- unit has finished training and is alive
#   "destroyed"     -- unit was killed or sacrificed

UNIT_LIFECYCLE_OVERRIDE_STATES: frozenset = frozenset({
    "unit_started",
    "building",
    "completed",
    "destroyed",
})


# ---------------------------------------------------------------------------
# BUILDING_LIFECYCLE_OVERRIDE_STATES
# ---------------------------------------------------------------------------
# The complete set of lifecycle state strings that a building entity can hold.
# These strings appear as values in the '_lifecycle' field of building data dicts.
# Used by schema_manager and wide_table_builder to validate or branch on state.
#
#   "building_started"  -- construction has begun (foundation placed)
#   "completed"         -- building is fully constructed and operational
#   "destroyed"         -- building was destroyed by enemy action or self-destruct
#   "cancelled"         -- construction was cancelled mid-build (partial refund issued)

BUILDING_LIFECYCLE_OVERRIDE_STATES: frozenset = frozenset({
    "building_started",
    "completed",
    "destroyed",
    "cancelled",
})


# ---------------------------------------------------------------------------
# NON_ARMY_TYPES
# ---------------------------------------------------------------------------
# Union of all building names and worker names. Used to exclude non-combat
# entities when computing army strength, army composition, or army direction.
# Any unit whose type name is in this set should be excluded from army metrics.

NON_ARMY_TYPES: frozenset = BUILDING_TYPES | WORKER_TYPES


# ---------------------------------------------------------------------------
# ECONOMY_COLUMN_SUFFIXES
# ---------------------------------------------------------------------------
# Ordered tuple of the economy metric column suffixes appended after the
# player prefix (e.g., "p1_minerals", "p2_vespene").
# Order is preserved for schema construction in schema_manager.py.
# Must remain a tuple (not frozenset) because column order matters for parquet
# schema stability and pandas DataFrame alignment.

ECONOMY_COLUMN_SUFFIXES: tuple = (
    "minerals",
    "vespene",
    "supply_used",
    "supply_cap",
    "collection_rate_minerals",
    "collection_rate_vespene",
)


# ---------------------------------------------------------------------------
# ENTITY_COL_RE
# ---------------------------------------------------------------------------
# Pre-compiled regular expression for parsing wide-table column names back
# into their component parts.
#
# Column naming convention used throughout the pipeline:
#   {player}_{botname}_{entitytype}_{sequence_id}_{attribute}
#   e.g.:  "p1_really_marine_003_health"
#          "p2_what_nexus_001_shields"
#          "p1_bot_v2_0_commandcenter_001_build_progress"
#
# Groups captured:
#   Group 1 -- player prefix  : "p1" or "p2"
#   Group 2 -- middle portion : combined {botname}_{entitytype} (e.g., "really_marine",
#              "bot_v2_0_commandcenter"). To extract the entity type alone, use:
#              entity_type = middle.rsplit('_', 1)[-1]
#              This works because SC2 type names never contain underscores after
#              sanitization (e.g., "Marine" -> "marine", "CommandCenter" -> "commandcenter").
#   Group 3 -- zero-padded 3-digit sequence id (e.g., "003")
#              Sequence IDs are assigned in order of first appearance per
#              player per type within a single replay.
#   Group 4 -- attribute name  : e.g., "health", "x", "y", "lifecycle"
#              May contain underscores (e.g., "build_progress", "is_flying").
#
# To reconstruct a column prefix (for building column names):
#   col_prefix = f"{player}_{middle}_{entity_id}"
#
# Usage example:
#   m = ENTITY_COL_RE.match(col_name)
#   if m:
#       player, middle, seq_id, attribute = m.groups()
#       entity_type = middle.rsplit('_', 1)[-1]

ENTITY_COL_RE = re.compile(r"^(p[12])_(.+)_(\d{3})_(.+)$")

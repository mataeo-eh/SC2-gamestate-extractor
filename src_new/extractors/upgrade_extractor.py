"""
UpgradeExtractor: Extracts upgrade data from SC2 observations.

This component handles:
- Extracting upgrade completion from replay tracker events (primary path)
- Parsing upgrade category and level from upgrade name strings
- Providing per-loop upgrade snapshots via get_upgrades_at_loop()

Primary API (used by extraction_pipeline.py):
    load_upgrade_snapshots(replay_path) -> dict
        Parses replay tracker events once before the game loop begins.
        Returns all SUpgradeEvent completions grouped by player, sorted by game_loop.
    get_upgrades_at_loop(snapshots, player_id, game_loop) -> dict
        Returns the set of upgrades completed at or before a given game_loop.
        Uses binary search for O(log n) lookup. Called once per player per loop.

    This mirrors the pattern used by economy_extractor.py for SPlayerStatsEvent.

    NOTE on upgrade_ids from the SC2 engine:
        In observer mode (observed_player_id=0), raw_data.player.upgrade_ids is
        always empty regardless of which player perspective is active via
        RequestObserverAction. This was verified empirically: switching perspective
        in observer mode does NOT populate upgrade_ids. Only a direct replay load
        with observed_player_id=1 or 2 exposes them, which requires a full replay
        restart. The UpgradeExtractor class below (engine-based) is therefore
        a no-op in the pipeline's observer mode path and is retained only for
        reference or non-observer use cases.

    NOTE on upgrade start time:
        Tracker events only record upgrade *completion*. started_loop is set to None.
        In the future, started_loop can be back-calculated by subtracting the known
        research duration for each upgrade type (a searchable table exists per patch
        version) from completed_loop. This is left as a future enhancement.
"""

from bisect import bisect_right
from typing import Dict, List, Set, Tuple, Optional
import logging
import re

import mpyq
from s2protocol import versions as s2versions

from pysc2.lib import upgrades as pysc2_upgrades


# ---------------------------------------------------------------------------
# Tracker-event-based upgrade loading (primary pipeline path)
# ---------------------------------------------------------------------------

# s2protocol tracker event type for upgrade completions
_UPGRADE_EVENT = 'NNet.Replay.Tracker.SUpgradeEvent'

# Cosmetic upgrade name prefixes to exclude from pipeline data.
# These are client-side visual effects, not tech upgrades.
_COSMETIC_PREFIXES = ('Spray',)

# Module-level pre-computed game loop arrays for binary search.
# Built once by load_upgrade_snapshots(), consumed by get_upgrades_at_loop().
# Keyed by player_id (int). Each worker process has its own copy.
_precomputed_upgrade_loops: Dict[int, List[int]] = {}


logger = logging.getLogger(__name__)


def load_upgrade_snapshots(replay_path: str) -> Dict[int, List[Tuple[int, str]]]:
    """
    Parse a replay file and extract all SUpgradeEvent upgrade completions.

    Opens the replay as an MPQ archive, decodes tracker events using the
    protocol version embedded in the replay header, and collects every
    SUpgradeEvent (upgrade completion) per player. Cosmetic upgrades
    (e.g. Sprays) are filtered out.

    Returns a dict mapping player_id (1-indexed) to a list of
    (game_loop, upgrade_name) tuples, sorted by game_loop ascending.
    This list is consumed by get_upgrades_at_loop() to forward-fill
    upgrade state into each row of the pipeline output.

    This follows the same pattern as load_economy_snapshots() in
    economy_extractor.py. Call this once before the game loop begins.

    Args:
        replay_path: Absolute or relative path to the .SC2Replay file.

    Returns:
        Dict mapping player_id (int, 1-indexed) to sorted list of
        (game_loop: int, upgrade_name: str) tuples. Players with no
        meaningful upgrades get an empty list.

    Raises:
        FileNotFoundError: If replay_path does not exist.

    Depends on / calls:
        - mpyq.MPQArchive: opens the MPQ container
        - s2protocol.versions: decodes the replay header and tracker events
        - Called by extraction_pipeline._observer_mode_processing()
    """
    logger.info(f"Loading upgrade snapshots from: {replay_path}")

    archive = mpyq.MPQArchive(replay_path)

    # Decode the header to determine the correct protocol version
    header_content = archive.header['user_data_header']['content']
    header = s2versions.latest().decode_replay_header(header_content)
    base_build = header['m_version']['m_baseBuild']

    logger.debug(f"Upgrade snapshot base build: {base_build}")

    protocol = s2versions.build(base_build)
    tracker_raw = archive.read_file('replay.tracker.events')

    # Accumulate (game_loop, upgrade_name) pairs per player
    events_per_player: Dict[int, List[Tuple[int, str]]] = {}

    event_count = 0
    for event in protocol.decode_replay_tracker_events(tracker_raw):
        if event['_event'] != _UPGRADE_EVENT:
            continue

        event_count += 1

        player_id = event['m_playerId']
        game_loop = event['_gameloop']

        # upgrade name arrives as bytes in some protocol versions
        name = event['m_upgradeTypeName']
        if isinstance(name, bytes):
            name = name.decode('utf-8')

        # Skip cosmetic upgrades (Sprays, etc.)
        if any(name.startswith(prefix) for prefix in _COSMETIC_PREFIXES):
            logger.debug(f"Skipping cosmetic upgrade: {name}")
            continue

        events_per_player.setdefault(player_id, []).append((game_loop, name))

    # Sort each player's events by game_loop (should already be ordered,
    # but sort defensively)
    for pid in events_per_player:
        events_per_player[pid].sort(key=lambda t: t[0])

    # Pre-compute game loop arrays for O(log n) binary search in get_upgrades_at_loop().
    # Built once here instead of rebuilding on every call.
    _precomputed_upgrade_loops.clear()
    for pid, pairs in events_per_player.items():
        _precomputed_upgrade_loops[pid] = [gl for gl, _ in pairs]

    logger.info(
        f"Loaded {event_count} SUpgradeEvent(s) across "
        f"{len(events_per_player)} player(s) from replay"
    )

    return events_per_player


def get_upgrades_at_loop(
    snapshots: Dict[int, List[Tuple[int, str]]],
    player_id: int,
    game_loop: int,
) -> Dict[str, Dict]:
    """
    Return all upgrades completed at or before a given game loop for a player.

    Uses binary search on the sorted (game_loop, upgrade_name) list from
    load_upgrade_snapshots() to find every upgrade with game_loop <= the
    requested game_loop. Builds and returns a dict in the same format as
    UpgradeExtractor.extract() so that downstream schema and row-building
    code requires no changes.

    Args:
        snapshots: Dict returned by load_upgrade_snapshots().
                   Maps player_id -> sorted list of (game_loop, upgrade_name).
        player_id: Which player to query (1-indexed, typically 1 or 2).
        game_loop: The current game loop. Returns all upgrades completed
                   at or before this loop.

    Returns:
        Dict mapping lowercase upgrade name to upgrade detail dict:
        {
            'terraninfantryweaponslevel1': {
                'upgrade_name': 'TerranInfantryWeaponsLevel1',
                'category': 'weapons',
                'level': 1,
                'status': 'completed',
                'completed_loop': 5510,
                'started_loop': None,
                # NOTE: started_loop is always None. It can be back-calculated
                # in a future enhancement by subtracting the known research
                # duration for each upgrade type (available in per-patch tables)
                # from completed_loop.
            },
            ...
        }
        Returns an empty dict if no upgrades have been completed yet.

    Depends on / calls:
        - _precomputed_upgrade_loops (module-level cache built by load_upgrade_snapshots)
        - parse_upgrade_details() for category and level parsing
        - bisect_right for O(log n) search
        - Called by extraction_pipeline._observer_mode_processing() per player per loop
    """
    player_events = snapshots.get(player_id, [])

    if not player_events:
        return {}

    # Use pre-computed game loop array for O(log n) binary search
    game_loops = _precomputed_upgrade_loops.get(player_id)
    if game_loops is None:
        # Fallback: build on the fly if called outside normal pipeline flow
        game_loops = [gl for gl, _ in player_events]

    # bisect_right gives insertion point after any equal game_loop values,
    # so [:idx] includes all upgrades completed at or before game_loop
    idx = bisect_right(game_loops, game_loop)

    completed = player_events[:idx]

    result = {}
    for completed_loop, name in completed:
        category, level = parse_upgrade_details(name)
        key = name.lower()
        result[key] = {
            'upgrade_name': name,
            'category': category,
            'level': level,
            'status': 'completed',
            'completed_loop': completed_loop,
            'started_loop': None,
        }

    return result


# ---------------------------------------------------------------------------
# Engine-based upgrade extraction (NOTE: no-op in observer mode)
# ---------------------------------------------------------------------------
# The functions and class below read raw_data.player.upgrade_ids from the SC2
# engine observation. In observer mode (observed_player_id=0), upgrade_ids is
# always empty regardless of perspective switching — this was confirmed by
# diagnostic testing. These remain for non-observer use cases only.
# ---------------------------------------------------------------------------


def get_upgrade_name(upgrade_id: int) -> str:
    """
    Convert upgrade ID to human-readable name.

    Args:
        upgrade_id: SC2 upgrade ID

    Returns:
        Upgrade name string
    """
    try:
        return pysc2_upgrades.Upgrades(upgrade_id).name
    except (ValueError, AttributeError):
        return f"Unknown({upgrade_id})"


def parse_upgrade_details(upgrade_name: str) -> Tuple[str, int]:
    """
    Parse upgrade name to extract category and level.

    NOTE (B-3): The categorization below is a best-effort heuristic based on keyword
    matching against the upgrade name string. It does NOT cover all edge cases.
    Known limitation: "ChitinousPlating" (a Zerg armor upgrade for Ultralisks)
    is categorized as "other" instead of "armor" because its name does not contain
    any of the armor keywords ("armor", "armour"). Other upgrades with non-obvious
    names may similarly be miscategorized. A comprehensive fix would require a
    hardcoded lookup table mapping upgrade names to their true SC2 categories.

    Args:
        upgrade_name: Human-readable upgrade name (e.g., "TerranInfantryWeaponsLevel1")

    Returns:
        Tuple of (category, level):
        - category: Type of upgrade (e.g., "weapons", "armor", "shields", "other")
        - level: Upgrade level (0 if not applicable, 1-3 for leveled upgrades)

    Examples:
        >>> parse_upgrade_details("TerranInfantryWeaponsLevel1")
        ("weapons", 1)
        >>> parse_upgrade_details("Stimpack")
        ("other", 0)
        >>> parse_upgrade_details("ProtossGroundArmorLevel2")
        ("armor", 2)
    """
    category = "other"
    level = 0

    # Convert to lowercase for easier matching
    name_lower = upgrade_name.lower()

    # Determine category
    if any(keyword in name_lower for keyword in ["weapon", "weapons", "melee", "missile", "ship", "attack"]):
        category = "weapons"
    elif "armor" in name_lower or "armour" in name_lower:
        category = "armor"
    elif "shield" in name_lower or "shields" in name_lower:
        category = "shields"
    elif "speed" in name_lower or "movement" in name_lower:
        category = "movement"
    elif "energy" in name_lower or "capacity" in name_lower:
        category = "energy"
    else:
        category = "other"

    # Extract level (look for patterns like "Level1", "Level2", etc.)
    level_match = re.search(r'level(\d)', name_lower)
    if level_match:
        level = int(level_match.group(1))

    return category, level


class UpgradeExtractor:
    """
    Extracts upgrade data from SC2 observations.

    This class tracks upgrades across frames, detects newly completed upgrades,
    and provides upgrade information with categorization for ground truth data.

    Lifecycle tracking:
        Each upgrade includes a 'status' field that can be 'completed', 'started',
        or 'cancelled'. Currently, only 'completed' status is populated from
        player.upgrade_ids. See the note in extract() for details on why start/cancel
        detection is deferred.

    Example usage:
        extractor = UpgradeExtractor(player_id=1)

        for obs in game_loop_iterator:
            upgrades_data = extractor.extract(obs)

            # Check for new upgrades
            new_upgrades = extractor.get_new_upgrades()
            if new_upgrades:
                print(f"New upgrades completed: {new_upgrades}")

            # Get all upgrades ever seen this game
            all_seen = extractor.get_all_discovered_upgrades()
            print(f"All discovered upgrades: {all_seen}")

            # Get upgrade summary
            summary = extractor.get_upgrade_summary(upgrades_data)
            print(f"Upgrades: {summary}")
    """

    def __init__(self, player_id: int):
        """
        Initialize the UpgradeExtractor.

        Args:
            player_id: Player ID this extractor is tracking (1 or 2)
        """
        self.player_id = player_id

        # Track upgrades from previous frame (set of upgrade_ids)
        self.previous_upgrades: Set[int] = set()

        # Track all completed upgrades with completion timestamps
        # Maps upgrade_id -> game_loop when the upgrade first appeared in player.upgrade_ids
        self.upgrade_completion_times: Dict[int, int] = {}

        # Track newly completed upgrades in the current frame
        self.newly_completed: Set[int] = set()

        # Track in-progress research ability_ids seen on buildings in the previous frame.
        # Maps ability_id -> game_loop when first observed on a building's order list.
        # Used for future start/cancel detection once ability-to-upgrade mapping is resolved.
        self._previous_research_ability_ids: Set[int] = set()

        # Track the game_loop when each research ability_id was first seen on a building.
        # Maps ability_id -> game_loop of first sighting.
        self._research_start_times: Dict[int, int] = {}

        # Registry of ALL upgrade names ever discovered during this game.
        # This includes completed upgrades. If start/cancel tracking were active,
        # it would also include started and cancelled upgrades.
        # Stores lowercase upgrade name strings for consistent lookups.
        self._all_discovered_upgrades: Set[str] = set()

    def extract(self, obs) -> Dict[str, Dict]:
        """
        Extract all upgrade data from observation.

        Scans player.upgrade_ids for completed upgrades and returns lifecycle
        info for each upgrade. Each entry includes a 'status' key.

        NOTE ON START/CANCEL DETECTION (deferred):
            Detecting upgrade START and CANCEL events is technically feasible by
            scanning raw_data.units for buildings owned by this player that have
            research orders (unit.orders[i].ability_id). When an ability_id appears
            on a building's order list, research has started; when it disappears
            without the corresponding upgrade_id appearing in player.upgrade_ids,
            the research was cancelled.

            However, this requires a reliable ability_id -> upgrade_id mapping.
            SC2's ability IDs for research commands do NOT match upgrade IDs, and
            pysc2 does not expose a clean mapping between them. Building such a
            mapping would require either:
              (a) A hardcoded lookup table of ~90+ ability_id -> upgrade_id pairs
                  that varies by game patch, or
              (b) Fuzzy name matching between ability names and upgrade names, which
                  is fragile and error-prone.

            Until a robust mapping is available, status is always 'completed' for
            upgrades returned here. The infrastructure for tracking building orders
            (_previous_research_ability_ids, _research_start_times) is in place
            so that start/cancel detection can be added without restructuring.

        Args:
            obs: SC2 observation from controller.observe()

        Returns:
            Dictionary mapping lowercase upgrade names to upgrade data:
            {
                'terraninfantryweaponslevel1': {
                    'upgrade_id': 7,
                    'upgrade_name': 'TerranInfantryWeaponsLevel1',
                    'category': 'weapons',
                    'level': 1,
                    'status': 'completed',
                    'completed_loop': 5000,
                    'started_loop': None,
                },
                'stimpack': {
                    'upgrade_id': 15,
                    'upgrade_name': 'Stimpack',
                    'category': 'other',
                    'level': 0,
                    'status': 'completed',
                    'completed_loop': 3500,
                    'started_loop': None,
                },
                ...
            }
        """
        try:
            raw_data = obs.observation.raw_data
            game_loop = obs.observation.game_loop

            # Get current upgrades from raw player data
            current_upgrades = set(raw_data.player.upgrade_ids)

            # Detect newly completed upgrades (present now but not in previous frame)
            self.newly_completed = current_upgrades - self.previous_upgrades

            # Record completion times for new upgrades
            for upgrade_id in self.newly_completed:
                if upgrade_id not in self.upgrade_completion_times:
                    self.upgrade_completion_times[upgrade_id] = game_loop
                    upgrade_name = get_upgrade_name(upgrade_id)
                    logger.info(
                        f"Player {self.player_id}: Upgrade {upgrade_name} "
                        f"completed at loop {game_loop}"
                    )

            # Build upgrades data dictionary with lifecycle status
            upgrades_data = {}

            for upgrade_id in current_upgrades:
                upgrade_name = get_upgrade_name(upgrade_id)
                category, level = parse_upgrade_details(upgrade_name)

                # Use lowercase name as key for consistency
                key = upgrade_name.lower()

                # Register this upgrade in the all-discovered set
                self._all_discovered_upgrades.add(key)

                upgrade_data = {
                    'upgrade_id': upgrade_id,
                    'upgrade_name': upgrade_name,
                    'category': category,
                    'level': level,
                    'status': 'completed',
                    # completed_loop: the game_loop when this upgrade first appeared
                    # in player.upgrade_ids
                    'completed_loop': self.upgrade_completion_times.get(upgrade_id, None),
                    # started_loop: would be the game_loop when research began on
                    # a building. Currently None because start detection is deferred
                    # (see docstring note on ability_id -> upgrade_id mapping).
                    'started_loop': None,
                }

                upgrades_data[key] = upgrade_data

            # Update previous upgrades for next iteration
            self.previous_upgrades = current_upgrades

            return upgrades_data

        except Exception as e:
            logger.error(f"Error extracting upgrade data: {e}")
            # Return empty dictionary on error
            return {}

    def get_new_upgrades(self) -> Set[int]:
        """
        Get upgrade IDs that were newly completed in the last extract() call.

        Returns:
            Set of upgrade IDs that were just completed
        """
        return self.newly_completed

    def get_all_discovered_upgrades(self) -> List[str]:
        """
        Get all upgrade names that have been seen at any point during this game.

        This returns every upgrade that has appeared in the extract() output across
        all frames processed so far. Currently this is equivalent to all completed
        upgrades, but if start/cancel tracking is added in the future, it would
        also include upgrades that were started but later cancelled.

        Returns:
            Sorted list of lowercase upgrade name strings seen during this game.
            Sorted for deterministic output ordering.

        Example:
            >>> extractor.get_all_discovered_upgrades()
            ['combatshield', 'stimpack', 'terraninfantryweaponslevel1']
        """
        return sorted(self._all_discovered_upgrades)

    def get_upgrade_summary(self, upgrades_data: Dict[str, Dict]) -> str:
        """
        Get a human-readable summary of upgrades.

        Args:
            upgrades_data: Output from extract()

        Returns:
            Formatted string summary

        Example:
            "3 upgrades (Weapons: 1, Armor: 1, Other: 1)"
        """
        if not upgrades_data:
            return "No upgrades completed"

        # Count by category
        category_counts: Dict[str, int] = {}
        for upgrade_data in upgrades_data.values():
            category = upgrade_data['category']
            category_counts[category] = category_counts.get(category, 0) + 1

        total = len(upgrades_data)
        categories_str = ", ".join([f"{cat.capitalize()}: {count}"
                                   for cat, count in sorted(category_counts.items())])

        return f"{total} upgrades ({categories_str})"

    def get_upgrades_by_category(self, upgrades_data: Dict[str, Dict]) -> Dict[str, list]:
        """
        Get upgrades grouped by category.

        Args:
            upgrades_data: Output from extract()

        Returns:
            Dictionary mapping categories to lists of upgrade names:
            {
                'weapons': ['TerranInfantryWeaponsLevel1', 'TerranInfantryWeaponsLevel2'],
                'armor': ['TerranInfantryArmorLevel1'],
                'other': ['Stimpack', 'CombatShield']
            }
        """
        by_category: Dict[str, list] = {}

        for upgrade_data in upgrades_data.values():
            category = upgrade_data['category']
            upgrade_name = upgrade_data['upgrade_name']

            if category not in by_category:
                by_category[category] = []

            by_category[category].append(upgrade_name)

        return by_category

    def get_upgrade_count(self, upgrades_data: Dict[str, Dict]) -> int:
        """
        Get total number of completed upgrades.

        Args:
            upgrades_data: Output from extract()

        Returns:
            Number of completed upgrades
        """
        return len(upgrades_data)

    def has_upgrade(self, upgrades_data: Dict[str, Dict], upgrade_name: str) -> bool:
        """
        Check if a specific upgrade has been completed.

        Args:
            upgrades_data: Output from extract()
            upgrade_name: Name of upgrade to check (case-insensitive)

        Returns:
            True if upgrade is completed, False otherwise

        Example:
            >>> has_upgrade(upgrades_data, "Stimpack")
            True
        """
        return upgrade_name.lower() in upgrades_data

    def reset(self):
        """
        Reset all tracking state.

        Clears completed upgrade history, newly-completed set, research tracking,
        and the all-discovered registry. Call this between games when reusing
        the same extractor instance.
        """
        self.previous_upgrades.clear()
        self.upgrade_completion_times.clear()
        self.newly_completed.clear()
        self._previous_research_ability_ids.clear()
        self._research_start_times.clear()
        self._all_discovered_upgrades.clear()

"""
EconomyExtractor: Extracts economy data from SC2 replay files using s2protocol.

This module replaces the previous engine-based EconomyExtractor class. In observer
mode (observed_player_id=0), the SC2 engine's player_common and score_details are
always empty (all zeros). Instead, this module parses the replay file directly using
s2protocol to extract SPlayerStatsEvent tracker events, which contain per-player
economy snapshots at ~160 game-loop intervals regardless of observation mode.

Provides two public functions:
  - load_economy_snapshots(replay_path) -> dict
      Parses the replay file once and returns all economy snapshots grouped by player.
  - get_economy_at_loop(snapshots, player_id, game_loop) -> dict
      Returns the most recent economy snapshot for a player at or before a given game loop.
      Uses binary search for efficient lookup across potentially thousands of snapshots.

The pipeline (extraction_pipeline.py) calls load_economy_snapshots() once before the
game loop begins, then calls get_economy_at_loop() each frame to forward-fill economy
values between the ~160-loop update intervals.

Dependencies:
  - mpyq: reads the MPQ archive format of SC2 replay files
  - s2protocol: Blizzard's official SC2 replay protocol decoder
"""

from typing import Dict, List
from bisect import bisect_right
import logging

import mpyq
from s2protocol import versions


logger = logging.getLogger(__name__)


# The set of fields we extract from each SPlayerStatsEvent, mapped to their
# output key names and any transformation needed. SPlayerStatsEvent stores
# supply values as fixed-point integers (multiply by 4096), so we divide
# those back to get the real float value.
#
# Source field name -> (output key, divisor)
# A divisor of 1 means "use raw integer value as-is".
# A divisor of 4096 means "fixed-point conversion: value / 4096".
_FIELD_MAP = {
    'm_scoreValueMineralsCurrent':        ('minerals',                   1),
    'm_scoreValueVespeneCurrent':         ('vespene',                    1),
    'm_scoreValueFoodUsed':               ('supply_used',             4096),
    'm_scoreValueFoodMade':               ('supply_cap',              4096),
    'm_scoreValueMineralsCollectionRate': ('collection_rate_minerals',   1),
    'm_scoreValueVespeneCollectionRate':  ('collection_rate_vespene',    1),
}

# Keys used in the output dicts, in a canonical order. Used by
# _make_zeroed_snapshot() to build a zero-valued default dict.
_ECONOMY_KEYS = [entry[0] for entry in _FIELD_MAP.values()]


def _make_zeroed_snapshot() -> Dict[str, float]:
    """
    Build a snapshot dict with all economy keys set to zero.

    Used as the default return value when no SPlayerStatsEvent has been
    emitted yet for a player at or before the requested game loop.

    Returns:
        Dict with keys matching the economy schema, all values 0.

    Called by:
        get_economy_at_loop() -- when no snapshot exists yet for a player.
    """
    return {key: 0 for key in _ECONOMY_KEYS}


def load_economy_snapshots(replay_path: str) -> Dict[int, List[Dict[str, float]]]:
    """
    Parse a replay file and extract all SPlayerStatsEvent economy snapshots.

    Opens the replay as an MPQ archive, decodes the tracker events using the
    protocol version embedded in the replay header, and collects economy fields
    from every SPlayerStatsEvent. Returns snapshots grouped by player_id (1-indexed)
    and sorted by game_loop ascending within each player.

    Args:
        replay_path: Absolute or relative path to the .SC2Replay file.

    Returns:
        Dict mapping player_id (int, 1-indexed) to a list of snapshot dicts,
        each containing:
            {
                'game_loop': int,
                'minerals': int,
                'vespene': int,
                'supply_used': float,
                'supply_cap': float,
                'collection_rate_minerals': float,
                'collection_rate_vespene': float,
            }
        Lists are sorted by game_loop ascending. Players with no events get
        an empty list.

    Raises:
        FileNotFoundError: If replay_path does not exist.
        KeyError: If the replay header is missing expected version fields.

    Depends on / calls:
        - mpyq.MPQArchive: opens the replay MPQ container
        - s2protocol.versions.latest().decode_replay_header(): reads version info
        - s2protocol.versions.build(base_build).decode_replay_tracker_events():
          iterates tracker events looking for SPlayerStatsEvent
    """
    logger.info(f"Loading economy snapshots from: {replay_path}")

    # Open the replay file as an MPQ archive
    archive = mpyq.MPQArchive(replay_path)

    # Decode the replay header to determine the protocol version.
    # The header is stored in the MPQ's user_data_header and contains the
    # SC2 build number needed to select the correct protocol decoder.
    header_content = archive.header['user_data_header']['content']
    header = versions.latest().decode_replay_header(header_content)
    base_build = header['m_version']['m_baseBuild']

    logger.debug(f"Replay base build: {base_build}")

    # Load the protocol matching this replay's build version
    protocol = versions.build(base_build)

    # Read and decode tracker events from the replay archive.
    # Tracker events include unit births/deaths, upgrades, and player stats.
    tracker_raw = archive.read_file('replay.tracker.events')

    # Accumulate snapshots per player.
    # Keys are player_id (1-indexed int), values are lists of snapshot dicts.
    snapshots: Dict[int, List[Dict[str, float]]] = {}

    event_count = 0
    for event in protocol.decode_replay_tracker_events(tracker_raw):
        if event['_event'] != 'NNet.Replay.Tracker.SPlayerStatsEvent':
            continue

        event_count += 1

        # SPlayerStatsEvent uses 1-indexed m_playerId.
        # m_playerId=1 is player 1, m_playerId=2 is player 2.
        player_id = event['m_playerId']
        game_loop = event['_gameloop']

        # Extract the stats sub-object that holds the score values.
        # All m_scoreValue* fields live inside event['m_stats'].
        stats = event['m_stats']

        # Build the snapshot dict by extracting each mapped field.
        # Apply the divisor for fixed-point fields (supply_used, supply_cap).
        snapshot = {'game_loop': game_loop}
        for src_field, (out_key, divisor) in _FIELD_MAP.items():
            raw_value = stats.get(src_field, 0)
            snapshot[out_key] = raw_value / divisor if divisor != 1 else raw_value

        # Append to this player's snapshot list
        if player_id not in snapshots:
            snapshots[player_id] = []
        snapshots[player_id].append(snapshot)

    # Sort each player's snapshots by game_loop (should already be in order,
    # but sort defensively in case the tracker events are not strictly ordered).
    for pid in snapshots:
        snapshots[pid].sort(key=lambda s: s['game_loop'])

    logger.info(
        f"Loaded {event_count} SPlayerStatsEvent(s) for "
        f"{len(snapshots)} player(s) from replay"
    )

    return snapshots


def get_economy_at_loop(
    snapshots: Dict[int, List[Dict[str, float]]],
    player_id: int,
    game_loop: int,
) -> Dict[str, float]:
    """
    Return the most recent economy snapshot for a player at or before a given game loop.

    Uses binary search (bisect_right) on the pre-sorted snapshot list to find
    the latest snapshot whose game_loop <= the requested game_loop. This gives
    O(log n) lookup per call, which is efficient even for long replays with
    thousands of snapshots.

    If the player has no snapshots at all, or the earliest snapshot is after
    the requested game_loop, returns a zeroed dict (all economy values = 0).

    Args:
        snapshots: The dict returned by load_economy_snapshots().
                   Maps player_id -> sorted list of snapshot dicts.
        player_id: Which player to look up (1-indexed, typically 1 or 2).
        game_loop: The game loop to query. Returns the most recent snapshot
                   at or before this loop.

    Returns:
        Dict with the same keys as a snapshot from load_economy_snapshots():
            {
                'game_loop': int,       # (absent if returning zeroed default)
                'minerals': int,
                'vespene': int,
                'supply_used': float,
                'supply_cap': float,
                'collection_rate_minerals': float,
                'collection_rate_vespene': float,
            }
        If no snapshot exists yet, returns a zeroed dict (no 'game_loop' key).

    Depends on / calls:
        _make_zeroed_snapshot() -- for the fallback when no data is available.
        bisect_right (from bisect module) -- for binary search on sorted list.
    """
    player_snaps = snapshots.get(player_id, [])

    if not player_snaps:
        # No snapshots at all for this player — return zeros
        return _make_zeroed_snapshot()

    # Binary search: find the rightmost snapshot with game_loop <= requested game_loop.
    # We build a list of game_loops for bisect_right. bisect_right returns the
    # insertion point AFTER any existing entries equal to game_loop, so index - 1
    # gives us the last snapshot at or before the target.
    game_loops = [s['game_loop'] for s in player_snaps]
    idx = bisect_right(game_loops, game_loop)

    if idx == 0:
        # All snapshots are after the requested game_loop — return zeros
        return _make_zeroed_snapshot()

    # Return a copy to prevent callers from mutating the cached snapshot
    return dict(player_snaps[idx - 1])

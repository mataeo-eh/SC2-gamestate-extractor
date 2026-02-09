"""
Feature Engineering Script: Army-Related Features for SC2 Replay Data

Processes parquet files that have ALREADY been through the create_unit_counts.py
pipeline (so *_count and *_total_unit_types columns exist) and engineers four
army-related features computed at 10-gameloop intervals:

1. Main Army Movement Direction — aggressive / defensive / neutral
2. Main Army Size — number of units in the largest spatial cluster
3. Number of Armies — distinct army groups (DBSCAN clusters with >= 3 units)
4. Army Complexity Over Time — unique army unit types / gameloop

All features are computed per player at every 10th gameloop and forward-filled
for intermediate rows.
"""

import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

# Ensure the project root is on sys.path so src_new can be imported
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.cluster import DBSCAN

from src_new.pipeline.logging_config import setup_logging

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Entity column pattern: p{player}_p{player}_{type}_{id}_{attribute}
ENTITY_COL_RE = re.compile(r"^(p[12])_p[12]_(.+?)_(\d+)_(.+)$")

# Known building types (structures that don't move) - matches create_unit_counts.py
BUILDING_TYPES = {
    # Terran
    "commandcenter", "commandcenterflying", "orbitalcommand", "planetaryfortress",
    "supplydepot", "supplydepotlowered", "barracks", "barrackstechlab",
    "barracksreactor", "factory", "factorytechlab", "factoryreactor",
    "starport", "starporttechlab", "starportreactor", "engineeringbay",
    "armory", "ghostacademy", "fusioncore", "bunker", "missileturret",
    "sensortower", "refinery",
    # Protoss
    "nexus", "pylon", "gateway", "forge", "cyberneticscore",
    "assimilator", "twilightcouncil", "templararchive", "darkshrine",
    "roboticsfacility", "roboticsbay", "stargate", "fleetbeacon",
    "photoncannon", "shieldbattery",
    # Zerg
    "hatchery", "lair", "hive", "spawningpool", "evolutionchamber",
    "extractor", "roachwarren", "banelingnest", "hydraliskden",
    "lurkerden", "infestationpit", "spire", "greaterspire",
    "ultraliskcavern", "nydusnetwork", "nyduscanal",
    "spinecrawler", "sporecrawler",
}

# Worker types (economic units, excluded from army clustering)
WORKER_TYPES = {"scv", "probe", "drone", "mule"}

# Base structure types (used to find starting positions)
BASE_TYPES = {
    "commandcenter", "nexus", "hatchery", "lair", "hive",
    "commandcenterflying", "orbitalcommand", "planetaryfortress",
}

# Non-army types = buildings + workers (excluded from army clustering)
NON_ARMY_TYPES = BUILDING_TYPES | WORKER_TYPES

# Alive states for units with a _state column
ALIVE_STATES = {"built", "existing"}

# DBSCAN parameters for army clustering
DBSCAN_EPS = 10.0          # Base spatial eps in map units
DBSCAN_MIN_SAMPLES = 3     # Minimum units to form an army cluster
MERGE_DISTANCE_FACTOR = 1.5  # Merge clusters whose centroids are within this * combined_radius


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def parse_entity_columns(columns):
    """Parse all entity columns and group by (player, entity_type, entity_id).

    Returns:
        dict: {(player, entity_type, entity_id): set of attribute names}
    """
    entities = defaultdict(set)
    for col in columns:
        m = ENTITY_COL_RE.match(col)
        if m:
            player, entity_type, entity_id, attribute = m.groups()
            entities[(player, entity_type, entity_id)].add(attribute)
    return dict(entities)


def group_entities_by_player_type(entities):
    """Group entity instances by (player, entity_type).

    Args:
        entities: dict from parse_entity_columns

    Returns:
        dict: {(player, entity_type): list of (entity_id, set of attributes)}
    """
    groups = defaultdict(list)
    for (player, entity_type, entity_id), attrs in entities.items():
        groups[(player, entity_type)].append((entity_id, attrs))
    return dict(groups)


def find_base_positions(df, entities):
    """Find starting base position for each player.

    Strategy:
    1. Look for the first base building (commandcenter/nexus/hatchery, id=001)
    2. Fallback: use worker_001 position (scv/probe/drone) at earliest non-NaN row

    Args:
        df: The full dataframe
        entities: dict from parse_entity_columns

    Returns:
        dict: {player: {"x": float, "y": float}} for each player found
    """
    bases = {}

    # Pass 1: base buildings with id=001
    for (player, etype, eid), attrs in entities.items():
        if etype in BASE_TYPES and eid == "001" and "x" in attrs and "y" in attrs:
            x_col = f"{player}_{player}_{etype}_{eid}_x"
            y_col = f"{player}_{player}_{etype}_{eid}_y"
            if x_col in df.columns and y_col in df.columns:
                x_vals = df[x_col].dropna()
                y_vals = df[y_col].dropna()
                if len(x_vals) > 0 and len(y_vals) > 0:
                    bases[player] = {
                        "x": float(x_vals.iloc[0]),
                        "y": float(y_vals.iloc[0]),
                    }

    # Pass 2: fallback to worker_001 for any player still missing
    for player in ["p1", "p2"]:
        if player in bases:
            continue
        for (p, etype, eid), attrs in entities.items():
            if (p == player and etype in WORKER_TYPES and eid == "001"
                    and "x" in attrs and "y" in attrs):
                x_col = f"{player}_{player}_{etype}_{eid}_x"
                y_col = f"{player}_{player}_{etype}_{eid}_y"
                if x_col in df.columns and y_col in df.columns:
                    x_vals = df[x_col].dropna()
                    y_vals = df[y_col].dropna()
                    if len(x_vals) > 0 and len(y_vals) > 0:
                        bases[player] = {
                            "x": float(x_vals.iloc[0]),
                            "y": float(y_vals.iloc[0]),
                        }
                        break

    return bases


def is_entity_alive(df, row_idx, player, entity_type, entity_id, attrs):
    """Check if a specific entity is alive at a given row index.

    Uses _state column if available (alive = 'built' or 'existing').
    Falls back to completed_loop / destroyed_loop lifecycle columns.

    Args:
        df: The full dataframe
        row_idx: Integer index into the dataframe
        player: 'p1' or 'p2'
        entity_type: e.g. 'marine'
        entity_id: e.g. '001'
        attrs: set of attribute names for this entity

    Returns:
        bool: True if the entity is alive at this row
    """
    col_prefix = f"{player}_{player}_{entity_type}_{entity_id}"

    if "state" in attrs:
        state_col = f"{col_prefix}_state"
        if state_col in df.columns:
            val = df.iat[row_idx, df.columns.get_loc(state_col)]
            return val in ALIVE_STATES

    if "completed_loop" in attrs and "destroyed_loop" in attrs:
        completed_col = f"{col_prefix}_completed_loop"
        destroyed_col = f"{col_prefix}_destroyed_loop"
        if completed_col in df.columns and destroyed_col in df.columns:
            completed = df.iat[row_idx, df.columns.get_loc(completed_col)]
            destroyed = df.iat[row_idx, df.columns.get_loc(destroyed_col)]
            return pd.notna(completed) and pd.isna(destroyed)

    if "completed_loop" in attrs:
        completed_col = f"{col_prefix}_completed_loop"
        if completed_col in df.columns:
            completed = df.iat[row_idx, df.columns.get_loc(completed_col)]
            return pd.notna(completed)

    return False


def get_entity_position(df, row_idx, player, entity_type, entity_id, attrs):
    """Get the (x, y) position of an entity at a given row index.

    Args:
        df: The full dataframe
        row_idx: Integer index into the dataframe
        player: 'p1' or 'p2'
        entity_type: e.g. 'marine'
        entity_id: e.g. '001'
        attrs: set of attribute names for this entity

    Returns:
        tuple (x, y) or None if position is not available
    """
    if "x" not in attrs or "y" not in attrs:
        return None

    col_prefix = f"{player}_{player}_{entity_type}_{entity_id}"
    x_col = f"{col_prefix}_x"
    y_col = f"{col_prefix}_y"

    if x_col not in df.columns or y_col not in df.columns:
        return None

    x = df.iat[row_idx, df.columns.get_loc(x_col)]
    y = df.iat[row_idx, df.columns.get_loc(y_col)]

    if pd.isna(x) or pd.isna(y):
        return None

    return (float(x), float(y))


def collect_army_positions(df, row_idx, player, entities):
    """Collect positions of all alive army units for a player at a given row.

    Army units = all non-building, non-worker entity types that are alive.

    Args:
        df: The full dataframe
        row_idx: Integer index into the dataframe
        player: 'p1' or 'p2'
        entities: dict from parse_entity_columns

    Returns:
        positions: np.ndarray of shape (N, 2) with x, y coordinates
        types: list of entity_type strings (parallel to positions)
    """
    positions = []
    types = []

    for (p, etype, eid), attrs in entities.items():
        if p != player:
            continue
        if etype in NON_ARMY_TYPES:
            continue

        if not is_entity_alive(df, row_idx, player, etype, eid, attrs):
            continue

        pos = get_entity_position(df, row_idx, player, etype, eid, attrs)
        if pos is not None:
            positions.append(pos)
            types.append(etype)

    if len(positions) == 0:
        return np.empty((0, 2)), []

    return np.array(positions, dtype=np.float64), types


def cluster_army_units(positions, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES):
    """Cluster army unit positions using DBSCAN with post-hoc merging.

    Uses a two-pass approach:
    1. DBSCAN with moderate eps and min_samples=3
    2. Merge clusters whose centroids are close relative to their combined size

    Args:
        positions: np.ndarray of shape (N, 2) with x, y coordinates
        eps: DBSCAN eps parameter
        min_samples: DBSCAN min_samples parameter

    Returns:
        labels: np.ndarray of cluster labels (-1 = noise), with merged clusters
        n_clusters: number of valid clusters (size >= min_samples)
    """
    if len(positions) < min_samples:
        return np.full(len(positions), -1, dtype=int), 0

    # Pass 1: initial DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(positions)

    # Get unique valid cluster labels (excluding noise = -1)
    unique_labels = set(labels)
    unique_labels.discard(-1)

    if len(unique_labels) <= 1:
        n_clusters = len(unique_labels)
        return labels, n_clusters

    # Pass 2: merge nearby clusters based on their size and proximity
    # Compute centroids and sizes for each cluster
    cluster_info = {}
    for label in unique_labels:
        mask = labels == label
        cluster_positions = positions[mask]
        centroid = cluster_positions.mean(axis=0)
        size = mask.sum()
        # Approximate radius: max distance from centroid to any member
        dists = np.sqrt(((cluster_positions - centroid) ** 2).sum(axis=1))
        radius = dists.max() if len(dists) > 0 else 0.0
        cluster_info[label] = {
            "centroid": centroid,
            "size": size,
            "radius": radius,
        }

    # Build merge map: for each pair of clusters, check if they should merge
    sorted_labels = sorted(unique_labels)
    merge_map = {l: l for l in sorted_labels}  # maps each label to its canonical label

    for i in range(len(sorted_labels)):
        for j in range(i + 1, len(sorted_labels)):
            li = sorted_labels[i]
            lj = sorted_labels[j]

            # Resolve to canonical labels
            ci = merge_map[li]
            while merge_map[ci] != ci:
                ci = merge_map[ci]
            cj = merge_map[lj]
            while merge_map[cj] != cj:
                cj = merge_map[cj]

            if ci == cj:
                continue  # Already merged

            info_i = cluster_info[li]
            info_j = cluster_info[lj]

            # Distance between centroids
            centroid_dist = np.sqrt(((info_i["centroid"] - info_j["centroid"]) ** 2).sum())

            # Combined radius: sum of both radii
            combined_radius = info_i["radius"] + info_j["radius"]

            # Merge if centroids are within MERGE_DISTANCE_FACTOR * combined_radius
            # Also use a minimum merge distance based on eps to handle small clusters
            merge_threshold = max(
                MERGE_DISTANCE_FACTOR * combined_radius,
                eps * 2.0,  # minimum merge distance
            )

            if centroid_dist <= merge_threshold:
                # Merge lj into li's canonical cluster
                merge_map[cj] = ci

    # Resolve all merge chains
    for label in sorted_labels:
        canonical = label
        while merge_map[canonical] != canonical:
            canonical = merge_map[canonical]
        merge_map[label] = canonical

    # Apply merge map to labels
    merged_labels = labels.copy()
    for old_label in sorted_labels:
        new_label = merge_map[old_label]
        if new_label != old_label:
            merged_labels[labels == old_label] = new_label

    # Recount valid clusters after merging
    final_unique = set(merged_labels)
    final_unique.discard(-1)

    # Re-validate cluster sizes after merging (must still have >= min_samples)
    valid_count = 0
    for label in final_unique:
        if (merged_labels == label).sum() >= min_samples:
            valid_count += 1
        else:
            # Mark undersized merged clusters as noise
            merged_labels[merged_labels == label] = -1

    return merged_labels, valid_count


def find_largest_cluster(positions, labels):
    """Find the centroid and size of the largest cluster.

    Args:
        positions: np.ndarray of shape (N, 2)
        labels: np.ndarray of cluster labels

    Returns:
        centroid: np.ndarray of shape (2,) or None if no clusters
        size: int, number of units in the largest cluster
    """
    unique_labels = set(labels)
    unique_labels.discard(-1)

    if not unique_labels:
        return None, 0

    best_label = None
    best_size = 0

    for label in unique_labels:
        mask = labels == label
        size = mask.sum()
        if size > best_size:
            best_size = size
            best_label = label

    if best_label is None:
        return None, 0

    mask = labels == best_label
    centroid = positions[mask].mean(axis=0)
    return centroid, int(best_size)


def determine_movement_direction(current_centroid, previous_centroid, own_base, opponent_base):
    """Determine if the main army moved aggressively or defensively.

    Args:
        current_centroid: np.ndarray (2,) - current main army centroid
        previous_centroid: np.ndarray (2,) - previous main army centroid
        own_base: dict with 'x', 'y' keys
        opponent_base: dict with 'x', 'y' keys

    Returns:
        str: "aggressive", "defensive", or "neutral"
    """
    if current_centroid is None or previous_centroid is None:
        return "neutral"
    if own_base is None or opponent_base is None:
        return "neutral"

    opponent_pos = np.array([opponent_base["x"], opponent_base["y"]])
    own_pos = np.array([own_base["x"], own_base["y"]])

    # Distance from current centroid to opponent base vs previous centroid to opponent base
    dist_to_opponent_current = np.sqrt(((current_centroid - opponent_pos) ** 2).sum())
    dist_to_opponent_previous = np.sqrt(((previous_centroid - opponent_pos) ** 2).sum())

    # Distance from current centroid to own base vs previous centroid to own base
    dist_to_own_current = np.sqrt(((current_centroid - own_pos) ** 2).sum())
    dist_to_own_previous = np.sqrt(((previous_centroid - own_pos) ** 2).sum())

    # Movement magnitude (to detect negligible movement)
    movement = np.sqrt(((current_centroid - previous_centroid) ** 2).sum())

    # Negligible movement threshold (less than 1 map unit)
    if movement < 1.0:
        return "neutral"

    # Net movement toward opponent vs toward own base
    moved_toward_opponent = dist_to_opponent_previous - dist_to_opponent_current
    moved_toward_own = dist_to_own_previous - dist_to_own_current

    if moved_toward_opponent > moved_toward_own:
        return "aggressive"
    elif moved_toward_own > moved_toward_opponent:
        return "defensive"
    else:
        return "neutral"


def count_alive_army_types(df, row_idx, player, entities):
    """Count the number of distinct alive army unit types for a player at a row.

    Army unit types exclude buildings and workers.

    Args:
        df: The full dataframe
        row_idx: Integer index
        player: 'p1' or 'p2'
        entities: dict from parse_entity_columns

    Returns:
        int: number of distinct army unit types with at least one alive instance
    """
    alive_types = set()

    for (p, etype, eid), attrs in entities.items():
        if p != player:
            continue
        if etype in NON_ARMY_TYPES:
            continue
        if etype in alive_types:
            continue  # Already found this type alive, skip further checks

        if is_entity_alive(df, row_idx, player, etype, eid, attrs):
            alive_types.add(etype)

    return len(alive_types)


def precompute_alive_masks(df, entities):
    """Precompute per-entity alive boolean arrays for all rows.

    This avoids repeated per-row lookups and dramatically speeds up processing.

    Args:
        df: The full dataframe
        entities: dict from parse_entity_columns

    Returns:
        dict: {(player, entity_type, entity_id): np.ndarray of bool}
    """
    n_rows = len(df)
    alive_masks = {}

    for (player, etype, eid), attrs in entities.items():
        col_prefix = f"{player}_{player}_{etype}_{eid}"
        mask = np.zeros(n_rows, dtype=np.bool_)

        if "state" in attrs:
            state_col = f"{col_prefix}_state"
            if state_col in df.columns:
                mask = df[state_col].isin(ALIVE_STATES).to_numpy(dtype=np.bool_)
                alive_masks[(player, etype, eid)] = mask
                continue

        if "completed_loop" in attrs and "destroyed_loop" in attrs:
            completed_col = f"{col_prefix}_completed_loop"
            destroyed_col = f"{col_prefix}_destroyed_loop"
            if completed_col in df.columns and destroyed_col in df.columns:
                completed_notna = df[completed_col].notna().to_numpy(dtype=np.bool_)
                destroyed_isna = df[destroyed_col].isna().to_numpy(dtype=np.bool_)
                mask = completed_notna & destroyed_isna
                alive_masks[(player, etype, eid)] = mask
                continue

        if "completed_loop" in attrs:
            completed_col = f"{col_prefix}_completed_loop"
            if completed_col in df.columns:
                mask = df[completed_col].notna().to_numpy(dtype=np.bool_)
                alive_masks[(player, etype, eid)] = mask
                continue

        alive_masks[(player, etype, eid)] = mask

    return alive_masks


def precompute_position_arrays(df, entities):
    """Precompute x, y position arrays for all entities.

    Args:
        df: The full dataframe
        entities: dict from parse_entity_columns

    Returns:
        dict: {(player, entity_type, entity_id): {"x": np.ndarray, "y": np.ndarray}}
              Only includes entities that have both x and y columns.
    """
    pos_arrays = {}

    for (player, etype, eid), attrs in entities.items():
        if "x" not in attrs or "y" not in attrs:
            continue

        col_prefix = f"{player}_{player}_{etype}_{eid}"
        x_col = f"{col_prefix}_x"
        y_col = f"{col_prefix}_y"

        if x_col in df.columns and y_col in df.columns:
            pos_arrays[(player, etype, eid)] = {
                "x": df[x_col].to_numpy(dtype=np.float64),
                "y": df[y_col].to_numpy(dtype=np.float64),
            }

    return pos_arrays


def collect_army_positions_fast(row_idx, player, entities, alive_masks, pos_arrays):
    """Collect positions of all alive army units for a player at a given row.

    Uses precomputed alive masks and position arrays for speed.

    Args:
        row_idx: Integer index into the dataframe
        player: 'p1' or 'p2'
        entities: dict from parse_entity_columns
        alive_masks: dict from precompute_alive_masks
        pos_arrays: dict from precompute_position_arrays

    Returns:
        positions: np.ndarray of shape (N, 2) with x, y coordinates
        types: list of entity_type strings (parallel to positions)
    """
    positions = []
    types = []

    for (p, etype, eid) in entities:
        if p != player:
            continue
        if etype in NON_ARMY_TYPES:
            continue

        # Check alive
        key = (p, etype, eid)
        if key not in alive_masks:
            continue
        if not alive_masks[key][row_idx]:
            continue

        # Get position
        if key not in pos_arrays:
            continue
        x = pos_arrays[key]["x"][row_idx]
        y = pos_arrays[key]["y"][row_idx]
        if np.isnan(x) or np.isnan(y):
            continue

        positions.append((x, y))
        types.append(etype)

    if len(positions) == 0:
        return np.empty((0, 2)), []

    return np.array(positions, dtype=np.float64), types


def count_alive_army_types_fast(row_idx, player, entities, alive_masks):
    """Count distinct alive army unit types using precomputed alive masks.

    Args:
        row_idx: Integer index
        player: 'p1' or 'p2'
        entities: dict from parse_entity_columns
        alive_masks: dict from precompute_alive_masks

    Returns:
        int: number of distinct army unit types with at least one alive instance
    """
    alive_types = set()

    for (p, etype, eid) in entities:
        if p != player:
            continue
        if etype in NON_ARMY_TYPES:
            continue
        if etype in alive_types:
            continue

        key = (p, etype, eid)
        if key not in alive_masks:
            continue
        if alive_masks[key][row_idx]:
            alive_types.add(etype)

    return len(alive_types)


# ---------------------------------------------------------------------------
# Core Feature Computation
# ---------------------------------------------------------------------------

def compute_army_features(df, entities, bases):
    """Compute all four army features for a single dataframe.

    Args:
        df: The full dataframe (with game_loop column)
        entities: dict from parse_entity_columns
        bases: dict from find_base_positions

    Returns:
        df: The dataframe with 8 new columns added
    """
    n_rows = len(df)
    game_loops = df["game_loop"].to_numpy()

    # Precompute alive masks and position arrays for performance
    alive_masks = precompute_alive_masks(df, entities)
    pos_arrays = precompute_position_arrays(df, entities)

    # Initialize feature arrays at computation intervals
    # We store results keyed by row index for interval rows only
    interval_rows = []
    for i in range(n_rows):
        if game_loops[i] % 10 == 0:
            interval_rows.append(i)

    # Initialize output arrays with defaults
    p1_direction = np.full(n_rows, "neutral", dtype=object)
    p2_direction = np.full(n_rows, "neutral", dtype=object)
    p1_army_size = np.zeros(n_rows, dtype=np.int64)
    p2_army_size = np.zeros(n_rows, dtype=np.int64)
    p1_army_count = np.zeros(n_rows, dtype=np.int64)
    p2_army_count = np.zeros(n_rows, dtype=np.int64)
    p1_complexity = np.zeros(n_rows, dtype=np.float64)
    p2_complexity = np.zeros(n_rows, dtype=np.float64)

    # Track previous centroids for movement direction
    prev_centroid = {"p1": None, "p2": None}

    # Get opponent base mapping
    opponent = {"p1": "p2", "p2": "p1"}

    # Process each interval row
    for idx in interval_rows:
        gl = game_loops[idx]

        for player in ["p1", "p2"]:
            # Collect alive army unit positions
            positions, unit_types = collect_army_positions_fast(
                idx, player, entities, alive_masks, pos_arrays
            )

            # Feature 4: Army Complexity Ratio
            n_unique_types = count_alive_army_types_fast(idx, player, entities, alive_masks)
            if gl > 0:
                complexity = n_unique_types / gl
            else:
                complexity = 0.0

            if player == "p1":
                p1_complexity[idx] = complexity
            else:
                p2_complexity[idx] = complexity

            # Clustering (used for Features 1, 2, 3)
            if len(positions) >= DBSCAN_MIN_SAMPLES:
                labels, n_clusters = cluster_army_units(positions)
                centroid, main_size = find_largest_cluster(positions, labels)
            elif len(positions) > 0:
                # Fewer units than min_samples: treat all as one group if > 0
                labels = np.zeros(len(positions), dtype=int)
                n_clusters = 0  # Not enough for a formal army cluster
                centroid = positions.mean(axis=0) if len(positions) > 0 else None
                main_size = len(positions)
            else:
                labels = np.array([], dtype=int)
                n_clusters = 0
                centroid = None
                main_size = 0

            # Feature 2: Main Army Size
            if player == "p1":
                p1_army_size[idx] = main_size
            else:
                p2_army_size[idx] = main_size

            # Feature 3: Number of Armies
            if player == "p1":
                p1_army_count[idx] = n_clusters
            else:
                p2_army_count[idx] = n_clusters

            # Feature 1: Main Army Movement Direction
            own_base = bases.get(player)
            opp_base = bases.get(opponent[player])
            direction = determine_movement_direction(
                centroid, prev_centroid[player], own_base, opp_base
            )

            if player == "p1":
                p1_direction[idx] = direction
            else:
                p2_direction[idx] = direction

            # Update previous centroid for next interval
            prev_centroid[player] = centroid

    # Forward-fill interval values to non-interval rows
    # Strategy: walk through all rows; at each interval row, update the current value;
    # at non-interval rows, carry forward the current value.
    curr_p1_dir = "neutral"
    curr_p2_dir = "neutral"
    curr_p1_size = 0
    curr_p2_size = 0
    curr_p1_count = 0
    curr_p2_count = 0
    curr_p1_complexity = 0.0
    curr_p2_complexity = 0.0

    interval_set = set(interval_rows)

    for i in range(n_rows):
        if i in interval_set:
            # This is an interval row - read the computed values and carry them
            curr_p1_dir = p1_direction[i]
            curr_p2_dir = p2_direction[i]
            curr_p1_size = p1_army_size[i]
            curr_p2_size = p2_army_size[i]
            curr_p1_count = p1_army_count[i]
            curr_p2_count = p2_army_count[i]
            curr_p1_complexity = p1_complexity[i]
            curr_p2_complexity = p2_complexity[i]
        else:
            # Non-interval row - forward-fill
            p1_direction[i] = curr_p1_dir
            p2_direction[i] = curr_p2_dir
            p1_army_size[i] = curr_p1_size
            p2_army_size[i] = curr_p2_size
            p1_army_count[i] = curr_p1_count
            p2_army_count[i] = curr_p2_count
            p1_complexity[i] = curr_p1_complexity
            p2_complexity[i] = curr_p2_complexity

    # Add columns to dataframe
    df["p1_main_army_direction"] = p1_direction
    df["p2_main_army_direction"] = p2_direction
    df["p1_main_army_size"] = p1_army_size.astype(np.int64)
    df["p2_main_army_size"] = p2_army_size.astype(np.int64)
    df["p1_army_count"] = p1_army_count.astype(np.int64)
    df["p2_army_count"] = p2_army_count.astype(np.int64)
    df["p1_army_complexity_ratio"] = p1_complexity
    df["p2_army_complexity_ratio"] = p2_complexity

    return df


# ---------------------------------------------------------------------------
# Single File Processing
# ---------------------------------------------------------------------------

def process_single_file(filepath):
    """Process a single parquet file and add army feature columns.

    Args:
        filepath: Path to the input parquet file

    Returns:
        pd.DataFrame: The augmented dataframe with army feature columns added
    """
    df = pd.read_parquet(filepath)

    # Parse all entity columns
    entities = parse_entity_columns(df.columns)

    # Find base positions
    bases = find_base_positions(df, entities)

    # Compute all army features
    df = compute_army_features(df, entities, bases)

    # Ensure no NaN values in new columns
    df["p1_main_army_direction"] = df["p1_main_army_direction"].fillna("neutral")
    df["p2_main_army_direction"] = df["p2_main_army_direction"].fillna("neutral")
    df["p1_main_army_size"] = df["p1_main_army_size"].fillna(0).astype(np.int64)
    df["p2_main_army_size"] = df["p2_main_army_size"].fillna(0).astype(np.int64)
    df["p1_army_count"] = df["p1_army_count"].fillna(0).astype(np.int64)
    df["p2_army_count"] = df["p2_army_count"].fillna(0).astype(np.int64)
    df["p1_army_complexity_ratio"] = df["p1_army_complexity_ratio"].fillna(0.0)
    df["p2_army_complexity_ratio"] = df["p2_army_complexity_ratio"].fillna(0.0)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir, output_dir, print_output=False, file_extension=".parquet", naming_pattern="_game_state"):
    """Process all game state parquet files and add army feature columns.

    Args:
        input_dir: Directory containing input parquet files.
        output_dir: Directory to write output parquet files.
        print_output: If True, print summary info to console.
        file_extension: File extension to match.
        naming_pattern: Pattern to match input files (glob: *{naming_pattern}{file_extension}).
    """
    # Setup logging
    setup_logging()

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Validate input directory
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        if print_output:
            print(f"Error: Input directory does not exist: {input_path}")
        return

    # Create output directory if needed
    output_path.mkdir(parents=True, exist_ok=True)

    # Find matching files
    glob_pattern = f"*{naming_pattern}{file_extension}"
    parquet_files = sorted(input_path.glob(glob_pattern))

    if not parquet_files:
        logger.warning(f"No files matching '{glob_pattern}' found in {input_path}")
        if print_output:
            print(f"No files matching '{glob_pattern}' found in {input_path}")
        return

    logger.info(f"Starting army feature engineering")
    logger.info(f"Input directory: {input_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Found {len(parquet_files)} files to process")

    if print_output:
        print(f"Starting army feature engineering")
        print(f"Input directory: {input_path}")
        print(f"Output directory: {output_path}")
        print(f"Found {len(parquet_files)} files to process")

    # Process files with progress bar
    processed_files = []
    failed_files = []

    for filepath in tqdm(parquet_files, desc="Processing files", unit="file"):
        tqdm.write(f"  Processing: {filepath.name}")
        try:
            df_out = process_single_file(filepath)

            # Write to output directory with the same filename
            out_filepath = output_path / filepath.name
            df_out.to_parquet(out_filepath, index=False)

            processed_files.append(filepath.name)
        except Exception as e:
            logger.warning(f"Failed to process {filepath.name}: {e}")
            failed_files.append(filepath.name)
            if print_output:
                print(f"  WARNING: Failed to process {filepath.name}: {e}")

    # Summary logging
    logger.info(f"Processing complete. {len(processed_files)} succeeded, {len(failed_files)} failed.")
    if print_output:
        print(f"\nProcessing complete. {len(processed_files)} succeeded, {len(failed_files)} failed.")

    # DEBUG-level summary with filenames
    if processed_files:
        logger.debug("Successfully processed files:\n  " + "\n  ".join(processed_files))
    if failed_files:
        logger.debug("Failed files:\n  " + "\n  ".join(failed_files))

    logger.info("Army feature engineering finished.")
    if print_output:
        print("Army feature engineering finished.")


if __name__ == "__main__":
    main(
        input_dir="data/quickstart/parquet",
        output_dir="data/quickstart/features",
        print_output=True,
    )

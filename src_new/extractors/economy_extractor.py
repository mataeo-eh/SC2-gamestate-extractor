"""
EconomyExtractor: Extracts economy data from SC2 observations.

This component handles:
- Extracting economy metrics from player_common (resources, supply, workers)
- Extracting collection statistics from score details
- Extracting spending totals from score details (spent_minerals, spent_vespene)
- Extracting race-specific counts (warp_gate_count, larva_count) from player_common
- Simple field extraction with no state tracking needed

Compatible with both replay extraction modes:
- Legacy two-pass mode: the replay is started with observed_player_id=N, so
  player_common and score_details are already scoped to that player.
- Observer mode (single-pass): the pipeline starts the replay without
  observed_player_id, steps forward once, then switches perspective to each
  player via ActionObserverPlayerPerspective before calling observe(). The
  resulting observation's player_common and score_details are scoped to the
  player whose perspective was activated. This extractor is called once per
  player per game-loop step, and does NOT perform any perspective switching
  itself — that is the pipeline's responsibility.
"""

from typing import Dict
import logging


logger = logging.getLogger(__name__)


class EconomyExtractor:
    """
    Extracts economy data from SC2 observations.

    This class extracts resource counts, supply information, worker counts,
    collection rates, spending totals, and race-specific counts from the
    observation data. Unlike unit/building extractors, this does not require
    state tracking across frames.

    Works identically in both legacy two-pass mode (observed_player_id set at
    replay start) and observer mode (pipeline switches perspective before each
    observe() call). The extractor simply reads whatever player_common and
    score_details the observation already contains — it is agnostic to how that
    perspective was established.

    Example usage:
        extractor = EconomyExtractor(player_id=1)

        for obs in game_loop_iterator:
            economy_data = extractor.extract(obs)

            # Print economy state
            print(f"Resources: {economy_data['minerals']}m, {economy_data['vespene']}g")
            print(f"Supply: {economy_data['supply_used']}/{economy_data['supply_cap']}")
            print(f"Workers: {economy_data['workers']}, Idle: {economy_data['idle_workers']}")
    """

    def __init__(self, player_id: int):
        """
        Initialize the EconomyExtractor.

        Args:
            player_id: Player ID this extractor is tracking (1 or 2)
        """
        self.player_id = player_id

    def extract(self, obs) -> Dict[str, float]:
        """
        Extract all economy data from observation.

        The observation's player_common and score_details are already scoped to
        the correct player by the time this method is called. In legacy two-pass
        mode this is because the replay was started with observed_player_id=N.
        In observer mode this is because the pipeline switched perspective via
        ActionObserverPlayerPerspective(player_id=N) and then called observe()
        before invoking this method. Either way, the extractor simply reads
        the fields — no perspective logic is needed here.

        Args:
            obs: SC2 observation from controller.observe()

        Returns:
            Dictionary containing economy metrics:
            {
                # Current resources
                'minerals': 450,
                'vespene': 200,

                # Supply
                'supply_used': 45,
                'supply_cap': 60,
                'food_army': 30,
                'workers': 15,

                # Worker and army counts
                'idle_workers': 2,
                'army_count': 30,

                # Race-specific counts
                'warp_gate_count': 0,   # Protoss only, 0 for other races
                'larva_count': 0,       # Zerg only, 0 for other races

                # Collection totals
                'collected_minerals': 15000,
                'collected_vespene': 8000,

                # Spending totals
                'spent_minerals': 12000,
                'spent_vespene': 6000,

                # Collection rates (per minute)
                'collection_rate_minerals': 1200.0,
                'collection_rate_vespene': 600.0,
            }

        Depends on / calls:
            _get_default_economy_data() — used as fallback on extraction errors
        """
        try:
            # --- player_common fields ---
            # player_common is perspective-dependent: it reflects whichever
            # player the observation is scoped to (set by the pipeline).
            player_common = obs.observation.player_common

            # Current resources
            minerals = player_common.minerals
            vespene = player_common.vespene

            # Supply information
            food_used = player_common.food_used
            food_cap = player_common.food_cap
            food_army = player_common.food_army
            food_workers = player_common.food_workers

            # Worker and army counts
            idle_worker_count = player_common.idle_worker_count
            army_count = player_common.army_count

            # Race-specific counts from player_common.
            # warp_gate_count is only non-zero for Protoss players;
            # larva_count is only non-zero for Zerg players.
            # Both default to 0 for races that don't use them.
            warp_gate_count = player_common.warp_gate_count
            larva_count = player_common.larva_count

            # --- score_details fields ---
            # score_details is also perspective-dependent, same scoping as
            # player_common above.
            score_details = obs.observation.score.score_details

            # Total resources collected over the course of the game
            collected_minerals = score_details.collected_minerals
            collected_vespene = score_details.collected_vespene

            # Total resources spent over the course of the game
            spent_minerals = score_details.spent_minerals
            spent_vespene = score_details.spent_vespene

            # Collection rates (resources per minute)
            collection_rate_minerals = score_details.collection_rate_minerals
            collection_rate_vespene = score_details.collection_rate_vespene

            # Build result dictionary
            # Keys match the names expected by wide_table_builder and schema_manager.
            # New fields (spent_minerals, spent_vespene, warp_gate_count,
            # larva_count) are additive — no existing keys are removed.
            economy_data = {
                # Current resources
                'minerals': minerals,
                'vespene': vespene,

                # Supply (named to match schema: supply_used, supply_cap)
                'supply_used': food_used,
                'supply_cap': food_cap,
                'food_army': food_army,
                'workers': food_workers,

                # Worker and army counts
                'idle_workers': idle_worker_count,
                'army_count': army_count,

                # Race-specific counts
                'warp_gate_count': warp_gate_count,
                'larva_count': larva_count,

                # Collection totals
                'collected_minerals': collected_minerals,
                'collected_vespene': collected_vespene,

                # Spending totals
                'spent_minerals': spent_minerals,
                'spent_vespene': spent_vespene,

                # Collection rates (per minute)
                'collection_rate_minerals': collection_rate_minerals,
                'collection_rate_vespene': collection_rate_vespene,
            }

            return economy_data

        except Exception as e:
            logger.error(f"Error extracting economy data: {e}")
            # Return default values on error
            return self._get_default_economy_data()

    def _get_default_economy_data(self) -> Dict[str, float]:
        """
        Get default economy data (all zeros) for error cases.

        Must stay in sync with the keys returned by extract(). Any field
        added to extract()'s output dictionary must also be added here.

        Returns:
            Dictionary with all economy fields set to 0

        Called by:
            extract() — when an exception occurs during field extraction
        """
        return {
            'minerals': 0,
            'vespene': 0,
            'supply_used': 0,
            'supply_cap': 0,
            'food_army': 0,
            'workers': 0,
            'idle_workers': 0,
            'army_count': 0,
            'warp_gate_count': 0,
            'larva_count': 0,
            'collected_minerals': 0,
            'collected_vespene': 0,
            'spent_minerals': 0,
            'spent_vespene': 0,
            'collection_rate_minerals': 0.0,
            'collection_rate_vespene': 0.0,
        }

    def get_summary(self, economy_data: Dict[str, float]) -> str:
        """
        Get a human-readable summary of economy data.

        Args:
            economy_data: Output from extract()

        Returns:
            Formatted string summary
        """
        summary = f"Resources: {economy_data['minerals']}m, {economy_data['vespene']}g | "
        summary += f"Supply: {economy_data['supply_used']}/{economy_data['supply_cap']} "
        summary += f"(Army: {economy_data['food_army']}, Workers: {economy_data['workers']}) | "
        summary += f"Idle Workers: {economy_data['idle_workers']} | "
        summary += f"Collection: {economy_data['collection_rate_minerals']:.0f}m/min, "
        summary += f"{economy_data['collection_rate_vespene']:.0f}g/min"
        return summary

    def reset(self):
        """
        Reset extractor state.

        Note: This extractor has no state to reset, but this method
        is provided for consistency with other extractors.
        """
        pass

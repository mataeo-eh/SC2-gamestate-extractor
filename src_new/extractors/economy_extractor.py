"""
EconomyExtractor: Extracts economy data from SC2 observations.

This component handles:
- Extracting economy metrics from player_common (resources, supply, workers)
- Extracting collection statistics from score details
- Simple field extraction with no state tracking needed
"""

from typing import Dict
import logging


logger = logging.getLogger(__name__)


class EconomyExtractor:
    """
    Extracts economy data from SC2 observations.

    This class extracts resource counts, supply information, worker counts,
    and collection rates from the observation data. Unlike unit/building extractors,
    this does not require state tracking across frames.

    Example usage:
        extractor = EconomyExtractor(player_id=1)

        for obs in game_loop_iterator:
            economy_data = extractor.extract(obs)

            # Print economy state
            print(f"Resources: {economy_data['minerals']}m, {economy_data['vespene']}g")
            print(f"Supply: {economy_data['food_used']}/{economy_data['food_cap']}")
            print(f"Workers: {economy_data['food_workers']}, Idle: {economy_data['idle_worker_count']}")
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

        Args:
            obs: SC2 observation from controller.observe()

        Returns:
            Dictionary containing economy metrics:
            {
                # Current resources
                'minerals': 450,
                'vespene': 200,

                # Supply (food)
                'food_used': 45,
                'food_cap': 60,
                'food_army': 30,
                'food_workers': 15,

                # Worker and army counts
                'idle_worker_count': 2,
                'army_count': 30,

                # Collection totals
                'collected_minerals': 15000,
                'collected_vespene': 8000,

                # Collection rates (per minute)
                'collection_rate_minerals': 1200.0,
                'collection_rate_vespene': 600.0,
            }
        """
        try:
            # Extract from player_common
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

            # Extract from score details (collection statistics)
            score_details = obs.observation.score.score_details

            # Total resources collected
            collected_minerals = score_details.collected_minerals
            collected_vespene = score_details.collected_vespene

            # Collection rates (resources per minute)
            collection_rate_minerals = score_details.collection_rate_minerals
            collection_rate_vespene = score_details.collection_rate_vespene

            # Build result dictionary
            economy_data = {
                # Current resources
                'minerals': minerals,
                'vespene': vespene,

                # Supply (food)
                'food_used': food_used,
                'food_cap': food_cap,
                'food_army': food_army,
                'food_workers': food_workers,

                # Worker and army counts
                'idle_worker_count': idle_worker_count,
                'army_count': army_count,

                # Collection totals
                'collected_minerals': collected_minerals,
                'collected_vespene': collected_vespene,

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

        Returns:
            Dictionary with all economy fields set to 0
        """
        return {
            'minerals': 0,
            'vespene': 0,
            'food_used': 0,
            'food_cap': 0,
            'food_army': 0,
            'food_workers': 0,
            'idle_worker_count': 0,
            'army_count': 0,
            'collected_minerals': 0,
            'collected_vespene': 0,
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
        summary += f"Supply: {economy_data['food_used']}/{economy_data['food_cap']} "
        summary += f"(Army: {economy_data['food_army']}, Workers: {economy_data['food_workers']}) | "
        summary += f"Idle Workers: {economy_data['idle_worker_count']} | "
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

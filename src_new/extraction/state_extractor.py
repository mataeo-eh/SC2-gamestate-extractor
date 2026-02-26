"""
StateExtractor: Extracts all required game state from pysc2 observations.

This component orchestrates the extraction of complete game state including
units, buildings, upgrades, and messages from SC2 observations.

Economy data is NOT extracted here. In observer mode (the only supported path),
economy is pre-loaded from the replay file via s2protocol by
economy_extractor.load_economy_snapshots() and injected into the state dict
by extraction_pipeline.py. This avoids the observer-mode bug where the engine's
player_common and score_details are always zero.

The primary extraction entry point is extract_observation_observer_mode(),
which takes two per-player observations (one for each player perspective) and
combines them into a single state dict with per-player upgrades data alongside
global units/buildings data. Economy keys (p1_economy, p2_economy) are absent
from the returned dict -- the pipeline adds them afterward.
"""

from typing import Dict, List, Any
import logging

from ..extractors.unit_extractor import UnitExtractor
from ..extractors.building_extractor import BuildingExtractor
from ..extractors.upgrade_extractor import UpgradeExtractor


logger = logging.getLogger(__name__)


class StateExtractor:
    """
    Extracts game state from pysc2 observations (units, buildings, upgrades, messages).

    This class orchestrates the individual extractors (units, buildings, upgrades)
    and provides a unified interface for extracting game state at each time step.
    Economy data is handled separately by economy_extractor.load_economy_snapshots()
    and injected by the pipeline, not by this class.

    The primary extraction entry point is extract_observation_observer_mode(),
    which takes two per-player observations and combines them into a single
    state dict. extract_observation() remains available for single-obs legacy
    use cases.
    """

    def __init__(self):
        """
        Initialize the StateExtractor with all component extractors.

        Creates unit, building, and upgrade extractors for both players.
        Economy extractors are NOT created here -- economy is pre-loaded
        from the replay file by economy_extractor.load_economy_snapshots()
        and injected by extraction_pipeline.py.
        """
        # Create extractors for both players
        self.unit_extractors = {
            1: UnitExtractor(player_id=1),
            2: UnitExtractor(player_id=2),
        }

        self.building_extractors = {
            1: BuildingExtractor(player_id=1),
            2: BuildingExtractor(player_id=2),
        }

        self.upgrade_extractors = {
            1: UpgradeExtractor(player_id=1),
            2: UpgradeExtractor(player_id=2),
        }

        logger.info("StateExtractor initialized")

    def extract_observation(self, obs, game_loop: int) -> Dict[str, Any]:
        """
        Extract state from a single observation (legacy single-obs path).

        Note: Economy data (p1_economy, p2_economy) is NOT included in the
        returned dict. Economy is now pre-loaded from the replay file via
        s2protocol and injected by the pipeline. Callers needing economy data
        should use economy_extractor.get_economy_at_loop() separately.

        Args:
            obs: SC2 observation from controller.observe()
            game_loop: Current game loop number

        Returns:
            Dictionary with extracted state:
            {
                'game_loop': int,
                'p1_units': dict,
                'p2_units': dict,
                'p1_buildings': dict,
                'p2_buildings': dict,
                'p1_upgrades': dict,
                'p2_upgrades': dict,
                'messages': list,
            }
        """
        state = {'game_loop': game_loop}

        # Extract units for both players
        state['p1_units'] = self.extract_units(obs, player_id=1)
        state['p2_units'] = self.extract_units(obs, player_id=2)

        # Extract buildings for both players
        state['p1_buildings'] = self.extract_buildings(obs, player_id=1)
        state['p2_buildings'] = self.extract_buildings(obs, player_id=2)

        # Economy is NOT extracted here — it comes from s2protocol via the pipeline.
        # See economy_extractor.load_economy_snapshots().

        # Extract upgrades for both players
        state['p1_upgrades'] = self.extract_upgrades(obs, player_id=1)
        state['p2_upgrades'] = self.extract_upgrades(obs, player_id=2)

        # Extract messages
        state['messages'] = self.extract_messages(obs)

        return state

    def extract_observation_observer_mode(
        self,
        obs_p1,
        obs_p2,
        game_loop: int,
    ) -> Dict[str, Any]:
        """
        Extract state from observer mode observations (units, buildings, upgrades, messages).

        In observer mode, the pipeline makes TWO observe() calls per game step:
        one after switching perspective to player 1, and one after switching to
        player 2. Both observations share the same raw_data.units (global),
        but have different upgrade data (perspective-dependent).

        Economy data (p1_economy, p2_economy) is NOT extracted here. In observer
        mode the engine's player_common and score_details are always zero, so
        economy is pre-loaded from the replay file via s2protocol and injected
        into the state dict by extraction_pipeline.py afterward.

        Args:
            obs_p1: Observation after switching to player 1 perspective.
                    Used for units, buildings, P1 upgrades, messages.
            obs_p2: Observation after switching to player 2 perspective.
                    Used for P2 upgrades.
            game_loop: Current game loop number.

        Returns:
            State dict (economy keys absent -- pipeline adds them):
            {
                'game_loop': int,
                'p1_units': dict,
                'p2_units': dict,
                'p1_buildings': dict,
                'p2_buildings': dict,
                'p1_upgrades': dict,  # from obs_p1 (correct P1 perspective)
                'p2_upgrades': dict,  # from obs_p2 (correct P2 perspective)
                'messages': list,
            }

        Depends on / calls:
            - extract_units() for both players (uses obs_p1)
            - extract_buildings() for both players (uses obs_p1)
            - extract_upgrades() with obs_p1 for P1, obs_p2 for P2
            - extract_messages() (uses obs_p1)
        """
        state = {'game_loop': game_loop}

        # Units and buildings come from obs_p1 — raw_data.units is identical
        # regardless of which player perspective is active, so the choice of
        # obs_p1 vs obs_p2 is arbitrary here.
        state['p1_units'] = self.extract_units(obs_p1, player_id=1)
        state['p2_units'] = self.extract_units(obs_p1, player_id=2)

        state['p1_buildings'] = self.extract_buildings(obs_p1, player_id=1)
        state['p2_buildings'] = self.extract_buildings(obs_p1, player_id=2)

        # Economy is NOT extracted from the engine here — player_common and
        # score_details are always zero in observer mode. Economy is pre-loaded
        # from the replay file via s2protocol (economy_extractor.load_economy_snapshots)
        # and injected by extraction_pipeline.py after this method returns.

        # Upgrades are perspective-dependent (raw_data.player.upgrade_ids
        # reflects the observed player).
        state['p1_upgrades'] = self.extract_upgrades(obs_p1, player_id=1)
        state['p2_upgrades'] = self.extract_upgrades(obs_p2, player_id=2)

        # Messages from obs_p1 — chat is global and identical in both obs
        state['messages'] = self.extract_messages(obs_p1)

        return state

    def extract_units(self, obs, player_id: int) -> Dict[str, Dict]:
        """
        Extract all units for a player.

        Args:
            obs: SC2 observation
            player_id: Player ID (1 or 2)

        Returns:
            Dictionary mapping unit IDs to unit data with '_lifecycle' key
            indicating lifecycle state (unit_started/building/completed/existing/destroyed)
        """
        extractor = self.unit_extractors[player_id]
        units = extractor.extract(obs)
        return units

    def extract_buildings(self, obs, player_id: int) -> Dict[str, Dict]:
        """
        Extract all buildings for a player.

        Args:
            obs: SC2 observation
            player_id: Player ID (1 or 2)

        Returns:
            Dictionary mapping building IDs to building data with '_lifecycle' key
            indicating lifecycle state (building_started/under_construction/completed/
            existing/destroyed/cancelled)
        """
        extractor = self.building_extractors[player_id]
        buildings = extractor.extract(obs)
        return buildings

    def extract_upgrades(self, obs, player_id: int) -> Dict[str, Any]:
        """
        Extract completed upgrades for a player.

        Args:
            obs: SC2 observation
            player_id: Player ID (1 or 2)

        Returns:
            Dictionary with upgrade data:
            {
                'upgrade_name': level,
                ...
            }

        # TODO: Test case - Extract upgrade levels
        """
        extractor = self.upgrade_extractors[player_id]
        upgrades = extractor.extract(obs)
        return upgrades

    def extract_messages(self, obs) -> List[Dict[str, Any]]:
        """
        Extract chat messages at this timestep.

        Args:
            obs: SC2 observation

        Returns:
            List of message dictionaries:
            [
                {
                    'game_loop': int,
                    'player_id': int,
                    'message': str,
                },
                ...
            ]

        # TODO: Test case - Extract messages
        """
        messages = []

        # Extract chat messages from observation
        # Note: Chat messages are on ResponseObservation directly, not on Observation
        if hasattr(obs, 'chat'):
            for msg in obs.chat:
                messages.append({
                    'game_loop': obs.observation.game_loop,
                    'player_id': msg.player_id,
                    'message': msg.message,
                })

        return messages

    def reset(self):
        """
        Reset all extractors.

        Resets unit, building, and upgrade extractors. Economy extractors
        are not managed here (economy is handled by
        economy_extractor.load_economy_snapshots() at the pipeline level).
        """
        for extractor in self.unit_extractors.values():
            extractor.reset()
        for extractor in self.building_extractors.values():
            extractor.reset()
        for extractor in self.upgrade_extractors.values():
            extractor.reset()

        logger.info("StateExtractor reset")

"""
ReplayLoader: Loads and initializes SC2 replays with pysc2.

This component handles:
- Loading replay data from .SC2Replay files
- Detecting and validating replay versions
- Configuring interface options for ground truth access
- Starting the SC2 controller
- Managing player perspective switching for multi-player ground truth
"""

from typing import Optional, Tuple
import logging

from pysc2 import run_configs
from pysc2.lib import replay
from s2clientprotocol import sc2api_pb2 as sc_pb


logger = logging.getLogger(__name__)


class ReplayLoader:
    """
    Loads SC2 replays and initializes the pysc2 controller.

    This class provides a clean interface for loading replays with the correct
    configuration for ground truth extraction.
    """

    def __init__(
        self,
        show_cloaked: bool = True,
        show_burrowed_shadows: bool = True,
        show_placeholders: bool = True,
    ):
        """
        Initialize the ReplayLoader.

        Args:
            show_cloaked: Show cloaked units (essential for ground truth)
            show_burrowed_shadows: Show burrowed units
            show_placeholders: Show queued buildings
        """
        self.show_cloaked = show_cloaked
        self.show_burrowed_shadows = show_burrowed_shadows
        self.show_placeholders = show_placeholders

        # Interface configuration for ground truth access
        self.interface = sc_pb.InterfaceOptions(
            raw=True,                              # CRITICAL: Enable raw data
            score=True,                            # Enable score information
            show_cloaked=show_cloaked,             # Show cloaked units
            show_burrowed_shadows=show_burrowed_shadows,  # Show burrowed units
            show_placeholders=show_placeholders,    # Show queued buildings
        )

        self.replay_data = None
        self.replay_version = None
        self.run_config = None
        self.controller = None

    def load_replay(self, replay_path: str) -> Tuple[bytes, replay.ReplayVersion]:
        """
        Load replay data and detect version.

        Args:
            replay_path: Path to .SC2Replay file

        Returns:
            Tuple of (replay_data, replay_version)

        Raises:
            FileNotFoundError: If replay file doesn't exist
            ValueError: If replay version detection fails
        """
        logger.info(f"Loading replay: {replay_path}")

        # Get initial run config
        run_config = run_configs.get()

        try:
            # Load replay data
            self.replay_data = run_config.replay_data(replay_path)

            # Get replay version
            self.replay_version = replay.get_replay_version(self.replay_data)

            logger.info(f"Replay version: {self.replay_version.game_version}")

            # Update run_config to match replay version
            self.run_config = run_configs.get(version=self.replay_version)

            return self.replay_data, self.replay_version

        except FileNotFoundError:
            logger.error(f"Replay file not found: {replay_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load replay: {e}")
            raise ValueError(f"Failed to load replay: {e}")

    def get_replay_info(self, controller) -> sc_pb.ResponseReplayInfo:
        """
        Get replay metadata information.

        Args:
            controller: SC2 controller instance

        Returns:
            ResponseReplayInfo with replay metadata
        """
        if self.replay_data is None:
            raise ValueError("No replay loaded. Call load_replay() first.")

        logger.info("Getting replay info...")
        info = controller.replay_info(self.replay_data)

        logger.info(f"Map: {info.map_name}")
        logger.info(f"Duration: {info.game_duration_loops} loops "
                   f"({info.game_duration_loops / 22.4:.1f} seconds)")
        logger.info(f"Players: {len(info.player_info)}")

        for i, player_info in enumerate(info.player_info):
            race_name = sc_pb.Race.Name(player_info.player_info.race_actual)
            logger.info(f"  Player {i+1}: {race_name}, "
                       f"APM: {player_info.player_apm}, "
                       f"MMR: {player_info.player_mmr}")

        return info

    def start_replay(
        self,
        controller,
        observed_player_id: int = 1,
        disable_fog: bool = False,
    ) -> None:
        """
        Start replay playback from a specific player's perspective.

        Args:
            controller: SC2 controller instance
            observed_player_id: Player ID to observe from (1 or 2)
            disable_fog: Disable fog of war (use with caution)
        """
        if self.replay_data is None:
            raise ValueError("No replay loaded. Call load_replay() first.")

        logger.info(f"Starting replay playback (observing player {observed_player_id})...")

        # Configure replay start request
        replay_request = sc_pb.RequestStartReplay(
            replay_data=self.replay_data,
            map_data=None,  # Map should be installed in SC2
            options=self.interface,
            observed_player_id=observed_player_id,
            disable_fog=disable_fog,
        )

        # Start the replay
        controller.start_replay(replay_request)
        logger.info(f"Replay started successfully")

    def start_sc2_instance(self, want_rgb: bool = False):
        """
        Start SC2 instance and return controller.

        This is a context manager that should be used with 'with' statement.

        Args:
            want_rgb: Whether to capture RGB pixels (not needed for ground truth)

        Returns:
            SC2 controller context manager

        Example:
            >>> loader = ReplayLoader()
            >>> loader.load_replay("replay.SC2Replay")
            >>> with loader.start_sc2_instance() as controller:
            >>>     loader.start_replay(controller, observed_player_id=1)
            >>>     # Process replay...
        """
        if self.run_config is None:
            raise ValueError("No replay loaded. Call load_replay() first.")

        logger.info("Starting SC2 instance...")
        return self.run_config.start(want_rgb=want_rgb)


# Convenience function for simple usage
def load_and_start_replay(
    replay_path: str,
    observed_player_id: int = 1,
) -> Tuple['ReplayLoader', 'controller', sc_pb.ResponseReplayInfo]:
    """
    Convenience function to load and start a replay in one call.

    Args:
        replay_path: Path to .SC2Replay file
        observed_player_id: Player ID to observe from (1 or 2)

    Returns:
        Tuple of (loader, controller, replay_info)

    Example:
        >>> loader, controller, info = load_and_start_replay("replay.SC2Replay")
        >>> # Use controller to process replay
        >>> controller.quit()
    """
    loader = ReplayLoader()
    loader.load_replay(replay_path)

    controller = loader.start_sc2_instance().__enter__()

    try:
        info = loader.get_replay_info(controller)
        loader.start_replay(controller, observed_player_id=observed_player_id)
        return loader, controller, info
    except Exception as e:
        controller.__exit__(None, None, None)
        raise

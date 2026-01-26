"""
ReplayLoader: Loads and initializes SC2 replays with pysc2.

This component wraps the pipeline.ReplayLoader and provides a clean interface
for the extraction pipeline with perfect information observation settings.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging

from pysc2 import run_configs
from pysc2.lib import replay
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import common_pb2

from ..pipeline.replay_loader import ReplayLoader as PipelineReplayLoader


logger = logging.getLogger(__name__)


class ReplayLoader:
    """
    Loads and initializes SC2 replays with pysc2 for ground truth extraction.

    This class provides perfect information observation settings required for
    complete ground truth game state extraction.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize with observation settings.

        Args:
            config: Optional configuration dictionary with keys:
                - show_cloaked (bool): Show cloaked units (default: True)
                - show_burrowed_shadows (bool): Show burrowed units (default: True)
                - show_placeholders (bool): Show queued buildings (default: True)
        """
        config = config or {}

        # Extract configuration
        show_cloaked = config.get('show_cloaked', True)
        show_burrowed_shadows = config.get('show_burrowed_shadows', True)
        show_placeholders = config.get('show_placeholders', True)

        # Initialize underlying pipeline loader with perfect information settings
        self._pipeline_loader = PipelineReplayLoader(
            show_cloaked=show_cloaked,
            show_burrowed_shadows=show_burrowed_shadows,
            show_placeholders=show_placeholders,
        )

        self.replay_data = None
        self.replay_version = None
        self.replay_info = None
        self.controller = None

        logger.info("ReplayLoader initialized with perfect information settings")

    def load_replay(self, replay_path: Path):
        """
        Load replay with full perfect information settings.

        This method loads the replay file and prepares it for iteration. It does
        NOT start the SC2 instance yet - use start_sc2_instance() for that.

        Args:
            replay_path: Path to .SC2Replay file

        Returns:
            Self (for method chaining)

        Raises:
            FileNotFoundError: If replay file doesn't exist
            ValueError: If replay is invalid or corrupt

        # TODO: Test case - Load valid replay
        # TODO: Test case - Handle invalid replay path
        # TODO: Test case - Handle corrupt replay file
        """
        replay_path = Path(replay_path)

        if not replay_path.exists():
            logger.error(f"Replay file not found: {replay_path}")
            raise FileNotFoundError(f"Replay file not found: {replay_path}")

        if not replay_path.suffix == '.SC2Replay':
            logger.error(f"Invalid replay file extension: {replay_path}")
            raise ValueError(f"Invalid replay file extension. Expected .SC2Replay, got {replay_path.suffix}")

        try:
            # Load replay data and version
            self.replay_data, self.replay_version = self._pipeline_loader.load_replay(str(replay_path))
            logger.info(f"Successfully loaded replay: {replay_path.name}")
            logger.info(f"Replay version: {self.replay_version.game_version}")

        except Exception as e:
            logger.error(f"Failed to load replay {replay_path}: {e}")
            raise ValueError(f"Failed to load replay: {e}")

        return self

    def start_sc2_instance(self):
        """
        Start SC2 instance and return controller context manager.

        Must be called after load_replay(). This returns a context manager
        that should be used with 'with' statement.

        Returns:
            SC2 controller context manager

        Raises:
            ValueError: If load_replay() hasn't been called

        Example:
            >>> loader = ReplayLoader()
            >>> loader.load_replay(Path("replay.SC2Replay"))
            >>> with loader.start_sc2_instance() as controller:
            >>>     info = loader.get_replay_info(controller)
            >>>     # Process replay...
        """
        if self.replay_data is None:
            raise ValueError("No replay loaded. Call load_replay() first.")

        return self._pipeline_loader.start_sc2_instance()

    def get_replay_info(self, controller) -> Dict[str, Any]:
        """
        Extract replay metadata (map, players, duration, etc.).

        Args:
            controller: SC2 controller instance

        Returns:
            Dictionary with replay metadata:
            {
                'map_name': str,
                'game_duration_loops': int,
                'game_duration_seconds': float,
                'num_players': int,
                'players': [
                    {
                        'player_id': int,
                        'race': str,
                        'apm': float,
                        'mmr': int,
                        'result': str,  # 'Victory', 'Defeat', etc.
                    },
                    ...
                ]
            }

        Raises:
            ValueError: If load_replay() hasn't been called

        # TODO: Test case - Extract correct metadata from known replay
        # TODO: Test case - Verify perfect information mode enabled in interface
        """
        if self.replay_data is None:
            raise ValueError("No replay loaded. Call load_replay() first.")

        # Get replay info from pipeline loader
        info_proto = self._pipeline_loader.get_replay_info(controller)
        self.replay_info = info_proto

        # Convert to dictionary for easier use
        metadata = {
            'map_name': info_proto.map_name,
            'game_duration_loops': info_proto.game_duration_loops,
            'game_duration_seconds': info_proto.game_duration_loops / 22.4,  # 22.4 loops/second
            'num_players': len(info_proto.player_info),
            'players': []
        }

        # Extract player information
        for i, player_info in enumerate(info_proto.player_info):
            player_data = {
                'player_id': i + 1,
                'race': common_pb2.Race.Name(player_info.player_info.race_actual),
                'apm': player_info.player_apm,
                'mmr': player_info.player_mmr,
                'result': sc_pb.Result.Name(player_info.player_result.result),
            }
            metadata['players'].append(player_data)

        logger.info(f"Extracted metadata for replay: {metadata['map_name']}")
        logger.info(f"  Duration: {metadata['game_duration_seconds']:.1f} seconds")
        logger.info(f"  Players: {metadata['num_players']}")

        return metadata

    def start_replay(self, controller, observed_player_id: int = 1, disable_fog: bool = False) -> None:
        """
        Start replay playback from a specific player's perspective.

        For ground truth extraction, you typically want disable_fog=False and will
        switch between player perspectives manually.

        Args:
            controller: SC2 controller instance
            observed_player_id: Player ID to observe from (1 or 2)
            disable_fog: Disable fog of war (use True for perfect information)

        Raises:
            ValueError: If load_replay() hasn't been called
        """
        if self.replay_data is None:
            raise ValueError("No replay loaded. Call load_replay() first.")

        self._pipeline_loader.start_replay(
            controller,
            observed_player_id=observed_player_id,
            disable_fog=disable_fog
        )

        logger.info(f"Replay started for player {observed_player_id}")

    def get_interface_options(self) -> sc_pb.InterfaceOptions:
        """
        Get the interface options being used.

        Returns:
            InterfaceOptions proto with current settings
        """
        return self._pipeline_loader.interface


# Convenience function for quick usage
def load_replay_with_metadata(replay_path: Path, config: Optional[Dict[str, Any]] = None) -> tuple:
    """
    Convenience function to load a replay and extract metadata in one call.

    Args:
        replay_path: Path to .SC2Replay file
        config: Optional configuration dictionary

    Returns:
        Tuple of (loader, controller, metadata)

    Example:
        >>> loader, controller, metadata = load_replay_with_metadata(Path("replay.SC2Replay"))
        >>> print(f"Map: {metadata['map_name']}")
        >>> # Process replay...
        >>> controller.quit()
    """
    loader = ReplayLoader(config)
    loader.load_replay(replay_path)

    controller = loader.start_sc2_instance().__enter__()

    try:
        metadata = loader.get_replay_info(controller)
        return loader, controller, metadata
    except Exception as e:
        controller.__exit__(None, None, None)
        raise

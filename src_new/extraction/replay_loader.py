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
import sc2reader
import mpyq
from s2protocol import versions as s2versions

from ..pipeline.replay_loader import ReplayLoader as PipelineReplayLoader


logger = logging.getLogger(__name__)

# Mapping from sc2reader's human-readable game speed name to the integer
# encoding used by s2protocol (m_gameSpeed) and expected by metadata_writer.
SPEED_NAME_TO_INT: Dict[str, int] = {
    "Slower": 0, "Slow": 1, "Normal": 2, "Fast": 3, "Faster": 4,
}


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
        self.replay_path = None

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

        if not replay_path.suffix.lower() == '.sc2replay':
            logger.error(f"Invalid replay file extension: {replay_path}")
            raise ValueError(f"Invalid replay file extension. Expected .SC2Replay, got {replay_path.suffix}")

        try:
            # Store the replay path for later use by get_replay_info() when
            # extracting map dimensions via sc2reader/s2protocol.
            self.replay_path = replay_path

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
                'game_version': str or absent,  # e.g. "5.0.13"
                'map_width': int or absent,     # pixels, set by _extract_replay_metadata()
                'map_height': int or absent,    # pixels, set by _extract_replay_metadata()
                'game_speed': int or absent,    # 0-4, set by _extract_replay_metadata()
                'data_build': int or absent,    # set by _extract_replay_metadata()
                'base_build': int or absent,    # set by _extract_replay_metadata()
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
                'player_name': player_info.player_info.player_name,
                'race': common_pb2.Race.Name(player_info.player_info.race_actual),
                'apm': player_info.player_apm,
                'mmr': player_info.player_mmr,
                'result': sc_pb.Result.Name(player_info.player_result.result),
            }
            metadata['players'].append(player_data)

        logger.info(f"Extracted metadata for replay: {metadata['map_name']}")
        logger.info(f"  Duration: {metadata['game_duration_seconds']:.1f} seconds")
        logger.info(f"  Players: {metadata['num_players']}")

        # --- Game version (from pysc2) ---
        # self.replay_version is set by load_replay() and exposes the game
        # version string (e.g. "5.0.13").  Set it here before calling into the
        # replay-file-based metadata extraction so it is always available.
        metadata['game_version'] = self.replay_version.game_version

        # --- Replay metadata extraction (map dims, speed, build numbers) ---
        # The SC2 engine (pysc2) does not expose map pixel dimensions, game
        # speed, or build numbers in its replay info protobuf.  We extract
        # them from the replay file itself using sc2reader or, as a fallback,
        # s2protocol + mpyq.
        #
        # Strategy:
        #   1. Try sc2reader.load_replay(path, load_map=True) which parses the
        #      embedded map archive and exposes width/height via map.map_info,
        #      game speed via replay.speed, and data_build via replay.build.
        #   2. If sc2reader fails (e.g. bot replays with missing cache_handles),
        #      fall back to s2protocol's decode_replay_header / initdata /
        #      details for base_build, data_build, game_speed, and map dims.
        #   3. Even when sc2reader succeeds, the s2protocol header is decoded
        #      to obtain base_build (not available from sc2reader).
        #
        # Both approaches read the .SC2Replay file directly — no SC2 game
        # engine required.
        self._extract_replay_metadata(metadata)

        return metadata

    def _extract_replay_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Populate map dimensions, game speed, and build numbers in the metadata dict.

        Extracts the following keys (when available) from the replay file:
        ``map_width``, ``map_height``, ``game_speed``, ``data_build``,
        ``base_build``.

        Attempts two strategies in order:

        1. **sc2reader** -- ``sc2reader.load_replay(path, load_map=True)``
           parses the embedded .s2ma map archive and provides pixel dimensions
           via ``replay.map.map_info.width / .height``.  Also provides
           ``replay.speed`` (human-readable game speed) and ``replay.build``
           (data build number).
        2. **s2protocol fallback / supplement** -- if sc2reader fails (common
           with bot-generated replays whose ``replay.details`` lacks
           cache_handles), we open the MPQ archive with *mpyq* and decode
           the replay header, initData, and details via *s2protocol* to get
           ``base_build``, ``data_build``, ``game_speed``, and map
           dimensions.  Even when sc2reader succeeds, the s2protocol header
           is always decoded to obtain ``base_build`` (not available from
           sc2reader).

        Both strategies are file-only (no SC2 engine needed) and wrapped in
        try/except so a failure here never crashes the pipeline.

        Only sets a metadata key when extraction succeeds — never sets a key
        to None.

        Args:
            metadata: The metadata dict built by get_replay_info().  Modified
                      in-place to add keys if extraction succeeds.

        Returns:
            None -- mutates *metadata* in-place.

        Depends on / calls:
            - sc2reader.load_replay (strategy 1)
            - mpyq.MPQArchive, s2protocol.versions (strategy 2)
            - SPEED_NAME_TO_INT module-level constant
            - Called by get_replay_info()
        """
        replay_path_str = str(self.replay_path)

        # Track whether sc2reader succeeded so we know whether s2protocol
        # needs to provide the full fallback or just the supplement.
        sc2reader_succeeded = False

        # ----- Strategy 1: sc2reader -----
        try:
            sc2_replay = sc2reader.load_replay(replay_path_str, load_map=True)

            # sc2reader stores map pixel dimensions inside the MapInfo object
            # accessible at replay.map.map_info after load_map=True.
            map_info = sc2_replay.map.map_info
            metadata['map_width'] = map_info.width
            metadata['map_height'] = map_info.height
            logger.info(
                f"  Map dimensions (sc2reader): "
                f"{metadata['map_width']}x{metadata['map_height']}"
            )

            # Game speed: sc2reader exposes a human-readable string like
            # "Faster".  Convert to the integer encoding (0-4) expected by
            # metadata_writer via the SPEED_NAME_TO_INT lookup table.
            speed_int = SPEED_NAME_TO_INT.get(sc2_replay.speed)
            if speed_int is not None:
                metadata['game_speed'] = speed_int
                logger.info(f"  Game speed (sc2reader): {sc2_replay.speed} -> {speed_int}")

            # Data build number (integer patch identifier).
            if hasattr(sc2_replay, 'build') and sc2_replay.build is not None:
                metadata['data_build'] = sc2_replay.build
                logger.info(f"  Data build (sc2reader): {sc2_replay.build}")

            sc2reader_succeeded = True
        except Exception as e:
            logger.debug(
                f"sc2reader could not extract replay metadata: {e}. "
                f"Falling back to s2protocol."
            )

        # ----- Strategy 2: s2protocol + mpyq (fallback AND supplement) -----
        # Always runs to obtain base_build (not available from sc2reader).
        # When sc2reader failed, also provides data_build, game_speed, and
        # map dimensions as a full fallback.
        try:
            archive = mpyq.MPQArchive(replay_path_str)

            # Decode the replay header to get base_build and data_build.
            header_content = archive.header['user_data_header']['content']
            header = s2versions.latest().decode_replay_header(header_content)
            base_build = header['m_version']['m_baseBuild']

            # base_build is only available from s2protocol — always set it.
            metadata['base_build'] = base_build
            logger.info(f"  Base build (s2protocol): {base_build}")

            # data_build: use s2protocol value only if sc2reader didn't
            # already provide it (setdefault avoids overwriting).
            # Note: data build lives at the header top-level as 'm_dataBuildNum',
            # NOT inside m_version (which has 'm_build' and 'm_baseBuild').
            metadata.setdefault('data_build', header['m_dataBuildNum'])

            protocol = s2versions.build(base_build)

            if not sc2reader_succeeded:
                # --- Full fallback: game_speed from replay.details ---
                if 'game_speed' not in metadata:
                    try:
                        details_raw = archive.read_file('replay.details')
                        details = protocol.decode_replay_details(details_raw)
                        metadata['game_speed'] = details['m_gameSpeed']
                        logger.info(
                            f"  Game speed (s2protocol): {metadata['game_speed']}"
                        )
                    except Exception as e:
                        logger.debug(f"Could not decode replay.details for game_speed: {e}")

                # --- Full fallback: map dimensions from replay.initData ---
                if 'map_width' not in metadata:
                    try:
                        init_data_raw = archive.read_file('replay.initData')
                        init_data = protocol.decode_replay_initdata(init_data_raw)
                        game_desc = init_data['m_syncLobbyState']['m_gameDescription']
                        metadata['map_width'] = game_desc['m_mapSizeX']
                        metadata['map_height'] = game_desc['m_mapSizeY']
                        logger.info(
                            f"  Map dimensions (s2protocol): "
                            f"{metadata['map_width']}x{metadata['map_height']}"
                        )
                    except Exception as e:
                        logger.debug(f"Could not decode replay.initData for map dims: {e}")

        except Exception as e:
            logger.warning(
                f"Could not extract replay metadata via s2protocol: {e}"
            )

    def start_replay(
        self,
        controller,
        observed_player_id: int = 1,
        disable_fog: bool = False,
        observer_mode: bool = False,
    ) -> None:
        """
        Start replay playback in either player-perspective or observer mode.

        In observer mode, the replay is started without a fixed player
        perspective, enabling per-player querying of player_common, score,
        and upgrade_ids via switch_player_perspective(). The observed_player_id
        parameter is ignored when observer_mode=True.

        In player-perspective mode (default), the replay is locked to one
        player's point of view, preserving the original behavior.

        Args:
            controller: SC2 controller instance (from start_sc2_instance())
            observed_player_id: Player ID to observe from (1 or 2).
                                Ignored when observer_mode=True.
            disable_fog: Disable fog of war. Automatically set to True
                         when observer_mode=True for full unit attributes.
            observer_mode: If True, start in observer mode. Enables
                           switch_player_perspective(). Default: False.

        Raises:
            ValueError: If load_replay() hasn't been called.

        Depends on:
            - load_replay() must be called first
            - Delegates to PipelineReplayLoader.start_replay()
        """
        if self.replay_data is None:
            raise ValueError("No replay loaded. Call load_replay() first.")

        if observer_mode:
            # In observer mode, force disable_fog and ignore observed_player_id
            self._pipeline_loader.start_replay(
                controller,
                observer_mode=True,
                disable_fog=True,
            )
            logger.info("Replay started in observer mode (no fixed player perspective)")
        else:
            # Player-perspective mode: preserve original behavior
            self._pipeline_loader.start_replay(
                controller,
                observed_player_id=observed_player_id,
                disable_fog=disable_fog,
            )
            logger.info(f"Replay started for player {observed_player_id}")

    def switch_player_perspective(self, controller, player_id: int) -> None:
        """
        Switch the observer's perspective to a specific player.

        Only valid when the replay was started with observer_mode=True.
        After switching, call controller.observe() to get data from the
        new player's perspective (player_common, score, upgrade_ids).

        Args:
            controller: SC2 controller instance (same one used in start_replay)
            player_id: The player ID to switch to (typically 1 or 2)

        Raises:
            ValueError: If player_id is not a positive integer.

        Depends on:
            - start_replay() must have been called with observer_mode=True
            - Delegates to PipelineReplayLoader.switch_player_perspective()
        """
        self._pipeline_loader.switch_player_perspective(controller, player_id)
        logger.debug(f"Switched observer perspective to player {player_id}")

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

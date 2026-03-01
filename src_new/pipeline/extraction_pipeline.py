"""
ReplayExtractionPipeline: Main pipeline for extracting ground truth from SC2 replays.

This component orchestrates all Phase 2 extraction components to process replays
end-to-end, from loading the replay to writing parquet files.

Supports observer mode: single-pass with per-player perspective switching.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from absl import flags

from ..extraction.replay_loader import ReplayLoader
from ..extraction.state_extractor import StateExtractor
from ..extraction.schema_manager import SchemaManager
from ..extraction.wide_table_builder import WideTableBuilder
from ..extraction.parquet_writer import ParquetWriter
from ..extractors.economy_extractor import load_economy_snapshots, get_economy_at_loop
from ..extraction.metadata_writer import build_metadata, save_metadata


logger = logging.getLogger(__name__)


class ReplayExtractionPipeline:
    """
    Main pipeline for extracting ground truth from SC2 replays.

    This class orchestrates the complete extraction workflow:
    1. Load replay with perfect information settings
    2. Build static schema columns from player names (economy, upgrades)
    3. Iterate through game loops in observer mode
    4. Extract state at each loop using per-player perspective switching
    5. Add entity columns (units, buildings) on-demand as they are first seen
    6. Build wide-format rows
    7. Collect messages
    8. Write parquet files

    Observer mode is the only supported processing path. It uses a single
    replay pass with per-player perspective switching to get correct
    economy/upgrade data for both players.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the extraction pipeline.

        Args:
            config: Optional configuration dictionary with keys:
                - show_cloaked (bool): Show cloaked units (default: True)
                - show_burrowed_shadows (bool): Show burrowed units (default: True)
                - show_placeholders (bool): Show queued buildings (default: True)
                - step_size (int): Game loops to step per iteration (default: 1)
                - compression (str): Parquet compression codec (default: 'snappy')
                - output_format (str): Output file naming format (default: 'standard')
        """
        self.config = config or {}

        # Initialize components
        self.replay_loader = ReplayLoader(config)
        self.state_extractor = StateExtractor()
        self.schema_manager = SchemaManager()
        self.wide_table_builder = None  # Created after schema is built
        self.parquet_writer = ParquetWriter(
            compression=self.config.get('compression', 'snappy')
        )

        # Pipeline configuration
        self.step_size = self.config.get('step_size', 1)

        logger.info("ReplayExtractionPipeline initialized")

    def process_replay(
        self,
        replay_path: Path,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Process single replay end-to-end.

        This is the main entry point for processing a replay. It orchestrates
        all extraction components and writes output parquet files.

        Args:
            replay_path: Path to .SC2Replay file
            output_dir: Directory for output files (default: data/processed)

        Returns:
            Processing result dictionary:
            {
                'success': bool,
                'replay_path': Path,
                'output_files': {
                    'game_state': Path,
                    'messages': Path,
                    'metadata': Path,
                },
                'metadata': dict,
                'stats': {
                    'total_loops': int,
                    'rows_written': int,
                    'messages_written': int,
                    'processing_time_seconds': float,
                },
                'error': str or None,
            }

        Raises:
            FileNotFoundError: If replay file doesn't exist
            ValueError: If replay is invalid

        # TODO: Test case - Process small replay end-to-end
        # TODO: Test case - Verify output files created
        # TODO: Test case - Validate output file naming
        # TODO: Test case - Check data integrity
        """
        import time

        replay_path = Path(replay_path)
        output_dir = Path(output_dir or 'data/processed')
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing replay: {replay_path.name}")
        logger.info(f"  Output directory: {output_dir}")

        start_time = time.time()

        # Initialize result
        result = {
            'success': False,
            'replay_path': replay_path,
            'output_files': {},
            'metadata': {},
            'stats': {},
            'error': None,
        }

        try:
            # Reset extractors for new replay
            self.state_extractor.reset()

            # Observer mode is the only supported processing path
            processing_result = self._observer_mode_processing(replay_path, output_dir)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Update result
            result['success'] = True
            result['output_files'] = processing_result['output_files']
            result['metadata'] = processing_result['metadata']
            result['stats'] = processing_result['stats']
            result['stats']['processing_time_seconds'] = processing_time

            logger.info(f"Successfully processed replay in {processing_time:.2f}s")
            logger.info(f"  Total loops: {result['stats']['total_loops']}")
            logger.info(f"  Rows written: {result['stats']['rows_written']}")
            logger.info(f"  Messages: {result['stats']['messages_written']}")

        except Exception as e:
            processing_time = time.time() - start_time
            result['error'] = str(e)
            result['stats']['processing_time_seconds'] = processing_time
            logger.error(f"Failed to process replay: {e}", exc_info=True)

        return result

    def _observer_mode_processing(
        self,
        replay_path: Path,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """
        Observer mode: single replay pass with per-player perspective switching.

        This is the only supported processing mode. The replay is started without
        a fixed player perspective (observer mode). At each game step:
        1. Step the replay forward one unit
        2. Switch to P1 perspective and observe (P1 economy, units, buildings)
        3. Switch to P2 perspective and observe (P2 economy, upgrades)
        4. Extract complete state combining both observations
        5. Add columns for any new entities seen this frame (dynamic schema)
        6. Build wide-format row and append to rows list

        Schema is built in two stages:
        - Static columns (economy, upgrades) are added once via build_base_schema()
          after player names are known from replay metadata.
        - Entity columns (units, buildings) are added on-demand via ensure_unit_columns()
          and ensure_building_columns() the first time each entity is encountered.

        Args:
            replay_path: Path to .SC2Replay file
            output_dir: Directory for parquet/json output files

        Returns:
            Dict with keys: output_files, metadata, stats

        Depends on / calls:
            - load_economy_snapshots() -- pre-parses s2protocol tracker events
            - get_economy_at_loop() -- forward-fills economy into each row
            - replay_loader.load_replay()
            - replay_loader.start_sc2_instance()
            - replay_loader.get_replay_info()
            - schema_manager.build_base_schema()
            - WideTableBuilder(schema_manager)
            - wide_table_builder.set_player_names()
            - replay_loader.start_replay(observer_mode=True)
            - replay_loader.switch_player_perspective()
            - state_extractor.extract_observation_observer_mode()
            - schema_manager.ensure_unit_columns()
            - schema_manager.ensure_building_columns()
            - wide_table_builder.build_row()
            - parquet_writer.write_game_state()
            - parquet_writer.write_messages()
            - build_metadata() + save_metadata() from metadata_writer
        """
        logger.info("Starting observer mode processing")

        # Load replay before opening SC2 instance — this is the ONLY load,
        # no pre-scan pass is needed.
        self.replay_loader.load_replay(replay_path)

        # Parse economy snapshots from the replay file via s2protocol BEFORE
        # launching the SC2 engine. This avoids the observer-mode bug where
        # the engine returns all-zero economy data (player_common/score_details
        # are empty when observed_player_id=0). The snapshots are emitted by
        # the replay tracker at ~160 game-loop intervals; we forward-fill them
        # into each row inside the game loop below.
        economy_snapshots = load_economy_snapshots(str(replay_path))

        # Log how many snapshots were loaded per player so operators can
        # verify that the replay contains economy data before the loop starts.
        for pid in sorted(economy_snapshots.keys()):
            snap_count = len(economy_snapshots[pid])
            logger.info(
                f"Economy snapshots loaded: player {pid} has {snap_count} snapshot(s)"
            )

        rows = []
        all_messages = []

        with self.replay_loader.start_sc2_instance() as controller:
            # Read replay metadata: player names, game duration, etc.
            metadata = self.replay_loader.get_replay_info(controller)

            # Extract player names — available from replay file header,
            # no replay scan required.
            player_names = {
                p['player_id']: p.get('player_name', '')
                for p in metadata.get('players', [])
            }

            # Build static schema columns: game_loop, economy, upgrades.
            # Entity columns (units, buildings) are added dynamically below.
            self.schema_manager.build_base_schema(player_names)

            # Create WideTableBuilder now that static schema columns exist.
            # WideTableBuilder reads schema.get_column_list() on each build_row() call,
            # so columns added dynamically later are automatically included.
            self.wide_table_builder = WideTableBuilder(self.schema_manager)
            self.wide_table_builder.set_player_names(player_names)

            # Start replay in observer mode — no fixed player perspective.
            self.replay_loader.start_replay(controller, observer_mode=True)

            game_loop = 0
            max_loops = metadata['game_duration_loops']
            # Report progress approximately every 5%
            progress_interval = max(1, max_loops // 20)

            logger.info(
                f"Observer mode: Processing {max_loops} game loops "
                f"(step size: {self.step_size})..."
            )

            while game_loop < max_loops:
                try:
                    # Step replay forward one unit
                    controller.step(self.step_size)

                    # Get P1-perspective observation (units, buildings, P1 economy)
                    self.replay_loader.switch_player_perspective(controller, player_id=1)
                    obs_p1 = controller.observe()

                    # player_result is populated when the game ends
                    if obs_p1.player_result:
                        logger.info(f"Observer mode: Replay ended at loop {game_loop}")
                        break

                    game_loop = obs_p1.observation.game_loop

                    # Get P2-perspective observation (P2 economy, P2 upgrades)
                    self.replay_loader.switch_player_perspective(controller, player_id=2)
                    obs_p2 = controller.observe()

                    # Combine both perspectives into a single state dict
                    state = self.state_extractor.extract_observation_observer_mode(
                        obs_p1, obs_p2, game_loop
                    )

                    # Inject economy data from s2protocol snapshots.
                    # state_extractor no longer populates p1_economy / p2_economy
                    # because the SC2 engine returns zeroed economy in observer mode.
                    # Instead we forward-fill from the pre-parsed tracker snapshots:
                    # get_economy_at_loop() returns the most recent snapshot at or
                    # before this game_loop via O(log n) binary search, giving each
                    # row accurate economy values without needing per-player engine obs.
                    state['p1_economy'] = get_economy_at_loop(economy_snapshots, 1, game_loop)
                    state['p2_economy'] = get_economy_at_loop(economy_snapshots, 2, game_loop)

                    # --- Dynamic column creation ---
                    # Add unit columns the first time each completed unit is seen.
                    # Units still under construction (lifecycle != 'completed') do not
                    # get columns until they finish — matching the pre-scan rule.
                    for player_num in [1, 2]:
                        player = f'p{player_num}'
                        units = state.get(f'{player}_units', {})
                        for readable_id, unit_data in units.items():
                            if unit_data.get('_lifecycle') == 'completed':
                                extra_attrs = (
                                    self.state_extractor
                                        .unit_extractors[player_num]
                                        .get_unit_attributes_for_id(readable_id)
                                )
                                self.schema_manager.ensure_unit_columns(
                                    player, readable_id, extra_attrs
                                )

                    # Add building columns on first appearance, regardless of lifecycle.
                    # Even cancelled buildings have meaningful position data.
                    for player_num in [1, 2]:
                        player = f'p{player_num}'
                        buildings = state.get(f'{player}_buildings', {})
                        for readable_id, building_data in buildings.items():
                            extra_attrs = (
                                self.state_extractor
                                    .building_extractors[player_num]
                                    .get_building_attributes_for_id(readable_id)
                            )
                            self.schema_manager.ensure_building_columns(
                                player, readable_id, extra_attrs
                            )

                    # Build wide-format row — schema is now complete for all entities
                    # seen so far; future entities will get columns in later iterations.
                    row = self.wide_table_builder.build_row(state)
                    rows.append(row)

                    messages = state.get('messages', [])
                    all_messages.extend(messages)

                    if game_loop % progress_interval == 0:
                        progress = (game_loop / max_loops) * 100
                        logger.info(
                            f"  Observer mode progress: {progress:.1f}% "
                            f"(loop {game_loop}/{max_loops})"
                        )

                except Exception as e:
                    logger.warning(
                        f"Error at game loop {game_loop} (observer mode): {e}"
                    )
                    continue

            logger.info(
                f"Observer mode complete. Extracted {len(rows)} rows, "
                f"{len(all_messages)} messages"
            )

        # --- Write output files ---
        replay_name = replay_path.stem
        parquet_dir = output_dir / 'parquet'
        json_dir = output_dir / 'json'
        parquet_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        output_files = {
            'game_state': parquet_dir / f"{replay_name}_game_state.parquet",
            'messages': parquet_dir / f"{replay_name}_messages.parquet",
            'metadata': json_dir / f"{replay_name}_metadata.json",
        }

        logger.info(f"Writing game state to {output_files['game_state']}")
        self.parquet_writer.write_game_state(
            rows, output_files['game_state'], self.schema_manager
        )

        if all_messages:
            logger.info(f"Writing messages to {output_files['messages']}")
            self.parquet_writer.write_messages(all_messages, output_files['messages'])
        else:
            logger.info("No messages to write")

        # Build and write comprehensive metadata JSON that makes the
        # parquet file self-documenting for dataset consumers.
        parquet_filename = f"{replay_name}_game_state.parquet"
        metadata_dict = build_metadata(
            metadata=metadata,
            columns=self.schema_manager.columns,
            total_rows=len(rows),
            parquet_filename=parquet_filename,
            all_messages=all_messages,
        )
        logger.info(f"Writing metadata to {output_files['metadata']}")
        save_metadata(metadata_dict, output_files['metadata'])

        return {
            'output_files': output_files,
            'metadata': metadata,
            'stats': {
                'total_loops': max_loops,
                'rows_written': len(rows),
                'messages_written': len(all_messages),
            },
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Get current pipeline configuration.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Update pipeline configuration.

        Args:
            config: New configuration dictionary
        """
        self.config.update(config)

        # Update step size if changed
        if 'step_size' in config:
            self.step_size = config['step_size']
            logger.info(f"Step size updated to: {self.step_size}")

    def validate_replay(self, replay_path: Path) -> Dict[str, Any]:
        """
        Validate replay without full processing.

        Quick check to see if replay can be loaded and basic info extracted.

        Args:
            replay_path: Path to replay file

        Returns:
            Validation result dictionary:
            {
                'valid': bool,
                'metadata': dict or None,
                'error': str or None,
            }
        """
        result = {
            'valid': False,
            'metadata': None,
            'error': None,
        }

        try:
            # Try to load replay
            self.replay_loader.load_replay(replay_path)

            # Try to extract metadata
            with self.replay_loader.start_sc2_instance() as controller:
                metadata = self.replay_loader.get_replay_info(controller)
                result['metadata'] = metadata
                result['valid'] = True

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Replay validation failed: {e}")

        return result


# Convenience function for quick processing
def process_replay_quick(
    replay_path: Path,
    output_dir: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to process a replay with default settings.

    Args:
        replay_path: Path to .SC2Replay file
        output_dir: Output directory (default: data/processed)
        config: Optional configuration dictionary

    Returns:
        Processing result dictionary

    Example:
        >>> result = process_replay_quick(Path("replay.SC2Replay"))
        >>> if result['success']:
        >>>     print(f"Processed {result['stats']['rows_written']} rows")
    """
    # Initialize absl flags if not already parsed (required for pysc2)
    if not flags.FLAGS.is_parsed():
        flags.FLAGS.mark_as_parsed()
    if config == None:
        config = {
            # Observation settings
            'show_cloaked': True,
            'show_burrowed_shadows': True,
            'show_placeholders': True,

            # Processing settings
            'step_size': 1,  # Game loops per step

            # Output settings
            'compression': 'snappy',  # options: 'snappy' 'gzip', 'brotli', 'zstd'
        }

    pipeline = ReplayExtractionPipeline(config)
    return pipeline.process_replay(replay_path, output_dir)

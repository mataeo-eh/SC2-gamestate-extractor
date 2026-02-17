"""
ReplayExtractionPipeline: Main pipeline for extracting ground truth from SC2 replays.

This component orchestrates all Phase 2 extraction components to process replays
end-to-end, from loading the replay to writing parquet files.

Supports three processing modes:
- 'observer' (default, preferred): Single-pass observer mode. The replay is started
  without a fixed player perspective, and at each game step the pipeline switches
  perspective to each player to get correct per-player economy/upgrades data.
- 'two_pass': Legacy fallback. Runs the replay twice — once from P1's perspective
  for full extraction, then again from P2's perspective to patch P2 economy/upgrades.
- 'single_pass': Dynamic schema mode with a single extraction pass. More memory-
  efficient but may produce ragged columns.
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


logger = logging.getLogger(__name__)


class ReplayExtractionPipeline:
    """
    Main pipeline for extracting ground truth from SC2 replays.

    This class orchestrates the complete extraction workflow:
    1. Load replay with perfect information settings
    2. Build schema (pre-scan pass or dynamic)
    3. Iterate through game loops
    4. Extract state at each loop (observer mode uses per-player perspective switching)
    5. Build wide-format rows
    6. Collect messages
    7. Write parquet files

    The preferred processing mode is 'observer', which uses a single replay pass
    with per-player perspective switching to get correct economy/upgrade data for
    both players. The legacy 'two_pass' mode is retained as a fallback.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the extraction pipeline.

        Args:
            config: Optional configuration dictionary with keys:
                - show_cloaked (bool): Show cloaked units (default: True)
                - show_burrowed_shadows (bool): Show burrowed units (default: True)
                - show_placeholders (bool): Show queued buildings (default: True)
                - processing_mode (str): 'observer', 'two_pass', or 'single_pass'
                    (default: 'observer')
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

        # Pipeline configuration — 'observer' is preferred over legacy 'two_pass'
        self.processing_mode = self.config.get('processing_mode', 'observer')
        self.step_size = self.config.get('step_size', 1)

        logger.info(f"ReplayExtractionPipeline initialized (mode: {self.processing_mode})")

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
                    'schema': Path,
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
        logger.info(f"  Processing mode: {self.processing_mode}")

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

            # Choose processing mode
            if self.processing_mode == 'observer':
                processing_result = self._observer_mode_processing(replay_path, output_dir)
            elif self.processing_mode == 'two_pass':
                processing_result = self._two_pass_processing(replay_path, output_dir)
            elif self.processing_mode == 'single_pass':
                processing_result = self._single_pass_processing(replay_path, output_dir)
            else:
                raise ValueError(f"Invalid processing mode: {self.processing_mode}")

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

        This is the preferred processing mode. The replay is started WITHOUT
        observed_player_id (observer mode), and at each game step:
        1. Step the replay forward
        2. Switch perspective to player 1, observe -> get P1 economy/score/upgrades
        3. Switch perspective to player 2, observe -> get P2 economy/score/upgrades
        4. Units and buildings come from either observation (identical in both)
        5. Build wide-format row and append

        This replaces the two-pass approach and gives correct per-player economy
        data without re-running the replay.

        The schema is still built via a separate pre-scan pass (using the legacy
        player-perspective mode internally in SchemaManager), then the actual
        data extraction happens in observer mode.

        Args:
            replay_path: Path to replay file
            output_dir: Output directory

        Returns:
            Processing result dictionary with output_files, metadata, and stats

        Depends on / calls:
            - schema_manager.build_schema_from_replay() for schema pre-scan
            - replay_loader.load_replay(), start_sc2_instance(), get_replay_info()
            - replay_loader.start_replay(observer_mode=True)
            - replay_loader.switch_player_perspective() for per-player observation
            - state_extractor.extract_observation_observer_mode()
            - wide_table_builder.build_row()
            - parquet_writer.write_game_state(), write_messages()
            - schema_manager.save_schema()
        """
        logger.info("Starting observer mode processing")

        # --- Schema pre-scan ---
        # build_schema_from_replay uses legacy player-perspective mode internally.
        # It only needs to discover entity IDs, not extract accurate economy data.
        logger.info("Schema scan: Building schema from replay...")
        self.schema_manager.build_schema_from_replay(
            replay_path,
            self.replay_loader,
            self.state_extractor,
        )

        # Create wide table builder with the completed schema
        self.wide_table_builder = WideTableBuilder(self.schema_manager)

        # Reset only per-frame state between schema scan and extraction pass,
        # preserving tag-to-ID mappings so the same readable IDs are reused.
        self.state_extractor.reset_frame_state()

        # --- Observer mode extraction pass ---
        # Load the replay fresh for the extraction pass
        self.replay_loader.load_replay(replay_path)

        # Storage for extracted data
        rows = []
        all_messages = []

        with self.replay_loader.start_sc2_instance() as controller:
            # Get replay metadata
            metadata = self.replay_loader.get_replay_info(controller)

            # Extract player names and pass to wide_table_builder and schema_manager
            player_names = {
                p['player_id']: p.get('player_name', '')
                for p in metadata.get('players', [])
            }
            if self.wide_table_builder is not None:
                self.wide_table_builder.set_player_names(player_names)
            self.schema_manager.set_player_names(player_names)

            # Start replay in observer mode — no observed_player_id is passed,
            # and disable_fog is forced True internally.
            self.replay_loader.start_replay(controller, observer_mode=True)

            # Process each game loop
            game_loop = 0
            max_loops = metadata['game_duration_loops']

            logger.info(
                f"Observer mode: Processing {max_loops} game loops "
                f"(step size: {self.step_size})..."
            )

            # Track progress — report approximately every 5%
            progress_interval = max(1, max_loops // 20)

            while game_loop < max_loops:
                try:
                    # Step forward
                    controller.step(self.step_size)

                    # Switch to player 1 perspective and observe
                    self.replay_loader.switch_player_perspective(controller, player_id=1)
                    obs_p1 = controller.observe()

                    # Check if replay has ended (player_result is populated when game ends)
                    if obs_p1.player_result:
                        logger.info(
                            f"Observer mode: Replay ended at loop {game_loop} "
                            f"(expected {max_loops})"
                        )
                        break

                    # Update game loop from the observation
                    game_loop = obs_p1.observation.game_loop

                    # Switch to player 2 perspective and observe
                    self.replay_loader.switch_player_perspective(controller, player_id=2)
                    obs_p2 = controller.observe()

                    # Extract complete state using observer mode method —
                    # uses obs_p1 for units/buildings/P1 economy/P1 upgrades/messages
                    # and obs_p2 for P2 economy/P2 upgrades
                    state = self.state_extractor.extract_observation_observer_mode(
                        obs_p1, obs_p2, game_loop
                    )

                    # Build wide-format row
                    row = self.wide_table_builder.build_row(state)
                    rows.append(row)

                    # Collect messages
                    messages = state.get('messages', [])
                    all_messages.extend(messages)

                    # Progress reporting
                    if game_loop % progress_interval == 0:
                        progress = (game_loop / max_loops) * 100
                        logger.info(
                            f"  Observer mode progress: {progress:.1f}% "
                            f"(loop {game_loop}/{max_loops})"
                        )

                except Exception as e:
                    logger.warning(f"Error at game loop {game_loop} (observer mode): {e}")
                    # Continue processing — don't fail entire replay for one frame
                    continue

            logger.info(
                f"Observer mode complete. Extracted {len(rows)} rows, "
                f"{len(all_messages)} messages"
            )

        # --- Write output files ---
        # Generate output file paths with directory structure
        replay_name = replay_path.stem
        parquet_dir = output_dir / 'parquet'
        json_dir = output_dir / 'json'

        # Create directories if they don't exist
        parquet_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        output_files = {
            'game_state': parquet_dir / f"{replay_name}_game_state.parquet",
            'messages': parquet_dir / f"{replay_name}_messages.parquet",
            'schema': json_dir / f"{replay_name}_schema.json",
        }

        # Write game state parquet
        logger.info(f"Writing game state to {output_files['game_state']}")
        self.parquet_writer.write_game_state(
            rows,
            output_files['game_state'],
            self.schema_manager,
        )

        # Write messages parquet (if any)
        if all_messages:
            logger.info(f"Writing messages to {output_files['messages']}")
            self.parquet_writer.write_messages(
                all_messages,
                output_files['messages'],
            )
        else:
            logger.info("No messages to write")

        # Write schema JSON
        logger.info(f"Writing schema to {output_files['schema']}")
        self.schema_manager.save_schema(output_files['schema'])

        # Return result
        return {
            'output_files': output_files,
            'metadata': metadata,
            'stats': {
                'total_loops': max_loops,
                'rows_written': len(rows),
                'messages_written': len(all_messages),
            },
        }

    def _two_pass_processing(
        self,
        replay_path: Path,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Two-pass approach for consistent schema (legacy fallback).

        NOTE: The preferred processing mode is 'observer', which achieves the
        same result in a single replay pass using perspective switching. This
        two-pass mode is retained as a fallback in case observer mode is not
        supported by a specific SC2 version.

        Pass 1: Scan replay to build complete schema
        Pass 2: Extract and write data with consistent schema

        This ensures all rows have the same columns, which is better for
        downstream ML pipelines, but requires processing the replay twice.

        Args:
            replay_path: Path to replay file
            output_dir: Output directory

        Returns:
            Processing result dictionary with output_files, metadata, and stats

        # TODO: Test case - Two-pass processing produces consistent schema
        # TODO: Test case - Verify no missing columns across rows
        """
        logger.info("Starting two-pass processing (legacy fallback)")

        # PASS 1: Build schema
        logger.info("Pass 1: Building schema...")
        self.schema_manager.build_schema_from_replay(
            replay_path,
            self.replay_loader,
            self.state_extractor
        )

        # Create wide table builder with schema
        self.wide_table_builder = WideTableBuilder(self.schema_manager)

        # Reset only per-frame state between passes, preserving tag-to-ID mappings
        # so that pass 2 uses the same readable IDs as pass 1 (schema scan).
        # This prevents column name mismatches between schema and data.
        self.state_extractor.reset_frame_state()

        # PASS 2: Extract data
        logger.info("Pass 2: Extracting data...")
        return self._extract_and_write(replay_path, output_dir)

    def _single_pass_processing(
        self,
        replay_path: Path,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Single-pass approach with dynamic schema.

        Extract and write data in one pass, building schema dynamically
        as new entities are discovered. More memory-efficient but may
        result in ragged columns (different rows may have different columns).

        Args:
            replay_path: Path to replay file
            output_dir: Output directory

        Returns:
            Processing result dictionary with output_files, metadata, and stats

        # TODO: Test case - Single-pass processing completes successfully
        # TODO: Test case - Verify dynamic schema updates during processing
        """
        logger.info("Starting single-pass processing")

        # Create wide table builder with empty schema
        # Schema will be built dynamically during extraction
        self.wide_table_builder = WideTableBuilder(self.schema_manager)

        return self._extract_and_write(replay_path, output_dir)

    def _extract_and_write(
        self,
        replay_path: Path,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Extract game state and write to parquet files.

        This performs the actual iteration through the replay, state extraction,
        and parquet writing.  It runs the replay twice:
          - Pass A (observed_player_id=1): full extraction (units, buildings,
            P1 economy, P1 upgrades, messages).  P2 economy/upgrades from this
            pass are WRONG (they reflect P1's perspective) and will be overwritten.
          - Pass B (observed_player_id=2): extracts ONLY P2 economy and P2
            upgrades, then patches them into the rows built during Pass A.

        Args:
            replay_path: Path to replay file
            output_dir: Output directory

        Returns:
            Processing result dictionary
        """
        # Load replay
        self.replay_loader.load_replay(replay_path)

        # Storage for extracted data
        rows = []
        all_messages = []

        # Start SC2 instance and process replay
        with self.replay_loader.start_sc2_instance() as controller:
            # Get replay metadata
            metadata = self.replay_loader.get_replay_info(controller)

            # Extract player names and pass to wide_table_builder
            player_names = {
                p['player_id']: p.get('player_name', '')
                for p in metadata.get('players', [])
            }
            if self.wide_table_builder is not None:
                self.wide_table_builder.set_player_names(player_names)
            # Also set on schema_manager (needed for single-pass mode where
            # schema building happens dynamically during extraction)
            self.schema_manager.set_player_names(player_names)

            # ------------------------------------------------------------------
            # Pass A: observed_player_id=1  (full extraction)
            # P1 economy/upgrades are correct.  P2 economy/upgrades are wrong
            # (perspective-dependent data reflects P1).  Units and buildings for
            # BOTH players are correct because they use unit.owner filtering
            # with disable_fog=True.
            # ------------------------------------------------------------------
            self.replay_loader.start_replay(controller, observed_player_id=1, disable_fog=True)

            # Process each game loop
            game_loop = 0
            max_loops = metadata['game_duration_loops']

            logger.info(f"Pass A (player 1 perspective): Processing {max_loops} game loops (step size: {self.step_size})...")

            # Track progress
            progress_interval = max(1, max_loops // 20)  # Report every 5%

            while game_loop < max_loops:
                try:
                    # Step forward
                    controller.step(self.step_size)
                    obs = controller.observe()

                    # Check if replay has ended
                    from pysc2.lib.protocol import Status
                    if obs.player_result:  # player_result is populated when game ends
                        logger.info(f"Pass A: Replay ended at loop {game_loop} (expected {max_loops})")
                        break # Break while loop after game ended

                    # Update game loop
                    game_loop = obs.observation.game_loop

                    # Extract state
                    state = self.state_extractor.extract_observation(obs, game_loop)

                    # In single-pass mode, update schema dynamically
                    if self.processing_mode == 'single_pass':
                        self.schema_manager._discover_entities_from_state(state)

                    # Build wide-format row
                    row = self.wide_table_builder.build_row(state)
                    rows.append(row)

                    # Collect messages
                    messages = state.get('messages', [])
                    all_messages.extend(messages)

                    # Progress reporting
                    if game_loop % progress_interval == 0:
                        progress = (game_loop / max_loops) * 100
                        logger.info(f"  Pass A progress: {progress:.1f}% (loop {game_loop}/{max_loops})")

                except Exception as e:
                    logger.warning(f"Error at game loop {game_loop} (Pass A): {e}")
                    # Continue processing - don't fail entire replay for one frame
                    continue

            logger.info(f"Pass A complete. Extracted {len(rows)} rows, {len(all_messages)} messages")

            # ------------------------------------------------------------------
            # Pass B: observed_player_id=2  (P2 economy + upgrades only)
            # Re-play the replay from player 2's perspective to get correct
            # player_common and score_details for P2.  Overwrites the incorrect
            # P2 economy/upgrade columns that were filled during Pass A.
            # ------------------------------------------------------------------
            self._patch_p2_economy(controller, rows, max_loops)

        # Generate output file paths with new directory structure
        replay_name = replay_path.stem
        parquet_dir = output_dir / 'parquet'
        json_dir = output_dir / 'json'

        # Create directories if they don't exist
        parquet_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        output_files = {
            'game_state': parquet_dir / f"{replay_name}_game_state.parquet",
            'messages': parquet_dir / f"{replay_name}_messages.parquet",
            'schema': json_dir / f"{replay_name}_schema.json",
        }

        # Write game state parquet
        logger.info(f"Writing game state to {output_files['game_state']}")
        self.parquet_writer.write_game_state(
            rows,
            output_files['game_state'],
            self.schema_manager
        )

        # Write messages parquet (if any)
        if all_messages:
            logger.info(f"Writing messages to {output_files['messages']}")
            self.parquet_writer.write_messages(
                all_messages,
                output_files['messages']
            )
        else:
            logger.info("No messages to write")

        # Write schema JSON
        logger.info(f"Writing schema to {output_files['schema']}")
        self.schema_manager.save_schema(output_files['schema'])

        # Return result
        return {
            'output_files': output_files,
            'metadata': metadata,
            'stats': {
                'total_loops': max_loops,
                'rows_written': len(rows),
                'messages_written': len(all_messages),
            },
        }

    def _patch_p2_economy(
        self,
        controller,
        rows: List[Dict[str, Any]],
        max_loops: int,
    ) -> None:
        """
        Run a second replay pass from player 2's perspective to fix P2 economy and upgrades.

        During Pass A (observed_player_id=1), the P2 economy and upgrade columns
        were filled with P1's perspective-dependent data (player_common,
        score_details, raw_data.player.upgrade_ids all reflect the observed
        player).  This method replays from P2's perspective and overwrites those
        columns with correct values.

        Rows are matched by game_loop value (not by index) to handle any
        differences in step counts between passes.

        Args:
            controller: Active SC2 controller (still within the context manager)
            rows: List of row dicts from Pass A — modified in place
            max_loops: Total game duration in loops
        """
        logger.info("Pass B (player 2 perspective): Extracting P2 economy and upgrades...")

        # Build a lookup from game_loop -> row for efficient patching
        row_by_game_loop: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            gl = row.get('game_loop')
            if gl is not None:
                row_by_game_loop[gl] = row

        # Reset P2 upgrade extractor state so it tracks upgrades fresh for the
        # new perspective.  P2 economy extractor is stateless (no reset needed).
        self.state_extractor.upgrade_extractors[2].reset()

        # Start replay from player 2's perspective
        self.replay_loader.start_replay(controller, observed_player_id=2, disable_fog=True)

        game_loop = 0
        patched_count = 0
        progress_interval = max(1, max_loops // 20)

        # The P2 economy columns that will be overwritten in each row
        p2_economy_columns = [
            'p2_minerals', 'p2_vespene',
            'p2_supply_used', 'p2_supply_cap',
            'p2_workers', 'p2_idle_workers',
        ]

        while game_loop < max_loops:
            try:
                controller.step(self.step_size)
                obs = controller.observe()

                if obs.player_result:
                    logger.info(f"Pass B: Replay ended at loop {game_loop} (expected {max_loops})")
                    break

                game_loop = obs.observation.game_loop

                # Extract only P2 economy + upgrades from this observation
                p2_state = self.state_extractor.extract_perspective_dependent(
                    obs, game_loop, observed_player_id=2
                )

                # Patch the matching row from Pass A
                target_row = row_by_game_loop.get(game_loop)
                if target_row is not None:
                    # Overwrite P2 economy columns
                    p2_economy = p2_state.get('p2_economy', {})
                    for col in p2_economy_columns:
                        attr = col.replace('p2_', '', 1)  # e.g. 'p2_minerals' -> 'minerals'
                        if attr in p2_economy:
                            target_row[col] = p2_economy[attr]

                    # Overwrite P2 upgrade columns
                    # The wide_table_builder maps upgrades via add_upgrades_to_row;
                    # replicate its logic here to patch upgrade columns directly.
                    p2_upgrades = p2_state.get('p2_upgrades', {})
                    self.wide_table_builder.add_upgrades_to_row(
                        target_row, 'p2', p2_upgrades
                    )

                    patched_count += 1

                # Progress reporting
                if game_loop % progress_interval == 0:
                    progress = (game_loop / max_loops) * 100
                    logger.info(f"  Pass B progress: {progress:.1f}% (loop {game_loop}/{max_loops})")

            except Exception as e:
                logger.warning(f"Error at game loop {game_loop} (Pass B): {e}")
                continue

        logger.info(f"Pass B complete. Patched P2 economy/upgrades in {patched_count}/{len(rows)} rows")

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

        # Update processing mode if changed
        if 'processing_mode' in config:
            self.processing_mode = config['processing_mode']
            logger.info(f"Processing mode updated to: {self.processing_mode}")

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
            'processing_mode': 'observer',  # preferred; fallback: 'two_pass'
            'step_size': 1,  # Game loops per step

            # Output settings
            'compression': 'snappy',  # options: 'snappy' 'gzip', 'brotli', 'zstd'
        }

    pipeline = ReplayExtractionPipeline(config)
    return pipeline.process_replay(replay_path, output_dir)

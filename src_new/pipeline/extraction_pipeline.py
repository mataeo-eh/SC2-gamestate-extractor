"""
ReplayExtractionPipeline: Main pipeline for extracting ground truth from SC2 replays.

This component orchestrates all Phase 2 extraction components to process replays
end-to-end, from loading the replay to writing parquet files.
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
    2. Build schema (one-pass or two-pass mode)
    3. Iterate through game loops
    4. Extract state at each loop
    5. Build wide-format rows
    6. Collect messages
    7. Write parquet files
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the extraction pipeline.

        Args:
            config: Optional configuration dictionary with keys:
                - show_cloaked (bool): Show cloaked units (default: True)
                - show_burrowed_shadows (bool): Show burrowed units (default: True)
                - show_placeholders (bool): Show queued buildings (default: True)
                - processing_mode (str): 'two_pass' or 'single_pass' (default: 'two_pass')
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
        self.processing_mode = self.config.get('processing_mode', 'two_pass')
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
            if self.processing_mode == 'two_pass':
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

    def _two_pass_processing(
        self,
        replay_path: Path,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Two-pass approach for consistent schema.

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
        logger.info("Starting two-pass processing")

        # PASS 1: Build schema
        logger.info("Pass 1: Building schema...")
        self.schema_manager.build_schema_from_replay(
            replay_path,
            self.replay_loader,
            self.state_extractor
        )

        # Create wide table builder with schema
        self.wide_table_builder = WideTableBuilder(self.schema_manager)

        # Reset state extractor after schema building
        self.state_extractor.reset()

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
        and parquet writing.

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

            # Start replay playback
            self.replay_loader.start_replay(controller, observed_player_id=1, disable_fog=True)

            # Process each game loop
            game_loop = 0
            max_loops = metadata['game_duration_loops']

            logger.info(f"Processing {max_loops} game loops (step size: {self.step_size})...")

            # Track progress
            progress_interval = max(1, max_loops // 20)  # Report every 5%

            while game_loop < max_loops:
                try:
                    # Step forward
                    controller.step(self.step_size)
                    obs = controller.observe()

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
                        logger.info(f"  Progress: {progress:.1f}% (loop {game_loop}/{max_loops})")

                except Exception as e:
                    logger.warning(f"Error at game loop {game_loop}: {e}")
                    # Continue processing - don't fail entire replay for one frame
                    continue

            logger.info(f"Extraction complete. Extracted {len(rows)} rows, {len(all_messages)} messages")

        # Generate output file paths
        replay_name = replay_path.stem
        output_files = {
            'game_state': output_dir / f"{replay_name}_game_state.parquet",
            'messages': output_dir / f"{replay_name}_messages.parquet",
            'schema': output_dir / f"{replay_name}_schema.json",
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

    pipeline = ReplayExtractionPipeline(config)
    return pipeline.process_replay(replay_path, output_dir)

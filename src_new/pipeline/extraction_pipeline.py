"""
ReplayExtractionPipeline: Main pipeline for extracting ground truth from SC2 replays.

This component orchestrates all Phase 2 extraction components to process replays
end-to-end, from loading the replay to writing parquet files.

Supports observer mode: single-pass with per-player perspective switching.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

import numpy as np
from absl import flags

from ..extraction.replay_loader import ReplayLoader
from ..extraction.state_extractor import StateExtractor
from ..extraction.schema_manager import SchemaManager
from ..extraction.wide_table_builder import WideTableBuilder
from ..extraction.parquet_writer import ParquetWriter
from ..extractors.economy_extractor import load_economy_snapshots
from ..extractors.upgrade_extractor import load_upgrade_snapshots
from ..extraction.metadata_writer import build_metadata, save_metadata


logger = logging.getLogger(__name__)


class ReplayExtractionPipeline:
    """
    Main pipeline for extracting ground truth from SC2 replays.

    This class orchestrates the complete extraction workflow:
    1. Load replay with perfect information settings
    2. Build static schema columns from player names (units and buildings only)
    3. Iterate through game loops in observer mode — single observe() per loop
    4. Add entity columns (units, buildings) on-demand as they are first seen
    5. Flush rows to part files in batches; stitch parts into final parquet
    6. Post-process the final parquet: read replay tracker events and append
       economy columns (forward-filled from SPlayerStatsEvent) and upgrade
       columns (p1_upgrades / p2_upgrades, cumulative chronological list of
       upgrade names completed at or before each game_loop; NaN before the
       first completion)
    7. Write metadata JSON

    Economy and upgrades are deliberately excluded from the hot extraction
    loop and added in a single post-processing pass after stitching. This
    keeps the core loop focused on unit/building state and avoids per-loop
    binary-search overhead.
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
                    'metadata': Path,
                },
                'metadata': dict,
                'stats': {
                    'total_loops': int,
                    'rows_written': int,
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
        Observer mode: single replay pass with one observation per game step.

        This is the only supported processing mode. The replay is started without
        a fixed player perspective (observer mode). At each game step:
        1. Step the replay forward one unit
        2. Observe once — raw_data.units is perspective-invariant in observer mode
        3. Extract unit and building state (economy and upgrades are intentionally
           excluded from this loop — they are added in a post-processing pass after
           all batches are stitched)
        4. Add entity columns for any new units/buildings seen this frame
        5. Build a wide-format row and flush to part files in batches

        Perspective switching is intentionally NOT used. Diagnostic testing
        confirmed that in observer mode, raw_data.player.upgrade_ids is always
        empty regardless of which player perspective is active, and raw_data.units
        is identical across all perspectives. One observe() per loop (down from
        five API calls: step + switch + obs + switch + obs) gives a ~60% reduction
        in SC2 engine round-trips per replay.

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
            - replay_loader.load_replay()
            - replay_loader.start_sc2_instance()
            - replay_loader.get_replay_info()
            - schema_manager.build_base_schema()
            - WideTableBuilder(schema_manager)
            - wide_table_builder.set_player_names()
            - replay_loader.start_replay(observer_mode=True)
            - state_extractor.extract_observation_observer_mode()
            - schema_manager.ensure_unit_columns()
            - schema_manager.ensure_building_columns()
            - wide_table_builder.build_row()
            - parquet_writer.reconcile_parts()
            - _add_tracker_columns() -- post-processing: economy + upgrades
            - build_metadata() + save_metadata() from metadata_writer
        """
        logger.info("Starting observer mode processing")

        # Load replay before opening SC2 instance — this is the ONLY load,
        # no pre-scan pass is needed.
        self.replay_loader.load_replay(replay_path)

        # --- Batch streaming accumulation ---
        # Instead of accumulating the entire replay in memory, we collect rows
        # into column_data and flush to a parquet part file every BATCH_SIZE
        # rows. This caps peak memory at BATCH_SIZE * column_count instead of
        # total_rows * column_count. After the game loop, reconcile_parts()
        # merges all part files into a single parquet with a unified schema
        # (PyArrow fills missing columns in earlier parts with null).
        BATCH_SIZE = 2000
        column_data: Dict[str, list] = {}
        batch_row_count = 0   # rows in current batch
        row_count = 0         # total rows across all batches
        part_index = 0        # current part file number
        all_messages = []

        # Parts directory sits next to the final output file and is cleaned up
        # after reconciliation.
        replay_name = replay_path.stem
        parquet_dir = output_dir / 'parquet'
        parts_dir = parquet_dir / f"_parts_{replay_name}"

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

            # --- FN-1: Build API-derived building type ID set from data_raw() ---
            # After start_replay(), call controller.data_raw() to get all
            # UnitTypeData entries. Build a frozenset of integer unit_type_ids
            # where Attribute.Structure (value 8) is present. This replaces the
            # hardcoded string-based BUILDING_TYPES lookup for runtime
            # is_building() checks, using the SC2 engine's own type metadata
            # instead of manually curated name sets.
            #
            # The string-based BUILDING_TYPES in shared_constants.py is retained
            # as a legacy fallback for post-extraction tools (metadata_writer,
            # create_unit_counts, etc.) that operate on entity type name strings.
            api_type_data = self._build_api_type_data(controller)

            # Pass API-derived type data to state_extractor so it can forward
            # the building type ID set to unit and building extractors.
            self.state_extractor.set_api_type_data(api_type_data)

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

                    # Single observe — raw_data.units is perspective-invariant in
                    # observer mode, so one call captures all unit/building data for
                    # both players. Perspective switching was previously used here to
                    # attempt per-player upgrade_ids reads, but diagnostic testing
                    # confirmed upgrade_ids is always empty in observer mode regardless
                    # of active perspective. Economy and upgrades are handled entirely
                    # in _add_tracker_columns() after stitching — not in this loop.
                    obs = controller.observe()

                    # player_result is populated when the game ends
                    if obs.player_result:
                        logger.info(f"Observer mode: Replay ended at loop {game_loop}")
                        break

                    game_loop = obs.observation.game_loop

                    # Extract unit and building state from the single observation.
                    # Economy and upgrades are excluded here — they are appended to
                    # the final parquet by _add_tracker_columns() after stitching.
                    state = self.state_extractor.extract_observation_observer_mode(
                        obs, obs, game_loop
                    )

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

                    # Accumulate in column-oriented format for faster DataFrame
                    # construction. When new columns appear (dynamic schema), we
                    # backfill them with NaN for previous rows IN THIS BATCH
                    # only — earlier batches are already flushed to disk and will
                    # gain these columns during reconcile_parts().
                    for col, val in row.items():
                        if col not in column_data:
                            # New column: backfill previous rows in current batch
                            column_data[col] = [np.nan] * batch_row_count
                        column_data[col].append(val)
                    # Defensive: ensure all existing columns get a value for this row
                    for col in column_data:
                        if len(column_data[col]) <= batch_row_count:
                            column_data[col].append(np.nan)
                    batch_row_count += 1
                    row_count += 1

                    # Flush batch to a part file when it reaches BATCH_SIZE.
                    # This frees the Python lists in column_data, capping peak
                    # memory at BATCH_SIZE * column_count.
                    if batch_row_count >= BATCH_SIZE:
                        self.parquet_writer.write_batch_part(
                            column_data, parts_dir, part_index,
                            self.schema_manager,
                        )
                        # write_batch_part calls column_data.clear() internally,
                        # but we reassign to a fresh dict for clarity.
                        column_data = {}
                        batch_row_count = 0
                        part_index += 1

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
                f"Observer mode complete. Extracted {row_count} rows, "
                f"{len(all_messages)} messages"
            )

        # --- Write output files ---
        # replay_name and parquet_dir were set up before the game loop for
        # the parts directory; json_dir is new here.
        json_dir = output_dir / 'json'
        parquet_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        output_files = {
            'game_state': parquet_dir / f"{replay_name}_game_state.parquet",
            'metadata': json_dir / f"{replay_name}_metadata.json",
        }

        # Flush any remaining rows in the last (partial) batch.
        if batch_row_count > 0:
            self.parquet_writer.write_batch_part(
                column_data, parts_dir, part_index,
                self.schema_manager,
            )
            column_data = {}
            part_index += 1

        # Reconcile all part files into one final parquet.
        # PyArrow fills columns missing in earlier parts with null.
        logger.info(f"Writing game state to {output_files['game_state']}")
        self.parquet_writer.reconcile_parts(
            parts_dir, output_files['game_state']
        )

        # Post-process: add economy and upgrade columns from tracker events.
        # These are intentionally excluded from the hot extraction loop for
        # speed. This single pandas pass appends them to the final parquet.
        self._add_tracker_columns(output_files['game_state'], replay_path)

        # Build and write comprehensive metadata JSON that makes the
        # parquet file self-documenting for dataset consumers.
        parquet_filename = f"{replay_name}_game_state.parquet"
        metadata_dict = build_metadata(
            metadata=metadata,
            columns=self.schema_manager.columns,
            total_rows=row_count,
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
                'rows_written': row_count,
            },
        }

    def _add_tracker_columns(
        self,
        parquet_path: Path,
        replay_path: Path,
    ) -> None:
        """
        Post-process the stitched parquet to append economy and upgrade columns.

        Called once after reconcile_parts() produces the final parquet. Reads the
        parquet into a DataFrame, parses replay tracker events for economy and
        upgrade data, joins them in, then writes the augmented DataFrame back to
        the same path.

        Economy columns (p1_minerals, p1_vespene, p1_supply_used, p1_supply_cap,
        p1_collection_rate_minerals, p1_collection_rate_vespene, and p2_*
        equivalents) are forward-filled from SPlayerStatsEvent snapshots using
        pandas merge_asof. Rows before the first snapshot receive 0.

        Upgrade columns (p1_upgrades, p2_upgrades) contain the cumulative,
        chronological list of upgrade names completed at or before each row's
        game_loop. Cells before the first completion event are None (NaN);
        every cell from the first completion onward is a list[str]. Within a
        single completion game_loop, names are alphabetically sorted so the
        per-loop ordering is deterministic and matches the migration script.

        Args:
            parquet_path: Path to the reconciled parquet file. Read and
                          overwritten in-place with the additional columns.
            replay_path: Path to the .SC2Replay file used to parse tracker events.

        Depends on / calls:
            - load_economy_snapshots() from economy_extractor
            - load_upgrade_snapshots() from upgrade_extractor
            - pandas.read_parquet / DataFrame.to_parquet
            - pandas.merge_asof for forward-filling economy snapshots
        """
        import pandas as pd
        from collections import defaultdict

        logger.info(f"Post-processing: adding tracker columns to {parquet_path.name}")

        # Read the stitched parquet. Rows are already ordered by game_loop
        # from extraction, but sort defensively to satisfy merge_asof.
        df = pd.read_parquet(parquet_path)
        df = df.sort_values('game_loop').reset_index(drop=True)

        # -----------------------------------------------------------------
        # Economy columns — forward-filled from SPlayerStatsEvent snapshots
        # -----------------------------------------------------------------
        economy_snapshots = load_economy_snapshots(str(replay_path))
        economy_fields = [
            'minerals', 'vespene', 'supply_used', 'supply_cap',
            'collection_rate_minerals', 'collection_rate_vespene',
        ]

        for player_id in [1, 2]:
            player_snaps = economy_snapshots.get(player_id, [])

            if not player_snaps:
                logger.warning(
                    f"No economy snapshots for player {player_id} — "
                    f"setting economy columns to 0"
                )
                for field in economy_fields:
                    df[f'p{player_id}_{field}'] = 0
                continue

            # Build a DataFrame of snapshots: columns are game_loop + field names
            eco_df = pd.DataFrame(player_snaps)
            eco_df = eco_df.sort_values('game_loop').reset_index(drop=True)

            # Prefix field columns with the player identifier so they do not
            # collide when merging both players in sequence.
            eco_df = eco_df.rename(columns={
                field: f'p{player_id}_{field}'
                for field in economy_fields
                if field in eco_df.columns
            })

            # merge_asof with direction='backward': for every row's game_loop,
            # attach the most recent snapshot whose game_loop <= that value.
            # This is a vectorised forward-fill — O(n log n) total.
            df = pd.merge_asof(df, eco_df, on='game_loop', direction='backward')

            # Rows before the first snapshot (early game_loops with no data yet)
            # receive NaN from merge_asof; fill with 0.
            for field in economy_fields:
                col = f'p{player_id}_{field}'
                if col in df.columns:
                    df[col] = df[col].fillna(0)

            logger.info(
                f"Economy columns added for player {player_id} "
                f"({len(player_snaps)} snapshot(s))"
            )

        # -----------------------------------------------------------------
        # Upgrade columns — cumulative chronological list per row
        # -----------------------------------------------------------------
        # For each row's game_loop, the cell holds the list of every upgrade
        # the player has completed at or before that loop. The first cells
        # (before any completion event) are None (NaN); every cell after the
        # first event is a list[str] that grows monotonically.
        from bisect import bisect_right

        upgrade_snapshots = load_upgrade_snapshots(str(replay_path))

        for player_id in [1, 2]:
            col = f'p{player_id}_upgrades'

            # Group completions by game_loop, then sort each per-loop batch
            # alphabetically so the within-loop ordering is deterministic.
            # This matches the migration script, which only has the legacy
            # alphabetical encoding to recover from.
            per_loop: Dict[int, list] = defaultdict(list)
            for gl, name in upgrade_snapshots.get(player_id, []):
                per_loop[gl].append(name)

            # Build two parallel arrays for binary-search lookup:
            #   completion_loops[i] = the i-th distinct completion game_loop
            #   cumulative_at[i]    = the cumulative list of upgrade names
            #                         completed at or before completion_loops[i]
            # Each cumulative_at entry is a fresh list so that mapping the
            # same list object to many rows is safe (pandas will not mutate
            # them, but a fresh list per step keeps the contract obvious).
            completion_loops: List[int] = sorted(per_loop.keys())
            cumulative_at: List[List[str]] = []
            running: List[str] = []
            for gl in completion_loops:
                running = running + sorted(per_loop[gl])
                cumulative_at.append(list(running))

            def _lookup(game_loop: int) -> Any:
                """Cumulative list as of game_loop, or None before any event."""
                idx = bisect_right(completion_loops, game_loop)
                if idx == 0:
                    return None
                return list(cumulative_at[idx - 1])

            if completion_loops:
                df[col] = df['game_loop'].map(_lookup)
            else:
                # No completions for this player — write an all-None object
                # column so the schema stays present and downstream code can
                # rely on the column existing.
                df[col] = pd.Series([None] * len(df), dtype=object)

            completed_count = sum(len(v) for v in per_loop.values())
            logger.info(
                f"Upgrade column {col} added "
                f"({completed_count} completion event(s), "
                f"{len(completion_loops)} distinct loop(s))"
            )

        # Write the augmented DataFrame back to the same parquet path.
        compression = self.config.get('compression', 'snappy')
        df.to_parquet(parquet_path, compression=compression, index=False)

        logger.info(
            f"Tracker columns written successfully to {parquet_path.name}"
        )

    def _build_api_type_data(self, controller) -> Dict[str, Any]:
        """
        Call controller.data_raw() and build API-derived type classification sets.

        Queries the SC2 engine's UnitTypeData to build a frozenset of integer
        unit_type_ids that have Attribute.Structure (value 8). This set is used
        by unit_extractor and building_extractor for runtime is_building()
        classification, replacing the hardcoded string-based BUILDING_TYPES
        lookup with an authoritative, engine-derived alternative.

        The Attribute.Structure flag marks ALL structures including neutral map
        objects (bridges, destructible rocks, mineral fields). The pipeline's
        existing owner filter (unit.owner in {1, 2}) already excludes these
        neutrals, so no additional filtering is needed here.

        Args:
            controller: SC2 controller instance (after start_replay())

        Returns:
            Dict with API-derived type data:
            {
                'building_type_ids': frozenset[int] -- unit type IDs with
                    Attribute.Structure from data_raw()
            }

        Depends on / calls:
            - controller.data_raw() from the SC2 API
        """
        # Attribute.Structure has enum value 8 in s2clientprotocol.data_pb2
        ATTRIBUTE_STRUCTURE = 8

        try:
            data_raw = controller.data_raw()
            building_ids = set()

            for unit_type_data in data_raw.units:
                if ATTRIBUTE_STRUCTURE in unit_type_data.attributes:
                    building_ids.add(unit_type_data.unit_id)

            building_type_ids = frozenset(building_ids)
            logger.info(
                f"API type data: {len(building_type_ids)} building type IDs "
                f"from data_raw() (Attribute.Structure)"
            )
            return {'building_type_ids': building_type_ids}

        except Exception as e:
            logger.warning(
                f"Failed to build API type data from data_raw(): {e}. "
                f"Falling back to string-based BUILDING_TYPES."
            )
            return {'building_type_ids': None}

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

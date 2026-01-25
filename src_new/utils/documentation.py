"""
Documentation Generator: Creates data dictionaries and processing reports.

This module generates comprehensive documentation for extracted parquet files,
including data dictionaries, replay reports, and batch summaries.
"""

from typing import Dict, Any, Optional, List, TYPE_CHECKING
from pathlib import Path
import logging
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from src_new.extraction.schema_manager import SchemaManager


logger = logging.getLogger(__name__)


def generate_data_dictionary(schema: Any, output_path: Path) -> None:
    """
    Generate complete data dictionary in markdown format.

    Creates comprehensive documentation of all columns in the parquet schema,
    organized by category with descriptions, types, examples, and metadata.

    Args:
        schema: SchemaManager instance with column definitions (accepts any object with
                generate_documentation() and get_column_list() methods)
        output_path: Path to save the markdown data dictionary

    The generated data dictionary includes:
    - Overview of the data structure
    - Base columns (game_loop, timestamp)
    - Unit columns (per player, per unit type)
    - Building columns (per player, per building type)
    - Economy columns (per player)
    - Upgrade columns (per player)
    - Unit count columns

    For each column:
    - Column name
    - Description
    - Data type
    - Missing value handling
    - Valid ranges (where applicable)

    # TODO: Test case - Generate data dictionary from schema
    # TODO: Test case - Verify markdown formatting
    # TODO: Test case - Check all column categories present
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating data dictionary: {output_path}")

    # Get column documentation from schema
    column_docs = schema.generate_documentation()
    columns = schema.get_column_list()

    # Build markdown document
    lines = []
    lines.append("# SC2 Replay Ground Truth Data Dictionary\n")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Total Columns**: {len(columns)}\n")

    # Overview
    lines.append("## Overview\n")
    lines.append("This data dictionary describes the schema for SC2 replay ground truth data.")
    lines.append("The data is stored in wide-format parquet files with one row per game loop.\n")
    lines.append("### Data Structure\n")
    lines.append("- **Format**: Wide-table parquet")
    lines.append("- **Granularity**: One row per game loop (frame)")
    lines.append("- **Game Speed**: 22.4 game loops per second")
    lines.append("- **Missing Values**: NaN for numeric, null for string")
    lines.append("- **Players**: Data for both player 1 (p1_) and player 2 (p2_)\n")

    # Categorize columns
    base_cols = []
    unit_cols = []
    building_cols = []
    economy_cols = []
    upgrade_cols = []
    count_cols = []
    other_cols = []

    for col in columns:
        if col in ['game_loop', 'timestamp_seconds']:
            base_cols.append(col)
        elif '_count' in col and not any(x in col for x in ['_x', '_y', '_z']):
            count_cols.append(col)
        elif any(x in col for x in ['minerals', 'vespene', 'supply_', 'workers', 'idle_workers']):
            economy_cols.append(col)
        elif 'upgrade' in col:
            upgrade_cols.append(col)
        elif any(x in col for x in ['_status', '_progress', '_started_loop', '_completed_loop', '_destroyed_loop']):
            building_cols.append(col)
        elif any(x in col for x in ['_x', '_y', '_z', '_health', '_shields', '_energy', '_state']):
            unit_cols.append(col)
        else:
            other_cols.append(col)

    # Base Columns
    lines.append("## Base Columns\n")
    lines.append("Core columns present in every row.\n")

    for col in base_cols:
        doc = column_docs.get(col, {})
        lines.append(f"### `{col}`\n")
        lines.append(f"- **Type**: `{doc.get('type', 'unknown')}`")
        lines.append(f"- **Description**: {doc.get('description', 'No description')}")
        lines.append(f"- **Missing Values**: {doc.get('missing_value', 'N/A')}")

        # Add range info
        if col == 'game_loop':
            lines.append(f"- **Range**: 0 to game_duration_loops")
        elif col == 'timestamp_seconds':
            lines.append(f"- **Range**: 0.0 to game_duration_seconds")
            lines.append(f"- **Calculation**: game_loop / 22.4")

        lines.append("")

    # Economy Columns
    if economy_cols:
        lines.append("## Economy Columns\n")
        lines.append("Resource and supply tracking for each player.\n")

        # Group by player
        for player in [1, 2]:
            player_economy_cols = [col for col in economy_cols if col.startswith(f'p{player}_')]

            if player_economy_cols:
                lines.append(f"### Player {player}\n")

                for col in sorted(player_economy_cols):
                    doc = column_docs.get(col, {})
                    lines.append(f"#### `{col}`\n")
                    lines.append(f"- **Type**: `{doc.get('type', 'unknown')}`")
                    lines.append(f"- **Description**: {doc.get('description', 'No description')}")
                    lines.append(f"- **Missing Values**: {doc.get('missing_value', 'N/A')}")

                    # Add range info
                    if 'minerals' in col or 'vespene' in col:
                        lines.append(f"- **Range**: >= 0")
                    elif 'supply' in col:
                        lines.append(f"- **Range**: >= 0")
                        if 'supply_used' in col:
                            lines.append(f"- **Constraint**: supply_used <= supply_cap")

                    lines.append("")

    # Unit Count Columns
    if count_cols:
        lines.append("## Unit Count Columns\n")
        lines.append("Aggregate counts of each unit type per player.\n")

        # Group by player
        for player in [1, 2]:
            player_count_cols = [col for col in count_cols if col.startswith(f'p{player}_')]

            if player_count_cols:
                lines.append(f"### Player {player}\n")

                for col in sorted(player_count_cols):
                    doc = column_docs.get(col, {})
                    lines.append(f"#### `{col}`\n")
                    lines.append(f"- **Type**: `{doc.get('type', 'unknown')}`")
                    lines.append(f"- **Description**: {doc.get('description', 'No description')}")
                    lines.append(f"- **Missing Values**: {doc.get('missing_value', 'N/A')}")
                    lines.append(f"- **Range**: >= 0")
                    lines.append("")

    # Unit Columns
    if unit_cols:
        lines.append("## Unit Columns\n")
        lines.append("Individual unit tracking with position, health, and state.\n")
        lines.append("**Note**: Each unit has multiple columns (x, y, z, health, shields, energy, state).")
        lines.append("Unit columns use NaN when the unit does not exist at a given game loop.\n")

        # Show examples from first few units
        sample_units = set()
        for col in unit_cols:
            # Extract unit identifier (e.g., "p1_marine_001" from "p1_marine_001_x")
            parts = col.split('_')
            if len(parts) >= 4:
                unit_id = '_'.join(parts[:3])  # p1_marine_001
                sample_units.add(unit_id)

            if len(sample_units) >= 10:  # Show first 10 units as examples
                break

        lines.append(f"\n**Total Unit Columns**: {len(unit_cols)}")
        lines.append(f"**Example Units**: {min(10, len(sample_units))} shown below\n")

        for unit_id in sorted(list(sample_units))[:10]:
            lines.append(f"### `{unit_id}_*`\n")

            # Find all columns for this unit
            unit_specific_cols = [col for col in unit_cols if col.startswith(f"{unit_id}_")]

            for col in sorted(unit_specific_cols):
                doc = column_docs.get(col, {})
                suffix = col.replace(f"{unit_id}_", "")

                lines.append(f"#### `{col}`")
                lines.append(f"- **Type**: `{doc.get('type', 'unknown')}`")
                lines.append(f"- **Description**: {doc.get('description', 'No description')}")
                lines.append(f"- **Missing Values**: {doc.get('missing_value', 'N/A')}")

                # Add range info based on suffix
                if suffix in ['x', 'y']:
                    lines.append(f"- **Range**: 0 to map_dimension")
                elif suffix == 'z':
                    lines.append(f"- **Range**: 0+ (height above ground)")
                elif 'health' in suffix or 'shields' in suffix or 'energy' in suffix:
                    lines.append(f"- **Range**: 0 to max_value")
                elif suffix == 'state':
                    lines.append(f"- **Valid Values**: created, alive, cancelled, dead, built, existing, killed")

                lines.append("")

        if len(unit_cols) > len(sample_units) * 10:
            lines.append(f"\n*... and {len(unit_cols) - len(sample_units) * 10} more unit columns*\n")

    # Building Columns
    if building_cols:
        lines.append("## Building Columns\n")
        lines.append("Building tracking with position, status, and progress.\n")
        lines.append("**Note**: Each building has multiple columns (x, y, z, status, progress, started_loop, completed_loop, destroyed_loop).")
        lines.append("Building columns use NaN when the building does not exist at a given game loop.\n")

        # Show examples from first few buildings
        sample_buildings = set()
        for col in building_cols:
            parts = col.split('_')
            if len(parts) >= 4:
                building_id = '_'.join(parts[:3])
                sample_buildings.add(building_id)

            if len(sample_buildings) >= 5:  # Show first 5 buildings as examples
                break

        lines.append(f"\n**Total Building Columns**: {len(building_cols)}")
        lines.append(f"**Example Buildings**: {min(5, len(sample_buildings))} shown below\n")

        for building_id in sorted(list(sample_buildings))[:5]:
            lines.append(f"### `{building_id}_*`\n")

            building_specific_cols = [col for col in building_cols if col.startswith(f"{building_id}_")]

            for col in sorted(building_specific_cols):
                doc = column_docs.get(col, {})
                suffix = col.replace(f"{building_id}_", "")

                lines.append(f"#### `{col}`")
                lines.append(f"- **Type**: `{doc.get('type', 'unknown')}`")
                lines.append(f"- **Description**: {doc.get('description', 'No description')}")
                lines.append(f"- **Missing Values**: {doc.get('missing_value', 'N/A')}")

                # Add range info based on suffix
                if suffix in ['x', 'y']:
                    lines.append(f"- **Range**: 0 to map_dimension")
                elif suffix == 'z':
                    lines.append(f"- **Range**: 0+ (height above ground)")
                elif suffix == 'status':
                    lines.append(f"- **Valid Values**: started, building, completed, destroyed")
                elif suffix == 'progress':
                    lines.append(f"- **Range**: 0-100")
                    lines.append(f"- **Constraint**: Monotonically increasing (never decreases)")
                elif 'loop' in suffix:
                    lines.append(f"- **Range**: 0 to game_duration_loops")

                lines.append("")

        if len(building_cols) > len(sample_buildings) * 8:
            lines.append(f"\n*... and {len(building_cols) - len(sample_buildings) * 8} more building columns*\n")

    # Upgrade Columns
    if upgrade_cols:
        lines.append("## Upgrade Columns\n")
        lines.append("Technology upgrades for each player.\n")

        for player in [1, 2]:
            player_upgrade_cols = [col for col in upgrade_cols if col.startswith(f'p{player}_')]

            if player_upgrade_cols:
                lines.append(f"### Player {player}\n")

                for col in sorted(player_upgrade_cols):
                    doc = column_docs.get(col, {})
                    lines.append(f"#### `{col}`\n")
                    lines.append(f"- **Type**: `{doc.get('type', 'unknown')}`")
                    lines.append(f"- **Description**: {doc.get('description', 'No description')}")
                    lines.append(f"- **Missing Values**: {doc.get('missing_value', 'N/A')}")
                    lines.append(f"- **Range**: >= 0 (level of upgrade)")
                    lines.append("")

    # Other columns
    if other_cols:
        lines.append("## Other Columns\n")

        for col in sorted(other_cols):
            doc = column_docs.get(col, {})
            lines.append(f"### `{col}`\n")
            lines.append(f"- **Type**: `{doc.get('type', 'unknown')}`")
            lines.append(f"- **Description**: {doc.get('description', 'No description')}")
            lines.append(f"- **Missing Values**: {doc.get('missing_value', 'N/A')}")
            lines.append("")

    # Footer
    lines.append("---\n")
    lines.append("*This data dictionary was automatically generated by the SC2 Replay Ground Truth Extraction Pipeline.*\n")

    # Write to file
    content = "\n".join(lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.info(f"Data dictionary written to {output_path}")
    logger.info(f"  Total columns documented: {len(columns)}")


def generate_replay_report(
    replay_path: Path,
    output_path: Path,
    validation_results: Optional[dict] = None
) -> None:
    """
    Generate processing report for a single replay.

    Creates a comprehensive report documenting the processing of a single replay,
    including metadata, statistics, validation results, and data previews.

    Args:
        replay_path: Path to the original replay file
        output_path: Path to save the markdown report
        validation_results: Optional validation results from OutputValidator

    The report includes:
    - Replay metadata (map, players, duration)
    - Processing statistics (time, rows generated)
    - Validation results (pass/fail, warnings)
    - Sample data preview (first/last rows)
    - Column statistics (min/max/mean for numeric columns)

    # TODO: Test case - Generate replay report
    # TODO: Test case - Include validation results
    # TODO: Test case - Verify sample data preview
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating replay report: {output_path}")

    # Build markdown document
    lines = []
    lines.append("# SC2 Replay Processing Report\n")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Replay information
    lines.append("## Replay Information\n")
    lines.append(f"- **File**: `{replay_path.name}`")
    lines.append(f"- **Path**: `{replay_path}`")

    if replay_path.exists():
        lines.append(f"- **Size**: {replay_path.stat().st_size / 1024:.2f} KB")
    else:
        lines.append(f"- **Status**: File not found")

    lines.append("")

    # Look for output files
    replay_stem = replay_path.stem
    replay_dir = replay_path.parent

    game_state_path = replay_dir / f"{replay_stem}_game_state.parquet"
    messages_path = replay_dir / f"{replay_stem}_messages.parquet"
    schema_path = replay_dir / f"{replay_stem}_schema.json"

    # Output files
    lines.append("## Output Files\n")

    if game_state_path.exists():
        lines.append(f"- ✅ **Game State**: `{game_state_path.name}`")

        # Get file info
        try:
            parquet_file = pq.ParquetFile(game_state_path)
            metadata = parquet_file.metadata

            lines.append(f"  - Rows: {metadata.num_rows}")
            lines.append(f"  - Columns: {metadata.num_columns}")
            lines.append(f"  - Size: {game_state_path.stat().st_size / 1024:.2f} KB")
            lines.append(f"  - Compression: {metadata.row_group(0).column(0).compression}")
        except Exception as e:
            lines.append(f"  - Error reading file: {e}")
    else:
        lines.append(f"- ❌ **Game State**: Not found")

    if messages_path.exists():
        lines.append(f"- ✅ **Messages**: `{messages_path.name}`")

        try:
            parquet_file = pq.ParquetFile(messages_path)
            metadata = parquet_file.metadata

            lines.append(f"  - Messages: {metadata.num_rows}")
            lines.append(f"  - Size: {messages_path.stat().st_size / 1024:.2f} KB")
        except Exception as e:
            lines.append(f"  - Error reading file: {e}")
    else:
        lines.append(f"- ⚠️ **Messages**: Not found (may be no messages)")

    if schema_path.exists():
        lines.append(f"- ✅ **Schema**: `{schema_path.name}`")
    else:
        lines.append(f"- ❌ **Schema**: Not found")

    lines.append("")

    # Validation results
    if validation_results:
        lines.append("## Validation Results\n")

        if validation_results.get('valid', False):
            lines.append("**Status**: ✅ Validation Passed\n")
        else:
            lines.append("**Status**: ❌ Validation Failed\n")

        # Errors
        errors = validation_results.get('errors', [])
        if errors:
            lines.append("### Errors\n")
            for error in errors:
                lines.append(f"- ❌ {error}")
            lines.append("")

        # Warnings
        warnings = validation_results.get('warnings', [])
        if warnings:
            lines.append("### Warnings\n")
            for warning in warnings:
                lines.append(f"- ⚠️ {warning}")
            lines.append("")

        # Statistics
        stats = validation_results.get('stats', {})
        if stats:
            lines.append("### Statistics\n")
            for key, value in stats.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

    # Data preview
    if game_state_path.exists():
        lines.append("## Data Preview\n")

        try:
            df = pd.read_parquet(game_state_path)

            lines.append("### First 5 Rows\n")
            lines.append("```")

            # Show only base columns and a few key columns
            preview_cols = ['game_loop', 'timestamp_seconds']
            economy_cols = [col for col in df.columns if any(x in col for x in ['minerals', 'vespene', 'supply'])]
            preview_cols.extend(economy_cols[:6])  # First 6 economy columns

            if all(col in df.columns for col in preview_cols):
                lines.append(df[preview_cols].head().to_string())
            else:
                lines.append(df.head().to_string())

            lines.append("```\n")

            # Column statistics for numeric columns
            lines.append("### Column Statistics\n")

            numeric_cols = df.select_dtypes(include=['float64', 'int64', 'Int64']).columns.tolist()

            # Focus on key columns
            key_cols = [col for col in numeric_cols if any(
                x in col for x in ['game_loop', 'timestamp', 'minerals', 'vespene', 'supply']
            )][:10]

            if key_cols:
                stats_df = df[key_cols].describe()
                lines.append("```")
                lines.append(stats_df.to_string())
                lines.append("```\n")

        except Exception as e:
            lines.append(f"Error loading data preview: {e}\n")

    # Footer
    lines.append("---\n")
    lines.append("*This report was automatically generated by the SC2 Replay Ground Truth Extraction Pipeline.*\n")

    # Write to file
    content = "\n".join(lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.info(f"Replay report written to {output_path}")


def generate_batch_summary(batch_results: Dict[str, Any], output_path: Path) -> None:
    """
    Generate summary report for batch processing.

    Creates a high-level summary of batch processing results across multiple
    replays, with aggregate statistics and common error patterns.

    Args:
        batch_results: Dictionary with batch processing results
            Expected format:
            {
                'results': List[dict],  # List of individual replay results
                'total_replays': int,
                'successful': int,
                'failed': int,
                'total_time_seconds': float,
                'config': dict,
            }
        output_path: Path to save the markdown summary

    The summary includes:
    - Total replays processed
    - Success/failure counts
    - Average processing time
    - Common errors
    - Validation summary across all replays
    - Resource usage statistics

    # TODO: Test case - Generate batch summary
    # TODO: Test case - Aggregate multiple replay results
    # TODO: Test case - Identify common error patterns
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating batch summary: {output_path}")

    # Build markdown document
    lines = []
    lines.append("# SC2 Replay Batch Processing Summary\n")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Overall statistics
    lines.append("## Overall Statistics\n")

    total_replays = batch_results.get('total_replays', 0)
    successful = batch_results.get('successful', 0)
    failed = batch_results.get('failed', 0)
    total_time = batch_results.get('total_time_seconds', 0)

    lines.append(f"- **Total Replays**: {total_replays}")
    lines.append(f"- **Successful**: {successful} ({successful/total_replays*100:.1f}%)" if total_replays > 0 else "- **Successful**: 0")
    lines.append(f"- **Failed**: {failed} ({failed/total_replays*100:.1f}%)" if total_replays > 0 else "- **Failed**: 0")
    lines.append(f"- **Total Time**: {total_time:.2f}s")
    lines.append(f"- **Average Time**: {total_time/total_replays:.2f}s per replay" if total_replays > 0 else "- **Average Time**: N/A")
    lines.append("")

    # Configuration
    config = batch_results.get('config', {})
    if config:
        lines.append("## Configuration\n")
        for key, value in config.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

    # Individual results
    results = batch_results.get('results', [])

    if results:
        lines.append("## Individual Results\n")

        # Success table
        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]

        if successful_results:
            lines.append(f"### Successful ({len(successful_results)})\n")
            lines.append("| Replay | Rows | Messages | Time (s) |")
            lines.append("|--------|------|----------|----------|")

            for result in successful_results:
                replay_name = Path(result.get('replay_path', 'unknown')).name
                stats = result.get('stats', {})
                rows = stats.get('rows_written', 0)
                messages = stats.get('messages_written', 0)
                time = stats.get('processing_time_seconds', 0)

                lines.append(f"| {replay_name} | {rows} | {messages} | {time:.2f} |")

            lines.append("")

        if failed_results:
            lines.append(f"### Failed ({len(failed_results)})\n")
            lines.append("| Replay | Error |")
            lines.append("|--------|-------|")

            for result in failed_results:
                replay_name = Path(result.get('replay_path', 'unknown')).name
                error = result.get('error', 'Unknown error')

                # Truncate long errors
                if len(error) > 100:
                    error = error[:97] + "..."

                lines.append(f"| {replay_name} | {error} |")

            lines.append("")

        # Aggregate statistics
        if successful_results:
            lines.append("## Aggregate Statistics\n")

            total_rows = sum(r.get('stats', {}).get('rows_written', 0) for r in successful_results)
            total_messages = sum(r.get('stats', {}).get('messages_written', 0) for r in successful_results)
            avg_rows = total_rows / len(successful_results) if successful_results else 0
            avg_messages = total_messages / len(successful_results) if successful_results else 0

            lines.append(f"- **Total Rows Generated**: {total_rows:,}")
            lines.append(f"- **Total Messages Extracted**: {total_messages:,}")
            lines.append(f"- **Average Rows per Replay**: {avg_rows:.0f}")
            lines.append(f"- **Average Messages per Replay**: {avg_messages:.0f}")

            # Game duration statistics
            all_durations = [
                r.get('stats', {}).get('total_loops', 0) / 22.4
                for r in successful_results
                if r.get('stats', {}).get('total_loops', 0) > 0
            ]

            if all_durations:
                lines.append(f"- **Average Game Duration**: {sum(all_durations)/len(all_durations):.1f}s")
                lines.append(f"- **Shortest Game**: {min(all_durations):.1f}s")
                lines.append(f"- **Longest Game**: {max(all_durations):.1f}s")

            lines.append("")

    # Common errors
    if failed_results:
        lines.append("## Common Errors\n")

        # Count error types
        error_counts: Dict[str, int] = {}

        for result in failed_results:
            error = result.get('error', 'Unknown error')

            # Simplify error message to group similar errors
            error_type = error.split(':')[0] if ':' in error else error

            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        # Sort by frequency
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)

        for error_type, count in sorted_errors[:10]:  # Top 10 errors
            lines.append(f"- **{error_type}**: {count} occurrence(s)")

        lines.append("")

    # Recommendations
    lines.append("## Recommendations\n")

    if failed > 0:
        lines.append(f"- Review {failed} failed replay(s) for common error patterns")

    if total_replays > 0 and successful > 0:
        success_rate = successful / total_replays
        if success_rate < 0.9:
            lines.append(f"- Success rate is {success_rate*100:.1f}%. Investigate common failure causes.")

    if not results:
        lines.append("- No results to analyze. Ensure batch processing completed successfully.")

    lines.append("")

    # Footer
    lines.append("---\n")
    lines.append("*This summary was automatically generated by the SC2 Replay Ground Truth Extraction Pipeline.*\n")

    # Write to file
    content = "\n".join(lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.info(f"Batch summary written to {output_path}")
    logger.info(f"  Total replays: {total_replays}")
    logger.info(f"  Success rate: {successful}/{total_replays}")

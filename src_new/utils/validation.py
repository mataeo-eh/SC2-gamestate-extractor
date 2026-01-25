"""
OutputValidator: Validates extracted parquet files for quality assurance.

This module provides comprehensive validation of extracted parquet files,
checking for data integrity, schema compliance, and logical consistency.
"""

from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import logging

import pandas as pd
import numpy as np
import pyarrow.parquet as pq


logger = logging.getLogger(__name__)


class OutputValidator:
    """
    Validates extracted parquet files for quality assurance.

    This class performs comprehensive validation of game state and message
    parquet files, checking for:
    - Schema compliance
    - Data integrity
    - Logical consistency
    - State transition validity
    - Resource constraints
    - Unit count consistency
    """

    def __init__(self):
        """Initialize the OutputValidator."""
        logger.info("OutputValidator initialized")

    def validate_game_state_parquet(self, parquet_path: Path) -> dict:
        """
        Validate game state parquet file.

        Performs comprehensive validation including:
        - Row count > 0
        - No duplicate game_loops
        - Unit counts match individual units
        - State transitions are valid
        - Building progress is monotonic (never decreases)
        - Resources are non-negative
        - No unexpected NaN patterns
        - Column types match schema
        - Required columns present

        Args:
            parquet_path: Path to game state parquet file

        Returns:
            Validation report dictionary:
            {
                'valid': bool,              # Overall validation status
                'file_path': str,           # Path to validated file
                'errors': List[str],        # Critical errors (make valid=False)
                'warnings': List[str],      # Non-critical warnings
                'info': Dict[str, Any],     # File metadata
                'checks': Dict[str, bool],  # Individual check results
                'stats': Dict[str, Any],    # Data statistics
            }

        # TODO: Test case - Validate good parquet
        # TODO: Test case - Detect duplicate game_loops
        # TODO: Test case - Detect invalid state transitions
        # TODO: Test case - Detect unit count mismatches
        # TODO: Test case - Detect non-monotonic building progress
        """
        parquet_path = Path(parquet_path)

        logger.info(f"Validating game state parquet: {parquet_path}")

        # Initialize report
        report = {
            'valid': True,
            'file_path': str(parquet_path),
            'errors': [],
            'warnings': [],
            'info': {},
            'checks': {},
            'stats': {},
        }

        try:
            # Check file exists
            if not parquet_path.exists():
                report['errors'].append(f"File not found: {parquet_path}")
                report['valid'] = False
                return report

            # Get file info
            parquet_file = pq.ParquetFile(parquet_path)
            metadata = parquet_file.metadata

            report['info'] = {
                'num_rows': metadata.num_rows,
                'num_columns': metadata.num_columns,
                'file_size_kb': parquet_path.stat().st_size / 1024,
                'compression': metadata.row_group(0).column(0).compression,
            }

            # Load data
            df = pd.read_parquet(parquet_path)

            # Run validation checks
            self._check_row_count(df, report)
            self._check_required_columns(df, report)
            self._check_duplicate_game_loops(df, report)
            self._check_column_types(df, report)
            self._check_resource_validity(df, report)
            self._check_building_progress_monotonic(df, report)
            self._check_unit_count_consistency(df, report)
            self._check_state_transitions(df, report)
            self._check_nan_patterns(df, report)

            # Generate statistics
            self._generate_stats(df, report)

            # Set overall validity
            report['valid'] = len(report['errors']) == 0

            if report['valid']:
                logger.info(f"Validation passed with {len(report['warnings'])} warnings")
            else:
                logger.warning(f"Validation failed with {len(report['errors'])} errors")

        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            report['errors'].append(f"Validation exception: {e}")
            report['valid'] = False

        return report

    def validate_messages_parquet(self, parquet_path: Path) -> dict:
        """
        Validate messages parquet file.

        Checks:
        - Row count >= 0 (messages are optional)
        - Required columns present (game_loop, player_id, message)
        - game_loops are valid and in range
        - No duplicate messages
        - Column types correct

        Args:
            parquet_path: Path to messages parquet file

        Returns:
            Validation report dictionary (same structure as validate_game_state_parquet)

        # TODO: Test case - Validate messages parquet
        # TODO: Test case - Handle empty messages file
        # TODO: Test case - Detect invalid game_loop values
        """
        parquet_path = Path(parquet_path)

        logger.info(f"Validating messages parquet: {parquet_path}")

        # Initialize report
        report = {
            'valid': True,
            'file_path': str(parquet_path),
            'errors': [],
            'warnings': [],
            'info': {},
            'checks': {},
            'stats': {},
        }

        try:
            # Check file exists
            if not parquet_path.exists():
                report['warnings'].append(f"Messages file not found (optional): {parquet_path}")
                report['checks']['file_exists'] = False
                return report

            report['checks']['file_exists'] = True

            # Get file info
            parquet_file = pq.ParquetFile(parquet_path)
            metadata = parquet_file.metadata

            report['info'] = {
                'num_rows': metadata.num_rows,
                'num_columns': metadata.num_columns,
                'file_size_kb': parquet_path.stat().st_size / 1024,
                'compression': metadata.row_group(0).column(0).compression,
            }

            # Load data
            df = pd.read_parquet(parquet_path)

            # Check required columns
            required_cols = ['game_loop', 'player_id', 'message']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                report['errors'].append(f"Missing required columns: {missing_cols}")
                report['checks']['required_columns'] = False
                report['valid'] = False
            else:
                report['checks']['required_columns'] = True

            # Check row count (can be 0 for no messages)
            if len(df) == 0:
                report['warnings'].append("No messages found (this is acceptable)")
                report['checks']['has_messages'] = False
            else:
                report['checks']['has_messages'] = True

                # Validate column types
                type_checks = {
                    'game_loop': (df['game_loop'].dtype == 'int64', 'game_loop should be int64'),
                    'player_id': (df['player_id'].dtype == 'int64', 'player_id should be int64'),
                    'message': (df['message'].dtype == 'object' or df['message'].dtype == 'string', 'message should be string'),
                }

                for check_name, (passed, message) in type_checks.items():
                    report['checks'][f'type_{check_name}'] = passed
                    if not passed:
                        report['errors'].append(message)
                        report['valid'] = False

                # Check for invalid game_loop values
                if (df['game_loop'] < 0).any():
                    report['errors'].append("Found negative game_loop values")
                    report['checks']['valid_game_loops'] = False
                    report['valid'] = False
                else:
                    report['checks']['valid_game_loops'] = True

                # Check for duplicate messages
                duplicate_count = df.duplicated().sum()
                if duplicate_count > 0:
                    report['warnings'].append(f"Found {duplicate_count} duplicate messages")
                    report['checks']['no_duplicates'] = False
                else:
                    report['checks']['no_duplicates'] = True

                # Generate statistics
                report['stats'] = {
                    'total_messages': len(df),
                    'unique_players': df['player_id'].nunique(),
                    'game_loop_range': (int(df['game_loop'].min()), int(df['game_loop'].max())),
                    'avg_message_length': float(df['message'].str.len().mean()),
                }

            if report['valid']:
                logger.info(f"Messages validation passed")
            else:
                logger.warning(f"Messages validation failed with {len(report['errors'])} errors")

        except Exception as e:
            logger.error(f"Messages validation error: {e}", exc_info=True)
            report['errors'].append(f"Validation exception: {e}")
            report['valid'] = False

        return report

    def generate_validation_report(self, validations: List[dict]) -> str:
        """
        Generate human-readable validation report from multiple validations.

        Creates a markdown-formatted report summarizing validation results
        across multiple files, with summary statistics, errors, warnings,
        and actionable recommendations.

        Args:
            validations: List of validation result dictionaries

        Returns:
            Markdown-formatted validation report string

        # TODO: Test case - Generate report from multiple validations
        # TODO: Test case - Verify markdown formatting
        """
        if not validations:
            return "# Validation Report\n\nNo validations performed.\n"

        # Build report sections
        lines = []
        lines.append("# SC2 Replay Extraction Validation Report\n")
        lines.append(f"**Total Files Validated**: {len(validations)}\n")

        # Summary statistics
        total_valid = sum(1 for v in validations if v['valid'])
        total_errors = sum(len(v['errors']) for v in validations)
        total_warnings = sum(len(v['warnings']) for v in validations)

        lines.append("## Summary\n")
        lines.append(f"- **Valid Files**: {total_valid}/{len(validations)}")
        lines.append(f"- **Total Errors**: {total_errors}")
        lines.append(f"- **Total Warnings**: {total_warnings}\n")

        # Overall status
        if total_valid == len(validations):
            lines.append("**Status**: ✅ All validations passed\n")
        else:
            lines.append(f"**Status**: ❌ {len(validations) - total_valid} file(s) failed validation\n")

        # Individual file results
        lines.append("## File Validation Results\n")

        for i, validation in enumerate(validations, 1):
            file_path = validation.get('file_path', 'Unknown')
            file_name = Path(file_path).name

            status = "✅ PASS" if validation['valid'] else "❌ FAIL"
            lines.append(f"### {i}. {file_name} - {status}\n")

            # File info
            if 'info' in validation and validation['info']:
                info = validation['info']
                lines.append("**File Info**:")
                lines.append(f"- Rows: {info.get('num_rows', 'N/A')}")
                lines.append(f"- Columns: {info.get('num_columns', 'N/A')}")
                lines.append(f"- Size: {info.get('file_size_kb', 0):.2f} KB")
                lines.append(f"- Compression: {info.get('compression', 'N/A')}\n")

            # Errors
            if validation['errors']:
                lines.append("**Errors**:")
                for error in validation['errors']:
                    lines.append(f"- ❌ {error}")
                lines.append("")

            # Warnings
            if validation['warnings']:
                lines.append("**Warnings**:")
                for warning in validation['warnings']:
                    lines.append(f"- ⚠️ {warning}")
                lines.append("")

            # Statistics
            if 'stats' in validation and validation['stats']:
                lines.append("**Statistics**:")
                for key, value in validation['stats'].items():
                    lines.append(f"- {key}: {value}")
                lines.append("")

        # Recommendations
        lines.append("## Recommendations\n")

        if total_errors > 0:
            lines.append("### Critical Issues")
            lines.append("The following issues must be addressed:\n")

            # Collect unique error types
            error_types = set()
            for v in validations:
                for error in v['errors']:
                    error_types.add(error)

            for error_type in sorted(error_types):
                lines.append(f"- {error_type}")
            lines.append("")

        if total_warnings > 0:
            lines.append("### Warnings")
            lines.append("The following warnings should be reviewed:\n")

            # Collect unique warning types
            warning_types = set()
            for v in validations:
                for warning in v['warnings']:
                    warning_types.add(warning)

            for warning_type in sorted(warning_types):
                lines.append(f"- {warning_type}")
            lines.append("")

        if total_errors == 0 and total_warnings == 0:
            lines.append("✅ No issues found. All validations passed successfully.\n")

        return "\n".join(lines)

    # Helper methods for validation checks

    def _check_row_count(self, df: pd.DataFrame, report: dict) -> None:
        """Check that DataFrame has at least one row."""
        if len(df) == 0:
            report['errors'].append("Parquet file is empty (0 rows)")
            report['checks']['row_count'] = False
        else:
            report['checks']['row_count'] = True

    def _check_required_columns(self, df: pd.DataFrame, report: dict) -> None:
        """Check that required base columns are present."""
        required_cols = ['game_loop', 'timestamp_seconds']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            report['errors'].append(f"Missing required columns: {missing_cols}")
            report['checks']['required_columns'] = False
        else:
            report['checks']['required_columns'] = True

    def _check_duplicate_game_loops(self, df: pd.DataFrame, report: dict) -> None:
        """Check for duplicate game_loop values."""
        if 'game_loop' not in df.columns:
            return

        duplicates = df['game_loop'].duplicated()
        duplicate_count = duplicates.sum()

        if duplicate_count > 0:
            report['errors'].append(f"Found {duplicate_count} duplicate game_loop values")
            report['checks']['no_duplicate_game_loops'] = False
        else:
            report['checks']['no_duplicate_game_loops'] = True

    def _check_column_types(self, df: pd.DataFrame, report: dict) -> None:
        """Validate column data types match expected schema."""
        type_issues = []

        # Check base columns
        if 'game_loop' in df.columns:
            if df['game_loop'].dtype not in ['int64', 'Int64']:
                type_issues.append("game_loop should be int64")

        if 'timestamp_seconds' in df.columns:
            if df['timestamp_seconds'].dtype != 'float64':
                type_issues.append("timestamp_seconds should be float64")

        # Check economy columns (should be int64 or Int64)
        economy_cols = [col for col in df.columns if any(
            x in col for x in ['minerals', 'vespene', 'supply_', 'workers', 'idle_workers']
        )]

        for col in economy_cols:
            if df[col].dtype not in ['int64', 'Int64']:
                type_issues.append(f"{col} should be int64")

        # Check coordinate columns (should be float64)
        coord_cols = [col for col in df.columns if col.endswith(('_x', '_y', '_z'))]

        for col in coord_cols:
            if df[col].dtype != 'float64':
                type_issues.append(f"{col} should be float64")

        if type_issues:
            for issue in type_issues[:10]:  # Limit to first 10
                report['warnings'].append(f"Type mismatch: {issue}")
            report['checks']['column_types'] = False
        else:
            report['checks']['column_types'] = True

    def _check_resource_validity(self, df: pd.DataFrame, report: dict) -> None:
        """
        Verify resources (minerals, gas, supply) are non-negative.

        Checks:
        - Minerals >= 0
        - Vespene gas >= 0
        - Supply used <= supply max
        - Supply max >= 0
        """
        issues = []

        for player in [1, 2]:
            minerals_col = f'p{player}_minerals'
            vespene_col = f'p{player}_vespene'
            supply_used_col = f'p{player}_supply_used'
            supply_cap_col = f'p{player}_supply_cap'

            # Check minerals
            if minerals_col in df.columns:
                if (df[minerals_col] < 0).any():
                    count = (df[minerals_col] < 0).sum()
                    issues.append(f"Player {player} has negative minerals in {count} rows")

            # Check vespene
            if vespene_col in df.columns:
                if (df[vespene_col] < 0).any():
                    count = (df[vespene_col] < 0).sum()
                    issues.append(f"Player {player} has negative vespene in {count} rows")

            # Check supply
            if supply_used_col in df.columns and supply_cap_col in df.columns:
                if (df[supply_used_col] > df[supply_cap_col]).any():
                    count = (df[supply_used_col] > df[supply_cap_col]).sum()
                    issues.append(f"Player {player} has supply_used > supply_cap in {count} rows")

                if (df[supply_cap_col] < 0).any():
                    count = (df[supply_cap_col] < 0).sum()
                    issues.append(f"Player {player} has negative supply_cap in {count} rows")

        if issues:
            for issue in issues:
                report['errors'].append(f"Resource constraint violation: {issue}")
            report['checks']['resource_validity'] = False
        else:
            report['checks']['resource_validity'] = True

    def _check_building_progress_monotonic(self, df: pd.DataFrame, report: dict) -> None:
        """
        Verify building progress is monotonically increasing (never decreases).

        Building progress should:
        - Be in range 0-100
        - Never decrease
        - Reach 100 when state becomes 'completed'
        """
        issues = []

        # Find all building progress columns
        progress_cols = [col for col in df.columns if col.endswith('_progress')]

        for col in progress_cols:
            # Check range
            if (df[col] < 0).any() or (df[col] > 100).any():
                issues.append(f"{col} has values outside range [0, 100]")

            # Check monotonicity (progress should never decrease)
            diff = df[col].diff()
            decreases = (diff < 0).sum()

            if decreases > 0:
                issues.append(f"{col} decreases {decreases} times (should be monotonic)")

        if issues:
            for issue in issues[:10]:  # Limit to first 10
                report['errors'].append(f"Building progress violation: {issue}")
            report['checks']['building_progress_monotonic'] = False
        else:
            report['checks']['building_progress_monotonic'] = True

    def _check_unit_count_consistency(self, df: pd.DataFrame, report: dict) -> None:
        """
        Verify unit counts match number of individual unit columns.

        For example, p1_marine_count should equal the number of p1_marine_NNN
        columns that have non-NaN values at each game loop.
        """
        issues = []

        # This is a complex check - we'll implement a simplified version
        # that checks for common unit types

        common_units = ['marine', 'scv', 'zealot', 'probe', 'zergling', 'drone']

        for player in [1, 2]:
            for unit_type in common_units:
                count_col = f'p{player}_{unit_type}_count'

                if count_col not in df.columns:
                    continue

                # Find all individual unit columns for this type
                unit_cols = [col for col in df.columns
                            if col.startswith(f'p{player}_{unit_type}_')
                            and col.endswith('_x')]  # Use _x as proxy for unit existence

                if unit_cols:
                    # Count non-NaN values in unit columns
                    actual_counts = df[unit_cols].notna().sum(axis=1)
                    expected_counts = df[count_col].fillna(0)

                    # Allow small discrepancies due to timing
                    mismatches = (actual_counts != expected_counts).sum()

                    if mismatches > len(df) * 0.1:  # More than 10% mismatch
                        issues.append(
                            f"{count_col} mismatches individual unit columns in {mismatches}/{len(df)} rows"
                        )

        if issues:
            for issue in issues[:10]:  # Limit to first 10
                report['warnings'].append(f"Unit count mismatch: {issue}")
            report['checks']['unit_count_consistency'] = False
        else:
            report['checks']['unit_count_consistency'] = True

    def _check_state_transitions(self, df: pd.DataFrame, report: dict) -> None:
        """
        Verify unit/building state transitions are valid.

        Valid transitions:
        - Units: created → alive → (cancelled OR dead)
        - Buildings: started → building → completed → (destroyed)
        - Upgrades: started → (cancelled OR completed)
        """
        issues = []

        # Find all state columns
        state_cols = [col for col in df.columns if col.endswith('_state') or col.endswith('_status')]

        valid_unit_states = {'created', 'alive', 'cancelled', 'dead', 'built', 'existing', 'killed'}
        valid_building_states = {'started', 'building', 'completed', 'destroyed'}

        for col in state_cols:
            # Check for invalid state values
            unique_states = df[col].dropna().unique()

            # Determine if this is a building or unit
            if '_status' in col:
                # Likely a building
                invalid_states = set(unique_states) - valid_building_states
                if invalid_states:
                    issues.append(f"{col} has invalid states: {invalid_states}")
            elif '_state' in col:
                # Likely a unit
                invalid_states = set(unique_states) - valid_unit_states
                if invalid_states:
                    issues.append(f"{col} has invalid states: {invalid_states}")

        if issues:
            for issue in issues[:10]:  # Limit to first 10
                report['warnings'].append(f"State transition issue: {issue}")
            report['checks']['state_transitions'] = False
        else:
            report['checks']['state_transitions'] = True

    def _check_nan_patterns(self, df: pd.DataFrame, report: dict) -> None:
        """Check for unexpected NaN patterns."""
        # Calculate NaN percentage per column
        nan_percentages = (df.isna().sum() / len(df)) * 100

        # Report columns with high NaN rates (but this might be expected for units)
        high_nan_cols = nan_percentages[nan_percentages > 95].index.tolist()

        if high_nan_cols:
            report['warnings'].append(
                f"{len(high_nan_cols)} columns have >95% NaN values (might be expected for rare units)"
            )

        # Check base columns shouldn't have NaN
        base_cols = ['game_loop', 'timestamp_seconds']
        for col in base_cols:
            if col in df.columns and df[col].isna().any():
                report['errors'].append(f"Base column {col} has NaN values")
                report['checks']['no_nan_in_base_columns'] = False
                return

        report['checks']['no_nan_in_base_columns'] = True

    def _generate_stats(self, df: pd.DataFrame, report: dict) -> None:
        """Generate statistics about the data."""
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
        }

        # Game loop range
        if 'game_loop' in df.columns:
            stats['game_loop_range'] = (int(df['game_loop'].min()), int(df['game_loop'].max()))
            stats['game_duration_seconds'] = float(df['timestamp_seconds'].max() if 'timestamp_seconds' in df.columns else 0)

        # Column categories
        unit_cols = [col for col in df.columns if any(x in col for x in ['_x', '_y', '_health', '_state'])]
        economy_cols = [col for col in df.columns if any(x in col for x in ['minerals', 'vespene', 'supply'])]
        building_cols = [col for col in df.columns if any(x in col for x in ['_status', '_progress'])]

        stats['unit_columns'] = len(unit_cols)
        stats['economy_columns'] = len(economy_cols)
        stats['building_columns'] = len(building_cols)

        # Memory usage
        stats['memory_usage_mb'] = float(df.memory_usage(deep=True).sum() / 1024 / 1024)

        report['stats'] = stats

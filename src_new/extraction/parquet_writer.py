"""
ParquetWriter: Writes wide-format data to parquet files.

This component handles writing the wide-format game state data to parquet
files with proper compression and schema handling.

Supports two write modes:
1. Single-shot: accumulate all rows, write once (write_game_state / write_game_state_columnar)
2. Batch streaming: write numbered part files during extraction, then reconcile
   all parts into one final parquet with a unified schema (write_batch_part /
   reconcile_parts). This caps peak memory at batch_size * column_count instead
   of total_rows * column_count.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schema_manager import SchemaManager


logger = logging.getLogger(__name__)

# Column name for chat messages (stable base column).
# Extracted as a module-level constant so all references stay in sync.
MESSAGES_COLUMN = 'Messages'


class ParquetWriter:
    """
    Writes wide-format data to parquet files.

    This class handles conversion of row dictionaries to DataFrames, proper
    type handling, compression, and both writing and appending operations.
    """

    def __init__(self, compression: str = 'snappy'):
        """
        Initialize the ParquetWriter.

        Args:
            compression: Compression codec ('snappy', 'gzip', 'brotli', 'zstd', None)
        """
        self.compression = compression
        logger.info(f"ParquetWriter initialized with {compression} compression")

    def write_game_state(
        self,
        rows: List[Dict[str, Any]],
        output_path: Path,
        schema: SchemaManager
    ) -> None:
        """
        Write game state rows to parquet.

        Handles:
        - Type conversion according to schema
        - Compression
        - Proper parquet schema
        - Column ordering

        Args:
            rows: List of row dictionaries
            output_path: Path to output parquet file
            schema: SchemaManager with column definitions

        Raises:
            ValueError: If rows is empty
            IOError: If write fails

        # TODO: Test case - Write DataFrame to parquet
        # TODO: Test case - Read back and verify
        # TODO: Test case - Handle NaN values
        # TODO: Test case - Test compression
        """
        if not rows:
            raise ValueError("Cannot write empty rows list")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writing {len(rows)} rows to {output_path}")

        # Convert rows to DataFrame
        df = pd.DataFrame(rows)

        # Reorder columns according to schema
        schema_columns = schema.get_column_list()
        df = df.reindex(columns=schema_columns)

        # Convert types according to schema
        df = self._convert_types(df, schema)

        # Write to parquet
        try:
            df.to_parquet(
                output_path,
                engine='pyarrow',
                compression=self.compression,
                index=False,
            )
            logger.info(f"Successfully wrote {len(rows)} rows to {output_path}")
            logger.info(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")

        except Exception as e:
            logger.error(f"Failed to write parquet: {e}")
            raise IOError(f"Failed to write parquet: {e}")

    def write_game_state_columnar(
        self,
        column_data: Dict[str, list],
        output_path: Path,
        schema: SchemaManager
    ) -> None:
        """
        Write game state from column-oriented data (dict of lists) to parquet.

        This is faster than write_game_state() because pd.DataFrame(column_dict)
        can directly wrap each list as a Series, avoiding the expensive key-scanning
        step that pd.DataFrame(list_of_dicts) requires for wide tables.

        Args:
            column_data: Dict mapping column names to lists of values.
                         All lists must have the same length.
            output_path: Path to output parquet file
            schema: SchemaManager with column definitions

        Raises:
            ValueError: If column_data is empty
            IOError: If write fails
        """
        if not column_data:
            raise ValueError("Cannot write empty column_data")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine row count from first column's list length
        first_col = next(iter(column_data.values()))
        num_rows = len(first_col)
        logger.info(f"Writing {num_rows} rows to {output_path} (columnar path)")

        # Build DataFrame from column-oriented dict (fast path).
        # Clear the source dict immediately after construction to free the
        # Python lists while the DataFrame (backed by contiguous numpy arrays)
        # is still alive. This avoids holding two full copies of the data at
        # peak memory (Fix 1).
        df = pd.DataFrame(column_data)
        column_data.clear()

        # Reorder columns according to schema
        schema_columns = schema.get_column_list()
        df = df.reindex(columns=schema_columns)

        # Convert types according to schema
        df = self._convert_types(df, schema)

        # Write to parquet
        try:
            df.to_parquet(
                output_path,
                engine='pyarrow',
                compression=self.compression,
                index=False,
            )
            logger.info(f"Successfully wrote {num_rows} rows to {output_path}")
            logger.info(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")

        except Exception as e:
            logger.error(f"Failed to write parquet: {e}")
            raise IOError(f"Failed to write parquet: {e}")

    def append_rows(
        self,
        rows: List[Dict[str, Any]],
        output_path: Path,
        schema: Optional[SchemaManager] = None
    ) -> None:
        """
        Append rows to existing parquet (for streaming processing).

        Args:
            rows: List of row dictionaries
            output_path: Path to parquet file
            schema: Optional SchemaManager (if None, infer from rows)

        # TODO: Test case - Append to existing file
        # TODO: Test case - Verify data integrity after append
        """
        if not rows:
            logger.warning("No rows to append")
            return

        output_path = Path(output_path)

        # Convert new rows to DataFrame
        new_df = pd.DataFrame(rows)

        # If schema provided, enforce it
        if schema:
            schema_columns = schema.get_column_list()
            new_df = new_df.reindex(columns=schema_columns)
            new_df = self._convert_types(new_df, schema)

        # Check if file exists
        if output_path.exists():
            logger.info(f"Appending {len(rows)} rows to existing {output_path}")

            # Read existing data
            existing_df = pd.read_parquet(output_path)

            # Combine
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

            # Write back
            combined_df.to_parquet(
                output_path,
                engine='pyarrow',
                compression=self.compression,
                index=False,
            )

            logger.info(f"Successfully appended. Total rows: {len(combined_df)}")

        else:
            # File doesn't exist, just write
            logger.info(f"File doesn't exist, creating new file with {len(rows)} rows")
            new_df.to_parquet(
                output_path,
                engine='pyarrow',
                compression=self.compression,
                index=False,
            )

    def _convert_types(self, df: pd.DataFrame, schema: SchemaManager) -> pd.DataFrame:
        """
        Convert DataFrame types according to schema.

        Uses vectorized pandas operations instead of per-element .apply(lambda)
        for significantly faster conversion on wide tables (Fix 2). For object
        columns, .astype('string') natively converts non-NaN values to their
        string representation and maps NaN to pd.NA — identical to the old
        lambda but executed at C level inside pandas.

        Args:
            df: DataFrame to convert
            schema: SchemaManager with type definitions

        Returns:
            DataFrame with converted types

        Depends on / calls:
            - schema.get_dtype() for per-column type lookup
            - _serialize_messages_for_parquet() for the Messages column
        """
        for col in df.columns:
            dtype = schema.get_dtype(col)

            try:
                if dtype == 'int64':
                    # Use Int64 (nullable integer) to handle NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                elif dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
                elif dtype == 'string':
                    df[col] = df[col].astype('string')
                elif dtype == 'bool':
                    df[col] = df[col].astype('boolean')
                elif dtype == 'object':
                    # Special handling for object columns that may contain lists
                    # (like Messages column which can be NaN, string, or list of strings)
                    # Convert lists to JSON strings for parquet compatibility
                    if col == MESSAGES_COLUMN:
                        df[col] = df[col].apply(self._serialize_messages_for_parquet)
                    else:
                        # Unit/building attribute columns contain mixed types:
                        # numeric values (float) for real data AND lifecycle state
                        # strings ("unit_started", "building", "completed", "destroyed",
                        # "cancelled", "building_started", "under_construction").
                        # PyArrow cannot handle mixed float+string columns, so convert
                        # everything to pandas StringDtype. NaN values are preserved as
                        # pd.NA automatically by .astype('string'). This is vectorized
                        # (C-level) and replaces the old per-element .apply(lambda).
                        df[col] = df[col].astype('string')
                else:
                    # Default to object
                    logger.warning(f"Unknown dtype '{dtype}' for column '{col}', using object")
                    df[col] = df[col].astype('object')

            except Exception as e:
                logger.warning(f"Failed to convert column '{col}' to {dtype}: {e}")

        return df

    def _serialize_messages_for_parquet(self, value):
        """
        Serialize Messages column values for parquet storage.

        PyArrow parquet doesn't support mixed types (strings and lists) in the same column,
        so we convert lists to JSON strings while keeping NaN and plain strings as-is.

        Args:
            value: Message value (NaN, string, or list of strings)

        Returns:
            NaN, string, or JSON-serialized string
        """
        if pd.isna(value):
            return value
        elif isinstance(value, str):
            return value
        elif isinstance(value, list):
            # Convert list to JSON string
            return json.dumps(value)
        else:
            # Fallback for unexpected types
            return str(value)

    def _deserialize_messages_from_parquet(self, value):
        """
        Deserialize Messages column values after reading from parquet.

        Converts JSON strings back to lists while keeping NaN and plain strings as-is.

        Args:
            value: Message value from parquet (NaN, string, or JSON string)

        Returns:
            NaN, string, or list of strings
        """
        if pd.isna(value):
            return value
        elif isinstance(value, str):
            # Try to parse as JSON (it might be a list)
            if value.startswith('['):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    # Not valid JSON, return as-is
                    return value
            else:
                return value
        else:
            return value

    def read_parquet(self, parquet_path: Path) -> pd.DataFrame:
        """
        Read parquet file into DataFrame.

        Args:
            parquet_path: Path to parquet file

        Returns:
            DataFrame with parquet data

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        parquet_path = Path(parquet_path)

        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        logger.info(f"Reading parquet from {parquet_path}")
        df = pd.read_parquet(parquet_path)
        logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

        # Deserialize Messages column if present
        if MESSAGES_COLUMN in df.columns:
            df[MESSAGES_COLUMN] = df[MESSAGES_COLUMN].apply(self._deserialize_messages_from_parquet)
            logger.info("  Deserialized Messages column from parquet")

        return df

    def get_parquet_info(self, parquet_path: Path) -> Dict[str, Any]:
        """
        Get information about a parquet file without loading all data.

        Args:
            parquet_path: Path to parquet file

        Returns:
            Dictionary with file information:
            {
                'num_rows': int,
                'num_columns': int,
                'file_size_kb': float,
                'columns': list,
                'compression': str,
            }
        """
        parquet_path = Path(parquet_path)

        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        # Read parquet metadata
        parquet_file = pq.ParquetFile(parquet_path)
        metadata = parquet_file.metadata

        info = {
            'num_rows': metadata.num_rows,
            'num_columns': metadata.num_columns,
            'file_size_kb': parquet_path.stat().st_size / 1024,
            'columns': [parquet_file.schema.names[i] for i in range(metadata.num_columns)],
            'compression': metadata.row_group(0).column(0).compression,
        }

        return info

    def validate_parquet(self, parquet_path: Path, schema: SchemaManager) -> Dict[str, Any]:
        """
        Validate a parquet file against expected schema.

        Args:
            parquet_path: Path to parquet file
            schema: Expected schema

        Returns:
            Validation report dictionary:
            {
                'valid': bool,
                'errors': list,
                'warnings': list,
                'info': dict,
            }
        """
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {},
        }

        try:
            # Get file info
            info = self.get_parquet_info(parquet_path)
            report['info'] = info

            # Check columns
            expected_columns = set(schema.get_column_list())
            actual_columns = set(info['columns'])

            missing_columns = expected_columns - actual_columns
            extra_columns = actual_columns - expected_columns

            if missing_columns:
                report['errors'].append(f"Missing columns: {list(missing_columns)[:10]}")
                report['valid'] = False

            if extra_columns:
                report['warnings'].append(f"Extra columns: {list(extra_columns)[:10]}")

            # Check row count
            if info['num_rows'] == 0:
                report['errors'].append("Parquet file is empty")
                report['valid'] = False

        except Exception as e:
            report['errors'].append(f"Validation failed: {e}")
            report['valid'] = False

        return report

    def write_batch_part(
        self,
        column_data: Dict[str, list],
        parts_dir: Path,
        part_index: int,
        schema: SchemaManager,
    ) -> Path:
        """
        Write one batch of column-oriented data to a numbered part file.

        Each part file is a standalone parquet file with whatever columns exist
        in the batch. Later, reconcile_parts() reads all parts and unifies them
        into a single parquet with a consistent schema (columns that don't exist
        in earlier parts are filled with NaN by PyArrow during concatenation).

        After building the DataFrame from column_data, the source dict is
        cleared immediately to free Python-list memory (Fix 1).

        Args:
            column_data: Dict mapping column names to lists of values.
                         All lists must have the same length.
            parts_dir: Directory to write part files into.
            part_index: Zero-based batch number (used in filename).
            schema: SchemaManager with column definitions and type info.

        Returns:
            Path to the written part file.

        Depends on / calls:
            - pd.DataFrame() for column-oriented construction
            - _convert_types() for schema-driven type casting
            - df.to_parquet() for writing
        """
        parts_dir.mkdir(parents=True, exist_ok=True)
        part_path = parts_dir / f"_part_{part_index:04d}.parquet"

        first_col = next(iter(column_data.values()))
        num_rows = len(first_col)
        logger.info(
            f"Writing batch part {part_index}: {num_rows} rows, "
            f"{len(column_data)} columns -> {part_path.name}"
        )

        # Build DataFrame and free the source lists immediately (Fix 1).
        df = pd.DataFrame(column_data)
        column_data.clear()

        # Reorder to match schema (only columns present in this batch)
        schema_columns = [c for c in schema.get_column_list() if c in df.columns]
        df = df.reindex(columns=schema_columns)

        # Convert types
        df = self._convert_types(df, schema)

        df.to_parquet(
            part_path,
            engine='pyarrow',
            compression=self.compression,
            index=False,
        )
        logger.info(
            f"  Part {part_index} written: {part_path.stat().st_size / 1024:.1f} KB"
        )
        return part_path

    def reconcile_parts(
        self,
        parts_dir: Path,
        output_path: Path,
    ) -> None:
        """
        Read all part files from parts_dir and write a single unified parquet.

        PyArrow handles schema reconciliation automatically: columns that exist
        in some parts but not others are filled with null. The result is
        identical to what write_game_state_columnar would produce from the full
        column_data dict, but peak memory only needs to hold one part at a time
        during writing plus the final concatenated table.

        After reconciliation, the part files are deleted.

        Args:
            parts_dir: Directory containing _part_NNNN.parquet files.
            output_path: Path to the final unified parquet file.

        Depends on / calls:
            - pq.read_table() for reading individual parts
            - pa.concat_tables() for schema-aware concatenation
            - pq.write_table() for writing the final file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect part files in order
        part_files = sorted(parts_dir.glob("_part_*.parquet"))
        if not part_files:
            raise ValueError(f"No part files found in {parts_dir}")

        logger.info(
            f"Reconciling {len(part_files)} part files into {output_path}"
        )

        # Read all parts as PyArrow tables. promote_options='default' fills
        # missing columns with null when schemas differ across parts.
        tables = []
        for pf in part_files:
            tables.append(pq.read_table(pf))

        unified = pa.concat_tables(tables, promote_options='default')
        logger.info(
            f"  Unified table: {unified.num_rows} rows x "
            f"{unified.num_columns} columns"
        )

        # Write final parquet
        pq.write_table(
            unified,
            output_path,
            compression=self.compression,
        )
        logger.info(
            f"  Final parquet: {output_path.stat().st_size / 1024:.1f} KB"
        )

        # Clean up part files and directory
        for pf in part_files:
            pf.unlink()
        # Remove parts directory if empty
        try:
            parts_dir.rmdir()
        except OSError:
            pass

    def write_batch_streaming(
        self,
        rows_iterator,
        output_path: Path,
        schema: SchemaManager,
        batch_size: int = 1000
    ) -> int:
        """
        Write rows in batches for memory-efficient streaming.

        NOTE: This is the older list-of-dicts streaming method. For the
        column-oriented batch approach used by the extraction pipeline, see
        write_batch_part() + reconcile_parts().

        Args:
            rows_iterator: Iterator yielding row dictionaries
            output_path: Path to output parquet file
            schema: SchemaManager with column definitions
            batch_size: Number of rows per batch

        Returns:
            Total number of rows written
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_rows = 0
        batch = []

        logger.info(f"Starting streaming write to {output_path} (batch_size={batch_size})")

        for row in rows_iterator:
            batch.append(row)

            if len(batch) >= batch_size:
                # Write batch
                if total_rows == 0:
                    # First batch - create file
                    self.write_game_state(batch, output_path, schema)
                else:
                    # Subsequent batches - append
                    self.append_rows(batch, output_path, schema)

                total_rows += len(batch)
                logger.info(f"  Written {total_rows} rows...")
                batch = []

        # Write remaining rows
        if batch:
            if total_rows == 0:
                self.write_game_state(batch, output_path, schema)
            else:
                self.append_rows(batch, output_path, schema)
            total_rows += len(batch)

        logger.info(f"Streaming write complete. Total rows: {total_rows}")

        return total_rows

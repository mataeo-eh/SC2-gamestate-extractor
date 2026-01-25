"""
ParquetWriter: Writes wide-format data to parquet files.

This component handles writing the wide-format game state data to parquet
files with proper compression and schema handling.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schema_manager import SchemaManager


logger = logging.getLogger(__name__)


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

    def write_messages(
        self,
        messages: List[Dict[str, Any]],
        output_path: Path
    ) -> None:
        """
        Write messages to separate parquet.

        Schema: game_loop, player_id, message

        Args:
            messages: List of message dictionaries with keys:
                - game_loop: int
                - player_id: int
                - message: str
            output_path: Path to output parquet file

        # TODO: Test case - Write and read back messages
        """
        if not messages:
            logger.info("No messages to write")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writing {len(messages)} messages to {output_path}")

        # Convert to DataFrame
        df = pd.DataFrame(messages)

        # Ensure correct schema
        df = df[['game_loop', 'player_id', 'message']]
        df['game_loop'] = df['game_loop'].astype('int64')
        df['player_id'] = df['player_id'].astype('int64')
        df['message'] = df['message'].astype('string')

        # Write to parquet
        try:
            df.to_parquet(
                output_path,
                engine='pyarrow',
                compression=self.compression,
                index=False,
            )
            logger.info(f"Successfully wrote {len(messages)} messages to {output_path}")

        except Exception as e:
            logger.error(f"Failed to write messages parquet: {e}")
            raise IOError(f"Failed to write messages parquet: {e}")

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

        Args:
            df: DataFrame to convert
            schema: SchemaManager with type definitions

        Returns:
            DataFrame with converted types
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
                else:
                    # Default to object
                    logger.warning(f"Unknown dtype '{dtype}' for column '{col}', using object")
                    df[col] = df[col].astype('object')

            except Exception as e:
                logger.warning(f"Failed to convert column '{col}' to {dtype}: {e}")

        return df

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

    def write_batch_streaming(
        self,
        rows_iterator,
        output_path: Path,
        schema: SchemaManager,
        batch_size: int = 1000
    ) -> int:
        """
        Write rows in batches for memory-efficient streaming.

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

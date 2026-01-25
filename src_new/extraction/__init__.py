"""
Extraction module for SC2 replay ground truth extraction.

This module contains the core extraction components that orchestrate
the replay loading, state extraction, schema management, wide table
building, and parquet writing.

Components:
- replay_loader: Loads SC2 replays with pysc2 (wrapper around pipeline.replay_loader)
- state_extractor: Extracts complete game state from observations
- schema_manager: Manages wide-table column schema
- wide_table_builder: Transforms extracted state to wide-format rows
- parquet_writer: Writes wide-format data to parquet files
"""

from .replay_loader import ReplayLoader
from .state_extractor import StateExtractor, UnitTracker, BuildingTracker
from .schema_manager import SchemaManager
from .wide_table_builder import WideTableBuilder
from .parquet_writer import ParquetWriter

__all__ = [
    'ReplayLoader',
    'StateExtractor',
    'UnitTracker',
    'BuildingTracker',
    'SchemaManager',
    'WideTableBuilder',
    'ParquetWriter',
]

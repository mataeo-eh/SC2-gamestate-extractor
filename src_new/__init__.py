"""
SC2 Replay Ground Truth Extraction Pipeline

This package provides a comprehensive system for extracting complete ground truth
game state from StarCraft II replays using pysc2.

Main components:
- pipeline: Core processing components (ReplayLoader, GameLoopIterator, etc.)
- extractors: Data extraction modules (units, buildings, economy, upgrades)
- extraction: High-level extraction orchestration (StateExtractor, SchemaManager, etc.)
- batch: Parallel processing and batch control
- utils: Utility functions and helpers
"""

__version__ = "1.0.0"
__author__ = "SC2 Ground Truth Extraction Team"

"""
Pipeline components for SC2 replay processing.

This module contains the core components that make up the extraction pipeline:
- ReplayLoader: Loads and initializes replays with pysc2
- GameLoopIterator: Steps through game loops and yields observations
- ReplayExtractionPipeline: Main end-to-end pipeline orchestrator
- ParallelReplayProcessor: Batch processing with multiprocessing
"""

from .replay_loader import ReplayLoader
from .game_loop_iterator import GameLoopIterator
from .extraction_pipeline import ReplayExtractionPipeline, process_replay_quick
from .parallel_processor import ParallelReplayProcessor, process_directory_quick

__all__ = [
    'ReplayLoader',
    'GameLoopIterator',
    'ReplayExtractionPipeline',
    'ParallelReplayProcessor',
    'process_replay_quick',
    'process_directory_quick',
]

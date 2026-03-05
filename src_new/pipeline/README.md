# pipeline/

Pipeline orchestration, batch processing, and supporting infrastructure.

## Modules

- **`extraction_pipeline.py`** - `ReplayExtractionPipeline`: the main single-replay orchestrator. Coordinates replay loading, game loop iteration, state extraction, and parquet writing end-to-end.
- **`parallel_processor.py`** - `ParallelReplayProcessor`: distributes replays across a `ProcessPoolExecutor` worker pool. Skips already-processed files, isolates worker failures, and aggregates batch results.
- **`game_loop_iterator.py`** - Iterates through game loops with configurable step size.
- **`dataset_pipeline.py`** - Uploads processed parquet datasets to Kaggle.
- **`logging_config.py`** - Centralized file-based and console logging setup.
- **`integration_check.py`** - Validates SC2 installation and dependency availability.

## Documentation

- `ARCHITECTURE.md` - Full system architecture diagrams and component responsibilities.
- `USAGE_EXAMPLES.md` - CLI usage examples.

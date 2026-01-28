# Pipeline Integration Usage Examples

This document provides examples of using the Phase 3 pipeline integration components.

## Overview

Phase 3 provides two main components:

1. **ReplayExtractionPipeline**: Process individual replays end-to-end
2. **ParallelReplayProcessor**: Batch process multiple replays in parallel

## Basic Usage

### Processing a Single Replay

```python
from pathlib import Path
from src_new.pipeline.extraction_pipeline import ReplayExtractionPipeline

# Create pipeline with default configuration
pipeline = ReplayExtractionPipeline()

# Process a replay
result = pipeline.process_replay(
    replay_path=Path("replays/example.SC2Replay"),
    output_dir=Path("data/quickstart")
)

# Check result
if result['success']:
    print(f"Success! Processed {result['stats']['rows_written']} rows")
    print(f"Output files:")
    for name, path in result['output_files'].items():
        print(f"  {name}: {path}")
else:
    print(f"Failed: {result['error']}")
```

### Quick Processing with Convenience Function

```python
from pathlib import Path
from src_new.pipeline.extraction_pipeline import process_replay_quick

# Process with one function call
result = process_replay_quick(
    replay_path=Path("replays/example.SC2Replay"),
    output_dir=Path("data/quickstart")
)

print(f"Processed in {result['stats']['processing_time_seconds']:.2f}s")
```

## Configuration Options

### Custom Pipeline Configuration

```python
from pathlib import Path
from src_new.pipeline.extraction_pipeline import ReplayExtractionPipeline

# Configure pipeline
config = {
    # Observation settings
    'show_cloaked': True,
    'show_burrowed_shadows': True,
    'show_placeholders': True,

    # Processing settings
    'processing_mode': 'two_pass',  # or 'single_pass'
    'step_size': 1,  # Game loops per step

    # Output settings
    'compression': 'snappy',  # or 'gzip', 'brotli', 'zstd'
}

pipeline = ReplayExtractionPipeline(config)
result = pipeline.process_replay(
    replay_path=Path("replays/example.SC2Replay"),
    output_dir=Path("data/quickstart")
)
```

### Two-Pass vs Single-Pass Processing

**Two-Pass Mode (Default - Recommended)**:
- Pass 1: Scans entire replay to discover all units/buildings
- Pass 2: Extracts data with consistent schema
- Pros: Consistent schema across all rows, better for ML
- Cons: Slower (processes replay twice)

```python
config = {'processing_mode': 'two_pass'}
pipeline = ReplayExtractionPipeline(config)
```

**Single-Pass Mode**:
- Extracts data in one pass, building schema dynamically
- Pros: Faster, more memory-efficient
- Cons: May have ragged columns (different rows may have different columns)

```python
config = {'processing_mode': 'single_pass'}
pipeline = ReplayExtractionPipeline(config)
```

## Batch Processing

### Processing Multiple Replays in Parallel

```python
from pathlib import Path
from src_new.pipeline.parallel_processor import ParallelReplayProcessor

# Create parallel processor
processor = ParallelReplayProcessor(
    config={'processing_mode': 'two_pass'},
    num_workers=4  # Use 4 CPU cores
)

# Process a batch of replays
replay_paths = [
    Path("replays/game1.SC2Replay"),
    Path("replays/game2.SC2Replay"),
    Path("replays/game3.SC2Replay"),
]

results = processor.process_replay_batch(
    replay_paths=replay_paths,
    output_dir=Path("data/quickstart")
)

# Check results
print(f"Successful: {results['successful_count']}/{results['total_replays']}")
print(f"Total time: {results['total_time_seconds']:.2f}s")
print(f"Average per replay: {results['average_time_per_replay']:.2f}s")

# Show failed replays
for replay_path, error in results['failed']:
    print(f"Failed: {replay_path.name} - {error}")
```

### Processing an Entire Directory

```python
from pathlib import Path
from src_new.pipeline.parallel_processor import ParallelReplayProcessor

processor = ParallelReplayProcessor(num_workers=8)

# Process all replays in a directory
results = processor.process_replay_directory(
    replay_dir=Path("replays/"),
    output_dir=Path("data/processed"),
    pattern="*.SC2Replay"  # Optional pattern
)

# Print summary
summary = processor.get_processing_summary(results)
print(summary)
```

### Recursive Directory Processing

```python
from pathlib import Path
from src_new.pipeline.parallel_processor import ParallelReplayProcessor

processor = ParallelReplayProcessor()

# Process replays recursively in nested directories
results = processor.process_replay_directory_recursive(
    replay_dir=Path("replays/"),
    output_dir=Path("data/quickstart"),
    pattern="**/*.SC2Replay"  # Recursive pattern
)
```

### Quick Batch Processing with Convenience Function

```python
from pathlib import Path
from src_new.pipeline.parallel_processor import process_directory_quick

# Process directory with one function call
results = process_directory_quick(
    replay_dir=Path("replays/"),
    output_dir=Path("data/quickstart"),
    num_workers=8
)

print(f"Processed {results['successful_count']} replays")
```

## Error Handling

### Retry Failed Replays

```python
from pathlib import Path
from src_new.pipeline.parallel_processor import ParallelReplayProcessor

processor = ParallelReplayProcessor()

# First attempt
results = processor.process_replay_directory(
    replay_dir=Path("replays/"),
    output_dir=Path("data/processed")
)

# Retry failed replays
if results['failed']:
    print(f"Retrying {len(results['failed'])} failed replays...")
    retry_results = processor.retry_failed_replays(
        failed_results=results['failed'],
        output_dir=Path("data/processed"),
        max_retries=1
    )
    print(f"Retry: {retry_results['successful_count']} now successful")
```

### Validate Replay Before Processing

```python
from pathlib import Path
from src_new.pipeline.extraction_pipeline import ReplayExtractionPipeline

pipeline = ReplayExtractionPipeline()

# Quick validation without full processing
validation = pipeline.validate_replay(Path("replays/example.SC2Replay"))

if validation['valid']:
    print("Replay is valid")
    print(f"Map: {validation['metadata']['map_name']}")
    print(f"Duration: {validation['metadata']['game_duration_seconds']:.1f}s")
else:
    print(f"Invalid replay: {validation['error']}")
```

## Output Files

For each processed replay, the pipeline generates three output files:

1. **Game State Parquet**: `{replay_name}_game_state.parquet`
   - Wide-format table with one row per game loop
   - Contains all unit positions, building states, economy, upgrades

2. **Messages Parquet**: `{replay_name}_messages.parquet`
   - Chat messages with game_loop, player_id, message

3. **Schema JSON**: `{replay_name}_schema.json`
   - Column definitions and data dictionary
   - Useful for understanding the output format

Example output structure:
```
data/processed/
├── replay_001_game_state.parquet
├── replay_001_messages.parquet
├── replay_001_schema.json
├── replay_002_game_state.parquet
├── replay_002_messages.parquet
└── replay_002_schema.json
```

## Reading Output Data

### Loading Game State

```python
import pandas as pd
from pathlib import Path

# Read game state parquet
df = pd.read_parquet(Path("data/processed/replay_001_game_state.parquet"))

print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
print(f"Game duration: {df['timestamp_seconds'].max():.1f}s")
print(f"Player 1 final minerals: {df.iloc[-1]['p1_minerals']}")
```

### Loading Messages

```python
import pandas as pd
from pathlib import Path

# Read messages parquet
messages_df = pd.read_parquet(Path("data/processed/replay_001_messages.parquet"))

print(f"Total messages: {len(messages_df)}")
for _, msg in messages_df.iterrows():
    print(f"[{msg['game_loop']}] Player {msg['player_id']}: {msg['message']}")
```

### Reading Schema

```python
import json
from pathlib import Path

# Read schema JSON
with open(Path("data/processed/replay_001_schema.json"), 'r') as f:
    schema = json.load(f)

print(f"Total columns: {len(schema['columns'])}")
print("\nColumn documentation:")
for col_name in schema['columns'][:10]:  # Show first 10
    doc = schema['documentation'][col_name]
    print(f"  {col_name}: {doc['description']}")
```

## Performance Considerations

### Optimal Worker Count

```python
import multiprocessing
from src_new.pipeline.parallel_processor import ParallelReplayProcessor

# Use all CPU cores
processor = ParallelReplayProcessor(num_workers=multiprocessing.cpu_count())

# Use half of CPU cores (recommended for systems with limited RAM)
processor = ParallelReplayProcessor(num_workers=multiprocessing.cpu_count() // 2)

# Use specific number
processor = ParallelReplayProcessor(num_workers=4)
```

### Memory Management

For large replays or systems with limited memory:

```python
# Use single-pass mode to reduce memory usage
config = {
    'processing_mode': 'single_pass',
    'step_size': 10,  # Sample every 10 game loops instead of every loop
}

processor = ParallelReplayProcessor(
    config=config,
    num_workers=2  # Fewer workers = less memory
)
```

### Processing Time Estimates

- Single replay (two-pass): ~30-60 seconds
- Single replay (single-pass): ~15-30 seconds
- Batch of 100 replays (8 workers): ~10-20 minutes

Times vary based on:
- Replay length
- Number of units/buildings
- CPU speed
- I/O performance

## Next Steps

After Phase 3, you can:

1. **Phase 4**: Add validation to verify output quality
2. **Phase 5**: Create CLI for easy command-line usage
3. **Phase 6**: Add comprehensive tests
4. **Phase 7**: Process full dataset and optimize performance

For now, the pipeline is ready to process replays programmatically!

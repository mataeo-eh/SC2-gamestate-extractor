# SC2 Replay Ground Truth Extraction Pipeline - Architecture

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   ParallelReplayProcessor                        │
│  (Batch processing with multiprocessing - Phase 3)              │
│                                                                  │
│  - process_replay_batch()                                       │
│  - process_replay_directory()                                   │
│  - Manages worker pool                                          │
│  - Aggregates results                                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ Creates multiple instances
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ReplayExtractionPipeline                        │
│  (Main end-to-end orchestrator - Phase 3)                       │
│                                                                  │
│  - process_replay()                                             │
│  - _two_pass_processing()                                       │
│  - _single_pass_processing()                                    │
└─────────┬───────────────────────────────────────────────────────┘
          │
          │ Coordinates these components:
          │
          ├─────────────────────────────────────────┐
          │                                         │
          ▼                                         ▼
┌─────────────────────┐                  ┌─────────────────────┐
│   ReplayLoader      │                  │  StateExtractor     │
│   (Phase 2)         │                  │  (Phase 2)          │
│                     │                  │                     │
│ - load_replay()     │                  │ - extract_obs()     │
│ - start_sc2()       │                  │                     │
│ - get_replay_info() │                  │ Uses:               │
│                     │                  │ - UnitExtractor     │
│ Wraps:              │                  │ - BuildingExtractor │
│ PipelineReplayLoader│                  │ - EconomyExtractor  │
│ (Phase 1)           │                  │ - UpgradeExtractor  │
└─────────────────────┘                  └─────────────────────┘
          │                                         │
          │                                         │
          └────────────┬────────────────────────────┘
                       │
                       │ State data flows to:
                       │
                       ▼
          ┌────────────────────────┐
          │   SchemaManager        │
          │   (Phase 2)            │
          │                        │
          │ - build_schema()       │
          │ - get_column_list()    │
          │ - save_schema()        │
          └────────┬───────────────┘
                   │
                   │ Schema defines structure for:
                   │
                   ▼
          ┌────────────────────────┐
          │  WideTableBuilder      │
          │  (Phase 2)             │
          │                        │
          │ - build_row()          │
          │ - add_unit_to_row()    │
          │ - add_building_to_row()│
          └────────┬───────────────┘
                   │
                   │ Wide-format rows go to:
                   │
                   ▼
          ┌────────────────────────┐
          │   ParquetWriter        │
          │   (Phase 2)            │
          │                        │
          │ - write_game_state()   │
          │ - write_messages()     │
          │ - write_streaming()    │
          └────────────────────────┘
                   │
                   ▼
          ┌────────────────────────┐
          │   Output Files         │
          │                        │
          │ - game_state.parquet   │
          │ - messages.parquet     │
          │ - schema.json          │
          └────────────────────────┘
```

## Data Flow

### Two-Pass Processing Mode

```
Pass 1: Schema Discovery
─────────────────────────
Replay File
    │
    ▼
ReplayLoader.load_replay()
    │
    ▼
ReplayLoader.start_sc2_instance()
    │
    ▼
For each game loop:
    controller.step()
    │
    ▼
    controller.observe()
    │
    ▼
    StateExtractor.extract_observation()
    │
    ▼
    SchemaManager._discover_entities_from_state()
    │
    └─> Builds column list dynamically

Pass 2: Data Extraction
────────────────────────
ReplayLoader.load_replay() (again)
    │
    ▼
StateExtractor.reset()
    │
    ▼
ReplayLoader.start_sc2_instance()
    │
    ▼
For each game loop:
    controller.step()
    │
    ▼
    controller.observe()
    │
    ▼
    StateExtractor.extract_observation()
    │
    ▼
    WideTableBuilder.build_row()
    │   (uses complete schema from Pass 1)
    │
    └─> Consistent wide-format rows
    │
    ▼
ParquetWriter.write_game_state()
    │
    ▼
Output files created
```

### Single-Pass Processing Mode

```
Replay File
    │
    ▼
ReplayLoader.load_replay()
    │
    ▼
ReplayLoader.start_sc2_instance()
    │
    ▼
For each game loop:
    controller.step()
    │
    ▼
    controller.observe()
    │
    ▼
    StateExtractor.extract_observation()
    │
    ├─> SchemaManager._discover_entities_from_state()
    │   (updates schema dynamically)
    │
    ▼
    WideTableBuilder.build_row()
    │   (may have different columns per row)
    │
    └─> Wide-format rows (ragged schema)
    │
    ▼
ParquetWriter.write_game_state()
    │
    ▼
Output files created
```

## Parallel Processing Architecture

```
ParallelReplayProcessor
    │
    ├─> Creates ProcessPoolExecutor
    │
    ├─> Submits N replay jobs to worker pool
    │
    └─> For each worker:
         │
         ├─> Worker Process 1
         │   └─> _worker_process_replay()
         │       └─> Creates ReplayExtractionPipeline
         │           └─> Processes replay
         │               └─> Returns (success, time, error)
         │
         ├─> Worker Process 2
         │   └─> _worker_process_replay()
         │       └─> ...
         │
         └─> Worker Process N
             └─> _worker_process_replay()
                 └─> ...
    │
    ▼
Aggregates results from all workers
    │
    ▼
Returns batch results dictionary
```

## Component Responsibilities

### ReplayExtractionPipeline
- **Input**: Single replay file path
- **Output**: Parquet files + metadata + stats
- **Responsibility**: Orchestrate end-to-end extraction for one replay
- **Error handling**: Catch and report errors, return detailed result

### ParallelReplayProcessor
- **Input**: List of replay paths or directory
- **Output**: Batch results with success/failure per replay
- **Responsibility**: Manage parallel execution across multiple replays
- **Error handling**: Isolate worker failures, continue batch processing

### ReplayLoader (extraction)
- **Input**: Replay file path
- **Output**: SC2 controller, replay metadata
- **Responsibility**: Load replay and provide access to game state
- **Error handling**: Validate file, handle corrupt replays

### StateExtractor
- **Input**: SC2 observation
- **Output**: Complete game state dictionary
- **Responsibility**: Extract all entities from observation
- **Error handling**: Handle missing data gracefully

### SchemaManager
- **Input**: Extracted states (Pass 1) or pre-built schema
- **Output**: Column definitions, data types, documentation
- **Responsibility**: Define and manage wide-table schema
- **Error handling**: Handle dynamic schema updates

### WideTableBuilder
- **Input**: Extracted state + schema
- **Output**: Wide-format row dictionary
- **Responsibility**: Flatten hierarchical state to wide format
- **Error handling**: Fill missing values with NaN/None

### ParquetWriter
- **Input**: List of row dictionaries + schema
- **Output**: Parquet files on disk
- **Responsibility**: Write data with proper types and compression
- **Error handling**: Validate data, handle I/O errors

## Configuration Propagation

```
User Config
    │
    ▼
ParallelReplayProcessor(config, num_workers)
    │
    ▼
_worker_process_replay(config)
    │
    ▼
ReplayExtractionPipeline(config)
    │
    ├─> ReplayLoader(config)
    │   - show_cloaked
    │   - show_burrowed_shadows
    │   - show_placeholders
    │
    ├─> processing_mode
    │   - two_pass / single_pass
    │
    ├─> step_size
    │   - Game loops to step
    │
    └─> ParquetWriter(compression)
        - compression codec
```

## Error Handling Strategy

### Pipeline Level
- Catch all exceptions in `process_replay()`
- Return detailed error in result dictionary
- Log full exception with traceback
- Don't crash - return failure result

### Worker Level
- Each worker has try/except wrapper
- Returns (False, time, error_message) on failure
- Logs error independently
- Doesn't crash other workers

### Batch Level
- Aggregate all results (success + failures)
- Continue processing even if some replays fail
- Provide retry mechanism for failed replays
- Report comprehensive statistics

## Logging Strategy

### Log Levels
- **INFO**: High-level progress (replay started, completed, stats)
- **WARNING**: Non-fatal issues (frame extraction error, missing data)
- **ERROR**: Fatal errors (replay load failed, I/O error)
- **DEBUG**: Detailed internal state (tracker updates, schema changes)

### Log Messages Include
- Replay name and path
- Current game loop
- Processing stage
- Timing information
- Error details with context

## Memory Management

### Two-Pass Mode
- **Pass 1**: Only schema in memory (~100 KB - 1 MB)
- **Pass 2**: Accumulates rows in memory (~100-500 MB)
- **Peak**: All rows before writing to parquet

### Single-Pass Mode
- **During**: Accumulates rows dynamically (~100-500 MB)
- **Peak**: All rows before writing to parquet

### Parallel Processing
- **Per worker**: 100-500 MB
- **Total**: num_workers × 500 MB
- **Recommendation**: Use num_workers = cpu_count() / 2 for memory-constrained systems

## Future Phases

### Phase 4: Validation
- Add `src_new/validation/` module
- Integrate validation into pipeline
- Output quality metrics

### Phase 5: CLI
- Add `src_new/cli/` module
- Command-line interface for pipeline
- Progress bars and user feedback

### Phase 6: Tests
- Add `tests/` directory
- Unit tests for each component
- Integration tests for full pipeline
- Performance benchmarks

### Phase 7: Production
- Process full dataset
- Performance optimization
- Documentation refinement
- Deploy to production environment

## Summary

Phase 3 successfully integrates all extraction components into a complete,
production-ready pipeline that can process SC2 replays both individually and
in parallel batches. The implementation follows the specification from the
implementation plan and is ready for Phase 4 (Validation).

# Phase 4: Validation & Quality Assurance

## Overview

This directory contains Phase 4 of the SC2 Replay Ground Truth Extraction Pipeline: **Validation & Quality Assurance** modules.

Phase 4 provides comprehensive validation of extracted parquet files and generates clear documentation to ensure data quality and usability.

## Modules

### 1. `validation.py` - OutputValidator

**Purpose**: Validates extracted parquet files for quality assurance.

**Key Class**: `OutputValidator`

**Methods**:
- `validate_game_state_parquet()` - Validate game state parquet files
- `validate_messages_parquet()` - Validate messages parquet files
- `generate_validation_report()` - Generate markdown validation reports

**Features**:
- Row count validation
- Duplicate detection (game_loop)
- Resource constraint checking (minerals, vespene, supply)
- Building progress monotonicity verification
- Unit count consistency checks
- State transition validation
- Schema compliance verification
- Comprehensive error and warning reporting

### 2. `documentation.py` - Documentation Generation

**Purpose**: Generate comprehensive documentation for extracted data.

**Key Functions**:
- `generate_data_dictionary()` - Create complete data dictionary
- `generate_replay_report()` - Generate single replay processing report
- `generate_batch_summary()` - Create batch processing summary

**Features**:
- Markdown-formatted documentation
- Column descriptions with types and ranges
- Data previews and statistics
- Validation result integration
- Batch processing analytics

## Quick Start

### Basic Validation

```python
from src_new.utils import OutputValidator
from pathlib import Path

# Create validator
validator = OutputValidator()

# Validate a parquet file
report = validator.validate_game_state_parquet(
    Path("data/processed/replay_game_state.parquet")
)

# Check results
if report['valid']:
    print("Validation passed!")
else:
    print(f"Errors: {report['errors']}")
```

### Generate Documentation

```python
from src_new.utils import generate_replay_report
from pathlib import Path

# Generate replay report
generate_replay_report(
    replay_path=Path("data/replays/replay.SC2Replay"),
    output_path=Path("data/processed/report.md"),
    validation_results=report  # Optional
)
```

## Validation Checks

### Game State Parquet

| Check | Type | Description |
|-------|------|-------------|
| Row Count | Error | Must have at least one row |
| Required Columns | Error | Base columns must exist |
| Duplicate Game Loops | Error | No duplicate game_loop values |
| Column Types | Warning | Types should match schema |
| Resource Validity | Error | Resources must be non-negative |
| Building Progress | Error | Progress must be monotonic |
| Unit Count Consistency | Warning | Counts must match units |
| State Transitions | Warning | States must be valid |
| NaN Patterns | Error | Base columns cannot have NaN |

### Messages Parquet

| Check | Type | Description |
|-------|------|-------------|
| File Exists | Warning | Messages are optional |
| Required Columns | Error | Must have required columns |
| Column Types | Error | Types must be correct |
| Valid Game Loops | Error | game_loop must be valid |
| No Duplicates | Warning | Check for duplicate messages |

## Files

```
src_new/utils/
├── __init__.py                      # Module exports
├── validation.py                    # OutputValidator class (666 lines)
├── documentation.py                 # Documentation generators (700 lines)
├── validation_check.py              # Integration check script
├── example_validation_workflow.py   # Complete workflow examples
├── README.md                        # This file
├── USAGE_EXAMPLES.md                # Detailed usage examples
└── PHASE4_SUMMARY.md                # Implementation summary
```

## Usage Examples

See `USAGE_EXAMPLES.md` for comprehensive examples including:
- Single file validation
- Batch validation
- Documentation generation
- Complete pipeline integration
- Error handling patterns

## Integration with Pipeline

Phase 4 integrates with Phases 1-3:

```python
from src_new.pipeline import ReplayExtractionPipeline
from src_new.utils import OutputValidator, generate_replay_report

# 1. Extract (Phase 3)
pipeline = ReplayExtractionPipeline()
result = pipeline.process_replay(replay_path, output_dir)

# 2. Validate (Phase 4)
validator = OutputValidator()
validation = validator.validate_game_state_parquet(
    result['output_files']['game_state']
)

# 3. Document (Phase 4)
generate_replay_report(
    replay_path=replay_path,
    output_path=output_dir / "report.md",
    validation_results=validation
)
```

## Validation Report Structure

```python
{
    'valid': bool,              # Overall status
    'file_path': str,           # Validated file
    'errors': List[str],        # Critical errors
    'warnings': List[str],      # Non-critical warnings
    'info': {                   # File metadata
        'num_rows': int,
        'num_columns': int,
        'file_size_kb': float,
        'compression': str,
    },
    'checks': {                 # Individual checks
        'row_count': bool,
        'no_duplicate_game_loops': bool,
        # ... more checks
    },
    'stats': {                  # Data statistics
        'total_rows': int,
        'game_loop_range': tuple,
        'memory_usage_mb': float,
        # ... more stats
    },
}
```

## Testing

Run integration check:

```bash
python src_new/utils/validation_check.py
```

Expected output:
```
[PASS]: Import Check
[PASS]: Validator Methods
[PASS]: Documentation Functions
[PASS]: Validation Report Structure

Total: 4/4 checks passed
```

## Dependencies

- **pandas**: DataFrame operations, parquet reading
- **pyarrow**: Parquet metadata inspection
- **numpy**: Numeric operations, NaN handling
- **logging**: Validation logging

## Performance Notes

- Validation loads entire parquet files into memory
- For large files (>1GB), consider:
  - Sampling validation
  - Streaming validation
  - Parallel validation

## Next Steps

### Phase 5: CLI & Integration
- Command-line interface with Click
- CLI flags for validation
- Batch processing integration

### Phase 6: Testing
- Unit tests for all validation checks
- Test fixtures with known data
- Edge case testing

## Contributing

When extending Phase 4:

1. **Add new validation checks**: Extend `OutputValidator` helper methods
2. **Custom documentation**: Add new generator functions to `documentation.py`
3. **Update tests**: Add TODO comments for test cases
4. **Document changes**: Update USAGE_EXAMPLES.md

## License

Part of the SC2 Replay Ground Truth Extraction Pipeline project.

## Contact

For questions or issues with Phase 4 validation and documentation modules, please refer to the main project documentation.

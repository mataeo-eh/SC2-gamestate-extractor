# Phase 4 Validation & Documentation Usage Examples

This document provides examples of how to use the Phase 4 validation and documentation modules.

## Table of Contents

1. [OutputValidator](#outputvalidator)
2. [Documentation Generation](#documentation-generation)
3. [Integration with Pipeline](#integration-with-pipeline)

---

## OutputValidator

The `OutputValidator` class validates extracted parquet files for quality assurance.

### Validate Game State Parquet

```python
from pathlib import Path
from src_new.utils.validation import OutputValidator

# Create validator
validator = OutputValidator()

# Validate a game state parquet file
parquet_path = Path("data/processed/replay_001_game_state.parquet")
report = validator.validate_game_state_parquet(parquet_path)

# Check results
if report['valid']:
    print("Validation passed!")
else:
    print(f"Validation failed with {len(report['errors'])} errors")
    for error in report['errors']:
        print(f"  - {error}")

# Review warnings
if report['warnings']:
    print(f"\nWarnings ({len(report['warnings'])})")
    for warning in report['warnings']:
        print(f"  - {warning}")

# View statistics
print("\nStatistics:")
for key, value in report['stats'].items():
    print(f"  {key}: {value}")
```

### Validate Messages Parquet

```python
from pathlib import Path
from src_new.utils.validation import OutputValidator

validator = OutputValidator()

# Validate messages file
messages_path = Path("data/processed/replay_001_messages.parquet")
report = validator.validate_messages_parquet(messages_path)

print(f"Valid: {report['valid']}")
print(f"Errors: {len(report['errors'])}")
print(f"Warnings: {len(report['warnings'])}")
```

### Generate Validation Report

```python
from pathlib import Path
from src_new.utils.validation import OutputValidator

validator = OutputValidator()

# Validate multiple files
validations = []

for parquet_file in Path("data/processed").glob("*_game_state.parquet"):
    report = validator.validate_game_state_parquet(parquet_file)
    validations.append(report)

# Generate comprehensive markdown report
markdown_report = validator.generate_validation_report(validations)

# Save to file
output_path = Path("data/processed/validation_report.md")
output_path.write_text(markdown_report, encoding='utf-8')

print(f"Validation report saved to {output_path}")
```

### Validation Report Structure

Each validation report returns a dictionary with the following structure:

```python
{
    'valid': bool,              # Overall validation status
    'file_path': str,           # Path to validated file
    'errors': List[str],        # Critical errors (make valid=False)
    'warnings': List[str],      # Non-critical warnings
    'info': {                   # File metadata
        'num_rows': int,
        'num_columns': int,
        'file_size_kb': float,
        'compression': str,
    },
    'checks': {                 # Individual check results
        'row_count': bool,
        'no_duplicate_game_loops': bool,
        'resource_validity': bool,
        'building_progress_monotonic': bool,
        'unit_count_consistency': bool,
        'state_transitions': bool,
        # ... more checks
    },
    'stats': {                  # Data statistics
        'total_rows': int,
        'total_columns': int,
        'game_loop_range': tuple,
        'game_duration_seconds': float,
        'unit_columns': int,
        'economy_columns': int,
        'building_columns': int,
        'memory_usage_mb': float,
    },
}
```

---

## Documentation Generation

### Generate Data Dictionary

```python
from pathlib import Path
from src_new.extraction.schema_manager import SchemaManager
from src_new.utils.documentation import generate_data_dictionary

# Load or create schema
schema = SchemaManager()
schema.load_schema(Path("data/processed/replay_001_schema.json"))

# Generate data dictionary
output_path = Path("docs/data_dictionary.md")
generate_data_dictionary(schema, output_path)

print(f"Data dictionary saved to {output_path}")
```

### Generate Replay Report

```python
from pathlib import Path
from src_new.utils.documentation import generate_replay_report
from src_new.utils.validation import OutputValidator

# Validate the replay first (optional)
validator = OutputValidator()
game_state_path = Path("data/processed/replay_001_game_state.parquet")
validation_results = validator.validate_game_state_parquet(game_state_path)

# Generate replay report
replay_path = Path("data/replays/replay_001.SC2Replay")
output_path = Path("data/processed/replay_001_report.md")

generate_replay_report(
    replay_path=replay_path,
    output_path=output_path,
    validation_results=validation_results
)

print(f"Replay report saved to {output_path}")
```

### Generate Batch Summary

```python
from pathlib import Path
from src_new.utils.documentation import generate_batch_summary

# Prepare batch results (typically from ParallelProcessor)
batch_results = {
    'total_replays': 10,
    'successful': 8,
    'failed': 2,
    'total_time_seconds': 245.5,
    'config': {
        'processing_mode': 'two_pass',
        'step_size': 1,
        'compression': 'snappy',
    },
    'results': [
        {
            'success': True,
            'replay_path': Path("data/replays/replay_001.SC2Replay"),
            'stats': {
                'rows_written': 15000,
                'messages_written': 25,
                'processing_time_seconds': 24.5,
                'total_loops': 336000,
            },
        },
        {
            'success': True,
            'replay_path': Path("data/replays/replay_002.SC2Replay"),
            'stats': {
                'rows_written': 18000,
                'messages_written': 32,
                'processing_time_seconds': 28.3,
                'total_loops': 403200,
            },
        },
        {
            'success': False,
            'replay_path': Path("data/replays/replay_003.SC2Replay"),
            'error': 'Failed to load replay: File corrupted',
            'stats': {
                'processing_time_seconds': 2.1,
            },
        },
        # ... more results
    ],
}

# Generate batch summary
output_path = Path("data/processed/batch_summary.md")
generate_batch_summary(batch_results, output_path)

print(f"Batch summary saved to {output_path}")
```

---

## Integration with Pipeline

### Complete Workflow Example

```python
from pathlib import Path
from src_new.pipeline.extraction_pipeline import ReplayExtractionPipeline
from src_new.utils.validation import OutputValidator
from src_new.utils.documentation import (
    generate_data_dictionary,
    generate_replay_report,
)

# 1. Process replay
pipeline = ReplayExtractionPipeline()
replay_path = Path("data/replays/my_replay.SC2Replay")
output_dir = Path("data/processed")

result = pipeline.process_replay(replay_path, output_dir)

if result['success']:
    print(f"Processing succeeded!")

    # 2. Validate output
    validator = OutputValidator()
    game_state_path = result['output_files']['game_state']
    messages_path = result['output_files']['messages']

    game_state_validation = validator.validate_game_state_parquet(game_state_path)
    messages_validation = validator.validate_messages_parquet(messages_path)

    # 3. Generate validation report
    validations = [game_state_validation, messages_validation]
    validation_report = validator.generate_validation_report(validations)

    validation_report_path = output_dir / f"{replay_path.stem}_validation.md"
    validation_report_path.write_text(validation_report, encoding='utf-8')

    # 4. Generate data dictionary
    schema_path = result['output_files']['schema']
    from src_new.extraction.schema_manager import SchemaManager
    schema = SchemaManager()
    schema.load_schema(schema_path)

    data_dict_path = output_dir / f"{replay_path.stem}_data_dictionary.md"
    generate_data_dictionary(schema, data_dict_path)

    # 5. Generate replay report
    replay_report_path = output_dir / f"{replay_path.stem}_report.md"
    generate_replay_report(
        replay_path=replay_path,
        output_path=replay_report_path,
        validation_results=game_state_validation
    )

    print(f"\nGenerated documentation:")
    print(f"  - Validation report: {validation_report_path}")
    print(f"  - Data dictionary: {data_dict_path}")
    print(f"  - Replay report: {replay_report_path}")

else:
    print(f"Processing failed: {result['error']}")
```

### Batch Processing with Validation

```python
from pathlib import Path
from src_new.pipeline.parallel_processor import ParallelProcessor
from src_new.utils.validation import OutputValidator
from src_new.utils.documentation import generate_batch_summary

# 1. Process batch of replays
processor = ParallelProcessor(num_workers=4)
replay_dir = Path("data/replays")
output_dir = Path("data/processed")

batch_results = processor.process_directory(replay_dir, output_dir)

# 2. Validate all outputs
validator = OutputValidator()
all_validations = []

for result in batch_results['results']:
    if result['success']:
        game_state_path = result['output_files']['game_state']
        validation = validator.validate_game_state_parquet(game_state_path)
        all_validations.append(validation)

# 3. Generate comprehensive validation report
validation_report = validator.generate_validation_report(all_validations)
validation_report_path = output_dir / "batch_validation.md"
validation_report_path.write_text(validation_report, encoding='utf-8')

# 4. Generate batch summary
batch_summary_path = output_dir / "batch_summary.md"
generate_batch_summary(batch_results, batch_summary_path)

print(f"Batch processing complete!")
print(f"  Total replays: {batch_results['total_replays']}")
print(f"  Successful: {batch_results['successful']}")
print(f"  Failed: {batch_results['failed']}")
print(f"\nReports generated:")
print(f"  - Validation: {validation_report_path}")
print(f"  - Summary: {batch_summary_path}")
```

---

## Validation Checks Reference

### Game State Validation Checks

| Check | Description | Failure Type |
|-------|-------------|--------------|
| `row_count` | Parquet has at least one row | Error |
| `required_columns` | Base columns (game_loop, timestamp_seconds) present | Error |
| `no_duplicate_game_loops` | No duplicate game_loop values | Error |
| `column_types` | Column types match expected schema | Warning |
| `resource_validity` | Resources (minerals, gas, supply) are valid | Error |
| `building_progress_monotonic` | Building progress never decreases | Error |
| `unit_count_consistency` | Unit counts match individual units | Warning |
| `state_transitions` | State transitions are valid | Warning |
| `no_nan_in_base_columns` | Base columns have no NaN values | Error |

### Messages Validation Checks

| Check | Description | Failure Type |
|-------|-------------|--------------|
| `file_exists` | Messages file exists (optional) | Warning |
| `required_columns` | Required columns present | Error |
| `has_messages` | File contains messages | Warning |
| `type_*` | Column types are correct | Error |
| `valid_game_loops` | game_loop values are valid | Error |
| `no_duplicates` | No duplicate messages | Warning |

---

## Tips and Best Practices

1. **Always validate after processing**: Immediately validate extracted parquet files to catch issues early.

2. **Review warnings**: Warnings don't fail validation but may indicate data quality issues.

3. **Generate documentation**: Create data dictionaries and replay reports for better data understanding.

4. **Batch validation**: When processing multiple replays, use batch validation to identify patterns.

5. **Custom validation**: Extend `OutputValidator` for domain-specific checks.

6. **Error handling**: Validation methods handle errors gracefully and return structured results.

7. **Performance**: Validation reads entire parquet files into memory. For large files, consider sampling.

---

## Next Steps

- **Phase 5**: CLI integration will add command-line flags for validation
- **Phase 6**: Tests will verify validation logic with known good/bad data
- **Future**: Add custom validation rules via configuration files

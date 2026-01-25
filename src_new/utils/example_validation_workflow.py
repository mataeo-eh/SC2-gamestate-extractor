"""
Example Validation Workflow: Complete demonstration of Phase 4 capabilities.

This script demonstrates how to use the validation and documentation modules
in a complete workflow.

Note: This is an example script. To run it, you need:
1. Processed parquet files from Phase 3
2. Schema JSON files from extraction pipeline
"""

import sys
from pathlib import Path

# Add src_new to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.validation import OutputValidator
from utils.documentation import (
    generate_data_dictionary,
    generate_replay_report,
    generate_batch_summary,
)


def example_1_validate_single_file():
    """Example 1: Validate a single game state parquet file."""
    print("=" * 60)
    print("Example 1: Validate Single File")
    print("=" * 60)

    # Create validator
    validator = OutputValidator()

    # Example parquet path (replace with actual path)
    parquet_path = Path("data/processed/example_game_state.parquet")

    print(f"\nValidating: {parquet_path}")

    if not parquet_path.exists():
        print(f"  [SKIP] File not found: {parquet_path}")
        print("  This is an example. Replace with actual parquet file path.")
        return None

    # Validate
    report = validator.validate_game_state_parquet(parquet_path)

    # Display results
    print(f"\nValidation Result: {'PASS' if report['valid'] else 'FAIL'}")
    print(f"  File: {report['file_path']}")
    print(f"  Rows: {report['info'].get('num_rows', 'N/A')}")
    print(f"  Columns: {report['info'].get('num_columns', 'N/A')}")
    print(f"  Size: {report['info'].get('file_size_kb', 0):.2f} KB")

    if report['errors']:
        print(f"\n  Errors ({len(report['errors'])}):")
        for error in report['errors'][:5]:  # Show first 5
            print(f"    - {error}")

    if report['warnings']:
        print(f"\n  Warnings ({len(report['warnings'])}):")
        for warning in report['warnings'][:5]:  # Show first 5
            print(f"    - {warning}")

    # Display statistics
    if 'stats' in report and report['stats']:
        print(f"\n  Statistics:")
        for key, value in list(report['stats'].items())[:5]:  # Show first 5
            print(f"    {key}: {value}")

    return report


def example_2_validate_messages():
    """Example 2: Validate messages parquet file."""
    print("\n" + "=" * 60)
    print("Example 2: Validate Messages File")
    print("=" * 60)

    validator = OutputValidator()

    messages_path = Path("data/processed/example_messages.parquet")

    print(f"\nValidating: {messages_path}")

    if not messages_path.exists():
        print(f"  [SKIP] File not found: {messages_path}")
        print("  Messages files are optional.")
        return None

    # Validate
    report = validator.validate_messages_parquet(messages_path)

    print(f"\nValidation Result: {'PASS' if report['valid'] else 'FAIL'}")

    if report.get('checks', {}).get('has_messages', False):
        print(f"  Messages found: {report['info'].get('num_rows', 0)}")
    else:
        print(f"  No messages in file")

    return report


def example_3_generate_validation_report():
    """Example 3: Generate comprehensive validation report."""
    print("\n" + "=" * 60)
    print("Example 3: Generate Validation Report")
    print("=" * 60)

    validator = OutputValidator()

    # Simulate multiple validations
    # In real usage, you would validate actual parquet files
    validations = []

    # Example: validate multiple files in a directory
    data_dir = Path("data/processed")

    if data_dir.exists():
        parquet_files = list(data_dir.glob("*_game_state.parquet"))

        if parquet_files:
            print(f"\nValidating {len(parquet_files)} files...")

            for parquet_file in parquet_files[:3]:  # Limit to first 3 for example
                print(f"  - {parquet_file.name}")
                report = validator.validate_game_state_parquet(parquet_file)
                validations.append(report)
        else:
            print(f"\n[SKIP] No game state parquet files found in {data_dir}")
            print("This is an example. Process replays first to generate parquet files.")
            return
    else:
        print(f"\n[SKIP] Directory not found: {data_dir}")
        return

    # Generate markdown report
    if validations:
        markdown_report = validator.generate_validation_report(validations)

        # Save to file
        output_path = data_dir / "validation_report.md"
        output_path.write_text(markdown_report, encoding='utf-8')

        print(f"\nValidation report saved to: {output_path}")
        print(f"  Total files validated: {len(validations)}")
        print(f"  Valid files: {sum(1 for v in validations if v['valid'])}")


def example_4_generate_data_dictionary():
    """Example 4: Generate data dictionary from schema."""
    print("\n" + "=" * 60)
    print("Example 4: Generate Data Dictionary")
    print("=" * 60)

    # This requires a schema JSON file from extraction pipeline
    schema_path = Path("data/processed/example_schema.json")

    print(f"\nLooking for schema: {schema_path}")

    if not schema_path.exists():
        print(f"  [SKIP] Schema file not found: {schema_path}")
        print("  Process a replay first to generate schema.")
        return

    # Load schema
    from extraction.schema_manager import SchemaManager

    schema = SchemaManager()
    schema.load_schema(schema_path)

    print(f"  Schema loaded with {len(schema.get_column_list())} columns")

    # Generate data dictionary
    output_path = Path("docs/data_dictionary.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_data_dictionary(schema, output_path)

    print(f"\nData dictionary saved to: {output_path}")


def example_5_generate_replay_report():
    """Example 5: Generate replay processing report."""
    print("\n" + "=" * 60)
    print("Example 5: Generate Replay Report")
    print("=" * 60)

    replay_path = Path("data/replays/example.SC2Replay")
    parquet_path = Path("data/processed/example_game_state.parquet")

    print(f"\nGenerating report for: {replay_path.name}")

    if not parquet_path.exists():
        print(f"  [SKIP] Parquet file not found: {parquet_path}")
        return

    # Validate first (optional)
    validator = OutputValidator()
    validation_results = validator.validate_game_state_parquet(parquet_path)

    # Generate report
    output_path = Path("data/processed/example_report.md")

    generate_replay_report(
        replay_path=replay_path,
        output_path=output_path,
        validation_results=validation_results
    )

    print(f"\nReplay report saved to: {output_path}")


def example_6_batch_summary():
    """Example 6: Generate batch processing summary."""
    print("\n" + "=" * 60)
    print("Example 6: Generate Batch Summary")
    print("=" * 60)

    # Simulate batch results
    # In real usage, this would come from ParallelProcessor
    batch_results = {
        'total_replays': 5,
        'successful': 4,
        'failed': 1,
        'total_time_seconds': 123.5,
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
        ],
    }

    print(f"\nGenerating batch summary...")
    print(f"  Total replays: {batch_results['total_replays']}")
    print(f"  Successful: {batch_results['successful']}")
    print(f"  Failed: {batch_results['failed']}")

    # Generate summary
    output_path = Path("data/processed/batch_summary.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_batch_summary(batch_results, output_path)

    print(f"\nBatch summary saved to: {output_path}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Phase 4 Validation & Documentation Examples")
    print("=" * 60)
    print("\nThis script demonstrates the Phase 4 validation and")
    print("documentation capabilities.")
    print("\nNote: Most examples require actual parquet files from")
    print("processing replays. Run the extraction pipeline first.")
    print()

    # Run examples
    examples = [
        example_1_validate_single_file,
        example_2_validate_messages,
        example_3_generate_validation_report,
        example_4_generate_data_dictionary,
        example_5_generate_replay_report,
        example_6_batch_summary,
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n[ERROR] {example_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Examples Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

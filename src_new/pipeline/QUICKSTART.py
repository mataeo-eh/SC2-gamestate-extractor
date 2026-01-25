"""
Quick Start Example for SC2 Replay Ground Truth Extraction Pipeline

This script demonstrates the simplest way to use the pipeline to process
SC2 replays and extract ground truth game state data.

Requirements:
    - pysc2 installed: pip install pysc2
    - SC2 game installed (pysc2 needs the game client)
    - Replay files (.SC2Replay)

Usage:
    python QUICKSTART.py
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src_new.pipeline import (
    process_replay_quick,
    process_directory_quick,
    ReplayExtractionPipeline,
    ParallelReplayProcessor,
)


def example_1_single_replay():
    """Example 1: Process a single replay with one function call."""
    print("=" * 60)
    print("Example 1: Process Single Replay")
    print("=" * 60)
    print()

    replay_path = Path("replays/example.SC2Replay")

    # Check if replay exists
    if not replay_path.exists():
        print(f"ERROR: Replay file not found: {replay_path}")
        print("Please update the path to point to an actual replay file.")
        return

    # Process the replay
    result = process_replay_quick(
        replay_path=replay_path,
        output_dir=Path("data/processed")
    )

    # Print results
    if result['success']:
        print("SUCCESS!")
        print(f"  Processed: {replay_path.name}")
        print(f"  Time: {result['stats']['processing_time_seconds']:.2f}s")
        print(f"  Rows written: {result['stats']['rows_written']}")
        print(f"  Messages: {result['stats']['messages_written']}")
        print()
        print("Output files:")
        for name, path in result['output_files'].items():
            print(f"  {name}: {path}")
    else:
        print("FAILED!")
        print(f"  Error: {result['error']}")


def example_2_batch_processing():
    """Example 2: Process multiple replays in parallel."""
    print()
    print("=" * 60)
    print("Example 2: Batch Processing")
    print("=" * 60)
    print()

    replay_dir = Path("replays/")

    # Check if directory exists
    if not replay_dir.exists():
        print(f"ERROR: Replay directory not found: {replay_dir}")
        print("Please update the path to point to a directory with replay files.")
        return

    # Process all replays in directory
    results = process_directory_quick(
        replay_dir=replay_dir,
        output_dir=Path("data/processed"),
        num_workers=4  # Use 4 CPU cores
    )

    # Print summary
    print(f"Total replays: {results['total_replays']}")
    print(f"Successful: {results['successful_count']}")
    print(f"Failed: {results['failed_count']}")
    print(f"Total time: {results['total_time_seconds']:.2f}s")
    print(f"Average per replay: {results['average_time_per_replay']:.2f}s")

    # Show failed replays
    if results['failed']:
        print()
        print("Failed replays:")
        for replay_path, error in results['failed'][:5]:
            print(f"  - {replay_path.name}: {error}")


def example_3_custom_configuration():
    """Example 3: Custom pipeline configuration."""
    print()
    print("=" * 60)
    print("Example 3: Custom Configuration")
    print("=" * 60)
    print()

    replay_path = Path("replays/example.SC2Replay")

    if not replay_path.exists():
        print(f"ERROR: Replay file not found: {replay_path}")
        return

    # Create pipeline with custom config
    config = {
        # Perfect information settings
        'show_cloaked': True,
        'show_burrowed_shadows': True,
        'show_placeholders': True,

        # Processing settings
        'processing_mode': 'single_pass',  # Faster mode
        'step_size': 10,  # Sample every 10 game loops

        # Output settings
        'compression': 'gzip',  # Better compression
    }

    pipeline = ReplayExtractionPipeline(config)

    # Process replay
    result = pipeline.process_replay(
        replay_path=replay_path,
        output_dir=Path("data/processed_custom")
    )

    if result['success']:
        print("SUCCESS with custom config!")
        print(f"  Mode: single-pass with step_size=10")
        print(f"  Time: {result['stats']['processing_time_seconds']:.2f}s")
        print(f"  Rows: {result['stats']['rows_written']}")


def example_4_advanced_batch():
    """Example 4: Advanced batch processing with retry."""
    print()
    print("=" * 60)
    print("Example 4: Advanced Batch Processing")
    print("=" * 60)
    print()

    replay_dir = Path("replays/")

    if not replay_dir.exists():
        print(f"ERROR: Replay directory not found: {replay_dir}")
        return

    # Create processor
    processor = ParallelReplayProcessor(
        config={'processing_mode': 'two_pass'},
        num_workers=8
    )

    # First attempt
    print("Processing replays...")
    results = processor.process_replay_directory(
        replay_dir=replay_dir,
        output_dir=Path("data/processed")
    )

    # Print formatted summary
    summary = processor.get_processing_summary(results)
    print(summary)

    # Retry failed replays if any
    if results['failed']:
        print()
        print(f"Retrying {len(results['failed'])} failed replays...")
        retry_results = processor.retry_failed_replays(
            failed_results=results['failed'],
            output_dir=Path("data/processed")
        )

        print(f"Retry: {retry_results['successful_count']} now successful")


def example_5_read_output():
    """Example 5: Reading pipeline output."""
    print()
    print("=" * 60)
    print("Example 5: Reading Output Data")
    print("=" * 60)
    print()

    import pandas as pd
    import json

    # Example output file
    game_state_path = Path("data/processed/example_game_state.parquet")
    messages_path = Path("data/processed/example_messages.parquet")
    schema_path = Path("data/processed/example_schema.json")

    # Read game state
    if game_state_path.exists():
        df = pd.read_parquet(game_state_path)
        print(f"Game State DataFrame:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Duration: {df['timestamp_seconds'].max():.1f}s")
        print(f"  P1 final minerals: {df.iloc[-1]['p1_minerals']}")
        print(f"  P2 final minerals: {df.iloc[-1]['p2_minerals']}")
    else:
        print(f"Game state file not found: {game_state_path}")

    print()

    # Read messages
    if messages_path.exists():
        messages_df = pd.read_parquet(messages_path)
        print(f"Messages DataFrame:")
        print(f"  Total messages: {len(messages_df)}")
        if len(messages_df) > 0:
            print("\n  First few messages:")
            for _, msg in messages_df.head(3).iterrows():
                print(f"    [{msg['game_loop']}] P{msg['player_id']}: {msg['message']}")
    else:
        print(f"Messages file not found: {messages_path}")

    print()

    # Read schema
    if schema_path.exists():
        with open(schema_path, 'r') as f:
            schema = json.load(f)

        print(f"Schema JSON:")
        print(f"  Total columns: {len(schema['columns'])}")
        print(f"\n  First 10 columns:")
        for col_name in schema['columns'][:10]:
            doc = schema['documentation'][col_name]
            print(f"    {col_name}: {doc['description']}")
    else:
        print(f"Schema file not found: {schema_path}")


def main():
    """Run all examples."""
    print()
    print("=" * 60)
    print("SC2 Replay Ground Truth Extraction Pipeline")
    print("Quick Start Examples")
    print("=" * 60)
    print()
    print("This script demonstrates basic usage of the pipeline.")
    print()
    print("NOTE: These examples require:")
    print("  1. pysc2 to be installed")
    print("  2. SC2 game to be installed")
    print("  3. Replay files to exist at the specified paths")
    print()
    print("Update the paths in each example to match your setup.")
    print()

    # Run examples
    # Uncomment the example you want to run

    # example_1_single_replay()
    # example_2_batch_processing()
    # example_3_custom_configuration()
    # example_4_advanced_batch()
    # example_5_read_output()

    print()
    print("=" * 60)
    print("To run an example, uncomment it in the main() function.")
    print()
    print("For more examples, see USAGE_EXAMPLES.md")
    print("=" * 60)


if __name__ == "__main__":
    main()

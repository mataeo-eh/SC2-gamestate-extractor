#!/usr/bin/env python3
"""
Quick Start Script for SC2 Replay Ground Truth Extraction Pipeline

This script demonstrates the simplest way to get started with the pipeline.
It processes a sample replay and displays the results.

Requirements:
    - pysc2 installed: pip install pysc2
    - SC2 game installed
    - At least one replay file

Usage:
    python quickstart.py

    # With custom replay:
    python quickstart.py --replay path/to/replay.SC2Replay
    python quickstart.py -r path/to/replay.SC2Replay

    # With custom output directory:
    python quickstart.py --output data/my_output
    python quickstart.py -o data/my_output

    # With parallel processing of full directory
    python quickstart.py --process-replay-directory path/to/replay/directory
    python quickstart.py -batch path/to/replay/directory

    # With custom number of workers for parallel processing 
    python quickstart.py --process-replay-directory path/to/replay/directory --workers 3
    python quickstart.py -batch path/to/replay/directory -w 3

    # With downloading new replays
    python quickstart.py --bots really why what --download-replays
    python quickstart.py --bots really why what -dr

        # Downloading custom number of replays (default is all)
        python quickstart.py --bots really why what --download-replays --num-replays 1
        python quickstart.py --bots really why what -dr -nr 1

    # With updating kaggle dataset
    python quickstart.py --update-kaggle-dataset
    python quickstart.py -dataset

    # With engineering features
    python quickstart.py --engineer-features
    python quickstart.py -e

    # With discretization 
    python quickstart.py --discretize
    python quickstart.py -d

    # Full pipeline usage
    python quickstart.py --process-replay-directory replays --output data/quickstart --workers 3 --download-replays --bots really what why -dataset --num-replays 105 -e -d

    # Feature Engineering and Discretization
    python quickstart.py --output data/quickstart --engineer-features --discretize -dataset

    # Testing
    python quickstart.py --process-replay-directory replays --output data/quickstart --workers 2 -dataset --num-replays 5 -e -d
"""
from absl import flags
import sys
import io
import logging
from pathlib import Path
import argparse
import multiprocessing

import papermill as pm
from papermill.exceptions import PapermillExecutionError

from src_new.pipeline.logging_config import setup_logging

from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

# Resolve paths relative to this script's location (robust to different cwd).
# SCRIPT_DIR always points to the directory containing quickstart.py regardless
# of where the user invokes the script from.
SCRIPT_DIR = Path(__file__).resolve().parent
EDA_NOTEBOOKS = [
    SCRIPT_DIR / "EDA" / "raw_data_summary.ipynb",
    SCRIPT_DIR / "EDA" / "data_verification.ipynb",
]

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def run_eda_notebooks(output_dir: Path):
    """Execute EDA notebooks via papermill with the resolved parquet directory path.

    Each notebook has a papermill ``parameters`` cell that declares a
    ``parquet_dir`` string variable.  This function resolves the parquet
    directory to an absolute path (so notebooks work regardless of cwd) and
    injects it into every notebook listed in ``EDA_NOTEBOOKS``.

    Notebooks are executed **in-place** -- the executed output overwrites the
    original ``.ipynb`` file.  On failure the partially-executed notebook is
    still written (with error markers), so the user can inspect what went wrong.

    Parameters:
        output_dir (Path): The ``--output`` argument from argparse
                           (e.g. ``Path('data/quickstart')``).  The parquet
                           sub-directory is appended automatically.

    Returns:
        None

    Depends on / calls:
        - ``papermill.execute_notebook()``
        - ``EDA_NOTEBOOKS`` (module-level list of notebook paths)
        - ``PapermillExecutionError`` for cell-level failures
    """
    # Resolve to an absolute path so notebooks work regardless of the
    # working directory papermill happens to use.
    parquet_dir_abs = str((output_dir / "parquet").resolve())

    for nb_path in EDA_NOTEBOOKS:
        nb_path_str = str(nb_path)
        print(f"  Running EDA notebook: {nb_path.name} ...")
        try:
            pm.execute_notebook(
                input_path=nb_path_str,
                output_path=nb_path_str,       # in-place execution
                parameters={"parquet_dir": parquet_dir_abs},
                cwd=str(nb_path.parent),        # match Jupyter's default cwd
                kernel_name="python3",
            )
            print(f"  OK: {nb_path.name}")
        except PapermillExecutionError as exc:
            # Cell-level error -- notebook is already saved with error markers.
            print(f"  FAILED: {nb_path.name}")
            print(f"    Error in cell [{exc.exec_count}]: {exc.ename}: {exc.evalue}")
            print("    The notebook has been saved with error details.")
        except Exception as exc:
            # Kernel death, missing dependencies, or other unexpected error.
            print(f"  FAILED: {nb_path.name}")
            print(f"    Unexpected error: {exc}")


def check_prerequisites():
    """Check if prerequisites are installed."""
    print("Checking prerequisites...")
    print()

    # Check Python version
    if sys.version_info < (3, 9):
        print(f"❌ Python 3.9+ required (you have {sys.version_info.major}.{sys.version_info.minor})")
        return False
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # Check dependencies
    missing = []

    try:
        import pandas
        print(f"✓ pandas {pandas.__version__}")
    except ImportError:
        print("❌ pandas not installed")
        missing.append("pandas")

    try:
        import pyarrow
        print(f"✓ pyarrow {pyarrow.__version__}")
    except ImportError:
        print("❌ pyarrow not installed")
        missing.append("pyarrow")

    try:
        import numpy
        print(f"✓ numpy {numpy.__version__}")
    except ImportError:
        print("❌ numpy not installed")
        missing.append("numpy")

    try:
        import pysc2
        print(f"✓ pysc2 installed")
    except ImportError:
        print("❌ pysc2 not installed")
        missing.append("pysc2")

    print()

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print()
        print("Install with:")
        print(f"  pip install {' '.join(missing)}")
        return False

    # Check SC2 installation
    try:
        from absl import flags
        from pysc2.run_configs import get
        
        # Parse flags if not already parsed
        if not flags.FLAGS.is_parsed():
            flags.FLAGS([''])  # Parse with empty args
        
        run_config = get()
        print(f"✓ SC2 found")
    except Exception as e:
        print(f"❌ SC2 not found: {e}")
        print()
        print("Please install StarCraft II:")
        print("  https://starcraft2.com/")
        return False

    print()
    return True


def find_sample_replay():
    """Find a sample replay file."""
    # Check common locations
    replay_dirs = [
        Path("replays")
    ]

    for replay_dir in replay_dirs:
        if replay_dir.exists():
            replays = list(replay_dir.rglob("*.SC2Replay"))
            if replays:
                return replays[0]

    return None


def process_replay_example(replay_path: Path, output_dir: Path):
    """Process a replay and display results."""
    print("=" * 70)
    print("Processing Replay")
    print("=" * 70)
    print()
    print(f"Replay: {replay_path}")
    print(f"Output: {output_dir}")
    print()

    # Import pipeline
    try:
        from src_new.pipeline import process_replay_quick
    except ImportError:
        print("❌ Pipeline not found. Make sure you're in the project root directory.")
        return False

    # Process replay
    print("Processing... (this may take 10-60 seconds)")
    print()

    try:
        result = process_replay_quick(
            replay_path=replay_path,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        print()
        print("Try:")
        print("  1. Verify the replay file is not corrupted")
        print("  2. Check that SC2 is properly installed")
        print("  3. See docs/troubleshooting.md for more help")
        return False

    # Display results
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print()

    if result['success']:
        print("✓ Processing successful!")
        print()
        print("Statistics:")
        print(f"  Time: {result['stats']['processing_time_seconds']:.2f}s")
        print(f"  Rows written: {result['stats']['rows_written']:,}")
        print(f"  Messages: {result['stats']['messages_written']}")
        print()
        print("Output files:")
        for name, path in result['output_files'].items():
            if path.exists():
                size_mb = path.stat().st_size / 1024 / 1024
                print(f"  {name}: {path} ({size_mb:.2f} MB)")
            else:
                print(f"  {name}: {path} (not created)")
        print()

        # Load and display sample data
        try:
            import pandas as pd

            game_state_path = result['output_files']['game_state']
            df = pd.read_parquet(game_state_path)

            print("Game State Preview:")
            print(f"  Duration: {df['timestamp_seconds'].max():.1f} seconds")
            print(f"  Total rows: {len(df):,}")
            print(f"  Total columns: {len(df.columns):,}")
            print()

            # Show first and last rows
            print("First 3 rows:")
            print(df.head(3)[['game_loop', 'timestamp_seconds', 'p1_minerals', 'p2_minerals']].to_string(index=False))
            print()

            print("Last 3 rows:")
            print(df.tail(3)[['game_loop', 'timestamp_seconds', 'p1_minerals', 'p2_minerals']].to_string(index=False))
            print()

            # Unit counts
            unit_count_cols = [col for col in df.columns if '_count' in col]
            if unit_count_cols:
                print("Final unit counts:")
                final_row = df.iloc[-1]
                for col in unit_count_cols[:10]:  # Show first 10
                    count = final_row[col]
                    if count > 0:
                        print(f"  {col}: {int(count)}")
                if len(unit_count_cols) > 10:
                    print(f"  ... and {len(unit_count_cols) - 10} more unit types")
                print()

        except Exception as e:
            print(f"Note: Could not load preview data: {e}")
            print()

        print("=" * 70)
        print("Next Steps")
        print("=" * 70)
        print()
        print("1. Explore the output:")
        print(f"   import pandas as pd")
        print(f"   df = pd.read_parquet('{result['output_files']['game_state']}')")
        print(f"   print(df.head())")
        print()
        print("2. Read the usage guide:")
        print("   docs/usage.md")
        print()
        print("3. Try batch processing:")
        print("   python -c \"from src_new.pipeline import process_directory_quick; \\")
        print("              results = process_directory_quick(Path('replays/'), Path('data/processed'))\"")
        print()
        print("4. Check out example notebooks:")
        print("   examples/01_basic_extraction.ipynb")
        print()

        return True
    else:
        print(f"❌ Processing failed: {result['error']}")
        print()
        print("See docs/troubleshooting.md for help")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Quick start for SC2 Replay Extraction Pipeline"
    )
    parser.add_argument(
        '-r', '--replay',
        type=Path,
        help='Path to single replay file (auto-detect if not provided)'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('data/quickstart'),
        help='Output directory (default: data/quickstart)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '-dr', '--download-replays',
        action='store_true',
        help='Download new replays from aiarena before processing'
    )
    parser.add_argument(
        '-nr', '--num-replays',
        type=int,
        help='Number of replays to download (default: all)'
    )
    parser.add_argument(
        '-b', '--bots',
        nargs='+',
        help='List of bot names to download replays for (default: None)'
    )
    parser.add_argument(
        '-batch', '--process-replay-directory',
        type=Path,
        help="Process all replays in specified directory using parallel replay processing"
    )
    parser.add_argument(
        '-dataset', '--update-kaggle-dataset',
        action="store_true",
        help="Optional: Use if you want to update the kaggle dataset after replays are parsed"
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=multiprocessing.cpu_count() // 2,
        help="CPU threads to use for parallel processing. Default is half of CPU's available threads"
    )
    parser.add_argument(
        "-e", "--engineer-features",
        action="store_true",
        help="Whether to engineer additional features like army composition and movement."
    )
    parser.add_argument(
        "-d", "--discretize",
        action="store_true",
        help="Whether to create a simplified dataset with only the engineered features for quick baseline model training."
    )
    parser.add_argument(
        "-EDA", "--EDA-notebook",
        action="store_true",
        help="Run EDA notebooks (raw_data_summary, data_verification) on the output data via papermill."
    )

    args = parser.parse_args()

    FLAGS = flags.FLAGS

    # Import pysc2 so its flags get registered
    from pysc2 import run_configs  

    # Now parse FLAGS - give it a clean argv without argparse flags
    FLAGS(['quickstart.py'], known_only=True)

    # Set up file-based logging before any processing begins
    log_file_path = setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("SC2 Replay Ground Truth Extraction Pipeline starting")

    print()
    print("=" * 70)
    print("SC2 Replay Ground Truth Extraction Pipeline")
    print("Quick Start")
    print("=" * 70)
    print()

    

    # Check prerequisites
    if not check_prerequisites():
        print("Please fix the issues above and try again.")
        sys.exit(1)

    # Downlaod replays if passed
    if args.download_replays:
        from src_new.data_processing.fetch_bot_replays import main as download_replays
        if not args.bots:
            raise RuntimeError("You must pass names of bots to download replays for to download_replays")
        download_replays(args.bots, max_replays = args.num_replays, print_output = args.verbose)

    # Process all replays in directory if passed
    if args.process_replay_directory:
        from src_new.pipeline.parallel_processor import ParallelReplayProcessor
        processor = ParallelReplayProcessor(num_workers=args.workers, log_file_path=log_file_path)
        # Process all replays in a directory
        results = processor.process_replay_directory(
            replay_dir=args.process_replay_directory,
            output_dir=args.output,
            pattern="*.SC2Replay"  # Optional pattern
        )
        # Print summary
        summary = processor.get_processing_summary(results)
        print(summary)
        # Add the unit count columns to all processed files
        from src_new.data_processing.create_unit_counts import main as create_unit_counts
        create_unit_counts(Path(f"{args.output}/parquet"), Path(f"{args.output}/parquet"))
        print("✓ Unit count columns added to all processed files")

    # Find replay if passed and process_replay_directory is not
    elif args.replay:
        replay_path = args.replay
        if not replay_path.exists():
            print(f"❌ Replay not found: {replay_path}")
            sys.exit(1)
        # Create output directory
        args.output.mkdir(parents=True, exist_ok=True)

        # Process replay
        success = process_replay_example(replay_path, args.output)

        if success:
            print("✓ Quick start complete!")
            print()
            print("For more information, see:")
            print("  README_SC2_PIPELINE.md - Project overview")
            print("  docs/usage.md - Detailed usage guide")
            print("  docs/data_dictionary.md - Output schema reference")
            sys.exit(0)
        else:
            print("❌ Quick start failed. See errors above.")
            sys.exit(1)
    else:
        print("Looking for sample replay...")
        replay_path = find_sample_replay()
        if not replay_path:
            print("❌ No replay found.")
            print()
            print("Please specify a replay:")
            print("  python quickstart.py --replay path/to/replay.SC2Replay")
            print()
            print("Or place replays in the 'replays/' directory")
            sys.exit(1)
        print(f"✓ Found: {replay_path}")
        print()

    # Create the engineered features dataset if passed
    if args.engineer_features:
        from src_new.data_processing.engineer_army_features import main as engineer_army_features
        engineer_army_features(Path(f"{args.output}/parquet"), Path(os.getenv("ENGINEER_FEATURES_OUTPUT_DIR")))
        print(f"✓ Engineered features dataset created at {os.getenv('ENGINEER_FEATURES_OUTPUT_DIR')}")

    # Create the discretized dataset if passed
    if args.discretize:
        from src_new.data_processing.discretize import main as discretize_main      
        discretize_main(Path(os.getenv("DISCRETIZE_INPUT_DIR")), Path(os.getenv("DISCRETIZE_OUTPUT_DIR")))
        print(f"✓ Discretized dataset created at {os.getenv('DISCRETIZE_OUTPUT_DIR')}")

    # Run EDA notebooks if passed
    if args.EDA_notebook:
        print("Running EDA notebooks...")
        run_eda_notebooks(args.output)
        print("EDA notebooks complete.")

    if args.discretize:
        # After discretization is done, update the kaggle dataset if passed
        # Note: The discretized data is a separate dataset from the raw information and engineered features dataset 
        if args.update_kaggle_dataset:
            from src_new.pipeline.dataset_pipeline import main as upload_to_kaggle
            upload_to_kaggle(dataset_name = "mataeoanderson/sc2-replay-data-discretized", download_path = Path("data/discretized"), Title = "SC2 Discretized Dataset")

    # After processing, update the kaggle dataset if passed
    if args.update_kaggle_dataset:
        from src_new.pipeline.dataset_pipeline import main as upload_to_kaggle
        upload_to_kaggle(dataset_name = "mataeoanderson/sc2-replay-data", download_path = Path("data/quickstart"))
    


if __name__ == "__main__":
    main()

"""
ParallelReplayProcessor: Process multiple SC2 replays in parallel.

This component provides batch processing capabilities with multiprocessing
for efficient processing of large replay datasets.
"""
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import logging
import time

from .extraction_pipeline import ReplayExtractionPipeline


logger = logging.getLogger(__name__)


class ParallelReplayProcessor:
    """
    Process multiple replays in parallel using multiprocessing.

    This class provides efficient batch processing of SC2 replays by utilizing
    multiple CPU cores. It handles error recovery, progress reporting, and
    detailed timing statistics.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        num_workers: Optional[int] = None,
        log_file_path: Optional[str] = None
    ):
        """
        Initialize the parallel processor.

        Args:
            config: Optional configuration dictionary (passed to ReplayExtractionPipeline)
            num_workers: Number of parallel workers (default: CPU count)
            log_file_path: Absolute path to the log file (passed to worker processes for logging)
        """
        self.config = config or {}
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.log_file_path = log_file_path

        logger.info(f"ParallelReplayProcessor initialized with {self.num_workers} workers")

    def process_replay_batch(
        self,
        replay_paths: List[Path],
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Process multiple replays in parallel.

        This method uses ProcessPoolExecutor to process replays in parallel,
        with comprehensive error handling and progress reporting.

        Args:
            replay_paths: List of paths to .SC2Replay files
            output_dir: Directory for output files

        Returns:
            Batch processing results:
            {
                'successful': [Path, ...],
                'failed': [(Path, error_message), ...],
                'processing_times': {Path: seconds, ...},
                'total_replays': int,
                'successful_count': int,
                'failed_count': int,
                'total_time_seconds': float,
                'average_time_per_replay': float,
            }

        # TODO: Test case - Process multiple replays in parallel
        # TODO: Test case - Handle worker errors gracefully
        # TODO: Test case - Verify all outputs created
        # TODO: Test case - Check progress reporting
        """
        logger.info(f"Processing batch of {len(replay_paths)} replays")
        logger.info(f"  Workers: {self.num_workers}")
        logger.info(f"  Output directory: {output_dir}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Make a copy of the replays list to iterate through it
        replays_list_copy = replay_paths.copy()
        replay_paths = [] # make the list empty so it can be populated with un-parsed replays
        for replay_path in replays_list_copy:
            stem = replay_path.stem
            suffix = ".parquet"
            processed_name = f"{stem}_game_state{suffix}" # Build the file name to check for

            # Check recursively in output_dir and all subdirectories with rglob
            matching_files = list(output_dir.rglob(processed_name))
            if matching_files:
                print(f"File: {stem} already processed (found at {matching_files[0].relative_to(output_dir)}). Skipping.")
                continue # Move onto next replay if current replay has already been parsed

            # Add the not yet processed replay path to the list to be processed
            replay_paths.append(replay_path)

        # Initialize results
        results = {
            'successful': [],
            'failed': [],
            'processing_times': {},
            'total_replays': len(replay_paths),
            'successful_count': 0,
            'failed_count': 0,
            'total_time_seconds': 0.0,
            'average_time_per_replay': 0.0,
        }

        start_time = time.time()

        # Process replays in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all jobs
            future_to_replay = {
                executor.submit(
                    _worker_process_replay,
                    replay_path,
                    output_dir,
                    self.config,
                    self.log_file_path
                ): replay_path
                for replay_path in replay_paths
            }

            # Process completed jobs
            completed = 0
            for future in tqdm(as_completed(future_to_replay), total=len(replay_paths), desc="Processing replays"):
                replay_path = future_to_replay[future]
                completed += 1

                try:
                    # Get result from worker
                    success, processing_time, error = future.result()

                    # Update results
                    results['processing_times'][replay_path] = processing_time

                    if success:
                        results['successful'].append(replay_path)
                        results['successful_count'] += 1
                        logger.info(
                            f"[{completed}/{len(replay_paths)}] SUCCESS: {replay_path.name} "
                            f"({processing_time:.2f}s)"
                        )
                    else:
                        results['failed'].append((replay_path, error))
                        results['failed_count'] += 1
                        logger.error(
                            f"[{completed}/{len(replay_paths)}] FAILED: {replay_path.name} - {error}"
                        )

                except Exception as e:
                    # Worker itself crashed
                    results['failed'].append((replay_path, f"Worker crashed: {e}"))
                    results['failed_count'] += 1
                    logger.error(
                        f"[{completed}/{len(replay_paths)}] CRASHED: {replay_path.name} - {e}"
                    )

        # Calculate statistics
        total_time = time.time() - start_time
        results['total_time_seconds'] = total_time

        if results['successful_count'] > 0:
            results['average_time_per_replay'] = (
                sum(results['processing_times'].values()) / results['successful_count']
            )

        # Log summary
        logger.info("=" * 60)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info(f"  Total replays: {results['total_replays']}")
        logger.info(f"  Successful: {results['successful_count']}")
        logger.info(f"  Failed: {results['failed_count']}")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Average time per replay: {results['average_time_per_replay']:.2f}s")
        logger.info("=" * 60)

        return results

    def process_replay_directory(
        self,
        replay_dir: Path,
        output_dir: Path,
        pattern: str = '*.SC2Replay'
    ) -> Dict[str, Any]:
        """
        Process all replays in a directory.

        Finds all replay files matching the pattern and processes them in parallel.

        Args:
            replay_dir: Directory containing replay files
            output_dir: Directory for output files
            pattern: Glob pattern for replay files (default: '*.SC2Replay')

        Returns:
            Batch processing results (same as process_replay_batch)

        Raises:
            FileNotFoundError: If replay directory doesn't exist
            ValueError: If no replays found

        # TODO: Test case - Process directory of replays
        # TODO: Test case - Handle empty directory
        # TODO: Test case - Filter by pattern correctly
        """
        replay_dir = Path(replay_dir)

        if not replay_dir.exists():
            raise FileNotFoundError(f"Replay directory not found: {replay_dir}")

        if not replay_dir.is_dir():
            raise ValueError(f"Path is not a directory: {replay_dir}")

        # Find all replay files
        replay_paths = list(replay_dir.glob(pattern))

        if not replay_paths:
            raise ValueError(f"No replay files found in {replay_dir} with pattern '{pattern}'")

        logger.info(f"Found {len(replay_paths)} replays in {replay_dir}")

        # Process batch
        return self.process_replay_batch(replay_paths, output_dir)

    def process_replay_directory_recursive(
        self,
        replay_dir: Path,
        output_dir: Path,
        pattern: str = '**/*.SC2Replay'
    ) -> Dict[str, Any]:
        """
        Recursively process all replays in a directory tree.

        Args:
            replay_dir: Root directory to search
            output_dir: Directory for output files
            pattern: Glob pattern (default: **/*.SC2Replay for recursive search)

        Returns:
            Batch processing results

        # TODO: Test case - Process nested directory structure
        """
        replay_dir = Path(replay_dir)

        if not replay_dir.exists():
            raise FileNotFoundError(f"Replay directory not found: {replay_dir}")

        # Find all replay files recursively
        replay_paths = list(replay_dir.glob(pattern))

        if not replay_paths:
            raise ValueError(f"No replay files found in {replay_dir} with pattern '{pattern}'")

        logger.info(f"Found {len(replay_paths)} replays in {replay_dir} (recursive)")

        # Process batch
        return self.process_replay_batch(replay_paths, output_dir)

    def get_processing_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate human-readable summary of processing results.

        Args:
            results: Results dictionary from process_replay_batch()

        Returns:
            Formatted summary string
        """
        summary_lines = [
            "=" * 60,
            "REPLAY PROCESSING SUMMARY",
            "=" * 60,
            f"Total replays: {results['total_replays']}",
            f"Successful: {results['successful_count']} ({results['successful_count']/results['total_replays']*100:.1f}%)",
            f"Failed: {results['failed_count']} ({results['failed_count']/results['total_replays']*100:.1f}%)",
            "",
            f"Total processing time: {results['total_time_seconds']:.2f}s",
            f"Average time per replay: {results['average_time_per_replay']:.2f}s",
        ]

        if results['failed']:
            summary_lines.extend([
                "",
                "Failed replays:",
            ])
            for replay_path, error in results['failed'][:10]:  # Show first 10
                summary_lines.append(f"  - {replay_path.name}: {error}")

            if len(results['failed']) > 10:
                summary_lines.append(f"  ... and {len(results['failed']) - 10} more")

        summary_lines.append("=" * 60)

        return "\n".join(summary_lines)

    def retry_failed_replays(
        self,
        failed_results: List[Tuple[Path, str]],
        output_dir: Path,
        max_retries: int = 1
    ) -> Dict[str, Any]:
        """
        Retry processing failed replays.

        Args:
            failed_results: List of (replay_path, error) tuples from previous batch
            output_dir: Output directory
            max_retries: Maximum number of retry attempts

        Returns:
            Batch processing results for retry attempt
        """
        retry_paths = [path for path, error in failed_results]

        logger.info(f"Retrying {len(retry_paths)} failed replays (max_retries: {max_retries})")

        return self.process_replay_batch(retry_paths, output_dir)


# Worker function for parallel processing
def _worker_process_replay(
    replay_path: Path,
    output_dir: Path,
    config: Dict[str, Any],
    log_file_path: Optional[str] = None
) -> Tuple[bool, float, Optional[str]]:
    """
    Worker function for parallel replay processing.

    This function is executed in a separate process by ProcessPoolExecutor.
    It creates its own ReplayExtractionPipeline instance and processes a
    single replay.

    Args:
        replay_path: Path to replay file
        output_dir: Output directory
        config: Configuration dictionary
        log_file_path: Absolute path to the shared log file for worker logging

    Returns:
        Tuple of (success, processing_time, error_message):
        - success: True if processing succeeded
        - processing_time: Time in seconds
        - error_message: Error string if failed, None if successful

    # TODO: Test case - Worker processes replay successfully
    # TODO: Test case - Worker handles exceptions gracefully
    # TODO: Test case - Worker returns correct timing information
    """
    import time
    import logging

    # Configure logging for this worker process (must happen before any log calls)
    # On Windows, spawned processes do not inherit the parent's logging config.
    if log_file_path:
        from src_new.pipeline.logging_config import setup_worker_logging
        setup_worker_logging(log_file_path)

    # Initialize FLAGS for pysc2 in this worker process
    from absl import flags
    from pysc2 import run_configs
    FLAGS = flags.FLAGS
    try:
        FLAGS(['worker'], known_only=True)
    except:
        pass  # FLAGS already parsed, ignore

    worker_logger = logging.getLogger(__name__)

    start_time = time.time()

    try:
        # Create pipeline instance for this worker
        pipeline = ReplayExtractionPipeline(config)

        # Process the replay
        result = pipeline.process_replay(replay_path, output_dir)

        processing_time = time.time() - start_time

        if result['success']:
            return (True, processing_time, None)
        else:
            return (False, processing_time, result.get('error', 'Unknown error'))

    except Exception as e:
        processing_time = time.time() - start_time
        error_message = f"{type(e).__name__}: {str(e)}"
        worker_logger.exception(f"Worker failed for {replay_path.name}: {error_message}")
        return (False, processing_time, error_message)


# Convenience function for quick batch processing
def process_directory_quick(
    replay_dir: Path,
    output_dir: Optional[Path] = None,
    num_workers: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    log_file_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to process a directory of replays.

    Args:
        replay_dir: Directory containing replay files
        output_dir: Output directory (default: data/processed)
        num_workers: Number of parallel workers (default: CPU count)
        config: Optional configuration dictionary
        log_file_path: Absolute path to the shared log file (passed to workers)

    Returns:
        Batch processing results

    Example:
        >>> results = process_directory_quick(Path("replays/"))
        >>> print(f"Processed {results['successful_count']} of {results['total_replays']} replays")
    """
    if config == None:
        config = {
            # Observation settings
            'show_cloaked': True,
            'show_burrowed_shadows': True,
            'show_placeholders': True,

            # Processing settings
            'processing_mode': 'observer',
            'step_size': 1,  # Game loops per step

            # Output settings
            'compression': 'snappy',  # options: 'snappy' 'gzip', 'brotli', 'zstd'
        }
    output_dir = output_dir or Path('data/processed')

    processor = ParallelReplayProcessor(config, num_workers, log_file_path=log_file_path)
    return processor.process_replay_directory(replay_dir, output_dir)

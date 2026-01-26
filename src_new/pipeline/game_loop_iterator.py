"""
GameLoopIterator: Steps through SC2 replay game loops.

This component handles:
- Iterating through game loops with configurable step size
- Yielding observations at each step
- Detecting game end conditions
- Managing step multiplier for performance/detail tradeoff
"""

from typing import Generator, Optional
import logging

from s2clientprotocol import sc2api_pb2 as sc_pb


logger = logging.getLogger(__name__)


class GameLoopIterator:
    """
    Iterator for stepping through SC2 replay game loops.

    This class provides a clean interface for iterating through a replay's
    game loops and extracting observations at regular intervals.
    """

    def __init__(
        self,
        controller,
        step_mul: int = 8,
        max_loops: Optional[int] = None,
    ):
        """
        Initialize the GameLoopIterator.

        Args:
            controller: SC2 controller instance (from ReplayLoader)
            step_mul: Number of game loops to step forward each iteration
                     (8 = ~0.35 seconds, 22 = ~1 second)
            max_loops: Maximum number of loops to process (None = all)
        """
        self.controller = controller
        self.step_mul = step_mul
        self.max_loops = max_loops

        self.current_loop = 0
        self.observation_count = 0
        self.game_ended = False

    def __iter__(self) -> Generator:
        """
        Iterate through game loops, yielding observations.

        Yields:
            observation: SC2 observation at current game loop

        Example:
            >>> iterator = GameLoopIterator(controller, step_mul=8)
            >>> for obs in iterator:
            >>>     game_loop = obs.observation.game_loop
            >>>     units = obs.observation.raw_data.units
            >>>     # Process observation...
        """
        # Get initial observation
        logger.info(f"Starting game loop iteration (step_mul={self.step_mul})")
        self.controller.step()

        while True:
            # Get current observation
            obs = self.controller.observe()

            # Check if game has ended
            if obs.player_result:
                logger.info("Game ended")
                for result in obs.player_result:
                    result_name = sc_pb.Result.Name(result.result)
                    logger.info(f"  Player {result.player_id}: {result_name}")
                self.game_ended = True
                break

            # Update current loop
            self.current_loop = obs.observation.game_loop
            self.observation_count += 1

            # Check max loops limit
            if self.max_loops and self.current_loop >= self.max_loops:
                logger.info(f"Reached max loops limit ({self.max_loops})")
                break

            # Yield observation
            yield obs

            # Step forward
            self.controller.step(self.step_mul)

        logger.info(f"Iteration complete. Processed {self.observation_count} observations")

    def get_observation(self):
        """
        Get current observation without stepping.

        Returns:
            Current observation from controller
        """
        return self.controller.observe()

    def step(self, step_mul: Optional[int] = None):
        """
        Manually step forward.

        Args:
            step_mul: Number of loops to step (uses default if None)
        """
        step_size = step_mul if step_mul is not None else self.step_mul
        self.controller.step(step_size)

    def reset(self):
        """Reset iteration state."""
        self.current_loop = 0
        self.observation_count = 0
        self.game_ended = False


def iterate_replay(
    controller,
    step_mul: int = 8,
    max_loops: Optional[int] = None,
    callback=None,
) -> int:
    """
    Convenience function to iterate through entire replay.

    Args:
        controller: SC2 controller instance
        step_mul: Number of game loops per step
        max_loops: Maximum loops to process
        callback: Optional function called for each observation: callback(obs, loop_num)

    Returns:
        Total number of observations processed

    Example:
        >>> def process_obs(obs, loop_num):
        >>>     print(f"Loop {loop_num}: {len(obs.observation.raw_data.units)} units")
        >>>
        >>> count = iterate_replay(controller, step_mul=22, callback=process_obs)
        >>> print(f"Processed {count} observations")
    """
    iterator = GameLoopIterator(controller, step_mul=step_mul, max_loops=max_loops)

    for obs in iterator:
        if callback:
            callback(obs, iterator.current_loop)

    return iterator.observation_count


def extract_all_observations(
    controller,
    step_mul: int = 8,
    max_loops: Optional[int] = None,
) -> list:
    """
    Extract all observations from replay into a list.

    Warning: This loads all observations into memory. For large replays,
    use GameLoopIterator directly and process observations incrementally.

    Args:
        controller: SC2 controller instance
        step_mul: Number of game loops per step
        max_loops: Maximum loops to process

    Returns:
        List of all observations

    Example:
        >>> observations = extract_all_observations(controller, step_mul=22)
        >>> print(f"Extracted {len(observations)} observations")
        >>> first_obs = observations[0]
    """
    observations = []
    iterator = GameLoopIterator(controller, step_mul=step_mul, max_loops=max_loops)

    for obs in iterator:
        observations.append(obs)

    return observations

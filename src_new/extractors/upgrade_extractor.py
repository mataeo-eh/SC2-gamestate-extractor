"""
UpgradeExtractor: Extracts upgrade data from SC2 observations.

This component handles:
- Extracting upgrade completion from raw player data
- Tracking upgrade completion across frames
- Looking up upgrade names via pysc2.lib.upgrades
- Parsing upgrade category and level
- Detecting newly completed upgrades
"""

from typing import Dict, Set, Tuple, Optional
import logging
import re

from pysc2.lib import upgrades as pysc2_upgrades


logger = logging.getLogger(__name__)


def get_upgrade_name(upgrade_id: int) -> str:
    """
    Convert upgrade ID to human-readable name.

    Args:
        upgrade_id: SC2 upgrade ID

    Returns:
        Upgrade name string
    """
    try:
        return pysc2_upgrades.Upgrades(upgrade_id).name
    except (ValueError, AttributeError):
        return f"Unknown({upgrade_id})"


def parse_upgrade_details(upgrade_name: str) -> Tuple[str, int]:
    """
    Parse upgrade name to extract category and level.

    Args:
        upgrade_name: Human-readable upgrade name (e.g., "TerranInfantryWeaponsLevel1")

    Returns:
        Tuple of (category, level):
        - category: Type of upgrade (e.g., "weapons", "armor", "shields", "other")
        - level: Upgrade level (0 if not applicable, 1-3 for leveled upgrades)

    Examples:
        >>> parse_upgrade_details("TerranInfantryWeaponsLevel1")
        ("weapons", 1)
        >>> parse_upgrade_details("Stimpack")
        ("other", 0)
        >>> parse_upgrade_details("ProtossGroundArmorLevel2")
        ("armor", 2)
    """
    category = "other"
    level = 0

    # Convert to lowercase for easier matching
    name_lower = upgrade_name.lower()

    # Determine category
    if any(keyword in name_lower for keyword in ["weapon", "weapons", "melee", "missile", "ship", "attack"]):
        category = "weapons"
    elif "armor" in name_lower or "armour" in name_lower:
        category = "armor"
    elif "shield" in name_lower or "shields" in name_lower:
        category = "shields"
    elif "speed" in name_lower or "movement" in name_lower:
        category = "movement"
    elif "energy" in name_lower or "capacity" in name_lower:
        category = "energy"
    else:
        category = "other"

    # Extract level (look for patterns like "Level1", "Level2", etc.)
    level_match = re.search(r'level(\d)', name_lower)
    if level_match:
        level = int(level_match.group(1))

    return category, level


class UpgradeExtractor:
    """
    Extracts upgrade data from SC2 observations.

    This class tracks upgrades across frames, detects newly completed upgrades,
    and provides upgrade information with categorization for ground truth data.

    Example usage:
        extractor = UpgradeExtractor(player_id=1)

        for obs in game_loop_iterator:
            upgrades_data = extractor.extract(obs)

            # Check for new upgrades
            new_upgrades = extractor.get_new_upgrades()
            if new_upgrades:
                print(f"New upgrades completed: {new_upgrades}")

            # Get upgrade summary
            summary = extractor.get_upgrade_summary(upgrades_data)
            print(f"Upgrades: {summary}")
    """

    def __init__(self, player_id: int):
        """
        Initialize the UpgradeExtractor.

        Args:
            player_id: Player ID this extractor is tracking (1 or 2)
        """
        self.player_id = player_id

        # Track upgrades from previous frame
        self.previous_upgrades: Set[int] = set()

        # Track all completed upgrades with completion timestamps
        self.upgrade_completion_times: Dict[int, int] = {}  # upgrade_id -> game_loop

        # Track newly completed upgrades in the current frame
        self.newly_completed: Set[int] = set()

    def extract(self, obs) -> Dict[str, Dict]:
        """
        Extract all upgrade data from observation.

        Args:
            obs: SC2 observation from controller.observe()

        Returns:
            Dictionary mapping upgrade names to upgrade data:
            {
                'terraninfantryweaponslevel1': {
                    'upgrade_id': 7,
                    'upgrade_name': 'TerranInfantryWeaponsLevel1',
                    'category': 'weapons',
                    'level': 1,
                    'completed': True,
                    'completed_loop': 5000,
                },
                'stimpack': {
                    'upgrade_id': 15,
                    'upgrade_name': 'Stimpack',
                    'category': 'other',
                    'level': 0,
                    'completed': True,
                    'completed_loop': 3500,
                },
                ...
            }
        """
        try:
            raw_data = obs.observation.raw_data
            game_loop = obs.observation.game_loop

            # Get current upgrades from raw player data
            current_upgrades = set(raw_data.player.upgrade_ids)

            # Detect newly completed upgrades
            self.newly_completed = current_upgrades - self.previous_upgrades

            # Record completion times for new upgrades
            for upgrade_id in self.newly_completed:
                if upgrade_id not in self.upgrade_completion_times:
                    self.upgrade_completion_times[upgrade_id] = game_loop
                    logger.info(f"Player {self.player_id}: Upgrade {get_upgrade_name(upgrade_id)} "
                              f"completed at loop {game_loop}")

            # Build upgrades data dictionary
            upgrades_data = {}

            for upgrade_id in current_upgrades:
                upgrade_name = get_upgrade_name(upgrade_id)
                category, level = parse_upgrade_details(upgrade_name)

                # Use lowercase name as key for consistency
                key = upgrade_name.lower()

                upgrade_data = {
                    'upgrade_id': upgrade_id,
                    'upgrade_name': upgrade_name,
                    'category': category,
                    'level': level,
                    'completed': True,
                    'completed_loop': self.upgrade_completion_times.get(upgrade_id, None),
                }

                upgrades_data[key] = upgrade_data

            # Update previous upgrades for next iteration
            self.previous_upgrades = current_upgrades

            return upgrades_data

        except Exception as e:
            logger.error(f"Error extracting upgrade data: {e}")
            # Return empty dictionary on error
            return {}

    def get_new_upgrades(self) -> Set[int]:
        """
        Get upgrade IDs that were newly completed in the last extract() call.

        Returns:
            Set of upgrade IDs that were just completed
        """
        return self.newly_completed

    def get_upgrade_summary(self, upgrades_data: Dict[str, Dict]) -> str:
        """
        Get a human-readable summary of upgrades.

        Args:
            upgrades_data: Output from extract()

        Returns:
            Formatted string summary

        Example:
            "3 upgrades (Weapons: 1, Armor: 1, Other: 1)"
        """
        if not upgrades_data:
            return "No upgrades completed"

        # Count by category
        category_counts = {}
        for upgrade_data in upgrades_data.values():
            category = upgrade_data['category']
            category_counts[category] = category_counts.get(category, 0) + 1

        total = len(upgrades_data)
        categories_str = ", ".join([f"{cat.capitalize()}: {count}"
                                   for cat, count in sorted(category_counts.items())])

        return f"{total} upgrades ({categories_str})"

    def get_upgrades_by_category(self, upgrades_data: Dict[str, Dict]) -> Dict[str, list]:
        """
        Get upgrades grouped by category.

        Args:
            upgrades_data: Output from extract()

        Returns:
            Dictionary mapping categories to lists of upgrade names:
            {
                'weapons': ['TerranInfantryWeaponsLevel1', 'TerranInfantryWeaponsLevel2'],
                'armor': ['TerranInfantryArmorLevel1'],
                'other': ['Stimpack', 'CombatShield']
            }
        """
        by_category = {}

        for upgrade_data in upgrades_data.values():
            category = upgrade_data['category']
            upgrade_name = upgrade_data['upgrade_name']

            if category not in by_category:
                by_category[category] = []

            by_category[category].append(upgrade_name)

        return by_category

    def get_upgrade_count(self, upgrades_data: Dict[str, Dict]) -> int:
        """
        Get total number of completed upgrades.

        Args:
            upgrades_data: Output from extract()

        Returns:
            Number of completed upgrades
        """
        return len(upgrades_data)

    def has_upgrade(self, upgrades_data: Dict[str, Dict], upgrade_name: str) -> bool:
        """
        Check if a specific upgrade has been completed.

        Args:
            upgrades_data: Output from extract()
            upgrade_name: Name of upgrade to check (case-insensitive)

        Returns:
            True if upgrade is completed, False otherwise

        Example:
            >>> has_upgrade(upgrades_data, "Stimpack")
            True
        """
        return upgrade_name.lower() in upgrades_data

    def reset(self):
        """Reset all tracking state."""
        self.previous_upgrades.clear()
        self.upgrade_completion_times.clear()
        self.newly_completed.clear()

"""
A discretization script for reducing dataset complexity to only the engineered features related to army composition and clustering
This module performs the following steps:
1. Loads the engineered features dataset (output from engineer_army_features.py)
2. Drops all columns except the engineered features added in engineer_army_features.py
3. Saves the resulting dataset to a new directory (e.g., "data/quickstart/discretized")

This is intended for training simpler models for a proof of concept of strategy prediction based on army composition, without the complexity of all the raw features.
"""


import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

# Ensure the project root is on sys.path so src_new can be imported
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from tqdm import tqdm

from src_new.pipeline.logging_config import setup_logging

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def drop_columns(input_dir, output_dir):
    """
    Loads a parquet file, drops all columns except the engineered features, and saves the result to a new parquet file in the "discretized" directory.
    """
    for file in tqdm(Path(input_dir).glob("*.parquet"), desc="Discretizing datasets"):
        df = pd.read_parquet(file)
        
        # Define the columns to keep (the engineered features added in engineer_army_features.py)
        columns_to_keep = [
            "p1_main_army_direction",
            "p1_army_complexity_ratio",
            "p1_main_army_size",
            "p1_army_count",
            "p1_supply_cap",
            "p1_supply_used",
            "p2_main_army_direction",
            "p2_main_army_size",
            "p2_army_count",
            "p2_army_complexity_ratio",
            "p2_supply_cap",
            "p2_supply_used",
        ]
        
        try:
            # Drop all other columns
            df_discretized = df[columns_to_keep]
        except KeyError as e:
            logger.error(f"Missing expected columns in {file}: {e}")
            continue
        # Save to new directory
        output_dir.mkdir(exist_ok=True)
        output_filepath = output_dir / Path(file).name
        df_discretized.to_parquet(output_filepath)

def main(input_dir, output_dir):
    setup_logging()
    logger.info(f"Starting discretization: input={input_dir}, output={output_dir}")
    drop_columns(input_dir, output_dir)
    logger.info("Discretization completed.")
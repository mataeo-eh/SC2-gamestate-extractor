import os
from kaggle.api.kaggle_api_extended import KaggleApi 
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]

def upload_to_kaggle():
    """
    Returns:
        success (bool): Whether the upload was successful
        message (str): Error message if any
    """

    try:
        # Initialize and authenticate
        api = KaggleApi()
        api.authenticate()

        # 1. Download current dataset to a local folder
        dataset_name = "mataeoanderson/sc2-replay-data"
        download_path = ROOT / "data" / "quickstart"

        api.dataset_download_files(dataset_name, path=download_path, unzip=True)

        # 2. Add your new parsed replay data
        # (your parsing code that generates new files)
        # Save new files into the download_path folder
        # e.g., new_features.csv, new_games.parquet, etc.

        # 3. Create a new version with all files (old + new)
        api.dataset_create_version(
            folder=download_path,
            version_notes="Added parsed data from replays",
            quiet=False
        )
    except Exception as e:
        print(f"Error during Kaggle upload: {e}")
        return False, str(e)
    return True

def create_metadata_file():
    """Create a metadata file required by Kaggle datasets."""
    download_path = ROOT / "data" / "quickstart"
    metadata = {
    "title": "StarCraft II Bot Replay Features",
    "id": "mataeoanderson/sc2-replay-data",  # slug for your dataset
    "licenses": [{"name": "MIT"}]  # or other license
    }
    with open(os.path.join(download_path, "dataset-metadata.json"), "w") as f:
        json.dump(metadata, f)



if __name__ == "__main__":
    upload_to_kaggle()
    
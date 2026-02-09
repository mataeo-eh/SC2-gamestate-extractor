import os
from kaggle.api.kaggle_api_extended import KaggleApi 
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]

def upload_to_kaggle(dataset_name, download_path):
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
        # Note: If you want to download existing files, uncomment the line below. Otherwise the dataset version will be created containing only the new files you add.
        #api.dataset_download_files(dataset_name, path=download_path, unzip=True)

        # 2. Add your new parsed replay data
        # (your parsing code that generates new files)
        # Save new files into the download_path folder
        # e.g., new_features.csv, new_games.parquet, etc.

        # 3. Create a new version with all files (old + new)
        api.dataset_create_version(
            folder=str(download_path),
            version_notes="Added parsed data from replays",
            dir_mode='zip',
            quiet=False
        )
    except Exception as e:
        print(f"Error during Kaggle upload: {e}")
        return False, str(e)
    return True, "Success"

def create_metadata_file(dataset_name, download_path, Title = "StarCraft II Bot Replay Features"):
    """Create a metadata file required by Kaggle datasets."""
    metadata = {
    "title": Title,
    "id": dataset_name,  # slug for your dataset
    "licenses": [{"name": "MIT"}]  # or other license
    }
    with open(download_path / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f)

def main(dataset_name, download_path, Title = "StarCraft II Bot Replay Features"):
    if not (download_path/"dataset-metadata.json").exists():
        create_metadata_file(dataset_name, download_path, Title)
    success, message = upload_to_kaggle(dataset_name, download_path)
    if success:
        print("Dataset uploaded successfully!")
    else:
        print(f"Failed to upload dataset: {message}")

if __name__ == "__main__":
    main(dataset_name = "mataeoanderson/sc2-replay-data", download_path = ROOT / "data" / "quickstart")
    
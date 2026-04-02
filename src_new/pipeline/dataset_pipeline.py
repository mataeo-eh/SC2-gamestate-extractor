import os
import shutil
import tempfile
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]

# Path to kernel-metadata.json and the Kaggle notebook in the EDA directory
EDA_DIR = ROOT / "EDA"
KERNEL_METADATA_PATH = EDA_DIR / "kernel-metadata.json"
KAGGLE_NOTEBOOK_NAME = "data_verification_kaggle.ipynb"

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

def push_kaggle_notebook():
    """
    Push the Kaggle EDA notebook to Kaggle using `kaggle kernels push`.

    Copies the Kaggle notebook and kernel-metadata.json into a temporary
    directory, pushes via the Kaggle API, then cleans up the temp directory.

    First push creates the kernel on Kaggle; subsequent pushes create new
    versions. The kernel-metadata.json in EDA/ controls the kernel slug,
    title, visibility, and attached dataset sources.

    Depends on / calls:
        - KERNEL_METADATA_PATH (module-level constant)
        - KAGGLE_NOTEBOOK_NAME (module-level constant)
        - EDA_DIR (module-level constant)
        - KaggleApi.kernels_push() for the actual upload

    Returns:
        success (bool): Whether the push was successful.
        message (str): Status or error message.
    """
    if not KERNEL_METADATA_PATH.exists():
        msg = f"kernel-metadata.json not found at {KERNEL_METADATA_PATH}"
        print(f"Error: {msg}")
        return False, msg

    notebook_path = EDA_DIR / KAGGLE_NOTEBOOK_NAME
    if not notebook_path.exists():
        msg = f"Kaggle notebook not found at {notebook_path}"
        print(f"Error: {msg}")
        return False, msg

    try:
        # Create a temp directory with the notebook and metadata for kaggle push
        tmp_dir = tempfile.mkdtemp(prefix="kaggle_notebook_push_")
        shutil.copy2(KERNEL_METADATA_PATH, Path(tmp_dir) / "kernel-metadata.json")
        shutil.copy2(notebook_path, Path(tmp_dir) / KAGGLE_NOTEBOOK_NAME)

        api = KaggleApi()
        api.authenticate()
        api.kernels_push(tmp_dir)
    except Exception as e:
        print(f"Error pushing notebook to Kaggle: {e}")
        return False, str(e)
    finally:
        # Clean up temp directory regardless of success or failure
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return True, "Success"


if __name__ == "__main__":
    main(dataset_name = "mataeoanderson/sc2-replay-data", download_path = ROOT / "data" / "quickstart")
    
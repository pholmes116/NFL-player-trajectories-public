import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

def download_nfl_data():
    # Define paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / 'raw_data'
    zip_file = data_dir / 'nfl-big-data-bowl-2025.zip'

    # Make sure the data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Authenticate Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download the dataset
    print("Downloading dataset...")
    api.competition_download_files(
        competition='nfl-big-data-bowl-2025',
        path=str(data_dir),
        quiet=False
    )

    # Extract the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    # Optionally, remove the zip file to save space
    print("Cleaning up...")
    zip_file.unlink()

    print(f"Data downloaded and extracted to {data_dir}")

if __name__ == "__main__":
    download_nfl_data()

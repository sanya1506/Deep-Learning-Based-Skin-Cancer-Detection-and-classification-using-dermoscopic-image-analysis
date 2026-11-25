"""Download the HAM10000 dataset from Kaggle into data/raw/.

Supports both Kaggle credential formats:
  * new API token:  ~/.kaggle/access_token  (or KAGGLE_API_TOKEN env var)
  * legacy:         ~/.kaggle/kaggle.json   (or KAGGLE_USERNAME / KAGGLE_KEY env vars)

Usage:
    python scripts/download_data.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

MANUAL_INSTRUCTIONS = f"""
------------------------------------------------------------------------------
Could not download HAM10000 automatically.

Option A - Kaggle API credentials
  1. Sign in at https://www.kaggle.com -> Settings -> API -> "Generate New Token".
  2. Save the token:
       macOS/Linux : ~/.kaggle/access_token        (new format, token string only)
                     or ~/.kaggle/kaggle.json      (legacy format)
       Windows     : %USERPROFILE%\\.kaggle\\access_token  (or kaggle.json)
     Alternatively set the KAGGLE_API_TOKEN environment variable.
  3. Re-run:  python scripts/download_data.py

Option B - Manual download
  1. Open https://www.kaggle.com/datasets/{config.KAGGLE_DATASET}
  2. Click "Download", unzip the archive into:
       {config.RAW_DIR}
  3. You should end up with {config.METADATA_CSV_NAME} and the folders
     HAM10000_images_part_1/ and HAM10000_images_part_2/ inside data/raw/.
------------------------------------------------------------------------------
"""


def has_credentials() -> bool:
    kaggle_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return any([
        os.environ.get("KAGGLE_API_TOKEN"),
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"),
        (kaggle_dir / "access_token").exists(),
        (kaggle_dir / "kaggle.json").exists(),
    ])


def dataset_present() -> bool:
    return any(config.RAW_DIR.rglob(config.METADATA_CSV_NAME)) and \
        sum(1 for _ in config.RAW_DIR.rglob("ISIC_*.jpg")) >= 10000


def main() -> int:
    if dataset_present():
        print(f"HAM10000 already present in {config.RAW_DIR} - nothing to do.")
        return 0

    if not has_credentials():
        print("No Kaggle credentials found (access_token, kaggle.json or env vars).")
        print(MANUAL_INSTRUCTIONS)
        return 1

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("The 'kaggle' package is not installed. Run: pip install -U kaggle")
        print(MANUAL_INSTRUCTIONS)
        return 1

    try:
        api = KaggleApi()
        api.authenticate()
        print(f"Downloading {config.KAGGLE_DATASET} into {config.RAW_DIR} (~2.7 GB) ...")
        api.dataset_download_files(config.KAGGLE_DATASET, path=str(config.RAW_DIR),
                                   unzip=True, quiet=False)
    except Exception as exc:  # network / auth / quota errors
        print(f"Kaggle download failed: {exc}")
        print(MANUAL_INSTRUCTIONS)
        return 1

    if not dataset_present():
        print("Download finished but the expected files were not found.")
        print(MANUAL_INSTRUCTIONS)
        return 1
    n_imgs = sum(1 for _ in config.RAW_DIR.rglob("ISIC_*.jpg"))
    print(f"Done. Found {n_imgs} image files and metadata in {config.RAW_DIR}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

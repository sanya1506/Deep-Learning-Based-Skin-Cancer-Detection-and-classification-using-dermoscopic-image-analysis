"""HAM10000 metadata loading, leakage-free splitting and the PyTorch Dataset.

Splits are stratified by diagnosis and grouped by lesion_id: every image of a given
lesion lands in exactly one of train / val / test (70 / 15 / 15).
"""
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from preprocess import load_image_rgb, preprocess_path  # noqa: E402

CLASS_TO_IDX = {c: i for i, c in enumerate(config.CLASS_NAMES)}


# --------------------------------------------------------------------------- metadata
def find_metadata_csv() -> Path:
    matches = sorted(config.RAW_DIR.rglob(config.METADATA_CSV_NAME))
    if not matches:
        raise FileNotFoundError(
            f"{config.METADATA_CSV_NAME} not found under {config.RAW_DIR}. "
            "Run: python scripts/download_data.py")
    return matches[0]


def load_metadata() -> pd.DataFrame:
    """Metadata with an absolute `path` column and integer `label`."""
    df = pd.read_csv(find_metadata_csv())
    paths = {}
    for p in config.RAW_DIR.rglob("ISIC_*.jpg"):
        paths.setdefault(p.stem, p)  # the Kaggle archive contains duplicate folders
    df["path"] = df["image_id"].map(paths)
    missing = df["path"].isna().sum()
    if missing:
        print(f"Warning: {missing} images listed in metadata were not found; dropping them.")
        df = df.dropna(subset=["path"])
    df["label"] = df["dx"].map(CLASS_TO_IDX)
    df["binary"] = df["dx"].map(config.BINARY_MAP)
    return df.reset_index(drop=True)


def _resolve_paths(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["path"] = df["path"].map(lambda rel: str(config.RAW_DIR / Path(rel)))
    return df


def make_splits(force: bool = False) -> dict:
    """Create (or load cached) train/val/test splits grouped by lesion_id."""
    files = {s: config.SPLITS_DIR / f"{s}.csv" for s in ("train", "val", "test")}
    if not force and all(f.exists() for f in files.values()):
        return {s: _resolve_paths(pd.read_csv(f)) for s, f in files.items()}

    df = load_metadata()
    lesions = df.groupby("lesion_id")["dx"].first().reset_index()
    train_l, rest_l = train_test_split(
        lesions, test_size=config.VAL_FRAC + config.TEST_FRAC,
        stratify=lesions["dx"], random_state=config.SEED)
    val_l, test_l = train_test_split(
        rest_l, test_size=config.TEST_FRAC / (config.VAL_FRAC + config.TEST_FRAC),
        stratify=rest_l["dx"], random_state=config.SEED)

    splits = {}
    for name, lesion_df in (("train", train_l), ("val", val_l), ("test", test_l)):
        part = df[df["lesion_id"].isin(set(lesion_df["lesion_id"]))].copy()
        # store paths relative to data/raw (POSIX style) so the CSVs are portable across machines/OSes
        part["path"] = part["path"].map(lambda p: Path(p).relative_to(config.RAW_DIR).as_posix())
        part.to_csv(files[name], index=False)
        splits[name] = _resolve_paths(part)
    assert not (set(splits["train"].lesion_id) & set(splits["test"].lesion_id)), "lesion leakage"
    assert not (set(splits["train"].lesion_id) & set(splits["val"].lesion_id)), "lesion leakage"
    for name, part in splits.items():
        print(f"{name:5s}: {len(part):5d} images, {part.lesion_id.nunique():5d} lesions")
    return splits


def quick_subset(df: pd.DataFrame, per_class: int) -> pd.DataFrame:
    shuffled = df.sample(frac=1.0, random_state=config.SEED)
    return shuffled.groupby("dx").head(per_class).reset_index(drop=True)


# --------------------------------------------------------------------------- transforms
def get_transforms(train: bool):
    norm = [transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)]
    if not train:
        return transforms.Compose([transforms.ToPILImage()] + norm)
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)),  # zoom
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
    ] + norm)


# --------------------------------------------------------------------------- dataset
class SkinLesionDataset(Dataset):
    """Returns (normalised tensor, label). Uses cached preprocessed images when present."""

    def __init__(self, df: pd.DataFrame, train: bool = False):
        self.df = df.reset_index(drop=True)
        self.transform = get_transforms(train)

    def __len__(self):
        return len(self.df)

    def load_preprocessed(self, idx: int):
        row = self.df.iloc[idx]
        cached = config.PROCESSED_DIR / f"{row.image_id}.jpg"
        if cached.exists():
            return load_image_rgb(cached)
        return preprocess_path(Path(row.path))

    def __getitem__(self, idx):
        img = self.load_preprocessed(idx)
        return self.transform(img), int(self.df.iloc[idx]["label"])


def get_dataloaders(batch_size: int = config.BATCH_SIZE, quick: bool = False):
    splits = make_splits()
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    if quick:
        train_df = quick_subset(train_df, config.QUICK_SAMPLES_PER_CLASS)
        val_df = quick_subset(val_df, config.QUICK_SAMPLES_PER_CLASS // 2)
        test_df = quick_subset(test_df, config.QUICK_SAMPLES_PER_CLASS // 2)
    pin = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(SkinLesionDataset(train_df, train=True), batch_size=batch_size,
                            shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=pin),
        "val": DataLoader(SkinLesionDataset(val_df), batch_size=batch_size,
                          shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=pin),
        "test": DataLoader(SkinLesionDataset(test_df), batch_size=batch_size,
                           shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=pin),
    }
    return loaders, {"train": train_df, "val": val_df, "test": test_df}


if __name__ == "__main__":
    s = make_splits(force=True)
    for k, v in s.items():
        print(k, v["dx"].value_counts().to_dict())

"""Shared pieces of the improved (v2) pipeline.

* Paths / settings for the v2 pipeline
* Shades-of-Gray colour constancy (replaces CLAHE, which distorts diagnostic colour)
* Lesion cropping from a segmentation mask (U-Net) or an Otsu fallback
* Metadata encoding (age, sex, anatomical site) shared by HAM10000 and ISIC 2019
* Building the training table: HAM10000 train split + optional ISIC 2019 extra images,
  with strict leakage filters against the HAM10000 val/test lesions.

The HAM10000 val/test splits are the SAME CSVs as the original project
(data/splits/*.csv), so v1 and v2 numbers are directly comparable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402  (original project config: class names, splits dir, RAW_DIR)

CLASS_NAMES = config.CLASS_NAMES                       # akiec bcc bkl df mel nv vasc
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
MALIGNANT_IDX = [CLASS_TO_IDX[c] for c in config.MALIGNANT_CLASSES]

V2_DIR = ROOT / "data" / "v2"                           # cropped + colour-corrected cache
V2_RESULTS = ROOT / "results" / "v2"
V2_MODELS = ROOT / "models" / "v2"
for _d in (V2_DIR, V2_RESULTS, V2_MODELS):
    _d.mkdir(parents=True, exist_ok=True)

CACHE_SIZE = 512          # cached crops are 512x512; training uses 384 random crops of them

# ISIC 2019 ground-truth columns -> HAM10000 classes.
# AK and SCC both go to akiec (HAM's akiec = actinic keratosis + intraepithelial carcinoma/Bowen's).
ISIC2019_MAP = {"MEL": "mel", "NV": "nv", "BCC": "bcc", "AK": "akiec", "SCC": "akiec",
                "BKL": "bkl", "DF": "df", "VASC": "vasc"}

# --------------------------------------------------------------------------- colour constancy
def shades_of_gray(img_rgb: np.ndarray, power: int = 6) -> np.ndarray:
    """Shades-of-Gray colour constancy (Finlayson & Trezzi 2004).

    Standard in ISIC-winning pipelines: removes colour casts from different
    dermatoscopes/lighting while keeping lesion colours (unlike CLAHE).
    """
    img = img_rgb.astype(np.float32)
    illum = np.power(np.mean(np.power(img, power), axis=(0, 1)), 1.0 / power)
    illum = illum / (np.sqrt(np.sum(illum ** 2)) + 1e-6)
    img = img / (illum * np.sqrt(3.0) + 1e-6)
    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- lesion cropping
def otsu_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Classical fallback segmentation (no training needed): Otsu on blurred grey image."""
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (11, 11), 0)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    return (m > 0).astype(np.uint8)


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(np.uint8)
    h, w = mask.shape
    best, best_score = 0, -1.0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        # prefer big components that are not glued to the image border (vignette corners)
        touches = (x == 0) + (y == 0) + (x + bw == w) + (y + bh == h)
        score = area * (0.5 if touches >= 2 else 1.0)
        if score > best_score:
            best, best_score = i, score
    return (lab == best).astype(np.uint8)


def crop_to_lesion(img_rgb: np.ndarray, mask: np.ndarray | None, margin: float = 0.25,
                   min_frac: float = 0.01, max_frac: float = 0.90) -> np.ndarray:
    """Square crop around the lesion bounding box (+margin). Falls back to the full
    image when the mask is empty, tiny, or covers almost everything."""
    h, w = img_rgb.shape[:2]
    if mask is None:
        return img_rgb
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    mask = largest_component(mask)
    frac = mask.mean()
    if frac < min_frac or frac > max_frac:
        return img_rgb
    ys, xs = np.where(mask > 0)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    side = max(y1 - y0, x1 - x0) * (1 + 2 * margin)
    side = int(min(max(side, 0.35 * min(h, w)), min(h, w)))   # never crop absurdly tight
    top = int(np.clip(cy - side / 2, 0, h - side))
    left = int(np.clip(cx - side / 2, 0, w - side))
    return img_rgb[top:top + side, left:left + side]


# --------------------------------------------------------------------------- metadata
SITE_GROUPS = ["head_neck", "upper_ext", "lower_ext", "torso", "palms_soles", "oral_genital", "unknown"]
_HAM_SITE = {"scalp": "head_neck", "face": "head_neck", "ear": "head_neck", "neck": "head_neck",
             "upper extremity": "upper_ext", "hand": "upper_ext",
             "lower extremity": "lower_ext", "foot": "lower_ext",
             "back": "torso", "trunk": "torso", "chest": "torso", "abdomen": "torso",
             "acral": "palms_soles", "genital": "oral_genital"}
_ISIC_SITE = {"head/neck": "head_neck", "upper extremity": "upper_ext", "lower extremity": "lower_ext",
              "anterior torso": "torso", "posterior torso": "torso", "lateral torso": "torso",
              "torso": "torso", "palms/soles": "palms_soles", "oral/genital": "oral_genital"}
META_DIM = 2 + 3 + len(SITE_GROUPS)     # age, age_missing, sex(m/f/unk), site one-hot


def site_group(value, source: str = "ham") -> str:
    if not isinstance(value, str):
        return "unknown"
    table = _HAM_SITE if source == "ham" else _ISIC_SITE
    return table.get(value.strip().lower(), "unknown")


def encode_meta(age, sex, site_grp: str) -> np.ndarray:
    v = np.zeros(META_DIM, np.float32)
    if age is None or (isinstance(age, float) and np.isnan(age)):
        v[1] = 1.0
    else:
        v[0] = float(age) / 100.0
    sex = str(sex).lower() if isinstance(sex, str) else "unknown"
    v[2 + {"male": 0, "female": 1}.get(sex, 2)] = 1.0
    v[5 + SITE_GROUPS.index(site_grp if site_grp in SITE_GROUPS else "unknown")] = 1.0
    return v


# --------------------------------------------------------------------------- tables
def load_ham_splits() -> dict[str, pd.DataFrame]:
    """Original leakage-free HAM10000 splits (grouped by lesion_id)."""
    out = {}
    for s in ("train", "val", "test"):
        df = pd.read_csv(config.SPLITS_DIR / f"{s}.csv")
        df["src_path"] = df["path"].map(lambda p: str(config.RAW_DIR / p))
        df["site_grp"] = df["localization"].map(lambda v: site_group(v, "ham"))
        df["source"] = "ham"
        df["label"] = df["dx"].map(CLASS_TO_IDX)
        out[s] = df[["image_id", "lesion_id", "dx", "label", "age", "sex", "site_grp", "src_path", "source"]]
    return out


def load_isic2019_extra(isic_dir: Path, ham: dict[str, pd.DataFrame], ham_all_ids: set[str]) -> pd.DataFrame:
    """ISIC 2019 training images usable as EXTRA training data.

    Leakage filters (critical - ISIC 2019 already contains every HAM10000 image):
      1. drop any image whose id is a HAM10000 image (train ones are already in our train set,
         val/test ones would leak the test set);
      2. drop any image whose lesion_id belongs to a HAM val/test lesion;
      3. drop UNK-labelled rows.
    """
    isic_dir = Path(isic_dir)
    gt_files = sorted(isic_dir.rglob("ISIC_2019_Training_GroundTruth.csv"))
    if not gt_files:
        raise FileNotFoundError(f"ISIC_2019_Training_GroundTruth.csv not found under {isic_dir}")
    gt = pd.read_csv(gt_files[0])
    meta_files = sorted(isic_dir.rglob("ISIC_2019_Training_Metadata.csv"))
    if meta_files:
        meta = pd.read_csv(meta_files[0])
    else:   # metadata is optional: images are still useful, metadata becomes "unknown"
        print("ISIC 2019 metadata CSV not found - using unknown age/sex/site for extra images")
        meta = pd.DataFrame({"image": gt["image"]})
    for col in ("age_approx", "anatom_site_general", "lesion_id", "sex"):
        if col not in meta:
            meta[col] = np.nan
    df = gt.merge(meta, on="image", how="left")
    cols = list(ISIC2019_MAP)
    df = df[df[cols].sum(axis=1) > 0].copy()
    df["dx"] = df[cols].idxmax(axis=1).map(ISIC2019_MAP)
    df["image_id"] = df["image"].str.replace("_downsampled", "", regex=False)

    blocked_lesions = set(ham["val"].lesion_id) | set(ham["test"].lesion_id)
    n0 = len(df)
    df = df[~df["image_id"].isin(ham_all_ids)]
    df = df[~df["lesion_id"].isin(blocked_lesions)]
    print(f"ISIC 2019: {n0} labelled -> {len(df)} extra images after leakage filtering")

    paths = {p.stem: str(p) for p in isic_dir.rglob("*.jpg")}
    df["src_path"] = df["image"].map(paths)
    df = df.dropna(subset=["src_path"])
    df["label"] = df["dx"].map(CLASS_TO_IDX)
    df["age"] = df["age_approx"]
    df["site_grp"] = df["anatom_site_general"].map(lambda v: site_group(v, "isic"))
    df["lesion_id"] = df["lesion_id"].fillna(df["image_id"])
    df["source"] = "isic2019"
    return df[["image_id", "lesion_id", "dx", "label", "age", "sex", "site_grp", "src_path", "source"]]


def cached_path(image_id: str) -> Path:
    return V2_DIR / f"{image_id}.jpg"


def build_tables(isic2019_dir: str | None = None) -> dict[str, pd.DataFrame]:
    """train (HAM train [+ ISIC 2019 extra]), val and test (HAM only) with cache paths."""
    ham = load_ham_splits()
    tables = {"train": ham["train"], "val": ham["val"], "test": ham["test"]}
    if isic2019_dir:
        all_ids = set(pd.concat(ham.values()).image_id)
        extra = load_isic2019_extra(Path(isic2019_dir), ham, all_ids)
        tables["train"] = pd.concat([ham["train"], extra], ignore_index=True)
    for k, df in tables.items():
        df = df.copy()
        df["path"] = df["image_id"].map(lambda i: str(cached_path(i)))
        tables[k] = df.reset_index(drop=True)
    assert not set(tables["train"].lesion_id) & set(tables["test"].lesion_id), "lesion leakage"
    assert not set(tables["train"].image_id) & set(tables["test"].image_id), "image leakage"
    return tables

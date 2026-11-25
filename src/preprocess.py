"""Dermoscopic image preprocessing.

Pipeline (identical for training and inference):
    1. Hair removal   - morphological black-hat on the grey image -> hair mask -> Telea inpainting
    2. Enhancement    - CLAHE on the L channel of the LAB colour space
    3. Resize         - 224 x 224 (INTER_AREA)
    4. Normalisation  - scale to [0, 1] then ImageNet mean / std (done in the torch transforms)

Run as a script to cache the preprocessed dataset to data/processed/ (speeds up training):
    python src/preprocess.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def load_image_rgb(path) -> np.ndarray:
    """Read an image from disk as RGB uint8 (works with non-ASCII Windows paths)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def hair_mask(img_rgb: np.ndarray,
              kernel_size: int = config.HAIR_KERNEL_SIZE,
              threshold: int = config.HAIR_THRESHOLD,
              min_length: int = config.HAIR_MIN_LENGTH):
    """Black-hat response and binary hair mask.

    The black-hat transform responds to thin dark structures. Thresholding alone also catches skin
    texture and the lesion's pigment network, so only connected components that are long
    (bounding box >= min_length px) and sparse (fill ratio < 0.4), i.e. hair-shaped, are kept.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    w, h, area = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_AREA]
    keep = (np.maximum(w, h) >= min_length) & (area / np.maximum(w * h, 1) < 0.4)
    keep[0] = False  # background
    mask = np.where(keep[labels], 255, 0).astype(np.uint8)
    return blackhat, cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)


def remove_hair(img_rgb: np.ndarray, radius: int = config.INPAINT_RADIUS) -> np.ndarray:
    """Remove dark hair strands: black-hat hair mask followed by Telea inpainting."""
    _, mask = hair_mask(img_rgb)
    return cv2.inpaint(img_rgb, mask, radius, cv2.INPAINT_TELEA)


def apply_clahe(img_rgb: np.ndarray,
                clip_limit: float = config.CLAHE_CLIP_LIMIT,
                tile_grid=config.CLAHE_TILE_GRID) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation on the luminance channel."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid))
    lab = cv2.merge((clahe.apply(l_chan), a_chan, b_chan))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def resize(img_rgb: np.ndarray, size: int = config.IMG_SIZE) -> np.ndarray:
    return cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)


def normalize(img_rgb: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> float32 HxWx3 normalised with ImageNet statistics."""
    img = img_rgb.astype(np.float32) / 255.0
    return (img - np.array(config.IMAGENET_MEAN, np.float32)) / np.array(config.IMAGENET_STD, np.float32)


def preprocess_image(img_rgb: np.ndarray, size: int = config.IMG_SIZE) -> np.ndarray:
    """Hair removal -> CLAHE -> resize. Returns an RGB uint8 image (size x size).

    Hair removal runs at a fixed working resolution so the morphological kernel
    behaves the same regardless of the input image size.
    """
    h, w = img_rgb.shape[:2]
    scale = 600.0 / max(h, w)
    if abs(scale - 1.0) > 0.05:
        img_rgb = cv2.resize(img_rgb, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)
    img = remove_hair(img_rgb)
    img = apply_clahe(img)
    return resize(img, size)


def preprocess_path(path, size: int = config.IMG_SIZE) -> np.ndarray:
    return preprocess_image(load_image_rgb(path), size)


def save_image_rgb(img_rgb: np.ndarray, path) -> None:
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError(f"Could not encode image for {path}")
    buf.tofile(str(path))


def cache_dataset(force: bool = False) -> int:
    """Preprocess every HAM10000 image once into data/processed/<image_id>.jpg."""
    from tqdm import tqdm
    from dataset import load_metadata

    df = load_metadata()
    done = 0
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Preprocessing"):
        out = config.PROCESSED_DIR / f"{row.image_id}.jpg"
        if out.exists() and not force:
            continue
        save_image_rgb(preprocess_path(row.path), out)
        done += 1
    print(f"Preprocessed {done} new images into {config.PROCESSED_DIR}")
    return done


if __name__ == "__main__":
    import argparse

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    parser = argparse.ArgumentParser(description="Cache preprocessed HAM10000 images")
    parser.add_argument("--force", action="store_true", help="re-process existing files")
    cache_dataset(parser.parse_args().force)

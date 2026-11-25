"""Figures for the report / presentation: pipeline diagram, class distribution, preprocessing steps.

Usage:
    python scripts/make_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import config  # noqa: E402

FIG_DIR = config.RESULTS_DIR / "figures"


def _box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#2c3e50", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.4,
                                 color="#2c3e50"))


def pipeline_diagram(out: Path):
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    w, h = 2.0, 1.25
    top = [
        ("Dermoscopic\nimage (RGB)", "#d6eaf8"),
        ("Hair removal\nblack-hat +\nTelea inpainting", "#d5f5e3"),
        ("CLAHE\n(LAB L-channel)", "#d5f5e3"),
        ("Resize 224×224\n+ ImageNet\nnormalisation", "#d5f5e3"),
        ("CNN backbone\nEfficientNet-B0 /\nResNet50 / DenseNet121", "#fdebd0"),
        ("Softmax\n7 HAM10000\nclasses", "#fdebd0"),
    ]
    xs = [0.2 + i * 2.3 for i in range(len(top))]
    for x, (t, c) in zip(xs, top):
        _box(ax, x, 3.4, w, h, t, c)
    for x1, x2 in zip(xs[:-1], xs[1:]):
        _arrow(ax, x1 + w, 4.02, x2, 4.02)

    bottom = [
        (xs[5], "Benign / Malignant\nmapping +\nconfidence scores", "#fadbd8"),
        (xs[3] + 1.1, "Grad-CAM\n(last conv layer)\nheat-map overlay", "#e8daef"),
        (xs[1] + 1.0, "Report export\nPDF + CSV\nStreamlit UI", "#d6eaf8"),
    ]
    for x, t, c in bottom:
        _box(ax, x, 0.7, w, h, t, c)
    _arrow(ax, xs[5] + w / 2, 3.4, xs[5] + w / 2, 0.7 + h)
    _arrow(ax, xs[4] + w / 2, 3.4, xs[3] + 1.1 + w / 2, 0.7 + h)
    _arrow(ax, xs[5], 1.32, xs[3] + 1.1 + w, 1.32)
    _arrow(ax, xs[3] + 1.1, 1.32, xs[1] + 1.0 + w, 1.32)
    ax.text(0.2, 2.75, "Training: stratified 70/15/15 split by lesion_id · augmentation (flip, rotate, "
                       "colour jitter, zoom) · class-weighted cross-entropy · AdamW · ReduceLROnPlateau · "
                       "early stopping", fontsize=9, color="#555555")
    ax.set_title("System pipeline", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def class_distribution(out: Path):
    from dataset import make_splits

    splits = make_splits()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    import numpy as np
    x = np.arange(len(config.CLASS_NAMES))
    width = 0.27
    for i, (name, color) in enumerate([("train", "#2a78b8"), ("val", "#e08a1e"), ("test", "#3a9d5d")]):
        counts = splits[name].dx.value_counts().reindex(config.CLASS_NAMES, fill_value=0)
        bars = axes[0].bar(x + (i - 1) * width, counts.values, width, label=f"{name} ({counts.sum()})", color=color)
        axes[0].bar_label(bars, fontsize=7, padding=1)
    axes[0].set_xticks(x, config.CLASS_NAMES)
    axes[0].set_ylabel("Images")
    axes[0].set_title("HAM10000 class distribution per split (grouped by lesion_id)")
    axes[0].legend()
    axes[0].spines[["top", "right"]].set_visible(False)

    total = sum(len(s) for s in splits.values())
    mal = sum(s.dx.isin(config.MALIGNANT_CLASSES).sum() for s in splits.values())
    axes[1].pie([total - mal, mal], labels=["Benign", "Malignant"], autopct="%1.1f%%",
                colors=["#1e8449", "#c0392b"], startangle=90, textprops={"color": "black"})
    axes[1].set_title(f"Benign vs malignant (n={total})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def preprocessing_steps(out: Path):
    import pandas as pd
    from preprocess import apply_clahe, load_image_rgb, preprocess_image, remove_hair

    test = pd.read_csv(config.SPLITS_DIR / "train.csv")
    # pick an image with visible hair: largest black-hat response among a few candidates
    import cv2
    import numpy as np
    best, best_score = None, -1
    for rel in test.sample(60, random_state=1)["path"]:
        img = load_image_rgb(config.RAW_DIR / Path(rel))
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17)))
        score = float((bh > config.HAIR_THRESHOLD).mean())
        if score > best_score:
            best, best_score = img, score
    img = best
    no_hair = remove_hair(img)
    steps = [img, no_hair, apply_clahe(no_hair), preprocess_image(img)]
    titles = ["1. Original", "2. Hair removed (black-hat + inpaint)", "3. CLAHE enhanced", "4. Resized 224×224"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    for ax, im, t in zip(axes, steps, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _ = np


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_diagram(FIG_DIR / "pipeline.png")
    class_distribution(FIG_DIR / "class_distribution.png")
    preprocessing_steps(FIG_DIR / "preprocessing_steps.png")
    print(f"Figures written to {FIG_DIR}")

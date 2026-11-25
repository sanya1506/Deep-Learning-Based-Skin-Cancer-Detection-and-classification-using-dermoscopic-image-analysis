"""Lesion segmentation (U-Net) + crop/colour-constancy cache.

Step 1  train  : U-Net (EfficientNet-B4 encoder, segmentation_models_pytorch) on the
                 ISIC 2018 Task 1 masks (2,594 images). Images that are HAM10000 val/test
                 images are removed from the segmentation training set.
Step 2  cache  : for every image used by the classifier (HAM train/val/test + ISIC 2019 extra):
                 predict mask -> crop to lesion (+25% margin) -> Shades-of-Gray -> 512x512 JPEG.

    python improved/segmentation.py train --task1-images <dir> --task1-masks <dir>
    python improved/segmentation.py cache [--isic2019 <dir>] [--no-unet]   # --no-unet = Otsu fallback
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import (CACHE_SIZE, V2_MODELS, build_tables, cached_path, crop_to_lesion,
                    load_ham_splits, otsu_mask, shades_of_gray)

SEG_SIZE = 384
SEG_CKPT = V2_MODELS / "unet_effb4_isic2018.pth"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_unet(pretrained: bool = True):
    import segmentation_models_pytorch as smp
    return smp.Unet("efficientnet-b4", encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=1)


def to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    x = (img_rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


class SegDS(Dataset):
    def __init__(self, items, train: bool):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, mp = self.items[i]
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        m = (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        img = cv2.resize(img, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_NEAREST)
        if self.train:
            if random.random() < .5: img, m = img[:, ::-1], m[:, ::-1]
            if random.random() < .5: img, m = img[::-1], m[::-1]
            k = random.randint(0, 3); img, m = np.rot90(img, k), np.rot90(m, k)
            img = np.ascontiguousarray(img); m = np.ascontiguousarray(m)
            img = np.clip(img.astype(np.float32) * random.uniform(.8, 1.2) + random.uniform(-20, 20), 0, 255).astype(np.uint8)
        return to_tensor(img), torch.from_numpy(m[None].astype(np.float32))


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum((1, 2, 3)) + eps
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps
    return 1 - (num / den).mean()


def train(args):
    imgs = {p.stem: p for p in Path(args.task1_images).rglob("ISIC_*.jpg")}
    masks = {p.stem[:12]: p for p in Path(args.task1_masks).rglob("*.png")
             if p.stem.lower().endswith("_segmentation")}          # ISIC_0000000_segmentation.png
    ham = load_ham_splits()
    blocked = set(ham["val"].image_id) | set(ham["test"].image_id)
    ids = sorted(k for k in imgs if k in masks and k not in blocked)
    print(f"segmentation set: {len(ids)} images (removed {len([k for k in imgs if k in blocked])} HAM val/test images)")
    random.Random(0).shuffle(ids)
    n_val = int(.1 * len(ids))
    tr = [(imgs[k], masks[k]) for k in ids[n_val:]]
    va = [(imgs[k], masks[k]) for k in ids[:n_val]]
    dl_tr = DataLoader(SegDS(tr, True), batch_size=args.bs, shuffle=True, num_workers=args.workers, drop_last=True)
    dl_va = DataLoader(SegDS(va, False), batch_size=args.bs, num_workers=args.workers)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    bce = nn.BCEWithLogitsLoss()
    best = 0
    for ep in range(args.epochs):
        model.train()
        for x, y in tqdm(dl_tr, desc=f"seg ep{ep + 1}", leave=False):
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                out = model(x)
                loss = bce(out, y) + dice_loss(out, y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        model.eval(); inter = union = 0
        with torch.no_grad():
            for x, y in dl_va:
                p = (model(x.to(dev)) > 0).float(); y = y.to(dev)
                inter += (p * y).sum().item(); union += ((p + y) > 0).float().sum().item()
        iou = inter / max(union, 1)
        print(f"epoch {ep + 1}: val Jaccard {iou:.4f}")
        if iou > best:
            best = iou
            torch.save(model.state_dict(), SEG_CKPT)
    print(f"best val Jaccard {best:.4f} -> {SEG_CKPT}")


class LesionSegmenter:
    """Predicts a binary lesion mask with the trained U-Net (+ flip TTA)."""

    def __init__(self, ckpt=SEG_CKPT, device=None):
        self.dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_unet(pretrained=False)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.to(self.dev).eval()

    @torch.no_grad()
    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        x = to_tensor(cv2.resize(img_rgb, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_AREA))[None].to(self.dev)
        p = torch.sigmoid(self.model(x)) + torch.sigmoid(self.model(x.flip(3))).flip(3)
        return ((p / 2)[0, 0].cpu().numpy() > 0.5).astype(np.uint8)


def prepare_image(img_rgb: np.ndarray, segmenter=None) -> np.ndarray:
    """Full v2 preprocessing used for training AND inference: crop -> colour constancy -> resize."""
    mask = segmenter(img_rgb) if segmenter is not None else otsu_mask(img_rgb)
    crop = crop_to_lesion(img_rgb, mask)
    crop = shades_of_gray(crop)
    return cv2.resize(crop, (CACHE_SIZE, CACHE_SIZE), interpolation=cv2.INTER_AREA)


def cache(args):
    tables = build_tables(args.isic2019)
    df = pd.concat(tables.values()).drop_duplicates("image_id")
    seg = None if args.no_unet else LesionSegmenter()
    done = 0
    for _, r in tqdm(df.iterrows(), total=len(df), desc="crop+cache"):
        out = cached_path(r.image_id)
        if out.exists() and not args.force:
            continue
        img = cv2.cvtColor(cv2.imread(r.src_path), cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(out), cv2.cvtColor(prepare_image(img, seg), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        done += 1
    print(f"cached {done} new images -> {out.parent}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--task1-images", required=True)
    t.add_argument("--task1-masks", required=True)
    t.add_argument("--epochs", type=int, default=25)
    t.add_argument("--bs", type=int, default=12)
    t.add_argument("--workers", type=int, default=4)
    c = sub.add_parser("cache")
    c.add_argument("--isic2019", default=None, help="ISIC 2019 folder (images + GT/metadata CSVs)")
    c.add_argument("--no-unet", action="store_true", help="use the Otsu fallback instead of the U-Net")
    c.add_argument("--force", action="store_true")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else cache(a)

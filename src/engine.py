"""Generic training / prediction engine used by every experiment in the report.

One function, `train_run`, trains any of the models (HAM10000 models, study models, Stage 1 gate):
  * architecture + settings of report Table 4.2 (AdamW, plateau or warm-up+cosine, early stopping)
  * class-imbalance strategy: none | sqrt | inverse | focal | sampler          (Table 4.3)
  * optional training-only skin-tone augmentation                              (Section 4.7.2)
  * any image set (full HAM images, U-Net crops, DermaCon-IN photos) via a `cache` column
  * best checkpoint by validation macro-F1; finished runs are skipped when re-run (resume)

`predict_probs` gives softmax outputs, optionally averaged over 3 views (original, h-flip, v-flip).
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from models import build_model  # noqa: E402
from preprocess import load_image_rgb, preprocess_path, save_image_rgb  # noqa: E402

WORKERS = int(os.environ.get("SKIN_WORKERS", config.NUM_WORKERS))
RUNS_DIR = config.RESULTS_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- utilities
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    return config.get_device()


def ensure_cached(df: pd.DataFrame, src_col: str = "path", cache_col: str = "cache",
                  preprocess=preprocess_path, workers: int | None = None) -> None:
    """Preprocess (hair removal, CLAHE, resize) every missing cache file, in parallel."""
    todo = [(s, c) for s, c in zip(df[src_col], df[cache_col]) if not Path(c).exists()]
    if not todo:
        return
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm
    n = workers or max(1, (os.cpu_count() or 2))
    print(f"preprocessing {len(todo)} images with {n} processes ...")
    with ProcessPoolExecutor(n) as ex:
        list(tqdm(ex.map(_cache_one, todo, chunksize=16), total=len(todo)))


def _cache_one(item):
    src, dst = item
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    save_image_rgb(preprocess_path(src), dst)


def smoke_subset(df: pd.DataFrame, label_col: str, per_class: int) -> pd.DataFrame:
    return df.sample(frac=1.0, random_state=0).groupby(label_col).head(per_class).reset_index(drop=True)


# --------------------------------------------------------------------------- skin-tone augmentation
def shades_of_grey(img: np.ndarray, power: int = 6) -> np.ndarray:
    x = img.astype(np.float32)
    illum = np.power(np.mean(np.power(x, power), axis=(0, 1)), 1.0 / power)
    illum = illum / (np.sqrt((illum ** 2).sum()) + 1e-6)
    return np.clip(x / (illum * np.sqrt(3.0) + 1e-6), 0, 255).astype(np.uint8)


def darker_skin_lab(img: np.ndarray, rng=random) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= rng.uniform(0.55, 0.95)        # lightness x 0.55-0.95
    lab[..., 2] += rng.uniform(0.0, 12.0)         # yellow-blue channel + 0..12
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


class SkinToneAugment:
    """Training-only: each of the three transforms is applied with probability 0.5 (report Fig. 4.3)."""

    def __init__(self):
        self.jitter = transforms.ColorJitter(brightness=0.25, contrast=0.25, hue=0.05)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            img = shades_of_grey(img)
        if random.random() < 0.5:
            img = darker_skin_lab(img)
        if random.random() < 0.5:
            from PIL import Image
            img = np.asarray(self.jitter(Image.fromarray(img)))
        return img


# --------------------------------------------------------------------------- data
NORM = [transforms.ToTensor(), transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)]


def train_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
    ] + NORM)


def eval_transform():
    return transforms.Compose([transforms.ToPILImage()] + NORM)


class CachedImageDataset(Dataset):
    """Reads preprocessed 224x224 images from df[cache_col]; returns (tensor, label)."""

    def __init__(self, df, label_col="label", cache_col="cache", train=False, skin_aug=False):
        self.paths = df[cache_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist() if label_col in df else [0] * len(df)
        self.tf = train_transform() if train else eval_transform()
        self.skin = SkinToneAugment() if (train and skin_aug) else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_image_rgb(self.paths[i])
        if img.shape[:2] != (config.IMG_SIZE, config.IMG_SIZE):
            img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
        if self.skin is not None:
            img = self.skin(img)
        return self.tf(img), self.labels[i]


# --------------------------------------------------------------------------- losses
def balanced_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    present = np.unique(labels)
    w = np.ones(n_classes, dtype=np.float32)
    w[present] = compute_class_weight("balanced", classes=present, y=labels)
    return w


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, y):
        logp = F.log_softmax(logits.float(), 1)
        logp_t = logp.gather(1, y[:, None])[:, 0]
        return (-(1 - logp_t.exp()) ** self.gamma * logp_t).mean()


def make_criterion(strategy: str, labels: np.ndarray, n_classes: int, smoothing: float, dev):
    if strategy == "focal":
        return FocalLoss(2.0)
    if strategy in ("none", "sampler"):
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
    w = balanced_weights(labels, n_classes)
    if strategy == "sqrt":
        w = np.sqrt(w)
    elif strategy != "inverse":
        raise ValueError(strategy)
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
    return nn.CrossEntropyLoss(weight=w, label_smoothing=smoothing)


# --------------------------------------------------------------------------- training
def run_done(run_name: str) -> bool:
    return (RUNS_DIR / run_name / "summary.json").exists() and (config.MODELS_DIR / f"{run_name}.pth").exists()


def train_run(run_name: str, arch_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, *,
              n_classes: int = config.NUM_CLASSES, class_names=None, label_col: str = "label",
              cache_col: str = "cache", seed: int = 42, imbalance: str = "sqrt", skin_aug: bool = False,
              settings: dict | None = None, force: bool = False) -> Path:
    """Train one model; returns the checkpoint path. Skips finished runs unless force=True."""
    ckpt_path = config.MODELS_DIR / f"{run_name}.pth"
    out_dir = RUNS_DIR / run_name
    if run_done(run_name) and not force:
        print(f"[skip] {run_name} already trained")
        return ckpt_path
    out_dir.mkdir(parents=True, exist_ok=True)
    s = dict(settings or config.TRAIN_SETTINGS.get(arch_name, config.CNN_SETTINGS))
    if config.SMOKE:
        train_df = smoke_subset(train_df, label_col, 8)
        val_df = smoke_subset(val_df, label_col, 4)
        s.update(epochs=1, patience=1, min_epochs=0)
    set_seed(seed)
    dev = device()
    use_amp = dev.type == "cuda"
    labels = train_df[label_col].to_numpy().astype(int)

    sampler = None
    if imbalance == "sampler":
        counts = np.bincount(labels, minlength=n_classes).astype(float)
        sw = 1.0 / counts[labels]
        sampler = WeightedRandomSampler(torch.tensor(sw, dtype=torch.double), num_samples=len(labels),
                                        replacement=True, generator=torch.Generator().manual_seed(seed))
    g = torch.Generator().manual_seed(seed)
    dl_tr = DataLoader(CachedImageDataset(train_df, label_col, cache_col, train=True, skin_aug=skin_aug),
                       batch_size=s["batch_size"], shuffle=sampler is None, sampler=sampler, generator=g,
                       num_workers=WORKERS, pin_memory=use_amp, drop_last=len(train_df) > s["batch_size"],
                       persistent_workers=WORKERS > 0)
    dl_va = DataLoader(CachedImageDataset(val_df, label_col, cache_col), batch_size=64,
                       num_workers=WORKERS, pin_memory=use_amp)

    model = build_model(arch_name, n_classes, pretrained=True, dropout=s["dropout"]).to(dev)
    crit = make_criterion(imbalance, labels, n_classes, s["label_smoothing"], dev)
    opt = torch.optim.AdamW(model.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    steps = s["epochs"] * len(dl_tr)
    if s["schedule"] == "cosine":
        warm = len(dl_tr)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda k: (k + 1) / warm if k < warm else 0.5 * (1 + math.cos(math.pi * (k - warm) / max(1, steps - warm))))
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.3, patience=2)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best, bad, hist, t0 = -1.0, 0, [], time.time()
    y_val = val_df[label_col].to_numpy().astype(int)
    for ep in range(1, s["epochs"] + 1):
        model.train()
        tl, n = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, enabled=use_amp):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if s["schedule"] == "cosine":
                sched.step()
            tl += loss.item() * len(y)
            n += len(y)
        pv = predict_loader(model, dl_va, tta=False)
        f1 = f1_score(y_val, pv.argmax(1), average="macro", labels=list(range(n_classes)), zero_division=0)
        acc = float((pv.argmax(1) == y_val).mean())
        if s["schedule"] != "cosine":
            sched.step(f1)
        hist.append({"epoch": ep, "train_loss": tl / max(n, 1), "val_acc": acc, "val_f1": f1,
                     "lr": opt.param_groups[0]["lr"], "minutes": (time.time() - t0) / 60})
        print(f"{run_name} ep {ep}: loss {tl / max(n, 1):.4f} val acc {acc:.4f} macro-F1 {f1:.4f}")
        if f1 > best:
            best, bad = f1, 0
            torch.save({"model_name": run_name, "arch": arch_name, "state_dict": model.state_dict(),
                        "class_names": class_names or config.CLASS_NAMES, "num_classes": n_classes,
                        "img_size": config.IMG_SIZE, "epoch": ep, "val_f1": f1, "val_acc": acc,
                        "seed": seed, "imbalance": imbalance, "skin_aug": skin_aug,
                        "cache_col": cache_col}, ckpt_path)
        else:
            bad += 1
            if bad >= s["patience"] and ep >= s["min_epochs"]:
                print(f"early stop at epoch {ep}")
                break
    pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)
    json.dump({"run": run_name, "arch": arch_name, "seed": seed, "imbalance": imbalance,
               "skin_aug": skin_aug, "epochs_run": len(hist), "best_val_f1": best,
               "train_minutes": round((time.time() - t0) / 60, 1), "device": str(dev),
               "n_train": len(train_df), "smoke": config.SMOKE},
              open(out_dir / "summary.json", "w"), indent=2)
    return ckpt_path


# --------------------------------------------------------------------------- prediction
@torch.no_grad()
def predict_loader(model, loader, tta: bool = True) -> np.ndarray:
    """Softmax probabilities; with tta the mean over original, horizontal and vertical flip."""
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for x, _ in loader:
        x = x.to(dev, non_blocking=True)
        views = [x, x.flip(3), x.flip(2)] if tta else [x]
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            p = sum(F.softmax(model(v).float(), 1) for v in views) / len(views)
        out.append(p.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1))


def predict_df(model, df: pd.DataFrame, cache_col: str = "cache", tta: bool = True,
               batch_size: int = 64) -> np.ndarray:
    dl = DataLoader(CachedImageDataset(df, label_col="__none__", cache_col=cache_col), batch_size=batch_size,
                    num_workers=WORKERS, pin_memory=torch.cuda.is_available())
    return predict_loader(model, dl, tta=tta)


def cached_predictions(key: str, model_or_ckpt, df: pd.DataFrame, cache_col="cache", tta=True,
                       force=False) -> np.ndarray:
    """Predictions saved in results/preds/<key>.npz (with image ids) and reused on later runs."""
    f = config.PREDS_DIR / f"{key}.npz"
    ids = df["image_id"].astype(str).to_numpy()
    if f.exists() and not force:
        z = np.load(f, allow_pickle=True)
        if len(z["ids"]) == len(ids) and (z["ids"] == ids).all():
            return z["probs"]
    model = model_or_ckpt
    if isinstance(model_or_ckpt, (str, Path)):
        from models import load_checkpoint
        model, _ = load_checkpoint(model_or_ckpt)
    p = predict_df(model, df, cache_col, tta)
    np.savez_compressed(f, probs=p, ids=ids)
    return p

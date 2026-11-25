"""v2 classifier training: strong ImageNet-21k backbones (timm) + metadata fusion.

What changed vs. the v1 src/train.py and why:
  * 384 px lesion crops instead of 224 px full images (fine texture/border detail)
  * ImageNet-21k / Noisy-Student backbones (EfficientNetV2-S, ConvNeXt-S, EfficientNet-B4)
  * optional ISIC 2019 extra training data (~3x more images)
  * metadata fusion: age, sex, anatomical site -> small MLP concatenated with image features
  * strong augmentation + mixup/cutmix, label smoothing, no class weights
    (class priors are re-balanced later on the validation set by ensemble.py)
  * AdamW, warm-up + cosine schedule, AMP, EMA of weights, 25 epochs
  * 8-view test-time augmentation; saves val/test probabilities for the ensemble

    python improved/train_cls.py --arch tf_efficientnetv2_s.in21k_ft_in1k --meta [--isic2019 DIR]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from common import CLASS_NAMES, META_DIM, V2_MODELS, V2_RESULTS, build_tables, encode_meta

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

RECOMMENDED = {   # arch -> (img size, batch size for a 16 GB GPU, lr)
    "tf_efficientnetv2_s.in21k_ft_in1k": (384, 24, 2e-4),
    "convnext_small.fb_in22k_ft_in1k_384": (384, 16, 1e-4),
    "tf_efficientnet_b4.ns_jft_in1k": (384, 16, 2e-4),
    "swin_base_patch4_window12_384.ms_in22k_ft_in1k": (384, 12, 5e-5),
}


# --------------------------------------------------------------------------- data
class LesionDS(Dataset):
    def __init__(self, df: pd.DataFrame, tf, use_meta: bool):
        self.df, self.tf, self.use_meta = df.reset_index(drop=True), tf, use_meta
        self.meta = np.stack([encode_meta(r.age, r.sex, r.site_grp) for r in self.df.itertuples()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = Image.open(self.df.path[i]).convert("RGB")
        meta = torch.from_numpy(self.meta[i]) if self.use_meta else torch.zeros(META_DIM)
        return self.tf(img), meta, int(self.df.label[i])


def train_tf(size):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation(180)], p=0.7),
        T.RandomAffine(0, shear=10),
        T.ColorJitter(0.2, 0.2, 0.15, 0.03),
        T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.2),
        T.ToTensor(), T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])


def eval_tf(size):
    return T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])


# --------------------------------------------------------------------------- model
class LesionNet(nn.Module):
    """timm backbone (+ optional metadata MLP) -> 7-class head."""

    def __init__(self, arch: str, use_meta: bool, pretrained: bool = True, drop: float = 0.3):
        super().__init__()
        import timm
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, drop_path_rate=0.2)
        nf = self.backbone.num_features
        self.use_meta = use_meta
        if use_meta:
            self.meta = nn.Sequential(nn.Linear(META_DIM, 64), nn.BatchNorm1d(64), nn.SiLU(), nn.Dropout(0.2),
                                      nn.Linear(64, 64), nn.BatchNorm1d(64), nn.SiLU())
            nf += 64
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(nf, len(CLASS_NAMES)))

    def forward(self, x, meta=None):
        f = self.backbone(x)
        if self.use_meta:
            f = torch.cat([f, self.meta(meta)], 1)
        return self.head(f)


def gradcam_target_layer(model: LesionNet) -> nn.Module:
    """Last conv stage, for the existing src/gradcam.py GradCAM class."""
    b = model.backbone
    for name in ("stages", "layers"):            # ConvNeXt / Swin
        if hasattr(b, name):
            return getattr(b, name)[-1]
    if hasattr(b, "conv_head"):                  # EfficientNet family
        return b.conv_head
    return list(b.children())[-2]


class EMA:
    def __init__(self, model, decay=0.999):
        self.m = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.m.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
            else:
                e.copy_(p)


# --------------------------------------------------------------------------- mixup / cutmix
def mix_batch(x, meta, y, n_cls, alpha_mix=0.4, alpha_cut=1.0, p=0.6, smoothing=0.1):
    y1 = F.one_hot(y, n_cls).float() * (1 - smoothing) + smoothing / n_cls
    if random.random() > p:
        return x, meta, y1
    idx = torch.randperm(x.size(0), device=x.device)
    if random.random() < 0.5:                                   # mixup
        lam = np.random.beta(alpha_mix, alpha_mix)
        x = lam * x + (1 - lam) * x[idx]
    else:                                                       # cutmix
        lam = np.random.beta(alpha_cut, alpha_cut)
        H, W = x.shape[2:]
        rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
        cy, cx = random.randint(0, H - 1), random.randint(0, W - 1)
        y0, y1_, x0, x1 = max(cy - rh // 2, 0), min(cy + rh // 2, H), max(cx - rw // 2, 0), min(cx + rw // 2, W)
        x = x.clone(); x[:, :, y0:y1_, x0:x1] = x[idx, :, y0:y1_, x0:x1]
        lam = 1 - (y1_ - y0) * (x1 - x0) / (H * W)
    meta = lam * meta + (1 - lam) * meta[idx]
    return x, meta, lam * y1 + (1 - lam) * y1[idx]


# --------------------------------------------------------------------------- eval / TTA
TTA_VIEWS = [(False, 0), (True, 0), (False, 1), (True, 1), (False, 2), (True, 2), (False, 3), (True, 3)]


@torch.no_grad()
def predict(model, loader, dev, tta: bool):
    model.eval()
    out = []
    for x, meta, _ in loader:
        x, meta = x.to(dev), meta.to(dev)
        views = TTA_VIEWS if tta else [(False, 0)]
        p = 0
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            for flip, k in views:
                v = torch.rot90(x.flip(3) if flip else x, k, (2, 3))
                p = p + F.softmax(model(v, meta).float(), 1)
        out.append((p / len(views)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="tf_efficientnetv2_s.in21k_ft_in1k")
    ap.add_argument("--meta", action="store_true", help="fuse age/sex/site metadata")
    ap.add_argument("--isic2019", default=None, help="add ISIC 2019 extra training data")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None, help="run name (default: arch[+meta][+isic])")
    a = ap.parse_args()

    size, bs, lr = RECOMMENDED.get(a.arch, (384, 16, 1e-4))
    size, bs, lr = a.size or size, a.bs or bs, a.lr or lr
    tag = a.tag or a.arch.split(".")[0] + ("_meta" if a.meta else "") + ("_isic" if a.isic2019 else "")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = V2_RESULTS / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = build_tables(a.isic2019)
    print({k: len(v) for k, v in tables.items()}, "| train class counts:",
          tables["train"].dx.value_counts().to_dict())
    dl_tr = DataLoader(LesionDS(tables["train"], train_tf(size), a.meta), batch_size=bs, shuffle=True,
                       num_workers=a.workers, drop_last=True, pin_memory=True, persistent_workers=a.workers > 0)
    dl_va = DataLoader(LesionDS(tables["val"], eval_tf(size), a.meta), batch_size=bs * 2, num_workers=a.workers)
    dl_te = DataLoader(LesionDS(tables["test"], eval_tf(size), a.meta), batch_size=bs * 2, num_workers=a.workers)

    model = LesionNet(a.arch, a.meta).to(dev)
    ema = EMA(model, decay=0.9993)
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": lr},
                             {"params": head_params, "lr": lr * 10}], weight_decay=0.02)
    total = a.epochs * len(dl_tr)
    warm = len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    y_val = tables["val"].label.to_numpy()

    best, history, t0 = -1, [], time.time()
    ckpt_path = V2_MODELS / f"{tag}.pth"
    for ep in range(a.epochs):
        model.train(); run_loss = 0
        for x, meta, y in tqdm(dl_tr, desc=f"{tag} ep{ep + 1}/{a.epochs}", leave=False):
            x, meta, y = x.to(dev, non_blocking=True), meta.to(dev), y.to(dev)
            if a.meta and random.random() < 0.1:
                meta = torch.zeros_like(meta)            # metadata dropout: robust when unknown
            x, meta, ys = mix_batch(x, meta, y, len(CLASS_NAMES))
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = torch.sum(-ys * F.log_softmax(model(x, meta).float(), 1), 1).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step(); ema.update(model)
            run_loss += loss.item()
        pv = predict(ema.m, dl_va, dev, tta=False)
        acc = (pv.argmax(1) == y_val).mean()
        bacc = np.mean([(pv.argmax(1)[y_val == c] == c).mean() for c in np.unique(y_val)])
        history.append({"epoch": ep + 1, "loss": run_loss / len(dl_tr), "val_acc": acc, "val_bal_acc": bacc})
        print(f"ep {ep + 1}: loss {run_loss / len(dl_tr):.4f}  val acc {acc:.4f}  bal acc {bacc:.4f}")
        if acc + 0.25 * bacc > best:          # mostly accuracy, but don't sacrifice minority classes
            best = acc + 0.25 * bacc
            torch.save({"arch": a.arch, "meta": a.meta, "size": size, "class_names": CLASS_NAMES,
                        "state_dict": ema.m.state_dict(), "epoch": ep + 1}, ckpt_path)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    for split, dl in (("val", dl_va), ("test", dl_te)):
        p = predict(model, dl, dev, tta=True)
        np.save(out_dir / f"{split}_probs.npy", p)
        df = tables[split][["image_id", "lesion_id", "dx"]].copy()
        for i, c in enumerate(CLASS_NAMES):
            df[f"p_{c}"] = p[:, i]
        df.to_csv(out_dir / f"{split}_predictions.csv", index=False)
        print(f"{split} accuracy with TTA: {(p.argmax(1) == tables[split].label.to_numpy()).mean():.4f}")
    json.dump({"tag": tag, "arch": a.arch, "meta": a.meta, "isic2019": bool(a.isic2019), "size": size,
               "best_epoch": ck["epoch"], "minutes": round((time.time() - t0) / 60, 1)},
              open(out_dir / "run.json", "w"), indent=2)


if __name__ == "__main__":
    main()

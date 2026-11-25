"""Supporting studies on HAM10000 (report Sections 4.7 and 6.5-6.9).
All use EfficientNet-B0 + the HAM10000 recipe, seeds 0/1/2, single-view evaluation.

  imbalance_study()      Table 6.10, Fig 6.7   (5 strategies x 3 seeds)
  skin_tone_study()      Table 6.11, Fig 4.3   (selected setup + skin-tone augmentation)
  segmentation_study()   Table 6.12, Fig 4.4   (U-Net, EfficientNet-B0 encoder, BCE+Dice; Otsu baseline)
  crop_study()           Table 6.13            (selected setup trained on predicted-mask crops)
  robustness_study()     Table 6.14, Fig 6.8   (brightness / blur / JPEG, severities 1-5)
  explanation_study()    Table 6.15            (Grad-CAM deletion / insertion, 200 test images)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import engine  # noqa: E402
import ham  # noqa: E402
import stats  # noqa: E402
from models import get_target_layer, load_checkpoint  # noqa: E402
from preprocess import load_image_rgb, preprocess_image, save_image_rgb  # noqa: E402

OUT = config.STUDY_DIR / "studies"
OUT.mkdir(parents=True, exist_ok=True)
ARCH = config.STUDY_ARCH
NAMES = {"none": "a) None", "sqrt": "b) sqrt balanced weights (default)", "inverse": "c) Inverse frequency",
         "focal": "d) Focal loss", "sampler": "e) Weighted sampler"}


def _run(prefix, seed):
    return f"study_{prefix}_s{seed}"


def _preds(run, df, split, cache_col="cache"):
    return engine.cached_predictions(f"{run}_{split}", config.MODELS_DIR / f"{run}.pth", df, cache_col, tta=False)


# =========================================================================== class imbalance
def imbalance_study():
    s = ham.splits()
    for strat in config.IMBALANCE_STRATEGIES:
        for seed in config.SEEDS:
            engine.train_run(_run(f"imb_{strat}", seed), ARCH, s["train"], s["val"], seed=seed, imbalance=strat)
    yv, yt, gt = s["val"].label.to_numpy(), s["test"].label.to_numpy(), s["test"].lesion_id.to_numpy()
    rows, val = [], {}
    for strat in config.IMBALANCE_STRATEGIES:
        pv = [_preds(_run(f"imb_{strat}", sd), s["val"], "val") for sd in config.SEEDS]
        pt = [_preds(_run(f"imb_{strat}", sd), s["test"], "test") for sd in config.SEEDS]
        mv = [stats.metrics(yv, p) for p in pv]
        val[strat] = (np.mean([m["macro_f1"] for m in mv]), np.mean([m["melanoma_recall"] for m in mv]))
        ts = stats.seed_summary(yt, pt, gt)
        rows.append({"strategy": strat, "label": NAMES[strat], "val_macro_f1": val[strat][0],
                     "val_mel_recall": val[strat][1], **{f"test_{k}": v for k, v in ts.items()}})
    # rule fixed in advance: best mean val macro-F1; within 0.005 -> higher val melanoma recall
    best_f1 = max(v[0] for v in val.values())
    close = [k for k, v in val.items() if best_f1 - v[0] <= 0.005]
    selected = max(close, key=lambda k: (val[k][1], val[k][0]))
    json.dump({"selected": selected, "val": {k: list(v) for k, v in val.items()}},
              open(OUT / "imbalance_selected.json", "w"), indent=2)
    json.dump(rows, open(OUT / "table_6_10_imbalance.json", "w"), indent=2, default=float)
    flat = [{"strategy": r["label"] + (" (selected)" if r["strategy"] == selected else ""),
             "val_macro_f1": r["val_macro_f1"], "val_mel_recall": r["val_mel_recall"],
             "test_macro_f1": stats.fmt_ms(r["test_macro_f1"]), "test_mel_recall": stats.fmt_ms(r["test_melanoma_recall"]),
             "test_mal_sens": stats.fmt_ms(r["test_malignant_sensitivity"]),
             "test_accuracy": f"{r['test_accuracy']['mean']:.3f} ± {r['test_accuracy']['std']:.3f}"} for r in rows]
    pd.DataFrame(flat).to_csv(OUT / "table_6_10_imbalance.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, m in zip(axes, ["macro_f1", "melanoma_recall", "malignant_sensitivity"]):
        for i, r in enumerate(rows):
            d = r[f"test_{m}"]
            ax.errorbar(i, d["mean"], yerr=[[d["mean"] - d["ci"][0]], [d["ci"][1] - d["mean"]]], fmt="o", capsize=4,
                        color="crimson" if r["strategy"] == selected else "steelblue")
        ax.set_xticks(range(len(rows))); ax.set_xticklabels([r["strategy"] for r in rows])
        ax.set_title(m.replace("_", " ")); ax.grid(alpha=.3)
    fig.suptitle("Class-imbalance strategies, HAM10000 test (mean, 95% CI; red = selected on validation)")
    fig.tight_layout(); fig.savefig(OUT / "fig_6_7_imbalance.png", dpi=150); plt.close(fig)
    print(pd.DataFrame(flat).to_string(index=False), "\nselected:", selected)
    return selected


def selected_setup() -> str:
    return json.load(open(OUT / "imbalance_selected.json"))["selected"]


def baseline_runs():
    return [_run(f"imb_{selected_setup()}", sd) for sd in config.SEEDS]


# =========================================================================== skin tone
def skin_tone_study():
    s = ham.splits()
    sel = selected_setup()
    for seed in config.SEEDS:
        engine.train_run(_run("skin", seed), ARCH, s["train"], s["val"], seed=seed, imbalance=sel, skin_aug=True)
    yt, gt = s["test"].label.to_numpy(), s["test"].lesion_id.to_numpy()
    base = [_preds(r, s["test"], "test") for r in baseline_runs()]
    aug = [_preds(_run("skin", sd), s["test"], "test") for sd in config.SEEDS]
    _paired_table(yt, gt, base, aug, "Baseline (selected setup)", "Augmented", OUT / "table_6_11_skin_tone")

    img = load_image_rgb(s["train"].cache.iloc[0])
    random.seed(1)
    views = [("original", img), ("Shades of Grey", engine.shades_of_grey(img)),
             ("darker skin (LAB)", engine.darker_skin_lab(img)),
             ("brightness/contrast/hue", np.asarray(engine.SkinToneAugment().jitter(
                 __import__("PIL.Image", fromlist=["Image"]).fromarray(img))))]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    for a, (t, v) in zip(axes, views):
        a.imshow(v); a.set_title(t, fontsize=9); a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "fig_4_3_skin_tone_aug.png", dpi=150); plt.close(fig)


def _paired_table(y, groups, probs_a, probs_b, name_a, name_b, out_stem):
    sa = stats.seed_summary(y, probs_a, groups)
    sb = stats.seed_summary(y, probs_b, groups)
    dd = stats.paired_diff(y, probs_a, probs_b, groups)
    rows = [{"metric": m.replace("_", " "), name_a: stats.fmt_ms(sa[m]), name_b: stats.fmt_ms(sb[m]),
             "difference [95% paired CI]": f"{dd[m]['diff']:+.4f} [{dd[m]['ci'][0]:+.4f}, {dd[m]['ci'][1]:+.4f}]"}
            for m in stats.REPORT_METRICS]
    pd.DataFrame(rows).to_csv(f"{out_stem}.csv", index=False)
    json.dump({"a": sa, "b": sb, "diff": dd}, open(f"{out_stem}.json", "w"), indent=2, default=float)
    print(pd.DataFrame(rows).to_string(index=False))


# =========================================================================== segmentation
class UNetEffB0(nn.Module):
    """U-Net decoder on an ImageNet EfficientNet-B0 encoder (torchvision)."""
    SKIPS = (1, 2, 3, 5)          # feature indices: 16@/2, 24@/4, 40@/8, 112@/16 ; bottleneck 1280@/32

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision import models
        self.enc = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None).features
        ch = [16, 24, 40, 112]

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
        self.d4 = block(1280 + ch[3], 256)
        self.d3 = block(256 + ch[2], 128)
        self.d2 = block(128 + ch[1], 64)
        self.d1 = block(64 + ch[0], 32)
        self.d0 = block(32, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        feats = {}
        h = x
        for i, layer in enumerate(self.enc):
            h = layer(h)
            if i in self.SKIPS:
                feats[i] = h
        for dec, i in ((self.d4, 5), (self.d3, 3), (self.d2, 2), (self.d1, 1)):
            h = F.interpolate(h, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            h = dec(torch.cat([h, feats[i]], 1))
        h = F.interpolate(h, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.d0(h))


def find_masks() -> dict:
    """image_id -> mask path, searched in data/ham_masks and /kaggle/input."""
    roots = [config.MASK_DIR] + ([Path("/kaggle/input")] if Path("/kaggle/input").exists() else [])
    out = {}
    for r in roots:
        for p in r.rglob("ISIC_*_segmentation.png"):
            out.setdefault(p.stem[:12], p)
    return out


def _seg_input(img):
    x = cv2.resize(img, (config.SEG_SIZE, config.SEG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x = (x - np.array(config.IMAGENET_MEAN, np.float32)) / np.array(config.IMAGENET_STD, np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1))


class SegDS(Dataset):
    def __init__(self, df, masks, train):
        self.df, self.masks, self.train = df.reset_index(drop=True), masks, train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = load_image_rgb(r.path)
        m = (cv2.imread(str(self.masks[r.image_id]), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        m = cv2.resize(m, (config.SEG_SIZE, config.SEG_SIZE), interpolation=cv2.INTER_NEAREST)
        x = _seg_input(img)
        y = torch.from_numpy(m[None].astype(np.float32))
        if self.train:
            if random.random() < .5: x, y = x.flip(2), y.flip(2)
            if random.random() < .5: x, y = x.flip(1), y.flip(1)
            k = random.randint(0, 3); x, y = torch.rot90(x, k, (1, 2)), torch.rot90(y, k, (1, 2))
        return x, y


def _dice_iou(pred, gt):
    inter = (pred & gt).sum(); s = pred.sum() + gt.sum(); u = (pred | gt).sum()
    return (2 * inter / s if s else 1.0), (inter / u if u else 1.0)


def train_unet(seed, s, masks, epochs=20):
    name = f"unet_s{seed}"
    path = config.MODELS_DIR / f"{name}.pth"
    if path.exists() and (engine.RUNS_DIR / name / "summary.json").exists():
        return path
    engine.set_seed(seed)
    dev = engine.device()
    tr = s["train"][s["train"].image_id.isin(masks)]
    va = s["val"][s["val"].image_id.isin(masks)]
    if config.SMOKE:
        tr, va, epochs = tr.head(16), va.head(8), 1
    dl = DataLoader(SegDS(tr, masks, True), batch_size=16, shuffle=True, num_workers=engine.WORKERS, drop_last=True)
    dv = DataLoader(SegDS(va, masks, False), batch_size=32, num_workers=engine.WORKERS)
    model = UNetEffB0().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best = -1
    for ep in range(epochs):
        model.train()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev.type, enabled=use_amp):
                o = model(x)
            o = o.float()
            p = torch.sigmoid(o)
            dice = 1 - (2 * (p * y).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
            loss = F.binary_cross_entropy_with_logits(o, y) + dice.mean()
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        model.eval(); ds = []
        with torch.no_grad():
            for x, y in dv:
                pr = (model(x.to(dev)).float() > 0).cpu().numpy().astype(bool)
                for a, b in zip(pr, y.numpy().astype(bool)):
                    ds.append(_dice_iou(a, b)[0])
        d = float(np.mean(ds))
        print(f"{name} ep {ep + 1}: val Dice {d:.4f}")
        if d > best:
            best = d
            torch.save({"state_dict": model.state_dict(), "val_dice": d, "seed": seed}, path)
    (engine.RUNS_DIR / name).mkdir(parents=True, exist_ok=True)
    json.dump({"run": name, "best_val_dice": best}, open(engine.RUNS_DIR / name / "summary.json", "w"))
    return path


class Segmenter:
    def __init__(self, path=None):
        path = path or best_unet_path()
        self.dev = engine.device()
        self.model = UNetEffB0(pretrained=False)
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
        self.model.to(self.dev).eval()

    @torch.no_grad()
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Binary mask at the image's own resolution."""
        p = torch.sigmoid(self.model(_seg_input(img)[None].to(self.dev)))[0, 0].cpu().numpy()
        return cv2.resize((p > 0.5).astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)


def best_unet_path():
    best = max(config.SEEDS, key=lambda sd: torch.load(config.MODELS_DIR / f"unet_s{sd}.pth", map_location="cpu",
                                                       weights_only=False)["val_dice"])
    return config.MODELS_DIR / f"unet_s{best}.pth"


def otsu_mask(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (11, 11), 0)
    _, m = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return m.astype(np.uint8)


def largest_component(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(np.uint8)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8)


def crop_with_mask(img, mask, margin=config.CROP_MARGIN, min_area=config.CROP_MIN_AREA):
    """Square crop around the largest predicted region + margin; full image if region < 1%."""
    m = largest_component(mask)
    if m.mean() < min_area:
        return img, False
    ys, xs = np.nonzero(m)
    h, w = m.shape
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    side = int(min(max(ys.max() - ys.min(), xs.max() - xs.min()) * (1 + 2 * margin), min(h, w)))
    side = max(side, 8)
    top = int(np.clip(cy - side / 2, 0, h - side)); left = int(np.clip(cx - side / 2, 0, w - side))
    return img[top:top + side, left:left + side], True


def segmentation_study():
    s = ham.splits()
    masks = find_masks()
    n_ham = sum(1 for k in masks if k in set(pd.concat(s.values()).image_id))
    print(f"HAM10000 masks found: {n_ham}")
    if n_ham < (50 if config.SMOKE else 5000):
        raise FileNotFoundError("HAM10000 lesion masks not found: add them under data/ham_masks or as a Kaggle input")
    for sd in config.SEEDS:
        train_unet(sd, s, masks)
    te = s["test"][s["test"].image_id.isin(masks)].reset_index(drop=True)
    g = te.lesion_id.to_numpy()
    per_seed, imgs, gts = [], [load_image_rgb(p) for p in te.path], []
    for p in te.image_id:
        gts.append(cv2.imread(str(masks[p]), cv2.IMREAD_GRAYSCALE) > 127)
    for sd in config.SEEDS:
        seg = Segmenter(config.MODELS_DIR / f"unet_s{sd}.pth")
        per_seed.append(np.array([_dice_iou(seg(i).astype(bool), gt) for i, gt in zip(imgs, gts)]))
    otsu = np.array([_dice_iou(otsu_mask(i).astype(bool), gt) for i, gt in zip(imgs, gts)])

    def seedmean_ci(col):
        return stats.bootstrap_ci(lambda ix: np.mean([ps[ix, col].mean() for ps in per_seed]), g)
    rows = [{"method": "U-Net (3 seeds)",
             "dice": f"{np.mean([ps[:, 0].mean() for ps in per_seed]):.3f} ± {np.std([ps[:, 0].mean() for ps in per_seed], ddof=1):.3f}",
             "dice_ci": seedmean_ci(0),
             "iou": f"{np.mean([ps[:, 1].mean() for ps in per_seed]):.3f} ± {np.std([ps[:, 1].mean() for ps in per_seed], ddof=1):.3f}",
             "iou_ci": seedmean_ci(1)},
            {"method": "Otsu baseline (no training)", "dice": f"{otsu[:, 0].mean():.3f}",
             "dice_ci": stats.bootstrap_ci(lambda ix: otsu[ix, 0].mean(), g),
             "iou": f"{otsu[:, 1].mean():.3f}", "iou_ci": stats.bootstrap_ci(lambda ix: otsu[ix, 1].mean(), g)}]
    pd.DataFrame(rows).to_csv(OUT / "table_6_12_segmentation.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    seg = Segmenter()
    fig, axes = plt.subplots(3, 5, figsize=(13, 8))
    for i in range(min(5, len(te))):
        m = seg(imgs[i]); c, _ = crop_with_mask(imgs[i], m)
        ov = imgs[i].copy()
        cnts, _ = cv2.findContours(largest_component(m), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (0, 255, 0), 3)
        for row, (im, t) in enumerate(((imgs[i], "test image"), (ov, "predicted mask"), (c, "crop"))):
            axes[row, i].imshow(im); axes[row, i].set_title(t, fontsize=8); axes[row, i].axis("off")
    fig.tight_layout(); fig.savefig(OUT / "fig_4_4_masks_crops.png", dpi=150); plt.close(fig)


def make_crop_cache(df, src_col="path", cache_col="crop_cache", seg=None):
    seg = seg or Segmenter()
    used = 0
    for src, dst in zip(df[src_col], df[cache_col]):
        if Path(dst).exists():
            continue
        img = load_image_rgb(src)
        crop, ok = crop_with_mask(img, seg(img))
        used += ok
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        save_image_rgb(preprocess_image(crop), dst)
    return used


def crop_study():
    s = ham.splits()
    seg = Segmenter()
    for df in s.values():
        make_crop_cache(df, seg=seg)
    sel = selected_setup()
    for seed in config.SEEDS:
        engine.train_run(_run("crop", seed), ARCH, s["train"], s["val"], seed=seed, imbalance=sel,
                         cache_col="crop_cache")
    yt, gt = s["test"].label.to_numpy(), s["test"].lesion_id.to_numpy()
    base = [_preds(r, s["test"], "test") for r in baseline_runs()]
    crop = [_preds(_run("crop", sd), s["test"], "test", "crop_cache") for sd in config.SEEDS]
    _paired_table(yt, gt, base, crop, "Full image", "Crop", OUT / "table_6_13_crop")


# =========================================================================== robustness
def corrupt(img, kind, level):
    if kind == "brightness":
        return np.clip(img.astype(np.float32) * level, 0, 255).astype(np.uint8)
    if kind == "gaussian_blur":
        return cv2.GaussianBlur(img, (0, 0), level)
    if kind == "jpeg_quality":
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(level)])
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    raise ValueError(kind)


def _corrupt_one(item):
    src, dst, kind, level = item
    img = load_image_rgb(src)
    if kind != "clean":
        img = corrupt(img, kind, level)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    save_image_rgb(preprocess_image(img), dst)


def robustness_study():
    from concurrent.futures import ProcessPoolExecutor
    s = ham.splits()
    te = s["test"].copy()
    yt, gt = te.label.to_numpy(), te.lesion_id.to_numpy()
    root = config.DATA_DIR / "corrupted"
    variants = [("clean", 0, None)] + [(k, i + 1, lv) for k, lvs in config.CORRUPTIONS.items() for i, lv in enumerate(lvs)]
    jobs = []
    for kind, sev, lv in variants:
        for src, iid in zip(te.path, te.image_id):
            dst = root / f"{kind}_{sev}" / f"{iid}.jpg"
            if not dst.exists():
                jobs.append((src, str(dst), kind, lv))
    if jobs:
        with ProcessPoolExecutor() as ex:
            list(ex.map(_corrupt_one, jobs, chunksize=32))
    wmap, _ = ham.ensemble_weights()
    models = {m: (config.MODELS_DIR / f"{m}.pth", True) for m in config.HAM_MODELS}
    models.update({f"study_s{sd}": (config.MODELS_DIR / f"{r}.pth", False) for sd, r in zip(config.SEEDS, baseline_runs())})
    rows = []
    for kind, sev, lv in variants:
        te["v"] = te.image_id.map(lambda i: str(root / f"{kind}_{sev}" / f"{i}.jpg"))
        P = {m: engine.cached_predictions(f"robust_{m}_{kind}_{sev}", path, te, "v", tta=tta)
             for m, (path, tta) in models.items()}
        P["ensemble"] = sum(w * P[m] for m, w in wmap.items())
        study = [P[f"study_s{sd}"] for sd in config.SEEDS]
        for m, p in list(P.items()) + [("study_setup", None)]:
            if m.startswith("study_s"):
                continue
            if m == "study_setup":
                ss = stats.seed_summary(yt, study, gt, ["macro_f1", "malignant_sensitivity", "accuracy"])
                r = {k: v["mean"] for k, v in ss.items()}
                ci = {k: v["ci"] for k, v in ss.items()}
            else:
                r = stats.metrics(yt, p)
                ci = {k: stats.bootstrap_ci(lambda ix, k=k: stats.metrics(yt[ix], p[ix])[k], gt, n=min(200, config.N_BOOTSTRAP))
                      for k in ("macro_f1", "malignant_sensitivity")}
            rows.append({"model": m, "corruption": kind, "severity": sev, "level": lv,
                         "macro_f1": r["macro_f1"], "malignant_sensitivity": r["malignant_sensitivity"],
                         "accuracy": r["accuracy"], "macro_f1_ci": ci["macro_f1"], "mal_sens_ci": ci["malignant_sensitivity"]})
    df = pd.DataFrame(rows)
    df.to_json(OUT / "robustness_all.json", orient="records", indent=1)
    clean = df[df.corruption == "clean"].set_index("model").macro_f1
    t = {"clean (severity 0)": clean}
    for kind in config.CORRUPTIONS:
        t[kind] = df[(df.corruption == kind) & (df.severity == 5)].set_index("model").macro_f1
    pd.DataFrame(t).T.to_csv(OUT / "table_6_14_robustness.csv")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, kind in enumerate(config.CORRUPTIONS):
        for i, met in enumerate(("macro_f1", "malignant_sensitivity")):
            ax = axes[i, j]
            for m in ["ensemble", "study_setup"] + config.HAM_MODELS:
                d = pd.concat([df[(df.model == m) & (df.corruption == "clean")],
                               df[(df.model == m) & (df.corruption == kind)]]).sort_values("severity")
                ax.plot(d.severity, d[met], "o-", label=m, lw=2.5 if m in ("ensemble", "study_setup") else 1)
                if m in ("ensemble", "study_setup"):
                    cic = "macro_f1_ci" if met == "macro_f1" else "mal_sens_ci"
                    ax.fill_between(d.severity, [c[0] for c in d[cic]], [c[1] for c in d[cic]], alpha=.15)
            ax.set_title(f"{kind} - {met.replace('_', ' ')}"); ax.set_xlabel("severity"); ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig_6_8_robustness.png", dpi=150); plt.close(fig)
    print(pd.DataFrame(t).T.round(3).to_string())


# =========================================================================== explanations
@torch.no_grad()
def _curve(model, x, blurred, order, cls, steps, mode):
    n = order.numel()
    batch = []
    for k in range(steps + 1):
        idx = order[: int(round(n * k / steps))]
        m = torch.zeros(n, device=x.device); m[idx] = 1
        m = m.view(1, 1, *x.shape[-2:])
        batch.append(x * (1 - m) + blurred * m if mode == "deletion" else blurred * (1 - m) + x * m)
    p = torch.softmax(model(torch.cat(batch)), 1)[:, cls].float().cpu().numpy()
    return float(np.trapezoid(p, dx=1 / steps)) if hasattr(np, "trapezoid") else float(np.trapz(p, dx=1 / steps))


def explanation_study():
    from gradcam import GradCAM
    s = ham.splits()
    te = s["test"].sample(n=min(config.EXPLAIN_IMAGES, len(s["test"])), random_state=42).reset_index(drop=True)
    tf = engine.eval_transform()
    runs = {m: config.MODELS_DIR / f"{m}.pth" for m in
            ("efficientnet_b0", "resnet50", "densenet121", "convnext_tiny", "swin_t")}
    runs["study_setup_seed0"] = config.MODELS_DIR / f"{baseline_runs()[0]}.pth"
    rows = []
    rng = np.random.default_rng(0)
    for name, path in runs.items():
        model, ck = load_checkpoint(path)
        dev = next(model.parameters()).device
        gc = GradCAM(model, get_target_layer(model, ck.get("arch", name)))
        res = []
        for r in te.itertuples():
            img = load_image_rgb(r.cache)
            x = tf(img)[None].to(dev)
            blurred = tf(cv2.GaussianBlur(img, (0, 0), 10))[None].to(dev)
            cam, cls, _ = gc(x)
            order = torch.from_numpy(np.argsort(-cam.ravel())).to(dev)
            rnd = torch.from_numpy(rng.permutation(cam.size)).to(dev)
            res.append([_curve(model, x, blurred, order, cls, config.EXPLAIN_STEPS, "deletion"),
                        _curve(model, x, blurred, order, cls, config.EXPLAIN_STEPS, "insertion"),
                        _curve(model, x, blurred, rnd, cls, config.EXPLAIN_STEPS, "deletion"),
                        _curve(model, x, blurred, rnd, cls, config.EXPLAIN_STEPS, "insertion")])
        gc.remove()
        res = np.array(res)
        g = te.lesion_id.to_numpy()
        rows.append({"model": name,
                     "deletion": res[:, 0].mean(), "deletion_ci": stats.bootstrap_ci(lambda ix: res[ix, 0].mean(), g),
                     "insertion": res[:, 1].mean(), "insertion_ci": stats.bootstrap_ci(lambda ix: res[ix, 1].mean(), g),
                     "random_deletion": res[:, 2].mean(), "random_insertion": res[:, 3].mean()})
        print(rows[-1])
        del model
    pd.DataFrame(rows).to_csv(OUT / "table_6_15_explanations.csv", index=False)

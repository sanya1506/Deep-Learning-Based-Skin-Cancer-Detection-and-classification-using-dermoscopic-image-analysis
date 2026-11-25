"""Part A of the report: six HAM10000 models, 3-view TTA, ensemble selection on validation,
Tables 6.1 / 6.2 / 6.16 and Figures 6.1-6.4 / 6.9.

    from ham import *; prepare(); train_all(); predict_all(); build_ensemble(); figures()
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc as sk_auc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import engine  # noqa: E402
import stats  # noqa: E402
from dataset import make_splits  # noqa: E402
from models import DISPLAY_NAMES, count_parameters, get_target_layer, load_checkpoint  # noqa: E402

OUT = config.STUDY_DIR / "ham"
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = config.CLASS_NAMES


# --------------------------------------------------------------------------- data
def splits(with_cache=True) -> dict:
    s = make_splits()
    for k, df in s.items():
        df = df.copy()
        df["cache"] = df["image_id"].map(lambda i: str(config.PROCESSED_DIR / f"{i}.jpg"))
        df["crop_cache"] = df["image_id"].map(lambda i: str(config.PROCESSED_CROP_DIR / f"{i}.jpg"))
        if config.SMOKE:
            df = engine.smoke_subset(df, "label", 12 if k == "train" else 6)
        s[k] = df.reset_index(drop=True)
    return s


def prepare():
    s = splits()
    for df in s.values():
        engine.ensure_cached(df, "path", "cache")
    t = pd.DataFrame({k: v["dx"].value_counts() for k, v in s.items()}).reindex(CLASSES).fillna(0).astype(int)
    t["total"] = t.sum(1)
    t.loc["images"] = t.sum()
    t.loc["lesions"] = [s[k].lesion_id.nunique() for k in ("train", "val", "test")] + [
        sum(s[k].lesion_id.nunique() for k in ("train", "val", "test"))]
    t.loc["malignant"] = [s[k].dx.isin(config.MALIGNANT_CLASSES).sum() for k in ("train", "val", "test")] + [
        sum(s[k].dx.isin(config.MALIGNANT_CLASSES).sum() for k in ("train", "val", "test"))]
    t.to_csv(OUT / "table_3_1_split.csv")
    print(t)
    return s


# --------------------------------------------------------------------------- training
def train_all(models=None):
    s = splits()
    for m in models or config.HAM_MODELS:
        engine.train_run(m, m, s["train"], s["val"], seed=config.SEED, imbalance="sqrt",
                         settings=config.TRAIN_SETTINGS[m])


def ckpt(m):
    return config.MODELS_DIR / f"{m}.pth"


def predict_all(models=None):
    s = splits()
    out = {}
    for m in models or config.HAM_MODELS:
        out[m] = {k: engine.cached_predictions(f"ham_{m}_{k}", ckpt(m), s[k]) for k in ("val", "test")}
    return out


def load_preds(models=None):
    s = splits()
    return {m: {k: engine.cached_predictions(f"ham_{m}_{k}", ckpt(m), s[k]) for k in ("val", "test")}
            for m in (models or config.HAM_MODELS)}


# --------------------------------------------------------------------------- ensemble
def _f1(y, p):
    return f1_score(y, p.argmax(1), average="macro", labels=list(range(len(CLASSES))), zero_division=0)


def build_ensemble():
    s = splits()
    yv, yt = s["val"].label.to_numpy(), s["test"].label.to_numpy()
    P = load_preds()
    models = list(P)
    V = np.stack([P[m]["val"] for m in models])
    T = np.stack([P[m]["test"] for m in models])

    cands = {"equal": np.ones(len(models)) / len(models)}
    best3, best3_f1 = None, -1
    for combo in itertools.combinations(range(len(models)), 3):
        w = np.zeros(len(models)); w[list(combo)] = 1 / 3
        f = _f1(yv, np.tensordot(w, V, 1))
        if f > best3_f1:
            best3, best3_f1 = w, f
    cands["best3"] = best3
    rng = np.random.default_rng(42)
    bw, bf = None, -1
    for _ in range(config.ENSEMBLE_RANDOM_VECTORS):
        w = rng.random(len(models)); w /= w.sum()
        f = _f1(yv, np.tensordot(w, V, 1))
        if f > bf:
            bw, bf = w, f
    cands["weighted"] = bw
    val_f1 = {k: _f1(yv, np.tensordot(w, V, 1)) for k, w in cands.items()}
    chosen = max(val_f1, key=val_f1.get)              # selection on VALIDATION only
    w = cands[chosen]
    np.savez(OUT / "ensemble_probs.npz", val=np.tensordot(w, V, 1), test=np.tensordot(w, T, 1))
    json.dump({"models": models, "candidates": {k: v.tolist() for k, v in cands.items()},
               "val_macro_f1": val_f1, "chosen": chosen, "weights": w.tolist()},
              open(OUT / "ensemble_choice.json", "w"), indent=2)
    print("validation macro-F1:", val_f1, "-> chosen:", chosen)
    return chosen, w


def ensemble_probs(split="test"):
    return np.load(OUT / "ensemble_probs.npz")[split]


def ensemble_weights():
    c = json.load(open(OUT / "ensemble_choice.json"))
    return dict(zip(c["models"], c["weights"])), c


# --------------------------------------------------------------------------- tables
def time_per_image(m, n=100):
    """Milliseconds per image with 3-view TTA, batch size 1."""
    s = splits()
    model, _ = load_checkpoint(ckpt(m))
    df = s["test"].head(min(n, len(s["test"])))
    dl = torch.utils.data.DataLoader(engine.CachedImageDataset(df), batch_size=1)
    xs = [x for x, _ in dl]
    dev = next(model.parameters()).device
    with torch.no_grad():
        for x in xs[:5]:
            model(x.to(dev))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for x in xs:
            x = x.to(dev)
            for v in (x, x.flip(3), x.flip(2)):
                model(v)
        if dev.type == "cuda":
            torch.cuda.synchronize()
    return 1000 * (time.time() - t0) / len(xs)


def tables():
    s = splits()
    yt, gt = s["test"].label.to_numpy(), s["test"].lesion_id.to_numpy()
    yv = s["val"].label.to_numpy()
    P = load_preds()
    wmap, choice = ensemble_weights()
    rows, params, ms = [], {}, {}
    for m in P:
        model, _ = load_checkpoint(ckpt(m))
        params[m] = count_parameters(model) / 1e6
        ms[m] = time_per_image(m)
        del model
        r = stats.metrics(yt, P[m]["test"])
        rows.append({"model": DISPLAY_NAMES[m], "key": m, "params_m": params[m], "val_f1": _f1(yv, P[m]["val"]),
                     **r, "ms_per_img": ms[m]})
    V = np.stack([P[m]["val"] for m in P]); T = np.stack([P[m]["test"] for m in P])
    for k, label in (("equal", "Ensemble, equal weights"), ("best3", "Ensemble, best 3"),
                     ("weighted", "Ensemble, weighted")):
        w = np.array(choice["candidates"][k])
        used = [m for m, wi in zip(P, w) if wi > 0]
        r = stats.metrics(yt, np.tensordot(w, T, 1))
        rows.append({"model": label + (" (selected)" if k == choice["chosen"] else ""), "key": f"ens_{k}",
                     "params_m": sum(params[m] for m in used), "val_f1": _f1(yv, np.tensordot(w, V, 1)),
                     **r, "ms_per_img": sum(ms[m] for m in used)})
    t61 = pd.DataFrame(rows)
    t61.to_csv(OUT / "table_6_1_models.csv", index=False)

    pe = ensemble_probs("test")
    pred = pe.argmax(1)
    cm = confusion_matrix(yt, pred, labels=range(len(CLASSES)))
    per = []
    for c, name in enumerate(CLASSES):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c].sum() - tp; tn = cm.sum() - tp - fp - fn
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        per.append({"class": name, "precision": prec, "recall": rec, "specificity": tn / max(tn + fp, 1),
                    "f1": 2 * prec * rec / max(prec + rec, 1e-9), "auc": stats._safe_auc(yt == c, pe[:, c]),
                    "test_images": int((yt == c).sum())})
    pd.DataFrame(per).to_csv(OUT / "table_6_2_per_class.csv", index=False)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(OUT / "confusion_matrix.csv")

    miss = []
    for name in config.MALIGNANT_CLASSES:
        c = CLASSES.index(name)
        sel = yt == c
        k = int((~np.isin(pred[sel], stats.MAL)).sum()); n = int(sel.sum())
        lo, hi = stats.wilson(k, n)
        miss.append({"class": name, "test_images": n, "called_benign": k, "miss_rate": k / max(n, 1),
                     "wilson_lo": lo, "wilson_hi": hi})
    pd.DataFrame(miss).to_csv(OUT / "table_6_16_missed_cancers.csv", index=False)

    sel = choice["chosen"]
    summary = {"selected": sel, "test": stats.metrics(yt, pe),
               "test_ci": {m: stats.bootstrap_ci(lambda ix: stats.metrics(yt[ix], pe[ix])[m], gt)
                           for m in ("accuracy", "macro_f1", "melanoma_recall")},
               "val_macro_f1": choice["val_macro_f1"]}
    json.dump(summary, open(OUT / "ensemble_summary.json", "w"), indent=2, default=float)
    print(t61.round(3).to_string(index=False))
    return t61


# --------------------------------------------------------------------------- figures
def ensemble_cam(img_tensor, class_idx=None):
    """Weighted mean of the members' Grad-CAMs for the ensemble's predicted class."""
    from gradcam import GradCAM
    wmap, _ = ensemble_weights()
    cams, probs = [], 0
    for m, w in wmap.items():
        if w <= 0:
            continue
        model, _ = load_checkpoint(ckpt(m))
        x = img_tensor.to(next(model.parameters()).device)
        with torch.no_grad():
            probs = probs + w * torch.softmax(model(x), 1)[0].cpu().numpy()
        cams.append((m, w))
        del model
    cls = int(np.argmax(probs)) if class_idx is None else class_idx
    total = 0
    for m, w in cams:
        model, _ = load_checkpoint(ckpt(m))
        gc = GradCAM(model, get_target_layer(model, m))
        cam, _, _ = gc(img_tensor.to(next(model.parameters()).device), class_idx=cls)
        gc.remove()
        total = total + w * cam
        del model
    total = (total - total.min()) / (total.max() - total.min() + 1e-8)
    return total, cls, probs


def figures():
    from gradcam import overlay_heatmap
    from preprocess import load_image_rgb
    s = splits()
    yt = s["test"].label.to_numpy()
    t61 = pd.read_csv(OUT / "table_6_1_models.csv")

    # Fig 6.1 - metrics of all models and ensembles
    mets = ["accuracy", "balanced_accuracy", "macro_f1", "melanoma_recall", "malignant_sensitivity"]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    x = np.arange(len(mets)); wbar = 0.8 / len(t61)
    for i, r in t61.iterrows():
        ax.bar(x + i * wbar, [r[m] for m in mets], wbar, label=r["model"],
               hatch="//" if str(r["key"]).startswith("ens") else None, edgecolor="white")
    ax.set_xticks(x + 0.4 - wbar / 2); ax.set_xticklabels([m.replace("_", " ") for m in mets])
    ax.set_ylim(0, 1); ax.legend(fontsize=7, ncol=3); ax.grid(axis="y", alpha=.3)
    ax.set_title("HAM10000 test metrics (hatched: ensembles)")
    fig.tight_layout(); fig.savefig(OUT / "fig_6_1_models.png", dpi=150); plt.close(fig)

    # Fig 6.2 - confusion matrix; Fig 6.3 - ROC
    pe = ensemble_probs("test")
    cm = confusion_matrix(yt, pe.argmax(1), labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > .5 else "black")
    ax.set_xticks(range(7)); ax.set_xticklabels(CLASSES); ax.set_yticks(range(7)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.colorbar(im)
    ax.set_title("Selected ensemble - HAM10000 test")
    fig.tight_layout(); fig.savefig(OUT / "fig_6_2_confusion.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c, name in enumerate(CLASSES):
        if 0 < (yt == c).sum() < len(yt):
            fpr, tpr, _ = roc_curve(yt == c, pe[:, c])
            axes[0].plot(fpr, tpr, label=f"{name} (AUC {sk_auc(fpr, tpr):.3f})")
    ym = np.isin(yt, stats.MAL)
    fpr, tpr, _ = roc_curve(ym, pe[:, stats.MAL].sum(1))
    axes[1].plot(fpr, tpr, color="crimson", label=f"malignant vs benign (AUC {sk_auc(fpr, tpr):.3f})")
    for a in axes:
        a.plot([0, 1], [0, 1], "k--", lw=.8); a.legend(fontsize=8); a.set_xlabel("False positive rate")
        a.set_ylabel("True positive rate"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_6_3_roc.png", dpi=150); plt.close(fig)

    # Fig 6.4 - ensemble Grad-CAM, one test image per class
    tf = engine.eval_transform()
    picks = s["test"].groupby("dx").head(1).sort_values("dx")
    fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.6))
    for i, r in enumerate(picks.itertuples()):
        img = load_image_rgb(r.cache)
        cam, cls, probs = ensemble_cam(tf(img)[None])
        axes[0, i].imshow(img); axes[0, i].set_title(f"true {r.dx}", fontsize=9)
        axes[1, i].imshow(overlay_heatmap(img, cam))
        axes[1, i].set_title(f"pred {CLASSES[cls]} {probs[cls]:.2f}", fontsize=9)
        axes[0, i].axis("off"); axes[1, i].axis("off")
    fig.tight_layout(); fig.savefig(OUT / "fig_6_4_gradcam.png", dpi=150); plt.close(fig)

    # Fig 6.9 - most confident missed cancers
    pred = pe.argmax(1)
    missed = np.flatnonzero(np.isin(yt, stats.MAL) & ~np.isin(pred, stats.MAL))
    missed = missed[np.argsort(-pe[missed].max(1))][:6]
    if len(missed):
        fig, axes = plt.subplots(2, len(missed), figsize=(2.6 * len(missed), 5.6), squeeze=False)
        for i, ix in enumerate(missed):
            r = s["test"].iloc[ix]
            img = load_image_rgb(r.cache)
            cam, cls, probs = ensemble_cam(tf(img)[None])
            axes[0, i].imshow(img); axes[0, i].set_title(f"true {r.dx}", fontsize=9)
            axes[1, i].imshow(overlay_heatmap(img, cam))
            axes[1, i].set_title(f"pred {CLASSES[cls]} {probs[cls]:.2f}", fontsize=9)
            axes[0, i].axis("off"); axes[1, i].axis("off")
        fig.tight_layout(); fig.savefig(OUT / "fig_6_9_missed.png", dpi=150); plt.close(fig)
    print("figures ->", OUT)

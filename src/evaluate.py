"""Evaluate a trained checkpoint on the held-out test split.

Outputs (results/<model>/):
    metrics.json, per_class_metrics.csv, classification_report.txt,
    confusion_matrix.png, confusion_matrix_binary.png, roc_curves.png, test_predictions.csv

Usage:
    python src/evaluate.py --model efficientnet_b0
    python src/evaluate.py --model efficientnet_b0 --quick
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix,  # noqa: E402
                             f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from dataset import get_dataloaders  # noqa: E402
from models import load_checkpoint  # noqa: E402

MALIGNANT_IDX = [config.CLASS_NAMES.index(c) for c in config.MALIGNANT_CLASSES]


@torch.no_grad()
def predict_loader(model, loader, device):
    probs, labels = [], []
    for x, y in tqdm(loader, desc="Evaluating", leave=False):
        probs.append(torch.softmax(model(x.to(device)), 1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def specificity_per_class(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    tn = cm.sum() - tp - fp - fn
    return tn / np.maximum(tn + fp, 1)


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    names = config.CLASS_NAMES
    preds = probs.argmax(1)
    k = len(names)
    cm = confusion_matrix(labels, preds, labels=range(k))
    prec, rec, f1, support = precision_recall_fscore_support(labels, preds, labels=range(k), zero_division=0)
    spec = specificity_per_class(cm)
    y_bin = label_binarize(labels, classes=range(k))
    aucs = []
    for i in range(k):
        aucs.append(roc_auc_score(y_bin[:, i], probs[:, i]) if 0 < y_bin[:, i].sum() < len(labels) else float("nan"))

    per_class = pd.DataFrame({
        "class": names, "full_name": [config.CLASS_FULL_NAMES[c] for c in names],
        "type": [config.BINARY_MAP[c] for c in names],
        "precision": prec, "recall_sensitivity": rec, "specificity": spec,
        "f1": f1, "roc_auc": aucs, "support": support})

    # benign vs malignant: class prediction mapped to its group; score = summed malignant prob
    true_m = np.isin(labels, MALIGNANT_IDX).astype(int)
    pred_m = np.isin(preds, MALIGNANT_IDX).astype(int)
    score_m = probs[:, MALIGNANT_IDX].sum(1)
    bcm = confusion_matrix(true_m, pred_m, labels=[0, 1])
    tn, fp, fn, tp = bcm.ravel()
    binary = {
        "accuracy": float((tp + tn) / bcm.sum()),
        "precision": float(tp / max(tp + fp, 1)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(true_m, pred_m, zero_division=0)),
        "roc_auc": float(roc_auc_score(true_m, score_m)) if 0 < true_m.sum() < len(true_m) else float("nan"),
        "confusion_matrix": bcm.tolist(),
    }
    overall = {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(np.mean(rec)),
        "macro_precision": float(np.mean(prec)),
        "macro_recall_sensitivity": float(np.mean(rec)),
        "macro_specificity": float(np.mean(spec)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "macro_roc_auc": float(np.nanmean(aucs)),
        "n_test": int(len(labels)),
    }
    return {"overall": overall, "per_class": per_class, "binary": binary, "confusion_matrix": cm}


def plot_confusion(cm, names, out, title, normalize=True):
    data = cm.astype(float) / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    annot = np.array([[f"{cm[i, j]}\n({data[i, j]:.0%})" for j in range(cm.shape[1])] for i in range(cm.shape[0])])
    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 2.5, 1.0 * len(names) + 2))
    sns.heatmap(data, annot=annot, fmt="", cmap="Blues", xticklabels=names, yticklabels=names,
                vmin=0, vmax=1, cbar_kws={"label": "Row-normalised rate"}, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_roc(probs, labels, out, title):
    names = config.CLASS_NAMES
    y_bin = label_binarize(labels, classes=range(len(names)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for i, c in enumerate(names):
        if 0 < y_bin[:, i].sum() < len(labels):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            axes[0].plot(fpr, tpr, lw=1.6, label=f"{c} (AUC={auc(fpr, tpr):.3f})")
    axes[0].set_title(f"{title}: one-vs-rest ROC (7 classes)")
    true_m = np.isin(labels, MALIGNANT_IDX).astype(int)
    if 0 < true_m.sum() < len(true_m):
        fpr, tpr, _ = roc_curve(true_m, probs[:, MALIGNANT_IDX].sum(1))
        axes[1].plot(fpr, tpr, lw=2.2, color="#c0392b", label=f"Malignant vs benign (AUC={auc(fpr, tpr):.3f})")
    axes[1].set_title(f"{title}: benign vs malignant ROC")
    for ax in axes:
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
        ax.set_xlabel("False positive rate (1 - specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def evaluate(model_name: str = config.DEFAULT_MODEL, quick: bool = False, batch_size: int = config.BATCH_SIZE) -> dict:
    suffix = "_quick" if quick else ""
    ckpt_path = config.model_path(model_name, quick)
    out_dir = config.RESULTS_DIR / f"{model_name}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = config.get_device()
    model, ckpt = load_checkpoint(ckpt_path, device)
    loaders, dfs = get_dataloaders(batch_size, quick=quick)
    probs, labels = predict_loader(model, loaders["test"], device)
    m = compute_metrics(probs, labels)

    names = config.CLASS_NAMES
    m["per_class"].to_csv(out_dir / "per_class_metrics.csv", index=False)
    report = classification_report(labels, probs.argmax(1), labels=range(len(names)),
                                   target_names=names, digits=4, zero_division=0)
    (out_dir / "classification_report.txt").write_text(report)
    plot_confusion(m["confusion_matrix"], names, out_dir / "confusion_matrix.png",
                   f"Confusion matrix - {model_name} (test set)")
    plot_confusion(np.array(m["binary"]["confusion_matrix"]), ["Benign", "Malignant"],
                   out_dir / "confusion_matrix_binary.png", f"Benign vs malignant - {model_name}")
    plot_roc(probs, labels, out_dir / "roc_curves.png", model_name)

    pred_df = dfs["test"][["image_id", "lesion_id", "dx"]].copy().reset_index(drop=True)
    pred_df["pred"] = [names[i] for i in probs.argmax(1)]
    pred_df["confidence"] = probs.max(1)
    for i, c in enumerate(names):
        pred_df[f"p_{c}"] = probs[:, i]
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    summary = {"model": model_name, "checkpoint": ckpt_path.name, "best_epoch": ckpt.get("epoch"),
               "overall": m["overall"], "binary": m["binary"],
               "per_class": m["per_class"].drop(columns=["full_name"]).to_dict(orient="records")}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(report)
    print("Overall:", json.dumps(m["overall"], indent=2))
    print("Benign vs malignant:", json.dumps({k: v for k, v in m["binary"].items() if k != "confusion_matrix"}, indent=2))
    print(f"Saved results to {out_dir}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a trained model on the test split")
    p.add_argument("--model", default=config.DEFAULT_MODEL, choices=config.AVAILABLE_MODELS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    evaluate(a.model, a.quick, a.batch_size)

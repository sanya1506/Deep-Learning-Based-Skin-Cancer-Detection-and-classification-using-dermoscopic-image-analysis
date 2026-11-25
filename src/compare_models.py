"""Train and evaluate EfficientNet-B0, ResNet50 and DenseNet121, then compare them.

Outputs: results/model_comparison.csv and results/model_comparison.png

Usage:
    python src/compare_models.py --epochs 10                 # train + evaluate all three
    python src/compare_models.py --reuse                     # reuse existing checkpoints where present
    python src/compare_models.py --quick                     # small subset, 2 epochs (pipeline check)
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from evaluate import evaluate  # noqa: E402
from models import DISPLAY_NAMES, build_model, count_parameters  # noqa: E402
from train import train  # noqa: E402

METRIC_COLS = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced acc."), ("macro_f1", "Macro F1"),
               ("macro_roc_auc", "Macro ROC-AUC"), ("bin_sensitivity", "Malignant sensitivity"),
               ("bin_specificity", "Malignant specificity")]


def plot_comparison(df: pd.DataFrame, out: Path, title: str):
    x = np.arange(len(METRIC_COLS))
    width = 0.8 / len(df)
    palette = ["#2a78b8", "#e08a1e", "#3a9d5d"]
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[k] for k, _ in METRIC_COLS]
        bars = ax.bar(x + (i - (len(df) - 1) / 2) * width, vals, width, label=row["model"],
                      color=palette[i % len(palette)])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, [label for _, label in METRIC_COLS])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (test set)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Compare the three transfer-learning backbones")
    p.add_argument("--models", nargs="+", default=config.AVAILABLE_MODELS, choices=config.AVAILABLE_MODELS)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--reuse", action="store_true", help="skip training when a checkpoint already exists")
    a = p.parse_args()

    suffix = "_quick" if a.quick else ""
    rows = []
    for name in a.models:
        ckpt = config.model_path(name, a.quick)
        if not (a.reuse and ckpt.exists()):
            train(name, a.epochs, a.batch_size, quick=a.quick)
        else:
            print(f"Reusing existing checkpoint {ckpt.name}")
        m = evaluate(name, quick=a.quick, batch_size=a.batch_size)
        summ_path = config.RESULTS_DIR / f"{name}{suffix}" / "training_summary.json"
        summ = json.loads(summ_path.read_text()) if summ_path.exists() else {}
        rows.append({
            "model": DISPLAY_NAMES[name],
            "params_millions": round(count_parameters(build_model(name, pretrained=False)) / 1e6, 2),
            **{k: round(v, 4) for k, v in m["overall"].items() if k != "n_test"},
            **{f"bin_{k}": round(v, 4) for k, v in m["binary"].items() if k != "confusion_matrix"},
            "epochs_run": summ.get("epochs_run"),
            "best_epoch": m.get("best_epoch"),
            "train_minutes": round(summ.get("train_minutes", float("nan")), 1),
        })

    df = pd.DataFrame(rows)
    out_csv = config.RESULTS_DIR / f"model_comparison{suffix}.csv"
    df.to_csv(out_csv, index=False)
    plot_comparison(df, config.RESULTS_DIR / f"model_comparison{suffix}.png",
                    "Model comparison on HAM10000 test set" + (" (quick mode)" if a.quick else ""))
    print(df.to_string(index=False))
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()

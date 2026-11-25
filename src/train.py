"""Train a transfer-learning classifier on HAM10000.

Examples:
    python src/train.py --quick                       # 2-epoch smoke test on a small subset
    python src/train.py --model efficientnet_b0 --epochs 15
    python src/train.py --model resnet50 --epochs 10 --batch-size 32
"""
import argparse
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
import torch.nn as nn  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from dataset import get_dataloaders  # noqa: E402
from models import build_model  # noqa: E402


def set_seed(seed: int = config.SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(train_df: pd.DataFrame, device) -> torch.Tensor:
    """Balanced class weights (n_samples / (n_classes * n_c)), square-rooted to avoid over-correction."""
    labels = train_df["label"].to_numpy()
    present = np.unique(labels)
    w = np.ones(config.NUM_CLASSES, dtype=np.float32)
    w[present] = compute_class_weight("balanced", classes=present, y=labels)
    w = np.sqrt(w)
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    train = optimizer is not None
    model.train(train)
    total_loss, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds.append(logits.argmax(1).cpu())
            targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return {
        "loss": total_loss / len(loader.dataset),
        "acc": float((preds == targets).mean()),
        "f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
    }


def plot_history(hist: pd.DataFrame, out: Path, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, label in zip(axes, ["loss", "acc", "f1"], ["Loss", "Accuracy", "Macro F1"]):
        ax.plot(hist["epoch"], hist[f"train_{key}"], "o-", label="train")
        ax.plot(hist["epoch"], hist[f"val_{key}"], "s-", label="validation")
        ax.set_xlabel("Epoch")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def train(model_name: str = config.DEFAULT_MODEL, epochs: int = config.EPOCHS,
          batch_size: int = config.BATCH_SIZE, lr: float = config.LEARNING_RATE,
          quick: bool = False, patience: int = config.EARLY_STOPPING_PATIENCE) -> Path:
    set_seed()
    device = config.get_device()
    if quick:
        epochs = min(epochs, config.QUICK_EPOCHS)
    print(f"Device: {device} | model: {model_name} | epochs: {epochs} | batch: {batch_size} | quick: {quick}")

    loaders, dfs = get_dataloaders(batch_size, quick=quick)
    print(f"train={len(dfs['train'])} val={len(dfs['val'])} test={len(dfs['test'])}")

    model = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(dfs["train"], device), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2)

    suffix = "_quick" if quick else ""
    ckpt_path = config.model_path(model_name, quick)
    out_dir = config.RESULTS_DIR / f"{model_name}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_f1, bad_epochs, history = -1.0, 0, []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        te = time.time()
        tr = run_epoch(model, loaders["train"], criterion, device, optimizer, f"train {epoch}/{epochs}")
        va = run_epoch(model, loaders["val"], criterion, device, None, f"val   {epoch}/{epochs}")
        scheduler.step(va["f1"])
        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in va.items()}, "lr": lr_now,
                        "seconds": time.time() - te})
        print(f"Epoch {epoch:2d}: train loss {tr['loss']:.4f} acc {tr['acc']:.4f} f1 {tr['f1']:.4f} | "
              f"val loss {va['loss']:.4f} acc {va['acc']:.4f} f1 {va['f1']:.4f} | "
              f"lr {lr_now:.1e} | {time.time() - te:.0f}s")

        if va["f1"] > best_f1:
            best_f1, bad_epochs = va["f1"], 0
            torch.save({
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "class_names": config.CLASS_NAMES,
                "img_size": config.IMG_SIZE,
                "epoch": epoch,
                "val_f1": va["f1"],
                "val_acc": va["acc"],
                "quick": quick,
            }, ckpt_path)
            print(f"  -> saved best model (val macro-F1 {best_f1:.4f}) to {ckpt_path.name}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience and (quick or epoch >= config.MIN_EPOCHS):
                print(f"Early stopping after {epoch} epochs (no improvement for {patience}).")
                break

    hist = pd.DataFrame(history)
    hist.to_csv(out_dir / "training_history.csv", index=False)
    plot_history(hist, out_dir / "training_curves.png", f"Training curves - {model_name}")
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump({"model": model_name, "epochs_run": len(history), "best_val_f1": best_f1,
                   "train_minutes": (time.time() - t0) / 60, "device": str(device),
                   "train_size": len(dfs["train"]), "quick": quick}, f, indent=2)
    print(f"Training done in {(time.time() - t0) / 60:.1f} min. Best val macro-F1: {best_f1:.4f}")
    return ckpt_path


def parse_args():
    p = argparse.ArgumentParser(description="Train a skin lesion classifier")
    p.add_argument("--model", default=config.DEFAULT_MODEL, choices=config.AVAILABLE_MODELS)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE)
    p.add_argument("--quick", action="store_true", help="small subset, 2 epochs (pipeline check)")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    train(a.model, a.epochs, a.batch_size, a.lr, a.quick, a.patience)

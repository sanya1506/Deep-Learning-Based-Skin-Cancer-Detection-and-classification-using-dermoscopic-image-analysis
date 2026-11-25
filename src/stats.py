"""Metrics and statistics used throughout the report (Section 4.8).

* `metrics`          - accuracy, balanced accuracy, macro-F1, melanoma recall, malignant sensitivity /
                       specificity / AUC, macro AUC
* `bootstrap_ci`     - 1,000 resamples of GROUPS (lesions for HAM10000, patients for DermaCon-IN)
* `seed_summary`     - mean +/- std over seeds and a bootstrap CI of the seed-mean
* `paired_diff`      - difference B - A with the SAME bootstrap resample for both (paired)
* `wilson`           - Wilson score interval for a proportion
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

MAL = [config.CLASS_NAMES.index(c) for c in config.MALIGNANT_CLASSES]
MEL = config.CLASS_NAMES.index("mel")
K = config.NUM_CLASSES


def _safe_auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < np.sum(y) < len(y) else float("nan")


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    pred = p.argmax(1)
    rec = [np.mean(pred[y == c] == c) if np.any(y == c) else np.nan for c in range(K)]
    ym, pm = np.isin(y, MAL), np.isin(pred, MAL)
    aucs = [_safe_auc(y == c, p[:, c]) for c in range(K)]
    return {
        "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": float(np.nanmean(rec)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(K)), zero_division=0)),
        "melanoma_recall": float(rec[MEL]) if not np.isnan(rec[MEL]) else float("nan"),
        "malignant_sensitivity": float(pm[ym].mean()) if ym.any() else float("nan"),
        "malignant_specificity": float((~pm[~ym]).mean()) if (~ym).any() else float("nan"),
        "malignant_auc": _safe_auc(ym, p[:, MAL].sum(1)),
        "macro_auc": float(np.nanmean(aucs)),
    }


REPORT_METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "melanoma_recall",
                  "malignant_sensitivity", "malignant_specificity", "malignant_auc"]


def group_index(groups):
    """List of index arrays, one per group (lesion / patient)."""
    groups = np.asarray(groups)
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    cuts = np.flatnonzero(g_sorted[1:] != g_sorted[:-1]) + 1
    return np.split(order, cuts)


def resamples(groups, n=None, seed=0):
    """Yield index arrays of group-level bootstrap resamples."""
    n = n or config.N_BOOTSTRAP
    idx = group_index(groups)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        pick = rng.integers(0, len(idx), len(idx))
        yield np.concatenate([idx[i] for i in pick])


def bootstrap_ci(fn, groups, n=None, seed=0, alpha=0.05):
    """fn(index_array) -> float. Percentile CI over group resamples."""
    vals = np.array([fn(ix) for ix in resamples(groups, n, seed)], dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2))))


def seed_summary(y, probs_per_seed: list, groups, metric_names=REPORT_METRICS, seed=0) -> dict:
    """{metric: {mean, std, ci: [lo, hi], per_seed: [...]}}; the CI is for the mean over seeds."""
    per = [metrics(y, p) for p in probs_per_seed]
    out = {}
    boot = {m: [] for m in metric_names}
    for ix in resamples(groups, seed=seed):
        ms = [metrics(y[ix], p[ix]) for p in probs_per_seed]
        for m in metric_names:
            boot[m].append(np.nanmean([d[m] for d in ms]))
    for m in metric_names:
        v = np.array([d[m] for d in per], float)
        b = np.array(boot[m], float)
        b = b[~np.isnan(b)]
        out[m] = {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
                  "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) else [None, None],
                  "per_seed": v.tolist()}
    return out


def paired_diff(y, probs_a: list, probs_b: list, groups, metric_names=REPORT_METRICS, seed=0) -> dict:
    """Mean over seeds of (B - A) with a paired bootstrap CI (same resample for A and B)."""
    point = {}
    for m in metric_names:
        a = np.nanmean([metrics(y, p)[m] for p in probs_a])
        b = np.nanmean([metrics(y, p)[m] for p in probs_b])
        point[m] = b - a
    boot = {m: [] for m in metric_names}
    for ix in resamples(groups, seed=seed):
        ma = [metrics(y[ix], p[ix]) for p in probs_a]
        mb = [metrics(y[ix], p[ix]) for p in probs_b]
        for m in metric_names:
            boot[m].append(np.nanmean([d[m] for d in mb]) - np.nanmean([d[m] for d in ma]))
    return {m: {"diff": float(point[m]),
                "ci": [float(np.nanpercentile(boot[m], 2.5)), float(np.nanpercentile(boot[m], 97.5))]}
            for m in metric_names}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def fmt_ms(d: dict, digits=3) -> str:
    """'0.695 ± 0.023 [0.641, 0.738]' from a seed_summary entry."""
    lo, hi = d["ci"]
    return f"{d['mean']:.{digits}f} ± {d['std']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"

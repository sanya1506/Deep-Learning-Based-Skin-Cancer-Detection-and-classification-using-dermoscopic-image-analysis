"""Ensemble the v2 runs and calibrate class priors - everything fitted on VALIDATION only.

1. Ensemble weights: weighted average of log-probabilities, weights chosen to minimise val NLL.
2. Prior correction: a per-class logit bias tuned by coordinate search to maximise val accuracy
   (fixes the prior shift caused by ISIC 2019 extra data / class imbalance).
3. Binary (benign vs malignant) threshold chosen on val; also reported at >=90% sensitivity.
4. Optional lesion-level averaging: images of the same lesion share one prediction
   (HAM10000 has several images of many lesions; only valid when lesion identity is known).

The test set is touched exactly once, at the end.

    python improved/ensemble.py                       # use every run in results/v2/
    python improved/ensemble.py --runs a_meta b_meta  # choose runs
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)

from common import CLASS_NAMES, CLASS_TO_IDX, MALIGNANT_IDX, V2_RESULTS


def load_run(tag):
    d = V2_RESULTS / tag
    val, test = pd.read_csv(d / "val_predictions.csv"), pd.read_csv(d / "test_predictions.csv")
    cols = [f"p_{c}" for c in CLASS_NAMES]
    return val, val[cols].to_numpy(), test, test[cols].to_numpy()


def combine(logps, w, bias=None):
    z = sum(wi * lp for wi, lp in zip(w, logps))
    if bias is not None:
        z = z + bias
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def fit_weights(logps, y):
    k = len(logps)
    if k == 1:
        return np.ones(1)

    def nll(theta):
        w = np.exp(theta) / np.exp(theta).sum()
        return -np.log(combine(logps, w)[np.arange(len(y)), y] + 1e-9).mean()
    th = minimize(nll, np.zeros(k), method="Nelder-Mead", options={"maxiter": 2000}).x
    return np.exp(th) / np.exp(th).sum()


def fit_bias(p, y, grid=np.arange(-1.5, 1.51, 0.05), passes=3, reg=0.002):
    """Coordinate ascent on val accuracy (small L1 penalty keeps biases near 0)."""
    lp, b = np.log(p + 1e-9), np.zeros(p.shape[1])
    score = lambda bb: (np.argmax(lp + bb, 1) == y).mean() - reg * np.abs(bb).sum()
    for _ in range(passes):
        for c in range(p.shape[1]):
            cands = []
            for g in grid:
                bb = b.copy(); bb[c] = g
                cands.append((score(bb), -abs(g), g))
            b[c] = max(cands)[2]
    return b


def lesion_average(p, lesion_ids):
    df = pd.DataFrame(p); df["l"] = np.asarray(lesion_ids)
    return df.groupby("l").transform("mean").to_numpy()


def metrics(y, p, thr=0.5):
    pred = p.argmax(1)
    yb = np.isin(y, MALIGNANT_IDX); pm = p[:, MALIGNANT_IDX].sum(1); pb = pm >= thr
    tp, tn = (pb & yb).sum(), (~pb & ~yb).sum()
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"), "weighted_f1": f1_score(y, pred, average="weighted"),
            "macro_roc_auc": roc_auc_score(y, p, multi_class="ovr", average="macro"),
            "bin_accuracy": (pb == yb).mean(), "bin_sensitivity": tp / yb.sum(),
            "bin_specificity": tn / (~yb).sum(), "bin_roc_auc": roc_auc_score(yb, pm), "bin_threshold": thr}


def pick_threshold(y, p, min_sens=None):
    yb = np.isin(y, MALIGNANT_IDX); pm = p[:, MALIGNANT_IDX].sum(1)
    best = (0, 0.5)
    for t in np.arange(0.05, 0.951, 0.01):
        pb = pm >= t
        if min_sens is not None:
            if (pb & yb).sum() / yb.sum() >= min_sens:
                best = max(best, ((pb == yb).mean(), t))
        else:
            best = max(best, ((pb == yb).mean(), t))
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--search-subsets", action="store_true", help="try every subset of runs (on val)")
    a = ap.parse_args()
    runs = a.runs or sorted(d.name for d in V2_RESULTS.iterdir() if (d / "test_predictions.csv").exists())
    data = {r: load_run(r) for r in runs}
    val0, test0 = data[runs[0]][0], data[runs[0]][2]
    yv, yt = val0.dx.map(CLASS_TO_IDX).to_numpy(), test0.dx.map(CLASS_TO_IDX).to_numpy()
    for r in runs:
        assert (data[r][0].image_id.values == val0.image_id.values).all(), f"{r}: val order differs"

    rows = []
    for r in runs:
        rows.append({"model": r, "val_acc": accuracy_score(yv, data[r][1].argmax(1)),
                     "test_acc": accuracy_score(yt, data[r][3].argmax(1))})

    subsets = [tuple(runs)]
    if a.search_subsets and len(runs) > 2:
        subsets = [s for k in range(2, len(runs) + 1) for s in itertools.combinations(runs, k)]

    def fit(sub):
        lv = [np.log(data[r][1] + 1e-9) for r in sub]
        w = fit_weights(lv, yv)
        pv = combine(lv, w)
        b = fit_bias(pv, yv)
        return w, b, (np.argmax(np.log(pv) + b, 1) == yv).mean()

    fits = {s: fit(s) for s in subsets}
    best_sub = max(fits, key=lambda s: fits[s][2])     # chosen on VAL accuracy
    w, b, _ = fits[best_sub]
    lv = [np.log(data[r][1] + 1e-9) for r in best_sub]
    lt = [np.log(data[r][3] + 1e-9) for r in best_sub]
    pv, pt = combine(lv, w, b), combine(lt, w, b)
    thr = pick_threshold(yv, pv)
    thr_s90 = pick_threshold(yv, pv, min_sens=0.90)

    results = {
        "ensemble_members": dict(zip(best_sub, np.round(w, 4).tolist())),
        "class_bias": dict(zip(CLASS_NAMES, np.round(b, 3).tolist())),
        "test_mean_ensemble_no_calibration": metrics(yt, combine(lt, np.ones(len(lt)) / len(lt))),
        "test_ensemble_calibrated": metrics(yt, pt, thr),
        "test_ensemble_calibrated_sens90": metrics(yt, pt, thr_s90),
        "test_ensemble_calibrated_lesion_avg": metrics(yt, lesion_average(pt, test0.lesion_id), thr),
        "individual_runs": rows,
    }
    results = json.loads(json.dumps(results, default=float))
    out = V2_RESULTS / "ensemble"
    out.mkdir(exist_ok=True)
    json.dump(results, open(out / "ensemble_metrics.json", "w"), indent=2)
    json.dump({"members": list(best_sub), "weights": w.tolist(), "bias": b.tolist(),
               "binary_threshold": thr, "binary_threshold_sens90": thr_s90},
              open(out / "ensemble_config.json", "w"), indent=2)
    tp = test0[["image_id", "lesion_id", "dx"]].copy()
    for i, c in enumerate(CLASS_NAMES):
        tp[f"p_{c}"] = pt[:, i]
    tp["pred"] = [CLASS_NAMES[i] for i in pt.argmax(1)]
    tp.to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame(confusion_matrix(yt, pt.argmax(1)), index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out / "confusion_matrix.csv")
    with open(out / "classification_report.txt", "w") as f:
        f.write(classification_report(yt, pt.argmax(1), target_names=CLASS_NAMES, digits=4))

    print(pd.DataFrame(rows).to_string(index=False))
    print("\nmembers/weights:", results["ensemble_members"])
    for k in ("test_mean_ensemble_no_calibration", "test_ensemble_calibrated",
              "test_ensemble_calibrated_sens90", "test_ensemble_calibrated_lesion_avg"):
        m = results[k]
        print(f"{k:40s} acc {m['accuracy']:.4f}  bal {m['balanced_accuracy']:.4f}  F1 {m['macro_f1']:.4f}  "
              f"| binary acc {m['bin_accuracy']:.4f} sens {m['bin_sensitivity']:.4f} spec {m['bin_specificity']:.4f}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()

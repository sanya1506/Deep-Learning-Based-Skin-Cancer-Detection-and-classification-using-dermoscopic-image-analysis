"""Automatic sanity checks (report Section 5.6): notebook 09 (HAM10000) and notebook 19 (Indian study).
Each check returns (name, passed, detail); results go to results/study/sanity_*.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

S = config.STUDY_DIR


def _save(name, checks):
    out = [{"check": c, "passed": bool(p), "detail": str(d)} for c, p, d in checks]
    json.dump(out, open(S / f"sanity_{name}.json", "w"), indent=2)
    n_ok = sum(c["passed"] for c in out)
    for c in out:
        print(("PASS " if c["passed"] else "FAIL ") + c["check"] + "  " + c["detail"])
    print(f"{n_ok}/{len(out)} checks passed")
    return out


def ham_checks():
    import ham
    s = ham.splits()
    L = {k: set(v.lesion_id) for k, v in s.items()}
    checks = [("no lesion in train and val", not L["train"] & L["val"], len(L["train"] & L["val"])),
              ("no lesion in train and test", not L["train"] & L["test"], len(L["train"] & L["test"])),
              ("no lesion in val and test", not L["val"] & L["test"], len(L["val"] & L["test"]))]
    n = sum(len(pd.read_csv(config.SPLITS_DIR / f"{k}.csv")) for k in ("train", "val", "test"))
    checks.append(("split covers all 10,015 images", n == 10015, n))
    c = json.load(open(S / "ham" / "ensemble_choice.json"))
    P = ham.load_preds()
    yv = s["val"].label.to_numpy()
    V = np.stack([P[m]["val"] for m in c["models"]])
    f = {k: f1_score(yv, np.tensordot(np.array(w), V, 1).argmax(1), average="macro", labels=list(range(7)), zero_division=0)
         for k, w in c["candidates"].items()}
    checks.append(("ensemble choice reproduced from validation", max(f, key=f.get) == c["chosen"], c["chosen"]))
    ok = all((config.MODELS_DIR / f"{m}.pth").exists() and len(P[m]["test"]) == len(s["test"]) for m in config.HAM_MODELS)
    checks.append(("six HAM10000 models trained and evaluated on the full test split", ok, len(s["test"])))
    return _save("ham", checks)


def indian_checks():
    import dermacon
    import studies
    checks = []
    parts = {k: pd.read_csv(f) for k, f in dermacon.SPLIT_FILES.items()}
    P = {k: set(v.patient) for k, v in parts.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        checks.append((f"no patient in DermaCon-IN {a} and {b}", not P[a] & P[b], len(P[a] & P[b])))
    df = dermacon.load()
    mal_test = df[df.image_id.isin(parts["test"].image_id) & df.malignant]
    checks.append(("DermaCon-IN test split contains malignant images", len(mal_test) > 0, len(mal_test)))
    for prefix, n in (("study_imb_", 15), ("study_skin_", 3), ("study_crop_", 3), ("unet_", 3), ("gate_", 3)):
        k = len(list(config.MODELS_DIR.glob(f"{prefix}*.pth")))
        checks.append((f"{prefix}* runs complete ({n} expected)", k >= n, k))
    sel = json.load(open(S / "studies" / "imbalance_selected.json"))
    val = {k: tuple(v) for k, v in sel["val"].items()}
    best = max(v[0] for v in val.values())
    re_sel = max([k for k, v in val.items() if best - v[0] <= 0.005], key=lambda k: (val[k][1], val[k][0]))
    checks.append(("imbalance setup reproduced from validation", re_sel == sel["selected"], sel["selected"]))
    t68 = json.load(open(S / "dermacon" / "table_6_8_two_stage.json"))
    checks.append(("gate thresholds stored for 3 seeds", len(t68["thresholds"]) == 3, t68["thresholds"]))
    fcr = json.load(open(S / "dermacon" / "table_6_3_fcr.json"))["rows"]
    n_nonneo = int((~dermacon.splits()[0]["test"].neoplasm).sum())
    checks.append(("FCR denominator = non-neoplastic test images", all(r.get("n", n_nonneo) == n_nonneo for r in fcr), n_nonneo))
    checks.append(("label mapping file written", (config.CONFIGS_DIR / "dermacon_mapping.csv").exists(), "configs/dermacon_mapping.csv"))
    import torch
    seeds_ok = all(torch.load(f, map_location="cpu", weights_only=False).get("seed") in config.SEEDS
                   for f in config.MODELS_DIR.glob("study_*.pth"))
    checks.append(("every study checkpoint was trained with seed 0, 1 or 2", seeds_ok, "study_*.pth"))
    nums = S / "report_numbers.json"
    checks.append(("report numbers file written", nums.exists(), nums.name))
    checks.append(("not a smoke run", not config.SMOKE, f"SMOKE={config.SMOKE}"))
    return _save("indian", checks)

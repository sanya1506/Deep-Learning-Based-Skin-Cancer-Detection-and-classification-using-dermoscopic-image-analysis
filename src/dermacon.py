"""Indian-skin study on DermaCon-IN (report Sections 3.2, 4.5, 4.6, 6.2-6.4).

  audit()          Tables 3.2-3.6, Fig 3.3, patient-level split, configs/dermacon_mapping.csv
  zero_shot()      Tables 6.3-6.6, Fig 6.5   (HAM models + ensemble, no retraining)
  gate()           Tables 6.7-6.8, Fig 6.6   (Stage 1 EfficientNet-B0 on 8 main classes, 3 seeds)
  ddi()            Table 6.9                 (runs only if data/ddi/ddi_metadata.csv exists)

DermaCon-IN is research-use data obtained after approval: put your copy under data/dermacon/
(any folder layout: the metadata CSV/XLSX and the images are found automatically). Column names
are detected automatically and printed; if detection is wrong, write configs/dermacon_columns.json
e.g. {"image": "Image Name", "patient": "Patient ID", ...}.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import engine  # noqa: E402
import ham  # noqa: E402
import stats  # noqa: E402

OUT = config.STUDY_DIR / "dermacon"
OUT.mkdir(parents=True, exist_ok=True)
SPLIT_FILES = {s: config.SPLITS_DIR / f"dermacon_{s}.csv" for s in ("train", "val", "test")}
COLS = ["image", "patient", "main_class", "sub_class", "diagnosis", "fst", "mst"]
KEYS = {
    "image": ["imagename", "imagefile", "filename", "imageid", "image", "img", "file"],
    "patient": ["patientid", "patient", "subjectid", "caseid", "pid"],
    "main_class": ["mainclass", "maincategory", "maindiagnosticclass", "main"],
    "sub_class": ["subclass", "subcategory", "etiology", "sub"],
    "diagnosis": ["diseaselabel", "disease", "diagnosis", "label"],
    "fst": ["fitzpatrick", "fst"],
    "mst": ["monk", "mst"],
}
MALIGNANT_RE = re.compile(r"melanoma|basal\s*cell|squamous\s*cell|bowen|keratoacanthoma", re.I)
MAPPING_RULES = [  # (regex on diagnosis, HAM10000 class); first match wins
    (r"(?<!non[- ])melanoma", "mel"), (r"basal\s*cell", "bcc"), (r"seborr?h?oeic\s*keratos|seborrheic\s*keratos", "bkl"),
    (r"nevocytic|melanocytic\s*nev|compound\s*nev|junctional\s*nev|intradermal\s*nev", "nv"),
    (r"dermatofibroma", "df"), (r"cherry\s*angioma|ha?emangioma|pyogenic\s*granuloma", "vasc"),
]


# --------------------------------------------------------------------------- loading
def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def find_metadata():
    roots = [config.DERMACON_DIR] + ([Path("/kaggle/input")] if Path("/kaggle/input").exists() else [])
    best = None
    for r in roots:
        for f in list(r.rglob("*.csv")) + list(r.rglob("*.xlsx")):
            if "ham10000" in f.name.lower() or "isic" in f.name.lower():
                continue
            try:
                df = pd.read_csv(f, nrows=5) if f.suffix == ".csv" else pd.read_excel(f, nrows=5)
            except Exception:
                continue
            score = sum(any(k in _norm(c) for c in df.columns for k in KEYS[x][:2]) for x in COLS)
            if score < 3 or (best is not None and score < best[0]):
                continue
            # tie-break on row count: the full per-image metadata beats a subset (e.g. train/test split files)
            rows = sum(1 for _ in open(f, encoding="utf-8", errors="ignore")) if f.suffix == ".csv" else len(pd.read_excel(f))
            if best is None or (score, rows) > best[:2]:
                best = (score, rows, f)
    if best is None or best[0] < 3:
        raise FileNotFoundError("DermaCon-IN metadata not found. Put the dataset under data/dermacon/ "
                                "(or attach it as a Kaggle input).")
    return best[2]


def detect_columns(df) -> dict:
    override = config.CONFIGS_DIR / "dermacon_columns.json"
    if override.exists():
        return json.load(open(override))
    found, used = {}, set()
    for col in COLS:
        for key in KEYS[col]:
            hit = [c for c in df.columns if key in _norm(c) and c not in used]
            if hit:
                found[col] = hit[0]; used.add(hit[0]); break
    return found


def load() -> pd.DataFrame:
    f = find_metadata()
    raw = pd.read_csv(f) if f.suffix == ".csv" else pd.read_excel(f)
    cols = detect_columns(raw)
    print(f"DermaCon-IN metadata: {f}\ncolumns in file: {list(raw.columns)}\ndetected: {cols}")
    missing = [c for c in ("image", "patient", "main_class") if c not in cols]
    if missing:
        raise KeyError(f"could not detect columns {missing}; write configs/dermacon_columns.json")
    df = pd.DataFrame({c: raw[cols[c]] if c in cols else np.nan for c in COLS})
    root = f.parent
    for _ in range(3):        # images may sit next to the CSV or a few folders above it
        if any(root.rglob("*.jp*g")) or any(root.rglob("*.png")) or root.parent in (root, Path("/kaggle/input"), config.DATA_DIR):
            break
        root = root.parent
    files = {}
    for p in list(root.rglob("*.jpg")) + list(root.rglob("*.jpeg")) + list(root.rglob("*.png")) + list(root.rglob("*.JPG")):
        files.setdefault(p.name.lower(), p); files.setdefault(p.stem.lower(), p)
    df["path"] = df["image"].map(lambda v: files.get(str(v).lower(), files.get(Path(str(v)).stem.lower())))
    n_miss = df.path.isna().sum()
    if n_miss:
        print(f"warning: {n_miss} metadata rows without an image file - dropped")
    df = df.dropna(subset=["path"]).reset_index(drop=True)
    df["path"] = df.path.astype(str)
    df["image_id"] = df["path"].map(lambda p: "dc_" + re.sub(r"[^A-Za-z0-9_-]", "_", Path(p).stem))
    df["cache"] = df.image_id.map(lambda i: str(config.PROCESSED_DERMACON_DIR / f"{i}.jpg"))
    df["crop_cache"] = df.image_id.map(lambda i: str(config.PROCESSED_DERMACON_CROP_DIR / f"{i}.jpg"))
    df["main_class"] = df.main_class.astype(str).str.strip()
    df["neoplasm"] = df.main_class.str.contains("neoplas", case=False)
    df["malignant"] = df.diagnosis.astype(str).str.contains(MALIGNANT_RE)
    df["ham_class"] = df.diagnosis.astype(str).map(map_to_ham)
    df["group"] = [disease_group(m, s, d) for m, s, d in zip(df.main_class, df.sub_class, df.diagnosis)]
    df["fst_num"] = df.fst.map(parse_scale)
    df["mst_num"] = df.mst.map(parse_scale)
    return df


def map_to_ham(diag: str) -> str:
    for rx, c in MAPPING_RULES:
        if re.search(rx, str(diag), re.I):
            return c
    return "unmapped"


def disease_group(main, sub, diag) -> str:
    t = f"{sub} {diag}".lower()
    m = str(main).lower()
    if "infect" in m:
        for k, g in (("fung", "Fungal infection"), ("tinea", "Fungal infection"), ("bacter", "Bacterial infection"),
                     ("viral", "Viral infection"), ("wart", "Viral infection"), ("parasit", "Parasitic (scabies etc.)"),
                     ("scabies", "Parasitic (scabies etc.)")):
            if k in t:
                return g
    if "inflam" in m:
        for k, g in (("eczema", "Eczema / dermatitis"), ("dermatitis", "Eczema / dermatitis"),
                     ("psoria", "Psoriasis / lichenoid"), ("lichen", "Psoriasis / lichenoid"),
                     ("papulosquam", "Psoriasis / lichenoid")):
            if k in t:
                return g
    if "append" in m:
        for k, g in (("acne", "Acne / sebaceous"), ("sebac", "Acne / sebaceous"), ("hair", "Hair disorders"),
                     ("alopecia", "Hair disorders")):
            if k in t:
                return g
    return str(main)


def parse_scale(v):
    if pd.isna(v):
        return np.nan
    s = str(v).upper().strip()
    roman = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
    m = re.search(r"\b(VI|IV|V|III|II|I)\b", s)
    if m and not re.search(r"\d", s):
        return roman[m.group(1)]
    m = re.search(r"\d+", s)
    return float(m.group()) if m else np.nan


# --------------------------------------------------------------------------- audit + split
def make_split(df, force=False):
    if all(f.exists() for f in SPLIT_FILES.values()) and not force:
        parts = {s: pd.read_csv(f) for s, f in SPLIT_FILES.items()}
        ids = {s: set(p.image_id) for s, p in parts.items()}
        return {s: df[df.image_id.isin(ids[s])].reset_index(drop=True) for s in parts}
    counts = df.main_class.value_counts()
    pat = df.groupby("patient").agg(mal=("malignant", "any"),
                                    rare=("main_class", lambda x: min(x, key=lambda c: counts[c]))).reset_index()
    pat["stratum"] = np.where(pat.mal, "malignant", pat.rare)
    small = pat.stratum.value_counts()
    pat.loc[pat.stratum.map(small) < 6, "stratum"] = "rare"     # tiny strata merged so the split can stratify
    tr, rest = train_test_split(pat, test_size=0.30, stratify=pat.stratum, random_state=config.SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=rest.stratum, random_state=config.SEED)
    out = {}
    for s, p in (("train", tr), ("val", va), ("test", te)):
        out[s] = df[df.patient.isin(set(p.patient))].reset_index(drop=True)
        out[s][["image_id", "patient", "main_class"]].to_csv(SPLIT_FILES[s], index=False)
    assert not set(out["train"].patient) & set(out["test"].patient)
    assert not set(out["train"].patient) & set(out["val"].patient)
    assert not set(out["val"].patient) & set(out["test"].patient)
    assert out["test"].malignant.any(), "test split has no malignant image"
    return out


def splits():
    df = load()
    s = make_split(df)
    if config.SMOKE:
        s = {k: v.groupby("main_class").head(6 if k == "train" else 4).reset_index(drop=True) for k, v in s.items()}
        keep = set(s["test"].image_id) | set(df[df.malignant].image_id)
        s["test"] = df[df.image_id.isin(keep) & df.image_id.isin(pd.read_csv(SPLIT_FILES["test"]).image_id)].reset_index(drop=True)
    main = sorted(df.main_class.unique())
    for v in s.values():
        v["main_label"] = v.main_class.map({c: i for i, c in enumerate(main)})
    return s, main


def audit():
    df = load()
    s = make_split(df, force=not all(f.exists() for f in SPLIT_FILES.values()))
    t32 = df.groupby("main_class").agg(images=("image_id", "size"), patients=("patient", "nunique"))
    t32.sort_values("images", ascending=False).to_csv(OUT / "table_3_2_main_classes.csv")
    t33 = pd.DataFrame({k: v.main_class.value_counts() for k, v in s.items()}).fillna(0).astype(int)
    t33.loc["TOTAL images"] = [len(s[k]) for k in t33.columns]
    t33.loc["TOTAL patients"] = [s[k].patient.nunique() for k in t33.columns]
    t33.to_csv(OUT / "table_3_3_split.csv")
    mal = df[df.malignant].copy()
    mal["split"] = mal.image_id.map({i: k for k, v in s.items() for i in v.image_id})
    t34 = mal.pivot_table(index="diagnosis", columns="split", values="image_id", aggfunc="count", fill_value=0)
    t34["patients"] = mal.groupby("diagnosis").patient.nunique()
    t34.to_csv(OUT / "table_3_4_malignant.csv")
    df.groupby("ham_class").agg(images=("image_id", "size"), patients=("patient", "nunique")).to_csv(
        OUT / "table_3_5_mapping.csv")
    (df[["diagnosis", "ham_class"]].drop_duplicates().sort_values(["ham_class", "diagnosis"])
     .to_csv(config.CONFIGS_DIR / "dermacon_mapping.csv", index=False))
    t36 = pd.concat([df.fst_num.value_counts().sort_index().rename(lambda x: f"Fitzpatrick type {int(x)}"),
                     df.mst_num.value_counts().sort_index().rename(lambda x: f"Monk tone {int(x)}")])
    t36.to_csv(OUT / "table_3_6_skin_tone.csv", header=["images"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    df.fst_num.value_counts().sort_index().plot.bar(ax=axes[0], color="sienna"); axes[0].set_title("Fitzpatrick type")
    df.mst_num.value_counts().sort_index().plot.bar(ax=axes[1], color="peru"); axes[1].set_title("Monk skin tone")
    fig.tight_layout(); fig.savefig(OUT / "fig_3_3_skin_tone.png", dpi=150); plt.close(fig)
    df.groupby("group").size().to_csv(OUT / "disease_groups.csv", header=["images"])
    print(t32, t33, t34, sep="\n\n")
    engine.ensure_cached(pd.concat(splits()[0].values()) if config.SMOKE else df, "path", "cache")
    return df


# --------------------------------------------------------------------------- zero-shot
def _fcr_row(name, called, patients, n_boot_seed=0):
    k, n = int(called.sum()), len(called)
    lo, hi = stats.wilson(k, n)
    plo, phi = stats.bootstrap_ci(lambda ix: called[ix].mean(), patients, seed=n_boot_seed)
    return {"model": name, "called_cancer": k, "n": n, "fcr": k / max(n, 1), "wilson": [lo, hi], "patient_ci": [plo, phi]}


def ham_probs(df, key):
    """HAM model + ensemble probabilities (3-view TTA) for DermaCon-IN images."""
    P = {m: engine.cached_predictions(f"dc_{key}_{m}", config.MODELS_DIR / f"{m}.pth", df) for m in config.HAM_MODELS}
    wmap, _ = ham.ensemble_weights()
    P["ensemble"] = sum(w * P[m] for m, w in wmap.items())
    return P


def zero_shot():
    import studies
    s, _ = splits()
    te = s["test"]; allx = pd.concat(s.values()).reset_index(drop=True)
    nn_ = te[~te.neoplasm].reset_index(drop=True)
    P = ham_probs(te, "test")
    called = {m: np.isin(p.argmax(1), stats.MAL)[~te.neoplasm.to_numpy()] for m, p in P.items()}
    rows = [_fcr_row(m, c, nn_.patient.to_numpy()) for m, c in called.items()]

    def study_row(label, runs, cache_col="cache"):
        cs = [np.isin(engine.cached_predictions(f"dc_test_{r}", config.MODELS_DIR / f"{r}.pth", nn_, cache_col,
                                                tta=False).argmax(1), stats.MAL) for r in runs]
        rates = [c.mean() for c in cs]
        ci = stats.bootstrap_ci(lambda ix: np.mean([c[ix].mean() for c in cs]), nn_.patient.to_numpy())
        return {"model": label, "fcr": float(np.mean(rates)), "fcr_std": float(np.std(rates, ddof=1)),
                "per_seed": rates, "patient_ci": ci}
    base_runs = studies.baseline_runs()
    rows.append(study_row("Selected study setup (3 seeds)", base_runs))
    rows.append(study_row("Skin-tone augmented (3 seeds)", [f"study_skin_s{sd}" for sd in config.SEEDS]))
    if (config.MODELS_DIR / "study_crop_s0.pth").exists():
        studies.make_crop_cache(nn_, seg=studies.Segmenter())
        rows.append(study_row("Crop model (3 seeds)", [f"study_crop_s{sd}" for sd in config.SEEDS], "crop_cache"))
    # BCCs called cancer by the study models (text of Section 6.6)
    mal_te = te[te.malignant].reset_index(drop=True)
    extra = {}
    for label, runs in (("study", base_runs), ("skin", [f"study_skin_s{sd}" for sd in config.SEEDS])):
        extra[label] = [int(np.isin(engine.cached_predictions(f"dc_testmal_{r}", config.MODELS_DIR / f"{r}.pth", mal_te,
                                                                tta=False).argmax(1), stats.MAL).sum()) for r in runs]
    json.dump({"rows": rows, "test_malignant_called_by_study_models": extra, "n_test_malignant": len(mal_te)},
              open(OUT / "table_6_3_fcr.json", "w"), indent=2, default=float)
    pd.DataFrame(rows).to_csv(OUT / "table_6_3_fcr.csv", index=False)

    pe = P["ensemble"][~te.neoplasm.to_numpy()]
    top = pd.Series([config.CLASS_NAMES[i] for i in pe.argmax(1)]).value_counts()
    top.to_csv(OUT / "ensemble_top_class_non_neoplastic.csv", header=["images"])
    ce = called["ensemble"]
    grp = []
    for g, ix in nn_.groupby("group").groups.items():
        ix = np.asarray(list(ix)); k, n = int(ce[ix].sum()), len(ix); lo, hi = stats.wilson(k, n)
        grp.append({"group": g, "images": n, "called_cancer": k, "fcr": k / n, "wilson_lo": lo, "wilson_hi": hi})
    grp = pd.DataFrame(grp).sort_values("fcr", ascending=False)
    grp.to_csv(OUT / "table_6_4_groups.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(grp) + 1))
    ax.errorbar(grp.fcr, range(len(grp)), xerr=[grp.fcr - grp.wilson_lo, grp.wilson_hi - grp.fcr], fmt="o", capsize=3)
    ax.set_yticks(range(len(grp))); ax.set_yticklabels(grp.group); ax.invert_yaxis()
    ax.set_xlabel("false-cancer rate (95% Wilson CI)"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_6_5_fcr_groups.png", dpi=150); plt.close(fig)

    tone = []
    for scale, col, bins in (("Fitzpatrick", "fst_num", [("FST III-IV", 3, 4), ("FST V-VI", 5, 6)]),
                             ("Monk", "mst_num", [("MST 1-5", 1, 5), ("MST 6", 6, 6), ("MST 7", 7, 7), ("MST 8-10", 8, 10)])):
        v = nn_[col].to_numpy()
        for name, lo_, hi_ in bins:
            sel = (v >= lo_) & (v <= hi_)
            k, n = int(ce[sel].sum()), int(sel.sum()); lo, hi = stats.wilson(k, n)
            tone.append({"scale": scale, "group": name, "images": n, "fcr": k / max(n, 1), "wilson_lo": lo, "wilson_hi": hi})
    pd.DataFrame(tone).to_csv(OUT / "table_6_5_skin_tone.csv", index=False)

    Pall = ham_probs(allx, "all")
    mapped = te[te.ham_class != "unmapped"]
    Pm = ham_probs(mapped.reset_index(drop=True), "mapped") if len(mapped) else None
    sec = []
    for m in P:
        acc = float((Pm[m].argmax(1) == mapped.ham_class.map(config.CLASS_NAMES.index).to_numpy()).mean()) if Pm else float("nan")
        mal = allx.malignant.to_numpy()
        det = np.isin(Pall[m].argmax(1), stats.MAL)
        k, n = int(det[mal].sum()), int(mal.sum()); lo, hi = stats.wilson(k, n)
        sec.append({"model": m, "mapped_acc_test": acc, "n_mapped_test": len(mapped), "malignant_detected": k / max(n, 1),
                    "malignant_k": k, "malignant_n": n, "wilson_lo": lo, "wilson_hi": hi,
                    "fcr_all_images": float(det[~allx.neoplasm.to_numpy()].mean())})
    pd.DataFrame(sec).to_csv(OUT / "table_6_6_secondary.csv", index=False)
    print(pd.DataFrame(rows)[["model", "fcr"]].to_string(index=False))


# --------------------------------------------------------------------------- gate / two-stage
def gate():
    s, main = splits()
    neo = [i for i, c in enumerate(main) if "neoplas" in c.lower()][0]
    for sd in config.SEEDS:
        engine.train_run(f"gate_s{sd}", "efficientnet_b0", s["train"], s["val"], n_classes=len(main),
                         class_names=main, label_col="main_label", seed=sd, imbalance="sqrt",
                         settings=config.TRAIN_SETTINGS["efficientnet_b0"])
    te, va = s["test"], s["val"]
    nonneo = ~te.neoplasm.to_numpy()
    pe = ham_probs(te, "test")["ensemble"]
    ens_called = np.isin(pe.argmax(1), stats.MAL)
    rows, two_called, thresholds = [], [], {}
    for sd in config.SEEDS:
        ck = config.MODELS_DIR / f"gate_s{sd}.pth"
        pv = engine.cached_predictions(f"gate_s{sd}_val", ck, va)
        pt = engine.cached_predictions(f"gate_s{sd}_test", ck, te)
        vneo = np.sort(pv[va.neoplasm.to_numpy(), neo])[::-1]
        k = int(np.ceil(config.GATE_NEOPLASM_RECALL * len(vneo)))
        t = float(vneo[k - 1]) if len(vneo) else 0.5           # highest t keeping >= 90% of val neoplasms
        thresholds[sd] = t
        passed = pt[:, neo] >= t
        yt = te.main_label.to_numpy()
        tn = te.neoplasm.to_numpy()
        rows.append({"seed": sd, "accuracy": float((pt.argmax(1) == yt).mean()),
                     "macro_f1": float(f1_score(yt, pt.argmax(1), average="macro", labels=list(range(len(main))), zero_division=0)),
                     "val_macro_f1": float(f1_score(va.main_label, pv.argmax(1), average="macro", labels=list(range(len(main))), zero_division=0)),
                     "neoplasm_recall": float((pt.argmax(1)[tn] == neo).mean()) if tn.any() else float("nan"),
                     "threshold": t, "test_neoplasms_passed": f"{int(passed[tn].sum())}/{int(tn.sum())}",
                     "non_neoplastic_passed": float(passed[nonneo].mean())})
        two_called.append((passed & ens_called)[nonneo])
    pd.DataFrame(rows).to_csv(OUT / "table_6_7_gate.csv", index=False)

    pats = te.patient.to_numpy()[nonneo]
    alone = ens_called[nonneo]
    rates = [c.mean() for c in two_called]
    ci_two = stats.bootstrap_ci(lambda ix: np.mean([c[ix].mean() for c in two_called]), pats)
    ci_alone = stats.bootstrap_ci(lambda ix: alone[ix].mean(), pats)
    ci_diff = stats.bootstrap_ci(lambda ix: np.mean([c[ix].mean() for c in two_called]) - alone[ix].mean(), pats)
    mal = te.malignant.to_numpy()
    bcc_alone = int(ens_called[mal].sum())
    bcc_two = []
    for sd in config.SEEDS:
        pt = engine.cached_predictions(f"gate_s{sd}_test", config.MODELS_DIR / f"gate_s{sd}.pth", te)
        bcc_two.append(int(((pt[:, neo] >= thresholds[sd]) & ens_called)[mal].sum()))
    t68 = {"ensemble_alone": {"fcr": float(alone.mean()), "patient_ci": ci_alone, "malignant_called": bcc_alone},
           "two_stage": {"fcr_mean": float(np.mean(rates)), "fcr_std": float(np.std(rates, ddof=1)), "per_seed": rates,
                         "patient_ci": ci_two, "malignant_called_per_seed": bcc_two},
           "change": {"points": float(np.mean(rates) - alone.mean()), "patient_ci": ci_diff},
           "n_test_malignant": int(mal.sum()), "thresholds": thresholds}

    # gate on HAM10000 dermoscopic validation images (best seed by DermaCon val macro-F1)
    best = max(rows, key=lambda r: r["val_macro_f1"])["seed"]
    hv = ham.splits()["val"]
    ph = engine.cached_predictions(f"gate_s{best}_hamval", config.MODELS_DIR / f"gate_s{best}.pth", hv)
    hp = ph[:, neo] >= thresholds[best]
    t68["gate_on_ham_val"] = {"seed": best, "passed_all": float(hp.mean()),
                              "passed_malignant": float(hp[hv.dx.isin(config.MALIGNANT_CLASSES).to_numpy()].mean())}
    json.dump(t68, open(OUT / "table_6_8_two_stage.json", "w"), indent=2, default=float)

    per = []
    for c in sorted(te.main_class.unique()):
        sel = (te.main_class == c).to_numpy()[nonneo]
        if sel.any():
            per.append({"main_class": c, "alone": alone[sel].mean(), "two_stage": np.mean([tc[sel].mean() for tc in two_called])})
    per = pd.DataFrame(per)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(per))
    ax.bar(x - .2, per.alone, .4, label="ensemble alone"); ax.bar(x + .2, per.two_stage, .4, label="two-stage (mean of 3 seeds)")
    ax.set_xticks(x); ax.set_xticklabels(per.main_class, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("false-cancer rate"); ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_6_6_two_stage.png", dpi=150); plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False)); print(json.dumps(t68, indent=1, default=float))


# --------------------------------------------------------------------------- DDI
def ddi():
    meta = config.DDI_DIR / "ddi_metadata.csv"
    out = OUT / "table_6_9_ddi.json"
    if not meta.exists():
        json.dump({"status": "PENDING: DDI", "reason": "data/ddi/ddi_metadata.csv not found"}, open(out, "w"), indent=2)
        print("DDI not available - Table 6.9 stays [PENDING: DDI]")
        return
    df = pd.read_csv(meta)
    files = {p.name: p for p in config.DDI_DIR.rglob("*.png")} | {p.name: p for p in config.DDI_DIR.rglob("*.jpg")}
    df["path"] = df.DDI_file.map(lambda f: str(files.get(f, "")))
    df = df[df.path != ""].reset_index(drop=True)
    df["image_id"] = "ddi_" + df.DDI_file.str.replace(r"\W", "_", regex=True)
    df["cache"] = df.image_id.map(lambda i: str(config.DATA_DIR / "processed_ddi" / f"{i}.jpg"))
    engine.ensure_cached(df, "path", "cache")
    pe = ham_probs(df, "ddi")["ensemble"]
    y = df.malignant.astype(str).str.lower().isin(["true", "1"]).to_numpy()
    called = np.isin(pe.argmax(1), stats.MAL)
    rows = []
    for g, sel in [("FST I-II", df.skin_tone == 12), ("FST III-IV", df.skin_tone == 34), ("FST V-VI", df.skin_tone == 56),
                   ("All", np.ones(len(df), bool))]:
        sel = np.asarray(sel)
        ys, cs = y[sel], called[sel]
        se = stats.wilson(int(cs[ys].sum()), int(ys.sum())); sp = stats.wilson(int((~cs[~ys]).sum()), int((~ys).sum()))
        rows.append({"group": g, "images": int(sel.sum()), "malignant": int(ys.sum()),
                     "sensitivity": cs[ys].mean() if ys.any() else np.nan, "sens_ci": se,
                     "specificity": (~cs[~ys]).mean() if (~ys).any() else np.nan, "spec_ci": sp,
                     "auc": stats._safe_auc(ys, pe[sel][:, stats.MAL].sum(1))})
    json.dump({"status": "done", "rows": rows}, open(out, "w"), indent=2, default=float)
    print(pd.DataFrame(rows).to_string(index=False))

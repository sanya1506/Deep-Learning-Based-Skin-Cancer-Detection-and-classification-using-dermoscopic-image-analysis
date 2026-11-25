"""Write Design_Project_13_Report_updated.docx from the merged copy of the original report.

Rules: front matter (P0-P67) untouched; text is changed only inside existing runs / table cells, so
fonts, styles, spacing and the invisible characters at paragraph ends stay as they are.
Every change is logged to changes.md.
Run:  .venv/Scripts/python kaggle_runs/report/update_report.py
"""
import copy
import io
import json
import re
import sys
from pathlib import Path

import docx
import pandas as pd
from docx.oxml.ns import qn
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
S = ROOT / "results" / "study"
V2 = ROOT / "results" / "v2"
SRC = HERE / "merged.docx"
OUT = HERE / "Design_Project_13_Report_updated.docx"

D = docx.Document(str(SRC))
BODY = list(D.element.body.iterchildren())
P = [docx.text.paragraph.Paragraph(e, D) for e in BODY if e.tag == qn("w:p")]
T = [docx.table.Table(e, D) for e in BODY if e.tag == qn("w:tbl")]
LOG = []
FRONT_MATTER_LAST = 67


# --------------------------------------------------------------------------- formatting
def f3(x): return f"{x:.3f}"
def pct(x, d=1): return f"{100 * x:.{d}f}%"
def num(n): return f"{int(n):,}"
def ci_pct(lo, hi, d=1): return f"{100 * lo:.{d}f}–{100 * hi:.{d}f}"
def pm(m, s): return f"{m:.3f} ± {s:.3f}"
def msci(d):  # {"mean","std","ci"} -> "0.671 ± 0.018 [0.612, 0.718]"
    return f"{d['mean']:.3f} ± {d['std']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"
def signed(x, d=4): return f"{x:+.{d}f}".replace("-", "−")
def diffci(m, lo, hi): return f"{signed(m)} [{signed(lo)}, {signed(hi)}]"


# --------------------------------------------------------------------------- edit helpers
def _guard(i):
    assert i > FRONT_MATTER_LAST, f"P{i} is front matter"


def cell(t, r, c, text, note=None):
    tb = T[t]
    cl = tb.rows[r].cells[c]
    old = cl.text
    if old == text:
        return
    runs = [rn for p in cl.paragraphs for rn in p.runs]
    if runs:
        runs[0].text = text
        for rn in runs[1:]:
            rn.text = ""
    else:
        cl.paragraphs[0].add_run(text)
    LOG.append(f"Table T{t} row {r} col {c}: '{old}' → '{text}'" + (f" ({note})" if note else ""))


def row(t, r, values):
    for c, v in enumerate(values):
        if v is not None:
            cell(t, r, c, v)


def replace(i, old, new, note=None):
    """Replace text inside paragraph i, run-safe (works across run boundaries)."""
    _guard(i)
    p = P[i]
    runs = p.runs
    full = "".join(r.text for r in runs)
    k = full.find(old)
    assert k >= 0, f"P{i}: not found: {old!r}\n in: {full[:300]!r}"
    # locate start/end runs
    pos, start = 0, None
    for ri, r in enumerate(runs):
        L = len(r.text)
        if start is None and k < pos + L:
            start, off = ri, k - pos
        if start is not None and k + len(old) <= pos + L:
            end, endoff = ri, k + len(old) - pos
            break
        pos += L
    if start == end:
        r = runs[start]; r.text = r.text[:off] + new + r.text[endoff:]
    else:
        runs[start].text = runs[start].text[:off] + new
        for r in runs[start + 1:end]:
            r.text = ""
        runs[end].text = runs[end].text[endoff:]
    LOG.append(f"P{i}: '{old}' → '{new}'" + (f" ({note})" if note else ""))


def image(par_index, png_path, note=""):
    """Swap the picture in paragraph par_index for png_path; keep the width, set height by aspect ratio."""
    _guard(par_index)
    el = P[par_index]._p
    blip = next(el.iter(qn("a:blip")))
    rid = blip.get(qn("r:embed"))
    part = D.part.related_parts[rid]
    data = Path(png_path).read_bytes()
    part._blob = data
    w, h = Image.open(io.BytesIO(data)).size
    ext = el.find(".//" + qn("wp:extent"))
    cx = int(ext.get("cx")); cy = int(round(cx * h / w))
    ext.set("cy", str(cy))
    for a in el.iter(qn("a:ext")):
        if a.get("cx") is not None:
            a.set("cx", str(cx)); a.set("cy", str(cy))
    LOG.append(f"P{par_index}: figure replaced by {Path(png_path).relative_to(ROOT).as_posix()} "
               f"(width kept {cx / 914400:.2f} in, height {cy / 914400:.2f} in) {note}")


# =========================================================================== A: HAM10000
t61 = pd.read_csv(S / "ham" / "table_6_1_models.csv")
t62 = pd.read_csv(S / "ham" / "table_6_2_per_class.csv")
t31 = pd.read_csv(S / "ham" / "table_3_1_split.csv").set_index("dx")
t616 = pd.read_csv(S / "ham" / "table_6_16_missed_cancers.csv").set_index("class")
ens = json.load(open(S / "ham" / "ensemble_summary.json"))
cm = pd.read_csv(S / "ham" / "confusion_matrix.csv", index_col=0)
runs = {m: json.load(open(ROOT / "results" / "runs" / m / "summary.json"))
        for m in ("efficientnet_b0", "efficientnet_b0_long", "resnet50", "densenet121", "convnext_tiny", "swin_t")}
timings = json.load(open(S / "timings.json"))

# Table 3.1 (T7)
for r, dx in enumerate(["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc", "images", "lesions", "malignant"], start=1):
    x = t31.loc[dx]
    row(7, r, [None, num(x.train), num(x.val), num(x.test), num(x.total)])

# Table 6.1 (T22) - same row order as the csv
for r, x in enumerate(t61.itertuples(), start=1):
    row(22, r, [None, f"{x.params_m:.1f}", f3(x.val_f1), f3(x.accuracy), f3(x.balanced_accuracy), f3(x.macro_f1),
                f3(x.melanoma_recall), f3(x.malignant_sensitivity), f3(x.macro_auc), f"{x.ms_per_img:.0f}"])
sel = t61[t61.key == "ens_weighted"].iloc[0]
eq = t61[t61.key == "ens_equal"].iloc[0]
b3 = t61[t61.key == "ens_best3"].iloc[0]

# Section 6.1 text (P243): the old text said the equal-weight ensemble was better on test; it no longer is.
replace(243, "The weighted ensemble was chosen because it had the validation macro-F1. The validation macro-F1 was 0.760. "
             "On the test set the weighted ensemble achieved an accuracy of 0.838. The weighted ensemble also had a macro-F1 of 0.747. "
             "The equal-weight ensemble had a better result on the test set. The macro-F1, for the equal-weight ensemble was 0.750. "
             "The melanoma recall for the equal-weight ensemble was 0.635. The decision was not changed after seeing these results. "
             "Choosing based on test results would introduce bias.",
        f"The weighted ensemble was chosen because it had the highest validation macro-F1 ({f3(sel.val_f1)}, against "
        f"{f3(b3.val_f1)} for the best three models and {f3(eq.val_f1)} for equal weights). On the test set the weighted ensemble "
        f"achieved an accuracy of {f3(sel.accuracy)} and a macro-F1 of {f3(sel.macro_f1)}, with melanoma recall {f3(sel.melanoma_recall)}. "
        f"The best-three ensemble had a slightly higher test accuracy ({f3(b3.accuracy)}) but a lower macro-F1 ({f3(b3.macro_f1)}). "
        f"EfficientNet-B0 and its long run ended with identical results: both reached their best validation macro-F1 at epoch 10, "
        f"so the longer schedule changed nothing. The choice was made on validation data only; choosing based on test results "
        f"would introduce bias.",
        note="rewritten: the equal-weight ensemble is no longer better on test")

# Table 6.2 (T23)
for r, x in enumerate(t62.itertuples(), start=1):
    row(23, r, [None, f3(x.precision), f3(x.recall), f3(x.specificity), f3(x.f1), f3(x.auc), num(x.test_images)])
mel = t62[t62["class"] == "mel"].iloc[0]
m616 = t616.loc["mel"]
replace(250, "Melanoma is the class: recall 0.563 F1 0.597. The ensemble misclassified 72 of 167 test melanomas (43.1%, 95% CI 35.8–50.7%) "
             "calling them mostly nevus or benign keratosis.",
        f"Melanoma is the weakest class: recall {f3(mel.recall)}, F1 {f3(mel.f1)}. The ensemble called {int(m616.called_benign)} of "
        f"{int(m616.test_images)} test melanomas benign ({pct(m616.miss_rate)}, 95% CI {ci_pct(m616.wilson_lo, m616.wilson_hi)}%), "
        f"mostly nevus ({int(cm.loc['mel', 'nv'])}) or benign keratosis ({int(cm.loc['mel', 'bkl'])}).")

# Table 6.16 (T37)
for r, c in enumerate(["mel", "bcc", "akiec"], start=1):
    x = t616.loc[c]
    row(37, r, [None, num(x.test_images), num(x.called_benign), pct(x.miss_rate), ci_pct(x.wilson_lo, x.wilson_hi)])

# Table 5.4 (T20)
for r, m in enumerate(["efficientnet_b0", "efficientnet_b0_long", "resnet50", "densenet121", "convnext_tiny", "swin_t"], start=1):
    row(20, r, [None, str(runs[m]["epochs_run"]), f"{runs[m]['train_minutes']:.1f}"])

# Figures from Part A
image(246, S / "ham" / "fig_6_1_models.png")
image(251, S / "ham" / "fig_6_2_confusion.png")
image(253, S / "ham" / "fig_6_3_roc.png")
image(255, S / "ham" / "fig_6_4_gradcam.png")
image(313, S / "ham" / "fig_6_9_missed.png")

# =========================================================================== B: supporting studies
imb = json.load(open(S / "studies" / "table_6_10_imbalance.json"))
imb_sel = json.load(open(S / "studies" / "imbalance_selected.json"))
skin = json.load(open(S / "studies" / "table_6_11_skin_tone.json"))
crop = json.load(open(S / "studies" / "table_6_13_crop.json"))
seg = pd.read_csv(S / "studies" / "table_6_12_segmentation.csv", encoding="utf-8")

labels = {"none": "a) None", "sqrt": "b) √ weights (default)", "inverse": "c) Inverse frequency",
          "focal": "d) Focal loss", "sampler": "e) Weighted sampler"}
byk = {r["strategy"]: r for r in imb}
for r, k in enumerate(["none", "sqrt", "inverse", "focal", "sampler"], start=1):
    x = byk[k]
    lab = labels[k] + (" (selected)" if k == imb_sel["selected"] else "")
    row(31, r, [lab, f3(x["val_macro_f1"]), f3(x["val_mel_recall"]), msci(x["test_macro_f1"]), msci(x["test_melanoma_recall"]),
                msci(x["test_malignant_sensitivity"]), pm(x["test_accuracy"]["mean"], x["test_accuracy"]["std"])])

v = imb_sel["val"]
best_test = max(imb, key=lambda r: r["test_macro_f1"]["mean"])
sel_r = byk[imb_sel["selected"]]
replace(290, "The rule chose the sampler, which gave a validation macro-F1 of 0.684. In the test the default project weights, which were square-rooted "
             "produced a score of 0.695 versus 0.670 but the confidence intervals overlap so the test set does not show a real difference. "
             "I did not change the choice after looking at the test results. Using inverse-frequency weights gave the highest melanoma recall of 0.739 "
             "and the highest malignant sensitivity but it reduced accuracy to 0.761. The original single-seed EfficientNet-B0, with a macro-F1 of 0.721 "
             "without augmentation scored higher, than the three-seed mean of the same recipe showing how much one run can vary.",
        f"The rule chose focal loss. Its validation macro-F1 ({f3(v['focal'][0])}) was within 0.005 of no weighting ({f3(v['none'][0])}), "
        f"so the tie-break applied, and focal loss had the higher validation melanoma recall ({f3(v['focal'][1])} vs {f3(v['none'][1])}). "
        f"On the test set the weighted sampler had the highest macro-F1 ({f3(best_test['test_macro_f1']['mean'])} versus "
        f"{f3(sel_r['test_macro_f1']['mean'])} for focal loss), but the confidence intervals overlap, so the test set does not show a real "
        f"difference. The choice was not changed after looking at the test results. The weighted sampler also gave the highest melanoma recall "
        f"({f3(byk['sampler']['test_melanoma_recall']['mean'])}) and inverse-frequency weights the highest malignant sensitivity "
        f"({f3(byk['inverse']['test_malignant_sensitivity']['mean'])}), but both reduced accuracy to about "
        f"{f3(byk['inverse']['test_accuracy']['mean'])}. The single EfficientNet-B0 of Table 6.1 (seed 42, with test-time augmentation) "
        f"reached a macro-F1 of {f3(t61[t61.key == 'efficientnet_b0'].iloc[0].macro_f1)}, higher than the three-seed mean of the same recipe "
        f"({f3(byk['sqrt']['test_macro_f1']['mean'])}), showing how much one run can vary.",
        note="rewritten: the rule now selects focal loss, not the weighted sampler")
image(290, S / "studies" / "fig_6_7_imbalance.png")


def study_table(t, d):
    names = ["accuracy", "balanced_accuracy", "macro_f1", "melanoma_recall", "malignant_sensitivity", "malignant_specificity", "malignant_auc"]
    for r, k in enumerate(names, start=1):
        a, b, df_ = d["a"][k], d["b"][k], d["diff"][k]
        row(t, r, [None, msci(a), msci(b), diffci(df_["diff"], df_["ci"][0], df_["ci"][1])])


study_table(32, skin)
study_table(34, crop)
row(33, 1, [None, seg.iloc[0].dice.replace("�", "±").replace("\xb1", "±"),
            "–".join(f"{float(x):.3f}" for x in seg.iloc[0].dice_ci.strip("()").split(",")),
            seg.iloc[0].iou.replace("�", "±").replace("\xb1", "±"),
            "–".join(f"{float(x):.3f}" for x in seg.iloc[0].iou_ci.strip("()").split(","))])
row(33, 2, [None, f"{float(seg.iloc[1].dice):.3f}", "–".join(f"{float(x):.3f}" for x in seg.iloc[1].dice_ci.strip("()").split(",")),
            f"{float(seg.iloc[1].iou):.3f}", "–".join(f"{float(x):.3f}" for x in seg.iloc[1].iou_ci.strip("()").split(","))])
image(209, S / "studies" / "fig_4_4_masks_crops.png", "(predicted masks of the new U-Net)")

# Table 5.5 (T21) - rows from Parts B (and C, below)
T55 = {"14_imbalance_study": 1, "15_skin_tone_augmentation": 2, "16_segmentation": 3, "17_robustness": 4,
       "18_explainability": 5, "10_indian_data_audit": 6, "11_indian_zero_shot": 7, "12_stage1_gate": 8}
# timings.json is merged by Part C; until then take Part B's file
for k, r in T55.items():
    if k in timings:
        cell(21, r, 2, f"{timings[k]:.0f}")

# =========================================================================== text: missing numbers, A/B-dependent sentences
# Abstract (P69): HAM10000 numbers from Part A; DermaCon-IN numbers from the report's own tables (3.2, 6.3, 6.8)
replace(69, "macro-F lands at ., accuracy at .%.", f"macro-F1 lands at {f3(sel.macro_f1)}, accuracy at {pct(sel.accuracy)}.")
replace(69, "DermaCon-IN: , clinical photos", "DermaCon-IN: 5,450 clinical photos")
replace(69, "On  non-cancer test images", "On 802 non-cancer test images")
replace(69, " .% of the time", " 11.0% of the time")
replace(69, "That cuts the-cancer rate down to .%.", "That cuts the false-cancer rate down to 5.6%.")
# Chapter 1 citation years (from the reference list) and dataset name
replace(81, "(Esteva et al., )", "(Esteva et al., 2017)")
replace(81, "(Tschandl et al., )", "(Tschandl et al., 2018)")
replace(81, "(Codella et al., )", "(Codella et al., 2019)")
replace(81, "(Adamson and Smith, ; Kinyanjui et al., )", "(Adamson and Smith, 2018; Kinyanjui et al., 2020)")
replace(81, "(Groh et al., ; Daneshjou et al., )", "(Groh et al., 2021; Daneshjou et al., 2022)")
replace(83, "on HAM00,", "on HAM10000,")
replace(83, "original HAM00 pipeline", "original HAM10000 pipeline")
replace(83, "(Madarkar et al., )", "(Madarkar et al., 2025)")
replace(98, "HAM10 (Tschandl", "HAM10000 (Tschandl")
replace(100, "HAM10, and DermIS", "HAM10000, and DermIS")

# Hardware (Table 5.1, T17) and the sentence under it
row(17, 1, [None, "Kaggle cloud notebook, 4 virtual CPU cores"])
row(17, 2, [None, "29 GB RAM"])
row(17, 3, [None, "NVIDIA Tesla T4, 16 GB (CUDA, PyTorch)"])
row(17, 4, [None, "Linux (Kaggle notebook container)"])
replace(223, "so it also runs on a machine with an NVIDIA GPU. No cloud service was used.",
        "so the same code also runs on Apple silicon or on a CPU-only machine. The models were trained and evaluated in "
        "Kaggle cloud notebooks; the web application runs locally.")
# Software (Table 5.2, T18): versions of the Kaggle runs as printed in the run logs
row(18, 1, [None, "3.13"])
row(18, 2, [None, "2.11.0 (CUDA 13.0 build)"])

# Small text gaps in Chapter 6 (no new numbers)
replace(258, "802 -neoplastic", "802 non-neoplastic")
replace(281, "(, with seed 2 the best result", "(seed 2, the best result")

# 6.6 skin tone - HAM10000 part from Part B; DermaCon-IN part unchanged
sk = skin["diff"]
replace(294, "On HAM10000 the augmentation did not make a difference; every interval, except for malignant specificity includes zero.",
        f"On HAM10000 the augmentation did not help: melanoma recall fell by {abs(sk['melanoma_recall']['diff']):.3f} "
        f"(interval {signed(sk['melanoma_recall']['ci'][0], 3)} to {signed(sk['melanoma_recall']['ci'][1], 3)}) while malignant "
        f"specificity rose slightly ({signed(sk['malignant_specificity']['diff'], 3)}); every other interval includes zero.",
        note="rewritten: melanoma recall now falls significantly")

# 6.7 segmentation and crops - HAM10000 part from Part B
cd_ = crop["diff"]
dice_u = float(seg.iloc[0].dice.split()[0]); dice_o = float(seg.iloc[1].dice)
replace(300, "The U-Net does a job of segmenting HAM10000 lesions (Dice score 0.943 compared to 0.652 for Otsu thresholding). "
             "Training on cropped images improved accuracy, macro-F1 (an increase of 0.0579) and malignant specificity with intervals that do not include zero. "
             "However it reduced sensitivity (a decrease of 0.0483 with intervals also not including zero) and melanoma recall. "
             "Cropping replaces missed cancers with false alarms but this is not always a benefit for a screening tool.",
        f"The U-Net does a good job of segmenting HAM10000 lesions (Dice score {dice_u:.3f} compared to {dice_o:.3f} for Otsu thresholding). "
        f"Training on cropped images improved macro-F1 ({signed(cd_['macro_f1']['diff'])}), melanoma recall ({signed(cd_['melanoma_recall']['diff'])}) "
        f"and malignant sensitivity ({signed(cd_['malignant_sensitivity']['diff'])}), all with intervals that do not include zero, "
        f"while malignant specificity fell ({signed(cd_['malignant_specificity']['diff'])}) and accuracy did not change clearly. "
        f"Cropping trades some extra false alarms for fewer missed cancers, which suits a screening tool.",
        note="rewritten: cropping now raises cancer sensitivity instead of lowering it")
replace(300, "On DermaCon-, IN", "On DermaCon-IN")

# Discussion / limitations / conclusion
replace(316, "misses about 43.1% of melanomas", f"misses about {pct(m616.miss_rate)} of melanomas")
replace(317, "cropping improved the headline metric while lowering cancer sensitivity",
        "cropping raised cancer sensitivity at the cost of specificity", note="meaning changed with the new crop result")
replace(323, "had a score, on the test compared to the default", "had a lower score on the test set than the default")
replace(323, "reported ass without", "reported as is, without")
replace(329, "(test macro-F1 0.747)", f"(test macro-F1 {f3(sel.macro_f1)})")
replace(329, "of -tumour Indian", "of non-tumour Indian")
replace(329, "A gate trained on data lowered", "A gate trained on Indian data lowered")
replace(329, "a trade-off from lesion cropping", "a gain in cancer sensitivity at some cost in specificity from lesion cropping")
replace(329, "do not mean results, for Indian patients", "do not mean good results for Indian patients")

# =========================================================================== v2: new Section 6.10
import numpy as np
from sklearn.metrics import roc_auc_score

v2m = json.load(open(V2 / "ensemble" / "ensemble_metrics.json"))
v2c = v2m["test_ensemble_calibrated"]; v2s = v2m["test_ensemble_calibrated_sens90"]
v2run = {r["model"]: r for r in v2m["individual_runs"]}
v2w = v2m["ensemble_members"]
v2p = pd.read_csv(V2 / "ensemble" / "test_predictions.csv")
CL = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
v2acc = float((v2p.pred == v2p.dx).mean())
_les = v2p.lesion_id.values; _u = np.unique(_les); _ix = {l: np.where(_les == l)[0] for l in _u}
_rng = np.random.default_rng(0); _bs = []
for _ in range(1000):
    s = np.concatenate([_ix[l] for l in _rng.choice(_u, len(_u))]); _bs.append((v2p.pred.values[s] == v2p.dx.values[s]).mean())
v2lo, v2hi = np.percentile(_bs, [2.5, 97.5])
assert abs(v2acc - v2c["accuracy"]) < 1e-9
v2cls = {}
for c in CL:
    t = (v2p.dx == c).values; p = (v2p.pred == c).values
    prec = (t & p).sum() / p.sum(); rec = (t & p).sum() / t.sum(); spec = (~t & ~p).sum() / (~t).sum()
    v2cls[c] = dict(precision=prec, recall=rec, specificity=spec, f1=2 * prec * rec / (prec + rec),
                    auc=roc_auc_score(t, v2p["p_" + c]), n=int(t.sum()))

ANCHOR = P[315]._p          # "6.10 Discussion" - the new section goes right before it


def _set_single_text(el, text):
    """Keep the first text run of a cloned paragraph (its formatting), drop the other runs."""
    runs = [r for r in el.iter(qn("w:r"))]
    keep = next(r for r in runs if r.find(qn("w:t")) is not None and r.find(qn("w:drawing")) is None)
    for r in runs:
        if r is not keep:
            r.getparent().remove(r)
    for t in keep.findall(qn("w:t"))[1:]:
        keep.remove(t)
    tt = keep.find(qn("w:t")); tt.text = text; tt.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def new_par(template_i, text):
    el = copy.deepcopy(P[template_i]._p); _set_single_text(el, text); ANCHOR.addprevious(el)
    LOG.append(f"NEW paragraph (style of P{template_i}) before 'Discussion': {text[:90]}…")
    return el


def new_table(template_t, rows):
    el = copy.deepcopy(T[template_t]._tbl); ANCHOR.addprevious(el)
    tb = docx.table.Table(el, P[315]._parent)
    for r, vals in enumerate(rows, start=1):
        for c, v in enumerate(vals):
            cl = tb.rows[r].cells[c]; rs = [x for p in cl.paragraphs for x in p.runs]
            rs[0].text = v
            for x in rs[1:]:
                x.text = ""
    LOG.append(f"NEW table (layout of T{template_t}) before 'Discussion': v2 per-class results")


def new_figure(template_i, png):
    el = copy.deepcopy(P[template_i]._p)
    rid, _img = D.part.get_or_add_image(str(png))
    next(el.iter(qn("a:blip"))).set(qn("r:embed"), rid)
    w, h = Image.open(png).size
    ext = el.find(".//" + qn("wp:extent")); cx = int(ext.get("cx")); cy = int(round(cx * h / w)); ext.set("cy", str(cy))
    for a in el.iter(qn("a:ext")):
        if a.get("cx") is not None:
            a.set("cx", str(cx)); a.set("cy", str(cy))
    dp = el.find(".//" + qn("wp:docPr")); dp.set("id", "9010"); dp.set("name", "Figure 6.10")
    ANCHOR.addprevious(el)
    LOG.append(f"NEW figure before 'Discussion': {Path(png).relative_to(ROOT).as_posix()}")


H2, BODYP, TCAP, FIGP, FCAP = 306, 316, 307, 313, 314   # templates: heading 2, body text, table caption, figure, figure caption
new_par(H2, "6.10 Improved pipeline (v2)")
new_par(BODYP,
        "To see how far the HAM10000 result can be pushed, a second pipeline (v2) was trained separately. It uses three stronger "
        "backbones at 384 × 384 pixels: EfficientNet-B4 (Noisy Student), EfficientNetV2-S (ImageNet-21k) and ConvNeXt-Small "
        "(ImageNet-22k). The lesion is first cut out with a U-Net trained on the ISIC 2018 segmentation masks, the patient's age, sex "
        "and body site are given to the model together with the image, and about 16,000 extra dermoscopic images from ISIC 2019 are "
        "added to the training data. Every HAM10000 image and every image of a validation or test lesion was removed from this extra "
        "data, so the test split stays free of leakage. Each model was trained for 18 epochs and predicts with 8-view test-time "
        "augmentation. The ensemble weights, a small per-class bias and the malignant threshold were all fitted on the validation set; "
        f"validation gave ConvNeXt-Small a weight of zero, so the final ensemble combines EfficientNet-B4 "
        f"({v2w['tf_efficientnet_b4_meta_isic']:.2f}) and EfficientNetV2-S ({v2w['tf_efficientnetv2_s_meta_isic']:.2f}). "
        f"Training took {json.load(open(V2 / 'run_settings.json'))['hours']:.1f} hours on Kaggle.")
new_par(BODYP,
        f"On the same HAM10000 test split (1,502 images) the v2 ensemble reaches an accuracy of {100 * v2acc:.2f}% "
        f"(95% CI {100 * v2lo:.1f}–{100 * v2hi:.1f}%), a macro-F1 of {v2c['macro_f1']:.3f} and a benign-vs-malignant accuracy of "
        f"{pct(v2c['bin_accuracy'])} (Table 6.17). Alone, the members reach {pct(v2run['tf_efficientnet_b4_meta_isic']['test_acc'])} "
        f"(EfficientNet-B4), {pct(v2run['tf_efficientnetv2_s_meta_isic']['test_acc'])} (EfficientNetV2-S) and "
        f"{pct(v2run['convnext_small_meta_isic']['test_acc'])} (ConvNeXt-Small). Because v2 uses extra ISIC 2019 training data and "
        f"384-pixel inputs, it is not directly comparable to the six 224-pixel models of Table 6.1.")
new_par(TCAP, "Table 6.17: Per-class test results of the v2 ensemble (HAM10000 test split)")
new_table(23, [[c, f3(x["precision"]), f3(x["recall"]), f3(x["specificity"]), f3(x["f1"]), f3(x["auc"]), num(x["n"])]
               for c, x in v2cls.items()])
new_par(BODYP,
        f"Most of the gain comes from the common classes: nevus recall rises to {f3(v2cls['nv']['recall'])} and basal cell carcinoma "
        f"recall to {f3(v2cls['bcc']['recall'])}, while melanoma recall ({f3(v2cls['mel']['recall'])}) stays close to that of the "
        f"selected ensemble ({f3(mel.recall)}). At the default threshold the malignant sensitivity is {pct(v2c['bin_sensitivity'])} at "
        f"{pct(v2c['bin_specificity'])} specificity; at the threshold chosen on validation for 90% sensitivity it is "
        f"{pct(v2s['bin_sensitivity'])} at {pct(v2s['bin_specificity'])} specificity. The v2 pipeline was not evaluated on "
        f"DermaCon-IN. Figure 6.10 shows Grad-CAM maps of the v2 ensemble.")
new_figure(FIGP, V2 / "ensemble" / "gradcam_examples.png")
new_par(FCAP, "Figure 6.10: Grad-CAM maps of the v2 ensemble on HAM10000 test images")

# renumber the following sections
replace(315, "6.10 Discussion", "6.11 Discussion")
replace(318, "6.11 Limitations", "6.12 Limitations")
# conclusion: one sentence on v2
replace(329, "and built a web application around them.",
        f"and built a web application around them. A separately trained improved pipeline (v2: 384-pixel inputs, extra ISIC 2019 "
        f"data, U-Net crops and patient metadata) reached {100 * v2acc:.2f}% accuracy on the same test split, though not a higher "
        f"melanoma recall.")


def clone_row(t, after_text, values):
    tb = T[t]
    src = next(r for r in tb.rows if any(after_text in c.text for c in r.cells))
    el = copy.deepcopy(src._tr); src._tr.addnext(el)
    nr = docx.table._Row(el, tb)
    for c, v in enumerate(values):
        if v is None:
            continue
        rs = [x for p in nr.cells[c].paragraphs for x in p.runs]
        rs[0].text = v
        for x in rs[1:]:
            x.text = ""
    LOG.append(f"NEW row in T{t} after '{after_text}': {values}")


# contents, list of tables, list of figures (page numbers are set afterwards from Word)
clone_row(2, "Explanation quality and missed cancers", [None, "Improved pipeline (v2)", None])
clone_row(3, "Cancers missed by the selected ensemble", ["6.17", "Per-class test results of the v2 ensemble (HAM10000 test split)", None])
clone_row(4, "Most confident missed cancers", ["6.10", "Grad-CAM maps of the v2 ensemble on HAM10000 test images", None])

# --------------------------------------------------------------------------- save
if __name__ == "__main__":
    D.save(str(OUT))
    (HERE / "changes.md").write_text("\n".join(LOG), encoding="utf-8")
    print(len(LOG), "changes ->", OUT.name)

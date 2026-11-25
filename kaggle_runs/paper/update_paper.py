"""Copy of 'fixed conference.docx' with the new Part A/B numbers; Part C numbers kept; format unchanged."""
import json
import re
from pathlib import Path

import docx
import pandas as pd
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
S = ROOT / "results" / "study"
D = docx.Document(str(HERE / "merged.docx"))
P = [docx.text.paragraph.Paragraph(e, D) for e in D.element.body.iterchildren() if e.tag == qn("w:p")]
T = [docx.table.Table(e, D) for e in D.element.body.iterchildren() if e.tag == qn("w:tbl")]
LOG = []


def f3(x): return f"{x:.3f}"
def pct(x): return f"{100 * x:.1f}"
def pm(d): return f"{d['mean']:.3f} ± {d['std']:.3f}"
def sg(x): return f"{x:+.3f}".replace("-", "−")
def dci(d): return f"{sg(d['diff'])} [{sg(d['ci'][0])}, {sg(d['ci'][1])}]"


def cell(t, r, c, text, bold=None):
    cl = T[t].rows[r].cells[c]
    rs = [x for p in cl.paragraphs for x in p.runs]
    old = cl.text
    rs[0].text = text
    for x in rs[1:]:
        x.text = ""
    if bold is not None:
        for x in rs:
            x.bold = bold
    if old != text or bold is not None:
        LOG.append(f"Table {t} r{r} c{c}: '{old}' → '{text}'" + ("" if bold is None else f" (bold={bold})"))


def replace(i, old, new):
    p = P[i]; runs = p.runs; full = "".join(r.text for r in runs); k = full.find(old)
    assert k >= 0, f"P{i}: not found {old!r}"
    pos, start = 0, None
    for ri, r in enumerate(runs):
        L = len(r.text)
        if start is None and k < pos + L:
            start, off = ri, k - pos
        if start is not None and k + len(old) <= pos + L:
            end, endoff = ri, k + len(old) - pos; break
        pos += L
    if start == end:
        runs[start].text = runs[start].text[:off] + new + runs[start].text[endoff:]
    else:
        runs[start].text = runs[start].text[:off] + new
        for r in runs[start + 1:end]:
            r.text = ""
        runs[end].text = runs[end].text[endoff:]
    LOG.append(f"P{i}: '{old}' → '{new}'")


t61 = pd.read_csv(S / "ham" / "table_6_1_models.csv").set_index("key")
t616 = pd.read_csv(S / "ham" / "table_6_16_missed_cancers.csv").set_index("class")
imb = {r["strategy"]: r for r in json.load(open(S / "studies" / "table_6_10_imbalance.json"))}
sel = json.load(open(S / "studies" / "imbalance_selected.json"))
skin = json.load(open(S / "studies" / "table_6_11_skin_tone.json"))["diff"]
crop = json.load(open(S / "studies" / "table_6_13_crop.json"))["diff"]
seg = pd.read_csv(S / "studies" / "table_6_12_segmentation.csv", encoding="utf-8")
w, eq = t61.loc["ens_weighted"], t61.loc["ens_equal"]
mel = t616.loc["mel"]

# ---------------------------------------------------------------- Table I (T3)
for r, k in enumerate(["efficientnet_b0", "efficientnet_b0_long", "resnet50", "densenet121", "convnext_tiny", "swin_t",
                       "ens_equal", "ens_weighted"], start=1):
    x = t61.loc[k]
    for c, v in enumerate([x.accuracy, x.macro_f1, x.melanoma_recall, x.malignant_sensitivity, x.macro_auc], start=1):
        cell(3, r, c, f3(v))
replace(54, "The weighted ensemble was chosen because it had the validation macro-F1 score of 0.760. In the test set the equal-weight "
            "ensemble achieved a higher macro-F1 of 0.750 but the weighted ensemble remained the choice we made. Unfortunately, the weighted "
            "ensemble still misclassified 43.1 % of the test melanomas with a 95 % confidence interval of 35.8 %, to 50.7 % labeling them as benign.",
        f"The weighted ensemble was chosen because it had the highest validation macro-F1 ({f3(w.val_f1)}). On the test set it also had "
        f"the higher macro-F1 ({f3(w.macro_f1)} vs {f3(eq.macro_f1)}) and melanoma recall ({f3(w.melanoma_recall)} vs "
        f"{f3(eq.melanoma_recall)}) of the two ensembles. Even so, it labelled {pct(mel.miss_rate)} % of the test melanomas benign "
        f"(95 % confidence interval {pct(mel.wilson_lo)} % to {pct(mel.wilson_hi)} %).")

# ---------------------------------------------------------------- Table III (T5) + text
for r, k in enumerate(["none", "sqrt", "inverse", "focal", "sampler"], start=1):
    x = imb[k]; b = (k == sel["selected"])
    cell(5, r, 0, T[5].rows[r].cells[0].text, bold=b)
    cell(5, r, 1, f3(x["val_macro_f1"]), bold=b)
    cell(5, r, 2, pm(x["test_macro_f1"]), bold=b)
    cell(5, r, 3, pm(x["test_melanoma_recall"]), bold=b)
v = sel["val"]; sm = imb["sampler"]; fo = imb["focal"]
replace(68, "The rule chosen the weighted sampler during validation. On the test set the default square-rooted weights achieved a macro-F1 "
            "(95% confidence interval 0.641–0.738 versus 0.615–0.712 for the sampler) yet the intervals overlap. Inverse-frequency weights "
            "produced the melanoma recall but at the expense of accuracy. A single seed can mislead. Our original single-seed EfficientNet-B0, "
            "evaluated on views reached a macro-F1 of 0.721 which is higher, than the mean macro-F1 of the same recipe when using three seeds.",
        f"The rule chose focal loss on validation (macro-F1 {f3(v['focal'][0])}, within 0.005 of no weighting, with the higher melanoma "
        f"recall). On the test set the weighted sampler achieved the highest macro-F1 (95% confidence interval "
        f"{sm['test_macro_f1']['ci'][0]:.3f}–{sm['test_macro_f1']['ci'][1]:.3f} versus {fo['test_macro_f1']['ci'][0]:.3f}–"
        f"{fo['test_macro_f1']['ci'][1]:.3f} for focal loss) yet the intervals overlap. The weighted sampler produced the highest melanoma "
        f"recall but at the expense of accuracy. A single seed can mislead: our single-seed EfficientNet-B0, evaluated on three views, "
        f"reached a macro-F1 of {f3(t61.loc['efficientnet_b0'].macro_f1)}, higher than the three-seed mean of the same recipe "
        f"({f3(imb['sqrt']['test_macro_f1']['mean'])}).")

# ---------------------------------------------------------------- Table IV (T6) + text
for r, d in ((1, skin), (2, crop)):
    for c, k in enumerate(["macro_f1", "malignant_sensitivity", "malignant_specificity"], start=1):
        cell(6, r, c, dci(d[k]))
replace(72, "(weighted sampler)", "(focal loss)")
dice_u = seg.iloc[0].dice.replace("�", "±").replace("\xb1", "±"); dice_o = float(seg.iloc[1].dice)
replace(73, "Table IV Skin-tone augmentation shows no effect on HAM10000.",
        f"Table IV: skin-tone augmentation did not help on HAM10000; it lowered melanoma recall "
        f"({sg(skin['melanoma_recall']['diff'])}, 95% CI {sg(skin['melanoma_recall']['ci'][0])} to "
        f"{sg(skin['melanoma_recall']['ci'][1])}) and slightly raised specificity.")
replace(73, "The U-Net reached a test Dice score of 0.943 ± 0.001 while Otsu thresholding gave 0.652. Training on crops raised macro-F1 and "
            "specificity. Training on crops lowered malignant sensitivity. Both confidence intervals exclude zero cropping trades missed "
            "cancers, for fewer false alarms.",
        f"The U-Net reached a test Dice score of {dice_u} while Otsu thresholding gave {dice_o:.3f}. Training on crops raised macro-F1 "
        f"and malignant sensitivity and lowered specificity, all with confidence intervals that exclude zero: cropping trades some false "
        f"alarms for fewer missed cancers.")

# ---------------------------------------------------------------- abstract, methods, discussion
replace(7, "test macro-F1 of 0.747", f"test macro-F1 of {f3(w.macro_f1)}")
replace(7, "cropping lesions increased macro-F1 while decreasing sensitivity for malignant lesions",
        "cropping lesions increased both macro-F1 and sensitivity for malignant lesions")
replace(43, "All experiments are implemented in PyTorch [29]. Run on an Apple M2 Pro GPU.",
        "All experiments are implemented in PyTorch [29] and run on NVIDIA Tesla T4 GPUs (Kaggle).")
replace(81, "macro-F1 score of 0.747", f"macro-F1 score of {f3(w.macro_f1)}")

# ---------------------------------------------------------------- missing words (no new numbers)
replace(13, "for -tumour conditions", "for non-tumour conditions")
replace(45, "percentage of -neoplastic images", "percentage of non-neoplastic images")
replace(25, "Melanoma (mel) cell carcinoma (bcc)", "Melanoma (mel), basal cell carcinoma (bcc)")
replace(28, "All of those five images are cell carcinomas.", "All of those five images are basal cell carcinomas.")
replace(64, "the ensemble was 12.5% of the time", "the ensemble was correct 12.5% of the time")
replace(66, "does not help with cell carcinomas", "does not help with basal cell carcinomas")
replace(84, "All of them are cell carcinomas", "All of them are basal cell carcinomas")

out = HERE / "fixed conference_updated.docx"
D.save(str(out))
(HERE / "changes.md").write_text("\n".join(LOG), encoding="utf-8")
print(len(LOG), "changes ->", out.name)

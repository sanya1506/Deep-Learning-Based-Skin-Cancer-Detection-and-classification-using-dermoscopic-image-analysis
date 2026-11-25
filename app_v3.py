"""Streamlit application with the Indian-study features (report Section 5.5, Appendix 2).

Run:  streamlit run app_v3.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402
from models import DISPLAY_NAMES  # noqa: E402
from predict_v3 import PredictorV3, ensemble_weights, gate_info, validation_operating_points  # noqa: E402
from report import DISCLAIMER, csv_bytes, generate_pdf_report, result_to_row  # noqa: E402

st.set_page_config(page_title="Skin lesion classifier - Indian skin study", page_icon="🔬", layout="wide")


V2_READY = (config.RESULTS_DIR / "v2" / "ensemble" / "ensemble_config.json").exists() and any(
    (config.MODELS_DIR / "v2").glob("*.pth"))


@st.cache_resource(show_spinner="Loading models ...")
def get_predictor(mode):
    if mode == "v2":
        from improved.predict_v2 import EnsemblePredictor   # needs: pip install timm segmentation-models-pytorch
        return EnsemblePredictor()
    return PredictorV3(mode)


def measured_fcr():
    f = config.STUDY_DIR / "dermacon" / "table_6_3_fcr.json"
    if not f.exists():
        return None
    for r in json.load(open(f))["rows"]:
        if r["model"] == "ensemble":
            return r
    return None


# --------------------------------------------------------------------------- sidebar
modes = ["v2"] if V2_READY else []
if ensemble_weights():
    modes.append("ensemble")
modes += [m for m in config.HAM_MODELS if (config.MODELS_DIR / f"{m}.pth").exists()]
if gate_info() and (gate_info()["path"]).exists():
    modes.append("two_stage")
if not modes:
    st.error("No trained models found in models/. Run the notebooks first.")
    st.stop()
label = {"v2": "v2 ensemble - ISIC 2019, 384 px (best accuracy)", "ensemble": "Ensemble (report, 224 px)", "two_stage": "Two-stage - for clinical photographs"}
mode = st.sidebar.selectbox("Model", modes, format_func=lambda m: label.get(m, DISPLAY_NAMES.get(m, m)))
pred = get_predictor(mode)

ops = validation_operating_points()
thr = st.sidebar.slider("Malignant threshold", 0.05, 0.95, 0.50, 0.05,
                        help="The cut-off on the malignant probability. At or above it the lesion is flagged MALIGNANT.")
st.sidebar.markdown(
    "**What is the malignant threshold?**\n\n"
    "The model gives a *malignant probability*: the chance that the lesion is one of the three cancer "
    "classes (melanoma, basal cell carcinoma, actinic keratosis / intra-epithelial carcinoma) added together. "
    "If this probability is **at or above** the threshold, the lesion is flagged **MALIGNANT**, otherwise **benign**.\n\n"
    "- **Lower threshold** → more lesions flagged: fewer cancers missed (higher sensitivity) but more false alarms.\n"
    "- **Higher threshold** → fewer false alarms (higher specificity) but more cancers missed.\n\n"
    "The default 0.50 is a balanced middle point. A screening tool would usually use a lower value.")
if ops and mode != "v2":
    se, sp = ops[round(thr, 2)]
    st.sidebar.markdown(f"Validation (HAM10000) at {thr:.2f}: **sensitivity {se:.1%}**, **specificity {sp:.1%}**")

# --------------------------------------------------------------------------- input
st.title("🔬 Skin lesion classification")
c1, c2 = st.columns([2, 1])
up = c1.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
samples = sorted(config.SAMPLE_DIR.glob("*.jpg"))
pick = c2.selectbox("...or a HAM10000 sample", ["-"] + [p.name for p in samples])
img = None
if up is not None:
    img = np.asarray(Image.open(up).convert("RGB"))
elif pick != "-":
    img = np.asarray(Image.open(config.SAMPLE_DIR / pick).convert("RGB"))
if img is None:
    st.info("Upload an image or choose a sample.")
    st.stop()

with st.spinner("Analysing ..."):
    r = pred.predict_array(img)

if r.get("gate") is not None:
    g = r["gate"]
    if g["passed"]:
        st.info(f"Stage 1 gate: P(neoplasm) = {g['p_neoplasm']:.3f} ≥ threshold {g['threshold']:.4f} → sent to the ensemble.")
    else:
        st.error(f"Stage 1 gate: P(neoplasm) = {g['p_neoplasm']:.3f} < threshold {g['threshold']:.4f} → treated as a "
                 "non-tumour condition and NOT classified. The gate was trained on clinical photographs and "
                 "rejects many dermoscopic images.")

if mode == "v2":
    tiles = [(r["original"], "1. Original image"),
             (r["preprocessed"], "2. U-Net lesion crop + colour correction"),
             (r["overlay"], "7. Grad-CAM: where the model looked")]
else:
    tiles = [(r.get("outline", r["original"]), "Image + U-Net lesion outline"),
             (r["crop"] if r.get("crop") is not None else r["original"], "Lesion crop"),
             (r["preprocessed"], "Preprocessed"), (r["overlay"], "Grad-CAM")]
cols = st.columns(len(tiles))
for col, (im, t) in zip(cols, tiles):
    col.image(im, caption=t, width="stretch")

p_mal = r["malignant_probability"]
st.subheader(f"Prediction: {r['pred_class']} – {r['pred_full_name']}  ({r['confidence']:.1%})")
st.markdown(f"Malignant probability **{p_mal:.1%}** → at threshold {thr:.2f}: "
            f"**{'MALIGNANT' if p_mal >= thr else 'benign'}**")
st.bar_chart(pd.Series(r["probabilities"]).sort_values())

if mode == "v2":
    names = {"tf_efficientnet_b4_meta_isic": "EfficientNet-B4", "tf_efficientnetv2_s_meta_isic": "EfficientNetV2-S"}
    members = " and ".join(f"**{names.get(t, t)}** (weight {w:.2f})" for w, (t, _, _) in zip(pred.weights, pred.members))
    with st.expander("What happened to this image in v2, step by step", expanded=True):
        st.markdown(
            "1. **Input.** The uploaded image (or the chosen sample) is read as a colour (RGB) picture.\n"
            "2. **Lesion crop (U-Net).** A U-Net segmentation network, trained on ISIC 2018 lesion masks, finds the "
            "lesion and the image is cropped around it, so the background skin, hair and ruler marks matter less. "
            "If the U-Net is not available, Otsu thresholding is used instead.\n"
            "3. **Colour correction (Shades of Gray).** The colours are normalised so that photos taken under "
            "different lights look alike. The result is resized to **384 × 384 px** (tile 2 above).\n"
            f"4. **Classification by two models.** {members}, both trained on the larger ISIC 2019 dataset, each look "
            "at the image **8 times** (original, mirrored and rotated by 90°, 180°, 270°) and the 8 answers are "
            "averaged (test-time augmentation).\n"
            "5. **Ensemble and calibration.** The two models' answers are combined with the weights above, which were "
            "chosen on the validation set. A small per-class correction, also fitted on validation, is added, and the "
            "result is turned into probabilities for the 7 classes (bar chart).\n"
            f"6. **Decision.** The class with the highest probability is the prediction "
            f"(**{r['pred_class']}**, {r['confidence']:.1%}). The malignant probability is the sum for melanoma, "
            f"basal cell carcinoma and actinic keratosis (**{p_mal:.1%}**), and is compared with the threshold "
            f"in the sidebar ({thr:.2f}).\n"
            "7. **Explanation (Grad-CAM).** Grad-CAM on the EfficientNet-B4 shows which parts of the image pushed the "
            "model towards its answer (last tile above, red = most important).")

d1, d2 = st.columns(2)
d1.download_button("PDF report", data=generate_pdf_report(r), file_name="skin_report.pdf", mime="application/pdf")
d2.download_button("CSV", data=csv_bytes([result_to_row(r)]), file_name="skin_report.csv", mime="text/csv")
st.markdown(f"> {DISCLAIMER}")

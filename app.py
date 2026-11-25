"""Streamlit web app for dermoscopic skin lesion classification.

Run:  streamlit run app.py
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402
from models import DISPLAY_NAMES  # noqa: E402
from predict import Predictor  # noqa: E402
from report import DISCLAIMER, csv_bytes, generate_pdf_report, result_to_row  # noqa: E402

st.set_page_config(page_title="Skin Cancer Detection", page_icon="🔬", layout="wide")

st.markdown("""
<style>
.badge {display:inline-block;padding:0.45rem 1.1rem;border-radius:999px;font-weight:700;
        font-size:1.15rem;letter-spacing:0.04em;color:white;}
.badge-benign {background:#1e8449;}
.badge-malignant {background:#c0392b;}
.disclaimer {border-left:5px solid #c0392b;background:rgba(192,57,43,0.08);padding:0.7rem 1rem;
             border-radius:4px;font-size:0.92rem;}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- helpers
def available_models():
    out = []
    for name in config.AVAILABLE_MODELS:
        if config.model_path(name).exists() or config.model_path(name, quick=True).exists():
            out.append(name)
    return out


@st.cache_resource(show_spinner="Loading model ...")
def get_predictor(model_name: str) -> Predictor:
    return Predictor(model_name)


def load_metrics(model_name: str):
    for sub in (model_name, f"{model_name}_quick"):
        f = config.RESULTS_DIR / sub / "metrics.json"
        if f.exists():
            return json.loads(f.read_text())
    return None


def confidence_chart(probs: dict):
    items = sorted(probs.items(), key=lambda kv: kv[1])
    labels = [f"{c} ({config.CLASS_FULL_NAMES[c].split(' /')[0]})" for c, _ in items]
    vals = [v * 100 for _, v in items]
    cols = ["#c0392b" if config.BINARY_MAP[c] == "Malignant" else "#1e8449" for c, _ in items]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.barh(labels, vals, color=cols)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Confidence (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- sidebar
models_ready = available_models()
st.sidebar.title("🔬 Model")
if not models_ready:
    st.sidebar.error("No trained model found in models/. Run `python src/train.py` first.")
    st.stop()
best_file = config.RESULTS_DIR / "best_model.json"   # written by notebooks/05_model_comparison.ipynb
preferred = json.loads(best_file.read_text())["best_model"] if best_file.exists() else config.DEFAULT_MODEL
default_idx = models_ready.index(preferred) if preferred in models_ready else 0
model_name = st.sidebar.selectbox("Architecture", models_ready, index=default_idx,
                                  format_func=lambda n: DISPLAY_NAMES.get(n, n))
predictor = get_predictor(model_name)
st.sidebar.markdown(
    f"**Backbone:** {predictor.display_name} (ImageNet transfer learning)  \n"
    f"**Checkpoint:** `{predictor.checkpoint.name}`  \n"
    f"**Best epoch:** {predictor.ckpt.get('epoch')}  \n"
    f"**Device:** `{predictor.device}`  \n"
    f"**Input:** {config.IMG_SIZE}×{config.IMG_SIZE} RGB, 7 classes")

metrics = load_metrics(model_name)
st.sidebar.subheader("Test-set metrics")
if metrics:
    o, b = metrics["overall"], metrics["binary"]
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Accuracy", f"{o['accuracy']:.1%}")
    c2.metric("Macro F1", f"{o['macro_f1']:.3f}")
    c1.metric("Macro AUC", f"{o['macro_roc_auc']:.3f}")
    c2.metric("Bal. acc.", f"{o['balanced_accuracy']:.1%}")
    st.sidebar.caption("Benign vs malignant")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Sensitivity", f"{b['sensitivity']:.1%}")
    c2.metric("Specificity", f"{b['specificity']:.1%}")
    st.sidebar.caption(f"n = {o['n_test']} held-out test images (split by lesion_id)")
else:
    st.sidebar.info("Run `python src/evaluate.py` to show test metrics here.")

with st.sidebar.expander("Class legend"):
    for c in config.CLASS_NAMES:
        st.markdown(f"- **{c}** – {config.CLASS_FULL_NAMES[c]} *({config.BINARY_MAP[c]})*")

# --------------------------------------------------------------------------- main
st.title("Skin Cancer Detection & Classification")
st.caption("Deep learning-based analysis of dermoscopic images · HAM10000 · Grad-CAM explainability")
st.markdown(f'<div class="disclaimer">⚠️ <b>Not a medical diagnosis.</b> {DISCLAIMER[12:]}</div>',
            unsafe_allow_html=True)
st.write("")

samples = sorted(config.SAMPLE_DIR.glob("*.jpg"))
query_sample = st.query_params.get("sample")
col_up, col_sample = st.columns([3, 2])
uploaded = col_up.file_uploader("Upload a dermoscopic image", type=["jpg", "jpeg", "png", "bmp"])
sample_names = ["—"] + [p.name for p in samples]
sample_choice = col_sample.selectbox(
    "…or try a sample image", sample_names,
    index=sample_names.index(query_sample) if query_sample in sample_names else 0)

img_bytes, filename = None, None
if uploaded is not None:
    img_bytes, filename = uploaded.getvalue(), uploaded.name
elif sample_choice != "—":
    img_bytes, filename = (config.SAMPLE_DIR / sample_choice).read_bytes(), sample_choice

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.seen = set()

if img_bytes is None:
    st.info("Upload an image or pick a sample to start the analysis.")
else:
    import io
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    with st.spinner("Preprocessing, classifying and computing Grad-CAM ..."):
        result = predictor.predict_array(img)
    result["filename"] = filename

    key = hashlib.md5(img_bytes).hexdigest() + model_name
    if key not in st.session_state.seen:
        st.session_state.seen.add(key)
        st.session_state.history.append(result_to_row(result))

    c1, c2, c3 = st.columns(3)
    c1.image(result["original"], caption="Original", width="stretch")
    c2.image(result["preprocessed"], caption="Preprocessed (hair removal + CLAHE, 224×224)",
             width="stretch")
    c3.image(result["overlay"], caption="Grad-CAM heatmap overlay", width="stretch")

    left, right = st.columns([2, 3])
    with left:
        st.subheader("Prediction")
        badge = "badge-malignant" if result["binary_label"] == "Malignant" else "badge-benign"
        st.markdown(f'<span class="badge {badge}">{result["binary_label"].upper()}</span>',
                    unsafe_allow_html=True)
        st.markdown(f"### {result['pred_class']} — {result['pred_full_name']}")
        m1, m2 = st.columns(2)
        m1.metric("Class confidence", f"{result['confidence']:.1%}")
        m2.metric("Malignant probability", f"{result['malignant_probability']:.1%}")
        st.caption("Grad-CAM: red/yellow regions contributed most to the predicted class.")

        stem = Path(filename).stem
        st.download_button("📄 Download PDF report", generate_pdf_report(result),
                           file_name=f"skin_report_{stem}.pdf", mime="application/pdf",
                           width="stretch")
        st.download_button("📊 Download CSV", csv_bytes([result_to_row(result)]),
                           file_name=f"skin_report_{stem}.csv", mime="text/csv", width="stretch")
    with right:
        st.subheader("Confidence scores")
        fig = confidence_chart(result["probabilities"])
        st.pyplot(fig)
        plt.close(fig)

st.divider()
st.subheader("Session history")
if st.session_state.history:
    hist = pd.DataFrame(st.session_state.history)
    st.dataframe(hist[["timestamp", "filename", "model", "predicted_class", "category", "confidence",
                       "malignant_probability"]], width="stretch", hide_index=True)
    h1, h2 = st.columns([1, 5])
    h1.download_button("Export history CSV", csv_bytes(st.session_state.history),
                       file_name=f"session_history_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")
    if h2.button("Clear history"):
        st.session_state.history, st.session_state.seen = [], set()
        st.rerun()
else:
    st.caption("Analysed images will appear here.")

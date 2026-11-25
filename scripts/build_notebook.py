"""Generate notebooks/skin_cancer_training.ipynb (Google Colab ready).

The notebook writes the project's own config.py and src/*.py via %%writefile, so it runs exactly
the same pipeline as the local scripts.

Usage:
    python scripts/build_notebook.py
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

SOURCE_FILES = ["config.py", "scripts/download_data.py", "scripts/prepare_samples.py", "src/preprocess.py",
                "src/dataset.py", "src/models.py", "src/train.py", "src/evaluate.py", "src/gradcam.py",
                "src/predict.py", "src/report.py", "src/compare_models.py"]

ABSTRACT = (ROOT / "docs" / "abstract.txt").read_text().strip() if (ROOT / "docs" / "abstract.txt").exists() else ""


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def build():
    cells = [
        md(f"""
# Deep Learning-Based Skin Cancer Detection and Classification Using Dermoscopic Image Analysis

**Abstract.** {ABSTRACT}

This notebook runs the complete pipeline end-to-end on **Google Colab** (Runtime → Change runtime type → **GPU**):

1. Download HAM10000 from Kaggle  2. Exploratory data analysis  3. Preprocessing (hair removal, CLAHE)
4. Leakage-free split by `lesion_id`  5. Transfer-learning training  6. Evaluation  7. Grad-CAM  8. Model comparison
"""),
        code("""
!nvidia-smi || echo "No NVIDIA GPU - switch the Colab runtime to GPU for reasonable training times"
!pip install -q -U kaggle reportlab
import torch; print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
"""),
        md("""
## 1. Project source files
The cells below write the project modules to disk (identical to the files in the repository).
"""),
        code("""
import os
for d in ["scripts", "src", "data/raw", "models", "results", "sample_images", "docs"]:
    os.makedirs(d, exist_ok=True)
"""),
    ]
    for rel in SOURCE_FILES:
        cells.append(code(f"%%writefile {rel}\n" + (ROOT / rel).read_text()))

    cells += [
        md("""
## 2. Kaggle credentials and download
Create a token at <https://www.kaggle.com/settings> → *API* → *Generate New Token*.
Either paste the **new-format token** when prompted, or upload a legacy **kaggle.json**.
"""),
        code("""
import os, getpass, pathlib
kdir = pathlib.Path.home() / ".kaggle"; kdir.mkdir(exist_ok=True)
if not ((kdir / "access_token").exists() or (kdir / "kaggle.json").exists() or os.environ.get("KAGGLE_API_TOKEN")):
    choice = input("Type 't' to paste an API token, or 'j' to upload kaggle.json: ").strip().lower()
    if choice == "j":
        from google.colab import files
        up = files.upload()
        (kdir / "kaggle.json").write_bytes(up["kaggle.json"]); os.chmod(kdir / "kaggle.json", 0o600)
    else:
        (kdir / "access_token").write_text(getpass.getpass("Kaggle API token: ").strip())
        os.chmod(kdir / "access_token", 0o600)
"""),
        code("!python scripts/download_data.py"),
        md("## 3. Exploratory data analysis"),
        code("""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "src")
import matplotlib.pyplot as plt, seaborn as sns, pandas as pd
import config
from dataset import load_metadata
df = load_metadata()
print(df.shape, "| unique lesions:", df.lesion_id.nunique())
df.head()
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
counts = df.dx.value_counts().reindex(config.CLASS_NAMES)
colors = ["#c0392b" if config.BINARY_MAP[c] == "Malignant" else "#1e8449" for c in counts.index]
bars = axes[0].bar(counts.index, counts.values, color=colors); axes[0].bar_label(bars)
axes[0].set_title("Class distribution (red = malignant, green = benign)"); axes[0].set_ylabel("images")
df.binary.value_counts().plot.pie(ax=axes[1], autopct="%1.1f%%", colors=["#1e8449", "#c0392b"])
axes[1].set_title("Benign vs malignant"); axes[1].set_ylabel("")
plt.tight_layout(); plt.show()
"""),
        code("""
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
sns.histplot(df, x="age", hue="binary", bins=20, multiple="stack", ax=axes[0]); axes[0].set_title("Age")
df.sex.value_counts().plot.bar(ax=axes[1], color="#5d6d7e"); axes[1].set_title("Sex")
df.localization.value_counts().plot.barh(ax=axes[2], color="#5d6d7e"); axes[2].set_title("Localization")
plt.tight_layout(); plt.show()
"""),
        code("""
from preprocess import load_image_rgb, remove_hair, apply_clahe, preprocess_image
fig, axes = plt.subplots(len(config.CLASS_NAMES), 4, figsize=(12, 3 * len(config.CLASS_NAMES)))
for r, c in enumerate(config.CLASS_NAMES):
    img = load_image_rgb(df[df.dx == c].iloc[0].path)
    no_hair = remove_hair(img)
    for ax, im, t in zip(axes[r], [img, no_hair, apply_clahe(no_hair), preprocess_image(img)],
                         ["original", "hair removed", "CLAHE", "final 224x224"]):
        ax.imshow(im); ax.set_title(f"{c}: {t}", fontsize=9); ax.axis("off")
plt.tight_layout(); plt.show()
"""),
        md("## 4. Leakage-free split (by `lesion_id`) and preprocessing cache"),
        code("!python src/dataset.py\n!python src/preprocess.py"),
        md("## 5. Training\nFirst a `--quick` run to verify the pipeline, then the full EfficientNet-B0 training (15 epochs, early stopping)."),
        code("!python src/train.py --quick && python src/evaluate.py --quick"),
        code("!python src/train.py --model efficientnet_b0 --epochs 15"),
        md("## 6. Evaluation on the held-out test set"),
        code("""
!python src/evaluate.py --model efficientnet_b0
from IPython.display import Image, display
for f in ["training_curves.png", "confusion_matrix.png", "confusion_matrix_binary.png", "roc_curves.png"]:
    display(Image(f"results/efficientnet_b0/{f}"))
"""),
        md("## 7. Grad-CAM explanations"),
        code("""
!python scripts/prepare_samples.py
!python src/gradcam.py --all-samples
import glob
for f in sorted(glob.glob("results/gradcam/*.png")):
    display(Image(f))
"""),
        code("!python src/predict.py --image $(ls sample_images/*.jpg | head -1) --pdf --csv"),
        md("## 8. Model comparison (EfficientNet-B0 vs ResNet50 vs DenseNet121)"),
        code("""
!python src/compare_models.py --epochs 15 --reuse
display(pd.read_csv("results/model_comparison.csv"))
display(Image("results/model_comparison.png"))
"""),
        md("## 9. Download trained models and results"),
        code("""
!zip -qr trained_outputs.zip models results sample_images
try:
    from google.colab import files; files.download("trained_outputs.zip")
except ImportError:
    print("Saved trained_outputs.zip")
"""),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}}
    out = ROOT / "notebooks" / "skin_cancer_training.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    print(f"Wrote {out} ({len(cells)} cells)")


if __name__ == "__main__":
    build()

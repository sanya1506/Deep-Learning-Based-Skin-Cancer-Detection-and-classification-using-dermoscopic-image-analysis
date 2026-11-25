"""Build docs/Project_Report.docx and docs/Presentation.pptx from the real numbers/figures in results/.

Run after training, evaluation, Grad-CAM, model comparison and screenshots:
    python scripts/build_docs.py
"""
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import config  # noqa: E402

from docx import Document  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Inches, Pt, RGBColor  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor as PRGB  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.enum.text import PP_ALIGN  # noqa: E402
from pptx.util import Emu  # noqa: E402
from pptx.util import Inches as PI  # noqa: E402
from pptx.util import Pt as PPt  # noqa: E402

MODEL = config.DEFAULT_MODEL
RES = config.RESULTS_DIR
FIG = RES / "figures"
SHOTS = config.DOCS_DIR / "screenshots"
TITLE = "Deep Learning-Based Skin Cancer Detection and Classification Using Dermoscopic Image Analysis"
ABSTRACT = (config.DOCS_DIR / "abstract.txt").read_text().strip()

REFERENCES = [
    "Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017). "
    "Dermatologist-level classification of skin cancer with deep neural networks. Nature, 542(7639), 115-118.",
    "Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of "
    "multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data, 5, 180161.",
    "Codella, N. C. F., Gutman, D., Celebi, M. E., et al. (2018). Skin lesion analysis toward melanoma "
    "detection: A challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), hosted by the "
    "International Skin Imaging Collaboration (ISIC). Proc. IEEE ISBI 2018, 168-172.",
    "Haenssle, H. A., Fink, C., Schneiderbauer, R., et al. (2018). Man against machine: diagnostic "
    "performance of a deep learning convolutional neural network for dermoscopic melanoma recognition in "
    "comparison to 58 dermatologists. Annals of Oncology, 29(8), 1836-1842.",
    "Brinker, T. J., Hekler, A., Enk, A. H., et al. (2019). Deep learning outperformed 136 of 157 "
    "dermatologists in a head-to-head dermoscopic melanoma image classification task. European Journal of "
    "Cancer, 113, 47-54.",
    "Tschandl, P., Codella, N., Akay, B. N., et al. (2019). Comparison of the accuracy of human readers versus "
    "machine-learning algorithms for pigmented skin lesion classification: an open, web-based, international, "
    "diagnostic study. The Lancet Oncology, 20(7), 938-947.",
    "Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual "
    "explanations from deep networks via gradient-based localization. Proc. IEEE ICCV, 618-626.",
    "Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. "
    "Proc. ICML, PMLR 97, 6105-6114.",
    "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. "
    "Proc. IEEE CVPR, 770-778.",
    "Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional "
    "networks. Proc. IEEE CVPR, 4700-4708.",
    "Lee, T., Ng, V., Gallagher, R., Coldman, A., & McLean, D. (1997). DullRazor: A software approach to hair "
    "removal from images. Computers in Biology and Medicine, 27(6), 533-543.",
    "Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. In Graphics Gems IV, "
    "Academic Press, 474-485.",
    "Telea, A. (2004). An image inpainting technique based on the fast marching method. Journal of Graphics "
    "Tools, 9(1), 23-34.",
    "Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. Proc. ICLR 2019.",
    "Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale "
    "hierarchical image database. Proc. IEEE CVPR, 248-255.",
]


# --------------------------------------------------------------------------- data
def load_results():
    m = json.loads((RES / MODEL / "metrics.json").read_text())
    per_class = pd.read_csv(RES / MODEL / "per_class_metrics.csv")
    summ = json.loads((RES / MODEL / "training_summary.json").read_text())
    hist = pd.read_csv(RES / MODEL / "training_history.csv")
    comp = pd.read_csv(RES / "model_comparison.csv") if (RES / "model_comparison.csv").exists() else None
    splits = {s: pd.read_csv(config.SPLITS_DIR / f"{s}.csv") for s in ("train", "val", "test")}
    return m, per_class, summ, hist, comp, splits


def pct(v):
    return f"{100 * v:.1f}%"


# --------------------------------------------------------------------------- DOCX helpers
def shade(cell, hex_color):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def add_table(doc, header, rows, widths=None, font=9):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(str(h))
        r.bold = True
        r.font.size = Pt(font)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(c, "2C3E50")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(font)
            if ri % 2:
                shade(cells[i], "F2F3F4")
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    return t


class Counter:
    def __init__(self):
        self.fig = 0
        self.tab = 0


def add_figure(doc, path, caption, cnt, width=6.3):
    if not Path(path).exists():
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cnt.fig += 1
    p = doc.add_paragraph(f"Figure {cnt.fig}: {caption}")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].italic = True
    p.runs[0].font.size = Pt(9)


def table_caption(doc, caption, cnt):
    cnt.tab += 1
    p = doc.add_paragraph(f"Table {cnt.tab}: {caption}")
    p.runs[0].bold = True
    p.runs[0].font.size = Pt(9.5)


def bullets(doc, items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


# --------------------------------------------------------------------------- DOCX
def build_docx(m, per_class, summ, hist, comp, splits):
    o, b = m["overall"], m["binary"]
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(11)
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(1)
    cnt = Counter()

    # title page
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(TITLE)
    r.bold = True
    r.font.size = Pt(22)
    r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Technical Project Report").font.size = Pt(14)
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(f"{date.today():%B %Y}").font.size = Pt(12)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Python · PyTorch · OpenCV · scikit-learn · Streamlit").font.size = Pt(11)
    doc.add_page_break()

    best = per_class.sort_values("f1", ascending=False).iloc[0]
    worst = per_class.sort_values("f1").iloc[0]
    mel = per_class.set_index("class").loc["mel"]
    n_total = sum(len(s) for s in splits.values())

    # 1 abstract
    doc.add_heading("Abstract", 1)
    doc.add_paragraph(ABSTRACT)
    doc.add_paragraph(
        f"In our implementation, an EfficientNet-B0 network fine-tuned on the HAM10000 dataset reached "
        f"{pct(o['accuracy'])} accuracy, a macro F1-score of {o['macro_f1']:.3f} and a macro ROC-AUC of "
        f"{o['macro_roc_auc']:.3f} on a held-out, lesion-disjoint test set of {o['n_test']} images. For the "
        f"clinically relevant benign-versus-malignant decision the system achieved a sensitivity of "
        f"{pct(b['sensitivity'])}, a specificity of {pct(b['specificity'])} and a ROC-AUC of {b['roc_auc']:.3f}.")

    # 2 introduction
    doc.add_heading("1. Introduction", 1)
    doc.add_paragraph(
        "Skin cancer is the most common group of cancers worldwide. Melanoma accounts for only a small fraction "
        "of skin cancers but causes the majority of skin-cancer deaths, while basal cell carcinoma and actinic "
        "keratoses / intraepithelial carcinoma are far more frequent and still require treatment. Survival "
        "depends strongly on early detection: thin melanomas detected early are usually curable by excision.")
    doc.add_paragraph(
        "Dermoscopy (epiluminescence microscopy) magnifies the skin surface and removes surface reflection, "
        "revealing pigment structures that are invisible to the naked eye. It improves diagnostic accuracy of "
        "trained dermatologists, but interpretation is subjective, time-consuming and requires years of "
        "experience. Convolutional neural networks (CNNs) have shown dermatologist-level performance on "
        "dermoscopic image classification [1, 4, 5, 6], which motivates an automated decision-support tool "
        "that is accurate, fast, and - critically for clinical acceptance - explainable.")
    doc.add_paragraph(
        "This project builds such a tool end-to-end: an image preprocessing pipeline, three transfer-learning "
        "CNN backbones, a rigorous evaluation protocol, Grad-CAM visual explanations, a Streamlit web "
        "application, and PDF/CSV report export.")

    # 3 problem statement
    doc.add_heading("2. Problem Statement", 1)
    doc.add_paragraph(
        "Given a single RGB dermoscopic image of a pigmented skin lesion, the system must (i) classify it into "
        "one of the seven diagnostic categories of HAM10000, (ii) map the result to a benign or malignant "
        "group, (iii) quantify its confidence for every class, (iv) visually explain which image regions drove "
        "the decision, and (v) produce an exportable report. The problem is challenging because:")
    bullets(doc, [
        "Classes are severely imbalanced (melanocytic nevi make up about two thirds of HAM10000).",
        "Inter-class similarity is high (e.g. melanoma vs. nevus, bkl vs. melanoma) and intra-class variation is large.",
        "Images contain artefacts: hair, ruler marks, gel bubbles, vignetting and uneven illumination.",
        "Many lesions have several images; naive random splits leak the same lesion into train and test sets "
        "and inflate reported accuracy.",
        "Black-box predictions are not acceptable in a medical setting - explanations are required.",
    ])
    doc.add_heading("Objectives", 2)
    bullets(doc, [
        "Implement hair removal and contrast enhancement preprocessing for dermoscopic images.",
        "Train and compare EfficientNet-B0, ResNet50 and DenseNet121 using transfer learning.",
        "Handle class imbalance with a class-weighted loss and data augmentation.",
        "Evaluate with clinical metrics: sensitivity, specificity, precision, F1, ROC-AUC, confusion matrices.",
        "Provide Grad-CAM explanations, a web interface and PDF/CSV report generation.",
    ])

    # 4 literature review
    doc.add_heading("3. Literature Review", 1)
    lit = [
        ("Esteva et al. (2017) [1]",
         "Fine-tuned a GoogLeNet Inception-v3 CNN on 129,450 clinical images covering 2,032 diseases and matched "
         "the performance of 21 board-certified dermatologists on keratinocyte carcinoma and melanoma "
         "classification. It established transfer learning from ImageNet [15] as the dominant approach."),
        ("Tschandl, Rosendahl & Kittler (2018) [2]",
         "Released HAM10000, 10,015 dermoscopic images from different populations and modalities, with seven "
         "diagnostic categories, more than 50% confirmed by histopathology. The metadata contains lesion_id, "
         "which this project uses to avoid data leakage."),
        ("Codella et al. (2018) [3]",
         "Summarised the ISIC 2017 challenge on lesion segmentation, dermoscopic feature detection and "
         "melanoma classification; top methods relied on deep CNN ensembles, data augmentation and external "
         "data, and the challenge popularised ROC-AUC and sensitivity/specificity reporting."),
        ("Haenssle et al. (2018) [4]",
         "Compared a fine-tuned Inception-v4 CNN with 58 dermatologists on dermoscopic melanoma recognition; the "
         "CNN achieved a higher specificity at the dermatologists' sensitivity level (ROC-AUC 0.86 vs 0.79 mean)."),
        ("Brinker et al. (2019) [5]",
         "A ResNet50 trained only on open ISIC images outperformed 136 of 157 dermatologists in a head-to-head "
         "dermoscopic melanoma classification task, demonstrating that standard ImageNet backbones such as "
         "ResNet50 (also used here) are competitive."),
        ("Tschandl et al. (2019) [6]",
         "An international reader study on the HAM10000 seven-class task: the top machine-learning algorithms "
         "outperformed 511 human readers, including experts, on average, while also highlighting weaknesses on "
         "out-of-distribution images."),
        ("Selvaraju et al. (2017) [7]",
         "Proposed Grad-CAM, which weights the last convolutional feature maps by the spatially pooled gradients "
         "of the class score to produce class-discriminative localisation maps without retraining - the "
         "explainability technique implemented in this project."),
    ]
    for head, txt in lit:
        p = doc.add_paragraph()
        p.add_run(head + ". ").bold = True
        p.add_run(txt)
    doc.add_paragraph(
        "Classic image-processing work also informs our preprocessing: DullRazor [11] introduced morphological "
        "hair detection followed by interpolation, and CLAHE [12] is widely used to enhance local contrast in "
        "medical images. EfficientNet [8], ResNet [9] and DenseNet [10] are the backbones evaluated.")

    # 5 dataset
    doc.add_heading("4. Dataset", 1)
    doc.add_paragraph(
        f"HAM10000 (\"Human Against Machine with 10000 training images\") [2] was downloaded from Kaggle "
        f"(kmader/skin-cancer-mnist-ham10000). It contains {n_total} dermoscopic images (600×450 px) of "
        f"{sum(s.lesion_id.nunique() for s in splits.values())} unique lesions. The three malignant classes "
        f"(mel, bcc, akiec) account for "
        f"{pct(sum(s.dx.isin(config.MALIGNANT_CLASSES).sum() for s in splits.values()) / n_total)} of images.")
    rows = []
    for c in config.CLASS_NAMES:
        rows.append([c, config.CLASS_FULL_NAMES[c], config.BINARY_MAP[c]] +
                    [int((splits[s].dx == c).sum()) for s in ("train", "val", "test")] +
                    [int(sum((splits[s].dx == c).sum() for s in splits))])
    rows.append(["Total", "", ""] + [len(splits[s]) for s in ("train", "val", "test")] + [n_total])
    table_caption(doc, "HAM10000 classes and the lesion-grouped 70/15/15 split (number of images)", cnt)
    add_table(doc, ["Code", "Diagnosis", "Group", "Train", "Val", "Test", "Total"], rows,
              [0.6, 2.6, 0.9, 0.6, 0.5, 0.5, 0.6])
    doc.add_paragraph()
    doc.add_paragraph(
        "Splitting protocol. Splits are made at the lesion level: unique lesion_ids are split 70/15/15, "
        "stratified by diagnosis (random seed 42), and every image is assigned to the split of its lesion. "
        "This guarantees that no lesion appears in more than one split (verified by assertion in "
        "src/dataset.py), so test performance reflects generalisation to unseen lesions.")
    add_figure(doc, FIG / "class_distribution.png", "Class distribution per split and benign/malignant ratio", cnt)

    # 6 methodology
    doc.add_heading("5. Methodology", 1)
    add_figure(doc, FIG / "pipeline.png", "End-to-end system pipeline", cnt)
    doc.add_heading("5.1 Preprocessing", 2)
    bullets(doc, [
        f"Hair removal: the grey-scale image is filtered with a morphological black-hat operation "
        f"({config.HAIR_KERNEL_SIZE}×{config.HAIR_KERNEL_SIZE} cross-shaped kernel), which responds to thin dark "
        f"structures; thresholding (T={config.HAIR_THRESHOLD}) gives a hair mask that is dilated and filled by "
        f"Telea fast-marching inpainting [13] (radius {config.INPAINT_RADIUS}).",
        f"Enhancement: CLAHE [12] (clip limit {config.CLAHE_CLIP_LIMIT}, {config.CLAHE_TILE_GRID[0]}×"
        f"{config.CLAHE_TILE_GRID[1]} tiles) applied to the L channel of the CIELAB colour space, enhancing "
        "texture and border contrast without distorting lesion colour.",
        f"Resizing to {config.IMG_SIZE}×{config.IMG_SIZE} pixels (area interpolation).",
        "Normalisation: pixel values scaled to [0, 1] and standardised with ImageNet channel mean and standard "
        "deviation, matching the pre-trained backbones.",
    ])
    doc.add_paragraph("The identical function (src/preprocess.py) is used for training, evaluation and inference; "
                      "preprocessed images are cached to disk to accelerate training.")
    add_figure(doc, FIG / "preprocessing_steps.png", "Preprocessing stages on a training image with visible hair", cnt)
    doc.add_heading("5.2 Data augmentation", 2)
    doc.add_paragraph("Applied on-the-fly to training images only: random resized crop (zoom, scale 0.75-1.0), "
                      "horizontal and vertical flips, rotation (±30°) and colour jitter (brightness/contrast 0.2, "
                      "saturation 0.15, hue 0.03). Dermoscopic images have no canonical orientation, so flips and "
                      "rotations are label-preserving.")
    doc.add_heading("5.3 Training procedure", 2)
    table_caption(doc, "Training configuration", cnt)
    add_table(doc, ["Setting", "Value"], [
        ["Loss", "Cross-entropy with class weights (square root of balanced weights), label smoothing 0.05"],
        ["Optimiser", f"AdamW [14], lr {config.LEARNING_RATE}, weight decay {config.WEIGHT_DECAY}"],
        ["LR schedule", "ReduceLROnPlateau on validation macro-F1 (factor 0.3, patience 2)"],
        ["Batch size / max epochs", f"{config.BATCH_SIZE} / {config.EPOCHS}"],
        ["Early stopping", f"Patience {config.EARLY_STOPPING_PATIENCE} epochs on validation macro-F1; best checkpoint kept"],
        ["Hardware", f"{summ['device'].upper()} ({'Apple Silicon GPU' if summ['device'] == 'mps' else summ['device']})"],
        ["Epochs run / training time", f"{summ['epochs_run']} / {summ['train_minutes']:.1f} min"],
    ], [1.8, 4.6])
    doc.add_paragraph()
    doc.add_paragraph("Model selection uses validation macro-F1 rather than accuracy because accuracy is dominated by "
                      "the nevus class; the test set is used once, for the final evaluation.")

    # 7 architecture
    doc.add_heading("6. Model Architecture", 1)
    doc.add_paragraph(
        "All three backbones are initialised with ImageNet [15] weights; the original 1000-way classifier is "
        "replaced with Dropout(0.3) followed by a linear layer with 7 outputs, and the whole network is "
        "fine-tuned.")
    arch_rows = [
        ["EfficientNet-B0 [8]", "Compound-scaled MBConv blocks with squeeze-and-excitation, SiLU", "~4.0 M",
         "features[-1] (1280×7×7)"],
        ["ResNet50 [9]", "Bottleneck residual blocks with identity shortcuts", "~23.5 M", "layer4[-1] (2048×7×7)"],
        ["DenseNet121 [10]", "Dense blocks with feature concatenation, transition layers", "~7.0 M",
         "denseblock4 (1024×7×7)"],
    ]
    table_caption(doc, "Backbones (parameter counts include the new 7-class head)", cnt)
    add_table(doc, ["Model", "Key idea", "Params", "Grad-CAM layer"], arch_rows, [1.4, 2.9, 0.7, 1.5])
    doc.add_paragraph()
    doc.add_paragraph(
        "EfficientNet-B0 is the default production model: it offers the best accuracy-per-parameter trade-off and "
        "runs in real time on a CPU. Its seven MBConv stages progressively extract low-level colour and edge "
        "features, mid-level texture (pigment network, dots, globules) and high-level lesion shape and border "
        "irregularity cues, followed by global average pooling and the classifier. The 7-class softmax output is "
        "mapped to Benign (nv, bkl, df, vasc) or Malignant (mel, bcc, akiec); the malignant probability is the "
        "sum of the three malignant class probabilities.")

    # 8 XAI
    doc.add_heading("7. Explainable AI: Grad-CAM", 1)
    doc.add_paragraph(
        "Grad-CAM [7] is implemented from scratch (src/gradcam.py). A forward hook on the last convolutional "
        "block stores the activations A^k and registers a tensor hook that captures the gradient of the "
        "predicted class score y^c. Channel weights are the spatially averaged gradients, "
        "α_k = (1/Z) Σ_ij ∂y^c/∂A^k_ij, and the map is L = ReLU(Σ_k α_k A^k). The map is bilinearly up-sampled "
        "to 224×224, min-max normalised, coloured with the JET colour map and blended (α = 0.45) over the "
        "preprocessed image. Red/yellow regions indicate the evidence that increased the predicted class score.")
    for f in sorted((RES / "gradcam").glob("gradcam_*.png"))[:4]:
        cls = f.stem.replace("gradcam_", "").split("_")[0]
        add_figure(doc, f, f"Grad-CAM explanation for a held-out test image (true class: {cls})", cnt, 6.0)

    # 9 results
    doc.add_heading("8. Results", 1)
    doc.add_paragraph(
        f"All results are on the lesion-disjoint test set ({o['n_test']} images) that was never used for "
        f"training or model selection. The EfficientNet-B0 checkpoint from epoch {m['best_epoch']} was used.")
    table_caption(doc, "Overall test metrics - EfficientNet-B0 (7-class)", cnt)
    add_table(doc, ["Metric", "Value"], [
        ["Accuracy", pct(o["accuracy"])], ["Balanced accuracy (mean recall)", pct(o["balanced_accuracy"])],
        ["Macro precision", f"{o['macro_precision']:.3f}"], ["Macro recall / sensitivity", f"{o['macro_recall_sensitivity']:.3f}"],
        ["Macro specificity", f"{o['macro_specificity']:.3f}"], ["Macro F1", f"{o['macro_f1']:.3f}"],
        ["Weighted F1", f"{o['weighted_f1']:.3f}"], ["Macro ROC-AUC (one-vs-rest)", f"{o['macro_roc_auc']:.3f}"],
    ], [3.2, 1.5])
    doc.add_paragraph()
    table_caption(doc, "Per-class test metrics - EfficientNet-B0", cnt)
    add_table(doc, ["Class", "Group", "Precision", "Sensitivity", "Specificity", "F1", "ROC-AUC", "Support"],
              [[r["class"], r["type"], f"{r.precision:.3f}", f"{r.recall_sensitivity:.3f}", f"{r.specificity:.3f}",
                f"{r.f1:.3f}", f"{r.roc_auc:.3f}", int(r.support)] for _, r in per_class.iterrows()],
              [0.6, 0.9, 0.8, 0.85, 0.85, 0.6, 0.75, 0.7])
    doc.add_paragraph()
    table_caption(doc, "Benign vs malignant test metrics - EfficientNet-B0", cnt)
    tn, fp = b["confusion_matrix"][0]
    fn, tp = b["confusion_matrix"][1]
    add_table(doc, ["Metric", "Value"], [
        ["Accuracy", pct(b["accuracy"])], ["Sensitivity (malignant recall)", pct(b["sensitivity"])],
        ["Specificity", pct(b["specificity"])], ["Precision (PPV)", pct(b["precision"])],
        ["F1", f"{b['f1']:.3f}"], ["ROC-AUC", f"{b['roc_auc']:.3f}"],
        ["TP / FN / FP / TN", f"{tp} / {fn} / {fp} / {tn}"],
    ], [3.2, 1.5])
    add_figure(doc, RES / MODEL / "training_curves.png", "Training and validation loss, accuracy and macro-F1", cnt)
    add_figure(doc, RES / MODEL / "confusion_matrix.png", "Seven-class confusion matrix (test set)", cnt, 5.2)
    add_figure(doc, RES / MODEL / "roc_curves.png", "ROC curves: one-vs-rest per class and benign vs malignant", cnt)
    add_figure(doc, RES / MODEL / "confusion_matrix_binary.png", "Benign vs malignant confusion matrix", cnt, 3.8)

    if comp is not None:
        doc.add_heading("8.1 Model comparison", 2)
        table_caption(doc, "Comparison of backbones on the test set", cnt)
        add_table(doc, ["Model", "Params (M)", "Accuracy", "Bal. acc.", "Macro F1", "Macro AUC", "Mal. sens.",
                        "Mal. spec.", "Epochs", "Train min"],
                  [[r.model, r.params_millions, pct(r.accuracy), pct(r.balanced_accuracy), f"{r.macro_f1:.3f}",
                    f"{r.macro_roc_auc:.3f}", pct(r.bin_sensitivity), pct(r.bin_specificity), int(r.epochs_run),
                    r.train_minutes] for r in comp.itertuples()], font=8)
        add_figure(doc, RES / "model_comparison.png", "Test-set metrics of the three backbones", cnt)

    shots = sorted(SHOTS.glob("*.png"))
    if shots:
        doc.add_heading("8.2 Application", 2)
        doc.add_paragraph("The Streamlit application shows the original, preprocessed and Grad-CAM images, the "
                          "predicted class with a Benign/Malignant badge, the confidence of all seven classes, "
                          "PDF/CSV download buttons, test metrics in the sidebar and an exportable session history.")
        for s in shots[:2]:
            add_figure(doc, s, "Streamlit web application (" + s.stem.replace("_", " ") + ")", cnt)

    # 10 discussion
    doc.add_heading("9. Discussion", 1)
    doc.add_paragraph(
        f"The model performs best on {best['class']} ({config.CLASS_FULL_NAMES[best['class']]}, F1 "
        f"{best['f1']:.3f}) and worst on {worst['class']} ({config.CLASS_FULL_NAMES[worst['class']]}, F1 "
        f"{worst['f1']:.3f}). Melanoma, the most dangerous class, reached a sensitivity of "
        f"{pct(mel['recall_sensitivity'])} with specificity {pct(mel['specificity'])} and ROC-AUC "
        f"{mel['roc_auc']:.3f}. The most frequent confusions are between visually similar pigmented lesions "
        "(melanoma, nevus and benign keratosis), which is consistent with the literature [6] and with the "
        "difficulty these cases pose to dermatologists.")
    doc.add_paragraph(
        f"The gap between accuracy ({pct(o['accuracy'])}) and balanced accuracy ({pct(o['balanced_accuracy'])}) "
        "reflects class imbalance: accuracy is dominated by the abundant nevus class. The class-weighted loss "
        "trades a little overall accuracy for better recall on rare classes, which is the clinically preferable "
        "behaviour. Because splits are grouped by lesion_id, the reported figures are lower than the >95% values "
        "sometimes reported with random image-level splits, but they are an honest estimate of performance on new "
        "lesions.")
    doc.add_paragraph(
        "Grad-CAM maps generally concentrate on the lesion body and its borders rather than on the surrounding "
        "skin, which increases trust in the predictions; maps that highlight artefacts (rulers, vignetting) are "
        "a useful warning sign that a prediction should not be trusted.")

    doc.add_heading("10. Limitations", 1)
    bullets(doc, [
        "HAM10000 is dominated by fair-skinned European/Australian patients; performance on darker skin types is unknown.",
        "Only dermoscopic images are supported - clinical (smartphone) photos are out of distribution.",
        "Rare classes (df, vasc, akiec) have few test examples, so their metrics have wide confidence intervals.",
        "No external validation set (e.g. ISIC 2019/2020) and no patient metadata (age, site) were used.",
        "Grad-CAM has coarse 7×7 resolution and indicates correlation, not causation.",
        f"Training budget was limited ({summ['epochs_run']} epochs on a laptop GPU); no ensembling or test-time augmentation.",
        "The system is not a certified medical device and has not been clinically validated.",
    ])
    doc.add_heading("11. Future Scope", 1)
    bullets(doc, [
        "Train on the larger ISIC 2019/2020 archives and validate externally across institutions and skin types.",
        "Lesion segmentation (e.g. U-Net) before classification, and fusion of patient metadata.",
        "Larger / newer backbones (EfficientNetV2, ConvNeXt, Vision Transformers) and model ensembles with TTA.",
        "Probability calibration and uncertainty estimation; abstain on out-of-distribution inputs.",
        "Additional explainability (Grad-CAM++, SHAP, concept-based explanations such as the ABCD rule).",
        "Mobile / edge deployment with model quantisation, and integration with clinical workflows (DICOM, EHR).",
    ])
    doc.add_heading("12. Conclusion", 1)
    doc.add_paragraph(
        f"This project delivered a complete, explainable skin-lesion classification system. Dermoscopic images "
        f"are cleaned with hair removal and CLAHE, classified by a fine-tuned EfficientNet-B0 into seven HAM10000 "
        f"classes and mapped to benign/malignant groups. On unseen lesions the model achieved {pct(o['accuracy'])} "
        f"accuracy, macro ROC-AUC {o['macro_roc_auc']:.3f}, and {pct(b['sensitivity'])} sensitivity / "
        f"{pct(b['specificity'])} specificity for malignancy. Grad-CAM heat maps make each decision inspectable, "
        "and the Streamlit interface with PDF/CSV export makes the system usable by non-programmers. The tool is "
        "intended for research and education and as a basis for future clinically validated decision support.")

    doc.add_heading("References", 1)
    for i, ref in enumerate(REFERENCES, 1):
        p = doc.add_paragraph(f"[{i}] {ref}")
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.first_line_indent = Inches(-0.35)
        p.runs[0].font.size = Pt(9.5)

    out = config.DOCS_DIR / "Project_Report.docx"
    doc.save(out)
    print(f"Wrote {out} ({cnt.fig} figures, {cnt.tab} tables)")


# --------------------------------------------------------------------------- PPTX
NAVY = PRGB(0x1F, 0x3A, 0x5F)
ACCENT = PRGB(0xC0, 0x39, 0x2B)
GREY = PRGB(0x55, 0x5F, 0x6B)
LIGHT = PRGB(0xF4, 0xF6, 0xF7)
W, H = PI(13.333), PI(7.5)


def _text(slide, x, y, w, h, text, size=18, bold=False, color=None, align=None):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = PPt(size)
        p.font.bold = bold
        p.font.color.rgb = color or PRGB(0x22, 0x22, 0x22)
        if align:
            p.alignment = align
    return tb


def _bullets(slide, x, y, w, h, items, size=18):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        sub = it.startswith("  ")
        p.text = ("– " if sub else "• ") + it.strip()
        p.level = 1 if sub else 0
        p.font.size = PPt(size - 3 if sub else size)
        p.font.color.rgb = GREY if sub else PRGB(0x22, 0x22, 0x22)
        p.space_after = PPt(7)
    return tb


def _slide(prs, title, num):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, PI(1.05))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    _text(s, PI(0.5), PI(0.18), PI(12), PI(0.8), title, 30, True, PRGB(0xFF, 0xFF, 0xFF))
    acc = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, PI(1.05), W, PI(0.06))
    acc.fill.solid()
    acc.fill.fore_color.rgb = ACCENT
    acc.line.fill.background()
    _text(s, PI(12.3), PI(7.0), PI(0.9), PI(0.4), str(num), 11, color=GREY, align=PP_ALIGN.RIGHT)
    _text(s, PI(0.5), PI(7.0), PI(9), PI(0.4), "Skin Cancer Detection with Deep Learning", 11, color=GREY)
    return s


def _pic(slide, path, x, y, w=None, h=None):
    if Path(path).exists():
        return slide.shapes.add_picture(str(path), x, y, width=w, height=h)
    return None


def _pic_fit(slide, path, x, y, max_w, max_h):
    """Insert a picture scaled to fit inside the box, centred."""
    if not Path(path).exists():
        return None
    from PIL import Image
    iw, ih = Image.open(path).size
    scale = min(max_w / iw, max_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    return slide.shapes.add_picture(str(path), x + (max_w - w) // 2, y + (max_h - h) // 2, width=w, height=h)


def _stat(slide, x, y, value, label, color=NAVY):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, PI(2.75), PI(1.35))
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = PRGB(0xD5, 0xD8, 0xDC)
    _text(slide, x, y + PI(0.1), PI(2.75), PI(0.7), value, 30, True, color, PP_ALIGN.CENTER)
    _text(slide, x, y + PI(0.8), PI(2.75), PI(0.45), label, 13, color=GREY, align=PP_ALIGN.CENTER)


def _table(slide, x, y, w, header, rows, size=12, col_w=None):
    shape = slide.shapes.add_table(len(rows) + 1, len(header), x, y, w, PI(0.38) * (len(rows) + 1))
    t = shape.table
    for i, h in enumerate(header):
        c = t.cell(0, i)
        c.text = str(h)
        c.fill.solid()
        c.fill.fore_color.rgb = NAVY
        p = c.text_frame.paragraphs[0]
        p.font.size = PPt(size)
        p.font.bold = True
        p.font.color.rgb = PRGB(0xFF, 0xFF, 0xFF)
    for r, row in enumerate(rows, 1):
        for i, v in enumerate(row):
            c = t.cell(r, i)
            c.text = str(v)
            c.fill.solid()
            c.fill.fore_color.rgb = LIGHT if r % 2 else PRGB(0xFF, 0xFF, 0xFF)
            c.text_frame.paragraphs[0].font.size = PPt(size)
    if col_w:
        for i, cw in enumerate(col_w):
            t.columns[i].width = Emu(int(PI(cw)))
    return t


def build_pptx(m, per_class, summ, hist, comp, splits):
    o, b = m["overall"], m["binary"]
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    n = 0

    # 1 title
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    strip = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, PI(4.55), W, PI(0.08))
    strip.fill.solid()
    strip.fill.fore_color.rgb = ACCENT
    strip.line.fill.background()
    _text(s, PI(0.8), PI(1.5), PI(11.7), PI(2.6), TITLE, 38, True, PRGB(0xFF, 0xFF, 0xFF))
    _text(s, PI(0.8), PI(4.9), PI(11.7), PI(0.6),
          "Transfer learning · Grad-CAM explainability · HAM10000", 20, color=PRGB(0xD6, 0xEA, 0xF8))
    _text(s, PI(0.8), PI(5.6), PI(11.7), PI(0.5), f"Project presentation · {date.today():%B %Y}", 16,
          color=PRGB(0xD6, 0xEA, 0xF8))
    n += 1

    # 2 problem
    n += 1
    s = _slide(prs, "Problem Statement", n)
    _bullets(s, PI(0.6), PI(1.5), PI(7.2), PI(5.2), [
        "Skin cancer is the most common cancer worldwide; melanoma causes most skin-cancer deaths",
        "Early detection is critical - thin melanomas are usually curable",
        "Dermoscopy improves accuracy, but reading it is subjective and needs years of expertise",
        "Challenges for automation:",
        "  severe class imbalance (≈67% nevi)",
        "  visually similar classes (melanoma vs nevus vs keratosis)",
        "  hair, rulers and illumination artefacts",
        "  black-box predictions are not acceptable clinically",
    ], 18)
    _pic_fit(s, FIG / "preprocessing_steps.png", PI(7.9), PI(2.3), PI(5.1), PI(3.0))

    # 3 objectives
    n += 1
    s = _slide(prs, "Objectives", n)
    _bullets(s, PI(0.6), PI(1.5), PI(12), PI(5.2), [
        "Preprocess dermoscopic images: hair removal, CLAHE enhancement, resize 224×224, normalise",
        "Classify lesions into the 7 HAM10000 classes with transfer-learning CNNs",
        "Map predictions to Benign (nv, bkl, df, vasc) or Malignant (mel, bcc, akiec) with confidence scores",
        "Compare EfficientNet-B0, ResNet50 and DenseNet121",
        "Explain every prediction with Grad-CAM heat maps",
        "Report clinical metrics: sensitivity, specificity, F1, ROC-AUC, confusion matrices",
        "Deliver a Streamlit web app with PDF / CSV report export",
    ], 20)

    # 4 dataset
    n += 1
    s = _slide(prs, "Dataset: HAM10000", n)
    n_total = sum(len(v) for v in splits.values())
    _bullets(s, PI(0.5), PI(1.4), PI(4.4), PI(5.4), [
        f"{n_total:,} dermoscopic images, 600×450 px",
        f"{sum(v.lesion_id.nunique() for v in splits.values()):,} unique lesions, 7 diagnoses",
        ">50% confirmed by histopathology",
        "Split 70 / 15 / 15, stratified by class",
        "Grouped by lesion_id → no lesion appears in two splits (no leakage)",
        f"Train {len(splits['train']):,} · Val {len(splits['val']):,} · Test {len(splits['test']):,}",
    ], 16)
    _pic_fit(s, FIG / "class_distribution.png", PI(5.0), PI(1.5), PI(8.1), PI(5.3))

    # 5 architecture / pipeline
    n += 1
    s = _slide(prs, "System Architecture & Pipeline", n)
    _pic_fit(s, FIG / "pipeline.png", PI(0.4), PI(1.3), PI(12.5), PI(5.6))

    # 6 preprocessing
    n += 1
    s = _slide(prs, "Image Preprocessing", n)
    _pic_fit(s, FIG / "preprocessing_steps.png", PI(0.4), PI(1.3), PI(12.5), PI(3.3))
    _bullets(s, PI(0.6), PI(4.7), PI(12.2), PI(2.3), [
        f"Hair removal: black-hat ({config.HAIR_KERNEL_SIZE}×{config.HAIR_KERNEL_SIZE} kernel) → threshold → "
        f"dilate → Telea inpainting",
        f"CLAHE on LAB L-channel (clip {config.CLAHE_CLIP_LIMIT}, {config.CLAHE_TILE_GRID[0]}×"
        f"{config.CLAHE_TILE_GRID[1]} tiles) → resize 224×224 → ImageNet normalisation",
        "Augmentation (train only): flips, ±30° rotation, colour jitter, random zoom",
    ], 17)

    # 7 models
    n += 1
    s = _slide(prs, "Models: Transfer Learning", n)
    _table(s, PI(0.6), PI(1.5), PI(12.1), ["Backbone", "Key idea", "Params", "Grad-CAM layer"], [
        ["EfficientNet-B0 (default)", "Compound scaling, MBConv + squeeze-excitation", "4.0 M", "features[-1]"],
        ["ResNet50", "Residual bottleneck blocks, identity shortcuts", "23.5 M", "layer4[-1]"],
        ["DenseNet121", "Dense connectivity, feature reuse", "7.0 M", "denseblock4"],
    ], 15, [3.0, 5.2, 1.4, 2.5])
    _bullets(s, PI(0.6), PI(3.4), PI(12.1), PI(3.4), [
        "ImageNet pre-trained weights; new head: Dropout(0.3) → Linear(7); full fine-tuning",
        "Class-weighted cross-entropy (√balanced weights) + label smoothing 0.05",
        f"AdamW (lr {config.LEARNING_RATE}, wd {config.WEIGHT_DECAY}), ReduceLROnPlateau, early stopping "
        f"(patience {config.EARLY_STOPPING_PATIENCE}) on validation macro-F1",
        f"Device: CUDA → Apple MPS → CPU (this run: {summ['device'].upper()}, {summ['epochs_run']} epochs, "
        f"{summ['train_minutes']:.0f} min)",
    ], 18)

    # 8 grad-cam
    n += 1
    s = _slide(prs, "Explainable AI: Grad-CAM", n)
    _bullets(s, PI(0.5), PI(1.35), PI(12.3), PI(1.3), [
        "Hooks on the last conv layer capture activations A and gradients ∂y_c/∂A;  "
        "weights α_k = mean(∂y_c/∂A_k);  map = ReLU(Σ α_k A_k)",
        "Red / yellow = regions that pushed the model towards the predicted class",
    ], 16)
    _pic_fit(s, RES / MODEL / "gradcam_test_images.png", PI(0.4), PI(2.3), PI(12.5), PI(4.65))

    # 9 results overall
    n += 1
    s = _slide(prs, "Results: EfficientNet-B0 on Unseen Test Lesions", n)
    _stat(s, PI(0.5), PI(1.45), pct(o["accuracy"]), "Accuracy (7-class)")
    _stat(s, PI(3.6), PI(1.45), f"{o['macro_roc_auc']:.3f}", "Macro ROC-AUC")
    _stat(s, PI(6.7), PI(1.45), pct(b["sensitivity"]), "Malignant sensitivity", ACCENT)
    _stat(s, PI(9.8), PI(1.45), pct(b["specificity"]), "Malignant specificity", ACCENT)
    _table(s, PI(0.5), PI(3.1), PI(7.0), ["Class", "Sens.", "Spec.", "F1", "AUC", "n"],
           [[r["class"], f"{r.recall_sensitivity:.2f}", f"{r.specificity:.2f}", f"{r.f1:.2f}", f"{r.roc_auc:.3f}",
             int(r.support)] for _, r in per_class.iterrows()], 12)
    _pic_fit(s, RES / MODEL / "confusion_matrix.png", PI(7.7), PI(3.0), PI(5.3), PI(3.9))

    # 10 curves
    n += 1
    s = _slide(prs, "Results: ROC Curves & Training Curves", n)
    _pic_fit(s, RES / MODEL / "roc_curves.png", PI(0.4), PI(1.25), PI(12.5), PI(3.4))
    _pic_fit(s, RES / MODEL / "training_curves.png", PI(0.4), PI(4.7), PI(12.5), PI(2.25))

    # 11 comparison
    if comp is not None:
        n += 1
        s = _slide(prs, "Model Comparison", n)
        _table(s, PI(0.5), PI(1.4), PI(12.3),
               ["Model", "Params", "Accuracy", "Bal. acc.", "Macro F1", "Macro AUC", "Mal. sens.", "Mal. spec."],
               [[r.model, f"{r.params_millions} M", pct(r.accuracy), pct(r.balanced_accuracy), f"{r.macro_f1:.3f}",
                 f"{r.macro_roc_auc:.3f}", pct(r.bin_sensitivity), pct(r.bin_specificity)]
                for r in comp.itertuples()], 13)
        _pic_fit(s, RES / "model_comparison.png", PI(0.5), PI(3.0), PI(8.4), PI(3.95))
        best_file = RES / "best_model.json"
        if best_file.exists():
            best = json.loads(best_file.read_text())
            _text(s, PI(9.1), PI(3.2), PI(3.9), PI(0.5), f"Selected: {best['display_name']}", 20, True, ACCENT)
            _bullets(s, PI(9.1), PI(3.8), PI(3.9), PI(3.0), [
                "Score = 0.4·macro-F1 + 0.3·malignant sensitivity + 0.3·binary AUC",
                "Best macro-F1 and accuracy",
                "6× fewer parameters than ResNet50 → fast CPU inference",
                "ResNet50: higher AUC / malignant sensitivity (alternative)",
            ], 13)

    # 12 demo
    shots = sorted(SHOTS.glob("*.png"))
    for i, shot in enumerate(shots[:1]):
        n += 1
        s = _slide(prs, "Demo: Streamlit Web App" + (" (cont.)" if i else ""), n)
        _pic_fit(s, shot, PI(0.4), PI(1.25), PI(12.5), PI(5.7))

    # 13 conclusion
    n += 1
    s = _slide(prs, "Conclusion", n)
    _bullets(s, PI(0.6), PI(1.5), PI(12), PI(5.3), [
        "Built a complete, explainable pipeline: preprocessing → CNN → benign/malignant → Grad-CAM → PDF/CSV",
        f"EfficientNet-B0: {pct(o['accuracy'])} accuracy, macro AUC {o['macro_roc_auc']:.3f}, macro F1 "
        f"{o['macro_f1']:.3f} on lesion-disjoint test data",
        f"Malignancy detection: sensitivity {pct(b['sensitivity'])}, specificity {pct(b['specificity'])}, "
        f"AUC {b['roc_auc']:.3f}",
        "Lesion-level splitting gives honest, leakage-free estimates",
        "Grad-CAM focuses on the lesion and its borders → transparent, inspectable decisions",
        "Streamlit app makes the model usable without programming",
        "Research / education tool - not a medical diagnosis",
    ], 22)

    # 14 future scope
    n += 1
    s = _slide(prs, "Future Scope", n)
    _bullets(s, PI(0.6), PI(1.5), PI(12), PI(5.3), [
        "Larger datasets (ISIC 2019/2020) and external validation across skin types",
        "Lesion segmentation (U-Net) + patient metadata fusion",
        "Vision Transformers / ConvNeXt, ensembles and test-time augmentation",
        "Calibrated probabilities, uncertainty estimation, out-of-distribution rejection",
        "Grad-CAM++, SHAP and concept-based (ABCD rule) explanations",
        "Mobile deployment (quantisation) and clinical workflow integration",
    ], 23)

    # 15 thank you
    n += 1
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    _text(s, PI(0.8), PI(2.6), PI(11.7), PI(1.2), "Thank you", 48, True, PRGB(0xFF, 0xFF, 0xFF), PP_ALIGN.CENTER)
    _text(s, PI(0.8), PI(3.9), PI(11.7), PI(0.8), "Questions?", 26, color=PRGB(0xD6, 0xEA, 0xF8),
          align=PP_ALIGN.CENTER)

    out = config.DOCS_DIR / "Presentation.pptx"
    prs.save(out)
    print(f"Wrote {out} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    data = load_results()
    build_docx(*data)
    build_pptx(*data)

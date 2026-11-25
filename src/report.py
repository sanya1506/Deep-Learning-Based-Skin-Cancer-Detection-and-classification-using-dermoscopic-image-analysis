"""PDF and CSV report export for a single prediction.

Usage:
    python src/report.py --image sample_images/<file>.jpg      # writes results/reports/report_<file>.pdf
"""
import argparse
import csv
import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

DISCLAIMER = ("DISCLAIMER: This report is produced by an experimental deep-learning system for research and "
              "educational purposes only. It is NOT a medical diagnosis and must not replace examination by a "
              "qualified dermatologist. Any suspicious skin lesion should be assessed by a medical professional.")

CSV_FIELDS = ["timestamp", "filename", "model", "predicted_class", "predicted_name", "category",
              "confidence", "malignant_probability"] + [f"p_{c}" for c in config.CLASS_NAMES]


def _rl_image(arr: np.ndarray, width_cm: float) -> RLImage:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    h, w = arr.shape[:2]
    return RLImage(buf, width=width_cm * cm, height=width_cm * cm * h / w)


def result_to_row(result: dict, timestamp: str = None) -> dict:
    row = {
        "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": result.get("filename", "uploaded_image"),
        "model": result.get("model", ""),
        "predicted_class": result["pred_class"],
        "predicted_name": result["pred_full_name"],
        "category": result["binary_label"],
        "confidence": round(result["confidence"], 4),
        "malignant_probability": round(result["malignant_probability"], 4),
    }
    row.update({f"p_{c}": round(p, 4) for c, p in result["probabilities"].items()})
    return row


def csv_bytes(rows) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def append_csv_row(result: dict, path: Path) -> Path:
    path = Path(path)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(result_to_row(result))
    return path


def generate_pdf_report(result: dict, out_path=None) -> object:
    """Build the PDF. Writes to out_path if given and returns the path, else returns the PDF bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            title="Skin Lesion Analysis Report")
    ss = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=ss["Normal"], fontSize=8, leading=10, textColor=colors.HexColor("#555555"))
    is_mal = result["binary_label"] == "Malignant"
    badge_color = colors.HexColor("#c0392b") if is_mal else colors.HexColor("#1e8449")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    story = [
        Paragraph("Skin Lesion Analysis Report", ss["Title"]),
        Paragraph("Deep Learning-Based Skin Cancer Detection and Classification "
                  "Using Dermoscopic Image Analysis", ss["Italic"]),
        Spacer(1, 0.3 * cm),
    ]
    meta = Table([
        ["Generated", ts], ["Image", result.get("filename", "uploaded_image")],
        ["Model", result.get("model", "")],
    ], colWidths=[3.5 * cm, 13.5 * cm])
    meta.setStyle(TableStyle([("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    story += [meta, Spacer(1, 0.3 * cm)]

    summary = Table([
        ["Predicted class", f"{result['pred_class']}  -  {result['pred_full_name']}"],
        ["Category", result["binary_label"].upper()],
        ["Confidence", f"{result['confidence']:.2%}"],
        ["Malignant probability", f"{result['malignant_probability']:.2%}"],
    ], colWidths=[4.5 * cm, 12.5 * cm])
    summary.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f6f7")),
        ("TEXTCOLOR", (1, 1), (1, 1), badge_color), ("FONTNAME", (1, 1), (1, 1), "Helvetica-Bold"),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.grey), ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [summary, Spacer(1, 0.4 * cm)]

    w = 5.4
    imgs = Table([[_rl_image(result["original"], w), _rl_image(result["preprocessed"], w),
                   _rl_image(result["overlay"], w)],
                  ["Original", "Preprocessed", "Grad-CAM heatmap"]], colWidths=[5.7 * cm] * 3)
    imgs.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("FONTSIZE", (0, 1), (-1, 1), 9)]))
    story += [imgs, Spacer(1, 0.2 * cm),
              Paragraph("Grad-CAM highlights (red/yellow) the image regions that most influenced the "
                        "predicted class.", small), Spacer(1, 0.4 * cm)]

    rows = [["Class", "Description", "Type", "Confidence"]]
    for c, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        rows.append([c, config.CLASS_FULL_NAMES[c], config.BINARY_MAP[c], f"{p:.2%}"])
    conf = Table(rows, colWidths=[2 * cm, 9 * cm, 3 * cm, 3 * cm])
    conf.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f3f4")]),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#fdebd0")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.grey), ("ALIGN", (3, 0), (3, -1), "RIGHT"),
    ]))
    story += [Paragraph("Confidence scores (all classes)", ss["Heading3"]), conf, Spacer(1, 0.6 * cm),
              Paragraph(DISCLAIMER, ParagraphStyle("disc", parent=small, textColor=colors.HexColor("#922b21"),
                                                   borderColor=colors.HexColor("#922b21"), borderWidth=0.6,
                                                   borderPadding=6, fontSize=8.5, leading=11))]
    doc.build(story)
    pdf = buf.getvalue()
    if out_path is None:
        return pdf
    Path(out_path).write_bytes(pdf)
    return Path(out_path)


if __name__ == "__main__":
    from predict import Predictor

    p = argparse.ArgumentParser(description="Generate a PDF + CSV report for one image")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--model", default=config.DEFAULT_MODEL, choices=config.AVAILABLE_MODELS)
    a = p.parse_args()
    out_dir = config.RESULTS_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = Predictor(a.model).predict(a.image)
    print("PDF:", generate_pdf_report(res, out_dir / f"report_{a.image.stem}.pdf"))
    print("CSV:", append_csv_row(res, out_dir / "predictions.csv"))

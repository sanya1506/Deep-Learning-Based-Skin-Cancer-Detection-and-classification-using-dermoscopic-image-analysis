"""Single-image inference with Grad-CAM.

Usage:
    python src/predict.py --image sample_images/<file>.jpg
    python src/predict.py --image path/to/img.jpg --model resnet50 --pdf --csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from gradcam import GradCAM, overlay_heatmap  # noqa: E402
from models import DISPLAY_NAMES, get_target_layer, load_checkpoint  # noqa: E402
from preprocess import load_image_rgb, normalize, preprocess_image  # noqa: E402


def find_checkpoint(model_name: str = config.DEFAULT_MODEL) -> Path:
    """Prefer the fully trained checkpoint, fall back to the --quick one."""
    for p in (config.model_path(model_name), config.model_path(model_name, quick=True)):
        if p.exists():
            return p
    raise FileNotFoundError(f"No checkpoint for {model_name} in {config.MODELS_DIR}. "
                            f"Train first: python src/train.py --model {model_name}")


class Predictor:
    def __init__(self, model_name: str = config.DEFAULT_MODEL, checkpoint: Path = None, device=None):
        self.device = device or config.get_device()
        self.checkpoint = Path(checkpoint) if checkpoint else find_checkpoint(model_name)
        self.model, self.ckpt = load_checkpoint(self.checkpoint, self.device)
        self.model_name = self.ckpt["model_name"]
        self.display_name = DISPLAY_NAMES.get(self.model_name, self.model_name)
        self.class_names = self.ckpt["class_names"]
        self.cam = GradCAM(self.model, get_target_layer(self.model, self.model_name))

    def predict_array(self, img_rgb: np.ndarray) -> dict:
        pre = preprocess_image(img_rgb)
        x = torch.from_numpy(normalize(pre).transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        cam, idx, probs = self.cam(x)
        cls = self.class_names[idx]
        malignant_prob = float(sum(probs[self.class_names.index(c)] for c in config.MALIGNANT_CLASSES))
        return {
            "pred_class": cls,
            "pred_full_name": config.CLASS_FULL_NAMES[cls],
            "binary_label": config.BINARY_MAP[cls],
            "confidence": float(probs[idx]),
            "malignant_probability": malignant_prob,
            "probabilities": {c: float(p) for c, p in zip(self.class_names, probs)},
            "original": img_rgb,
            "preprocessed": pre,
            "cam": cam,
            "overlay": overlay_heatmap(pre, cam),
            "model": self.display_name,
        }

    def predict(self, path) -> dict:
        r = self.predict_array(load_image_rgb(path))
        r["filename"] = Path(path).name
        return r


def main():
    p = argparse.ArgumentParser(description="Classify a single dermoscopic image")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--model", default=config.DEFAULT_MODEL, choices=config.AVAILABLE_MODELS)
    p.add_argument("--pdf", action="store_true", help="also write a PDF report to results/reports/")
    p.add_argument("--csv", action="store_true", help="also append a row to results/reports/predictions.csv")
    a = p.parse_args()
    if not a.image.exists():
        p.error(f"image not found: {a.image}")

    r = Predictor(a.model).predict(a.image)
    print(f"\nImage      : {a.image}")
    print(f"Model      : {r['model']}")
    print(f"Prediction : {r['pred_class']} - {r['pred_full_name']}")
    print(f"Category   : {r['binary_label'].upper()}  (malignant probability {r['malignant_probability']:.1%})")
    print(f"Confidence : {r['confidence']:.2%}\n")
    print("Class confidences:")
    for c, prob in sorted(r["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {c:6s} {config.BINARY_MAP[c]:9s} {prob:7.2%}  {'#' * int(prob * 40)}")

    if a.pdf or a.csv:
        from report import append_csv_row, generate_pdf_report
        out_dir = config.RESULTS_DIR / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        if a.pdf:
            pdf = generate_pdf_report(r, out_dir / f"report_{a.image.stem}.pdf")
            print(f"\nPDF report: {pdf}")
        if a.csv:
            print(f"CSV row appended: {append_csv_row(r, out_dir / 'predictions.csv')}")
    print("\nDISCLAIMER: research/educational tool only - not a medical diagnosis.")


if __name__ == "__main__":
    main()

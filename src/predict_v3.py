"""Inference for app_v3.py: any single HAM10000 model, the selected ensemble, or the two-stage
pipeline (Stage 1 DermaCon-IN gate -> ensemble). Returns the same dict as src/predict.py (so the
PDF/CSV report code is shared) plus lesion outline, crop and gate information."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import engine  # noqa: E402
from gradcam import GradCAM, overlay_heatmap  # noqa: E402
from models import DISPLAY_NAMES, get_target_layer, load_checkpoint  # noqa: E402
from preprocess import preprocess_image  # noqa: E402

MAL = [config.CLASS_NAMES.index(c) for c in config.MALIGNANT_CLASSES]


def ensemble_weights() -> dict:
    f = config.STUDY_DIR / "ham" / "ensemble_choice.json"
    if not f.exists():
        return {}
    c = json.load(open(f))
    return {m: w for m, w in zip(c["models"], c["weights"]) if w > 0}


def gate_info():
    f = config.STUDY_DIR / "dermacon" / "table_6_8_two_stage.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    seed = d["gate_on_ham_val"]["seed"]
    return {"seed": seed, "threshold": d["thresholds"][str(seed)] if str(seed) in d["thresholds"] else d["thresholds"][seed],
            "path": config.MODELS_DIR / f"gate_s{seed}.pth"}


class PredictorV3:
    def __init__(self, mode: str = "ensemble"):
        """mode: a model name from config.HAM_MODELS, 'ensemble' or 'two_stage'."""
        self.mode = mode
        self.device = config.get_device()
        names = list(ensemble_weights()) if mode in ("ensemble", "two_stage") else [mode]
        self.weights = ensemble_weights() if mode in ("ensemble", "two_stage") else {mode: 1.0}
        self.models = {n: load_checkpoint(config.MODELS_DIR / f"{n}.pth", self.device)[0] for n in names}
        self.gate = None
        if mode == "two_stage":
            g = gate_info()
            gm, gck = load_checkpoint(g["path"], self.device)
            self.gate = (gm, g["threshold"], [i for i, c in enumerate(gck["class_names"]) if "neoplas" in c.lower()][0])
        self.segmenter = None
        unets = sorted(config.MODELS_DIR.glob("unet_s*.pth"))
        if unets:
            import studies
            self.segmenter = studies.Segmenter()
        self.display_name = {"ensemble": "Ensemble (selected on validation)",
                             "two_stage": "Two-stage (DermaCon-IN gate -> ensemble)"}.get(mode, DISPLAY_NAMES.get(mode, mode))
        self.tf = engine.eval_transform()

    @torch.no_grad()
    def _probs(self, model, x):
        views = [x, x.flip(3), x.flip(2)]
        return (sum(torch.softmax(model(v), 1) for v in views) / 3)[0].cpu().numpy()

    def predict_array(self, img_rgb: np.ndarray) -> dict:
        pre = preprocess_image(img_rgb)
        x = self.tf(pre)[None].to(self.device)
        probs = sum(w * self._probs(m, x) for (n, m), w in zip(self.models.items(), self.weights.values()))
        probs = probs / probs.sum()
        idx = int(probs.argmax()); cls = config.CLASS_NAMES[idx]
        cam = 0
        for (n, m), w in zip(self.models.items(), self.weights.values()):
            gc = GradCAM(m, get_target_layer(m, n))
            c, _, _ = gc(x, class_idx=idx)
            gc.remove()
            cam = cam + w * c
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        outline, crop = img_rgb, None
        if self.segmenter is not None:
            import studies
            mask = self.segmenter(img_rgb)
            outline = img_rgb.copy()
            cnts, _ = cv2.findContours(studies.largest_component(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(outline, cnts, -1, (0, 255, 0), max(2, img_rgb.shape[1] // 200))
            crop, _ = studies.crop_with_mask(img_rgb, mask)
        gate = None
        if self.gate is not None:
            gm, t, neo = self.gate
            p_neo = float(self._probs(gm, x)[neo])
            gate = {"p_neoplasm": p_neo, "threshold": t, "passed": p_neo >= t}
        p_mal = float(probs[MAL].sum())
        res = {"pred_class": cls, "pred_full_name": config.CLASS_FULL_NAMES[cls],
               "binary_label": config.BINARY_MAP[cls], "confidence": float(probs[idx]),
               "malignant_probability": p_mal, "probabilities": dict(zip(config.CLASS_NAMES, map(float, probs))),
               "original": img_rgb, "preprocessed": pre, "cam": cam, "overlay": overlay_heatmap(pre, cam),
               "outline": outline, "crop": crop, "gate": gate, "model": self.display_name}
        if gate is not None and not gate["passed"]:
            res["binary_label"] = "Not routed (non-tumour by gate)"
        return res


def validation_operating_points():
    """Malignant-probability thresholds -> validation sensitivity/specificity of the ensemble."""
    f = config.STUDY_DIR / "ham" / "ensemble_probs.npz"
    if not f.exists():
        return None
    import ham
    pv = np.load(f)["val"]
    y = np.isin(ham.splits()["val"].label.to_numpy(), MAL)
    s = pv[:, MAL].sum(1)
    ts = np.round(np.arange(0.05, 0.96, 0.05), 2)
    return {float(t): (float((s[y] >= t).mean()), float((s[~y] < t).mean())) for t in ts}

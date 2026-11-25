"""Inference with the v2 ensemble. Returns the same dict as src/predict.py's Predictor,
so the Streamlit app can switch with a one-line change:

    from improved.predict_v2 import EnsemblePredictor as Predictor   # in app.py

Pipeline: U-Net lesion crop (Otsu fallback) -> Shades-of-Gray -> every ensemble member with
8-view TTA -> weighted log-prob ensemble + calibrated class bias -> Grad-CAM from the first CNN member.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE.parent))
import config  # noqa: E402
from common import CLASS_NAMES, MALIGNANT_IDX, V2_MODELS, V2_RESULTS, encode_meta, site_group  # noqa: E402
from gradcam import GradCAM, overlay_heatmap  # noqa: E402  (existing project implementation)
from segmentation import SEG_CKPT, LesionSegmenter, prepare_image  # noqa: E402
from train_cls import MEAN, STD, TTA_VIEWS, LesionNet, gradcam_target_layer  # noqa: E402


class _BindMeta(nn.Module):
    """Lets the single-input GradCAM class drive a model that also takes metadata."""

    def __init__(self, model, meta):
        super().__init__()
        self.model, self.meta = model, meta

    def forward(self, x):
        return self.model(x, self.meta)


class EnsemblePredictor:
    def __init__(self, device=None):
        self.device = device or config.get_device()
        cfg = json.load(open(V2_RESULTS / "ensemble" / "ensemble_config.json"))
        self.weights, self.bias = np.array(cfg["weights"]), np.array(cfg["bias"])
        self.threshold = cfg["binary_threshold"]
        self.members = []
        for tag in cfg["members"]:
            ck = torch.load(V2_MODELS / f"{tag}.pth", map_location="cpu", weights_only=False)
            m = LesionNet(ck["arch"], ck["meta"], pretrained=False)
            m.load_state_dict(ck["state_dict"]); m.to(self.device).eval()
            self.members.append((tag, m, ck))
        self.segmenter = LesionSegmenter(device=self.device) if SEG_CKPT.exists() else None
        self.class_names = CLASS_NAMES
        self.display_name = "v2 ensemble (" + ", ".join(t for t, _, _ in self.members) + ")"

    def _tensor(self, img, size):
        import cv2
        x = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = (x - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
        return torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.device)

    @torch.no_grad()
    def _member_logp(self, model, x, meta):
        p = 0
        for flip, k in TTA_VIEWS:
            v = torch.rot90(x.flip(3) if flip else x, k, (2, 3))
            p = p + F.softmax(model(v, meta).float(), 1)
        return torch.log(p / len(TTA_VIEWS) + 1e-9)[0].cpu().numpy()

    def predict_array(self, img_rgb: np.ndarray, age=None, sex=None, localization=None) -> dict:
        pre = prepare_image(img_rgb, self.segmenter)
        meta_vec = torch.from_numpy(encode_meta(age, sex, site_group(localization, "ham")))[None].to(self.device)
        z = 0
        for w, (tag, m, ck) in zip(self.weights, self.members):
            meta = meta_vec if ck["meta"] else torch.zeros_like(meta_vec)
            z = z + w * self._member_logp(m, self._tensor(pre, ck["size"]), meta)
        z = z + self.bias
        probs = np.exp(z - z.max()); probs /= probs.sum()
        idx = int(probs.argmax()); cls = CLASS_NAMES[idx]

        cam_member = next(((t, m, c) for t, m, c in self.members if "swin" not in c["arch"]), self.members[0])
        _, m, ck = cam_member
        meta = meta_vec if ck["meta"] else torch.zeros_like(meta_vec)
        wrapped = _BindMeta(m, meta)
        gc = GradCAM(wrapped, gradcam_target_layer(m))
        cam, _, _ = gc(self._tensor(pre, ck["size"]), class_idx=idx)
        gc.remove()

        p_mal = float(probs[MALIGNANT_IDX].sum())
        return {
            "pred_class": cls,
            "pred_full_name": config.CLASS_FULL_NAMES[cls],
            # the class decides the label, as in v1; malignant_probability is reported alongside
            "binary_label": config.BINARY_MAP[cls],
            "confidence": float(probs[idx]),
            "malignant_probability": p_mal,
            "probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
            "original": img_rgb,
            "preprocessed": pre,
            "cam": cam,
            "overlay": overlay_heatmap(pre, cam),
            "model": self.display_name,
        }

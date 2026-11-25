"""Grad-CAM (Selvaraju et al., 2017) implemented from scratch with PyTorch hooks.

    alpha_k  = mean_{i,j} dy_c / dA^k_{ij}            (channel importance weights)
    L_c      = ReLU( sum_k alpha_k * A^k )             (class activation map)

Usage:
    python src/gradcam.py --image sample_images/mel_ISIC_0000000.jpg
    python src/gradcam.py --all-samples              # every image in sample_images/
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()
        # tensor hook: called with dL/d(output) during backward
        # (skipped when the model is run without gradients, e.g. the deletion/insertion curves under no_grad)
        if output.requires_grad:
            output.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self.gradients = grad.detach()

    def __call__(self, x: torch.Tensor, class_idx: int = None):
        """x: (1,3,H,W). Returns (cam HxW in [0,1], class_idx, probs)."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, 1)[0].detach().cpu().numpy()
            if class_idx is None:
                class_idx = int(logits.argmax(1).item())
            logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)            # (1,K,1,1)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))    # (1,1,h,w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, probs

    def remove(self):
        self._handle.remove()


def overlay_heatmap(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a JET-coloured CAM over an RGB uint8 image."""
    cam = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8(np.clip((1 - alpha) * img_rgb + alpha * heat, 0, 255))


def save_figure(original, preprocessed, overlay, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for ax, im, t in zip(axes, [original, preprocessed, overlay],
                         ["Original", "Preprocessed (hair removal + CLAHE)", "Grad-CAM overlay"]):
        ax.imshow(im)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    from predict import Predictor

    p = argparse.ArgumentParser(description="Generate Grad-CAM visualisations")
    p.add_argument("--image", type=Path, help="path to a dermoscopic image")
    p.add_argument("--all-samples", action="store_true", help="run on every file in sample_images/")
    p.add_argument("--model", default=config.DEFAULT_MODEL, choices=config.AVAILABLE_MODELS)
    p.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR / "gradcam")
    a = p.parse_args()

    images = sorted(config.SAMPLE_DIR.glob("*.jpg")) if a.all_samples else [a.image]
    if not images or images[0] is None:
        p.error("give --image PATH or --all-samples")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    predictor = Predictor(model_name=a.model)
    for img_path in images:
        r = predictor.predict(img_path)
        title = (f"{img_path.name}: predicted {r['pred_class']} ({r['confidence']:.1%}) - "
                 f"{r['binary_label']}")
        out = a.out_dir / f"gradcam_{img_path.stem}.png"
        save_figure(r["original"], r["preprocessed"], r["overlay"], title, out)
        print(f"{title}  ->  {out}")

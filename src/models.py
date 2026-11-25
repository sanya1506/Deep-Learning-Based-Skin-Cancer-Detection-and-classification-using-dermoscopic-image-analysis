"""ImageNet-pretrained backbones (torchvision) with a new head: dropout -> linear.

Six HAM10000 models (report Table 6.1): EfficientNet-B0, EfficientNet-B0 (long run), ResNet50,
DenseNet121, ConvNeXt-Tiny and Swin-T. The same builder makes the 8-class Stage 1 gate.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

DISPLAY_NAMES = {
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b0_long": "EfficientNet-B0 (long)",
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "convnext_tiny": "ConvNeXt-Tiny",
    "swin_t": "Swin-T",
}


def arch_of(name: str) -> str:
    return config.ARCH_OF.get(name, name)


def build_model(name: str, num_classes: int = config.NUM_CLASSES, pretrained: bool = True,
                dropout: float = 0.3) -> nn.Module:
    arch = arch_of(name)
    head = lambda f: nn.Sequential(nn.Dropout(dropout), nn.Linear(f, num_classes))  # noqa: E731
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = head(m.classifier[1].in_features)
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = head(m.fc.in_features)
    elif arch == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = head(m.classifier.in_features)
    elif arch == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[2] = head(m.classifier[2].in_features)
    elif arch == "swin_t":
        m = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
        m.head = head(m.head.in_features)
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from {config.AVAILABLE_MODELS}")
    return m


def get_target_layer(model: nn.Module, name: str) -> nn.Module:
    """Grad-CAM target: last convolutional stage, output must be (N, C, H, W)."""
    arch = arch_of(name)
    if arch == "efficientnet_b0":
        return model.features[-1]
    if arch == "resnet50":
        return model.layer4[-1]
    if arch == "densenet121":
        # denseblock4 output (norm5 is followed by an in-place ReLU, which breaks tensor hooks)
        return model.features.denseblock4
    if arch == "convnext_tiny":
        return model.features[-1]
    if arch == "swin_t":
        # Swin features are (N, H, W, C); model.permute turns them into (N, C, H, W)
        return model.permute
    raise ValueError(name)


def load_checkpoint(path, device=None):
    """Load a checkpoint saved by the training code. Returns (model, checkpoint_dict)."""
    device = device or config.get_device()
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    n = ckpt.get("num_classes", len(ckpt["class_names"]))
    model = build_model(ckpt.get("arch", ckpt["model_name"]), n, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

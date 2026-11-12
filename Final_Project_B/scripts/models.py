"""Model definitions for the B stage: image-only baseline and metadata fusion.

Both models are trained from the same ImageNet-pretrained ResNet-50
(``ResNet50_Weights.IMAGENET1K_V2``) so that the only difference between the
baseline and the fusion model is the presence of the metadata stream. That is
what makes the paired Benefit/Neutral/Harm comparison meaningful later.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

IMAGE_FEATURE_DIM = 2048


def _resnet50_backbone(pretrained: bool) -> nn.Sequential:
    """ResNet-50 without its global-average-pool flattening head."""
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = torchvision.models.resnet50(weights=weights)
    return nn.Sequential(*list(model.children())[:-1])


class ImageOnlyNet(nn.Module):
    """Baseline: dermoscopic image -> 7-class logits."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.0, pretrained: bool = True):
        super().__init__()
        self.features = _resnet50_backbone(pretrained)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(IMAGE_FEATURE_DIM, num_classes)

    def forward(self, image: torch.Tensor, meta: torch.Tensor | None = None) -> torch.Tensor:
        features = torch.flatten(self.features(image), 1)
        return self.classifier(self.dropout(features))


class MetadataBranch(nn.Module):
    """Small MLP projecting tabular metadata into the visual feature space."""

    def __init__(self, n_features: int, hidden=(64, 128), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for width in hidden:
            layers += [
                nn.Linear(prev, width),
                nn.BatchNorm1d(width),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = width
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        return self.net(meta)


class TwoStreamNet(nn.Module):
    """Image branch + metadata branch, concatenated at mid-level, then fused."""

    def __init__(
        self,
        n_features: int,
        num_classes: int = 7,
        meta_hidden=(64, 128),
        fusion_dim: int = 256,
        dropout: float = 0.3,
        meta_dropout: float = 0.2,
        image_dropout: float = 0.2,
        pretrained: bool = True,
    ):
        super().__init__()
        if n_features <= 0:
            raise ValueError("TwoStreamNet needs at least one metadata feature")
        self.features = _resnet50_backbone(pretrained)
        self.meta = MetadataBranch(n_features, meta_hidden, meta_dropout)
        self.image_dropout = nn.Dropout(image_dropout)
        self.fusion = nn.Sequential(
            nn.Linear(IMAGE_FEATURE_DIM + self.meta.out_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, image: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        visual = self.image_dropout(torch.flatten(self.features(image), 1))
        tabular = self.meta(meta)
        return self.fusion(torch.cat([visual, tabular], dim=1))


class SeparableConvBlock(nn.Module):
    """Depthwise-separable convolution + BatchNorm + Swish, as described by the
    sequential-CNN HAM10000 papers (cheap convolutions, dilated receptive field)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel, padding=padding, dilation=dilation,
                      groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SeqCNN(nn.Module):
    """Small sequential CNN: the architecture family used by the 96%+ papers.

    Deliberately *not* a pretrained network, so training starts from scratch the way
    those papers describe. Intended for the low-resolution (28 px) reproduction.
    """

    def __init__(self, num_classes: int = 2, width: int = 32, dropout: float = 0.4):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            SeparableConvBlock(3, w),
            SeparableConvBlock(w, w),
            nn.MaxPool2d(2),
            SeparableConvBlock(w, w * 2, dilation=2),
            SeparableConvBlock(w * 2, w * 2),
            nn.MaxPool2d(2),
            SeparableConvBlock(w * 2, w * 4, dilation=2),
            SeparableConvBlock(w * 4, w * 4),
            nn.MaxPool2d(2),
            SeparableConvBlock(w * 4, w * 4, dilation=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(w * 4, num_classes),
        )

    def forward(self, image: torch.Tensor, meta: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(self.features(image))


def build_model(cfg: dict, n_features: int, num_classes: int = 7) -> nn.Module:
    """Instantiate the model described by ``cfg['model']``."""
    model_cfg = cfg["model"]
    kind = model_cfg["kind"]
    pretrained = bool(model_cfg.get("pretrained", True))

    if kind == "image_only":
        return ImageOnlyNet(
            num_classes=num_classes,
            dropout=float(model_cfg.get("dropout", 0.0)),
            pretrained=pretrained,
        )
    if kind == "fusion":
        return TwoStreamNet(
            n_features=n_features,
            num_classes=num_classes,
            meta_hidden=tuple(model_cfg.get("meta_hidden", (64, 128))),
            fusion_dim=int(model_cfg.get("fusion_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.3)),
            meta_dropout=float(model_cfg.get("meta_dropout", 0.2)),
            image_dropout=float(model_cfg.get("image_dropout", 0.2)),
            pretrained=pretrained,
        )
    if kind == "seqcnn":
        return SeqCNN(
            num_classes=num_classes,
            width=int(model_cfg.get("width", 32)),
            dropout=float(model_cfg.get("dropout", 0.4)),
        )
    raise ValueError(f"unknown model kind: {kind!r}")


def describe_model(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{type(model).__name__}: {total/1e6:.1f}M parameters ({trainable/1e6:.1f}M trainable)"

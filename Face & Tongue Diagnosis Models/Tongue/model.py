# model.py
from __future__ import annotations

from typing import Dict, Callable
import torch
import torch.nn as nn
import torchvision.models as tvm
import timm 

_BACKBONE_FACTORY: Dict[str, Callable[[], nn.Module]] = {
    "resnet50": lambda: tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2),
    "densenet121": lambda: tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1),
    "efficientnet_b0": lambda: timm.create_model("efficientnet_b0", pretrained=True),
}
_BACKBONE_FACTORY.update({
    # 22K‑pretrained then 1K‑fine‑tuned weights, 224‑pixel input
    "convnext_tiny": lambda: timm.create_model(
        "convnext_tiny",  # or "convnext_tiny.fb_in22k" for raw 22K 384px
        pretrained=True),
})

def _get_feat_dim(backbone_name: str, backbone: nn.Module) -> int:
    if backbone_name.startswith("resnet"):
        return backbone.fc.in_features
    if backbone_name.startswith("densenet"):
        return backbone.classifier.in_features
    if backbone_name.startswith("efficientnet"):
        clf = backbone.classifier
        return clf[-1].in_features if isinstance(clf, nn.Sequential) else clf.in_features
    if backbone_name.startswith("convnext"):
        # torchvision variant has `classifier`; timm variant has `head`
        if hasattr(backbone, "classifier"):
            clf = backbone.classifier
            return clf[-1].in_features if isinstance(clf, nn.Sequential) else clf.in_features
        return backbone.head.in_features
    raise ValueError(f"Unsupported backbone {backbone_name}")


class MultiTaskNet(nn.Module):
    """Backbone + independent linear heads for each task."""

    def __init__(
        self,
        num_classes_dict: Dict[str, int],
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        if backbone not in _BACKBONE_FACTORY:
            raise ValueError(f"backbone must be one of {list(_BACKBONE_FACTORY)}, got {backbone}")

        # instantiate backbone
        self.backbone_name = backbone
        self.backbone: nn.Module = _BACKBONE_FACTORY[backbone]()
        if not pretrained:
            # re‑initialise weights if pretrained is False
            for p in self.backbone.parameters():
                if p.requires_grad:
                    nn.init.normal_(p.data, mean=0.0, std=0.02)

        feat_dim = _get_feat_dim(backbone, self.backbone)
        # strip classifier head
        if backbone.startswith("resnet"):
            self.backbone.fc = nn.Identity()
        elif backbone.startswith("densenet"):
            self.backbone.classifier = nn.Identity()
        elif backbone.startswith("efficientnet"):
            self.backbone.classifier = nn.Identity()
        elif backbone.startswith("convnext"):
            if hasattr(self.backbone, "classifier"):
                self.backbone.classifier = nn.Identity()
            else:
                self.backbone.head = nn.Identity()
        else:
            raise RuntimeError("Unexpected backbone switch failure")

        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.heads = nn.ModuleDict({
            name: nn.Linear(feat_dim, n_cls) for name, n_cls in num_classes_dict.items()
        })
        # self.heads = nn.ModuleDict({
        #     name: nn.Sequential(nn.Dropout(p=0.15), nn.Linear(feat_dim, n_cls)) for name, n_cls in num_classes_dict.items()
        # })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        if feat.ndim == 4:
            feat = feat.mean(dim=[2, 3])
        feat = self.dropout(feat)
        return {name: head(feat) for name, head in self.heads.items()}

"""Multi‑task face/tongue classifier with ConvNeXt‑v2 + OLS options

Key features
- Supports powerful timm backbones:
    * "convnextv2_base.fcmae_ft_in22k_in1k_384"
    * "convnext_nano_ols.d1h_in1k"
  (and keeps your older torchvision/timm convnext for continuity.)
- Clean separation of concerns:
    * model handles feature extraction + multi‑head logits
    * trainer decides the policy and calls model.freeze_policy(...)
- Robust partial unfreezing:
    * freeze all, train only heads (default)
    * train last K stages of the backbone
    * or unfreeze everything
- Optimizer param‑groups helper for different LRs on backbone vs heads
- Optional dropout before heads
- Optional head bias init from label priors (mitigates biased top‑3)

Notes
- For timm models we build with num_classes=0 and global_pool='avg' so the
  forward returns pooled features (B, C) directly.
- For ConvNeXt variants, stages are detected via `model.stages`.
- Make sure your dataloader image size matches the chosen backbone (e.g., 384
  for convnextv2_base... if you want to follow pretrain resolution).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm
import timm
try:
    import timm
    from timm.layers import SelectAdaptivePool2d
except Exception as e:  # pragma: no cover
    raise ImportError("This module requires 'timm'. Please `pip install timm`.\n" + str(e))
# --------------------------- Backbone creation ---------------------------

def _make_timm_backbone(name: str,
                        pretrained: bool = True,
                        in_chans: int = 3,
                        global_pool: str = 'avg') -> nn.Module:
    """Create a timm model that *outputs pooled features* (B, C).
    This is achieved by setting num_classes=0.
    """
    m = timm.create_model(
        name,
        pretrained=pretrained,
        in_chans=in_chans,
        num_classes=0,           # <- pooled features out
        global_pool=global_pool, # 'avg' or 'max'
        features_only=False,
    )
    if not hasattr(m, 'num_features'):
        # Fallback: try to infer feature dim later via a dummy forward if needed
        m.num_features = None  # type: ignore[attr-defined]
    return m


def _make_torchvision_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Torchvision models: strip the classifier and return (model, feat_dim)."""
    if name == 'resnet50':
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat_dim
    if name == 'densenet121':
        m = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = m.classifier.in_features
        m.classifier = nn.Identity()
        return m, feat_dim
    raise ValueError(f"Unsupported torchvision backbone: {name}")


# ------------------------------- Model -----------------------------------
@dataclass
class FreezePolicy:
    # How many final stages of the backbone to train. -1 => all, 0 => none (heads only)
    train_last_k_stages: int = 0
    # If True, keep BatchNorm/LayerNorm in eval mode when frozen; set False to train norms
    freeze_norm_layers: bool = True


class MultiTaskNet(nn.Module):
    def __init__(
        self,
        num_classes_dict: Dict[str, int],
        backbone: str = 'convnextv2_base.fcmae_ft_in22k_in1k_384',
        pretrained: bool = True,
        in_chans: int = 3,
        dropout: float | None = 0.0,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone

        # --- Build backbone
        timm_ok = backbone in (
            'convnextv2_base.fcmae_ft_in22k_in1k_384',
            'convnext_nano_ols.d1h_in1k',
            'convnext_tiny',
        ) or backbone in timm.list_models(backbone)

        if timm_ok:
            self.backbone = _make_timm_backbone(backbone, pretrained=pretrained, in_chans=in_chans)
            feat_dim = getattr(self.backbone, 'num_features', None)
            if feat_dim is None or isinstance(feat_dim, (list, tuple)):
                # Robustly infer by a dummy forward (lazy), avoid allocating on init
                feat_dim = None
        elif backbone in {'resnet50', 'densenet121'}:
            self.backbone, feat_dim = _make_torchvision_backbone(backbone, pretrained=pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self._feature_dim_hint = feat_dim  # may be None and lazily inferred

        # --- Heads
        self.dropout = nn.Dropout(dropout) if (dropout and dropout > 0) else nn.Identity()
        self.heads = nn.ModuleDict({
            name: nn.Linear(self.feature_dim(), n_cls) for name, n_cls in num_classes_dict.items()
        })

        # default: freeze all backbone params; trainer can call freeze_policy to change
        self.freeze_policy(FreezePolicy(train_last_k_stages=0))

    # --------------------------- Feature dim ---------------------------
    def feature_dim(self) -> int:
        if self._feature_dim_hint is not None:
            return int(self._feature_dim_hint)
        # Lazy inference via dummy tensor (keeps code robust across models)
        with torch.no_grad():
            dev = next(self.parameters()).device
            x = torch.zeros(1, 3, 224, 224, device=dev)  # size doesn't matter; model pools
            f = self.backbone(x)
            if f.ndim == 4:
                f = f.mean(dim=(2, 3))
            self._feature_dim_hint = f.shape[-1]
        return int(self._feature_dim_hint)

    # --------------------------- Freezing utils ------------------------
    def _iter_stages(self) -> List[nn.Module]:
        m = self.backbone
        # ConvNeXt(v1/v2) in timm expose `.stages`
        if hasattr(m, 'stages') and isinstance(m.stages, nn.Sequential):
            return [s for s in m.stages]
        # Torchvision ResNet
        if self.backbone_name.startswith('resnet'):
            return [m.layer1, m.layer2, m.layer3, m.layer4]
        # Torchvision DenseNet
        if self.backbone_name.startswith('densenet'):
            fs = m.features
            return [fs.denseblock1, fs.denseblock2, fs.denseblock3, fs.denseblock4]
        # Fallback: single block
        return [m]

    @staticmethod
    def _set_requires_grad(module: nn.Module, flag: bool) -> None:
        for p in module.parameters():
            p.requires_grad = flag

    def _set_norm_eval(self, module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def freeze_policy(self, policy: FreezePolicy) -> None:
        """Apply a freezing policy to the backbone; heads always trainable."""
        # Freeze everything first
        self._set_requires_grad(self.backbone, False)

        stages = self._iter_stages()
        k = policy.train_last_k_stages
        if k == -1:
            # unfreeze all stages
            for s in stages:
                self._set_requires_grad(s, True)
        elif k > 0:
            # unfreeze last k stages
            for s in stages[-k:]:
                self._set_requires_grad(s, True)
        # else: keep all frozen

        if policy.freeze_norm_layers:
            self._set_norm_eval(self.backbone)

        # heads always trainable
        for h in self.heads.values():
            self._set_requires_grad(h, True)

    # --------------------------- Optimizer groups ----------------------
    def param_groups(self, lr_backbone: float, lr_heads: float, weight_decay: float = 0.0):
        bb_params: List[nn.Parameter] = []
        head_params: List[nn.Parameter] = []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (head_params if n.startswith('heads.') else bb_params).append(p)
        groups = []
        if bb_params:
            groups.append({"params": bb_params, "lr": lr_backbone, "weight_decay": weight_decay})
        if head_params:
            groups.append({"params": head_params, "lr": lr_heads, "weight_decay": weight_decay})
        return groups

    # --------------------------- Bias init (optional) ------------------
    @torch.no_grad()
    def init_head_bias_from_priors(self, priors: Dict[str, Iterable[float]]):
        """Set classifier bias to log(p/(1-p)) per class for better early calibration.
        `priors[task]` is a list/array that sums to 1.0.
        """
        import math
        for task, head in self.heads.items():
            if task not in priors:
                continue
            p = torch.tensor(list(priors[task]), dtype=head.bias.dtype, device=head.bias.device)
            p = p.clamp_(1e-6, 1 - 1e-6)
            head.bias.copy_(torch.log(p / (1 - p)))

    # --------------------------- Forward --------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)  # expected shape (B, C) thanks to num_classes=0
        if feat.ndim == 4:
            feat = feat.mean(dim=(2, 3))
        feat = self.dropout(feat)
        return {name: head(feat) for name, head in self.heads.items()}

_SANITY_SMALL_KEYS = ("atto", "femto", "pico", "nano")

def _find_first(items: List[str], *needles: str) -> Optional[str]:
    for needle in needles:
        for it in items:
            if needle in it:
                return it
    return items[0] if items else None

def sanity_list_supported_variants() -> Dict[str, List[str]]:
    """Map 'convnext_atto/femto/pico/nano' -> available timm model names (pretrained only)."""
    out: Dict[str, List[str]] = {}
    for key in _SANITY_SMALL_KEYS:
        names = sorted(set(
            timm.list_models(f"*convnext*{key}*", pretrained=True) +
            timm.list_models(f"*convnextv2*{key}*", pretrained=True)
        ))
        out[f"convnext_{key}"] = names
    return out

def sanity_resolve_timm_name(variant: str, prefer_ols: bool = True) -> str:
    """Resolve logical variant (e.g. 'convnext_atto') to concrete timm model name."""
    var = variant.lower().strip()
    if not any(k in var for k in _SANITY_SMALL_KEYS):
        raise ValueError(f"Unknown variant='{variant}'. Expected one of {_SANITY_SMALL_KEYS}.")
    key = next(k for k in _SANITY_SMALL_KEYS if k in var)
    candidates = sorted(set(
        timm.list_models(f"*convnext*{key}*", pretrained=True) +
        timm.list_models(f"*convnextv2*{key}*", pretrained=True)
    ))
    if not candidates:
        raise RuntimeError(f"No pretrained timm models found for variant='{variant}'.")
    if prefer_ols:
        chosen = _find_first(candidates, "_ols.", ".ols.", "-ols.")
    else:
        chosen = _find_first(candidates, ".d1_in1k")
    return chosen or candidates[0]

class SanityConvNeXt(nn.Module):
    """Thin wrapper around timm ConvNeXt/ConvNeXtV2 for 3-class sanity check."""
    def __init__(self,
                 timm_name: str,
                 num_classes: int = 3,
                 pretrained: bool = True,
                 global_pool: str = "avg",
                 drop_rate: float = 0.0,
                 drop_path_rate: float = 0.1) -> None:
        super().__init__()
        self.timm_name = timm_name
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            global_pool=global_pool,
        )
        if hasattr(self.backbone, "reset_classifier"):
            self.backbone.reset_classifier(num_classes=num_classes, global_pool=global_pool)
        self.num_features: Optional[int] = getattr(self.backbone, "num_features", None)
        if self.num_features is None:
            clf = self._get_classifier_module()
            if isinstance(clf, nn.Linear):
                self.num_features = clf.in_features
        self.global_pool = getattr(self.backbone, "global_pool",
                                   SelectAdaptivePool2d(pool_type=global_pool))

    def _get_classifier_module(self) -> nn.Module:
        name_or_mod = self.backbone.get_classifier()
        if isinstance(name_or_mod, str):
            return getattr(self.backbone, name_or_mod)
        return name_or_mod

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if not return_features:
            return self.backbone(x)
        if hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x)
            pooled = self.global_pool(feats)
            # Some timm versions return name vs module; handle both
            clf = self._get_classifier_module()
            logits = clf(pooled) if isinstance(clf, nn.Module) else self.backbone(x)
            return logits, pooled
        return self.backbone(x), None

    @torch.no_grad()
    def predict(self, x: torch.Tensor, apply_softmax: bool = True) -> torch.Tensor:
        self.eval()
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1) if apply_softmax else logits

    def freeze_backbone(self, trainable_keywords: Tuple[str, ...] = ("head","classifier","fc")) -> None:
        for n, p in self.backbone.named_parameters():
            keep = any(kw in n for kw in trainable_keywords)
            p.requires_grad = bool(keep)

    def unfreeze_all(self) -> None:
        for _, p in self.backbone.named_parameters():
            p.requires_grad = True

    def load_checkpoint(self, path: str, strict: bool = True) -> None:
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        self.load_state_dict(state, strict=strict)

@dataclass
class SanityConfig:
    variant: str = "convnext_atto"   # <- your chosen deployment model
    num_classes: int = 3
    pretrained: bool = True
    global_pool: str = "avg"
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    prefer_ols: bool = True          # prefer *_ols if available

def build_sanity_model(cfg: SanityConfig = SanityConfig()) -> SanityConvNeXt:
    timm_name = sanity_resolve_timm_name(cfg.variant, prefer_ols=cfg.prefer_ols)
    return SanityConvNeXt(
        timm_name=timm_name,
        num_classes=cfg.num_classes,
        pretrained=cfg.pretrained,
        global_pool=cfg.global_pool,
        drop_rate=cfg.drop_rate,
        drop_path_rate=cfg.drop_path_rate,
    )


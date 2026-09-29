"""Trainable image + auxiliary anatomy model for nonclinical retention targets.

This module never needs source images, crop recipes, masks, or labels in forward.
Empty predicted lungs are diagnostics, not a reason to discard an input. Its
whole/basal/rim regions are geometric proxies, not clinical localizations.
The factory reuses a local, SHA-verified XRV DenseNet tensor checkpoint offline.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import math
from pathlib import Path
import re

import torch
from torch import Tensor, nn
from torch.nn import functional as F

FINDINGS = ("pleural_effusion", "pneumothorax", "consolidation")
VERSION = "direct-image-auxiliary-anatomy-retention-v1"


def _finite(value: Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")


def _probability(value: Tensor, name: str) -> None:
    _finite(value, name)
    if (value < 0).any() or (value > 1).any():
        raise ValueError(f"{name} must be in [0, 1]")


def soft_region_pool(features: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    """Return [B,R,C] pooled features and real [B,R] normalized region mass.

Zero mass yields an exactly zero vector, not a fabricated nonempty mask. The
mass remains explicit, so callers can distinguish a missing from a real region.
Accumulation is float32 even inside BF16/FP16 autocast.
"""
    if (features.ndim != 4 or weights.ndim != 4 or min(features.shape) < 1 or min(weights.shape) < 1 or
            features.shape[0] != weights.shape[0] or features.shape[-2:] != weights.shape[-2:]):
        raise ValueError("features/weights must have matching [B,channels,H,W] axes")
    _finite(features, "features")
    _probability(weights, "region weights")
    f, w = features.float().flatten(2), weights.float().flatten(2)
    total = w.sum(-1)
    with torch.autocast(device_type=features.device.type, enabled=False):
        pooled = torch.bmm(w, f.transpose(1, 2)) / total.clamp_min(1e-6).unsqueeze(-1)
    return pooled, total / w.shape[-1]


def lung_summary(probabilities: Tensor) -> tuple[Tensor, Tensor]:
    """Per lung: soft mass, confidence, four edge contacts and hard presence.

No bounding box or hard-mask division is needed. Channel identity is the
training target's image-coordinate convention, not inferred clinical laterality.
"""
    if probabilities.ndim != 4 or probabilities.shape[1] != 2:
        raise ValueError("lung probabilities must have shape [B,2,H,W]")
    _probability(probabilities, "lung probabilities")
    p = probabilities.float()
    hard_present = (p >= 0.5).flatten(2).any(-1)
    summary = torch.stack((p.mean((-2, -1)), p.amax((-2, -1)),
                           p[..., 0, :].mean(-1), p[..., -1, :].mean(-1),
                           p[..., :, 0].mean(-1), p[..., :, -1].mean(-1),
                           hard_present.float()), dim=-1)
    return summary.flatten(1), hard_present


def finding_region_weights(probabilities: Tensor) -> tuple[Tensor, Tensor]:
    """Predicted whole, basal bbox-quarter, and inner-rim weighted regions.

    Returns weights [B,3,2,H,W] and validity [B,3,2], in FINDINGS order.
Region membership uses the predicted >=0.5 mask; nonempty pixels retain their
soft confidence. Empty lungs are left empty. The rim radius is ceil(2% of the
predicted lung bbox short edge), minimum one decoder-grid pixel. These features
do not redefine the original continuous source-mask retention supervision.
"""
    if probabilities.ndim != 4 or probabilities.shape[1] != 2:
        raise ValueError("lung probabilities must have shape [B,2,H,W]")
    _probability(probabilities, "lung probabilities")
    p = probabilities.float()
    hard = p >= 0.5
    height, width = p.shape[-2:]
    ys = torch.arange(height, device=p.device)[None, None, :, None]
    xs = torch.arange(width, device=p.device)[None, None, None, :]
    y0 = torch.where(hard, ys, height).amin((-2, -1))
    y1 = torch.where(hard, ys, -1).amax((-2, -1)) + 1
    x0 = torch.where(hard, xs, width).amin((-2, -1))
    x1 = torch.where(hard, xs, -1).amax((-2, -1)) + 1
    whole = p * hard
    basal = whole * (ys >= (y0 + 0.75 * (y1 - y0))[:, :, None, None])
    radii = ((torch.minimum(y1 - y0, x1 - x0).float() * .02).ceil().clamp_min(1)).long()
    rim = torch.zeros_like(p)
    # Group kernels by radius; loops only over a few image-scale radii, not
    # pixels. Zero padding treats an image-boundary lung as touching the border.
    for radius_tensor in torch.unique(radii):
        radius = int(radius_tensor.item())
        selected = radii == radius
        m = hard[selected].float().unsqueeze(1)
        eroded = -F.max_pool2d(-F.pad(m, (radius,) * 4, value=0),
                              2 * radius + 1, stride=1)
        rim[selected] = p[selected] * (m[:, 0] - eroded[:, 0])
    regions = torch.stack((basal, rim, whole), dim=1)
    return regions, (regions > 0).flatten(3).any(-1)


class DirectAnatomyRetention(nn.Module):
    """Shared trainable encoder, auxiliary lungs, finding-conditioned regressor.

    `encoder` takes [B,1,H,W] floats in [-1024,1024] and returns a spatial map.
    The caller letterboxes the current image and supervision consistently; this
    module does not resize/crop/normalize input pixels a second time.
There is no pretrained download or random-backbone fallback in this constructor.
Inject a real pretrained encoder, or use a small test encoder explicitly.
"""

    def __init__(self, encoder: nn.Module, encoder_channels: int, *,
                 width: int = 64, finding_embedding_dim: int = 16,
                 region_dropout: float = 0.2, freeze_encoder_batchnorm: bool = True):
        super().__init__()
        if encoder_channels < 1 or width < 4 or width % 4:
            raise ValueError("encoder_channels must be positive; width must be a positive multiple of four")
        if finding_embedding_dim < 1 or not 0 <= region_dropout <= 1:
            raise ValueError("invalid embedding dimension or region_dropout")
        self.encoder = encoder
        self.encoder_channels = encoder_channels
        self.region_dropout = float(region_dropout)
        self.freeze_encoder_batchnorm = bool(freeze_encoder_batchnorm)
        self.anatomy_decoder = nn.Sequential(
            nn.Conv2d(encoder_channels, width, 1), nn.GroupNorm(4, width), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(4, width), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(4, width), nn.SiLU())
        self.lung_head = nn.Conv2d(width, 2, 1)
        self.finding_embeddings = nn.Embedding(len(FINDINGS), finding_embedding_dim)
        # global image + own regional feature + bilateral lung features +
        # bilateral summary + own mass + both lung masses + region-use flag.
        self.retention_head = nn.Sequential(
            nn.Linear(encoder_channels + 3 * width + 14 + 4 + finding_embedding_dim, 128),
            nn.SiLU(), nn.Linear(128, 1))
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder_batchnorm:
            # Running statistics stay at the pretrained values with small
            # batches; convolution and BN affine parameters remain trainable.
            for module in self.encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        if (images.ndim != 4 or images.shape[0] < 1 or images.shape[1] != 1 or
                min(images.shape[-2:]) < 32 or not images.is_floating_point()):
            raise ValueError("images must be floating [B,1,H,W], with H,W >=32")
        _finite(images, "images")
        if (images < -1024).any() or (images > 1024).any():
            raise ValueError("images must be XRV-normalized into [-1024,1024]")
        features = self.encoder(images)
        if (features.ndim != 4 or features.shape[:2] != (images.shape[0], self.encoder_channels)):
            raise ValueError("encoder returned an incompatible spatial feature map")
        _finite(features, "encoder features")
        anatomy = self.anatomy_decoder(features)
        coarse_logits = self.lung_head(anatomy)
        lungs = coarse_logits.float().sigmoid()
        summary, hard_present = lung_summary(lungs)
        per_lung_regions, region_valid = finding_region_weights(lungs)
        region_weights = per_lung_regions.amax(2)
        region, region_mass = soft_region_pool(anatomy, region_weights)
        bilateral, bilateral_mass = soft_region_pool(anatomy, per_lung_regions[:, 2])
        batch = images.shape[0]
        use_regions = torch.ones((batch, 1), device=images.device, dtype=torch.float32)
        if self.training and self.region_dropout:
            use_regions = (torch.rand_like(use_regions) >= self.region_dropout).float()
        # Feature dropout teaches the image branch to operate independently;
        # the actual segmentation output and its missing flags are not changed.
        region = region * use_regions[:, :, None]
        bilateral = bilateral.flatten(1) * use_regions
        summary = summary * use_regions
        regional_context = torch.cat((
            region,
            bilateral[:, None, :].expand(-1, len(FINDINGS), -1),
            summary[:, None, :].expand(-1, len(FINDINGS), -1),
            (region_mass * use_regions).unsqueeze(-1),
            (bilateral_mass * use_regions)[:, None, :].expand(-1, len(FINDINGS), -1),
            use_regions[:, None, :].expand(-1, len(FINDINGS), -1)), dim=-1)
        global_image = F.adaptive_avg_pool2d(features.float(), 1).flatten(1)
        embeddings = self.finding_embeddings.weight.float()[None].expand(batch, -1, -1)
        joined = torch.cat((global_image[:, None].expand(-1, len(FINDINGS), -1),
                            regional_context, embeddings), dim=-1)
        retention_logits = self.retention_head(joined).squeeze(-1).float()
        _finite(retention_logits, "retention logits")
        full_lung_logits = F.interpolate(coarse_logits.float(), size=images.shape[-2:],
                                         mode="bilinear", align_corners=False)
        full_lungs = full_lung_logits.sigmoid()
        return {
            "retention_logits": retention_logits,
            "retention": retention_logits.sigmoid(),
            "lung_logits": full_lung_logits,
            "lung_hard_present": (full_lungs >= 0.5).flatten(2).any(-1),
            "lung_soft_mass": full_lungs.mean((-2, -1)),
            "region_lung_hard_present": hard_present,
            "finding_region_soft_mass": per_lung_regions.mean((-2, -1)),
            "finding_region_valid": region_valid,
            "regional_branch_used": use_regions.bool().squeeze(-1),
        }


def multitask_loss(outputs: dict[str, Tensor], retention_targets: Tensor,
                   lung_targets: Tensor | None = None, *,
                   retention_valid: Tensor | None = None,
                   lung_valid: Tensor | None = None,
                   segmentation_weight: float = 0.25,
                   dice_weight: float = 0.5) -> dict[str, Tensor]:
    """Finite complete retention targets + optionally annotated bilateral masks.

Missing labels use explicit boolean validity, never a zero-valued substitute.
An observed, genuinely empty lung mask remains valid segmentation supervision.
Mask channel order and geometry must already match the presented input.
"""
    prediction = outputs["retention"].float()
    if prediction.ndim != 2 or prediction.shape[1] != len(FINDINGS) or retention_targets.shape != prediction.shape:
        raise ValueError("retention targets/predictions require [B,3] in FINDINGS order")
    _probability(prediction, "retention predictions")
    target = retention_targets.to(device=prediction.device, dtype=torch.float32)
    valid = torch.ones_like(prediction, dtype=torch.bool) if retention_valid is None else retention_valid
    if valid.shape != prediction.shape or valid.dtype != torch.bool or valid.device != prediction.device:
        raise ValueError("retention_valid must be a matching boolean tensor on the prediction device")
    if not valid.any():
        raise ValueError("at least one observed retention target is required")
    _probability(target[valid], "observed retention targets")
    retention = F.smooth_l1_loss(prediction[valid], target[valid], beta=0.1)
    if (not math.isfinite(segmentation_weight) or not math.isfinite(dice_weight) or
            segmentation_weight < 0 or dice_weight < 0):
        raise ValueError("loss weights must be finite and nonnegative")
    segmentation = prediction.sum() * 0
    observed_lungs = torch.zeros((), device=prediction.device, dtype=torch.long)
    if lung_targets is None:
        if lung_valid is not None:
            raise ValueError("lung_valid requires lung_targets")
    else:
        logits = outputs["lung_logits"].float()
        masks = lung_targets.to(device=logits.device, dtype=torch.float32)
        if logits.ndim != 4 or logits.shape[1] != 2 or masks.shape != logits.shape:
            raise ValueError("lung targets must match presented [B,2,H,W] logits; no implicit mask resize")
        observed = torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device) if lung_valid is None else lung_valid
        if observed.shape != logits.shape[:2] or observed.dtype != torch.bool or observed.device != logits.device:
            raise ValueError("lung_valid must be boolean [B,2] on the logits device")
        observed_lungs = observed.sum()
        if observed.any():
            selected_logits, selected_masks = logits[observed], masks[observed]
            _finite(selected_logits, "observed lung logits")
            _probability(selected_masks, "observed lung masks")
            bce = F.binary_cross_entropy_with_logits(selected_logits, selected_masks)
            p = selected_logits.sigmoid().flatten(1)
            t = selected_masks.flatten(1)
            dice = 1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)
            segmentation = bce + dice_weight * dice.mean()
    return {"loss": retention + segmentation_weight * segmentation,
            "retention_loss": retention, "segmentation_loss": segmentation,
            "observed_retention_targets": valid.sum(), "observed_lung_masks": observed_lungs}


class XrvDenseNetEncoder(nn.Module):
    """XRV's DenseNet features; caller has already applied original normalization."""
    def __init__(self, features: nn.Module):
        super().__init__()
        self.features = features

    def forward(self, images: Tensor) -> Tensor:
        return F.relu(self.features(images), inplace=False)


def from_xrv_densenet_checkpoint(path: str | Path, expected_sha256: str, **kwargs) -> DirectAnatomyRetention:
    """Load only an explicit, checksum-bound local tensor state; never download.

Uses the existing converted CheXpert XRV DenseNet-121 state dictionary. The
original disease classifier is loaded strictly for identity, then discarded;
the new encoder is trainable. A legacy pickled full-model file is not accepted.
"""
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be an explicit lowercase SHA256")
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    if hasher.hexdigest() != expected_sha256:
        raise ValueError("pretrained encoder checkpoint SHA256 mismatch")
    import torchxrayvision as xrv
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or any(not isinstance(k, str) or not torch.is_tensor(v)
                                                      for k, v in state.items()):
        raise ValueError("expected a plain tensor state dictionary")
    disease = xrv.models.DenseNet(weights=None, op_threshs=None, apply_sigmoid=False)
    disease.load_state_dict(state, strict=True)
    encoder = XrvDenseNetEncoder(disease.features)
    model = DirectAnatomyRetention(encoder, disease.classifier.in_features, **kwargs)
    model.initialization_provenance = {"version": VERSION, "path": str(path.resolve()),
                                       "sha256": expected_sha256, "pretrained": True,
                                       "encoder_trainable": True, "findings": list(FINDINGS),
                                       "torchxrayvision": importlib.metadata.version("torchxrayvision"),
                                       "input": "letterboxed grayscale [B,1,H,W], [-1024,1024]",
                                       "clinical_support_model": False}
    return model

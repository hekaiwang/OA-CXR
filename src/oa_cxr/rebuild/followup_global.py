"""Matched global-only DenseNet control, without anatomy decoder or mask input.

This is a new two-stage control, not a re-labelled previous ablation. Only the
current presented image enters forward. Three query embeddings are learned;
global pooled image features are shared by all three queries.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F

from .vision import FINDINGS, XrvDenseNetEncoder, _finite, _probability

VERSION = "oa-cxr-matched-global-control-v1"


class GlobalRetention(nn.Module):
    def __init__(self, encoder: nn.Module, encoder_channels: int, *, finding_embedding_dim: int = 16):
        super().__init__()
        if encoder_channels < 1 or finding_embedding_dim < 1:
            raise ValueError("positive feature and query dimensions required")
        self.encoder = encoder
        self.encoder_channels = encoder_channels
        self.finding_embeddings = nn.Embedding(len(FINDINGS), finding_embedding_dim)
        self.retention_head = nn.Sequential(nn.Linear(encoder_channels + finding_embedding_dim, 128),
                                            nn.SiLU(), nn.Linear(128, 1))
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        # Exact pretrained BN running statistics; affine parameters train.
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, images):
        if (images.ndim != 4 or images.shape[0] < 1 or images.shape[1] != 1
                or min(images.shape[-2:]) < 32 or not images.is_floating_point()):
            raise ValueError("images require floating [B,1,H,W] with H,W >=32")
        _finite(images, "images")
        if (images < -1024).any() or (images > 1024).any():
            raise ValueError("images must be XRV-normalized into [-1024,1024]")
        spatial = self.encoder(images)
        if spatial.ndim != 4 or spatial.shape[:2] != (len(images), self.encoder_channels):
            raise ValueError("incompatible encoder feature map")
        _finite(spatial, "encoder features")
        pooled = F.adaptive_avg_pool2d(spatial.float(), 1).flatten(1)
        global_features = pooled[:, None].expand(-1, len(FINDINGS), -1)
        embeddings = self.finding_embeddings.weight.float()[None].expand(len(images), -1, -1)
        joined = torch.cat((global_features, embeddings), dim=-1)
        logits = self.retention_head(joined).squeeze(-1).float()
        _finite(logits, "retention logits")
        return {"retention_logits": logits, "retention": logits.sigmoid(), "readout_features": joined}


def retention_loss(output, targets):
    prediction = output["retention"].float()
    if prediction.ndim != 2 or prediction.shape[1] != len(FINDINGS) or targets.shape != prediction.shape:
        raise ValueError("targets/predictions require matching [B,3]")
    target = targets.to(prediction.device, dtype=torch.float32)
    _probability(prediction, "retention predictions")
    _probability(target, "retention targets")
    return F.smooth_l1_loss(prediction, target, beta=0.1)


def feature_names(model):
    names = [f"image_encoder_{i:04d}" for i in range(model.encoder_channels)]
    names += [f"finding_embedding_{i:03d}" for i in range(model.finding_embeddings.embedding_dim)]
    if len(names) != model.retention_head[0].in_features:
        raise ValueError("feature names differ from actual first Linear input")
    return names + [f"finding_{name}" for name in FINDINGS]


def append_onehot(features):
    if features.ndim != 3 or features.shape[1] != len(FINDINGS):
        raise ValueError("features must be [B,3,D] in declared finding order")
    _finite(features, "readout features")
    onehot = torch.eye(len(FINDINGS), device=features.device, dtype=torch.float32)[None].expand(len(features), -1, -1)
    return torch.cat((features.float(), onehot), dim=-1).flatten(0, 1)


def from_pretrained(path, expected_sha256):
    """Strictly load local plain CheXpert tensor state; no downloads/fallback."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("explicit lowercase SHA256 required")
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("pretrained checkpoint SHA256 mismatch")
    import torchxrayvision as xrv
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or any(not isinstance(k, str) or not torch.is_tensor(v) for k, v in state.items()):
        raise ValueError("plain tensor state dictionary required")
    disease = xrv.models.DenseNet(weights=None, op_threshs=None, apply_sigmoid=False)
    disease.load_state_dict(state, strict=True)
    model = GlobalRetention(XrvDenseNetEncoder(disease.features), disease.classifier.in_features)
    model.initialization_provenance = {
        "version": VERSION, "path": str(path.resolve()), "sha256": expected_sha256,
        "pretrained": True, "encoder_trainable": True, "finding_embedding_dim": 16,
        "torchxrayvision": importlib.metadata.version("torchxrayvision"),
        "classifier_discarded": True, "anatomy_decoder_present": False,
        "regional_features_present": False, "segmentation_supervision": False,
        "new_query_embedding_and_regression_head": "fresh seed17 initialization; no OA decoder constructed",
        "input": "letterboxed grayscale [B,1,H,W], [-1024,1024]", "clinical_support_model": False}
    return model

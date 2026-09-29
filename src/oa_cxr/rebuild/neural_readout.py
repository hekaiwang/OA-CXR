"""Trainable neural readouts of frozen, label-free image/anatomy features.

This module has no dataset or checkpoint-file discovery. Only explicitly marked
fit features can initialize normalization; the caller controls fit/dev selection
and must record their identities. The original image encoder is not fine-tuned.
"""
from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

VERSION = "frozen-feature-neural-readout-v1"


def _features(value: Tensor, width: int) -> None:
    if not torch.is_tensor(value) or value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != width:
        raise ValueError(f"features must be nonempty [N,{width}]")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError("features must be finite floating values")


class FitStandardizer(nn.Module):
    """Persistent population moments, fitted once from fit batches only.

Batch moments are merged in float64. Constant columns use scale one; no rows
are dropped, no labels are used, and forward never updates these statistics.
"""
    def __init__(self, input_dim: int, minimum_scale: float = 1e-6):
        super().__init__()
        if not isinstance(input_dim, int) or isinstance(input_dim, bool) or input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        if not math.isfinite(minimum_scale) or minimum_scale <= 0:
            raise ValueError("minimum_scale must be finite and positive")
        self.input_dim, self.minimum_scale = input_dim, float(minimum_scale)
        self.register_buffer("mean", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer("scale", torch.ones(input_dim, dtype=torch.float32))
        self.register_buffer("row_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, batches: Tensor | Iterable[Tensor], *, split: str) -> None:
        if split != "fit":
            raise ValueError("standardization may only be fitted on split='fit'")
        if bool(self.fitted):
            raise RuntimeError("normalization is already frozen; construct a new readout to refit")
        count = 0
        mean = torch.zeros(self.input_dim, dtype=torch.float64)
        m2 = torch.zeros_like(mean)
        for batch in (batches,) if torch.is_tensor(batches) else batches:
            _features(batch, self.input_dim)
            x = batch.detach().to(device="cpu", dtype=torch.float64)
            n = len(x)
            batch_mean = x.mean(0)
            delta = batch_mean - mean
            m2 += ((x - batch_mean) ** 2).sum(0) + delta.square() * count * n / (count + n)
            mean += delta * n / (count + n)
            count += n
        if count < 1:
            raise ValueError("at least one fit feature row is required")
        scale = (m2 / count).clamp_min(0).sqrt()
        scale = torch.where(scale < self.minimum_scale, torch.ones_like(scale), scale)
        mean, scale = mean.float(), scale.float()
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("fit moments cannot be represented in float32")
        self.mean.copy_(mean)
        self.scale.copy_(scale)
        self.row_count.fill_(count)
        self.fitted.fill_(True)

    def forward(self, features: Tensor) -> Tensor:
        _features(features, self.input_dim)
        if not bool(self.fitted):
            raise RuntimeError("fit-only standardization has not been fitted")
        standardized = (features.float() - self.mean.float()) / self.scale.float()
        if not torch.isfinite(standardized).all():
            raise ValueError("standardized features contain nonfinite values")
        return standardized


class FrozenRetentionHead(nn.Module):
    """Replay the old 1250 -> 128 -> 1 head without added one-hot columns.

The old head takes unstandardized exported features. This float32 replay is a
control; CUDA BF16 generation rounding need not be bitwise identical to replay.
"""
    def __init__(self, head_input_dim: int, hidden_dim: int = 128, appended_features: int = 3):
        super().__init__()
        if (any(not isinstance(v, int) or isinstance(v, bool) or v < 1
                for v in (head_input_dim, hidden_dim)) or
                not isinstance(appended_features, int) or isinstance(appended_features, bool) or appended_features < 0):
            raise ValueError("invalid retained head dimensions")
        self.head_input_dim, self.hidden_dim = head_input_dim, hidden_dim
        self.appended_features = appended_features
        self.layers = nn.Sequential(nn.Linear(head_input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.requires_grad_(False)

    @classmethod
    def from_head(cls, head: nn.Sequential, *, appended_features: int = 3):
        if (not isinstance(head, nn.Sequential) or len(head) != 3 or
                not isinstance(head[0], nn.Linear) or not isinstance(head[1], nn.SiLU) or
                not isinstance(head[2], nn.Linear) or head[2].out_features != 1 or
                head[0].out_features != head[2].in_features):
            raise ValueError("retained head must be Linear -> SiLU -> Linear(1)")
        result = cls(head[0].in_features, head[0].out_features, appended_features)
        result.layers.load_state_dict(deepcopy(head.state_dict()))
        return result

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        _features(features, self.head_input_dim + self.appended_features)
        with torch.autocast(device_type=features.device.type, enabled=False):
            logits = self.layers(features[:, :self.head_input_dim].float()).squeeze(-1)
        if not torch.isfinite(logits).all():
            raise ValueError("retained head produced nonfinite logits")
        score = logits.sigmoid()
        return {"raw_prediction": logits, "retention": score, "loss_prediction": score}

    def config(self) -> dict:
        return {"head_input_dim": self.head_input_dim, "hidden_dim": self.hidden_dim,
                "appended_features": self.appended_features}


class ResidualBlock(nn.Module):
    def __init__(self, width: int, inner_width: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, inner_width),
                                     nn.SiLU(), nn.Dropout(dropout), nn.Linear(inner_width, width))

    def forward(self, features: Tensor) -> Tensor:
        return features + self.network(features)


class NeuralReadout(nn.Module):
    """Residual MLP, optionally learning a correction to frozen old logits.

In linear_clamp mode only reported retention is clipped: training uses the
unclipped value, preserving a corrective gradient outside [0,1]. In sigmoid
mode training uses sigmoid scores. old_logit requires sigmoid and starts with
an exactly zero correction, retaining the old head's initial predictions.
"""
    def __init__(self, input_dim: int, *, width: int = 256, inner_width: int = 128,
                 blocks: int = 2, dropout: float = 0.0,
                 activation: str = "linear_clamp", residual_mode: str = "none",
                 retained_head: FrozenRetentionHead | None = None,
                 minimum_scale: float = 1e-6):
        super().__init__()
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (width, inner_width, blocks)):
            raise ValueError("widths and block count must be positive integers")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        if activation not in ("linear_clamp", "sigmoid") or residual_mode not in ("none", "old_logit"):
            raise ValueError("unknown activation or residual mode")
        if residual_mode == "old_logit":
            if activation != "sigmoid" or retained_head is None:
                raise ValueError("old_logit correction requires sigmoid and a retained head")
            if retained_head.head_input_dim + retained_head.appended_features != input_dim:
                raise ValueError("retained head and exported feature widths differ")
        elif retained_head is not None:
            raise ValueError("retained head is only used in old_logit mode")
        self.standardizer = FitStandardizer(input_dim, minimum_scale)
        self.width, self.inner_width, self.blocks = width, inner_width, blocks
        self.dropout, self.activation, self.residual_mode = float(dropout), activation, residual_mode
        self.projection = nn.Sequential(nn.Linear(input_dim, width), nn.SiLU())
        self.residual_blocks = nn.Sequential(*(ResidualBlock(width, inner_width, dropout) for _ in range(blocks)))
        self.output = nn.Linear(width, 1)
        self.retained_head = deepcopy(retained_head)
        if residual_mode == "old_logit":
            self.retained_head.requires_grad_(False)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def fit_standardizer(self, batches: Tensor | Iterable[Tensor], *, split: str) -> None:
        self.standardizer.fit(batches, split=split)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        # Disabling autocast makes head arithmetic and returned values float32.
        # Callers should keep model parameters float32 rather than .half().
        with torch.autocast(device_type=features.device.type, enabled=False):
            normalized = self.standardizer(features)
            correction = self.output(self.residual_blocks(self.projection(normalized))).squeeze(-1)
            raw = correction
            if self.retained_head is not None:
                raw = raw + self.retained_head(features)["raw_prediction"]
            if not torch.isfinite(raw).all():
                raise ValueError("neural readout produced nonfinite values")
            score = raw.sigmoid() if self.activation == "sigmoid" else raw.clamp(0, 1)
            loss_prediction = score if self.activation == "sigmoid" else raw
        return {"raw_prediction": raw, "retention": score, "loss_prediction": loss_prediction}

    def config(self) -> dict:
        return {"input_dim": self.standardizer.input_dim, "width": self.width,
                "inner_width": self.inner_width, "blocks": self.blocks, "dropout": self.dropout,
                "activation": self.activation, "residual_mode": self.residual_mode,
                "minimum_scale": self.standardizer.minimum_scale,
                "retained_head": None if self.retained_head is None else self.retained_head.config()}

    @classmethod
    def from_config(cls, config: dict):
        config = deepcopy(config)
        if config.get("retained_head") is not None:
            config["retained_head"] = FrozenRetentionHead(**config["retained_head"])
        return cls(**config)


class NeuralReadoutImageAdapter(nn.Module):
    """Inference-only wrapper preserving the base model's lung diagnostics.

The original pre-Linear features and finding one-hot columns are extracted in
the same order as export.py. No masks/targets enter inference. No base weights
or source implementation are modified. Put both modules in eval mode first;
an accidental training-mode call fails rather than enabling region dropout.

For a linear readout, retention_logits is a finite diagnostic logit using
float32 epsilon clipping; the actual retention remains its original [0,1]
clamped value, and the epsilon is explicitly returned. Sigmoid readouts return
their exact raw logits and epsilon zero. This wrapper does not establish that
float32 replay equals the original CUDA BF16 report-generation numerics.
"""
    def __init__(self, base: nn.Module, readout: NeuralReadout):
        super().__init__()
        if not isinstance(getattr(base, "retention_head", None), nn.Sequential):
            raise ValueError("base model must expose the original retention_head")
        first = base.retention_head[0]
        if not isinstance(first, nn.Linear) or first.in_features + 3 != readout.standardizer.input_dim:
            raise ValueError("base head input plus three finding columns must match readout")
        self.base, self.readout = base, readout

    @torch.no_grad()
    def forward(self, images: Tensor) -> dict[str, Tensor]:
        if self.base.training or self.readout.training:
            raise RuntimeError("single-image adapter requires base and readout eval mode")
        captured = []
        hook = self.base.retention_head[0].register_forward_pre_hook(
            lambda _module, values: captured.append(values[0].detach().float()))
        try:
            base_output = self.base(images)
        finally:
            hook.remove()
        if len(captured) != 1 or captured[0].ndim != 3 or captured[0].shape[:2] != (len(images), 3):
            raise ValueError("base must produce exactly one [B,3,F] head input")
        head = captured[0]
        onehot = torch.eye(3, device=head.device, dtype=torch.float32)[None].expand(len(head), -1, -1)
        features = torch.cat((head, onehot), dim=-1).reshape(-1, self.readout.standardizer.input_dim)
        result = self.readout(features)
        score = result["retention"].reshape(len(head), 3)
        raw = result["raw_prediction"].reshape(len(head), 3)
        epsilon = 0.0 if self.readout.activation == "sigmoid" else torch.finfo(torch.float32).eps
        logits = raw if epsilon == 0 else torch.logit(score.clamp(epsilon, 1 - epsilon))
        return {**base_output, "retention": score, "retention_logits": logits,
                "neural_readout_raw_prediction": raw,
                "retention_logit_clipping_epsilon": torch.tensor(epsilon, device=score.device)}


def readout_loss(outputs: dict[str, Tensor], targets: Tensor, *, kind: str = "huber",
                 beta: float = 0.1) -> Tensor:
    """L1 or SmoothL1/Huber-style loss on explicitly observed [N] targets.

SmoothL1(beta) is Huber(beta) / beta, matching the existing model's scale.
Missing rows must be handled explicitly by the caller, never silently dropped.
"""
    prediction = outputs["loss_prediction"]
    if (prediction.ndim != 1 or prediction.numel() == 0 or targets.shape != prediction.shape or
            not targets.is_floating_point() or targets.device != prediction.device):
        raise ValueError("prediction/targets must be matching nonempty floating [N] on the same device")
    if (not torch.isfinite(prediction).all() or not torch.isfinite(targets).all() or
            (targets < 0).any() or (targets > 1).any()):
        raise ValueError("predictions must be finite and targets must lie in [0,1]")
    if kind == "l1":
        return F.l1_loss(prediction.float(), targets.float())
    if kind != "huber" or not math.isfinite(beta) or beta <= 0:
        raise ValueError("kind must be l1 or huber, with a finite positive Huber beta")
    return F.smooth_l1_loss(prediction.float(), targets.float(), beta=beta)

"""Controlled, matched-capacity ablations of a frozen anatomical readout.

These are conditional tests of an already trained representation. Removing its
regional inputs does not retrain the encoder or remove its auxiliary loss.
The public forward validates inputs; the trainer may use forward_validated only
after validating the complete, immutable feature matrix once.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .neural_readout import FitStandardizer, ResidualBlock

VERSION = "frozen-feature-ablation-readout-v1"
INPUT_DIM = 1253
FINDINGS = ("pleural_effusion", "pneumothorax", "consolidation")
SEEDS = (17,)


@dataclass(frozen=True)
class Arm:
    name: str
    zero_ranges: tuple[tuple[int, int], ...] = ()
    standardize: bool = True
    skip_connections: bool = True
    activation: str = "linear_clamp"
    loss: str = "l1"
    beta: float = .02

    def specification(self) -> dict:
        result = asdict(self)
        result["zero_ranges"] = [list(pair) for pair in self.zero_ranges]
        return result


ARMS = (
    Arm("full"),
    Arm("without_regional", ((1024, 1234),)),
    Arm("without_global", ((0, 1024),)),
    Arm("without_geometry_summary", ((1216, 1234),)),
    Arm("without_finding_encoding", ((1234, 1253),)),
    Arm("without_standardization", standardize=False),
    Arm("without_skip_connections", skip_connections=False),
    Arm("smooth_l1", loss="huber"),
    Arm("sigmoid_output", activation="sigmoid"),
)


def arm_definitions() -> list[dict]:
    """Fresh JSON-serializable copies; callers cannot mutate the fixed arms."""
    return [arm.specification() for arm in ARMS]


def get_arm(name: str) -> Arm:
    for arm in ARMS:
        if arm.name == name:
            return arm
    raise ValueError("unknown ablation arm: " + str(name))


def feature_names() -> list[str]:
    names = [f"image_encoder_{i:04d}" for i in range(1024)]
    names += [f"finding_region_{i:03d}" for i in range(64)]
    names += [f"lung_{lung}_region_{i:03d}" for lung in (0, 1) for i in range(64)]
    names += [f"lung_{lung}_{name}" for lung in (0, 1) for name in
              ("soft_mass", "maximum", "top_contact", "bottom_contact", "left_contact", "right_contact", "hard_present")]
    names += ["finding_region_mass", "lung_0_region_mass", "lung_1_region_mass", "regional_branch_used"]
    names += [f"finding_embedding_{i:03d}" for i in range(16)]
    names += ["finding_" + finding for finding in FINDINGS]
    return names


def validate_feature_names(names: list[str]) -> None:
    if names != feature_names():
        raise ValueError("the ordered 1253-column exported feature contract differs")


class AblationReadout(nn.Module):
    """Same parameter roster, dimensions and random initialization for each arm.

Zeroing happens after fit-only normalization, so zero is the neutral fit mean.
Without standardization uses raw features and still stores fit statistics for
audit. Removing skips preserves every LayerNorm/Linear/nonlinearity parameter.
Finding-region features remain finding dependent after removal of the explicit
16-dimensional embedding and three one-hot columns.
"""

    def __init__(self, *, arm: str = "full", input_dim: int = INPUT_DIM,
                 width: int = 256, inner_width: int = 256, blocks: int = 2,
                 dropout: float = 0., minimum_scale: float = 1e-6):
        super().__init__()
        specification = get_arm(arm)
        if input_dim != INPUT_DIM:
            raise ValueError("ablation feature contract requires exactly 1253 columns")
        if any(type(value) is not int or value < 1 for value in (width, inner_width, blocks)):
            raise ValueError("positive integer widths and block count required")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        self.arm = specification
        self.width, self.inner_width, self.blocks = width, inner_width, blocks
        self.dropout = float(dropout)
        self.standardizer = FitStandardizer(input_dim, minimum_scale)
        keep = torch.ones(input_dim, dtype=torch.float32)
        for start, stop in specification.zero_ranges:
            keep[start:stop] = 0
        self.register_buffer("input_keep_mask", keep)
        self.projection = nn.Sequential(nn.Linear(input_dim, width), nn.SiLU())
        # Use the original exact module roster for all arms. Only the addition
        # is bypassed in without_skip_connections, not any trainable layer.
        self.residual_blocks = nn.Sequential(*(ResidualBlock(width, inner_width, dropout) for _ in range(blocks)))
        self.output = nn.Linear(width, 1)

    def fit_standardizer(self, batches, *, split: str) -> None:
        self.standardizer.fit(batches, split=split)

    def prepare_features(self, features: Tensor, *, validate: bool = True) -> Tensor:
        if validate:
            if (not torch.is_tensor(features) or features.ndim != 2 or features.shape[0] < 1
                    or features.shape[1] != INPUT_DIM or not features.is_floating_point()
                    or not bool(torch.isfinite(features).all())):
                raise ValueError("features must be finite, nonempty floating [N,1253]")
            if not bool(self.standardizer.fitted):
                raise RuntimeError("fit-only standardization has not been fitted")
        value = features.float()
        if self.arm.standardize:
            value = (value - self.standardizer.mean) / self.standardizer.scale
        value = value * self.input_keep_mask
        if validate and not bool(torch.isfinite(value).all()):
            raise ValueError("feature transformation produced nonfinite values")
        return value

    def forward_validated(self, features: Tensor) -> dict[str, Tensor]:
        """No device-to-host input scans; caller must validate the full dataset.

The training loop separately checks loss and gradient finiteness before every
optimizer update. Evaluation checks the assembled prediction vector once.
"""
        with torch.autocast(device_type=features.device.type, enabled=False):
            hidden = self.projection(self.prepare_features(features, validate=False))
            for block in self.residual_blocks:
                hidden = block(hidden) if self.arm.skip_connections else block.network(hidden)
            raw = self.output(hidden).squeeze(-1)
            score = raw.sigmoid() if self.arm.activation == "sigmoid" else raw.clamp(0, 1)
            loss_prediction = score if self.arm.activation == "sigmoid" else raw
        return dict(raw_prediction=raw, retention=score, loss_prediction=loss_prediction)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        self.prepare_features(features, validate=True)
        output = self.forward_validated(features)
        if not bool(torch.isfinite(output["raw_prediction"]).all()):
            raise ValueError("readout produced nonfinite values")
        return output

    def config(self) -> dict:
        return dict(arm=self.arm.name, input_dim=INPUT_DIM, width=self.width,
                    inner_width=self.inner_width, blocks=self.blocks, dropout=self.dropout,
                    minimum_scale=self.standardizer.minimum_scale)

    @classmethod
    def from_config(cls, configuration: dict):
        return cls(**deepcopy(configuration))

    def parameter_counts(self) -> dict:
        count = sum(parameter.numel() for parameter in self.parameters())
        active = int(self.input_keep_mask.count_nonzero().item())
        return dict(total=count, trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    input_columns=INPUT_DIM, active_input_columns=active,
                    structurally_silenced_projection_weights=(INPUT_DIM - active) * self.width,
                    matched_allocated_parameter_count=True)


def loss_validated(outputs: dict[str, Tensor], targets: Tensor, arm: str) -> Tensor:
    """Loss for fully validated labels. No target values enter model.forward."""
    specification = get_arm(arm)
    if specification.loss == "l1":
        return F.l1_loss(outputs["loss_prediction"].float(), targets.float())
    return F.smooth_l1_loss(outputs["loss_prediction"].float(), targets.float(), beta=specification.beta)

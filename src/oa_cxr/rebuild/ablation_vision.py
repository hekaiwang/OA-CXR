"""Isolated image ablations of the sealed anatomy-retention architecture.

State-dict names remain compatible, but a checkpoint must always be constructed
with its recorded arm. Regional masking is an instance-local pre-Linear hook,
registered before any downstream feature-capture hook; it applies in train and
eval. Predicted masks and their validity diagnostics remain truthful.
"""
from __future__ import annotations

import torch
from torch import nn

from .vision import DirectAnatomyRetention, from_xrv_densenet_checkpoint

ARMS = ("full", "without_segmentation_loss", "without_regional_features",
        "without_region_dropout", "frozen_encoder")
VERSION = "anatomy-retention-image-ablation-v1"


def arm_settings(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown image ablation arm: {arm}")
    return {"arm": arm, "segmentation_weight": 0.0 if arm == "without_segmentation_loss" else 0.25,
            "region_dropout": 0.0 if arm == "without_region_dropout" else 0.2,
            "regional_features_enabled": arm != "without_regional_features",
            "encoder_trainable": arm != "frozen_encoder"}


class AblationAnatomyRetention(DirectAnatomyRetention):
    def __init__(self, encoder: nn.Module, encoder_channels: int, *, arm: str, **kwargs):
        self.ablation = arm_settings(arm)
        if "region_dropout" in kwargs:
            raise ValueError("region_dropout is fixed by the ablation arm")
        super().__init__(encoder, encoder_channels,
                         region_dropout=self.ablation["region_dropout"], **kwargs)
        # 1024:1234 for the production 1024-channel, width64 model. Compute
        # bounds from the same architecture to permit small real-tensor tests.
        self.regional_feature_slice = (encoder_channels,
                                      encoder_channels + 3 * self.lung_head.in_channels + 18)
        if not self.ablation["regional_features_enabled"]:
            self._regional_mask_hook = self.retention_head[0].register_forward_pre_hook(self._mask_regional)
        if not self.ablation["encoder_trainable"]:
            self.encoder.requires_grad_(False)
        self.train(self.training)

    def _mask_regional(self, _module, values):
        joined = values[0].clone()
        start, stop = self.regional_feature_slice
        joined[..., start:stop] = 0
        return (joined, *values[1:])

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.ablation["encoder_trainable"]:
            # Keep all encoder state, including BN buffers, unchanged. Merely
            # setting requires_grad(False) would not freeze running statistics.
            self.encoder.eval()
        return self

    def forward(self, images):
        output = super().forward(images)
        if not self.ablation["regional_features_enabled"]:
            output["regional_branch_used"] = torch.zeros_like(output["regional_branch_used"])
        return output


def from_pretrained(path, expected_sha256, *, arm):
    """Use the exact sealed pretrained loader; make no download or fallback."""
    # Preserve exactly the original loader's initial state and RNG position.
    # Extra wrapper construction must not change dropout/shuffle trajectories.
    base = from_xrv_densenet_checkpoint(path, expected_sha256)
    rng = torch.get_rng_state()
    model = AblationAnatomyRetention(base.encoder, base.encoder_channels, arm=arm,
                                   width=base.lung_head.in_channels,
                                   finding_embedding_dim=base.finding_embeddings.embedding_dim)
    model.load_state_dict(base.state_dict(), strict=True)
    torch.set_rng_state(rng)
    model.initialization_provenance = {**base.initialization_provenance, "ablation_version": VERSION,
                                      **model.ablation, "identical_fresh_module_initialization_within_seed": True}
    return model

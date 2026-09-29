"""Real torch CPU checks of missing anatomy, supervision and gradient routes."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from oa_cxr.rebuild.vision import (
    DirectAnatomyRetention, FINDINGS, XrvDenseNetEncoder,
    finding_region_weights, from_xrv_densenet_checkpoint, lung_summary, multitask_loss, soft_region_pool,
)


def model(**kwargs):
    encoder = nn.Sequential(nn.Conv2d(1, 8, 3, stride=4, padding=1), nn.BatchNorm2d(8), nn.ReLU())
    return DirectAnatomyRetention(encoder, 8, width=8, **kwargs)


def test_zero_mass_is_not_fabricated_and_negative_features_are_preserved():
    features = -torch.ones(2, 3, 4, 5)
    weights = torch.zeros(2, 2, 4, 5)
    weights[:, 1, 1, 2] = 1
    pooled, mass = soft_region_pool(features, weights)
    assert torch.equal(pooled[:, 0], torch.zeros(2, 3))
    assert torch.equal(pooled[:, 1], -torch.ones(2, 3))
    assert torch.equal(mass[:, 0], torch.zeros(2))
    assert torch.allclose(mass[:, 1], torch.full((2,), 1 / 20))


def test_hard_empty_does_not_erase_real_soft_evidence():
    probabilities = torch.full((2, 2, 5, 6), 0.1)
    summary, present = lung_summary(probabilities)
    assert not present.any()
    assert torch.allclose(summary[:, 0], torch.full((2,), 0.1))
    regions, valid = finding_region_weights(probabilities)
    assert not valid.any() and regions.sum() == 0


def test_empty_predicted_lungs_still_score_every_image_without_masks():
    net = model(region_dropout=0).eval()
    with torch.no_grad():
        net.lung_head.weight.zero_()
        net.lung_head.bias.fill_(-1000)
        out = net(torch.rand(3, 1, 32, 40))
    assert out["retention"].shape == (3, len(FINDINGS))
    assert torch.isfinite(out["retention"]).all()
    assert not out["lung_hard_present"].any()
    assert torch.equal(out["lung_soft_mass"], torch.zeros(3, 2))
    assert torch.equal(out["finding_region_soft_mass"], torch.zeros(3, 3, 2))
    assert not out["finding_region_valid"].any()
    assert out["lung_logits"].shape == (3, 2, 32, 40)


def test_joint_loss_trains_encoder_and_segmentation_head():
    torch.manual_seed(18)
    net = model(region_dropout=0)
    out = net(torch.rand(2, 1, 32, 32))
    masks = torch.zeros(2, 2, 32, 32)
    masks[:, 0, 5:28, 3:14] = 1
    masks[:, 1, 5:28, 18:29] = 1
    losses = multitask_loss(out, torch.tensor([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]]), masks)
    losses["loss"].backward()
    for parameter in (net.encoder[0].weight, net.lung_head.weight,
                      net.finding_embeddings.weight, net.retention_head[-1].weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
        assert torch.isfinite(parameter.grad).all()
    assert losses["observed_retention_targets"] == 6
    assert losses["observed_lung_masks"] == 4


def test_direct_route_has_gradient_with_entire_anatomy_branch_dropped():
    torch.manual_seed(21)
    net = model(region_dropout=1)
    out = net(torch.rand(2, 1, 32, 32))
    assert not out["regional_branch_used"].any()
    multitask_loss(out, torch.zeros(2, 3))["loss"].backward()
    assert net.encoder[0].weight.grad.abs().sum() > 0


def test_missing_mask_supervision_is_not_a_negative_mask_label():
    out = model(region_dropout=0)(torch.rand(2, 1, 32, 32))
    masks = torch.full((2, 2, 32, 32), float("nan"))
    no_masks = torch.zeros(2, 2, dtype=torch.bool)
    result = multitask_loss(out, torch.ones(2, 3), masks, lung_valid=no_masks)
    assert torch.isfinite(result["loss"])
    assert result["observed_lung_masks"] == 0
    assert result["segmentation_loss"] == 0
    with pytest.raises(ValueError, match="nonfinite"):
        multitask_loss(out, torch.ones(2, 3), masks)


def test_known_empty_mask_remains_supervised():
    out = model(region_dropout=0)(torch.rand(1, 1, 32, 32))
    result = multitask_loss(out, torch.zeros(1, 3), torch.zeros(1, 2, 32, 32))
    assert result["observed_lung_masks"] == 2
    assert result["segmentation_loss"] > 0


def test_loss_rejects_unaligned_masks_and_unobserved_retention():
    out = model()(torch.rand(1, 1, 32, 32))
    with pytest.raises(ValueError, match="no implicit mask resize"):
        multitask_loss(out, torch.ones(1, 3), torch.zeros(1, 2, 16, 16))
    with pytest.raises(ValueError, match="at least one"):
        multitask_loss(out, torch.ones(1, 3), retention_valid=torch.zeros(1, 3, dtype=torch.bool))


@pytest.mark.parametrize("bad", [float("nan"), -1024.01, 1024.01])
def test_malformed_pixels_fail_explicitly(bad):
    image = torch.rand(1, 1, 32, 32)
    image[0, 0, 0, 0] = bad
    with pytest.raises(ValueError):
        model()(image)


def test_eval_is_deterministic_and_encoder_batchnorm_stays_frozen():
    net = model(region_dropout=0.75)
    net.train()
    assert not net.encoder[1].training
    assert net.encoder[0].weight.requires_grad
    net.eval()
    image = torch.rand(1, 1, 32, 32)
    with torch.no_grad():
        a, b = net(image), net(image)
    assert torch.equal(a["retention"], b["retention"])
    assert a["regional_branch_used"].all()


def test_xrv_normalization_is_not_applied_twice():
    encoder = XrvDenseNetEncoder(nn.Identity())
    # ReLU is part of the real DenseNet feature wrapper, so negative endpoints
    # become zero here; capture the actual encoder input separately.
    values = []
    hook = encoder.features.register_forward_pre_hook(lambda module, args: values.append(args[0].clone()))
    encoder(torch.tensor([[[[-1024.0, 0.0, 1024.0]]]]))
    hook.remove()
    assert torch.equal(values[0], torch.tensor([[[[-1024.0, 0.0, 1024.0]]]]))


def test_weight_hash_checked_before_importing_or_loading_xrv(tmp_path):
    file = tmp_path / "fake.pt"
    file.write_bytes(b"not weights")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        from_xrv_densenet_checkpoint(file, "0" * 64)


def test_bfloat16_autocast_keeps_loss_and_pooling_finite():
    net = model(region_dropout=0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = net(torch.rand(2, 1, 32, 32))
        losses = multitask_loss(out, torch.rand(2, 3), torch.zeros(2, 2, 32, 32))
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])


def test_whole_basal_rim_and_missing_lung_have_explicit_validity():
    probabilities = torch.zeros(1, 2, 16, 16)
    probabilities[0, 0, 2:14, 3:11] = 0.8
    regions, valid = finding_region_weights(probabilities)
    assert regions.shape == (1, 3, 2, 16, 16)
    assert valid[0, :, 0].all() and not valid[0, :, 1].any()
    assert torch.allclose(regions[0, 2, 0].sum(), torch.tensor(12 * 8 * .8))
    assert torch.allclose(regions[0, 0, 0].sum(), torch.tensor(3 * 8 * .8))
    assert regions[0, 0, 0, :11].sum() == 0
    assert regions[0, 1, 0, 3:13, 4:10].sum() == 0
    assert regions[0, 1, 0, 2, 3] == .8

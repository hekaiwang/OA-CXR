"""Real CPU gradients and serialization; skipped if torch is unavailable."""
from __future__ import annotations

import io

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from oa_cxr.rebuild.neural_readout import (
    FitStandardizer, FrozenRetentionHead, NeuralReadout, NeuralReadoutImageAdapter, readout_loss,
)


def fit_features():
    torch.manual_seed(81)
    x = torch.randn(17, 9)
    x[:, 2] = 7
    return x


def test_standardization_is_fit_only_population_streaming_and_frozen():
    x = fit_features()
    norm = FitStandardizer(9)
    with pytest.raises(RuntimeError, match="not been fitted"):
        norm(x)
    for split in ("dev", "calibration", "test", "external"):
        with pytest.raises(ValueError, match="split='fit'"):
            norm.fit(x, split=split)
    norm.fit([x[:4], x[4:11], x[11:]], split="fit")
    assert norm.row_count.item() == len(x)
    assert torch.allclose(norm.mean, x.mean(0), atol=1e-6)
    expected = x.std(0, unbiased=False)
    expected[2] = 1
    assert torch.allclose(norm.scale, expected, atol=1e-6)
    assert torch.equal(norm(x)[:, 2], torch.zeros(len(x)))
    before = {k: v.clone() for k, v in norm.state_dict().items()}
    norm(x + 100)
    assert all(torch.equal(before[k], v) for k, v in norm.state_dict().items())
    with pytest.raises(RuntimeError, match="already frozen"):
        norm.fit(x + 100, split="fit")


@pytest.mark.parametrize("activation", ["linear_clamp", "sigmoid"])
@pytest.mark.parametrize("kind", ["l1", "huber"])
def test_real_optimizer_updates_mlp_and_keeps_statistics_fixed(activation, kind):
    x = fit_features()
    model = NeuralReadout(9, width=16, inner_width=8, activation=activation)
    model.fit_standardizer(x, split="fit")
    assert not any(isinstance(m, nn.BatchNorm1d) for m in model.modules())
    assert any(isinstance(m, nn.LayerNorm) for m in model.modules())
    before = {k: v.clone() for k, v in model.named_parameters()}
    moments = model.standardizer.mean.clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = model(x)
    assert result["retention"].shape == (len(x),)
    assert result["retention"].dtype == torch.float32
    assert ((result["retention"] >= 0) & (result["retention"] <= 1)).all()
    loss = readout_loss(result, torch.linspace(0, 1, len(x)), kind=kind)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    optimizer.step()
    assert all(not torch.equal(before[k], v) for k, v in model.named_parameters())
    assert torch.equal(model.standardizer.mean, moments)


def test_linear_clamp_preserves_training_gradient_outside_range():
    x = fit_features()
    model = NeuralReadout(9, width=8, inner_width=4)
    model.fit_standardizer(x, split="fit")
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.fill_(2)
    output = model(x)
    assert torch.equal(output["retention"], torch.ones(len(x)))
    readout_loss(output, torch.zeros(len(x)), kind="l1").backward()
    assert model.output.bias.grad.item() == pytest.approx(1)


def test_retained_control_uses_raw_features_and_ignores_appended_onehot():
    x = fit_features()
    head = nn.Sequential(nn.Linear(6, 5), nn.SiLU(), nn.Linear(5, 1))
    control = FrozenRetentionHead.from_head(head)
    expected = head(x[:, :6]).squeeze(-1)
    assert torch.equal(control(x)["raw_prediction"], expected)
    altered = x.clone()
    altered[:, 6:] = 1234
    assert torch.equal(control(altered)["retention"], expected.sigmoid())
    assert all(not p.requires_grad for p in control.parameters())
    with torch.no_grad():
        head[0].weight.zero_()
    assert torch.equal(control(x)["raw_prediction"], expected)


def test_old_logit_starts_exactly_at_control_then_learns_without_updating_control():
    x = fit_features()
    control = FrozenRetentionHead.from_head(nn.Sequential(nn.Linear(6, 5), nn.SiLU(), nn.Linear(5, 1)))
    model = NeuralReadout(9, width=16, inner_width=8, activation="sigmoid",
                          residual_mode="old_logit", retained_head=control)
    model.fit_standardizer(x, split="fit")
    assert torch.equal(model(x)["retention"], control(x)["retention"])
    before = {k: v.clone() for k, v in model.retained_head.state_dict().items()}
    first_projection = model.projection[0].weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        readout_loss(model(x), torch.zeros(len(x))).backward()
        optimizer.step()
    assert not torch.equal(model(x)["retention"], control(x)["retention"])
    assert not torch.equal(first_projection, model.projection[0].weight)
    assert all(torch.equal(before[k], v) for k, v in model.retained_head.state_dict().items())
    assert all(p.grad is None for p in model.retained_head.parameters())


@pytest.mark.parametrize("residual_mode", ["none", "old_logit"])
def test_state_dict_and_config_roundtrip_preserves_all_predictions_and_moments(residual_mode):
    x = fit_features()
    control = FrozenRetentionHead(6, 5) if residual_mode == "old_logit" else None
    model = NeuralReadout(9, width=16, inner_width=8, activation="sigmoid",
                          residual_mode=residual_mode, retained_head=control)
    model.fit_standardizer(x, split="fit")
    model.eval()
    buffer = io.BytesIO()
    torch.save({"config": model.config(), "state_dict": model.state_dict()}, buffer)
    buffer.seek(0)
    payload = torch.load(buffer, weights_only=True)
    restored = NeuralReadout.from_config(payload["config"])
    restored.load_state_dict(payload["state_dict"], strict=True)
    restored.eval()
    assert restored.config() == model.config()
    assert restored.standardizer.row_count.item() == len(x)
    assert torch.equal(restored(x + 0.1)["retention"], model(x + 0.1)["retention"])
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())


def test_cpu_autocast_still_returns_float32():
    x = fit_features()
    model = NeuralReadout(9, width=8, inner_width=4)
    model.fit_standardizer(x, split="fit")
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(x)
    assert all(value.dtype == torch.float32 for value in output.values())


def test_invalid_features_targets_and_incompatible_residuals_fail_explicitly():
    x = fit_features()
    model = NeuralReadout(9, width=8, inner_width=4)
    with pytest.raises(ValueError, match="features"):
        model.fit_standardizer(x[:, :4], split="fit")
    assert not model.standardizer.fitted
    model.fit_standardizer(x, split="fit")
    broken = x.clone()
    broken[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        model(broken)
    with pytest.raises(ValueError, match="targets"):
        readout_loss(model(x), torch.full((len(x),), 2.0))
    with pytest.raises(ValueError, match="kind"):
        readout_loss(model(x), torch.zeros(len(x)), kind="unknown")
    with pytest.raises(ValueError, match="requires sigmoid"):
        NeuralReadout(9, residual_mode="old_logit", retained_head=FrozenRetentionHead(6))
    with pytest.raises(ValueError, match="widths differ"):
        NeuralReadout(9, activation="sigmoid", residual_mode="old_logit", retained_head=FrozenRetentionHead(7))


@pytest.mark.parametrize("activation", ["linear_clamp", "sigmoid"])
def test_image_adapter_matches_exported_features_preserves_lungs_and_removes_hook(activation):
    class TinyBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.retention_head = nn.Sequential(nn.Linear(6, 5), nn.SiLU(), nn.Linear(5, 1))

        def forward(self, image):
            joined = image.reshape(len(image), 3, 6)
            logits = self.retention_head(joined).squeeze(-1)
            return {"retention": logits.sigmoid(), "retention_logits": logits,
                    "lung_hard_present": torch.zeros(len(image), 2, dtype=torch.bool),
                    "lung_logits": image}

    x = fit_features()
    model = NeuralReadout(9, width=8, inner_width=4, activation=activation)
    model.fit_standardizer(x, split="fit")
    base = TinyBase()
    adapter = NeuralReadoutImageAdapter(base, model)
    image = torch.randn(2, 1, 3, 6)
    with pytest.raises(RuntimeError, match="eval mode"):
        adapter(image)
    adapter.eval()
    joined = image.reshape(2, 3, 6)
    features = torch.cat((joined, torch.eye(3)[None].expand(2, -1, -1)), -1).reshape(6, 9)
    expected = model(features)
    before = {k: v.clone() for k, v in base.state_dict().items()}
    output = adapter(image)
    assert torch.equal(output["retention"], expected["retention"].reshape(2, 3))
    assert not output["lung_hard_present"].any()
    assert torch.equal(output["lung_logits"], image)
    assert torch.isfinite(output["retention_logits"]).all()
    assert output["retention_logit_clipping_epsilon"].item() == (0 if activation == "sigmoid" else torch.finfo(torch.float32).eps)
    assert all(torch.equal(before[k], v) for k, v in base.state_dict().items())
    assert not base.retention_head[0]._forward_pre_hooks

    broken = image.clone()
    broken[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        adapter(broken)
    assert not base.retention_head[0]._forward_pre_hooks

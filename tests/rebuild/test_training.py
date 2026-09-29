"""Training contracts and real-tensor metric accounting; no GPU or large corpus."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("rebuild_train", Path(__file__).resolve().parents[2] / "scripts/rebuild/train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def row(i, split="fit"):
    return {"source_id": f"{split}-{i}", "source_image_sha256": f"{i:064x}",
            "split_group_id": f"group-{split}-{i}", "dataset": "fixture", "split": split,
            "source_image_path": f"/fixture/{i}.png", "row_index": i}


def test_each_source_sees_all_thirteen_variants_once():
    for source in ["one", "two", "patient-胸片"]:
        assert sorted(train.variant_for_epoch(source, i) for i in range(13)) == list(range(13))
    with pytest.raises(ValueError):
        train.variant_for_epoch("one", 13)


def test_ten_thousand_means_unique_source_images_not_variants():
    rows = [row(i) for i in range(10_000)]
    train.validate_source_rows(rows, "fit")
    with pytest.raises(ValueError, match="at least 10000"):
        train.validate_source_rows(rows[:9999], "fit")
    rows[-1]["source_image_sha256"] = rows[0]["source_image_sha256"]
    with pytest.raises(ValueError, match="duplicate"):
        train.validate_source_rows(rows, "fit")


@pytest.mark.parametrize("key", ["source_id", "source_image_sha256", "split_group_id"])
def test_fit_dev_leakage_rejected_for_all_identity_keys(key):
    first, second = row(1), row(2, "dev")
    second[key] = first[key]
    with pytest.raises(ValueError, match="overlap"):
        train.assert_disjoint([first], [second])


def test_fixed_development_selection_is_order_independent_and_bounded():
    rows = [row(i, "dev") for i in range(600)]
    selected = {rows[i]["source_id"] for i in train.select_dev(rows)}
    reverse = list(reversed(rows))
    assert selected == {reverse[i]["source_id"] for i in train.select_dev(reverse)}
    assert len(selected) == 512


def test_refuses_heldout_before_opening_any_array(tmp_path, monkeypatch):
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("heldout array opened"))
    for split in ("calibration", "test", "external_nih"):
        with pytest.raises(ValueError, match="refuses"):
            train.PreparedDataset(tmp_path, split, [])


def fixture_arrays(tmp_path):
    directory = tmp_path / "fit"
    directory.mkdir()
    images = np.zeros((2, 13, 256, 256), dtype=np.uint8)
    images[0, :, :, :] = 255
    masks = np.zeros((2, 13, 2, 256, 256), dtype=np.uint8)
    masks[0, :, 1, 20, 40] = 1
    targets = np.full((2, 13, 3), .75, dtype=np.float32)
    np.save(directory / "images.npy", images)
    np.save(directory / "masks.npy", np.packbits(masks.reshape(2, 13, 2, -1), axis=-1, bitorder="big"))
    np.save(directory / "targets.npy", targets)
    return [row(0), row(1)]


def test_arrays_decode_masks_and_normalize_without_metadata_input(tmp_path):
    rows = fixture_arrays(tmp_path)
    dataset = train.PreparedDataset(tmp_path, "fit", rows)
    sample = dataset[0]
    assert sample["images"].shape == (1, 256, 256) and sample["images"].min() == 1024
    assert dataset[1]["images"].max() == -1024
    assert sample["masks"][1, 20, 40] == 1 and sample["masks"].sum() == 1
    assert "dataset" not in sample and "source_image_path" not in sample
    assert set(sample) == {"images", "masks", "targets", "source_index", "variant_index"}


def test_smoke_uses_distinct_variants_of_fixed_sources_and_dev_uses_all13(tmp_path):
    rows = fixture_arrays(tmp_path)
    smoke = train.PreparedDataset(tmp_path, "fit", rows, indices=[0, 1], smoke_batch=4)
    pairs = [(smoke[i]["source_index"], smoke[i]["variant_index"]) for i in range(len(smoke))]
    assert len(pairs) == len(set(pairs)) == 4
    assert {p[0] for p in pairs} == {0, 1}
    full = train.PreparedDataset(tmp_path, "fit", rows, indices=[1], all_variants=True)
    assert len(full) == 13 and {full[i]["variant_index"] for i in range(13)} == set(range(13))


@pytest.mark.parametrize("batch_size,sources", [(32, 16), (64, 16), (128, 32), (256, 64), (512, 128)])
def test_every_supported_smoke_batch_has_exactly_distinct_inputs(tmp_path, monkeypatch, batch_size, sources):
    class Array:
        def __init__(self, name): self.name = name
        def __getitem__(self, _):
            return {"images.npy": np.zeros((256, 256), dtype=np.uint8),
                    "masks.npy": np.zeros((2, 8192), dtype=np.uint8),
                    "targets.npy": np.ones(3, dtype=np.float32)}[self.name]
    monkeypatch.setattr(np, "load", lambda path, **kwargs: Array(Path(path).name))
    assert train.smoke_source_count(batch_size) == sources
    population = [row(i) for i in range(sources)]
    dataset = train.PreparedDataset(tmp_path, "fit", population, indices=range(sources), smoke_batch=batch_size)
    pairs = [(dataset[i]["source_index"], dataset[i]["variant_index"]) for i in range(len(dataset))]
    assert len(pairs) == len(set(pairs)) == batch_size
    assert len({source for source, _ in pairs}) == sources
    assert all(sum(source == i for source, _ in pairs) == batch_size // sources for i in range(sources))


@pytest.mark.parametrize("indices,batch", [([0, 0], 4), ([0, 1], 27), ([0, 1], 0), ([-1, 1], 4)])
def test_smoke_rejects_duplicates_or_exhausted_variant_capacity_before_arrays(tmp_path, monkeypatch, indices, batch):
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("invalid smoke accessed arrays"))
    with pytest.raises(ValueError):
        train.PreparedDataset(tmp_path, "fit", [row(0), row(1)], indices=indices, smoke_batch=batch)


@pytest.mark.parametrize("batch,steps", [(32, 597), (64, 299), (128, 150), (256, 75), (512, 38)])
def test_declared_optimizer_steps_include_last_partial_batch(batch, steps):
    protocol = train.optimization_plan(19086, batch)
    assert protocol["expected_optimizer_steps_per_epoch"] == steps
    assert protocol["expected_optimizer_steps_total"] == steps * 13
    assert protocol["physical_batch_size"] == protocol["effective_batch_size"] == batch
    assert protocol["gradient_accumulation_steps"] == 1 and protocol["drop_last"] is False


@pytest.mark.parametrize("change", ["batch", "code"])
def test_formal_training_rejects_smoke_from_old_batch_or_code_before_gpu(tmp_path, monkeypatch, change):
    checkpoint = tmp_path / "initial.pt"; checkpoint.write_bytes(b"fixture")
    smoke = tmp_path / "smoke"; smoke.mkdir()
    args = SimpleNamespace(action="train", batch_size=256, data=tmp_path / "cache", output=tmp_path / "train",
        checkpoint=checkpoint, checkpoint_sha256=train.sha256(checkpoint), gpu_uuid="GPU-1234",
        smoke_dir=smoke)
    data = {"root": args.data, "bindings": {"protocol.json": "1" * 64, "fit/sources.jsonl": "2" * 64},
            "cache_review_sha256": "3" * 64, "cache_status_sha256": "4" * 64}
    identity = train.common_identity(args, data)
    if change == "batch": identity["batch_size"] = 64
    else: identity["implementation"]["train.py"] = "0" * 64
    (smoke / "status.json").write_text(json.dumps({"status": "passed", "action": "smoke", "identity": identity}))
    monkeypatch.setattr(train, "validate_data", lambda *a, **k: data)
    monkeypatch.setattr(train, "check_gpu", lambda *a: pytest.fail("incompatible smoke reached GPU"))
    with pytest.raises(ValueError, match="different data/model/batch/implementation"):
        train.run(args)


def test_corrupt_target_is_not_silently_skipped(tmp_path):
    rows = fixture_arrays(tmp_path)
    targets = np.load(tmp_path / "fit/targets.npy")
    targets[:] = np.nan
    np.save(tmp_path / "fit/targets.npy", targets)
    dataset = train.PreparedDataset(tmp_path, "fit", rows)
    with pytest.raises(ValueError, match="no silent"):
        dataset[0]


def test_bound_files_reject_substituted_bytes(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text("{}")
    bindings = {"protocol.json": train.sha256(path)}
    assert train._bound_file(tmp_path.resolve(), "protocol.json", bindings) == path.resolve()
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="mismatch"):
        train._bound_file(tmp_path.resolve(), "protocol.json", bindings)


def test_smoke_validation_reads_only_fit_not_dev_or_heldout(tmp_path, monkeypatch):
    protocol = {"findings": train.FINDINGS, "variants": 13,
                "data_protocol": {"normalization": "(gray/255*2-1)*1024"}}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    fit = tmp_path / "fit"; fit.mkdir()
    rows = [row(i) for i in range(10000)]
    (fit / "sources.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    for name in ("images.npy", "masks.npy", "targets.npy"):
        (fit / name).write_bytes(b"header stub inspected by fake numpy loader")
    paths = ["protocol.json"] + [f"fit/{n}" for n in ("sources.jsonl", "images.npy", "masks.npy", "targets.npy")]
    status = {"status": "completed", "files": {p: train.sha256(tmp_path / p) for p in paths},
              "splits": {"fit": {"sources": 10000, "inputs": 130000, "targets": 390000}}}
    (tmp_path / "status.json").write_text(json.dumps(status))
    review = {"status": "passed", "inspected": True,
              "cache_status_sha256": train.sha256(tmp_path / "status.json")}
    (tmp_path / "visual_review.json").write_text(json.dumps(review))
    opened = []
    def inspect(path, **kwargs):
        opened.append(Path(path))
        assert Path(path).parent.name == "fit"
        shape, dtype = {"images.npy": ((10000, 13, 256, 256), np.uint8),
                        "masks.npy": ((10000, 13, 2, 8192), np.uint8),
                        "targets.npy": ((10000, 13, 3), np.float32)}[Path(path).name]
        return SimpleNamespace(shape=shape, dtype=np.dtype(dtype))
    monkeypatch.setattr(np, "load", inspect)
    data = train.validate_data(tmp_path, include_dev=False)
    assert set(data["rows"]) == {"fit"} and len(opened) == 3
    protocol["findings"] = list(reversed(train.FINDINGS))
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    status["files"]["protocol.json"] = train.sha256(tmp_path / "protocol.json")
    (tmp_path / "status.json").write_text(json.dumps(status))
    review["cache_status_sha256"] = train.sha256(tmp_path / "status.json")
    (tmp_path / "visual_review.json").write_text(json.dumps(review))
    with pytest.raises(ValueError, match="findings"):
        train.validate_data(tmp_path, include_dev=False)


@pytest.mark.parametrize("change", ["not_inspected", "pending", "stale"])
def test_visual_review_is_required_before_any_prepared_array_access(tmp_path, monkeypatch, change):
    (tmp_path / "status.json").write_text(json.dumps({"status": "completed"}))
    review = {"status": "passed", "inspected": True,
              "cache_status_sha256": train.sha256(tmp_path / "status.json")}
    if change == "not_inspected": review["inspected"] = False
    elif change == "pending": review["status"] = "pending"
    else: review["cache_status_sha256"] = "0" * 64
    (tmp_path / "visual_review.json").write_text(json.dumps(review))
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("array accessed before visual gate"))
    with pytest.raises(ValueError, match="actual cache technical visual review"):
        train.validate_data(tmp_path, include_dev=False)


def test_atomic_status_is_valid_json_and_rejects_nan(tmp_path):
    path = tmp_path / "status.json"
    train.atomic_json(path, {"status": "running"})
    train.atomic_json(path, {"status": "completed", "n": 10000})
    assert json.loads(path.read_text())["n"] == 10000
    with pytest.raises(ValueError):
        train.atomic_json(path, {"loss": float("nan")})
    assert json.loads(path.read_text())["status"] == "completed"


def test_metrics_separate_empty_mask_from_finite_retention_scores():
    torch = pytest.importorskip("torch")
    metrics = train.MetricAccumulator()
    prediction = {"retention": torch.full((2, 3), .5), "lung_logits": torch.full((2, 2, 4, 4), -10.)}
    batch = {"targets": torch.full((2, 3), .75), "masks": torch.zeros(2, 2, 4, 4),
             "source_index": torch.tensor([0, 1])}
    batch["masks"][0, 0] = 1
    metrics.add(prediction, batch, torch.tensor(.2))
    result = metrics.result()
    assert result["mae"] == .25 and result["finite_score_rate"] == 1
    assert result["empty_mask_rate"] == 1 and result["mask_dice"] == .75
    assert result["unique_sources"] == 2


def audit_parameters():
    return {
        "encoder.features.conv0.weight": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "anatomy_decoder.0.weight": np.ones((2, 2), dtype=np.float32),
        "lung_head.weight": np.ones((2, 1), dtype=np.float32),
        "finding_embeddings.weight": np.ones((3, 2), dtype=np.float32),
        "retention_head.0.weight": np.ones((2, 3), dtype=np.float32),
    }


def test_finetuning_evidence_measures_selected_parameters_and_actual_steps():
    initial = audit_parameters()
    selected = {name: value.copy() + .25 for name, value in initial.items()}
    selected["encoder.features.conv0.weight"] = initial["encoder.features.conv0.weight"].copy()
    selected["encoder.features.conv0.weight"][0, 0] += 3
    result = train.parameter_update_evidence(initial, selected, optimizer_steps=260, selected_optimizer_steps=100)
    assert result["status"] == "passed" and result["all_audited_components_changed"] is True
    assert result["optimizer_steps"] == 260 and result["selected_checkpoint_optimizer_steps"] == 100
    conv = result["components"]["encoder_conv0"]
    assert conv["changed_elements"] == conv["changed_parameter_count"] == 1
    assert conv["difference_l2_norm"] == conv["max_abs_difference"] == 3
    assert conv["relative_difference_l2"] == pytest.approx(3 / np.sqrt(30))
    assert conv["initial_parameters_sha256"] != conv["selected_parameters_sha256"]
    assert result["clinical_segmentation_success_established"] is False
    np.testing.assert_array_equal(initial["encoder.features.conv0.weight"], [[1, 2], [3, 4]])
    json.dumps(result, allow_nan=False)


def test_finite_unchanged_parameters_are_not_successful_finetuning():
    parameters = audit_parameters()
    result = train.parameter_update_evidence(parameters, parameters, optimizer_steps=13, selected_optimizer_steps=1)
    assert result["status"] == "failed" and result["all_audited_components_changed"] is False
    assert all(v["changed_elements"] == 0 for v in result["components"].values())


@pytest.mark.parametrize("total,selected", [(0, 0), (-1, 1), (1, 2), (True, 1), (1, True)])
def test_finetuning_evidence_rejects_zero_or_inconsistent_step_claims(total, selected):
    parameters = audit_parameters()
    with pytest.raises(ValueError, match="actual optimizer steps"):
        train.parameter_update_evidence(parameters, parameters,
            optimizer_steps=total, selected_optimizer_steps=selected)


@pytest.mark.parametrize("change", ["missing", "shape", "nan"])
def test_finetuning_evidence_rejects_invalid_selected_weight_identity(change):
    initial = audit_parameters()
    selected = {name: value.copy() for name, value in initial.items()}
    key = "encoder.features.conv0.weight"
    if change == "missing": del selected[key]
    elif change == "shape": selected[key] = np.zeros((1, 1), dtype=np.float32)
    else: selected[key][0, 0] = np.nan
    with pytest.raises(ValueError):
        train.parameter_update_evidence(initial, selected, optimizer_steps=1, selected_optimizer_steps=1)


def test_real_optimizer_update_is_captured_without_snapshot_aliasing():
    torch = pytest.importorskip("torch")
    named = [(name, torch.nn.Parameter(torch.from_numpy(value.copy()))) for name, value in audit_parameters().items()]
    model = SimpleNamespace(named_parameters=lambda: iter(named))
    initial = train.capture_audit_parameters(model)
    optimizer = torch.optim.AdamW([p for _, p in named], lr=1e-4, weight_decay=1e-4)
    sum(p.square().sum() for _, p in named).backward()
    optimizer.step()
    result = train.parameter_update_evidence(initial, dict(named),
        optimizer_steps=1, selected_optimizer_steps=1)
    assert result["status"] == "passed"
    for name, original in audit_parameters().items():
        np.testing.assert_array_equal(initial[name], original)

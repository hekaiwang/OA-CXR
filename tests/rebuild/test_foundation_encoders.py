"""Field-of-view, official-asset binding and independent-label adapter tests."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from baselines import foundation_encoders as encoders

spec = importlib.util.spec_from_file_location("test_foundation_export", Path(__file__).resolve().parents[2] / "scripts/rebuild/export_foundation_features.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


@pytest.mark.parametrize("method", encoders.METHODS)
def test_full_frame_normalization_preserves_borders_and_rgb(method):
    gray = np.full((256, 256), 127, dtype=np.uint8)
    gray[:12] = 255; gray[-12:] = 0; gray[:, :12] = 63; gray[:, -12:] = 192
    result = encoders.preprocess_image(gray, method)
    policy = encoders.preprocessing_policy(method)
    assert result.dtype == np.float32 and result.shape == (3, 224, 224) and result.flags.c_contiguous
    recovered = result * np.array(policy["std"], np.float32)[:, None, None] + np.array(policy["mean"], np.float32)[:, None, None]
    rgb = np.clip(np.rint(recovered.transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
    assert np.array_equal(rgb, encoders.presented_rgb(gray))
    assert rgb[0, 112, 0] == 255 and rgb[-1, 112, 0] == 0
    assert rgb[112, 0, 0] == 63 and rgb[112, -1, 0] == 192
    assert np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])


@pytest.mark.parametrize("value", [np.zeros((256, 256), np.float32), np.zeros((224, 224), np.uint8), np.zeros((256, 256, 3), np.uint8)])
def test_preprocessor_rejects_noncache_inputs(value):
    with pytest.raises(ValueError, match="uint8 256x256"):
        encoders.preprocess_image(value, "biomedclip")


def test_feature_encoding_repeats_visual_information_and_only_adds_finding_identity():
    image = np.array([[1, 2], [3, 4]], np.float32)
    actual = encoders.query_features(image)
    np.testing.assert_array_equal(actual[:, :2], np.repeat(image, 3, axis=0))
    np.testing.assert_array_equal(actual[:, 2:], np.tile(np.eye(3, dtype=np.float32), (2, 1)))
    assert encoders.feature_names("dinov2", export.FINDINGS)[-3:] == ["finding_" + finding for finding in export.FINDINGS]
    assert len(encoders.feature_names("biomedclip", export.FINDINGS)) == 515
    assert len(encoders.feature_names("dinov2", export.FINDINGS)) == 771


@pytest.mark.parametrize("value", [np.ones((2, 3), np.float64), np.array([[np.nan]], np.float32), np.ones(3, np.float32)])
def test_query_features_rejects_invalid_embedding(value):
    with pytest.raises(ValueError, match="float32"):
        encoders.query_features(value)


def asset_fixture(root, method="biomedclip"):
    root.mkdir()
    (root / "weights.pt").write_bytes(b"synthetic test artifact")
    (root / "config.json").write_text("{}")
    upstream = root / "upstream"; upstream.mkdir()
    (upstream / "model.py").write_text("# synthetic fixture\n")
    names = ["weights.pt", "config.json", "upstream/model.py"]
    value = {"schema": "oa-cxr-foundation-assets-v1", "method": method, "status": "completed",
             "files": [{"path": name, "sha256": encoders.sha256(root / name)} for name in names],
             "loader": {"kind": {"biomedclip": "biomedclip_visual", "chexzero": "chexzero_visual", "dinov2": "dinov2_vitb14"}[method],
                        "weights": "weights.pt", "config": "config.json", "code_root": "upstream"}}
    path = root / "asset_manifest.json"
    path.write_text(json.dumps(value))
    return path, value


@pytest.mark.parametrize("method", encoders.METHODS)
def test_asset_manifest_requires_byte_bound_weights_and_code(tmp_path, method):
    path, _ = asset_fixture(tmp_path / method, method)
    verified = encoders.verify_assets(path, encoders.sha256(path), method)
    assert verified["method"] == method and verified["paths"]["weights.pt"].is_file()
    (path.parent / "weights.pt").write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="file SHA256"):
        encoders.verify_assets(path, encoders.sha256(path), method)


@pytest.mark.parametrize("change", ["manifest_sha", "method", "status", "duplicate", "escape", "unbound_code", "unbound_weights"])
def test_asset_binding_rejects_substitution(tmp_path, change):
    path, value = asset_fixture(tmp_path / "assets", "chexzero")
    if change == "method": value["method"] = "dinov2"
    elif change == "status": value["status"] = "running"
    elif change == "duplicate": value["files"].append(value["files"][0])
    elif change == "escape":
        (tmp_path / "outside.py").write_text("x=1")
        value["files"].append({"path": "../outside.py", "sha256": encoders.sha256(tmp_path / "outside.py")})
    elif change == "unbound_code": (path.parent / "upstream/extra.py").write_text("# unbound")
    elif change == "unbound_weights": value["loader"]["weights"] = "other.pt"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        encoders.verify_assets(path, "0" * 64 if change == "manifest_sha" else encoders.sha256(path), "chexzero")


def cache_fixture(root):
    root.mkdir(); folder = root / "external_fixture"; folder.mkdir()
    row = {"row_index": 0, "source_id": "fixture", "source_image_sha256": "1" * 64,
           "split_group_id": "group", "dataset": "fixture", "split": "external_fixture",
           "source_image_path": "/not-opened/source.png"}
    (folder / "sources.jsonl").write_text(json.dumps(row) + "\n")
    np.save(folder / "images.npy", np.broadcast_to(np.arange(256, dtype=np.uint8)[None, None, None, :], (1, 13, 256, 256)))
    (root / "protocol.json").write_text(json.dumps({"findings": export.FINDINGS, "variants": 13,
             "data_protocol": {"normalization": "(gray/255*2-1)*1024"}}))
    names = ["protocol.json", "external_fixture/sources.jsonl", "external_fixture/images.npy"]
    (root / "status.json").write_text(json.dumps({"status": "completed", "files": {name: export.sha256(root / name) for name in names},
                "splits": {"external_fixture": {"sources": 1, "inputs": 13, "targets": 39}}}))
    (root / "visual_review.json").write_text(json.dumps({"status": "passed", "inspected": True,
                 "cache_status_sha256": export.sha256(root / "status.json")}))
    return row


def test_image_only_cache_with_no_masks_or_targets_is_sufficient_for_preview(tmp_path):
    cache = tmp_path / "cache"; cache_fixture(cache)
    result = export.preview(SimpleNamespace(data=cache, split="external_fixture", method="biomedclip", output=tmp_path / "preview"))
    assert result["status"] == "completed" and result["inputs"] == 3
    assert result["masks_loaded"] is False and result["targets_loaded"] is False
    assert result["actual_visual_review"].startswith("pending")
    dataset = export.ImageOnlyDataset(cache / "external_fixture/images.npy", "dinov2")
    assert len(dataset) == 13 and dataset[12][1] == 12 and dataset[12][0].shape == (3, 224, 224)


def bridge_fixture(root):
    exported, truth = root / "exported", root / "truth"
    exported.mkdir(); truth.mkdir()
    cache = root / "cache"; row = cache_fixture(cache)
    rows = list(export.original.query_records([row], "external_fixture"))
    ids = np.asarray([r["row_id"] for r in rows], dtype="U64")
    X = np.arange(len(rows) * 5, dtype=np.float32).reshape(len(rows), 5)
    y = np.linspace(0, 1, len(rows), dtype=np.float32)
    for location in (exported, truth):
        (location / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    np.savez(exported / "features.npz", X=X, row_ids=ids, available=np.ones(len(rows), bool))
    np.savez(truth / "labels.npz", y=y, row_ids=ids)
    (exported / "provenance.json").write_text("{}")
    (exported / "manifest.json").write_text(json.dumps(export.original.make_manifest(exported / "samples.jsonl", exported / "features.npz",
                    ["image0", "image1"] + ["finding_" + f for f in export.FINDINGS], "2" * 64)))
    for location, action, files in ((exported, "export-foundation", ("samples.jsonl", "features.npz", "manifest.json", "provenance.json")),
                                    (truth, "attach-labels", ("samples.jsonl", "labels.npz"))):
        (location / "status.json").write_text(json.dumps({"status": "completed", "action": action, "method": "biomedclip",
                    "output_sha256": {name: export.sha256(location / name) for name in files}}))
    return exported, truth, X, y, rows


def test_label_bridge_preserves_query_dataset_and_features_without_loading_old_x(tmp_path, monkeypatch):
    exported, truth, X, y, rows = bridge_fixture(tmp_path)
    # No masks, original X, source image files or encoder weights even exist.
    monkeypatch.setattr(encoders, "load_encoder", lambda *a: pytest.fail("label bridge loaded encoder"))
    result = export.attach_labels(SimpleNamespace(export_dir=exported, truth_dir=truth, output=tmp_path / "attached"))
    assert result["status"] == "completed" and result["original_features_loaded"] is False
    with np.load(tmp_path / "attached/features.npz") as value:
        np.testing.assert_array_equal(value["X"], X)
        np.testing.assert_array_equal(value["y"], y)
    assert (tmp_path / "attached/samples.jsonl").read_bytes() == (exported / "samples.jsonl").read_bytes()
    assert {r["dataset"] for r in rows} == {"fixture"}


@pytest.mark.parametrize("change", ["identity", "label_ids", "label_range", "label_nan", "changed_sha", "failed_source"])
def test_bridge_rejects_changed_or_misaligned_labels(tmp_path, change):
    exported, truth, X, y, rows = bridge_fixture(tmp_path)
    if change == "identity":
        rows[0]["dataset"] = "different"
        (truth / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    elif change in ("label_ids", "label_range", "label_nan"):
        ids = np.asarray([r["row_id"] for r in rows], dtype="U64")
        if change == "label_ids": ids = ids[::-1]
        elif change == "label_range": y[0] = 2
        else: y[0] = np.nan
        np.savez(truth / "labels.npz", y=y, row_ids=ids)
    state = export.read_json(truth / "status.json")
    if change != "changed_sha":
        state["output_sha256"] = {name: export.sha256(truth / name) for name in ("samples.jsonl", "labels.npz")}
    else:
        (truth / "labels.npz").write_bytes(b"changed")
    if change == "failed_source": state["status"] = "failed"
    (truth / "status.json").write_text(json.dumps(state))
    with pytest.raises(ValueError):
        export.attach_labels(SimpleNamespace(export_dir=exported, truth_dir=truth, output=tmp_path / "attached"))
    assert not (tmp_path / "attached").exists()


def test_feature_protocol_excludes_split_or_outcome_selection():
    value = export.feature_protocol("biomedclip", "0" * 64)
    assert value["inference_batch_size"] == 128 and value["targets_or_masks_read"] is False
    assert "data" not in value and "split" not in value and "checkpoint_selection" not in value


@pytest.mark.parametrize("method", encoders.METHODS)
def test_frozen_forward_shape_and_normalization_on_cpu(method):
    torch = pytest.importorskip("torch")
    class SyntheticEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(encoders.DIMENSIONS[method]), requires_grad=False)
        def forward(self, image):
            return self.scale[None, :].expand(len(image), -1)
    model = SyntheticEncoder().eval()
    result = encoders.encode_batch(model, torch.zeros(2, 3, 224, 224), method)
    assert result.shape == (2, encoders.DIMENSIONS[method]) and result.dtype == torch.float32
    if method != "dinov2":
        torch.testing.assert_close(torch.linalg.vector_norm(result, dim=1), torch.ones(2))
    else:
        torch.testing.assert_close(result, torch.ones_like(result))
    model.train()
    with pytest.raises(ValueError, match="eval"):
        encoders.encode_batch(model, torch.zeros(2, 3, 224, 224), method)

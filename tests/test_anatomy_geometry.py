"""Tests for geometry truth, input isolation, and the manual-mask proxy contract."""
import importlib.util
import inspect
import json

import numpy as np
from PIL import Image
import pytest

from oa_cxr.anatomy_geometry import (
    FEATURE_PROTOCOL, FEATURE_PROTOCOL_SHA256, GEOMETRY_PROTOCOL_SHA256,
    LABEL_PROTOCOL_SHA256, anatomy_retention_labels, array_sha256,
    current_image_geometry_features, maira_geometry, mask_box_area,
    transform_manual_mask,
)
from oa_cxr.io import stable_hash


def label_args():
    return {"source_image_sha256": "a" * 64,
            "manual_mask_sha256": {"left": "b" * 64, "right": "c" * 64},
            "annotation_source": "manual-mask-fixture-v1", "processor_revision": "d" * 40}


def feature_args():
    return {"presented_image_sha256": "a" * 64, "producer_revision": "pspnet-fixture-v1",
            "weights_sha256": "e" * 64}


@pytest.fixture
def lungs():
    left, right = np.zeros((100, 100), dtype=bool), np.zeros((100, 100), dtype=bool)
    left[10:90, 10:40] = True
    right[10:90, 60:90] = True
    return left, right


def test_uncropped_rectangular_source_still_loses_fov():
    geometry = maira_geometry((100, 200))
    assert geometry["resized_size"] == [518, 1036]
    assert geometry["center_crop_box"] == [0, 259, 518, 777]
    assert geometry["visible_box"] == [0.0, 50.0, 100.0, 150.0]
    assert geometry["protocol_sha256"] == GEOMETRY_PROTOCOL_SHA256
    mask = np.ones((200, 100), dtype=bool)
    assert mask_box_area(mask, geometry["visible_box"]) / mask.sum() == 0.5


def test_native_crop_translates_viewport_and_can_admit_previously_invisible_anatomy():
    original = maira_geometry((100, 200))
    cropped = maira_geometry((100, 200), (0, 20, 100, 200))
    assert cropped["resized_size"] == [518, 932]
    assert cropped["center_crop_box"] == [0, 207, 518, 725]
    assert cropped["visible_box"] == pytest.approx([0, 59.97854077253219, 100, 160.0214592274678])
    new_anatomy = np.zeros((200, 100), dtype=bool)
    new_anatomy[151:158, 30:70] = True
    assert mask_box_area(new_anatomy, original["visible_box"]) == 0
    assert mask_box_area(new_anatomy, cropped["visible_box"]) == new_anatomy.sum()


@pytest.mark.parametrize("size,resized,crop", [
    ((1000, 1400), [518, 725], [0, 103, 518, 621]),
    ((1400, 1000), [725, 518], [103, 0, 621, 518]),
    ((1001, 1403), [518, 726], [0, 104, 518, 622]),
    ((517, 518), [518, 519], [0, 0, 518, 518]),
    ((518, 517), [519, 518], [0, 0, 518, 518]),
    ((13, 13), [518, 518], [0, 0, 518, 518]),
])
def test_hf_integer_resize_and_odd_center_offset_golden_cases(size, resized, crop):
    geometry = maira_geometry(size)
    assert geometry["resized_size"] == resized
    assert geometry["center_crop_box"] == crop


def test_continuous_pixel_intersection_not_pixel_count():
    mask = np.ones((2, 2), dtype=bool)
    assert mask_box_area(mask, (0.25, 0.5, 1.5, 1.75)) == pytest.approx(1.5625)
    sparse = np.array([[1, 0], [0, 1]], dtype=bool)
    assert mask_box_area(sparse, (0.25, 0.5, 1.5, 1.75)) == pytest.approx(0.75)
    assert mask_box_area(mask, (1, 1, 1, 2)) == 0


def test_labels_are_independent_regions_and_worst_lung_not_resized_area(lungs):
    left, right = lungs
    labels = anatomy_retention_labels(left, right, maira_geometry((100, 100)), **label_args())
    assert labels["targets"] == {"consolidation": 1.0, "pleural_effusion": 1.0, "pneumothorax": 1.0}
    assert labels["per_finding"]["pleural_effusion"]["per_lung"]["manual_left"]["source_area"] == 600
    # A square native crop removes bottom20 and background columns, giving an
    # exact analytic viewport; odd resize effects are checked separately above.
    geometry = maira_geometry((100, 100), (10, 0, 90, 80))
    labels = anatomy_retention_labels(left, right, geometry, **label_args())
    assert labels["targets"]["consolidation"] == pytest.approx(70 / 80)
    assert labels["targets"]["pleural_effusion"] == pytest.approx(0.5)
    # Rectangle 30x80: 1-pixel inner ring has 216 pixels; removed bottom10
    # takes the bottom30 and 2*9 side pixels.
    assert labels["targets"]["pneumothorax"] == pytest.approx(168 / 216)
    origin = labels["fov_label_provenance"]
    assert origin["clinical_support_labels"] is False
    assert origin["protocol_sha256"] == LABEL_PROTOCOL_SHA256
    assert "claim_supported" not in json.dumps(labels)


def test_continuous_basal_band_uses_fractional_row_area():
    left, right = np.zeros((20, 20), bool), np.zeros((20, 20), bool)
    left[2:7, 2:5] = True
    right[2:7, 15:18] = True
    labels = anatomy_retention_labels(left, right, maira_geometry((20, 20)), **label_args())
    assert labels["per_finding"]["pleural_effusion"]["per_lung"]["manual_left"]["source_area"] == 3.75


def test_two_lung_min_does_not_average_away_unilateral_loss(lungs):
    labels = anatomy_retention_labels(*lungs, maira_geometry((100, 100), (20, 0, 100, 100)), **label_args())
    details = labels["per_finding"]["consolidation"]["per_lung"]
    assert labels["targets"]["consolidation"] == min(v["retention"] for v in details.values())
    assert details["manual_left"]["retention"] < details["manual_right"]["retention"]


def test_mask_overlay_audit_uses_real_mapping_but_is_not_target_area():
    mask = np.zeros((23, 11), dtype=bool)
    mask[4:18, 2:9] = True
    geometry = maira_geometry((11, 23), (1, 2, 11, 21))
    transformed, audit = transform_manual_mask(mask, geometry)
    assert transformed.shape == (518, 518)
    assert audit["used_for_target_area"] is False
    assert audit["source_mask_array_sha256"] == array_sha256(mask, kind="mask")
    expected = Image.fromarray(mask.astype(np.uint8)).crop((1, 2, 11, 21)).resize(
        (518, 984), Image.Resampling.NEAREST).crop((0, 233, 518, 751))
    np.testing.assert_array_equal(transformed, np.asarray(expected).astype(bool))


def test_entirely_removed_region_has_zero_retention_not_missing_label():
    left, right = np.zeros((200, 100), bool), np.zeros((200, 100), bool)
    left[0:20, 10:30] = True
    right[0:20, 70:90] = True
    geometry = maira_geometry((100, 200))
    labels = anatomy_retention_labels(left, right, geometry, **label_args())
    assert set(labels["targets"].values()) == {0.0}
    transformed, audit = transform_manual_mask(left, geometry)
    assert not transformed.any() and audit["presented_mask_foreground_pixels"] == 0


def test_inner_boundary_handles_native_frame_border():
    left, right = np.zeros((100, 100), bool), np.zeros((100, 100), bool)
    left[:, :30] = True
    right[:, 70:] = True
    labels = anatomy_retention_labels(left, right, maira_geometry((100, 100)), **label_args())
    assert labels["per_finding"]["pneumothorax"]["per_lung"]["manual_left"]["source_area"] == 256


def test_current_features_are_channel_permutation_invariant_without_patient_side(lungs):
    probabilities = np.stack(lungs).astype(np.float32) * 0.8
    first = current_image_geometry_features(probabilities, **feature_args())
    swapped = current_image_geometry_features(probabilities[::-1], **feature_args())
    assert first["features"] == swapped["features"]
    assert first["lung_channel_order"] == {"image_left": 0, "image_right": 1}
    assert swapped["lung_channel_order"] == {"image_left": 1, "image_right": 0}
    assert first["clinical_laterality_inferred"] is False
    assert first["features"]["bilateral_reflected_dice"] == 1.0
    assert first["features"]["image_left_extent"] == 1.0
    assert first["features"]["image_left_area_fraction"] == 0.24
    assert first["features"]["image_left_foreground_probability_mean"] == pytest.approx(0.8)
    assert first["features"]["image_left_contact_left"] == 0
    assert sum(first["features"][f"image_left_mass_{part}_third"] for part in ("upper", "middle", "lower")) == pytest.approx(1)
    assert all(0 <= value <= 1 for value in first["features"].values())
    assert set(first["feature_provenance"]) == set(first["features"])
    assert {p["source"] for p in first["feature_provenance"].values()} == {"deployable_current_image"}
    assert FEATURE_PROTOCOL_SHA256 == stable_hash(FEATURE_PROTOCOL)


def test_actual_edge_contact_and_confidence_extremes():
    probabilities = np.zeros((2, 6, 6), dtype=float)
    probabilities[0, :, :2] = 1
    probabilities[1, :, 4:] = 1
    features = current_image_geometry_features(probabilities, **feature_args())["features"]
    assert features["image_left_contact_left"] == 1
    assert features["image_left_contact_top"] == pytest.approx(1 / 3)
    assert features["image_right_contact_right"] == 1
    assert features["image_left_foreground_entropy_mean"] == 0


def test_deployment_signature_cannot_receive_geometry_or_manual_labels(lungs):
    assert set(inspect.signature(current_image_geometry_features).parameters) == {
        "lung_probabilities", "presented_image_sha256", "producer_revision", "weights_sha256"}
    probabilities = np.stack(lungs).astype(float)
    with pytest.raises(TypeError):
        current_image_geometry_features(probabilities, geometry=maira_geometry((100, 100)), **feature_args())


@pytest.mark.parametrize("source,crop", [
    ((0, 10), None), ((10.0, 10), None), ((True, 10), None),
    ((10, 10), (0, 0, 11, 10)), ((10, 10), (0, 0, 0, 10)),
    ((10, 10), (-1, 0, 10, 10)), ((10, 10), (0, 0.5, 10, 10)),
])
def test_reject_invalid_native_geometry(source, crop):
    with pytest.raises(ValueError):
        maira_geometry(source, crop)


@pytest.mark.parametrize("mask", [np.zeros((2, 2)), np.ones((2, 2, 1)),
                                 np.array([[0, 255]]), np.array([[np.nan, 1]]), np.array([[0.5, 1]])])
def test_reject_empty_or_nonbinary_manual_masks(mask):
    with pytest.raises(ValueError):
        mask_box_area(mask, (0, 0, 1, 1))


def test_reject_misalignment_overlap_tampered_geometry_and_hashes(lungs):
    with pytest.raises(ValueError, match="dimensions"):
        anatomy_retention_labels(*lungs, maira_geometry((99, 100)), **label_args())
    with pytest.raises(ValueError, match="overlap"):
        anatomy_retention_labels(lungs[0], lungs[0], maira_geometry((100, 100)), **label_args())
    geometry = maira_geometry((100, 100))
    geometry["visible_box"][0] = 1
    with pytest.raises(ValueError, match="fixed MAIRA"):
        anatomy_retention_labels(*lungs, geometry, **label_args())
    args = label_args()
    args["source_image_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="SHA256"):
        anatomy_retention_labels(*lungs, maira_geometry((100, 100)), **args)
    args = label_args()
    args["manual_mask_sha256"] = {"left": "b" * 64}
    with pytest.raises(ValueError, match="left/right"):
        anatomy_retention_labels(*lungs, maira_geometry((100, 100)), **args)


@pytest.mark.parametrize("bad", [np.zeros((2, 2, 2)), np.ones((3, 2, 2)),
                                np.ones((2, 1, 2)), np.full((2, 2, 2), np.nan),
                                np.full((2, 2, 2), 1.01), np.full((2, 2, 2), -0.01),
                                np.ones((2, 2, 2), dtype=np.uint8), np.ones((2, 2, 2))])
def test_reject_missing_invalid_or_ambiguous_predicted_lungs(bad):
    with pytest.raises(ValueError):
        current_image_geometry_features(bad, **feature_args())


@pytest.mark.parametrize("field,value", [("presented_image_sha256", "a" * 63),
                                       ("weights_sha256", "A" * 64),
                                       ("producer_revision", "main")])
def test_feature_binding_requires_fixed_valid_identifiers(lungs, field, value):
    args = feature_args()
    args[field] = value
    with pytest.raises(ValueError):
        current_image_geometry_features(np.stack(lungs).astype(float), **args)


def test_array_hashes_bind_contents_and_shape_without_mutating_inputs(lungs):
    p = np.stack(lungs).astype(float)
    before = p.copy()
    first = current_image_geometry_features(p, **feature_args())
    np.testing.assert_array_equal(p, before)
    p[0, 0, 0] = 0.1
    second = current_image_geometry_features(p, **feature_args())
    assert first["probabilities_sha256"] != second["probabilities_sha256"]
    assert array_sha256(lungs[0], kind="mask") == array_sha256(lungs[0].astype(np.uint8), kind="mask")
    json.dumps(first, allow_nan=False)


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="optional actual HF processor cross-check")
@pytest.mark.parametrize("size,crop", [((17, 25), None), ((25, 17), (1, 2, 25, 16)),
                                      ((1000, 1400), None), ((517, 518), None)])
def test_optional_crosscheck_actual_hf_bit_processor(size, crop):
    # Module itself never imports torch/transformers. On the server this runs an
    # independent real-processor CPU check, with nearest interpolation for masks.
    from transformers import BitImageProcessor
    width, height = size
    mask = (np.indices((height, width)).sum(axis=0) % 7 < 3)
    geometry = maira_geometry(size, crop)
    actual, _ = transform_manual_mask(mask, geometry)
    image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB").crop(tuple(geometry["native_crop_box"]))
    processor = BitImageProcessor(size={"shortest_edge": 518}, crop_size={"height": 518, "width": 518},
                                  resample=Image.Resampling.NEAREST, do_rescale=False, do_normalize=False)
    hf = processor(images=image, return_tensors="np")["pixel_values"][0, 0] > 0
    np.testing.assert_array_equal(actual, hf)

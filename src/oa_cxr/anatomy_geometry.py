"""Independent anatomical-retention proxies, never clinical O-minus labels.

Label functions may see native manual masks and construction geometry. The
deployment feature function accepts only probabilities predicted from the current
presented image. Keep these artifacts separate from clinical ``claim_supported``
labels and from each other when constructing model inputs.

Coordinates are pixel *edges*: pixel (y, x) occupies [x,x+1] x [y,y+1]. Source
images and masks must already have the same orientation/alignment; this module
does not infer EXIF transforms, patient side, or registration. MAIRA's nominal
viewport is modeled exactly; the bicubic interpolation kernel's small boundary
footprint is not counted as observable anatomy.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .io import stable_hash


GEOMETRY_PROTOCOL = {
    "version": "maira-native-viewport-v1",
    "processor": "transformers==4.51.3/BitImageProcessor",
    "resize": "shortest edge 518; long edge int(518*long/short)",
    "center_crop": "518x518; offset=(resized_edge-518)//2",
    "coordinates": "pixel edges in aligned native source; x0,y0,x1,y1",
    "viewport": "inverse affine of nominal crop rectangle; excludes bicubic kernel halo",
    "mask_audit_interpolation": "nearest, not the bicubic image interpolation",
}
LABEL_PROTOCOL = {
    "version": "manual-visible-lung-retention-v1",
    "geometry_protocol_sha256": stable_hash(GEOMETRY_PROTOCOL),
    "area": "continuous viewport intersection with native unit mask pixels",
    "denominator": "annotated region area in source radiograph, not resized mask area",
    "aggregation": "minimum of the two independently annotated lungs",
    "consolidation": "whole-visible-lung proxy",
    "pleural_effusion": "basal-lung proxy: bottom 25% of each native lung bbox",
    "pneumothorax": "inner-boundary proxy: mask minus Euclidean erosion, radius=max(1,0.02*bbox_short_edge)",
    "clinical_support_labels": False,
    "scope": "retention of source-visible annotated anatomy, not full lung/pleura or O-minus",
}
FEATURE_PROTOCOL = {
    "version": "current-presented-lung-probabilities-v1",
    "input": "two independent sigmoid probability maps of the current presented image only",
    "threshold": 0.5,
    "channel_order": "ascending hard-mask x centroid: image_left,image_right; no clinical laterality",
    "coordinates": "pixel centers normalized by current width/height",
    "extent": "hard-mask area divided by its axis-aligned bbox area",
    "compactness": "4*pi*area/(four-neighbor exposed pixel-edge perimeter squared)",
    "contact": "occupied pixels on exact image edge / corresponding edge length",
    "vertical_distribution": "hard-mask mass in image-frame thirds, fractional row intersections",
    "confidence": "mean foreground probability and binary entropy within predicted mask; not calibrated accuracy",
    "symmetry": "min/max area,height,width; reflected-mask Dice; centroid mirror agreement",
    "original_image_or_crop_parameters": False,
}
GEOMETRY_PROTOCOL_SHA256 = stable_hash(GEOMETRY_PROTOCOL)
LABEL_PROTOCOL_SHA256 = stable_hash(LABEL_PROTOCOL)
FEATURE_PROTOCOL_SHA256 = stable_hash(FEATURE_PROTOCOL)


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _digest(value: str, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _revision(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty fixed revision/source identifier")
    if value.lower() in {"main", "master", "latest", "unknown"}:
        raise ValueError(f"{name} must be fixed, not a moving or unknown revision")
    return value


def _mask(mask: np.ndarray, name: str = "mask") -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or min(mask.shape) < 1:
        raise ValueError(f"{name} must be a nonempty HxW numpy array")
    if mask.dtype.kind not in "buif" or not np.isfinite(mask).all():
        raise ValueError(f"{name} must contain finite binary values")
    # Preserve explicit mask semantics: never silently threshold an antialiased
    # mask, a disease map, or a probability map as a manual annotation.
    if not np.logical_or(mask == 0, mask == 1).all():
        raise ValueError(f"{name} must be binary 0/1; decode PNG 0/255 explicitly")
    result = mask.astype(bool, copy=False)
    if not result.any():
        raise ValueError(f"{name} has no annotated/predicted region")
    return result


def array_sha256(array: np.ndarray, *, kind: str) -> str:
    """Hash canonical array shape/type/bytes, distinct from a PNG/NPY file hash.

    ``mask`` canonicalizes validated binary arrays to uint8. ``probabilities``
    canonicalizes to little-endian float64 without silently rounding to float32.
    File-byte hashes, when supplied to other functions, remain separate bindings.
    """
    if kind == "mask":
        canonical = np.ascontiguousarray(_mask(array), dtype=np.uint8)
    elif kind == "probabilities":
        canonical = _probabilities(array).astype("<f8", copy=False)
        canonical = np.ascontiguousarray(canonical)
    else:
        raise ValueError("Array hash kind must be mask or probabilities")
    header = {"schema": "anatomy-array-sha256-v1", "kind": kind,
              "shape": list(canonical.shape), "dtype": canonical.dtype.str}
    digest = hashlib.sha256(stable_hash(header).encode("ascii") + b"\n")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def maira_geometry(source_size: Sequence[int], crop_box: Sequence[int] | None = None) -> dict:
    """Compose integer native crop, HF resize, and center crop back to source.

    ``source_size`` is (width,height); native crop is an in-bounds integer PIL
    box. No padding or alternative processor configuration is accepted. The HF
    rules are pinned in transformers v4.51.3 ``image_transforms.py`` functions
    ``get_resize_output_image_size`` and ``center_crop``.
    """
    if not isinstance(source_size, (tuple, list)) or len(source_size) != 2:
        raise ValueError("source_size must be (width,height)")
    width, height = [_integer(v, "source dimension", 1) for v in source_size]
    if crop_box is None:
        crop_box = (0, 0, width, height)
    if not isinstance(crop_box, (tuple, list)) or len(crop_box) != 4:
        raise ValueError("crop_box must be an integer (x0,y0,x1,y1)")
    x0, y0, x1, y1 = [_integer(v, "crop coordinate") for v in crop_box]
    if not (x0 < x1 <= width and y0 < y1 <= height):
        raise ValueError("crop_box must be nonempty and inside the native source")
    crop_width, crop_height = x1 - x0, y1 - y0
    if crop_width <= crop_height:
        resized_width, resized_height = 518, int(518 * crop_height / crop_width)
    else:
        resized_width, resized_height = int(518 * crop_width / crop_height), 518
    left, top = (resized_width - 518) // 2, (resized_height - 518) // 2
    scale_x, scale_y = resized_width / crop_width, resized_height / crop_height
    return {
        "source_size": [width, height], "native_crop_box": [x0, y0, x1, y1],
        "resized_size": [resized_width, resized_height],
        "center_crop_box": [left, top, left + 518, top + 518],
        "presented_size": [518, 518],
        "visible_box": [x0 + left / scale_x, y0 + top / scale_y,
                        x0 + (left + 518) / scale_x, y0 + (top + 518) / scale_y],
        "protocol_sha256": GEOMETRY_PROTOCOL_SHA256,
    }


def _geometry(geometry: dict) -> dict:
    if not isinstance(geometry, dict):
        raise ValueError("geometry must be the complete maira_geometry artifact")
    expected = maira_geometry(geometry.get("source_size"), geometry.get("native_crop_box"))
    if geometry != expected:
        raise ValueError("Geometry differs from the fixed MAIRA processor protocol")
    return expected


def transform_manual_mask(mask: np.ndarray, geometry: dict) -> tuple[np.ndarray, dict]:
    """Return nearest-resampled presented mask for overlay audit, not target area."""
    geometry = _geometry(geometry)
    mask = _mask(mask)
    if list(mask.shape[::-1]) != geometry["source_size"]:
        raise ValueError("Manual mask does not match source dimensions/alignment")
    image = Image.fromarray(mask.astype(np.uint8))
    transformed = np.asarray(image.crop(tuple(geometry["native_crop_box"])).resize(
        tuple(geometry["resized_size"]), Image.Resampling.NEAREST).crop(
        tuple(geometry["center_crop_box"])), dtype=bool)
    # An entirely removed region is a valid zero-retention audit result.
    output_hash = hashlib.sha256(np.ascontiguousarray(transformed, dtype=np.uint8).tobytes()).hexdigest()
    audit = {"geometry": geometry, "source_mask_array_sha256": array_sha256(mask, kind="mask"),
             "presented_mask_uint8_bytes_sha256": output_hash,
             "presented_mask_shape": list(transformed.shape),
             "presented_mask_foreground_pixels": int(transformed.sum()),
             "resampling": "nearest", "used_for_target_area": False,
             "definition": "same geometric mapping as MAIRA; mask interpolation intentionally differs from bicubic image"}
    return transformed, audit


def _box(box, width: int, height: int) -> tuple[float, float, float, float]:
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        raise ValueError("visible/region box must have four numeric coordinates")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number)) for v in box):
        raise ValueError("Box coordinates must be finite real numbers")
    values = tuple(float(v) for v in box)
    if not np.isfinite(values).all():
        raise ValueError("Box coordinates must be finite")
    x0, y0, x1, y1 = values
    if not (0 <= x0 <= x1 <= width and 0 <= y0 <= y1 <= height):
        raise ValueError("Box is outside source dimensions or has reversed edges")
    return values


def mask_box_area(mask: np.ndarray, box: Sequence[float]) -> float:
    """Continuous intersection area of binary unit mask pixels with a rectangle.

    Empty rectangles return zero; empty *source regions* are rejected so callers
    cannot accidentally obtain a fabricated retention denominator.
    """
    mask = _mask(mask)
    height, width = mask.shape
    x0, y0, x1, y1 = _box(box, width, height)
    return _mask_box_area_validated(mask, (x0, y0, x1, y1))


def _mask_box_area_validated(mask: np.ndarray, box) -> float:
    """Internal fast path for immutable, once-validated prepared regions."""
    x0, y0, x1, y1 = box
    if x0 == x1 or y0 == y1:
        return 0.0
    ix0, iy0 = int(np.floor(x0)), int(np.floor(y0))
    ix1, iy1 = int(np.ceil(x1)), int(np.ceil(y1))
    xs, ys = np.arange(ix0, ix1), np.arange(iy0, iy1)
    wx = np.maximum(0.0, np.minimum(xs + 1, x1) - np.maximum(xs, x0))
    wy = np.maximum(0.0, np.minimum(ys + 1, y1) - np.maximum(ys, y0))
    # Avoid a full-size float mask on 4K source images.
    return float(np.einsum("ij,i,j->", mask[iy0:iy1, ix0:ix1], wy, wx,
                           dtype=np.float64, optimize=False))


def _intersection(a, b) -> list[float]:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    return [x0, y0, max(x0, min(a[2], b[2])), max(y0, min(a[3], b[3]))]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _inner_boundary(mask: np.ndarray) -> tuple[np.ndarray, float]:
    x0, y0, x1, y1 = _bbox(mask)
    radius = max(1.0, 0.02 * min(x1 - x0, y1 - y0))
    # Explicit zero padding makes boundaries correct even for masks touching a
    # source edge. Only allocate the distance map for the lung bounding box.
    local = mask[y0:y1, x0:x1]
    distance = distance_transform_edt(np.pad(local, 1, constant_values=False))[1:-1, 1:-1]
    rim = np.zeros_like(mask)
    rim[y0:y1, x0:x1] = local & (distance <= radius)
    return _mask(rim, "inner-boundary region"), radius


@dataclass(frozen=True)
class PreparedRegion:
    finding: str
    lung: str
    mask: np.ndarray
    box: tuple[float, float, float, float]
    source_area: float
    erosion_radius: float | None = None


@dataclass(frozen=True)
class PreparedManualRegions:
    """Read-only native ROIs reusable across variants; never deployment input."""
    source_size: tuple[int, int]
    regions: tuple[PreparedRegion, ...]
    mask_array_sha256: tuple[tuple[str, str], ...]


def _immutable_mask(mask: np.ndarray) -> np.ndarray:
    # Immutable bytes backing prevents setflags(write=True), not only ordinary
    # assignment. Callers may mutate their original arrays after preparation.
    return np.frombuffer(mask.tobytes(order="C"), dtype=np.bool_).reshape(mask.shape)


def prepare_manual_regions(left_mask: np.ndarray, right_mask: np.ndarray) -> PreparedManualRegions:
    """Compute and freeze native regions/EDT once per source radiograph."""
    masks = {"manual_left": _mask(left_mask, "left manual mask"),
             "manual_right": _mask(right_mask, "right manual mask")}
    if masks["manual_left"].shape != masks["manual_right"].shape:
        raise ValueError("Manual masks must share source dimensions and alignment")
    if np.logical_and(*masks.values()).any():
        raise ValueError("Manual left/right lung masks overlap; check annotation alignment")
    height, width = masks["manual_left"].shape
    full = (0.0, 0.0, float(width), float(height))
    regions, hashes = [], []
    for name, input_mask in masks.items():
        mask = _immutable_mask(input_mask)
        hashes.append((name, array_sha256(mask, kind="mask")))
        x0, y0, x1, y1 = _bbox(mask)
        basal_box = (float(x0), y0 + 0.75 * (y1 - y0), float(x1), float(y1))
        rim, radius = _inner_boundary(mask)
        rim = _immutable_mask(rim)
        for finding, region, box, erosion_radius in (
            ("consolidation", mask, full, None),
            ("pleural_effusion", mask, basal_box, None),
            ("pneumothorax", rim, full, radius),
        ):
            denominator = _mask_box_area_validated(region, box)
            if denominator <= 0:
                raise ValueError(f"Empty {finding} region in {name}; no proxy can be defined")
            regions.append(PreparedRegion(finding, name, region, box, denominator, erosion_radius))
    return PreparedManualRegions((width, height), tuple(regions), tuple(hashes))


def anatomy_retention_labels(
    left_mask: np.ndarray, right_mask: np.ndarray, geometry: dict, *,
    source_image_sha256: str, manual_mask_sha256: Mapping[str, str],
    annotation_source: str, processor_revision: str,
) -> dict:
    """Generate continuous finding-conditioned *geometric* targets from masks.

    ``manual_mask_sha256`` contains the actual annotation file digests under
    ``left`` and ``right``; callers verify file contents/alignment before passing
    decoded arrays. Canonical array hashes are additionally recorded here.
    ``left_mask`` and ``right_mask`` retain the annotation file naming only;
    their minimum aggregation does not infer or require clinical laterality.
    """
    return anatomy_retention_labels_from_regions(
        prepare_manual_regions(left_mask, right_mask), geometry,
        source_image_sha256=source_image_sha256, manual_mask_sha256=manual_mask_sha256,
        annotation_source=annotation_source, processor_revision=processor_revision)


def anatomy_retention_labels_from_regions(
    prepared: PreparedManualRegions, geometry: dict, *,
    source_image_sha256: str, manual_mask_sha256: Mapping[str, str],
    annotation_source: str, processor_revision: str,
) -> dict:
    """Same exact labels, reusing the output of ``prepare_manual_regions``.

    Treat prepared instances as opaque: construct them only with the preparation
    function. They contain immutable native annotations and must never enter the
    current-image deployment feature pipeline.
    """
    geometry = _geometry(geometry)
    if not isinstance(prepared, PreparedManualRegions):
        raise ValueError("prepared must come from prepare_manual_regions")
    if list(prepared.source_size) != geometry["source_size"]:
        raise ValueError("Manual masks must share the source image dimensions and alignment")
    _digest(source_image_sha256, "source_image_sha256")
    if not isinstance(manual_mask_sha256, Mapping) or set(manual_mask_sha256) != {"left", "right"}:
        raise ValueError("manual_mask_sha256 must have exactly left/right file digests")
    hashes = {name: _digest(value, f"{name} manual mask SHA256") for name, value in manual_mask_sha256.items()}
    _revision(annotation_source, "annotation_source")
    _revision(processor_revision, "processor_revision")
    definitions = {"consolidation": "whole-visible-lung proxy",
                   "pleural_effusion": "basal-lung proxy",
                   "pneumothorax": "inner-boundary proxy"}
    details = {finding: {"proxy_region": region, "per_lung": {}}
               for finding, region in definitions.items()}
    for region in prepared.regions:
        numerator = _mask_box_area_validated(region.mask, _intersection(region.box, geometry["visible_box"]))
        retention = numerator / region.source_area
        if not np.isfinite(retention) or not 0 <= retention <= 1 + 1e-12:
            raise ValueError("Invalid anatomical-retention calculation")
        entry = {"retention": float(min(1.0, retention)), "source_area": region.source_area,
                 "visible_area": numerator}
        if region.finding == "pleural_effusion":
            entry["native_basal_box"] = list(region.box)
        if region.finding == "pneumothorax":
            entry["native_erosion_radius"] = region.erosion_radius
        details[region.finding]["per_lung"][region.lung] = entry
    return {
        "targets": {finding: min(v["retention"] for v in item["per_lung"].values())
                    for finding, item in details.items()},
        "per_finding": details,
        "fov_label_provenance": {
            "source": "manual_mask_fov_label", "protocol_sha256": LABEL_PROTOCOL_SHA256,
            "geometry_protocol_sha256": GEOMETRY_PROTOCOL_SHA256,
            "source_image_sha256": source_image_sha256, "manual_mask_sha256": hashes,
            "manual_mask_array_sha256": dict(prepared.mask_array_sha256),
            "annotation_source": annotation_source, "processor_revision": processor_revision,
            "geometry": geometry, "geometry_sha256": stable_hash(geometry),
            "scope": LABEL_PROTOCOL["scope"], "clinical_support_labels": False,
            "clinical_laterality_inferred": False,
            "binding_verification": "file digests and image/mask alignment are caller-verified; array digests computed here",
        },
    }


def _probabilities(probabilities: np.ndarray) -> np.ndarray:
    if not isinstance(probabilities, np.ndarray) or probabilities.ndim != 3 or probabilities.shape[0] != 2:
        raise ValueError("Lung probabilities must be a numpy array of shape [2,H,W]")
    if min(probabilities.shape[1:]) < 2 or probabilities.dtype.kind != "f":
        raise ValueError("Lung probabilities require floating-point H,W >= 2")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Lung probabilities must be finite sigmoid probabilities in [0,1], not logits")
    return probabilities


def _lung_features(probability: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = _bbox(mask)
    area = int(mask.sum())
    padded = np.pad(mask.astype(np.int8), 1)
    perimeter = int(np.abs(np.diff(padded, axis=0)).sum() + np.abs(np.diff(padded, axis=1)).sum())
    p = probability[mask].astype(np.float64)
    # Entropy is only a segmenter's self-confidence measure, not mask accuracy.
    entropy = -(p * np.log2(np.clip(p, 1e-15, 1)) + (1 - p) * np.log2(np.clip(1 - p, 1e-15, 1)))
    result = {
        "area_fraction": area / (height * width),
        "centroid_x": float((xs.mean() + 0.5) / width),
        "centroid_y": float((ys.mean() + 0.5) / height),
        "bbox_width": (x1 - x0) / width, "bbox_height": (y1 - y0) / height,
        "extent": area / ((x1 - x0) * (y1 - y0)),
        "compactness": float(4 * np.pi * area / perimeter**2),
        "contact_top": float(mask[0, :].mean()), "contact_bottom": float(mask[-1, :].mean()),
        "contact_left": float(mask[:, 0].mean()), "contact_right": float(mask[:, -1].mean()),
        "foreground_probability_mean": float(p.mean()), "foreground_entropy_mean": float(entropy.mean()),
    }
    for index, region in enumerate(("upper", "middle", "lower")):
        result[f"mass_{region}_third"] = mask_box_area(
            mask, [0.0, index * height / 3, float(width), (index + 1) * height / 3]) / area
    return result


def current_image_geometry_features(
    lung_probabilities: np.ndarray, *, presented_image_sha256: str,
    producer_revision: str, weights_sha256: str,
) -> dict:
    """Extract deployable features with no source image, crop, mask, or label input.

    The caller must bind the segmenter inference to the verified presented-image
    file digest and pin the segmenter/preprocessing revision and weight digest.
    PSPNet channels are independent sigmoid probabilities, not a two-class
    softmax. No clinical left/right side is inferred from channel order.
    """
    probabilities = _probabilities(lung_probabilities)
    _digest(presented_image_sha256, "presented_image_sha256")
    _digest(weights_sha256, "weights_sha256")
    _revision(producer_revision, "producer_revision")
    masks = [_mask(p >= 0.5, f"predicted lung channel {i}") for i, p in enumerate(probabilities)]
    per_channel = [_lung_features(p, mask) for p, mask in zip(probabilities, masks)]
    if abs(per_channel[0]["centroid_x"] - per_channel[1]["centroid_x"]) <= 1e-12:
        raise ValueError("Lung x centroids are indistinguishable; image-left/right order is undefined")
    order = sorted(range(2), key=lambda index: per_channel[index]["centroid_x"])
    left, right = [per_channel[index] for index in order]
    left_mask, right_mask = [masks[index] for index in order]
    features = {f"image_{side}_{key}": value for side, values in (("left", left), ("right", right))
                for key, value in values.items()}
    for key in ("area_fraction", "bbox_width", "bbox_height"):
        features[f"bilateral_{key}_symmetry"] = min(left[key], right[key]) / max(left[key], right[key])
    features["bilateral_reflected_dice"] = float(
        2 * (left_mask & right_mask[:, ::-1]).sum() / (left_mask.sum() + right_mask.sum()))
    features["bilateral_centroid_x_mirror_agreement"] = 1 - abs(left["centroid_x"] + right["centroid_x"] - 1)
    features["bilateral_centroid_y_agreement"] = 1 - abs(left["centroid_y"] - right["centroid_y"])
    features["bilateral_overlap_fraction"] = float((left_mask & right_mask).sum() / (left_mask | right_mask).sum())
    if any(not np.isfinite(v) or not 0 <= v <= 1 + 1e-12 for v in features.values()):
        raise ValueError("Nonfinite or out-of-range geometry feature")
    features = {key: float(np.clip(value, 0, 1)) for key, value in features.items()}
    origin = {"source": "deployable_current_image", "producer_revision": producer_revision,
              "weights_sha256": weights_sha256, "protocol_sha256": FEATURE_PROTOCOL_SHA256,
              "definition": "current predicted lung shape/confidence; not anatomical retention or clinical support probability"}
    return {
        "features": features, "feature_provenance": {key: dict(origin) for key in features},
        "feature_protocol_sha256": FEATURE_PROTOCOL_SHA256,
        "presented_image_sha256": presented_image_sha256,
        "probabilities_sha256": array_sha256(probabilities, kind="probabilities"),
        "probabilities_shape": list(probabilities.shape),
        "lung_channel_order": {"image_left": order[0], "image_right": order[1]},
        "clinical_laterality_inferred": False, "clinical_support_labels": False,
        "binding_verification": "presented file digest and segmenter input are caller-verified; probability array digest computed here",
    }

"""Mask-only structural comparator using deployed predictions, never source GT.

The auxiliary segmenter is shared with OA-CXR. This is not an independent
pretrained-encoder baseline. Coordinates below describe the presented predicted
mask, not crop recipes or original source-image coordinates.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter

FINDINGS = ("pleural_effusion", "pneumothorax", "consolidation")
VERSION = "predicted-mask-geometry-v1"
MEASURES = ("area", "bbox_x", "bbox_y", "bbox_width", "bbox_height", "centroid_x",
            "centroid_y", "top_contact", "bottom_contact", "left_contact", "right_contact", "empty")
REGRESSOR = dict(loss="squared_error", learning_rate=.05, max_iter=200, max_leaf_nodes=15,
                 max_depth=None, min_samples_leaf=30, l2_regularization=1., max_bins=255,
                 early_stopping=False, random_state=17, categorical_features=None)


def feature_names():
    block = [f"lung_{lung}_{name}" for lung in (0, 1) for name in MEASURES]
    block += ["bilateral_abs_difference_" + name for name in MEASURES]
    return [prefix + name for prefix in ("whole_", "query_region_") for name in block] + [
        "finding_" + name for name in FINDINGS]


def _binary(masks):
    value = np.asarray(masks)
    if (value.ndim != 4 or value.shape[1] != 2 or min(value.shape) < 1 or
            value.dtype not in (np.dtype(bool), np.dtype(np.uint8))):
        raise ValueError("boolean/binary uint8 [inputs,2,height,width] masks required")
    if value.dtype == np.uint8 and np.any(value > 1):
        raise ValueError("uint8 masks must be binary 0/1, not probabilities or packed bytes")
    return value.astype(bool, copy=False)


def _boxes(m):
    ys, xs = m.any(-1), m.any(-2)
    present = ys.any(-1)
    y0, x0 = ys.argmax(-1), xs.argmax(-1)
    y1, x1 = m.shape[-2] - ys[..., ::-1].argmax(-1), m.shape[-1] - xs[..., ::-1].argmax(-1)
    return tuple(np.where(present, a, 0) for a in (x0, y0, x1, y1)), present


def _summaries(m):
    height, width = m.shape[-2:]
    (x0, y0, x1, y1), present = _boxes(m)
    row_mass, col_mass = m.sum(-1, dtype=np.int64), m.sum(-2, dtype=np.int64)
    mass = row_mass.sum(-1, dtype=np.int64)
    cx = (col_mass @ (np.arange(width, dtype=np.float64) + .5)) / np.maximum(mass, 1) / width
    cy = (row_mass @ (np.arange(height, dtype=np.float64) + .5)) / np.maximum(mass, 1) / height
    values = np.stack((mass / (height * width), x0 / width, y0 / height,
                       (x1 - x0) / width, (y1 - y0) / height, cx, cy,
                       m[..., 0, :].mean(-1), m[..., -1, :].mean(-1),
                       m[..., :, 0].mean(-1), m[..., :, -1].mean(-1), ~present), axis=-1)
    return np.concatenate((values.reshape(len(m), -1), np.abs(values[:, 0] - values[:, 1])), axis=-1)


def geometry_features(predicted_masks):
    """Return [inputs,3,75] in FINDINGS order; empty masks remain valid evidence.

    Hard final-resolution predicted masks are the sole input. The basal region
    is the bottom quarter of the *current predicted* bbox. The rim uses square
    erosion, ceil(2% of bbox short edge), minimum one pixel; it includes medial
    boundaries. Neither is claimed to be a clinical localization or source GT.
    """
    m = _binary(predicted_masks)
    (x0, y0, x1, y1), _ = _boxes(m)
    basal = m & (np.arange(m.shape[-2])[None, None, :, None] >=
                 (y0 + .75 * (y1 - y0))[:, :, None, None])
    radii = np.maximum(1, np.ceil(.02 * np.minimum(x1 - x0, y1 - y0))).astype(int)
    rim = np.empty_like(m)
    for radius in np.unique(radii):
        selected = radii == radius
        eroded = minimum_filter(m[selected], size=(1, 2 * int(radius) + 1, 2 * int(radius) + 1),
                                 mode="constant", cval=0)
        rim[selected] = m[selected] & ~eroded
    whole = np.broadcast_to(_summaries(m)[:, None, :], (len(m), 3, 36))
    regional = np.stack([_summaries(region) for region in (basal, rim, m)], axis=1)
    onehot = np.broadcast_to(np.eye(3), (len(m), 3, 3))
    result = np.concatenate((whole, regional, onehot), axis=-1).astype(np.float32)
    if result.shape != (len(m), 3, len(feature_names())) or not np.isfinite(result).all():
        raise ValueError("geometry extraction produced invalid features")
    return result


def validate_query_rows(rows, split, expected_sources):
    """Strict original query order links each packed-mask index to three rows."""
    from oa_cxr.io import stable_hash
    if len(rows) != expected_sources * 39:
        raise ValueError("query/source denominator mismatch")
    source_keys, images, groups = set(), set(), set()
    for source_index in range(expected_sources):
        first = rows[source_index * 39]
        key = (first.get("dataset"), first.get("source_id"))
        if (not all(isinstance(v, str) and v for v in key) or key in source_keys or
                not isinstance(first.get("source_image_sha256"), str) or
                not isinstance(first.get("split_group_id"), str)):
            raise ValueError("missing/duplicate source identity")
        source_keys.add(key); images.add(first["source_image_sha256"]); groups.add(first["split_group_id"])
        for variant in range(13):
            for finding_index, finding in enumerate(FINDINGS):
                row = rows[source_index * 39 + variant * 3 + finding_index]
                if (any(row.get(k) != first.get(k) for k in
                        ("dataset", "source_id", "source_image_sha256", "split_group_id", "source_row_index")) or
                        row.get("cache_split") != split or row.get("split") != ("external" if split.startswith("external") else split) or
                        row.get("variant_index") != variant or row.get("finding") != finding or
                        row.get("input_id") != stable_hash(["rebuild-input-v1", *key, variant]) or
                        row.get("row_id") != stable_hash(["rebuild-query-v1", *key, variant, finding])):
                    raise ValueError("query row order/identity differs from packed mask order")
    if len(images) != expected_sources:
        raise ValueError("source image SHA identities are duplicated")
    return dict(source_keys=source_keys, images=images, groups=groups)


def predict_from_masks(regressor, predicted_masks):
    """Deployment/challenge inference, with no labels or original-image access."""
    features = geometry_features(predicted_masks)
    result = np.asarray(regressor.predict(features.reshape(-1, features.shape[-1])))
    if result.shape != (len(features) * 3,) or not np.isfinite(result).all():
        raise ValueError("regressor emitted missing/nonfinite scores")
    return np.clip(result, 0, 1).reshape(len(features), 3)

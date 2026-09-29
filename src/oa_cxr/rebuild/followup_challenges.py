"""Frozen, image-only challenge construction for relative anatomy retention.

Native reference masks determine targets and construction eligibility only.
Photometric invariance here means geometric retention, never clinical utility.
Invalid slots are explicit; their zero storage placeholders are not inputs.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from PIL import Image, ImageFilter

from oa_cxr.io import stable_hash
from oa_cxr.rebuild.data import FINDINGS, IMAGE_SIZE, letterbox, retention_targets

VERSION = "followup-fixed-challenges-20260925-v1"
DIRECTIONS = ("top", "bottom", "left", "right")
VARIANTS = (
    ("original", "anchor"),
    *((f"{side}_{percent:02d}", "unseen_single") for side in DIRECTIONS for percent in (5, 30)),
    ("left07_right23", "unseen_asymmetric"), ("left23_right07", "unseen_asymmetric"),
    ("top07_bottom23", "unseen_asymmetric"), ("top23_bottom07", "unseen_asymmetric"),
    ("background_only", "background"),
    ("brightness070", "photometric"), ("brightness130", "photometric"), ("blur_sigma2", "photometric"),
    ("left30_gray127", "padding"), ("left30_topleft", "padding"),
    ("top30_gray127", "padding"), ("top30_topleft", "padding"),
)
VARIANT_NAMES = tuple(name for name, _ in VARIANTS)
N_VARIANTS = len(VARIANTS)
NESTED_PAIRS = tuple((a, b) for side in DIRECTIONS
                     for a, b in (("original", f"{side}_05"), (f"{side}_05", f"{side}_30")))
SPECIFICATION = {
    "version": VERSION, "findings": list(FINDINGS), "image_size": IMAGE_SIZE,
    "variants": [{"name": name, "family": family} for name, family in VARIANTS],
    "unseen_single_crop_fractions": [.05, .30],
    "asymmetric_opposing_crop_fractions": [.07, .23],
    "crop_rounding": "Python round, matching current native data contract",
    "background_crop": "union lung bounding box expanded by ceil(2% native width/height), clipped; require removable margin and all three targets unchanged",
    "brightness": "multiply original 256 uint8 image by 0.7 or 1.3, np.rint then clip0..255; keep no-effect valid and flag",
    "blur": "Pillow GaussianBlur(radius=2.0) on original 256 uint8 image; keep no-effect valid and flag",
    "padding": "reuse exact resized raster of left30/top30; centered gray127 fill or black top-left placement; no new resizing; zero padding/offset means explicit ineligible",
    "base_presentation": "native integer crop then preserve-aspect bilinear resize, centered black256 letterbox; exact existing letterbox implementation",
    "nested_pairs": [list(pair) for pair in NESTED_PAIRS],
    "numerical_tolerance": 1e-6,
    "invalid_storage": "valid=false, targets=NaN, zero uint8 placeholder; never infer on placeholder",
    "clinical_invariance_claimed": False, "reference_masks_model_input": False,
    "old_observed_sources_are_new_blind_test": False,
}
SPECIFICATION_SHA256 = stable_hash(SPECIFICATION)


def _pixel_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def crop_boxes(size):
    """The thirteen fixed anchor/unseen boxes, without inspecting predictions."""
    w, h = map(int, size)
    if min(w, h) < 4:
        raise ValueError("Native image too small for predefined crop challenge")
    result = {"original": (0, 0, w, h)}
    for side in DIRECTIONS:
        for percent in (5, 30):
            dx, dy = round(percent / 100 * w), round(percent / 100 * h)
            result[f"{side}_{percent:02d}"] = (
                dx if side == "left" else 0, dy if side == "top" else 0,
                w - dx if side == "right" else w, h - dy if side == "bottom" else h)
    result.update(left07_right23=(round(.07*w), 0, w-round(.23*w), h),
                  left23_right07=(round(.23*w), 0, w-round(.07*w), h),
                  top07_bottom23=(0, round(.07*h), w, h-round(.23*h)),
                  top23_bottom07=(0, round(.23*h), w, h-round(.07*h)))
    return result


def background_box(masks, size):
    w, h = size
    if masks.shape != (2, h, w) or masks.dtype != np.bool_ or masks.all(axis=0).any():
        raise ValueError("Aligned disjoint bilateral Boolean source masks required")
    if not all(mask.any() for mask in masks):
        raise ValueError("Both native source masks must be nonempty")
    ys, xs = np.nonzero(masks.any(0))
    dx, dy = math.ceil(.02*w), math.ceil(.02*h)
    return (max(0, int(xs.min())-dx), max(0, int(ys.min())-dy),
            min(w, int(xs.max())+1+dx), min(h, int(ys.max())+1+dy))


def resized_raster(image, box):
    current = image.crop(box).convert("L")
    w, h = current.size
    scale = IMAGE_SIZE / max(w, h)
    nw, nh = max(1, round(w*scale)), max(1, round(h*scale))
    raster = np.asarray(current.resize((nw, nh), Image.Resampling.BILINEAR), dtype=np.uint8)
    offset = ((IMAGE_SIZE-nw)//2, (IMAGE_SIZE-nh)//2)
    return raster, offset


def _canvas(raster, offset, fill):
    result = np.full((IMAGE_SIZE, IMAGE_SIZE), fill, dtype=np.uint8)
    x, y = offset
    h, w = raster.shape
    result[y:y+h, x:x+w] = raster
    return result


def build_source(image, masks):
    """Return 21 fixed slots, separate targets/validity, and transform receipts.

    Ineligible conditional slots remain present and do not silently shrink any
    nominal denominator. Unexpected construction errors fail the source caller.
    """
    image = image.convert("L")
    masks = np.asarray(masks)
    background = background_box(masks, image.size)
    boxes = crop_boxes(image.size)
    images = np.zeros((N_VARIANTS, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    targets = np.full((N_VARIANTS, len(FINDINGS)), np.nan, dtype=np.float32)
    valid = np.zeros(N_VARIANTS, dtype=np.bool_)
    records = []
    index = {name: i for i, name in enumerate(VARIANT_NAMES)}
    base_targets = retention_targets(masks, list(boxes.values()))
    targets_by_name = dict(zip(boxes, base_targets))
    for i, (name, family) in enumerate(VARIANTS):
        anchor = "left_30" if name.startswith("left30_") else "top_30" if name.startswith("top30_") else "original"
        row = {"variant_index": i, "variant": name, "family": family, "anchor_variant": anchor,
               "status": "eligible", "reason": None, "geometric_retention_invariant": family in ("background", "photometric", "padding"),
               "clinical_assessability_invariant": False}
        box = boxes.get(name, boxes.get(anchor))
        if name in boxes:
            current = letterbox(image.crop(box))
            target = targets_by_name[name]
        elif name == "background_only":
            box = background
            if box == boxes["original"]:
                row.update(status="construction_ineligible", reason="no_removable_background_margin")
                current = target = None
            else:
                current = letterbox(image.crop(box))
                target = retention_targets(masks, [box])[0]
                if not np.allclose(target, targets_by_name["original"], atol=1e-6, rtol=0):
                    raise ValueError("Background-only crop changed reference anatomy retention")
        elif family == "photometric":
            original = images[index["original"]]
            if name.startswith("brightness"):
                factor = .7 if name == "brightness070" else 1.3
                current = np.clip(np.rint(original.astype(np.float32)*factor), 0, 255).astype(np.uint8)
                row["brightness_factor"] = factor
            else:
                current = np.asarray(Image.fromarray(original).filter(ImageFilter.GaussianBlur(radius=2.0)), dtype=np.uint8)
                row["gaussian_radius_256_pixels"] = 2.0
            target = targets_by_name["original"].copy()
        elif family == "padding":
            raster, offset = resized_raster(image, box)
            original_canvas = _canvas(raster, offset, 0)
            if not np.array_equal(original_canvas, images[index[anchor]]):
                raise ValueError("Padding anchor raster does not exactly reproduce canonical letterbox")
            fill = 127 if name.endswith("gray127") else 0
            new_offset = offset if name.endswith("gray127") else (0, 0)
            row.update(resized_raster_shape=list(raster.shape), resized_raster_sha256=_pixel_hash(raster),
                       original_offset_xy=list(offset), presented_offset_xy=list(new_offset), padding_fill=fill,
                       unchanged_anatomy_raster_verified=True)
            if (name.endswith("gray127") and raster.shape == (IMAGE_SIZE, IMAGE_SIZE)) or (name.endswith("topleft") and offset == (0, 0)):
                row.update(status="construction_ineligible", reason="no_padding_to_modify" if name.endswith("gray127") else "no_padding_offset_to_move")
                current = target = None
            else:
                current = _canvas(raster, new_offset, fill)
                target = targets_by_name[anchor].copy()
        else:
            raise ValueError("Unknown fixed challenge variant")
        row["native_crop_box_xyxy"] = list(box)
        if current is not None:
            images[i] = current
            targets[i] = target
            valid[i] = True
            row.update(pixel_sha256=_pixel_hash(current), effective_transform=not np.array_equal(current, images[index[anchor]]) if name != "original" else False)
        else:
            row.update(pixel_sha256=None, effective_transform=False)
        records.append(row)
    for outer, inner in NESTED_PAIRS:
        a, b = boxes[outer], boxes[inner]
        if not (a[0] <= b[0] < b[2] <= a[2] and a[1] <= b[1] < b[3] <= a[3]):
            raise ValueError("Frozen nested crop containment failed")
        if (targets[index[inner]] > targets[index[outer]] + 1e-6).any():
            raise ValueError("Native reference target violates containment monotonicity")
    if not np.isfinite(targets[valid]).all() or not np.isnan(targets[~valid]).all():
        raise ValueError("Invalid target/validity partition")
    return images, targets, valid, records

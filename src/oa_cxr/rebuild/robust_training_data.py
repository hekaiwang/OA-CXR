"""Frozen source-native fit augmentation for a new OA-CXR candidate.

Targets remain native annotation retention. Photometry changes geometric inputs,
not target geometry; that does not imply preserved clinical assessability.
The 13 slot identities belong to this protocol, not the old 13 crop definitions.
"""
from __future__ import annotations

import hashlib
import math
import re

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from oa_cxr.io import stable_hash
from oa_cxr.rebuild.data import DATA_PROTOCOL as ORIGINAL_DATA_PROTOCOL, FINDINGS, retention_targets

VERSION = "oa-cxr-source-native-robust-fit-v1"
SEED = 17
SLOTS = 13
DATA_PROTOCOL = {
    **ORIGINAL_DATA_PROTOCOL,
    "version": VERSION,
    "image_transform": "fit: native random crop then optional fixed nuisance and preserve-aspect letterbox256; dev: original sealed 13 variants unchanged",
    "variant_scope": "fit slots 0..12 are new source-specific recipes; dev slots retain original definitions; never join by slot alone across feature-protocol namespaces",
    "augmentation": {
        "seed": 17, "variants_per_source": 13, "original_slot": 0,
        "single_edge_slots": [1, 2, 3, 4, 5, 6], "adjacent_edge_slots": [7, 8, 9, 10],
        "opposed_edge_slots": [11, 12], "fraction_range": [.02, .28],
        "per_axis_total_fraction_cap": .4,
        "sampling": "SHA256-derived 53-bit uniform per source_id, image SHA, seed, slot and named parameter; fixed before new scores",
        "rounding": "floor removed pixels; minimum one pixel for active edges; axis>=32; both capped continuous fractions and realized pixel limits verified",
        "opposed_fraction_rule": "independent side fractions; scale both proportionally only when sum exceeds0.4",
        "nuisance_slots": {"3": "brightness0.8 or1.2 chosen by hash", "7": "GaussianBlur radius1.0 in rendered content pixels",
            "11": "hash chooses centered padding fill63 OR zero padding at a hash-selected corner"},
        "nuisance_order": "native crop; bilinear resize content; content-only brightness/blur; pad without further resizing",
        "mask_transform": "same integer native crop, nearest content resize, identical placement, always zero mask padding",
        "target_transform": "native annotation retention_targets evaluated on integer crop before any rendering",
        "non_nuisance_slots": [0, 1, 2, 4, 5, 6, 8, 9, 10, 12],
        "clinical_assessability_invariance_claimed": False},
}
PROTOCOL_SHA256 = stable_hash(DATA_PROTOCOL)


def uniform(source_id, source_sha256, slot, parameter):
    if not isinstance(source_id, str) or not source_id or not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("explicit source identity and image SHA256 required")
    if type(slot) is not int or not 0 <= slot < 13 or not isinstance(parameter, str) or not parameter:
        raise ValueError("fixed slot and named random parameter required")
    digest = stable_hash([VERSION, SEED, source_id, source_sha256, slot, parameter])
    return (int(digest[:16], 16) >> 11) / (2 ** 53)


def _choice(values, source_id, source_sha256, slot, parameter):
    return values[int(uniform(source_id, source_sha256, slot, parameter) * len(values))]


def recipes(size, source_id, source_sha256):
    width, height = size
    if any(type(x) is not int or x < 32 for x in size):
        raise ValueError("native image dimensions must be integer and at least32")
    # Validate identity even before the unmodified original slot.
    uniform(source_id, source_sha256, 0, "identity_check")
    result = []
    for slot in range(13):
        if slot == 0: edges, family = (), "original"
        elif slot <= 4: edges, family = (("top", "bottom", "left", "right")[slot - 1],), "single_edge"
        elif slot <= 6:
            edges, family = (_choice(("top", "bottom", "left", "right"), source_id, source_sha256, slot, "edge"),), "single_edge"
        elif slot <= 10:
            edges, family = (("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right"))[slot - 7], "adjacent_edges"
        else: edges, family = (("top", "bottom") if slot == 11 else ("left", "right")), "opposed_edges"
        fractions = {edge: .02 + .26 * uniform(source_id, source_sha256, slot, edge) for edge in edges}
        if family == "opposed_edges" and sum(fractions.values()) > .4:
            factor = .4 / sum(fractions.values()); fractions = {edge: value * factor for edge, value in fractions.items()}
        removed = {edge: max(1, math.floor(fractions[edge] * (height if edge in ("top", "bottom") else width))) for edge in edges}
        box = (removed.get("left", 0), removed.get("top", 0), width - removed.get("right", 0), height - removed.get("bottom", 0))
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError("invalid constructed crop")
        for axis, dimension in ((("top", "bottom"), height), (("left", "right"), width)):
            if sum(removed.get(k, 0) for k in axis) > math.floor(.4 * dimension):
                raise ValueError("realized axis removal exceeds fixed cap")
            if any(removed.get(k, 0) > math.floor(.28 * dimension) for k in axis):
                raise ValueError("realized edge removal exceeds fixed cap")
        nuisance = {"kind": "none", "brightness": 1.0, "blur_radius": 0.0, "padding_fill": 0, "placement": "center"}
        if slot == 3:
            nuisance.update(kind="brightness", brightness=_choice((.8, 1.2), source_id, source_sha256, slot, "brightness"))
        elif slot == 7:
            nuisance.update(kind="blur", blur_radius=1.0)
        elif slot == 11:
            if uniform(source_id, source_sha256, slot, "padding_mode") < .5:
                nuisance.update(kind="padding_fill", padding_fill=63)
            else:
                nuisance.update(kind="padding_offset", placement=_choice(("top_left", "top_right", "bottom_left", "bottom_right"),
                    source_id, source_sha256, slot, "corner"))
        result.append({"slot": slot, "family": family, "edges": list(edges), "sampled_fractions": fractions,
            "removed_pixels": removed, "box": list(box), "nuisance": nuisance})
    return result


def render(image, masks, recipe):
    if image.mode != "L": image = image.convert("L")
    if masks.dtype != np.bool_ or masks.shape != (2, image.height, image.width):
        raise ValueError("native boolean masks must align with source image")
    box = recipe["box"]
    if len(box) != 4 or any(type(v) is not int for v in box) or not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height):
        raise ValueError("integer nonempty native crop required")
    cropped = image.crop(box); scale = 256 / max(cropped.size)
    nw, nh = max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))
    nuisance = recipe["nuisance"]
    placement = nuisance["placement"]
    if placement == "center": position = ((256 - nw) // 2, (256 - nh) // 2)
    elif placement in ("top_left", "top_right", "bottom_left", "bottom_right"):
        position = (256 - nw if placement.endswith("right") else 0, 256 - nh if placement.startswith("bottom") else 0)
    else: raise ValueError("unknown content placement")
    content = cropped.resize((nw, nh), Image.Resampling.BILINEAR)
    if nuisance["brightness"] != 1: content = ImageEnhance.Brightness(content).enhance(nuisance["brightness"])
    if nuisance["blur_radius"]: content = content.filter(ImageFilter.GaussianBlur(nuisance["blur_radius"]))
    canvas = Image.new("L", (256, 256), nuisance["padding_fill"]); canvas.paste(content, position)
    rendered_masks = []
    for mask in masks:
        foreground = Image.fromarray(mask.astype(np.uint8) * 255).crop(box).resize((nw, nh), Image.Resampling.NEAREST)
        target = Image.new("L", (256, 256), 0); target.paste(foreground, position)
        rendered_masks.append(np.asarray(target) > 0)
    x, y = np.asarray(canvas, dtype=np.uint8), np.stack(rendered_masks)
    return x, y, {"resized_content_size": [nw, nh], "content_position": list(position),
                 "has_padding": nw < 256 or nh < 256, "mask_padding_fill": 0}


def render_source(image, masks, source_id, source_sha256):
    plan = recipes(image.size, source_id, source_sha256)
    targets = retention_targets(masks, [item["box"] for item in plan])
    views = [render(image, masks, item) for item in plan]
    images = np.stack([x for x, _, _ in views])
    presented = np.stack([m for _, m, _ in views])
    packed = np.packbits(presented.reshape(13, 2, -1), axis=-1, bitorder="big")
    for index, (_, _, placement) in enumerate(views):
        plan[index].update(placement)
        plan[index]["image_pixels_sha256"] = hashlib.sha256(images[index].tobytes()).hexdigest()
        plan[index]["packed_masks_sha256"] = hashlib.sha256(packed[index].tobytes()).hexdigest()
        plan[index]["retention_targets"] = targets[index].tolist()
    return images, packed, targets, plan

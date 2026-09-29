"""Parse SCR hand-drawn PFS lung contours without reading images or outcomes.

The author's ``points`` files use a 1024-square coordinate system. These are
not the resampled ``landmarks`` files. ``* fixed`` blocks contain anatomical
anchor points and are deliberately excluded. Observer identity is unspecified.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, __version__ as PILLOW_VERSION


SCR_PROTOCOL = {
    "version": "scr-pfs-lung-polygons-v2",
    "source": "https://doi.org/10.5281/zenodo.7056076",
    "paper": "https://doi.org/10.1016/j.media.2005.02.002",
    "input": "points/*.pfs hand-drawn contours; not landmarks",
    "coordinate_canvas": [1024, 1024],
    "coordinate_order": "x,y; image origin at top-left",
    "allowed_vertex_extent": "closed continuous canvas [0,1024] on both axes",
    "labels": ["right lung", "left lung"],
    "required_line_mode": "ClosedContour",
    "fixed_blocks": "excluded; never substituted for missing contours",
    "observer": "unspecified in the released PFS; not asserted to be observer1",
    "target_sizes": [[1024, 1024], [2048, 2048]],
    "transform": "multiply original floating vertices by target_size/1024, then floor",
    "rasterizer": "Pillow ImageDraw.polygon fill=1 outline=1 on mode L; inclusive boundary pixels",
    "final_mask": "bool [height,width]; separately rasterize anatomical right and left lung",
    "canonicalization": "remove only exact consecutive duplicate vertices and explicit duplicate closure; retain original coordinates and removed zero-based indices; never repair non-adjacent repeats or crossings",
    "validation": "unique simple nonzero polygon per lung; finite in-range vertices; no mask overlap",
    "target_semantics": "manually delineated visible lung fields, not complete anatomical lung or clinical O-minus",
}
SCR_PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(SCR_PROTOCOL, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
).hexdigest()

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_POINT = re.compile(rf"\{{\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\}}\s*,?")
_ATTRIBUTE = re.compile(r"\[([A-Za-z][A-Za-z0-9]*)=([^\[\]\r\n]*)\]")
_ALLOWED_LABELS = frozenset(
    ["right lung", "left lung", "heart", "right clavicle", "left clavicle"]
    + [f"{name} fixed" for name in ["right lung", "left lung", "heart", "right clavicle", "left clavicle"]]
)
_EPS = 1e-8


@dataclass(frozen=True)
class SCRContours:
    """Immutable native-coordinate contours bound to exact annotation bytes."""

    annotation_sha256: str
    right_lung: tuple[tuple[float, float], ...]
    left_lung: tuple[tuple[float, float], ...]
    excluded_labels: tuple[str, ...]
    right_lung_original: tuple[tuple[float, float], ...]
    left_lung_original: tuple[tuple[float, float], ...]
    right_lung_removed_indices: tuple[int, ...]
    left_lung_removed_indices: tuple[int, ...]
    protocol_sha256: str = SCR_PROTOCOL_SHA256


def _sha(value: str, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")
    return value


def _cross(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p) -> bool:
    return (
        abs(_cross(a, b, p)) <= _EPS
        and min(a[0], b[0]) - _EPS <= p[0] <= max(a[0], b[0]) + _EPS
        and min(a[1], b[1]) - _EPS <= p[1] <= max(a[1], b[1]) + _EPS
    )


def _intersects(a, b, c, d) -> bool:
    ab_c, ab_d = _cross(a, b, c), _cross(a, b, d)
    cd_a, cd_b = _cross(c, d, a), _cross(c, d, b)
    if ((ab_c > _EPS and ab_d < -_EPS) or (ab_c < -_EPS and ab_d > _EPS)) and (
        (cd_a > _EPS and cd_b < -_EPS) or (cd_a < -_EPS and cd_b > _EPS)
    ):
        return True
    return any((_on_segment(a, b, c), _on_segment(a, b, d), _on_segment(c, d, a), _on_segment(c, d, b)))


def _validate_polygon(points, label: str) -> tuple[tuple[float, float], ...]:
    result = tuple(tuple(float(value) for value in point) for point in points)
    if any(len(point) != 2 for point in result):
        raise ValueError(f"{label}: coordinates must be x,y pairs")
    if not all(math.isfinite(value) and 0 <= value <= 1024 for point in result for value in point):
        raise ValueError(f"{label}: non-finite or out-of-range native coordinates")
    if len(result) > 1 and result[0] == result[-1]:
        result = result[:-1]  # A single explicit closing vertex is equivalent.
    if not 3 <= len(result) <= 2048 or len(set(result)) != len(result):
        raise ValueError(f"{label}: need 3..2048 unique vertices")
    area2 = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(result, result[1:] + result[:1]))
    if abs(area2) <= _EPS:
        raise ValueError(f"{label}: zero-area polygon")
    count = len(result)
    for i in range(count):
        a, b, c = result[i - 1], result[i], result[(i + 1) % count]
        if abs(_cross(a, b, c)) <= _EPS and (
            (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1]) < 0
        ):
            raise ValueError(f"{label}: adjacent edges backtrack")
        for j in range(i + 1, count):
            if j == i + 1 or (i == 0 and j == count - 1):
                continue
            if _intersects(result[i], result[(i + 1) % count], result[j], result[(j + 1) % count]):
                raise ValueError(f"{label}: self-intersecting or self-touching contour")
    return result


def _canonicalize(points):
    """Remove zero-length edges only; report indices in original point order."""
    original = tuple(tuple(float(value) for value in point) for point in points)
    kept, indices, removed = [], [], []
    for index, point in enumerate(original):
        if kept and point == kept[-1]:
            removed.append(index)
        else:
            kept.append(point)
            indices.append(index)
    if len(kept) > 1 and kept[0] == kept[-1]:
        kept.pop()
        removed.append(indices.pop())
    return original, tuple(kept), tuple(sorted(removed))


def parse_pfs(payload: bytes | str, *, expected_sha256: str | None = None) -> SCRContours:
    """Read strict PFS blocks and select exact closed lung labels only.

    Prefer bytes so the hash covers the downloaded file, including CRLF. A str
    is hashed as UTF-8 as supplied. Auxiliary fixed blocks are excluded even if
    they happen to contain a ClosedContour attribute.
    """
    if isinstance(payload, str):
        raw = payload.encode("utf-8")
    elif isinstance(payload, bytes):
        raw = payload
    else:
        raise TypeError("payload must be bytes or str")
    if not raw or len(raw) > 2 * 1024 * 1024:
        raise ValueError("Empty or unexpectedly large PFS")
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != _sha(expected_sha256, "expected_sha256"):
        raise ValueError("Annotation SHA256 mismatch")
    text = raw.decode("utf-8-sig", errors="strict")
    blocks = {}
    attributes = None
    points = []
    for number, source_line in enumerate(text.splitlines(), 1):
        line = source_line.strip()
        if not line or line.startswith(";"):
            continue
        if line == "{":
            if attributes is not None:
                raise ValueError(f"Line {number}: nested block")
            attributes, points = {}, []
        elif line in ("}", "},"):
            if attributes is None:
                raise ValueError(f"Line {number}: closing unopened block")
            label = attributes.get("Label")
            if label not in _ALLOWED_LABELS:
                raise ValueError(f"Line {number}: unknown/missing label {label!r}")
            if label in blocks:
                raise ValueError(f"Duplicate annotation block: {label}")
            blocks[label] = (attributes, tuple(points))
            attributes = None
        else:
            if attributes is None:
                raise ValueError(f"Line {number}: data outside a block")
            attribute, point = _ATTRIBUTE.fullmatch(line), _POINT.fullmatch(line)
            if attribute:
                key, value = attribute.groups()
                if key in attributes or points:
                    raise ValueError(f"Line {number}: duplicate or late attribute")
                attributes[key] = value.strip()
            elif point:
                points.append(tuple(float(value) for value in point.groups()))
            else:
                raise ValueError(f"Line {number}: malformed PFS syntax")
    if attributes is not None:
        raise ValueError("Unclosed PFS block")
    selected, originals, removed = {}, {}, {}
    for label in ("right lung", "left lung"):
        if label not in blocks:
            raise ValueError(f"Missing {label} ClosedContour; fixed anchors cannot substitute")
        attr, vertices = blocks[label]
        if attr.get("LineMode") != "ClosedContour":
            raise ValueError(f"{label}: LineMode must be ClosedContour")
        originals[label], canonical, removed[label] = _canonicalize(vertices)
        selected[label] = _validate_polygon(canonical, label)
    return SCRContours(
        digest, selected["right lung"], selected["left lung"],
        tuple(sorted(set(blocks) - set(selected))),
        originals["right lung"], originals["left lung"],
        removed["right lung"], removed["left lung"],
    )


def load_pfs(path: str | Path, *, expected_sha256: str | None = None) -> SCRContours:
    return parse_pfs(Path(path).read_bytes(), expected_sha256=expected_sha256)


def rasterize_lungs(
    contours: SCRContours, *, target_size: tuple[int, int] = (2048, 2048), source_image_sha256: str,
) -> dict:
    """Rasterize original floating vertices after explicit 1x or 2x scaling.

    Returns immutable masks with anatomical names from the file, plus provenance.
    Do not resize the already rasterized 1024 mask to make the 2048 mask: that is
    a different boundary convention and is not this protocol.
    """
    if not isinstance(contours, SCRContours) or contours.protocol_sha256 != SCR_PROTOCOL_SHA256:
        raise ValueError("Expected SCRContours from the fixed PFS protocol")
    _sha(contours.annotation_sha256, "annotation_sha256")
    _sha(source_image_sha256, "source_image_sha256")
    if not isinstance(target_size, (tuple, list)) or len(target_size) != 2 or any(type(v) is not int for v in target_size):
        raise ValueError("target_size must be integer (width,height)")
    size = tuple(target_size)
    if size not in ((1024, 1024), (2048, 2048)):
        raise ValueError("Only explicit native 1024 or 2x 2048 canvas is supported")
    masks, audits = {}, {}
    scale = size[0] / 1024
    for name in ("right_lung", "left_lung"):
        points = _validate_polygon(getattr(contours, name), name)
        original, canonical, removed = _canonicalize(getattr(contours, f"{name}_original"))
        if canonical != points or removed != getattr(contours, f"{name}_removed_indices"):
            raise ValueError(f"{name}: original/canonical vertex audit mismatch")
        vertices = [(math.floor(x * scale), math.floor(y * scale)) for x, y in points]
        canvas = Image.new("L", size, 0)
        ImageDraw.Draw(canvas).polygon(vertices, fill=1, outline=1)
        mask = np.asarray(canvas, dtype=bool)
        if not mask.any():
            raise ValueError(f"{name}: empty rasterized region")
        # bytes-backed arrays cannot be made writable by a caller.
        mask = np.frombuffer(mask.tobytes(order="C"), dtype=np.bool_).reshape(size[1], size[0])
        masks[name] = mask
        audits[name] = {
            "label": name.replace("_", " "), "vertices": len(points),
            "polygon_sha256": hashlib.sha256(json.dumps(points, separators=(",", ":")).encode()).hexdigest(),
            "mask_bool_c_order_sha256": hashlib.sha256(mask.tobytes(order="C")).hexdigest(),
            "foreground_pixels": int(mask.sum()),
            "original_vertices": [list(point) for point in original],
            "canonical_vertices": [list(point) for point in points],
            "removed_original_zero_based_indices": list(removed),
        }
    if np.any(masks["right_lung"] & masks["left_lung"]):
        raise ValueError("Left/right lung masks overlap; requires annotation audit")
    return {
        **masks,
        "provenance": {
            "protocol_sha256": SCR_PROTOCOL_SHA256,
            "annotation_source": SCR_PROTOCOL["source"],
            "annotation_file_sha256": contours.annotation_sha256,
            "source_image_sha256": source_image_sha256,
            "observer": "unspecified",
            "native_coordinate_size": [1024, 1024],
            "target_mask_size": list(size),
            "vertex_scale_xy": [scale, scale],
            "rasterizer": "Pillow.ImageDraw.polygon; scaled coordinates floor; inclusive boundary",
            "pillow_version": PILLOW_VERSION,
            "excluded_labels": list(contours.excluded_labels),
            "clinical_support_label": False,
            "masks": audits,
        },
    }

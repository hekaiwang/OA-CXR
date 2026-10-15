"""Frozen, image-only official encoders for the new coverage task adapters.

All three adapters preserve the entire already-presented field of view. Their
224px resize is a task adaptation, not an assertion of original task replication.
Torch and upstream libraries are imported only by ``load_encoder``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

METHODS = ("biomedclip", "chexzero", "dinov2")
DIMENSIONS = {"biomedclip": 512, "chexzero": 512, "dinov2": 768}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocessing_policy(method):
    if method not in METHODS:
        raise ValueError("unsupported foundation encoder")
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if method == "dinov2" else (CLIP_MEAN, CLIP_STD)
    return {"input": "accepted presented uint8 grayscale 256x256 cache; no masks or targets",
            "resize": [224, 224], "interpolation": "Pillow BICUBIC on RGB uint8",
            "crop": None, "rgb": "three identical grayscale channels", "scale": 1 / 255,
            "mean": list(mean), "std": list(std), "augmentation": None,
            "readout": "LayerNorm CLS token" if method == "dinov2" else "L2-normalized projected image embedding",
            "task_adaptation": "full presented field of view retained; no new crop or padding"}


def presented_rgb(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.shape != (256, 256):
        raise ValueError("presented image must be uint8 256x256")
    return np.asarray(Image.fromarray(image).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)).copy()


def preprocess_image(image, method):
    policy = preprocessing_policy(method)
    rgb = presented_rgb(image)
    tensor = rgb.astype(np.float32).transpose(2, 0, 1) / np.float32(255)
    mean = np.asarray(policy["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(policy["std"], dtype=np.float32)[:, None, None]
    return np.ascontiguousarray((tensor - mean) / std, dtype=np.float32)


def feature_names(method, findings):
    if method not in METHODS or len(findings) != 3 or len(set(findings)) != 3:
        raise ValueError("encoder and exactly three ordered findings required")
    return [f"{method}_image_{i:04d}" for i in range(DIMENSIONS[method])] + [f"finding_{finding}" for finding in findings]


def query_features(embeddings):
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2 or embeddings.dtype != np.float32 or not np.isfinite(embeddings).all():
        raise ValueError("finite float32 image embeddings required")
    visual = np.broadcast_to(embeddings[:, None, :], (len(embeddings), 3, embeddings.shape[1]))
    identity = np.broadcast_to(np.eye(3, dtype=np.float32), (len(embeddings), 3, 3))
    return np.concatenate((visual, identity), axis=-1).reshape(-1, embeddings.shape[1] + 3)


def verify_assets(manifest_path, expected_sha256, method):
    manifest_path = Path(manifest_path).resolve(strict=True)
    if not re.fullmatch("[a-f0-9]{64}", expected_sha256 or "") or sha256(manifest_path) != expected_sha256:
        raise ValueError("asset manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (method not in METHODS or manifest.get("schema") != "oa-cxr-foundation-assets-v1" or
            manifest.get("method") != method or manifest.get("status") != "completed"):
        raise ValueError("completed method-specific foundation asset manifest required")
    root = manifest_path.parent
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("bound code and weights required")
    paths = {}
    for item in files:
        relative = item.get("path")
        if not isinstance(relative, str) or Path(relative).is_absolute() or relative in paths:
            raise ValueError("unique relative asset path required")
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("asset path escapes manifest root")
        if not re.fullmatch("[a-f0-9]{64}", item.get("sha256", "")) or sha256(path) != item["sha256"]:
            raise ValueError("asset file SHA256 mismatch: " + relative)
        paths[relative] = path
    loader = manifest.get("loader", {})
    expected_kind = {"biomedclip": "biomedclip_visual", "chexzero": "chexzero_visual", "dinov2": "dinov2_vitb14"}[method]
    if loader.get("kind") != expected_kind or loader.get("weights") not in paths:
        raise ValueError("explicit bound official model weights and loader required")
    if method == "biomedclip" and loader.get("config") not in paths:
        raise ValueError("bound BiomedCLIP model config required")
    if method in ("chexzero", "dinov2"):
        relative = loader.get("code_root")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("bound upstream code root required")
        code_root = (root / relative).resolve(strict=True)
        if not code_root.is_relative_to(root) or not code_root.is_dir():
            raise ValueError("invalid code root")
        # Code executed by a local hub or local Python import must be bound too.
        code = [p for p in code_root.rglob("*.py") if "__pycache__" not in p.parts]
        if not code or any(p.resolve() not in set(paths.values()) for p in code):
            raise ValueError("all upstream Python source files must be hash bound")
    return {"manifest": manifest, "path": manifest_path, "root": root, "paths": paths,
            "sha256": expected_sha256, "method": method}


def _state_dict(path):
    import torch
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError("official checkpoint must contain a state dictionary")
    if "state_dict" in value:
        value = value["state_dict"]
    if not value or not isinstance(value, dict):
        raise ValueError("empty or invalid checkpoint state")
    if all(name.startswith("module.") for name in value):
        value = {name[7:]: tensor for name, tensor in value.items()}
    return value


def load_encoder(assets):
    """Load verified local weights without network access or random fallbacks."""
    import torch
    method = assets["method"]
    loader, paths = assets["manifest"]["loader"], assets["paths"]
    weights = _state_dict(paths[loader["weights"]])
    if method == "biomedclip":
        from open_clip.model import _build_vision_tower
        config = json.loads(paths[loader["config"]].read_text(encoding="utf-8"))
        if config.get("preprocess_cfg") != {"mean": list(CLIP_MEAN), "std": list(CLIP_STD)}:
            raise ValueError("BiomedCLIP official normalization changed")
        cfg = config["model_cfg"]
        if cfg["embed_dim"] != 512 or cfg["vision_cfg"].get("timm_model_pretrained") is not False:
            raise ValueError("expected official locally initialized BiomedCLIP ViT-B encoder")
        model = _build_vision_tower(cfg["embed_dim"], cfg["vision_cfg"], quick_gelu=False, cast_dtype=None)
        visual = {key[7:]: value for key, value in weights.items() if key.startswith("visual.")}
        if not visual:
            raise ValueError("BiomedCLIP checkpoint is missing its image tower")
        model.load_state_dict(visual, strict=True)
    elif method == "chexzero":
        code = assets["root"] / loader["code_root"] / "model.py"
        spec = importlib.util.spec_from_file_location("oa_cxr_official_chexzero_model", code)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.build_model(weights).visual
        if getattr(model, "output_dim", None) != 512 or getattr(model, "input_resolution", None) != 224:
            raise ValueError("expected official CheXzero 224px 512-dimensional image encoder")
    elif method == "dinov2":
        model = torch.hub.load(str(assets["root"] / loader["code_root"]), "dinov2_vitb14",
                               source="local", pretrained=False)
        model.load_state_dict(weights, strict=True)
        if model.num_features != 768:
            raise ValueError("expected DINOv2 ViT-B/14 CLS dimension 768")
    else:
        raise ValueError("unsupported method")
    model.float().eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()) or model.training:
        raise ValueError("encoder must be entirely frozen and in eval mode")
    return model


def encode_batch(model, tensor, method):
    """Output float32 feature vectors; no learned head, labels or masks involved."""
    import torch
    if method not in METHODS or tensor.ndim != 4 or tuple(tensor.shape[1:]) != (3, 224, 224):
        raise ValueError("NCHW 224px RGB encoder input required")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("encoder became trainable or left eval mode")
    value = model(tensor).float()
    if value.shape != (len(tensor), DIMENSIONS[method]) or not bool(torch.isfinite(value).all()):
        raise ValueError("invalid encoder output")
    if method in ("biomedclip", "chexzero"):
        norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
        if bool((norm <= 0).any()):
            raise ValueError("zero CLIP embedding cannot be normalized")
        value = value / norm
    return value

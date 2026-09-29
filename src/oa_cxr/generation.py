"""Frozen reporting with resumable, input-addressed caches and processor audit."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import platform
import random
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__
from .io import append_jsonl, exclusive_writer, read_jsonl, sha256_file, stable_hash, write_json

PIPELINE_VERSION = "reporting-v1"
ALLOWED_METADATA = {"indication", "technique", "comparison"}


def validate_config(config: dict) -> dict:
    config = copy.deepcopy(config)
    required = {"backend", "model_source", "model_revision", "processor_revision", "generation", "seed", "dtype", "device"}
    if missing := required - config.keys():
        raise ValueError(f"Missing generation configuration: {sorted(missing)}")
    if config["backend"] not in {"mock", "maira"}:
        raise ValueError("backend must be mock or maira")
    if config["backend"] == "maira":
        for key in ("model_revision", "processor_revision"):
            if not re.fullmatch(r"[a-f0-9]{40}", config[key]):
                raise ValueError(f"{key} must be a full immutable HF commit SHA")
        if config["dtype"] not in {"float32", "bfloat16"}:
            raise ValueError("Only official float32 and explicit bfloat16 are supported")
    generation = config["generation"]
    allowed = {"do_sample", "max_new_tokens", "use_cache", "num_beams", "temperature", "top_p"}
    if generation.keys() - allowed:
        raise ValueError("Unsupported generation option")
    if generation.get("max_new_tokens", 0) <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not generation.get("do_sample", False) and ({"temperature", "top_p"} & generation.keys()):
        raise ValueError("temperature/top_p require do_sample=true")
    return config


def input_identity(row: dict, config: dict) -> dict:
    """Only current inputs + permitted clinical sections enter this identity."""
    paths = row["local_image_paths"]
    if "frontal" not in paths or paths.keys() - {"frontal", "lateral"}:
        raise ValueError("Current frontal and optional lateral images only")
    metadata = row.get("allowed_input_metadata", {})
    if metadata.keys() - ALLOWED_METADATA:
        raise ValueError("Input metadata contains unapproved fields (Findings/Impression are prohibited)")
    if any(value is not None and not isinstance(value, str) for value in metadata.values()):
        raise ValueError("Clinical metadata values must be strings or null")
    hashes = {view: sha256_file(path) for view, path in paths.items()}
    for view, expected in row.get("image_sha256", {}).items():
        if expected and expected != hashes.get(view):
            raise ValueError(f"Image hash mismatch for {view}; rebuild manifest after changing files")
    return {
        "subject_id": str(row["subject_id"]), "study_id": str(row["study_id"]),
        "input_variant_id": row.get("input_variant_id", "original"),
        "partition": row["partition"], "official_split": row.get("official_split"),
        "image_sha256": hashes, "allowed_input_metadata": metadata,
        "view_positions": row.get("view_positions", {}),
        "preprocessing_version": row.get("preprocessing_version", "unspecified"),
        "pipeline_version": PIPELINE_VERSION, "package_version": __version__,
        "implementation_sha256": sha256_file(__file__),
        "runtime_packages": environment_info()["packages"],
        "config": config,
    }


def cache_key(row: dict, config: dict) -> str:
    return stable_hash(input_identity(row, validate_config(config)))


@lru_cache(maxsize=1)
def environment_info() -> dict:
    packages = {}
    for package in ("oa-cxr", "numpy", "Pillow", "torch", "torchvision", "transformers", "accelerate", "huggingface-hub"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def verify_audit_artifacts(record: dict) -> None:
    audit = record.get("processor_audit", {})
    paths, hashes = audit.get("presented_image_paths", {}), audit.get("presented_image_sha256", {})
    if not paths or paths.keys() != hashes.keys():
        raise ValueError("Missing processor image audit; regenerate this record")
    for view, path in paths.items():
        if not Path(path).is_file() or sha256_file(path) != hashes[view]:
            raise ValueError("Processor audit hash missing or changed; use --retry-errors to regenerate artifacts")
    if record.get("backend") == "maira":
        path = audit.get("tensor_path")
        if not path or not Path(path).is_file() or sha256_file(path) != audit.get("tensor_sha256"):
            raise ValueError("Processor tensor audit missing or changed; use --retry-errors to regenerate")


def processor_protocol(record: dict) -> dict:
    config = record.get("input_identity", {}).get("config", {})
    return {"processor_revision": record.get("processor_revision"), "dtype": config.get("dtype"),
            "image_processor_config_hash": stable_hash(record.get("processor_audit", {}).get("image_processor", {})),
            "view_order": list(record.get("processor_audit", {}).get("presented_image_paths", {}))}


def verify_processor_review(record: dict, review: dict | None) -> None:
    verify_audit_artifacts(record)
    if record.get("backend") == "mock" and record.get("synthetic"):
        return
    protocol = processor_protocol(record)
    if not review or not any(
        item.get("protocol") == protocol and item.get("reviewer_id") and item.get("evidence_record_ids")
        for item in review.get("protocols", [])
    ):
        raise ValueError("Real processor images need a documented view/normalization review; run record_processor_review.py on smoke/pilot audit images")


class MockBackend:
    """Deliberately fixed synthetic fixture, never a radiology model or prediction."""

    def __init__(self, config: dict):
        self.config = config

    def generate(self, row: dict, audit_dir: Path) -> dict:
        if not row.get("synthetic", False):
            raise ValueError("Mock backend accepts explicitly synthetic manifests only")
        audit_dir.mkdir(parents=True, exist_ok=True)
        images = {}
        for view, path in row["local_image_paths"].items():
            dest = audit_dir / f"{view}.png"
            with Image.open(path) as im:
                im.convert("RGB").save(dest)
            images[view] = str(dest.resolve())
        return {
            "generated_findings": "No pleural effusion or pneumothorax. No focal consolidation. Cardiac silhouette is enlarged.",
            "processor_audit": {"status": "synthetic_identity", "presented_image_paths": images,
                                "presented_image_sha256": {v: sha256_file(p) for v,p in images.items()}},
            "peak_memory_bytes": 0, "peak_reserved_memory_bytes": 0,
            "generation_calls": 0, "synthetic": True,
        }


class MairaBackend:
    def __init__(self, config: dict):
        # Heavy imports occur only when explicitly requesting real inference.
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoProcessor

        if transformers.__version__ != "4.51.3":
            raise RuntimeError("MAIRA environment must use transformers==4.51.3; see requirements/maira-cu121.txt")
        if not config["device"].startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("MAIRA requires an available CUDA device. CPU verification uses the synthetic demo.")
        self.torch, self.config = torch, config
        self.device = torch.device(config["device"])
        common = {"trust_remote_code": True, "local_files_only": config.get("local_files_only", True)}
        self.processor = AutoProcessor.from_pretrained(config["model_source"], revision=config["processor_revision"], **common)
        dtype = getattr(torch, config["dtype"])
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This device does not support bfloat16")
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_source"], revision=config["model_revision"], torch_dtype=dtype,
            low_cpu_mem_usage=True, **common).eval().to(self.device)

    def _audit(self, inputs, row: dict, directory: Path) -> dict:
        """Save actual normalized tensors; export a PNG only when layout is verified."""
        directory.mkdir(parents=True, exist_ok=True)
        tensor = inputs.get("pixel_values")
        if tensor is None:
            raise ValueError("Processor has no pixel_values; implement an audited adapter before this experiment")
        array = tensor.detach().float().cpu().numpy()
        tensor_path = directory / "pixel_values.npy"
        np.save(tensor_path, array, allow_pickle=False)
        ip = getattr(self.processor, "image_processor", None)
        if ip is None:
            raise ValueError("Cannot audit missing image_processor")
        audit = {"tensor_path": str(tensor_path.resolve()), "tensor_sha256": sha256_file(tensor_path),
                 "shape": list(array.shape), "image_processor": ip.to_dict(), "presented_image_paths": {}}
        # Standard HF single-batch [N,C,H,W] or [1,N,C,H,W]. Do not guess tile/order semantics.
        if array.ndim == 5 and array.shape[0] == 1:
            array = array[0]
        views = [v for v in ("frontal", "lateral") if v in row["local_image_paths"]]
        if array.ndim != 4 or array.shape[0] != len(views) or array.shape[1] not in (1, 3):
            write_json(directory / "processor_audit.json", {**audit, "status": "unsupported_tensor_layout"})
            raise ValueError("Unexpected processor tensor layout; raw tensor saved, view mapping requires inspection")
        if not hasattr(ip, "do_normalize") or not hasattr(ip, "do_rescale"):
            raise ValueError("Unknown processor normalization; implement explicit tensor inversion")
        mean = np.asarray(ip.image_mean if ip.do_normalize else [0] * array.shape[1]).reshape(-1, 1, 1)
        std = np.asarray(ip.image_std if ip.do_normalize else [1] * array.shape[1]).reshape(-1, 1, 1)
        if mean.shape[0] != array.shape[1] or std.shape[0] != array.shape[1] or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
            raise ValueError("Invalid image normalization metadata")
        factor = getattr(ip, "rescale_factor", None) if ip.do_rescale else 1.0
        if factor is None or not np.isfinite(factor) or factor <= 0:
            raise ValueError("Unknown image rescale_factor")
        for view, pixels in zip(views, array):
            if ip.do_normalize:
                pixels = pixels * std + mean
            pixels = pixels / (factor * 255.0)
            if not np.all(np.isfinite(pixels)) or pixels.min() < -.02 or pixels.max() > 1.02:
                raise ValueError("Image tensor inversion produced out-of-range pixels; inspect processor before clipping")
            pixels = np.clip(pixels, 0, 1)
            pixels = np.moveaxis(np.rint(pixels * 255).astype(np.uint8), 0, -1)
            if pixels.shape[-1] == 1:
                pixels = pixels[..., 0]
            path = directory / f"{view}.png"
            Image.fromarray(pixels).save(path)
            audit["presented_image_paths"][view] = str(path.resolve())
        audit["presented_image_sha256"] = {v: sha256_file(p) for v,p in audit["presented_image_paths"].items()}
        audit["status"] = "tensor_exported_pending_visual_review"
        audit["view_order_assumption"] = "official current_frontal then current_lateral; verify on server smoke"
        write_json(directory / "processor_audit.json", audit)
        return audit

    def generate(self, row: dict, audit_dir: Path) -> dict:
        torch = self.torch
        seed = int(self.config["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        images = {}
        for view, path in row["local_image_paths"].items():
            with Image.open(path) as im:
                images[view] = im.convert("RGB")
        metadata = row.get("allowed_input_metadata", {})
        inputs = self.processor.format_and_preprocess_reporting_input(
            current_frontal=images["frontal"], current_lateral=images.get("lateral"),
            prior_frontal=None, prior_report=None,
            indication=metadata.get("indication"), technique=metadata.get("technique"),
            comparison=metadata.get("comparison"), get_grounding=False, return_tensors="pt")
        inputs = inputs.to(self.device)
        if self.config["dtype"] != "float32":
            for key, value in inputs.items():
                if hasattr(value, "is_floating_point") and value.is_floating_point():
                    inputs[key] = value.to(dtype=getattr(torch, self.config["dtype"]))
        audit = self._audit(inputs, row, audit_dir)
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, **self.config["generation"])
        torch.cuda.synchronize(self.device)
        prefix_len = inputs["input_ids"].shape[-1]
        raw_text = self.processor.decode(output[0, prefix_len:], skip_special_tokens=True).lstrip()
        findings = self.processor.convert_output_to_plaintext_or_grounded_sequence(raw_text)
        if not isinstance(findings, str) or not findings.strip():
            raise ValueError("Expected nonempty ungrounded Findings string")
        return {"generated_findings": findings, "decoded_completion": raw_text,
                "processor_audit": audit, "generation_calls": 1, "synthetic": False,
                "prompt_tokens": int(prefix_len), "completion_tokens": int(output.shape[-1] - prefix_len),
                "possibly_truncated": int(output.shape[-1] - prefix_len) >= self.config["generation"]["max_new_tokens"],
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                "peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved(self.device))}


def run_cache(rows: list[dict], config: dict, output: str | Path, *, audit_dir: str | Path,
              retry_errors: bool = False, allow_test: bool = False,
              shard_index: int = 0, num_shards: int = 1, backend=None) -> dict:
    config = validate_config(config)
    if not 0 <= shard_index < num_shards:
        raise ValueError("Require 0 <= shard_index < num_shards")
    output, audit_dir = Path(output), Path(audit_dir)
    subjects, identities = {}, set()
    for row in rows:
        patient = str(row["subject_id"])
        partition = row["partition"]
        if patient in subjects and subjects[patient] != partition:
            raise ValueError("Patient crosses partitions")
        subjects[patient] = partition
        identity = (str(row["study_id"]), row.get("input_variant_id", "original"))
        if identity in identities:
            raise ValueError(f"Duplicate study/variant in manifest: {identity}")
        identities.add(identity)
        if (partition == "test" or row.get("official_split") == "test") and not allow_test:
            raise ValueError("Test is sealed; --allow-test is required for a frozen final protocol")
    summary = {"success": 0, "failed": 0, "skipped": 0, "skipped_errors": 0,
               "sharded_out": 0, "backend": config["backend"]}
    with exclusive_writer(output):
        existing = read_jsonl(output) if output.exists() else []
        latest = {r["cache_key"]: r for r in existing}
        loaded = backend
        for row in rows:
            shard = int(hashlib.sha256(str(row["subject_id"]).encode()).hexdigest(), 16) % num_shards
            if shard != shard_index:
                summary["sharded_out"] += 1
                continue
            # Validate input hashes before inference. Unreadable files still receive a failure record.
            try:
                identity = input_identity(row, config)
                key = stable_hash(identity)
                input_error = None
            except (OSError, ValueError) as exc:
                key = stable_hash({"invalid_input": row, "config": config, "pipeline_version": PIPELINE_VERSION})
                identity, input_error = None, exc
            previous = latest.get(key)
            if previous and (previous.get("error_status") is None or not retry_errors):
                valid = True
                if previous.get("error_status") is None:
                    try:
                        verify_audit_artifacts(previous)
                    except (OSError, ValueError):
                        valid = False
                        if not retry_errors:
                            raise ValueError("Successful cache has missing/corrupt processor artifacts; rerun with --retry-errors")
                if valid:
                    summary["skipped"] += 1
                    if previous.get("error_status") is not None:
                        summary["skipped_errors"] += 1
                    continue
            base = {k: row.get(k) for k in ("subject_id", "study_id", "dicom_id", "partition", "official_split", "input_variant_id", "sampling_stratum", "dataset", "split_source")}
            base.update({"cache_key": key, "record_id": key, "backend": config["backend"],
                         "model_revision": config["model_revision"], "processor_revision": config["processor_revision"],
                         "decoding_config_hash": stable_hash(config["generation"]), "input_sha256": stable_hash(identity) if identity else None,
                         "input_identity": identity, "local_image_paths": row.get("local_image_paths"),
                         "error_status": None, "generated_findings": None, "synthetic": config["backend"] == "mock",
                         "attempt": int(previous.get("attempt", 0)) + 1 if previous else 1})
            start = time.perf_counter()
            stage = "input_validation"
            try:
                if input_error:
                    raise input_error
                if loaded is None:
                    stage = "backend_initialization"
                    loaded = MockBackend(config) if config["backend"] == "mock" else MairaBackend(config)
                stage = "generation"
                result = loaded.generate(row, audit_dir / key)
                if not isinstance(result.get("generated_findings"), str) or not result["generated_findings"].strip():
                    raise ValueError("Empty Findings returned")
                base.update(result)
                summary["success"] += 1
            except Exception as exc:
                # Deliberately omit provider URLs/credential-bearing exception text from patient cache.
                base["error_status"] = {"type": type(exc).__name__, "stage": stage,
                                        "message": "Generation/input validation failed; run single-study smoke to diagnose."}
                summary["failed"] += 1
            base["runtime_sec"] = time.perf_counter() - start
            append_jsonl(output, base)
            latest[key] = base
            if base["error_status"] and stage == "backend_initialization":
                summary["stopped_on_backend_error"] = True
                break
        write_json(str(output) + ".run.json", {"config": config, "environment": environment_info(), "summary": summary,
                   "shard_index": shard_index, "num_shards": num_shards,
                   "timing_note": "First uncached record includes model load. Peak allocated/reserved measures generation including resident weights."})
    return summary


def successful_records(rows: list[dict]) -> list[dict]:
    latest = {}
    for row in rows:
        key = row.get("cache_key", row.get("record_id"))
        if not key:
            raise ValueError("Missing cache key")
        latest[key] = row
    result = [r for r in latest.values() if r.get("error_status") is None and r.get("generated_findings")]
    modes = {r.get("backend") for r in result}
    if len(modes) > 1:
        raise ValueError("Do not mix synthetic and real generation caches")
    return result

"""Opt-in greedy MAIRA batching, isolated from the singleton sampling protocol.

The fixed MAIRA processor formats each study; its inherited LlavaProcessor then
accepts a list of prompts and a flat, study-major list of images. Transformers
4.51.3 LlavaForConditionalGeneration scatters image features into image-token
positions in that same order. We check those counts and audit the actual slices.
Sources: microsoft/maira-2 processing_maira2.py and modeling_maira2.py at the
configured immutable revision; transformers v4.51.3 models/llava/{processing,
modeling}_llava.py; https://huggingface.co/docs/transformers/v4.51.3/model_doc/llava
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .generation import MairaBackend, environment_info, input_identity, validate_config, verify_audit_artifacts
from .io import append_jsonl, exclusive_writer, read_jsonl, sha256_file, stable_hash, write_json

BATCH_VERSION = "maira-greedy-batch-v1"


def batch_config(config: dict, batch_size: int) -> dict:
    config = validate_config(config)
    if config["backend"] != "maira":
        raise ValueError("Batched inference requires backend=maira")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if config["generation"].get("do_sample", False):
        raise ValueError("Stochastic batching changes per-study seed semantics; use the singleton sampling CLI")
    if config["generation"].get("num_beams", 1) != 1:
        raise ValueError("Only greedy num_beams=1 batching has been implemented")
    protocol = {"version": BATCH_VERSION, "batch_size": batch_size, "padding_side": "left",
                "grouping": "manifest_order_fixed_chunks", "implementation_sha256": sha256_file(__file__)}
    if config.get("batching") not in (None, protocol):
        raise ValueError("Existing batching protocol differs; use the original unbatched config")
    config["batching"] = protocol
    return config


def inspect_layout(input_ids, attention_mask, pixel_shape, view_counts, *, image_token_id: int,
                   tokens_per_image: int, pad_token_id: int) -> dict:
    """Reject incorrect padding and image ownership before a model forward pass."""
    ids, mask = np.asarray(input_ids), np.asarray(attention_mask)
    if ids.ndim != 2 or mask.shape != ids.shape or ids.shape[0] != len(view_counts):
        raise ValueError("Unexpected batched token layout")
    if len(pixel_shape) != 4 or pixel_shape[0] != sum(view_counts) or pixel_shape[1] != 3:
        raise ValueError("Expected flat [sum(views),3,H,W] pixel_values")
    if tokens_per_image < 1 or not np.isin(mask, [0, 1]).all():
        raise ValueError("Invalid image token count or attention mask")
    offsets, lengths, offset = [], [], 0
    for index, count in enumerate(view_counts):
        if count not in (1, 2):
            raise ValueError("Only current frontal and optional lateral are supported")
        length = int(mask[index].sum())
        if length < 1 or not np.array_equal(mask[index], [0] * (ids.shape[1] - length) + [1] * length):
            raise ValueError("Decoder-only generation requires contiguous left padding")
        if not np.all(ids[index, :ids.shape[1] - length] == pad_token_id):
            raise ValueError("Unexpected token in masked padding")
        if int(np.count_nonzero(ids[index] == image_token_id)) != count * tokens_per_image:
            raise ValueError("Image-token count does not match this study's views")
        offsets.append((offset, offset + count))
        lengths.append(length)
        offset += count
    return {"image_offsets": offsets, "prompt_tokens": lengths, "padded_prompt_tokens": int(ids.shape[1])}


def split_completions(output_ids, input_ids, eos_token_ids, *, max_new_tokens: int) -> list[dict]:
    """Slice after the padded prompt width, then trim at each row's first EOS."""
    output, inputs = np.asarray(output_ids), np.asarray(input_ids)
    if output.ndim != 2 or inputs.ndim != 2 or output.shape[0] != inputs.shape[0]:
        raise ValueError("Unexpected generation output layout")
    width = inputs.shape[1]
    if output.shape[1] < width or not np.array_equal(output[:, :width], inputs):
        raise ValueError("Generation output does not preserve its padded input prefix")
    if output.shape[1] - width > max_new_tokens:
        raise ValueError("Generation exceeded max_new_tokens")
    eos = {int(x) for x in eos_token_ids}
    if not eos:
        raise ValueError("Explicit EOS token IDs are required for batch accounting")
    result = []
    for row in output[:, width:]:
        first_eos = next((i for i, token in enumerate(row) if int(token) in eos), None)
        stop = len(row) if first_eos is None else first_eos + 1
        result.append({"token_ids": [int(token) for token in row[:stop]], "completion_tokens": stop,
                       "ended_with_eos": first_eos is not None,
                       "possibly_truncated": first_eos is None and stop >= max_new_tokens})
    return result


class BatchedMairaBackend(MairaBackend):
    """One resident model, greedy batches, actual per-study processor artifacts."""

    def __init__(self, config: dict):
        batch_config(config, config.get("batching", {}).get("batch_size", 1))
        super().__init__(config)
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            raise ValueError("MAIRA tokenizer must define a padding token")

    def generate_batch(self, rows: list[dict], audit_dirs: list[Path]) -> list[dict | Exception]:
        if not rows or len(rows) != len(audit_dirs):
            raise ValueError("One audit directory per nonempty batch member is required")
        if self.config["generation"].get("do_sample") or self.config["generation"].get("num_beams", 1) != 1:
            raise ValueError("Only greedy batching is supported")
        torch = self.torch
        seed = int(self.config["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        texts, images, counts = [], [], []
        for row in rows:
            input_identity(row, self.config)  # Includes hashes and the clinical input allowlist.
            views = {}
            for view in ("frontal", "lateral"):
                if view in row["local_image_paths"]:
                    with Image.open(row["local_image_paths"][view]) as im:
                        views[view] = im.convert("RGB")
            metadata = row.get("allowed_input_metadata", {})
            text, study_images = self.processor.format_reporting_input(
                current_frontal=views["frontal"], current_lateral=views.get("lateral"),
                prior_frontal=None, prior_report=None, get_grounding=False,
                indication=metadata.get("indication"), technique=metadata.get("technique"),
                comparison=metadata.get("comparison"))
            if len(study_images) != len(views):
                raise ValueError("Official processor returned an unexpected number of images")
            texts.append(text)
            images.extend(study_images)
            counts.append(len(study_images))
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        if set(inputs) != {"input_ids", "attention_mask", "pixel_values"}:
            raise ValueError("Unexpected processor fields; inspect before adapting batch layout")
        pixels = inputs["pixel_values"]
        ip = self.processor
        patch = int(ip.patch_size)
        tokens_per_image = (pixels.shape[-2] // patch) * (pixels.shape[-1] // patch) + int(ip.num_additional_image_tokens)
        if ip.vision_feature_select_strategy == "default":
            tokens_per_image -= 1
        elif ip.vision_feature_select_strategy != "full":
            raise ValueError("Unsupported vision feature selection strategy")
        ids = inputs["input_ids"].detach().cpu().numpy()
        mask = inputs["attention_mask"].detach().cpu().numpy()
        layout = inspect_layout(ids, mask, pixels.shape, counts,
                                image_token_id=int(self.model.config.image_token_index),
                                tokens_per_image=tokens_per_image,
                                pad_token_id=int(self.processor.tokenizer.pad_token_id))
        inputs = inputs.to(self.device)
        for key, value in inputs.items():
            if value.is_floating_point():
                inputs[key] = value.to(dtype=getattr(torch, self.config["dtype"]))
        audits = []
        for index, (row, directory, (start, end)) in enumerate(zip(rows, audit_dirs, layout["image_offsets"])):
            directory = Path(directory)
            audit = self._audit({"pixel_values": inputs["pixel_values"][start:end]}, row, directory)
            # These are exactly the token tensors passed to generate, including padding.
            path = directory / "prompt_tokens.npz"
            np.savez(path, input_ids=ids[index], attention_mask=mask[index])
            audit.update({"batch_image_offset": [start, end], "batch_member_index": index,
                          "batch_size": len(rows), "tokens_per_image": tokens_per_image,
                          "prompt_tensor_path": str(path.resolve()), "prompt_tensor_sha256": sha256_file(path),
                          "padding_side": "left", "padded_prompt_tokens": layout["padded_prompt_tokens"]})
            write_json(directory / "processor_audit.json", audit)
            audits.append(audit)
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, **self.config["generation"])
        torch.cuda.synchronize(self.device)
        eos = self.model.generation_config.eos_token_id
        completions = split_completions(output.detach().cpu().numpy(), ids,
                                       [eos] if isinstance(eos, int) else eos,
                                       max_new_tokens=self.config["generation"]["max_new_tokens"])
        results = []
        for index, completion in enumerate(completions):
            result = {"generated_findings": None, "processor_audit": audits[index], "synthetic": False,
                      "generation_calls": 1 / len(rows), "generation_batch_size": len(rows),
                      "prompt_tokens": layout["prompt_tokens"][index],
                      "padded_prompt_tokens": layout["padded_prompt_tokens"],
                      **{key: value for key, value in completion.items() if key != "token_ids"},
                      "peak_memory_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                      "peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved(self.device))}
            try:
                raw = self.processor.decode(completion.pop("token_ids"), skip_special_tokens=True).lstrip()
                result["decoded_completion"] = raw
                findings = self.processor.convert_output_to_plaintext_or_grounded_sequence(raw)
                if not isinstance(findings, str) or not findings.strip():
                    raise ValueError("Expected nonempty ungrounded Findings string")
                result["generated_findings"] = findings
            except Exception as exc:
                result["error_status"] = {"type": type(exc).__name__, "stage": "decoding",
                                          "message": "Invalid or empty ungrounded Findings output."}
            results.append(result)
        return results


def verify_batch_artifacts(record: dict) -> None:
    verify_audit_artifacts(record)
    audit = record.get("processor_audit", {})
    path = audit.get("prompt_tensor_path")
    if not path or not Path(path).is_file() or sha256_file(path) != audit.get("prompt_tensor_sha256"):
        raise ValueError("Batched prompt audit is missing or changed")


def plan_batches(rows: list[dict], config: dict, batch_size: int, *, allow_test=False) -> list[list[dict]]:
    """Bind batch membership to identity; a resumed partial batch keeps its shape."""
    config = batch_config(config, batch_size)
    seen, subjects, items = set(), {}, []
    for row in rows:
        patient, partition = str(row["subject_id"]), row["partition"]
        if patient in subjects and subjects[patient] != partition:
            raise ValueError("Patient crosses partitions")
        subjects[patient] = partition
        key = (str(row["study_id"]), row.get("input_variant_id", "original"))
        if key in seen:
            raise ValueError("Duplicate study/variant in manifest")
        seen.add(key)
        if (partition == "test" or row.get("official_split") == "test") and not allow_test:
            raise ValueError("Test is sealed")
        try:
            identity, error = input_identity(row, config), None
        except (OSError, ValueError) as exc:
            identity, error = {"invalid_input": row, "config": config}, exc
        items.append({"row": row, "identity": identity, "input_error": error})
    batches = []
    for offset in range(0, len(items), batch_size):
        group = items[offset:offset + batch_size]
        members = [stable_hash(item["identity"]) for item in group]
        for index, item in enumerate(group):
            item["identity"] = {**item["identity"], "batch_context": {"member_input_hashes": members, "member_index": index}}
            item["key"] = stable_hash(item["identity"])
        batches.append(group)
    return batches


def run_cache_batched(rows: list[dict], config: dict, output: str | Path, *, audit_dir: str | Path,
                      batch_size: int = 1, retry_errors=False, allow_test=False, backend=None) -> dict:
    config = batch_config(config, batch_size)
    batches = plan_batches(rows, config, batch_size, allow_test=allow_test)
    output, audit_dir = Path(output), Path(audit_dir)
    summary = {"success": 0, "failed": 0, "skipped": 0, "skipped_errors": 0,
               "batch_calls": 0, "recomputed_cached_members": 0, "backend": "maira", "batch_size": batch_size}
    loaded = backend
    if loaded is not None and getattr(loaded, "config", config) != config:
        raise ValueError("Injected backend config differs from the bound batch protocol")
    with exclusive_writer(output):
        existing = read_jsonl(output) if output.exists() else []
        planned = {item["key"] for group in batches for item in group}
        if any(record["cache_key"] not in planned for record in existing):
            raise ValueError("Cache belongs to another manifest/batch protocol; choose a new output")
        latest = {record["cache_key"]: record for record in existing}
        for group in batches:
            pending = []
            for item in group:
                previous = latest.get(item["key"])
                skip = previous is not None and (previous.get("error_status") is None or not retry_errors)
                if skip and previous.get("error_status") is None:
                    try:
                        verify_batch_artifacts(previous)
                    except (OSError, ValueError) as exc:
                        if not retry_errors:
                            raise ValueError("Successful cache audit changed; use --retry-errors") from exc
                        skip = False
                if skip:
                    summary["skipped"] += 1
                    summary["skipped_errors"] += int(previous.get("error_status") is not None)
                else:
                    pending.append(item)
            if not pending:
                continue
            start, stage = time.perf_counter(), "input_validation"
            stopped = False
            try:
                # Never silently drop an invalid member, changing padding/numerics of its peers.
                if any(item["input_error"] is not None for item in group):
                    raise ValueError("Invalid batch member; fix the input manifest before retrying")
                if loaded is None:
                    stage = "backend_initialization"
                    loaded = BatchedMairaBackend(config)
                stage = "generation"
                results = loaded.generate_batch([item["row"] for item in group], [audit_dir / item["key"] for item in group])
                if len(results) != len(group):
                    raise ValueError("Backend returned the wrong number of batch members")
                summary["batch_calls"] += 1
                summary["recomputed_cached_members"] += len(group) - len(pending)
            except Exception as exc:
                results = [exc] * len(group)
                stopped = stage == "backend_initialization"
            elapsed = time.perf_counter() - start
            for item, result in zip(group, results):
                if item not in pending:
                    continue
                row, key = item["row"], item["key"]
                record = {field: row.get(field) for field in ("subject_id", "study_id", "dicom_id", "partition",
                          "official_split", "input_variant_id", "sampling_stratum", "dataset", "split_source")}
                record.update({"cache_key": key, "record_id": key, "backend": "maira", "synthetic": False,
                               "model_revision": config["model_revision"], "processor_revision": config["processor_revision"],
                               "decoding_config_hash": stable_hash(config["generation"]), "input_sha256": key,
                               "input_identity": item["identity"], "local_image_paths": row.get("local_image_paths"),
                               "generated_findings": None, "error_status": None,
                               "attempt": int(latest.get(key, {}).get("attempt", 0)) + 1,
                               "runtime_sec": elapsed / len(group), "batch_runtime_sec": elapsed})
                if not isinstance(result, Exception) and not result.get("error_status") and (not isinstance(result.get("generated_findings"), str) or not result["generated_findings"].strip()):
                    result = ValueError("Empty Findings returned")
                if isinstance(result, Exception):
                    record["error_status"] = {"type": type(result).__name__, "stage": stage,
                                              "message": "Batch inference failed; inspect a separate diagnostic smoke run."}
                    summary["failed"] += 1
                else:
                    record.update(result)
                    summary["failed" if result.get("error_status") else "success"] += 1
                append_jsonl(output, record)
                latest[key] = record
            if stopped:
                summary["stopped_on_backend_error"] = True
                break
        write_json(str(output) + ".run.json", {"config": config, "environment": environment_info(), "summary": summary,
                   "note": "Greedy batches form a distinct protocol; no equivalence to singleton caches is assumed. "
                           "Partial batches are recomputed with original members; only missing/error records are appended. "
                           "Peak memory is shared by the batch; runtime_sec is amortized batch time."})
    return summary

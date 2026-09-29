"""Canonical-text MAIRA report likelihood; not original-trajectory probability.

The report is retokenized separately with add_special_tokens=False. Only those
suffix tokens are scored; prompt, padding and EOS are excluded. This baseline
is report-level and is not an official SCUQ reproduction or a support label.
"""
from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image

from .batched_generation import inspect_layout, plan_batches, verify_batch_artifacts
from .claims import extract_claims
from .generation import MairaBackend, PIPELINE_VERSION, environment_info, input_identity, validate_config
from .io import sha256_file, stable_hash, write_json, write_jsonl

VERSION = "maira-canonical-report-mean-loglikelihood-v1"


def pack_teacher_forcing(prompts, suffixes, *, pad_token_id):
    """Right-pad prompt+suffix; label only the canonical suffix, without EOS."""
    if not prompts or len(prompts) != len(suffixes):
        raise ValueError("One nonempty prompt and suffix are required per report")
    prompts, suffixes = [list(p) for p in prompts], [list(s) for s in suffixes]
    for tokens in [*prompts, *suffixes]:
        if not tokens or any(isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer)) or t < 0 for t in tokens):
            raise ValueError("Prompt/suffix tokens must be nonempty nonnegative integer sequences")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0:
        raise ValueError("An explicit padding token ID is required")
    lengths = [len(p) + len(s) for p, s in zip(prompts, suffixes)]
    ids = np.full((len(prompts), max(lengths)), pad_token_id, dtype=np.int64)
    mask = np.zeros_like(ids)
    labels = np.full_like(ids, -100)
    for i, (prompt, suffix) in enumerate(zip(prompts, suffixes)):
        length, split = lengths[i], len(prompt)
        ids[i, :length] = prompt + suffix
        mask[i, :length] = 1
        labels[i, split:length] = suffix
    positions = np.maximum(mask.cumsum(axis=1) - 1, 0)
    positions[mask == 0] = 0
    return {"input_ids": ids, "attention_mask": mask, "labels": labels, "position_ids": positions,
            "prompt_lengths": [len(p) for p in prompts], "suffix_lengths": [len(s) for s in suffixes]}


def causal_suffix_positions(batch):
    """Return (report row, preceding-logit position, target ID), checking masks."""
    ids, mask, labels = (np.asarray(batch[key]) for key in ("input_ids", "attention_mask", "labels"))
    if (ids.ndim != 2 or mask.shape != ids.shape or labels.shape != ids.shape or not np.isin(mask, [0, 1]).all()
            or not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(labels.dtype, np.integer) or np.any(ids < 0)):
        raise ValueError("Invalid teacher-forcing tensor layout")
    if len(batch["prompt_lengths"]) != len(ids) or len(batch["suffix_lengths"]) != len(ids):
        raise ValueError("Length metadata does not match reports")
    for index, (prompt, suffix) in enumerate(zip(batch["prompt_lengths"], batch["suffix_lengths"])):
        length = prompt + suffix
        if prompt < 1 or suffix < 1 or length > ids.shape[1]:
            raise ValueError("Invalid prompt/suffix lengths")
        if not np.array_equal(mask[index], [1] * length + [0] * (ids.shape[1] - length)):
            raise ValueError("Teacher forcing requires contiguous right padding")
        expected = np.full(ids.shape[1], -100, dtype=np.int64)
        expected[prompt:length] = ids[index, prompt:length]
        if not np.array_equal(labels[index], expected):
            raise ValueError("Only suffix targets may be scored; prompt/padding labels are prohibited")
    owners, targets_at = np.nonzero(labels != -100)
    return owners, targets_at - 1, labels[owners, targets_at]


def summarize_token_log_probs(log_probs, owners, batch_size):
    values, owners = np.asarray(log_probs, dtype=np.float64), np.asarray(owners)
    if values.ndim != 1 or owners.shape != values.shape or not np.isfinite(values).all() or np.any(values > 1e-6):
        raise ValueError("Invalid token log probabilities")
    result = []
    for index in range(batch_size):
        member = values[owners == index]
        if not len(member):
            raise ValueError("Empty canonical report has no normalized likelihood")
        mean = float(member.mean())
        result.append({"score": mean, "mean_token_log_probability": mean,
                       "mean_token_nll": -mean, "sum_token_log_probability": float(member.sum()),
                       "scored_token_count": len(member)})
    return result


def numpy_teacher_forced_scores(logits, batch):
    """Small CPU reference for auditing causal shift and length normalization."""
    logits = np.asarray(logits)
    owners, positions, targets = causal_suffix_positions(batch)
    if logits.ndim != 3 or logits.shape[:2] != batch["input_ids"].shape or targets.max() >= logits.shape[2]:
        raise ValueError("Logits do not align with the full input sequence/vocabulary")
    selected = np.asarray(logits[owners, positions], dtype=np.float64)
    if not np.isfinite(selected).all():
        raise ValueError("Nonfinite scored logits")
    maximum = selected.max(axis=1)
    log_normalizer = maximum + np.log(np.exp(selected - maximum[:, None]).sum(axis=1))
    values = selected[np.arange(len(targets)), targets] - log_normalizer
    return summarize_token_log_probs(values, owners, len(batch["input_ids"]))


def unpad_prompt(ids, mask, pad_token_id):
    ids, mask = np.asarray(ids), np.asarray(mask)
    if ids.ndim != 1 or mask.shape != ids.shape or not np.isin(mask, [0, 1]).all():
        raise ValueError("Invalid cached prompt mask")
    length = int(mask.sum())
    if length < 1 or not np.array_equal(mask, [0] * (len(ids) - length) + [1] * length):
        raise ValueError("Audited generation prompt must have contiguous left padding")
    if not np.all(ids[:len(ids) - length] == pad_token_id):
        raise ValueError("Masked generation prompt contains non-pad tokens")
    return ids[-length:].tolist()


def verify_replayed_audit(record, replayed_audit, prompt_ids, prompt_mask, *, pad_token_id):
    """Verify original artifacts and exact pixels/prompt, ignoring old left pads."""
    verify_batch_artifacts(record)
    original = record["processor_audit"]
    for field in ("tensor_sha256", "presented_image_sha256"):
        if replayed_audit.get(field) != original.get(field):
            raise ValueError(f"Replayed processor {field} differs from generation cache")
    if stable_hash(replayed_audit.get("image_processor")) != stable_hash(original.get("image_processor")):
        raise ValueError("Replayed image processor configuration differs")
    with np.load(original["prompt_tensor_path"], allow_pickle=False) as saved:
        if set(saved.files) != {"input_ids", "attention_mask"}:
            raise ValueError("Unknown generation prompt audit fields")
        old = unpad_prompt(saved["input_ids"], saved["attention_mask"], pad_token_id)
    new = unpad_prompt(prompt_ids, prompt_mask, pad_token_id)
    if old != new or record.get("prompt_tokens") != len(new):
        raise ValueError("Replayed official prompt tokens differ from generation cache")
    return new


def prepare_cache_reports(cache_rows, manifest_rows, supplied_config):
    """Require the original full generation manifest and reconstruct every key.

    The scoring batch size is unrelated to the original generation batch size.
    Missing cache entries remain explicit denominator rows. All historical cache
    attempts must match this exact input/configuration; the last attempt wins.
    """
    config = validate_config(supplied_config)
    if config["backend"] != "maira" or not cache_rows or not manifest_rows:
        raise ValueError("Nonempty real MAIRA cache and its original manifest are required")
    identities = [row.get("input_identity") for row in cache_rows if row.get("input_identity")]
    configurations = {stable_hash(value.get("config")): value.get("config") for value in identities}
    if len(configurations) != 1:
        raise ValueError("Cache must have one bound generation configuration")
    cached_config = next(iter(configurations.values()))
    if not isinstance(cached_config, dict):
        raise ValueError("Cache lacks its generation configuration")
    comparable = lambda value: {key: item for key, item in value.items() if key != "batching"}
    if comparable(cached_config) != comparable(config) or ("batching" in config and config["batching"] != cached_config.get("batching")):
        raise ValueError("Supplied configuration differs from cache generation configuration")
    if cached_config.get("batching"):
        size = cached_config["batching"].get("batch_size")
        batches = plan_batches(manifest_rows, cached_config, size, allow_test=True)
        planned = [item for group in batches for item in group]
    else:
        planned, seen, patients = [], set(), {}
        for row in manifest_rows:
            member = (str(row["study_id"]), row.get("input_variant_id", "original"))
            if member in seen:
                raise ValueError("Duplicate study/variant in generation manifest")
            seen.add(member)
            patient = (row.get("dataset"), str(row["subject_id"]))
            if patient in patients and patients[patient] != row["partition"]:
                raise ValueError("Patient crosses generation partitions")
            patients[patient] = row["partition"]
            try:
                identity, error = input_identity(row, cached_config), None
                key = stable_hash(identity)
            except (OSError, ValueError) as exc:
                identity, error = None, exc
                key = stable_hash({"invalid_input": row, "config": cached_config, "pipeline_version": PIPELINE_VERSION})
            planned.append({"row": row, "identity": identity, "key": key, "input_error": error})
    by_key = {item["key"]: item for item in planned}
    latest = {}
    for cache in cache_rows:
        key = cache.get("cache_key")
        if key not in by_key:
            raise ValueError("Cache identity does not match the original manifest/configuration")
        item = by_key[key]
        expected_sha = key if item["identity"] is not None else None
        if (cache.get("backend") != "maira" or cache.get("synthetic") is not False
                or cache.get("input_identity") != item["identity"] or cache.get("input_sha256") != expected_sha
                or cache.get("model_revision") != cached_config["model_revision"]
                or cache.get("processor_revision") != cached_config["processor_revision"]
                or cache.get("decoding_config_hash") != stable_hash(cached_config["generation"])):
            raise ValueError("Cache input SHA, model/processor revision or generation identity differs")
        for field in ("subject_id", "study_id", "partition"):
            if str(cache.get(field)) != str(item["row"].get(field)):
                raise ValueError("Cache top-level input identity differs from manifest")
        if (cache.get("input_variant_id") or "original") != (item["row"].get("input_variant_id") or "original"):
            raise ValueError("Cache variant differs from manifest")
        if cache.get("dataset") != item["row"].get("dataset"):
            raise ValueError("Cache dataset differs from manifest")
        if "error_status" not in cache:
            raise ValueError("Cache must explicitly record generation error status")
        latest[key] = cache
    reports = []
    for item in planned:
        cache, row = latest.get(item["key"]), item["row"]
        result = {field: row.get(field) for field in ("input_id", "subject_id", "study_id", "dataset", "partition", "input_variant_id")}
        result.update(cache_key=item["key"], score=None, mean_token_log_probability=None, mean_token_nll=None,
                      generated_findings_sha256=None, eligible_negative_claims=None, status="pending")
        if cache is None:
            result["status"] = "missing_cache"
        elif cache["error_status"] is not None:
            result.update(status="generation_failure", generation_error_type=cache["error_status"].get("type"))
        else:
            if type(cache.get("possibly_truncated")) is not bool:
                raise ValueError("Successful generation needs an explicit truncation flag")
            text = cache.get("generated_findings")
            if not isinstance(text, str) or not text.strip():
                result["status"] = "empty_findings"
            else:
                result.update(generated_findings_sha256=stable_hash(text),
                              eligible_negative_claims=sum(claim["eligible"] for claim in extract_claims(text)))
                if cache["possibly_truncated"]:
                    result["status"] = "truncated_generation"
                else:
                    # No old singleton prompt audit is silently treated as verified.
                    verify_batch_artifacts(cache)
        reports.append({"row": row, "cache": cache, "result": result})
    return reports, cached_config


class MairaReportLikelihoodBackend(MairaBackend):
    """Reuse official loading/processor and exact inherited pixel audit exporter."""

    def __init__(self, config):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if (not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", visible)
                or config.get("device") != "cuda:0"):
            raise RuntimeError("Set exactly one complete GPU UUID before loading report-likelihood weights")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one UUID-masked CUDA device is required before model loading")
        super().__init__(config)
        if self.config["device"] != "cuda:0" or self.torch.cuda.device_count() != 1:
            raise RuntimeError("Report likelihood requires one UUID-masked cuda:0 device")
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            raise ValueError("MAIRA tokenizer must define pad_token_id")

    def score_batch(self, reports, audit_dirs):
        if not reports or len(reports) != len(audit_dirs):
            raise ValueError("Each report requires a separate audit directory")
        torch = self.torch
        texts, images, counts = [], [], []
        for item in reports:
            row = item["row"]
            views = {}
            for view in ("frontal", "lateral"):
                if view in row["local_image_paths"]:
                    with Image.open(row["local_image_paths"][view]) as im:
                        views[view] = im.convert("RGB")
            metadata = row.get("allowed_input_metadata", {})
            text, study_images = self.processor.format_reporting_input(
                current_frontal=views["frontal"], current_lateral=views.get("lateral"),
                prior_frontal=None, prior_report=None, get_grounding=False,
                indication=metadata.get("indication"), technique=metadata.get("technique"), comparison=metadata.get("comparison"))
            if len(study_images) != len(views):
                raise ValueError("Official processor returned unexpected view count")
            texts.append(text); images.extend(study_images); counts.append(len(study_images))
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        if set(inputs) != {"input_ids", "attention_mask", "pixel_values"}:
            raise ValueError("Unexpected official processor fields")
        ids, mask = (inputs[key].detach().cpu().numpy() for key in ("input_ids", "attention_mask"))
        pixels, ip = inputs["pixel_values"], self.processor
        patch = int(ip.patch_size)
        image_tokens = (pixels.shape[-2] // patch) * (pixels.shape[-1] // patch) + int(ip.num_additional_image_tokens)
        if ip.vision_feature_select_strategy == "default": image_tokens -= 1
        elif ip.vision_feature_select_strategy != "full": raise ValueError("Unknown vision feature strategy")
        image_id, pad_id = int(self.model.config.image_token_index), int(ip.tokenizer.pad_token_id)
        layout = inspect_layout(ids, mask, pixels.shape, counts, image_token_id=image_id,
                                tokens_per_image=image_tokens, pad_token_id=pad_id)
        pixels = pixels.to(device=self.device, dtype=getattr(torch, self.config["dtype"]))
        prompts, suffixes, replayed = [], [], []
        forbidden = set(ip.tokenizer.all_special_ids) | {image_id, pad_id}
        for index, (item, directory, (start, end)) in enumerate(zip(reports, audit_dirs, layout["image_offsets"])):
            audit = self._audit({"pixel_values": pixels[start:end]}, item["row"], Path(directory))
            prompt = verify_replayed_audit(item["cache"], audit, ids[index], mask[index], pad_token_id=pad_id)
            suffix = ip.tokenizer.encode(item["cache"]["generated_findings"], add_special_tokens=False)
            if not suffix or any(token in forbidden for token in suffix):
                raise ValueError("Canonical report must contain nonempty ordinary tokens; special tokens/EOS are not scored")
            prompts.append(prompt); suffixes.append(suffix); replayed.append(audit)
        batch = pack_teacher_forcing(prompts, suffixes, pad_token_id=pad_id)
        max_positions = int(self.model.config.text_config.max_position_embeddings)
        if batch["input_ids"].shape[1] > max_positions:
            raise ValueError("Full prompt plus canonical report exceeds context; silent truncation is prohibited")
        owners, positions, targets = causal_suffix_positions(batch)
        tensor_inputs = {key: torch.as_tensor(batch[key], device=self.device) for key in ("input_ids", "attention_mask", "position_ids")}
        for index, directory in enumerate(audit_dirs):
            path = Path(directory) / "teacher_forcing_tokens.npz"
            np.savez(path, **{key: batch[key][index] for key in ("input_ids", "attention_mask", "labels", "position_ids")})
            replayed[index].update(teacher_forcing_tokens_path=str(path.resolve()), teacher_forcing_tokens_sha256=sha256_file(path),
                                   canonical_suffix_ids_sha256=stable_hash(suffixes[index]), verified_generation_prompt=True)
            write_json(Path(directory) / "likelihood_audit.json", replayed[index])
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode():
            # No generate(), no labels averaging over prompt, no past KV cache.
            output = self.model(**tensor_inputs, pixel_values=pixels, use_cache=False, return_dict=True, logits_to_keep=0)
            if tuple(output.logits.shape[:2]) != tuple(batch["input_ids"].shape):
                raise ValueError("Model logits do not preserve full teacher-forcing positions")
            log_probs = np.empty(len(targets), dtype=np.float64)
            for start in range(0, len(targets), 256):
                stop = min(start + 256, len(targets))
                row_index = torch.as_tensor(owners[start:stop], device=self.device)
                position_index = torch.as_tensor(positions[start:stop], device=self.device)
                target_ids = torch.as_tensor(targets[start:stop], device=self.device)
                selected = output.logits[row_index, position_index].float()
                values = -torch.nn.functional.cross_entropy(selected, target_ids, reduction="none")
                log_probs[start:stop] = values.detach().cpu().numpy()
        torch.cuda.synchronize(self.device)
        results = summarize_token_log_probs(log_probs, owners, len(reports))
        for index, result in enumerate(results):
            result.update(processor_audit=replayed[index], prompt_tokens=len(prompts[index]),
                          canonical_suffix_ids_sha256=stable_hash(suffixes[index]),
                          peak_memory_bytes=int(torch.cuda.max_memory_allocated(self.device)),
                          peak_reserved_memory_bytes=int(torch.cuda.max_memory_reserved(self.device)))
        return results


def run_report_likelihood(cache_rows, manifest_rows, config, output_dir, *, batch_size=4, backend=None, gpu_snapshot=None):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    reports, effective_config = prepare_cache_reports(cache_rows, manifest_rows, config)
    if isinstance(backend, MairaReportLikelihoodBackend) and backend.config != effective_config:
        raise ValueError("Preloaded scoring backend differs from the bound cache configuration")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    definition = {"schema_version": VERSION, "score_definition": "mean log P(canonical report token | official prompt, current images, earlier canonical report tokens)",
                "higher_means": "higher report-level canonical textual likelihood, not clinical support",
                "suffix_tokenization": "generated_findings exactly as stored; encode(add_special_tokens=False); append no EOS; reject any special token",
                "original_generation_trajectory_probability": False, "official_scuq_reproduction": False,
                "claim_level_offsets_reconstructed": False, "eos_included": False, "prompt_or_pad_tokens_scored": False,
                "teacher_forcing_use_cache": False, "padding_side": "right", "batch_size": batch_size,
                "batch_membership": "original manifest order, successful nontruncated reports only; no eligible-negative filtering",
                "implementation_sha256": sha256_file(__file__), "claims_parser_sha256": sha256_file(Path(__file__).with_name("claims.py")),
                "conditional_model": {key: effective_config[key] for key in ("model_source", "model_revision", "processor_revision", "dtype")},
                "runtime_packages": environment_info()["packages"],
                "synthetic_test_double": backend is not None and not isinstance(backend, MairaReportLikelihoodBackend),
                "clinical_effectiveness_evaluated": False}
    protocol = {"definition": definition, "protocol_sha256": stable_hash(definition), "generation_config": effective_config,
                "cache_rows_sha256": stable_hash(cache_rows), "manifest_rows_sha256": stable_hash(manifest_rows),
                "environment": environment_info(), "gpu_snapshot": gpu_snapshot}
    protocol["run_sha256"] = stable_hash(protocol)
    write_json(output / "protocol.json", protocol)
    results = [item["result"] for item in reports]
    for result in results:
        result.update(score_protocol_sha256=protocol["protocol_sha256"], clinical_support_prediction=False,
                      synthetic_test_double=definition["synthetic_test_double"])
    write_jsonl(output / "scores.jsonl", results)
    pending = [item for item in reports if item["result"]["status"] == "pending"]
    state = {"status": "running", "source_manifest_inputs": len(reports), "cache_rows_including_retries": len(cache_rows),
             "latest_cache_records": sum(item["cache"] is not None for item in reports), "protocol_sha256": protocol["protocol_sha256"]}
    write_json(output / "status.json", state)
    start = time.perf_counter()
    try:
        if pending and backend is None:
            backend = MairaReportLikelihoodBackend(effective_config)
        for offset in range(0, len(pending), batch_size):
            group = pending[offset:offset + batch_size]
            directories = [output / "audit" / item["result"]["cache_key"] for item in group]
            values = backend.score_batch(group, directories)
            if len(values) != len(group):
                raise ValueError("Backend did not return one score per report")
            for item, value in zip(group, values):
                if not np.isfinite(value.get("score", np.nan)) or value["score"] > 1e-6 or value.get("scored_token_count", 0) < 1:
                    raise ValueError("Backend returned an invalid report score")
                item["result"].update(value, status="scored")
            write_jsonl(output / "scores.jsonl", results)
            state.update(scored=sum(result["status"] == "scored" for result in results), runtime_sec=time.perf_counter() - start)
            write_json(output / "status.json", state)
        state["status"] = "completed"
    except Exception as exc:
        for result in results:
            if result["status"] == "pending":
                result.update(status="scoring_not_completed", scoring_error_type=type(exc).__name__)
        state.update(status="failed", scoring_error_type=type(exc).__name__)
        write_jsonl(output / "scores.jsonl", results)
        raise
    finally:
        state.update(runtime_sec=time.perf_counter() - start, status_counts=dict(Counter(row["status"] for row in results)),
                     reports_without_eligible_negatives=sum(row["eligible_negative_claims"] == 0 for row in results),
                     denominator_note="every original manifest input remains; zero eligible claims is counted but does not suppress report scoring")
        write_json(output / "status.json", state)
    return state

"""Fresh, audited image-model training; no calibration/test array access."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from oa_cxr.io import stable_hash

SEED = 17
VARIANTS = 13
MIN_FIT_SOURCES = 10_000
EPOCHS = 13
FINDINGS = ["pleural_effusion", "pneumothorax", "consolidation"]
TRAIN_BATCH_SIZES = (32, 64, 128, 256, 512)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, document):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


AUDIT_COMPONENTS = {
    "encoder_conv0": "encoder.features.conv0.",
    "anatomy_decoder": "anatomy_decoder.",
    "lung_head": "lung_head.",
    "finding_embeddings": "finding_embeddings.",
    "retention_head": "retention_head.",
}


def _parameter_array(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.array(value, dtype=np.float32, copy=True)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("audit parameters must be finite nonempty arrays")
    return array


def capture_audit_parameters(model):
    """Small immutable CPU snapshot, not another checkpoint or optimizer state."""
    result = {name: _parameter_array(value) for name, value in model.named_parameters()
              if any(name.startswith(prefix) for prefix in AUDIT_COMPONENTS.values())}
    if "encoder.features.conv0.weight" not in result or any(
            not any(name.startswith(prefix) for name in result) for prefix in AUDIT_COMPONENTS.values()):
        raise ValueError("expected encoder stem and all new trainable audit components")
    return result


def parameter_update_evidence(initial, selected, *, optimizer_steps, selected_optimizer_steps):
    """Measure actual selected-weight changes, not merely finite forward scores."""
    if (type(optimizer_steps) is not int or type(selected_optimizer_steps) is not int
            or not 0 < selected_optimizer_steps <= optimizer_steps):
        raise ValueError("positive actual optimizer steps and selected-checkpoint step count required")
    if "encoder.features.conv0.weight" not in initial:
        raise ValueError("initial encoder conv0 audit is missing")
    components = {}
    for component, prefix in AUDIT_COMPONENTS.items():
        names = sorted(name for name in initial if name.startswith(prefix))
        if not names or any(name not in selected for name in names):
            raise ValueError("selected checkpoint is missing an audited component")
        parameters = {}; squared_delta = squared_initial = 0.0
        changed = elements = changed_parameters = 0; maximum = 0.0
        initial_hashes = []; selected_hashes = []
        for name in names:
            before, after = _parameter_array(initial[name]), _parameter_array(selected[name])
            if before.shape != after.shape: raise ValueError("audited parameter shape changed")
            delta = after.astype(np.float64) - before.astype(np.float64)
            count = int(np.count_nonzero(delta)); norm = float(np.linalg.norm(delta.ravel()))
            max_abs = float(np.max(np.abs(delta)))
            initial_digest = hashlib.sha256(before.tobytes(order="C")).hexdigest()
            selected_digest = hashlib.sha256(after.tobytes(order="C")).hexdigest()
            parameters[name] = {"shape": list(before.shape), "elements": before.size,
                "changed_elements": count, "difference_l2_norm": norm, "max_abs_difference": max_abs,
                "initial_float32_bytes_sha256": initial_digest, "selected_float32_bytes_sha256": selected_digest}
            initial_hashes.append([name, list(before.shape), initial_digest])
            selected_hashes.append([name, list(after.shape), selected_digest])
            squared_delta += float(np.square(delta).sum())
            squared_initial += float(np.square(before.astype(np.float64)).sum())
            changed += count; elements += before.size; changed_parameters += int(count > 0)
            maximum = max(maximum, max_abs)
        components[component] = {"parameter_count": len(names), "changed_parameter_count": changed_parameters,
            "elements": elements, "changed_elements": changed, "difference_l2_norm": squared_delta ** .5,
            "relative_difference_l2": (squared_delta / squared_initial) ** .5 if squared_initial else None,
            "max_abs_difference": maximum, "initial_parameters_sha256": stable_hash(initial_hashes),
            "selected_parameters_sha256": stable_hash(selected_hashes), "parameters": parameters}
    changed = all(group["changed_elements"] > 0 for group in components.values())
    return {"status": "passed" if changed else "failed", "all_audited_components_changed": changed,
            "optimizer_steps": optimizer_steps, "selected_checkpoint_optimizer_steps": selected_optimizer_steps,
            "components": components,
            "scope": "encoder first convolution and all new decoder/embedding/head parameters; not every encoder layer",
            "interpretation": "weight changes can include AdamW decay; they do not establish segmentation repair or clinical benefit",
            "clinical_segmentation_success_established": False}


def variant_for_epoch(source_id, epoch):
    if type(epoch) is not int or not 0 <= epoch < EPOCHS:
        raise ValueError("epoch must be a zero-based index in [0,12]")
    return (int(stable_hash(source_id), 16) + epoch) % VARIANTS


def select_dev(rows, limit=512):
    return sorted(range(len(rows)), key=lambda i: (stable_hash(["rebuild-dev", SEED, rows[i]["source_id"]]),
                                                  rows[i]["source_id"]))[:limit]


def smoke_source_count(batch_size):
    if batch_size not in TRAIN_BATCH_SIZES:
        raise ValueError("unsupported training batch size")
    return max(16, batch_size // 4)


def optimization_plan(source_count, batch_size):
    if type(source_count) is not int or source_count < 1 or batch_size not in TRAIN_BATCH_SIZES:
        raise ValueError("positive source count and supported training batch required")
    per_epoch = (source_count + batch_size - 1) // batch_size
    return {"physical_batch_size": batch_size, "effective_batch_size": batch_size,
            "gradient_accumulation_steps": 1, "drop_last": False,
            "expected_optimizer_steps_per_epoch": per_epoch,
            "expected_optimizer_steps_total": per_epoch * EPOCHS,
            "scope": "formal training; smoke always performs zero optimizer steps"}


def _bound_file(root, relative, bindings):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("data path escapes the prepared root")
    expected = bindings.get(relative)
    if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError(f"missing explicit SHA256 for {relative}")
    if sha256(path) != expected:
        raise ValueError(f"prepared data SHA256 mismatch: {relative}")
    return path


def validate_source_rows(rows, split, *, minimum_fit_sources=MIN_FIT_SOURCES):
    if not rows:
        raise ValueError(f"{split} source list is empty")
    required = ("source_id", "source_image_sha256", "split_group_id", "dataset", "source_image_path")
    for i, row in enumerate(rows):
        if row.get("row_index") != i or type(row.get("row_index")) is not int or row.get("split") != split:
            raise ValueError("source row indices/splits do not match the array contract")
        if any(not isinstance(row.get(k), str) or not row[k] for k in required):
            raise ValueError("source identity/provenance fields are missing")
        if not re.fullmatch(r"[a-f0-9]{64}", row["source_image_sha256"]):
            raise ValueError("invalid source image SHA256")
    for key in ("source_id", "source_image_sha256"):
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError(f"duplicate {key} in {split}")
    if split == "fit" and len(rows) < minimum_fit_sources:
        raise ValueError(f"at least {minimum_fit_sources} unique source images are required for fitting")


def assert_disjoint(fit, dev):
    for key in ("source_id", "source_image_sha256", "split_group_id"):
        if {r[key] for r in fit} & {r[key] for r in dev}:
            raise ValueError(f"fit/dev overlap in {key}")


def validate_data(root, *, include_dev):
    """Open only fit and optionally dev. Heldout arrays are never enumerated."""
    root = Path(root).resolve(strict=True)
    status = read_json(root / "status.json")
    if status.get("status") != "completed":
        raise ValueError("prepared data status must be completed")
    from oa_cxr.rebuild.cache_validation import validation_receipt
    review_path = validation_receipt(root)
    bindings = status.get("files", {})
    _bound_file(root, "protocol.json", bindings)
    protocol = read_json(root / "protocol.json")
    if (protocol.get("findings") != FINDINGS or protocol.get("variants") != VARIANTS or
            protocol.get("data_protocol", {}).get("normalization") != "(gray/255*2-1)*1024"):
        raise ValueError("prepared findings/variants/input normalization do not match the training contract")
    splits = ("fit", "dev") if include_dev else ("fit",)
    result = {"root": root, "rows": {}, "cache_review_sha256": sha256(review_path),
              "cache_status_sha256": sha256(root / "status.json"),
              "bindings": {"protocol.json": bindings["protocol.json"]}}
    for split in splits:
        files = {name: _bound_file(root, f"{split}/{name}", bindings)
                 for name in ("sources.jsonl", "images.npy", "masks.npy", "targets.npy")}
        rows = [json.loads(line) for line in files["sources.jsonl"].read_text(encoding="utf-8").splitlines() if line.strip()]
        validate_source_rows(rows, split)
        n = len(rows)
        counts = status.get("splits", {}).get(split, {})
        if any(counts.get(k) != v for k, v in {"sources": n, "inputs": n * 13, "targets": n * 39}.items()):
            raise ValueError(f"prepared {split} counts do not match the source rows")
        for name, dtype, shape in (("images.npy", np.uint8, (n, 13, 256, 256)),
                                   ("masks.npy", np.uint8, (n, 13, 2, 8192)),
                                   ("targets.npy", np.float32, (n, 13, 3))):
            values = np.load(files[name], mmap_mode="r", allow_pickle=False)
            if values.dtype != dtype or values.shape != shape:
                raise ValueError(f"invalid {split}/{name} shape/dtype")
            del values
        result["rows"][split] = rows
        result["bindings"].update({f"{split}/{name}": bindings[f"{split}/{name}"] for name in files})
    if include_dev:
        assert_disjoint(result["rows"]["fit"], result["rows"]["dev"])
    return result


class PreparedDataset:
    """Memory-mapped arrays. Metadata controls grouping only, never model input."""
    def __init__(self, root, split, rows, *, epoch=0, indices=None, all_variants=False, smoke_batch=None):
        if split not in ("fit", "dev"):
            raise ValueError("training dataset refuses calibration/test/external splits")
        self.root, self.split, self.rows = Path(root), split, rows
        self.epoch, self.all_variants, self.smoke_batch = epoch, all_variants, smoke_batch
        self.indices = list(range(len(rows))) if indices is None else list(indices)
        if (not self.indices or len(set(self.indices)) != len(self.indices) or
                any(type(i) is not int or not 0 <= i < len(rows) for i in self.indices)):
            raise ValueError("source indices must be nonempty, unique and within the declared split")
        if smoke_batch is not None and (
                type(smoke_batch) is not int or smoke_batch < 1 or smoke_batch > VARIANTS * len(self.indices)
                or all_variants):
            raise ValueError("smoke batch exceeds distinct source-variant capacity or conflicts with all_variants")
        self.images = np.load(self.root / split / "images.npy", mmap_mode="r", allow_pickle=False)
        self.masks = np.load(self.root / split / "masks.npy", mmap_mode="r", allow_pickle=False)
        self.targets = np.load(self.root / split / "targets.npy", mmap_mode="r", allow_pickle=False)

    def __len__(self):
        return self.smoke_batch if self.smoke_batch else len(self.indices) * (13 if self.all_variants else 1)

    def __getitem__(self, index):
        if self.smoke_batch:
            source_index = self.indices[index % len(self.indices)]
            variant = (variant_for_epoch(self.rows[source_index]["source_id"], 0) + index // len(self.indices)) % 13
        elif self.all_variants:
            source_index, variant = self.indices[index // 13], index % 13
        else:
            source_index = self.indices[index]
            variant = variant_for_epoch(self.rows[source_index]["source_id"], self.epoch)
        image = np.array(self.images[source_index, variant], dtype=np.float32)[None]
        image = (image / 255.0 * 2.0 - 1.0) * 1024.0
        mask = np.unpackbits(self.masks[source_index, variant], axis=-1, bitorder="big").reshape(2, 256, 256).astype(np.float32)
        target = np.array(self.targets[source_index, variant], dtype=np.float32)
        if not np.isfinite(target).all() or (target < 0).any() or (target > 1).any():
            raise ValueError("invalid retention target; no silent sample skipping")
        return {"images": image, "masks": mask, "targets": target,
                "source_index": source_index, "variant_index": variant}


class MetricAccumulator:
    def __init__(self):
        self.inputs = self.targets = self.finite_targets = self.empty_lungs = self.lungs = 0
        self.absolute_error = self.dice_sum = self.loss_sum = 0.0
        self.unique_sources = set()
        self.inputs_with_empty_lung = self.both_empty_inputs = 0

    def add(self, output, batch, loss):
        import torch
        pred = output["retention"].detach().float()
        target = batch["targets"].detach().float()
        finite = torch.isfinite(pred)
        self.inputs += len(pred)
        self.targets += pred.numel()
        self.finite_targets += int(finite.sum())
        self.absolute_error += float((pred[finite] - target[finite]).abs().sum())
        self.loss_sum += float(loss.detach()) * len(pred)
        predicted_masks = output["lung_logits"].detach() >= 0
        masks = batch["masks"] > .5
        intersection = (predicted_masks & masks).flatten(2).sum(-1)
        denominator = predicted_masks.flatten(2).sum(-1) + masks.flatten(2).sum(-1)
        dice = torch.where(denominator > 0, 2 * intersection.float() / denominator.clamp_min(1), torch.ones_like(denominator).float())
        self.dice_sum += float(dice.sum())
        self.lungs += dice.numel()
        self.empty_lungs += int((~predicted_masks.flatten(2).any(-1)).sum())
        present = predicted_masks.flatten(2).any(-1)
        self.inputs_with_empty_lung += int((~present.all(-1)).sum())
        self.both_empty_inputs += int((~present.any(-1)).sum())
        self.unique_sources.update(int(i) for i in batch["source_index"])

    def result(self):
        return {"inputs": self.inputs, "retention_targets": self.targets,
                "unique_sources": len(self.unique_sources),
                "mae": self.absolute_error / self.finite_targets if self.finite_targets else None,
                "finite_score_rate": self.finite_targets / self.targets if self.targets else None,
                "mask_dice": self.dice_sum / self.lungs if self.lungs else None,
                "empty_mask_rate": self.empty_lungs / self.lungs if self.lungs else None,
                "empty_masks": self.empty_lungs, "mask_denominator": self.lungs,
                "inputs_with_any_empty_lung": self.inputs_with_empty_lung,
                "inputs_with_both_empty_lungs": self.both_empty_inputs,
                "mean_loss": self.loss_sum / self.inputs if self.inputs else None}


def check_gpu(uuid):
    if not re.fullmatch(r"GPU-[a-fA-F0-9-]+", uuid):
        raise ValueError("an explicit GPU UUID is required")
    available = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
    if uuid not in [u.strip() for u in available]:
        raise ValueError("requested GPU UUID is unavailable")
    applications = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    for line in applications.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if fields and fields[0] == uuid:
            raise RuntimeError(f"requested GPU is occupied: {line}")


def implementation_hashes():
    return {"train.py": sha256(__file__), "vision.py": sha256(ROOT / "src/oa_cxr/rebuild/vision.py")}


def common_identity(args, data):
    return {"data_root": str(data["root"]), "fit_bindings": {k: v for k, v in data["bindings"].items()
            if k == "protocol.json" or k.startswith("fit/")},
            "cache_review_sha256": data["cache_review_sha256"],
            "cache_status_sha256": data["cache_status_sha256"],
            "checkpoint_sha256": args.checkpoint_sha256, "batch_size": args.batch_size,
            "gpu_uuid": args.gpu_uuid, "implementation": implementation_hashes()}


def _loader(torch, dataset, batch_size, *, workers, shuffle=False, epoch=0):
    generator = torch.Generator().manual_seed(SEED + epoch)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                      num_workers=workers, pin_memory=True, drop_last=False,
                                      generator=generator, persistent_workers=False)


def _to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if k in ("images", "masks", "targets") else v
            for k, v in batch.items()}


def run(args):
    if args.action not in ("smoke", "train") or args.batch_size not in TRAIN_BATCH_SIZES:
        raise ValueError("action must be smoke/train and batch must be an explicitly supported fixed size")
    if Path(args.output).exists():
        raise FileExistsError("smoke/training output must be fresh; no overwrite or hidden resume")
    data = validate_data(args.data, include_dev=args.action == "train")
    if sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    identity = common_identity(args, data)
    if args.action == "train":
        if not args.smoke_dir:
            raise ValueError("formal training requires an accepted real GPU smoke")
        smoke = read_json(Path(args.smoke_dir) / "status.json")
        if smoke.get("status") != "passed" or smoke.get("action") != "smoke" or smoke.get("identity") != identity:
            raise ValueError("GPU smoke is missing, failed, or bound to different data/model/batch/implementation")
    check_gpu(args.gpu_uuid)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    import torch
    from oa_cxr.rebuild.vision import from_xrv_densenet_checkpoint, multitask_loss
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected GPU must support real CUDA BF16")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    state = {"status": "running", "action": args.action, "started_at": started, "identity": identity,
             "fit_unique_source_images": len(data["rows"]["fit"]), "pid": os.getpid(), "optimizer_steps": 0}
    atomic_json(output / "status.json", state)
    try:
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        protocol = {"version": "large-image-retention-training-v1", "identity": identity,
                    "seed": SEED, "findings": FINDINGS, "epochs": EPOCHS,
                    "variant_schedule": "(int(stable_hash(source_id),16)+zero_based_epoch)%13",
                    "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 1e-4},
                    "optimization_plan": optimization_plan(len(data["rows"]["fit"]), args.batch_size),
                    "gradient_clip_l2_norm": 5.0,
                    "precision": "CUDA BF16 autocast; float32 master parameters",
                    "retention_loss": "SmoothL1 beta=0.1", "segmentation_loss": "0.25*(BCE+0.5*DiceLoss)",
                    "dev_selection": "first 512 by stable_hash(['rebuild-dev',17,source_id]), all13 variants",
                    "best_selection": "minimum dev all-query MAE; ties keep earlier epoch",
                    "evaluation_header": {"reads": ["fit", "dev"] if args.action == "train" else ["fit"],
                                          "calibration_test_external_access": False,
                                          "heldout_results_not_used_for_selection": True},
                    "data_bindings": data["bindings"], "torch": torch.__version__,
                    "numpy": np.__version__, "gpu_name": torch.cuda.get_device_name(0),
                    "parameter_update_audit": "initial versus dev-selected best: encoder conv0 and all new decoder/embedding/head parameters; actual optimizer-step counters",
                    "segmentation_metrics": "all annotated lungs; both-empty Dice=1, empty predictions reported separately"}
        atomic_json(output / "protocol.json", protocol)
        model = from_xrv_densenet_checkpoint(args.checkpoint, args.checkpoint_sha256).to("cuda")
        atomic_json(output / "initialization.json", model.initialization_provenance)
        torch.cuda.reset_peak_memory_stats()
        if args.action == "smoke":
            # Two variants/source for batch32; four for larger supported sizes.
            # Dynamic source count prevents wrapping beyond 13 unique variants.
            smoke_sources = smoke_source_count(args.batch_size)
            dataset = PreparedDataset(data["root"], "fit", data["rows"]["fit"],
                                      indices=range(smoke_sources), smoke_batch=args.batch_size)
            batch = _to_device(next(iter(_loader(torch, dataset, args.batch_size, workers=0))), "cuda")
            smoke_pairs = set(zip(batch["source_index"].tolist(), batch["variant_index"].tolist()))
            if len(smoke_pairs) != args.batch_size or len(set(batch["source_index"].tolist())) != smoke_sources:
                raise ValueError("smoke batch contains repeated inputs or an incorrect unique-source count")
            before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(batch["images"])
                losses = multitask_loss(prediction, batch["targets"], batch["masks"])
            losses["loss"].backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                raise ValueError("smoke backward produced missing/nonfinite gradients")
            if not any(g.abs().sum() > 0 for g in gradients):
                raise ValueError("smoke backward produced only zero gradients")
            if any(not torch.equal(before[k], v.detach().cpu()) for k, v in model.state_dict().items()):
                raise ValueError("smoke unexpectedly changed parameters/buffers")
            metrics = MetricAccumulator(); metrics.add(prediction, batch, losses["loss"])
            state.update(status="passed", smoke_metrics=metrics.result(), optimizer_steps=0,
                         checkpoint_saved=False, weights_unchanged=True,
                         smoke_source_images=smoke_sources, smoke_unique_input_pairs=len(smoke_pairs))
        else:
            initial_parameters = capture_audit_parameters(model)
            initial_parameter_path = output / "initial_audit_parameters.npz"
            np.savez(initial_parameter_path, **initial_parameters)
            initial_parameter_sha256 = sha256(initial_parameter_path)
            dev_indices = select_dev(data["rows"]["dev"])
            atomic_json(output / "dev_cohort.json", {"row_indices": dev_indices,
                        "source_ids": [data["rows"]["dev"][i]["source_id"] for i in dev_indices],
                        "selection_before_first_epoch": True, "inputs": len(dev_indices) * 13})
            dev_dataset = PreparedDataset(data["root"], "dev", data["rows"]["dev"], indices=dev_indices, all_variants=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            best, best_epoch = float("inf"), None
            seen_sources = set()
            for epoch in range(EPOCHS):
                epoch_start_steps = state["optimizer_steps"]
                model.train(); metrics = MetricAccumulator()
                dataset = PreparedDataset(data["root"], "fit", data["rows"]["fit"], epoch=epoch)
                state.update(epoch=epoch + 1, phase="fit", epoch_inputs=0)
                atomic_json(output / "status.json", state)
                for step, batch in enumerate(_loader(torch, dataset, args.batch_size, workers=args.workers, shuffle=True, epoch=epoch)):
                    batch = _to_device(batch, "cuda")
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        prediction = model(batch["images"])
                        losses = multitask_loss(prediction, batch["targets"], batch["masks"])
                    if not torch.isfinite(losses["loss"]):
                        raise ValueError("nonfinite training loss; no hidden retry")
                    losses["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                    optimizer.step()
                    state["optimizer_steps"] += 1
                    metrics.add(prediction, batch, losses["loss"])
                    seen_sources.update(metrics.unique_sources)
                    if step % 50 == 0:
                        state.update(epoch_inputs=metrics.inputs, fit_metrics=metrics.result(),
                                     actual_unique_training_sources=len(seen_sources), updated_at=time.time())
                        atomic_json(output / "status.json", state)
                        print(json.dumps({"epoch": epoch + 1, "step": step, **metrics.result()}), flush=True)
                if metrics.inputs != len(data["rows"]["fit"]) or len(metrics.unique_sources) != len(data["rows"]["fit"]):
                    raise ValueError("epoch did not consume every unique fit source exactly once")
                if state["optimizer_steps"] - epoch_start_steps != protocol["optimization_plan"]["expected_optimizer_steps_per_epoch"]:
                    raise ValueError("actual optimizer steps differ from the explicit no-accumulation batch protocol")
                state.update(phase="dev"); atomic_json(output / "status.json", state)
                model.eval(); development = MetricAccumulator()
                with torch.inference_mode():
                    for batch in _loader(torch, dev_dataset, args.batch_size, workers=args.workers):
                        batch = _to_device(batch, "cuda")
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            prediction = model(batch["images"])
                            losses = multitask_loss(prediction, batch["targets"], batch["masks"])
                        development.add(prediction, batch, losses["loss"])
                if development.inputs != len(dev_indices) * 13:
                    raise ValueError("incomplete fixed development population")
                dev_metrics = development.result()
                if dev_metrics["finite_score_rate"] != 1.0:
                    raise ValueError("nonfinite development score; never select on a reduced subset")
                checkpoint = output / f"epoch_{epoch + 1:02d}.pt"
                torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                            "epoch": epoch + 1, "protocol_sha256": sha256(output / "protocol.json"),
                            "findings": FINDINGS, "optimizer_steps": state["optimizer_steps"]}, checkpoint)
                record = {"epoch": epoch + 1, "fit": metrics.result(), "dev": dev_metrics,
                          "checkpoint": checkpoint.name, "checkpoint_sha256": sha256(checkpoint),
                          "optimizer_steps": state["optimizer_steps"],
                          "optimizer_steps_in_epoch": state["optimizer_steps"] - epoch_start_steps}
                atomic_json(output / f"epoch_{epoch + 1:02d}.json", record)
                if dev_metrics["mae"] < best:
                    best, best_epoch = dev_metrics["mae"], epoch + 1
                    atomic_json(output / "best.json", {**record, "selection": "dev MAE only"})
                state.update(completed_epochs=epoch + 1, best_epoch=best_epoch, best_dev_mae=best,
                             actual_unique_training_sources=len(seen_sources), last_epoch=record)
                atomic_json(output / "status.json", state)
                print(json.dumps({"completed_epoch": epoch + 1, "dev": dev_metrics, "best_epoch": best_epoch}), flush=True)
            best_record = read_json(output / "best.json")
            if state["optimizer_steps"] != protocol["optimization_plan"]["expected_optimizer_steps_total"]:
                raise ValueError("total actual optimizer steps differ from the fixed training plan")
            selected_path = output / best_record["checkpoint"]
            if (sha256(selected_path) != best_record["checkpoint_sha256"] or
                    sha256(initial_parameter_path) != initial_parameter_sha256):
                raise ValueError("selected checkpoint or immutable initial parameter snapshot changed")
            selected = torch.load(selected_path, map_location="cpu", weights_only=True)
            if (selected.get("epoch") != best_record["epoch"] or
                    selected.get("optimizer_steps") != best_record["optimizer_steps"] or
                    selected.get("protocol_sha256") != sha256(output / "protocol.json")):
                raise ValueError("selected checkpoint step/epoch/protocol identity differs")
            evidence = parameter_update_evidence(initial_parameters, selected["model_state"],
                optimizer_steps=state["optimizer_steps"], selected_optimizer_steps=selected["optimizer_steps"])
            evidence.update(initial_audit_parameters_sha256=initial_parameter_sha256,
                            selected_checkpoint=selected_path.name,
                            selected_checkpoint_sha256=best_record["checkpoint_sha256"],
                            selected_epoch=best_record["epoch"], protocol_sha256=sha256(output / "protocol.json"))
            atomic_json(output / "finetuning_evidence.json", evidence)
            del selected, initial_parameters
            state.update(finetuning_evidence_status=evidence["status"], output_sha256={
                "initial_audit_parameters.npz": initial_parameter_sha256,
                "finetuning_evidence.json": sha256(output / "finetuning_evidence.json")})
            if evidence["status"] != "passed":
                raise ValueError("selected model has unchanged audited components; finetuning evidence failed")
            state.update(status="completed", phase="finished", total_fit_inputs=len(data["rows"]["fit"]) * EPOCHS,
                         all_13_variants_seen_per_fit_source=True)
        torch.cuda.synchronize()
        state.update(finished_at=time.time(), elapsed_seconds=time.time() - started,
                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        atomic_json(output / "status.json", state)
        return state
    except BaseException as error:
        state.update(status="failed", failed_at=time.time(), error=repr(error), traceback=traceback.format_exc())
        atomic_json(output / "status.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("smoke", "train"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--batch-size", type=int, choices=TRAIN_BATCH_SIZES, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke-dir")
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("workers must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()

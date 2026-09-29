"""Export frozen new-model predictions/features, then attach labels separately.

export reads images and identities only. attach-labels is a separate CPU action:
it joins independent targets and scores already saved binary lung predictions.
No heldout labels influence checkpoint selection, features, or predictions.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rebuild_training_contract", HERE / "train.py")
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)
sha256, read_json, atomic_json = training.sha256, training.read_json, training.atomic_json
stable_hash, FINDINGS = training.stable_hash, training.FINDINGS


def cache_inputs(root, split, *, labels=False):
    if not re.fullmatch(r"fit|dev|calibration|test|external(?:_[a-z0-9_]+)?", split):
        raise ValueError("explicit supported cache split required")
    root = Path(root).resolve(strict=True)
    state = read_json(root / "status.json")
    from oa_cxr.rebuild.cache_validation import validation_receipt
    if state.get("status") != "completed":
        raise ValueError("Completed cache required")
    validation_receipt(root)
    bindings = state.get("files", {})
    training._bound_file(root, "protocol.json", bindings)
    protocol = read_json(root / "protocol.json")
    if (protocol.get("findings") != FINDINGS or protocol.get("variants") != 13 or
            protocol.get("data_protocol", {}).get("normalization") != "(gray/255*2-1)*1024"):
        raise ValueError("cache image/finding protocol differs")
    names = ("sources.jsonl", "targets.npy", "masks.npy") if labels else ("sources.jsonl", "images.npy")
    paths = {name: training._bound_file(root, f"{split}/{name}", bindings) for name in names}
    rows = [json.loads(line) for line in paths["sources.jsonl"].read_text(encoding="utf-8").splitlines() if line.strip()]
    training.validate_source_rows(rows, split, minimum_fit_sources=10000)
    n = len(rows)
    if state.get("splits", {}).get(split) != {"sources": n, "inputs": n * 13, "targets": n * 39}:
        raise ValueError("cache split counts differ")
    contracts = {"images.npy": (np.uint8, (n, 13, 256, 256)),
                 "masks.npy": (np.uint8, (n, 13, 2, 8192)), "targets.npy": (np.float32, (n, 13, 3))}
    for name in names:
        if name == "sources.jsonl": continue
        array = np.load(paths[name], mmap_mode="r", allow_pickle=False)
        dtype, shape = contracts[name]
        if array.dtype != dtype or array.shape != shape:
            raise ValueError(f"invalid {name} shape/dtype")
        del array
    return {"root": root, "split": split, "rows": rows, "paths": paths,
            "protocol": protocol, "cache_status_sha256": sha256(root / "status.json"),
            "bindings": {f"{split}/{name}": bindings[f"{split}/{name}"] for name in names},
            "target_values_loaded": labels}


def selected_checkpoint(train_dir):
    """Reproduce the frozen development-only choice without any test access."""
    root = Path(train_dir).resolve(strict=True)
    state, protocol, best = (read_json(root / name) for name in ("status.json", "protocol.json", "best.json"))
    if state.get("status") != "completed" or state.get("action") != "train" or state.get("completed_epochs") != 13:
        raise ValueError("only a fully completed new 13-epoch training run can be exported")
    if (state.get("fit_unique_source_images", 0) < 10000 or
            state.get("actual_unique_training_sources") != state.get("fit_unique_source_images")):
        raise ValueError("completed fitting must document at least 10000 actual unique source images")
    if protocol.get("version") != "large-image-retention-training-v1" or protocol.get("findings") != FINDINGS:
        raise ValueError("unsupported new training contract; legacy model import is prohibited")
    epochs = [read_json(root / f"epoch_{i:02d}.json") for i in range(1, 14)]
    for i, entry in enumerate(epochs, 1):
        if (entry.get("epoch") != i or not np.isfinite(entry.get("dev", {}).get("mae", np.nan)) or
                entry["dev"].get("finite_score_rate") != 1):
            raise ValueError("incomplete or invalid development checkpoint history")
    winner = min(epochs, key=lambda entry: (entry["dev"]["mae"], entry["epoch"]))
    if (best != {**winner, "selection": "dev MAE only"} or state.get("best_epoch") != winner["epoch"] or
            protocol.get("identity", {}).get("implementation", {}).get("vision.py") !=
            sha256(training.ROOT / "src/oa_cxr/rebuild/vision.py")):
        raise ValueError("best checkpoint selection or current vision implementation differs")
    path = (root / winner["checkpoint"]).resolve(strict=True)
    if not path.is_relative_to(root) or sha256(path) != winner["checkpoint_sha256"]:
        raise ValueError("selected checkpoint path/SHA mismatch")
    return {"path": path, "sha256": winner["checkpoint_sha256"], "epoch": winner["epoch"],
            "protocol": protocol, "protocol_sha256": sha256(root / "protocol.json"),
            "best_sha256": sha256(root / "best.json"), "training_root": str(root)}


def query_records(rows, split):
    for row in rows:
        for variant in range(13):
            identity = [row["dataset"], row["source_id"], variant]
            input_id = stable_hash(["rebuild-input-v1", *identity])
            for finding in FINDINGS:
                yield {"row_id": stable_hash(["rebuild-query-v1", *identity, finding]), "input_id": input_id,
                       "source_id": row["source_id"], "split_group_id": row["split_group_id"],
                       "dataset": row["dataset"], "split": "external" if split.startswith("external") else split,
                       "cache_split": split, "finding": finding, "source_image_sha256": row["source_image_sha256"],
                       "source_row_index": row["row_index"], "variant_index": variant}


def feature_names(model):
    channels, width = model.encoder_channels, model.lung_head.in_channels
    names = [f"image_encoder_{i:04d}" for i in range(channels)]
    names += [f"finding_region_{i:03d}" for i in range(width)]
    names += [f"lung_{lung}_region_{i:03d}" for lung in (0, 1) for i in range(width)]
    names += [f"lung_{lung}_{name}" for lung in (0, 1)
              for name in ("soft_mass", "maximum", "top_contact", "bottom_contact", "left_contact", "right_contact", "hard_present")]
    names += ["finding_region_mass", "lung_0_region_mass", "lung_1_region_mass", "regional_branch_used"]
    names += [f"finding_embedding_{i:03d}" for i in range(model.finding_embeddings.embedding_dim)]
    if len(names) != model.retention_head[0].in_features:
        raise ValueError("feature names do not describe the actual first Linear input")
    return names + [f"finding_{name}" for name in FINDINGS]


def make_manifest(samples_path, features_path, names, protocol_sha256):
    return {"schema": "oa-cxr-tabular-dataset-v1", "target": "anatomical_retention",
            "label_free_features": True, "feature_protocol_sha256": protocol_sha256,
            "feature_names": names, "samples": {"path": samples_path.name, "sha256": sha256(samples_path)},
            "features": {"path": features_path.name, "sha256": sha256(features_path)}}


class ImageOnlyDataset:
    def __init__(self, path):
        self.images = np.load(path, mmap_mode="r", allow_pickle=False)

    def __len__(self):
        return self.images.shape[0] * 13

    def __getitem__(self, index):
        image = np.array(self.images[index // 13, index % 13], dtype=np.float32)[None]
        return (image / 255.0 * 2.0 - 1.0) * 1024.0, index


def load_model(checkpoint):
    import torch
    import torchxrayvision as xrv
    from oa_cxr.rebuild.vision import DirectAnatomyRetention, XrvDenseNetEncoder
    if torch.__version__ != checkpoint["protocol"].get("torch"):
        raise ValueError("Torch version differs from frozen training; explicit portability validation required")
    payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
    if (payload.get("epoch") != checkpoint["epoch"] or payload.get("protocol_sha256") != checkpoint["protocol_sha256"] or
            payload.get("findings") != FINDINGS):
        raise ValueError("checkpoint payload is not bound to the frozen training protocol")
    base = xrv.models.DenseNet(weights=None, op_threshs=None, apply_sigmoid=False)
    model = DirectAnatomyRetention(XrvDenseNetEncoder(base.features), base.classifier.in_features)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.eval()


def export(args):
    if Path(args.output).exists(): raise FileExistsError("fresh export output required")
    cache = cache_inputs(args.data, args.split, labels=False)
    checkpoint = selected_checkpoint(args.train_dir)
    training.check_gpu(args.gpu_uuid)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("real CUDA BF16 GPU required")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "action": "export", "started_at": time.time(), "target_values_loaded": False,
             "sources": len(cache["rows"]), "inputs": len(cache["rows"]) * 13,
             "queries": len(cache["rows"]) * 39, "completed_inputs": 0, "cache_split": args.split}
    atomic_json(out / "status.json", state)
    try:
        model = load_model(checkpoint).to("cuda")
        names = feature_names(model)
        feature_protocol = {"version": "trained-retention-head-input-v1", "model_checkpoint_sha256": checkpoint["sha256"],
                            "training_protocol_sha256": checkpoint["protocol_sha256"], "feature_names": names,
                            "input_data_protocol": cache["protocol"]["data_protocol"], "inference_batch_size": args.batch_size,
                            "precision": "CUDA BF16 autocast; exact pre-Linear values exported as float32",
                            "readout": "retention_head[0] forward_pre_hook; three explicit finding one-hot columns appended",
                            "model_mode": "eval; regional dropout disabled", "targets_or_masks_read": False,
                            "implementation": {"export.py": sha256(__file__), "vision.py": checkpoint["protocol"]["identity"]["implementation"]["vision.py"]}}
        protocol_sha = stable_hash(feature_protocol)
        atomic_json(out / "feature_protocol.json", {**feature_protocol, "artifact_sha256": protocol_sha})
        provenance = {"checkpoint_sha256": checkpoint["sha256"], "checkpoint_epoch": checkpoint["epoch"],
                      "training_root": checkpoint["training_root"], "selection": "frozen dev-only best; no external selection",
                      "cache_root": str(cache["root"]), "cache_split": args.split, "cache_bindings": cache["bindings"],
                      "cache_status_sha256": cache["cache_status_sha256"], "target_values_loaded": False,
                      "feature_protocol_sha256": protocol_sha, "gpu_uuid": args.gpu_uuid, "torch": torch.__version__}
        atomic_json(out / "provenance.json", provenance)
        rows = list(query_records(cache["rows"], args.split))
        with (out / "samples.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows: stream.write(json.dumps(row, sort_keys=True) + "\n")
        X = np.lib.format.open_memmap(out / "feature_work.npy", mode="w+", dtype=np.float32,
                                     shape=(len(rows), len(names)))
        packed = np.lib.format.open_memmap(out / "mask_work.npy", mode="w+", dtype=np.uint8,
                                          shape=(state["inputs"], 2, 8192))
        available = np.zeros(len(rows), dtype=bool)
        mask_available = np.zeros(state["inputs"], dtype=bool)
        captured = {}
        hook = model.retention_head[0].register_forward_pre_hook(
            lambda module, values: captured.update(head_input=values[0].detach().float()))
        loader = torch.utils.data.DataLoader(ImageOnlyDataset(cache["paths"]["images.npy"]),
                                            batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                            drop_last=False, pin_memory=True)
        empty_lungs = 0
        with (out / "predictions.jsonl").open("x", encoding="utf-8") as stream, torch.inference_mode():
            for step, (images, indices) in enumerate(loader):
                indices = indices.numpy()
                captured.clear()
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        prediction = model(images.to("cuda", non_blocking=True))
                    head = captured["head_input"].cpu().numpy()
                    scores = prediction["retention"].float().cpu().numpy()
                    if head.shape != (len(indices), 3, len(names) - 3) or not np.isfinite(head).all() or not np.isfinite(scores).all():
                        raise ValueError("nonfinite/misaligned model or actual head features")
                    onehot = np.broadcast_to(np.eye(3, dtype=np.float32), (len(indices), 3, 3))
                    vectors = np.concatenate((head, onehot), axis=-1).reshape(-1, len(names))
                    query_indices = (indices[:, None] * 3 + np.arange(3)[None]).ravel()
                    X[query_indices] = vectors; available[query_indices] = True
                    masks = (prediction["lung_logits"] >= 0).cpu().numpy()
                    packed[indices] = np.packbits(masks.reshape(len(indices), 2, -1), axis=-1, bitorder="big")
                    mask_available[indices] = True
                    present = masks.reshape(len(indices), 2, -1).any(-1)
                    empty_lungs += int((~present).sum())
                    for k, index in enumerate(indices):
                        for finding_index in range(3):
                            row = rows[int(index) * 3 + finding_index]
                            record = {**row, "method": "direct_anatomy_retention", "status": "success",
                                      "score": float(scores[k, finding_index]), "failure_reason": None,
                                      "lung_hard_present": present[k].tolist(),
                                      "finding_region_valid": prediction["finding_region_valid"][k, finding_index].cpu().tolist()}
                            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                except Exception as error:
                    # Preserve every affected input/finding as a real failed
                    # batch attempt, stop, and leave unattempted rows in the
                    # declared sample denominator. No retry or fabricated mask.
                    for index in indices:
                        for finding_index in range(3):
                            row = rows[int(index) * 3 + finding_index]
                            stream.write(json.dumps({**row, "method": "direct_anatomy_retention", "status": "failed",
                                "score": None, "failure_reason": f"batch_inference_failed:{type(error).__name__}:{error}"}) + "\n")
                    stream.flush()
                    state.update(failed_batch_inputs=[int(i) for i in indices], failure_policy="stop; no hidden retry")
                    raise
                state.update(completed_inputs=state["completed_inputs"] + len(indices), empty_predicted_lungs=empty_lungs)
                if step % 50 == 0:
                    atomic_json(out / "status.json", state); stream.flush()
                    print(json.dumps({"completed_inputs": state["completed_inputs"], "inputs": state["inputs"]}), flush=True)
        hook.remove()
        if state["completed_inputs"] != state["inputs"] or not available.all() or not mask_available.all():
            raise ValueError("incomplete input accounting")
        X.flush(); packed.flush()
        ids = np.asarray([r["row_id"] for r in rows], dtype="U64")
        np.savez(out / "features.npz", X=X, row_ids=ids, available=available)
        np.savez_compressed(out / "predicted_masks.npz", packed=packed, available=mask_available)
        del X, packed
        (out / "feature_work.npy").unlink(); (out / "mask_work.npy").unlink()
        manifest = make_manifest(out / "samples.jsonl", out / "features.npz", names, protocol_sha)
        atomic_json(out / "manifest.json", manifest)
        state.update(status="completed", finished_at=time.time(), finite_score_rate=float(available.mean()),
                     empty_mask_rate=empty_lungs / (state["inputs"] * 2), feature_dimension=len(names),
                     output_sha256={n: sha256(out / n) for n in ("predictions.jsonl", "samples.jsonl", "features.npz",
                       "predicted_masks.npz", "manifest.json", "provenance.json", "feature_protocol.json")})
        atomic_json(out / "status.json", state)
        return state
    except BaseException as error:
        state.update(status="failed", failed_at=time.time(), error=repr(error), traceback=traceback.format_exc())
        atomic_json(out / "status.json", state)
        raise


def packed_dice(predicted, target):
    """Exact binary Dice for each [input,lung], without uint8 overflow."""
    if predicted.dtype != np.uint8 or target.dtype != np.uint8 or predicted.shape != target.shape:
        raise ValueError("aligned uint8 packed predicted and target masks required")
    counts = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    intersection = counts[np.bitwise_and(predicted, target)].sum(-1, dtype=np.int64)
    denominator = counts[predicted].sum(-1, dtype=np.int64) + counts[target].sum(-1, dtype=np.int64)
    dice = np.ones(denominator.shape, dtype=np.float64)
    np.divide(2 * intersection, denominator, out=dice, where=denominator > 0)
    return dice


class SegmentationAudit:
    """Separate legitimate cropped-empty GT, erroneous empty masks and failures."""
    CHANNELS = ("image_left", "image_right")
    COUNTS = ("declared", "scored", "unavailable", "gt_nonempty", "scored_gt_nonempty",
              "unavailable_gt_nonempty", "gt_empty", "scored_gt_empty", "unavailable_gt_empty",
              "predicted_empty", "false_empty", "false_nonempty", "correct_empty")
    SUMS = ("dice_sum", "iou_sum", "gt_nonempty_dice_sum", "gt_nonempty_iou_sum")

    def __init__(self):
        self.groups = {}

    def add(self, predicted, target, available, datasets):
        if (predicted.dtype != np.uint8 or target.dtype != np.uint8 or predicted.shape != target.shape
                or predicted.ndim != 3 or predicted.shape[1] != 2 or
                available.dtype != np.bool_ or available.shape != (len(target),) or len(datasets) != len(target)):
            raise ValueError("aligned packed [input,2,bytes] masks, boolean availability and dataset IDs required")
        if any(not isinstance(d, str) or not d for d in datasets):
            raise ValueError("explicit dataset identity is required for every presented input")
        lookup = np.asarray([i.bit_count() for i in range(256)], dtype=np.uint8)
        p = lookup[predicted].sum(-1, dtype=np.int64)
        t = lookup[target].sum(-1, dtype=np.int64)
        intersection = lookup[np.bitwise_and(predicted, target)].sum(-1, dtype=np.int64)
        dice = np.ones(p.shape, dtype=np.float64); iou = np.ones_like(dice)
        np.divide(2 * intersection, p + t, out=dice, where=p + t > 0)
        np.divide(intersection, p + t - intersection, out=iou, where=p + t - intersection > 0)
        datasets = np.asarray(datasets)
        selections = [(None, np.ones(len(target), dtype=bool))]
        selections += [(name, datasets == name) for name in sorted(set(datasets))]
        for name, selected in selections:
            entries = self.groups.setdefault(name, [{key: 0 for key in self.COUNTS + self.SUMS} for _ in range(2)])
            for lung, entry in enumerate(entries):
                active = selected & available
                positive = t[:, lung] > 0; empty = p[:, lung] == 0
                flags = dict(declared=selected, scored=active, unavailable=selected & ~available,
                    gt_nonempty=selected & positive, scored_gt_nonempty=active & positive,
                    unavailable_gt_nonempty=selected & ~available & positive,
                    gt_empty=selected & ~positive, scored_gt_empty=active & ~positive,
                    unavailable_gt_empty=selected & ~available & ~positive,
                    predicted_empty=active & empty, false_empty=active & positive & empty,
                    false_nonempty=active & ~positive & ~empty, correct_empty=active & ~positive & empty)
                for key, mask in flags.items(): entry[key] += int(mask.sum())
                entry["dice_sum"] += float(dice[active, lung].sum())
                entry["iou_sum"] += float(iou[active, lung].sum())
                entry["gt_nonempty_dice_sum"] += float(dice[active & positive, lung].sum())
                entry["gt_nonempty_iou_sum"] += float(iou[active & positive, lung].sum())

    @classmethod
    def _result(cls, entry):
        divide = lambda value, count: float(value / count) if count else None
        return {**{key + "_count": int(entry[key]) for key in cls.COUNTS},
            "mask_dice": divide(entry["dice_sum"], entry["scored"]),
            "mask_iou": divide(entry["iou_sum"], entry["scored"]),
            "empty_prediction_rate": divide(entry["predicted_empty"], entry["scored"]),
            "false_empty_rate": divide(entry["false_empty"], entry["scored_gt_nonempty"]),
            "false_empty_rate_denominator": "scored_gt_nonempty_count",
            "false_empty_or_unavailable_rate": divide(entry["false_empty"] + entry["unavailable_gt_nonempty"], entry["gt_nonempty"]),
            "false_nonempty_rate": divide(entry["false_nonempty"], entry["scored_gt_empty"]),
            "false_nonempty_rate_denominator": "scored_gt_empty_count",
            "nonempty_gt_dice": divide(entry["gt_nonempty_dice_sum"], entry["scored_gt_nonempty"]),
            "nonempty_gt_iou": divide(entry["gt_nonempty_iou_sum"], entry["scored_gt_nonempty"])}

    def result(self):
        def group(entries):
            combined = {key: sum(entry[key] for entry in entries) for key in self.COUNTS + self.SUMS}
            return {**self._result(combined), "per_lung":
                    {name: self._result(entry) for name, entry in zip(self.CHANNELS, entries)}}
        return {"overall": group(self.groups[None]) if None in self.groups else None,
                "by_dataset": {name: group(entries) for name, entries in sorted(
                    ((key, value) for key, value in self.groups.items() if key is not None))},
                "channel_semantics": "image-left/image-right by source mask x-centroid; not asserted patient laterality",
                "missing_prediction_policy": "unavailable is not an empty mask; report separately and in conservative failure-inclusive rate",
                "both_empty_dice_and_iou": 1,
                "clinical_segmentation_success_established": False}


def attach_labels(args):
    out, exported = Path(args.output), Path(args.export_dir).resolve(strict=True)
    if out.exists(): raise FileExistsError("fresh label bridge output required")
    cache = cache_inputs(args.data, args.split, labels=True)
    receipt = read_json(exported / "status.json")
    provenance = read_json(exported / "provenance.json")
    if (receipt.get("status") != "completed" or receipt.get("action") != "export" or
            provenance.get("cache_root") != str(cache["root"]) or provenance.get("cache_split") != args.split or
            provenance.get("cache_status_sha256") != cache["cache_status_sha256"]):
        raise ValueError("completed predictions do not belong to this exact cache/split")
    for name, digest in receipt["output_sha256"].items():
        if sha256(exported / name) != digest: raise ValueError("exported bytes changed")
    samples = [json.loads(s) for s in (exported / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    if samples != list(query_records(cache["rows"], args.split)):
        raise ValueError("label/query identities differ")
    targets = np.load(cache["paths"]["targets.npy"], mmap_mode="r", allow_pickle=False)
    y = np.asarray(targets).reshape(-1)
    if not np.isfinite(y).all() or (y < 0).any() or (y > 1).any(): raise ValueError("invalid independent target")
    out.mkdir(parents=True)
    state = {"status": "running", "action": "attach-labels", "started_at": time.time(),
             "prediction_rerun": False, "model_selection_performed": False, "target_values_loaded": True}
    atomic_json(out / "status.json", state)
    try:
        with np.load(exported / "features.npz", allow_pickle=False) as features:
            X, row_ids, available = features["X"], features["row_ids"], features["available"]
        if row_ids.tolist() != [r["row_id"] for r in samples] or X.shape[0] != len(y):
            raise ValueError("feature/label row identity mismatch")
        np.savez(out / "labels.npz", y=y, row_ids=row_ids)
        np.savez(out / "features.npz", X=X, row_ids=row_ids, available=available, y=y)
        del X
        shutil.copyfile(exported / "samples.jsonl", out / "samples.jsonl")
        manifest = read_json(exported / "manifest.json")
        atomic_json(out / "manifest.json", make_manifest(out / "samples.jsonl", out / "features.npz",
                    manifest["feature_names"], manifest["feature_protocol_sha256"]))
        ground_truth = np.load(cache["paths"]["masks.npy"], mmap_mode="r", allow_pickle=False).reshape(-1, 2, 8192)
        with np.load(exported / "predicted_masks.npz", allow_pickle=False) as blob:
            predicted, mask_available = blob["packed"], blob["available"]
        if predicted.shape != ground_truth.shape or mask_available.shape != (len(ground_truth),):
            raise ValueError("predicted mask identity/count mismatch")
        segmentation = SegmentationAudit()
        input_datasets = np.repeat([row["dataset"] for row in cache["rows"]], 13)
        for start in range(0, len(ground_truth), 512):
            segmentation.add(predicted[start:start + 512], ground_truth[start:start + 512],
                             mask_available[start:start + 512], input_datasets[start:start + 512])
        segmentation_result = segmentation.result()
        aggregate = segmentation_result["overall"]
        atomic_json(out / "segmentation_evaluation.json", {
            "mask_dice": aggregate["mask_dice"], "scored_lungs": aggregate["scored_count"],
            "declared_lungs": len(ground_truth) * 2, "unavailable_inputs": int((~mask_available).sum()),
            "both_empty_dice": 1, "empty_prediction_rate": receipt["empty_mask_rate"],
            "ground_truth_stratified": segmentation_result,
            "repair_acceptance": "finite retention scores alone do not establish repaired segmentation; assess per-dataset false-empty and nonempty-GT Dice/IoU",
            "annotation_types": sorted({r.get("annotation_type", "unspecified") for r in cache["rows"]}),
            "scope": "auxiliary segmentation against independent presented masks; not clinical validation"})
        atomic_json(out / "provenance.json", {"source_export_status_sha256": sha256(exported / "status.json"),
                    "independent_label_bindings": cache["bindings"], "label_free_features_preserved": True,
                    "feature_protocol_sha256": manifest["feature_protocol_sha256"], "prediction_rerun": False})
        state.update(status="completed", finished_at=time.time(), queries=len(y),
                     output_sha256={n: sha256(out / n) for n in ("labels.npz", "features.npz", "samples.jsonl",
                                               "manifest.json", "segmentation_evaluation.json", "provenance.json")})
        atomic_json(out / "status.json", state)
        return state
    except BaseException as error:
        state.update(status="failed", failed_at=time.time(), error=repr(error), traceback=traceback.format_exc())
        atomic_json(out / "status.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("export", "attach-labels"), required=True)
    parser.add_argument("--data", required=True); parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--train-dir")
    parser.add_argument("--gpu-uuid"); parser.add_argument("--batch-size", type=int, choices=(32, 64), default=32)
    parser.add_argument("--workers", type=int, default=4); parser.add_argument("--export-dir")
    arguments = parser.parse_args()
    if arguments.action == "export" and (not arguments.train_dir or not arguments.gpu_uuid):
        parser.error("export requires --train-dir and --gpu-uuid")
    if arguments.action == "attach-labels" and not arguments.export_dir:
        parser.error("attach-labels requires --export-dir")
    if arguments.workers < 0: parser.error("workers must be nonnegative")
    print(json.dumps(export(arguments) if arguments.action == "export" else attach_labels(arguments)), flush=True)

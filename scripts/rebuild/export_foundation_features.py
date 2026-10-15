"""Export official frozen image features; attach existing truth in a separate action."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from contextlib import nullcontext

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from baselines import foundation_encoders as encoders

spec = importlib.util.spec_from_file_location("foundation_original_export", ROOT / "scripts/rebuild/export.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
sha256, atomic_json, read_json = original.sha256, original.atomic_json, original.read_json
FINDINGS = original.FINDINGS
ALLOWED_GPUS = ("GPU-b3971740-75c4-b619-ec3b-d5b16d158389", "GPU-c51f1b05-61b8-753d-271f-b937bd7eb108")


class ImageOnlyDataset:
    def __init__(self, path, method):
        self.path, self.method = str(path), method
        self.images = np.load(path, mmap_mode="r", allow_pickle=False)
        if self.images.dtype != np.uint8 or self.images.ndim != 4 or self.images.shape[1:] != (13, 256, 256):
            raise ValueError("accepted presented-image cache array required")

    def __len__(self):
        return self.images.shape[0] * 13

    def __getitem__(self, index):
        return encoders.preprocess_image(self.images[index // 13, index % 13], self.method), index


def feature_protocol(method, assets_sha256):
    # No split, path or outcome enters this identity: one reusable feature map.
    return {"version": "foundation-image-only-retention-adapter-v1", "method": method,
            "assets_sha256": assets_sha256, "preprocessing": encoders.preprocessing_policy(method),
            "feature_names": encoders.feature_names(method, FINDINGS), "frozen_encoder": True,
            "model_mode": "eval", "targets_or_masks_read": False, "finding_encoding": FINDINGS,
            "precision": "CUDA BF16 autocast; normalized/readout features float32",
            "inference_batch_size": 128,
            "implementation": {"foundation_encoders.py": sha256(encoders.__file__),
                               "export_foundation_features.py": sha256(__file__),
                               "original_export.py": sha256(original.__file__),
                               "original_train.py": sha256(original.training.__file__)}}


def export(args):
    if args.batch_size != 128 or args.workers != 4:
        raise ValueError("fixed batch128/workers4 required; any explicitly selected idle GPU is accepted")
    if Path(args.output).exists():
        raise FileExistsError("fresh feature export output required")
    assets = encoders.verify_assets(args.assets, args.assets_sha256, args.method)
    cache = original.cache_inputs(args.data, args.split, labels=False)
    original.training.check_gpu(args.gpu_uuid)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("real authorized CUDA BF16 GPU required")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "action": "export-foundation", "method": args.method,
             "started_at": time.time(), "sources": len(cache["rows"]), "inputs": len(cache["rows"]) * 13,
             "queries": len(cache["rows"]) * 39, "completed_inputs": 0, "cache_split": args.split,
             "target_values_loaded": False, "masks_loaded": False, "gpu_uuid": args.gpu_uuid}
    atomic_json(out / "status.json", state)
    try:
        model = encoders.load_encoder(assets).to("cuda")
        protocol = feature_protocol(args.method, args.assets_sha256)
        protocol_sha = original.stable_hash(protocol)
        atomic_json(out / "feature_protocol.json", {**protocol, "artifact_sha256": protocol_sha})
        atomic_json(out / "provenance.json", {"method": args.method, "assets_manifest": str(assets["path"]),
                    "assets_sha256": args.assets_sha256, "cache_root": str(cache["root"]),
                    "cache_split": args.split, "cache_status_sha256": cache["cache_status_sha256"],
                    "cache_bindings": cache["bindings"], "target_values_loaded": False, "masks_loaded": False,
                    "torch": torch.__version__, "gpu_uuid": args.gpu_uuid,
                    "feature_protocol_sha256": protocol_sha, "encoder_trainable_parameters": 0})
        rows = list(original.query_records(cache["rows"], args.split))
        with (out / "samples.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        names = protocol["feature_names"]
        X = np.lib.format.open_memmap(out / "feature_work.npy", mode="w+", dtype=np.float32,
                                     shape=(len(rows), len(names)))
        available = np.zeros(len(rows), dtype=bool)
        loader = torch.utils.data.DataLoader(ImageOnlyDataset(cache["paths"]["images.npy"], args.method),
                    batch_size=128, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
        with torch.inference_mode():
            for step, (images, indices) in enumerate(loader):
                indices = indices.numpy()
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        encoded = encoders.encode_batch(model, images.to("cuda", non_blocking=True), args.method)
                    features = encoders.query_features(encoded.cpu().numpy())
                    query_indices = (indices[:, None] * 3 + np.arange(3)[None, :]).ravel()
                    X[query_indices] = features
                    available[query_indices] = True
                except Exception:
                    state.update(failed_batch_inputs=indices.tolist(), failure_policy="stop; preserve denominator; no retry")
                    raise
                state["completed_inputs"] += len(indices)
                if step % 50 == 0:
                    X.flush()
                    atomic_json(out / "status.json", state)
                    print(json.dumps({"method": args.method, "split": args.split,
                                      "completed_inputs": state["completed_inputs"], "inputs": state["inputs"]}), flush=True)
        if not available.all() or state["completed_inputs"] != state["inputs"]:
            raise ValueError("incomplete feature export")
        X.flush()
        np.savez(out / "features.npz", X=X, row_ids=np.asarray([row["row_id"] for row in rows], dtype="U64"),
                 available=available)
        del X
        (out / "feature_work.npy").unlink()
        atomic_json(out / "manifest.json", original.make_manifest(out / "samples.jsonl", out / "features.npz", names, protocol_sha))
        # A completed export is only valid for the exact assets read at launch.
        encoders.verify_assets(args.assets, args.assets_sha256, args.method)
        state.update(status="completed", finished_at=time.time(), finite_feature_rate=1.,
                     failed_inputs=0, feature_dimension=len(names),
                     output_sha256={name: sha256(out / name) for name in
                         ("samples.jsonl", "features.npz", "manifest.json", "feature_protocol.json", "provenance.json")})
        atomic_json(out / "status.json", state)
        return state
    except BaseException as error:
        state.update(status="failed", failed_at=time.time(), error=repr(error), traceback=traceback.format_exc())
        atomic_json(out / "status.json", state)
        raise


def completed_files(root, required):
    root = Path(root).resolve(strict=True)
    state = read_json(root / "status.json")
    if state.get("status") != "completed":
        raise ValueError("completed source artifact required")
    for name in required:
        if sha256(root / name) != state.get("output_sha256", {}).get(name):
            raise ValueError("bound source artifact mismatch: " + name)
    return root, state


def attach_labels(args):
    """Join the old independent y by exact identity; do not read old X or masks."""
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("fresh label bridge output required")
    exported, receipt = completed_files(args.export_dir, ("samples.jsonl", "features.npz", "manifest.json", "provenance.json"))
    truth, truth_receipt = completed_files(args.truth_dir, ("samples.jsonl", "labels.npz"))
    if receipt.get("action") != "export-foundation" or truth_receipt.get("action") != "attach-labels":
        raise ValueError("foundation features and original independent label action required")
    samples = [json.loads(line) for line in (exported / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    truth_rows = [json.loads(line) for line in (truth / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if samples != truth_rows or not samples:
        raise ValueError("full feature/truth query identities differ")
    ids = np.asarray([row["row_id"] for row in samples], dtype="U64")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("duplicate query identities")
    with np.load(truth / "labels.npz", allow_pickle=False) as labels:
        if set(labels.files) != {"y", "row_ids"} or not np.array_equal(labels["row_ids"], ids):
            raise ValueError("independent labels must align by exact row identity")
        y = labels["y"]
    if y.dtype.kind != "f" or y.shape != (len(ids),) or not np.isfinite(y).all() or (y < 0).any() or (y > 1).any():
        raise ValueError("finite independent anatomical retention targets in [0,1] required")
    with np.load(exported / "features.npz", allow_pickle=False) as features:
        if set(features.files) != {"X", "row_ids", "available"} or not np.array_equal(features["row_ids"], ids):
            raise ValueError("label-free feature identity differs")
        X, available = features["X"], features["available"]
    manifest = read_json(exported / "manifest.json")
    if (X.dtype != np.float32 or X.shape != (len(ids), len(manifest["feature_names"])) or
            available.dtype != np.bool_ or available.shape != (len(ids),) or not available.all() or not np.isfinite(X).all()):
        raise ValueError("complete finite features required")
    out.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "action": "attach-labels", "method": receipt["method"], "started_at": time.time(),
             "prediction_rerun": False, "model_selection_performed": False, "target_values_loaded": True,
             "masks_loaded": False, "original_features_loaded": False}
    atomic_json(out / "status.json", state)
    try:
        np.savez(out / "features.npz", X=X, row_ids=ids, available=available, y=y)
        np.savez(out / "labels.npz", y=y, row_ids=ids)
        shutil.copyfile(exported / "samples.jsonl", out / "samples.jsonl")
        atomic_json(out / "manifest.json", original.make_manifest(out / "samples.jsonl", out / "features.npz",
                    manifest["feature_names"], manifest["feature_protocol_sha256"]))
        atomic_json(out / "provenance.json", {"source_export_status_sha256": sha256(exported / "status.json"),
                    "source_truth_status_sha256": sha256(truth / "status.json"),
                    "source_labels_sha256": sha256(truth / "labels.npz"), "label_free_features_preserved": True,
                    "feature_protocol_sha256": manifest["feature_protocol_sha256"], "join": "exact ordered row_id and complete query metadata",
                    "masks_loaded": False, "original_features_loaded": False, "prediction_rerun": False})
        state.update(status="completed", finished_at=time.time(), queries=len(ids),
                     output_sha256={name: sha256(out / name) for name in
                         ("labels.npz", "features.npz", "samples.jsonl", "manifest.json", "provenance.json")})
        atomic_json(out / "status.json", state)
        return state
    except BaseException as error:
        state.update(status="failed", failed_at=time.time(), error=repr(error), traceback=traceback.format_exc())
        atomic_json(out / "status.json", state)
        raise


def preview(args):
    """CPU check and technical PNG. Actual visual inspection is a separate receipt."""
    from PIL import Image, ImageDraw
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("fresh technical preview directory required")
    cache = original.cache_inputs(args.data, args.split, labels=False)
    images = np.load(cache["paths"]["images.npy"], mmap_mode="r", allow_pickle=False)
    indices = [(source, variant) for source in range(min(3, len(images))) for variant in (0, 4, 12)]
    out.mkdir(parents=True, exist_ok=False)
    sheet = Image.new("RGB", (3 * 224, len(indices) * 248), "white")
    draw = ImageDraw.Draw(sheet)
    records = []
    for index, (source, variant) in enumerate(indices):
        gray = images[source, variant]
        rgb = encoders.presented_rgb(gray)
        tensor = encoders.preprocess_image(gray, args.method)
        policy = encoders.preprocessing_policy(args.method)
        recovered = tensor * np.asarray(policy["std"], dtype=np.float32)[:, None, None] + np.asarray(policy["mean"], dtype=np.float32)[:, None, None]
        recovered = np.clip(np.rint(recovered.transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
        if not np.array_equal(recovered, rgb) or not np.isfinite(tensor).all():
            raise ValueError("actual normalization round trip changed presented pixels")
        tile = Image.fromarray(gray).convert("RGB").resize((224, 224))
        sheet.paste(tile, (0, index * 248 + 24))
        sheet.paste(Image.fromarray(rgb), (224, index * 248 + 24))
        sheet.paste(Image.fromarray(recovered), (448, index * 248 + 24))
        draw.text((4, index * 248 + 4), f"source {source} variant {variant}: cache | full-frame RGB224 | denormalized actual input", fill="black")
        name = f"input_{source:02d}_{variant:02d}.npy"
        np.save(out / name, tensor)
        records.append({"source_index": source, "variant_index": variant, "tensor_file": name,
                        "tensor_file_sha256": sha256(out / name), "shape": list(tensor.shape),
                        "min": float(tensor.min()), "max": float(tensor.max()), "finite": True,
                        "roundtrip_pixels_equal": True})
    sheet.save(out / "preprocessing_contact_sheet.png")
    result = {"status": "completed", "action": "technical-preview", "method": args.method,
              "sources": min(3, len(images)), "inputs": len(records), "records": records,
              "cache_root": str(cache["root"]), "cache_split": args.split,
              "cache_status_sha256": cache["cache_status_sha256"], "cache_bindings": cache["bindings"],
              "policy": encoders.preprocessing_policy(args.method), "masks_loaded": False, "targets_loaded": False,
              "actual_visual_review": "pending; generator does not claim human visual inspection",
              "contact_sheet_sha256": sha256(out / "preprocessing_contact_sheet.png")}
    atomic_json(out / "status.json", result)
    return result


def smoke(args):
    """Actual pretrained forward check, never a partial production export."""
    if args.device not in ("cpu", "cuda") or args.batch_size != 128:
        raise ValueError("explicit CPU or CUDA technical smoke and fixed production batch128 required")
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("fresh technical smoke output required")
    assets = encoders.verify_assets(args.assets, args.assets_sha256, args.method)
    cache = original.cache_inputs(args.data, args.split, labels=False)
    if args.device == "cuda":
        if args.gpu_uuid not in ALLOWED_GPUS:
            raise ValueError("authorized GPU UUID required")
        original.training.check_gpu(args.gpu_uuid)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    torch.set_num_threads(4)
    if args.device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("real CUDA BF16 GPU required")
    out.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "action": "smoke-foundation", "production_export": False,
             "method": args.method, "device": args.device, "started_at": time.time(),
             "assets_sha256": args.assets_sha256, "cache_root": str(cache["root"]), "cache_split": args.split,
             "cache_status_sha256": cache["cache_status_sha256"], "cache_bindings": cache["bindings"],
             "target_values_loaded": False, "masks_loaded": False, "gpu_uuid": args.gpu_uuid if args.device == "cuda" else None}
    atomic_json(out / "status.json", state)
    try:
        model = encoders.load_encoder(assets).to(args.device)
        dataset = ImageOnlyDataset(cache["paths"]["images.npy"], args.method)
        actual_indices = [source * 13 + variant for source in range(min(3, len(cache["rows"]))) for variant in (0, 4, 12)]
        indices = [actual_indices[index % len(actual_indices)] for index in range(128 if args.device == "cuda" else 3)]
        inputs = np.stack([dataset[index][0] for index in indices])
        tensor = torch.from_numpy(inputs).to(args.device)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else nullcontext():
            embeddings = encoders.encode_batch(model, tensor, args.method)
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        features = encoders.query_features(embeddings.cpu().numpy())
        np.save(out / "technical_features.npy", features)
        state.update(status="completed", finished_at=time.time(), actual_input_indices=indices,
                     distinct_presented_inputs=len(set(indices)), batch_inputs=len(indices),
                     batch_repeats_for_technical_memory_profile=args.device == "cuda", forward_seconds=elapsed,
                     peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                     finite=True, features_shape=list(features.shape), torch=torch.__version__,
                     encoder_trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                     feature_protocol_sha256=original.stable_hash(feature_protocol(args.method, args.assets_sha256)),
                     technical_features_sha256=sha256(out / "technical_features.npy"),
                     interpretation="actual pretrained forward and input pipeline only; no scientific metrics or production acceptance")
        atomic_json(out / "status.json", state)
        return state
    except BaseException as error:
        state.update(status="failed", failed_at=time.time(), error=repr(error), traceback=traceback.format_exc())
        atomic_json(out / "status.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("export", "attach-labels", "preview", "smoke"), required=True)
    parser.add_argument("--method", choices=encoders.METHODS)
    parser.add_argument("--assets"); parser.add_argument("--assets-sha256")
    parser.add_argument("--data"); parser.add_argument("--split"); parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-uuid"); parser.add_argument("--batch-size", type=int, choices=(128,), default=128)
    parser.add_argument("--workers", type=int, choices=(4,), default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--export-dir"); parser.add_argument("--truth-dir")
    args = parser.parse_args()
    if args.action in ("export", "preview", "smoke") and not all((args.method, args.data, args.split)):
        parser.error("export/preview requires method, data and split")
    if args.action == "export" and not all((args.assets, args.assets_sha256, args.gpu_uuid)):
        parser.error("export requires assets, assets-sha256 and gpu-uuid")
    if args.action == "smoke" and not all((args.assets, args.assets_sha256)):
        parser.error("smoke requires assets and assets-sha256")
    if args.action == "attach-labels" and not all((args.export_dir, args.truth_dir)):
        parser.error("attach-labels requires export-dir and truth-dir")
    print(json.dumps({"export": export, "attach-labels": attach_labels, "preview": preview, "smoke": smoke}[args.action](args)), flush=True)


if __name__ == "__main__":
    main()

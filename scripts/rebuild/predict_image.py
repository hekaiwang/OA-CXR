"""Predict anatomical-retention proxies from one current PNG/JPEG and a verified package.

Requires this repository at the package's bound implementation and environment.
No original image, crop metadata, annotation, target or source catalog is read.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import warnings

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from oa_cxr.io import exclusive_writer, load_json, sha256_file, write_json

SCHEMA = 'oa-cxr-retention-weight-package-v1'
FINDINGS = ['pleural_effusion', 'pneumothorax', 'consolidation']
PREPROCESS = {'image_size': 256, 'grayscale': "Pillow convert('L')", 'resize': 'bilinear preserve aspect',
              'rounding': 'Python round; minimum resized side 1', 'padding': 'zero uint8; centered floor offset',
              'normalization': '(gray/255*2-1)*1024', 'source': 'current image only',
              'exif_orientation': 'must be absent or 1; no hidden rotation', 'clinical_laterality': False}
ARCHITECTURE = {'version': 'direct-image-auxiliary-anatomy-retention-v1', 'encoder': 'XRV DenseNet121 features',
                'width': 64, 'finding_embedding_dim': 16, 'region_dropout': 0.2, 'freeze_encoder_batchnorm': True}
RUNTIME_FILES = ('scripts/rebuild/predict_image.py', 'src/oa_cxr/rebuild/vision.py', 'src/oa_cxr/rebuild/data.py',
                 'src/oa_cxr/anatomy_geometry.py', 'src/oa_cxr/io.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def runtime_versions():
    return {name: importlib.metadata.version(name) for name in ('torch', 'torchxrayvision', 'numpy', 'Pillow', 'scipy')}


def verify_package(package, manifest_sha256, *, root=ROOT):
    require(isinstance(manifest_sha256, str) and re.fullmatch('[0-9a-f]{64}', manifest_sha256), 'Explicit manifest SHA256 required')
    package = Path(package).resolve(strict=True)
    path = package / 'manifest.json'
    require(not path.is_symlink() and path.is_file() and sha256_file(path) == manifest_sha256, 'Package manifest SHA mismatch')
    manifest = load_json(path)
    require(manifest.get('schema') == SCHEMA and manifest.get('findings') == FINDINGS and
            manifest.get('architecture') == ARCHITECTURE and manifest.get('preprocessing') == PREPROCESS,
            'Unsupported package architecture or preprocessing')
    require(manifest.get('repository_required') is True and manifest.get('clinical_support_model') is False
            and manifest.get('optimizer_included') is False,
            'Unsupported package scope')
    evidence = manifest.get('training_evidence', {})
    epoch = evidence.get('selected_epoch')
    require(type(epoch) is int and 1 <= epoch <= 13 and evidence.get('completed_epochs') == 13
            and evidence.get('optimizer_steps') == 975 and evidence.get('selected_checkpoint_optimizer_steps') == epoch * 75
            and type(evidence.get('unique_training_sources')) is int and evidence['unique_training_sources'] >= 10000
            and evidence.get('all_13_variants_seen_per_fit_source') is True
            and evidence.get('all_audited_components_changed') is True
            and evidence.get('clinical_effectiveness_established') is False,
            'Incomplete training evidence in package')
    for name in ('selected_checkpoint_sha256', 'training_protocol_sha256', 'training_status_sha256',
                 'finetuning_evidence_sha256', 'initial_pretrained_sha256'):
        require(isinstance(evidence.get(name), str) and re.fullmatch('[0-9a-f]{64}', evidence[name]),
                'Missing training evidence SHA: ' + name)
    implementation = manifest.get('runtime_implementation_sha256', {})
    require(set(implementation) == set(RUNTIME_FILES), 'Incomplete runtime implementation binding')
    for name, expected in implementation.items():
        require(sha256_file(root / name) == expected, 'Repository implementation differs: ' + name)
    require(manifest.get('runtime_versions') == runtime_versions(), 'Package runtime versions differ; explicit portability validation required')
    record = manifest.get('weights', {})
    require(record.get('path') == 'model_state.pt' and type(record.get('bytes')) is int and 0 < record['bytes'] <= 512 * 1024**2
            and re.fullmatch('[0-9a-f]{64}', record.get('sha256', '')), 'Invalid plain weight identity')
    weights = package / record['path']
    require(not weights.is_symlink() and weights.is_file() and weights.stat().st_size == record['bytes']
            and sha256_file(weights) == record['sha256'], 'Package weight length/SHA mismatch')
    return manifest, weights


def model_from_state(state):
    import torch
    import torchxrayvision as xrv
    from oa_cxr.rebuild.vision import DirectAnatomyRetention, XrvDenseNetEncoder
    require(isinstance(state, dict) and state and all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()),
            'Expected only a nonempty tensor state dictionary')
    require(all(torch.isfinite(v).all() for v in state.values()), 'Nonfinite package parameter')
    base = xrv.models.DenseNet(weights=None, op_threshs=None, apply_sigmoid=False)
    model = DirectAnatomyRetention(XrvDenseNetEncoder(base.features), base.classifier.in_features)
    model.load_state_dict(state, strict=True)
    return model.eval()


def load_package(package, manifest_sha256):
    manifest, weights = verify_package(package, manifest_sha256)
    import torch
    state = torch.load(weights, map_location='cpu', weights_only=True)
    return model_from_state(state), manifest


def preprocess_image(path):
    from oa_cxr.rebuild.data import letterbox
    path = Path(path).resolve(strict=True)
    require(path.is_file() and path.suffix.lower() in ('.png', '.jpg', '.jpeg'), 'Use one local PNG or JPEG file')
    require(0 < path.stat().st_size <= 100 * 1024**2, 'Image file size outside supported range')
    raw = path.read_bytes()
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as image:
            require(image.format in ('PNG', 'JPEG') and getattr(image, 'n_frames', 1) == 1, 'Expected a single-frame PNG/JPEG')
            require(image.getexif().get(274, 1) == 1, 'EXIF-rotated input requires explicit preparation')
            require(image.mode in ('L', 'RGB', 'RGBA'), 'Only explicit 8-bit grayscale/RGB inputs are supported')
            require(min(image.size) >= 32 and image.width * image.height <= 40_000_000, 'Decoded image size outside supported range')
            image.load()
            if image.mode == 'RGBA':
                require(image.getchannel('A').getextrema() == (255, 255), 'Transparent pixels have no declared grayscale interpretation')
            w, h = image.size
            array = letterbox(image, 256)
            metadata = {'filename': path.name, 'sha256': hashlib.sha256(raw).hexdigest(), 'format': image.format,
                        'mode': image.mode, 'original_size': [w, h]}
    scale = 256 / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    metadata.update(resized_size=[nw, nh], padding_xy=[(256 - nw) // 2, (256 - nh) // 2],
                    presented_gray_sha256=hashlib.sha256(array.tobytes()).hexdigest())
    normalized = (array.astype(np.float32)[None, None] / 255.0 * 2.0 - 1.0) * 1024.0
    require(normalized.shape == (1, 1, 256, 256) and np.isfinite(normalized).all(), 'Invalid normalized pixels')
    return normalized, metadata


def check_gpu(uuid):
    require(isinstance(uuid, str) and re.fullmatch('GPU-[0-9a-fA-F-]{36}', uuid), 'Explicit GPU UUID required')
    names = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=20)
    require(uuid in [s.strip() for s in names.splitlines()], 'GPU UUID is unavailable')
    applications = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True, timeout=20)
    if any(row.split(',')[0].strip() == uuid for row in applications.splitlines()):
        raise RuntimeError('Requested GPU is occupied; do not preempt another task')
    snapshot = subprocess.check_output(['nvidia-smi', '-i', uuid, '--query-gpu=memory.used,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True, timeout=20).strip()
    memory, utilization = [int(s.strip()) for s in snapshot.split(',')]
    if memory > 256 or utilization > 5:
        raise RuntimeError('Requested GPU is not idle')


def summarize_prediction(prediction):
    import torch
    scores = prediction['retention'].detach().float().cpu()
    lungs = prediction['lung_hard_present'].detach().cpu()
    regions = prediction['finding_region_valid'].detach().cpu()
    logits = prediction['lung_logits'].detach().float().cpu()
    require(scores.shape == (1, 3) and torch.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(), 'Invalid retention scores')
    require(lungs.shape == (1, 2) and lungs.dtype == torch.bool and regions.shape == (1, 3, 2)
            and regions.dtype == torch.bool and logits.shape == (1, 2, 256, 256) and torch.isfinite(logits).all(), 'Invalid anatomy output')
    masks = (logits[0] >= 0).numpy()
    require(np.array_equal(masks.reshape(2, -1).any(-1), lungs[0].numpy()), 'Mask and lung-presence diagnostics disagree')
    findings = {}
    for i, name in enumerate(FINDINGS):
        score = float(scores[0, i])
        validity = regions[0, i].tolist()
        review = score < .9
        diagnostic_review = not all(validity) or not bool(lungs.all())
        findings[name] = {'anatomical_retention': score, 'predicted_region_present': validity,
                          'predicted_region_empty': [not x for x in validity], 'review_recommended': review,
                          'review_threshold': .9,
                          'review_rule': 'fixed score < 0.9; matches batch geometry_review threshold',
                          'diagnostic_review_recommended': diagnostic_review,
                          'diagnostic_review_reason': 'empty predicted lung or finding region' if diagnostic_review else None,
                          'diagnostic_rule_scope': 'runtime anatomy diagnostic; not the report-editing metric',
                          'message': ('相关解剖区域覆盖可能不足，请复核当前图像。' if review else '未触发固定解剖覆盖复核规则。') +
                                     ('预测解剖区域为空，另需人工复核。' if diagnostic_review else '') +
                                     '该分数不判断疾病存在或不存在，也不证明阴性描述得到临床支持。'}
    return {'findings': findings, 'predicted_lung_empty': [not x for x in lungs[0].tolist()],
            'mask_channel_meaning': 'image-x ordered annotation regions; not clinical laterality',
            'clinical_support_model': False, 'ground_truth_or_source_metadata_read': False}, masks


def run(package, manifest_sha256, image, output, *, gpu_uuid=None, cpu_technical_check=False, save_masks=False):
    require(bool(gpu_uuid) != bool(cpu_technical_check), 'Choose one explicit GPU UUID or CPU technical mode')
    package, output = Path(package).resolve(strict=True), Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(package), 'Use a fresh output directory outside the package')
    pixels, metadata = preprocess_image(image)
    if gpu_uuid:
        check_gpu(gpu_uuid)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid or ''
    import torch
    torch.set_num_threads(4)
    model, manifest = load_package(package, manifest_sha256)
    device = 'cpu' if cpu_technical_check else 'cuda:0'
    if gpu_uuid:
        check_gpu(gpu_uuid)
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(), 'One masked BF16 CUDA device required')
        actual = getattr(torch.cuda.get_device_properties(0), 'uuid', None)
        require(actual is None or str(actual).lower().removeprefix('gpu-') == gpu_uuid.lower().removeprefix('gpu-'), 'CUDA runtime UUID differs')
    model = model.to(device)
    with torch.inference_mode(), torch.autocast(device_type='cpu' if cpu_technical_check else 'cuda',
                                               dtype=torch.bfloat16, enabled=not cpu_technical_check):
        prediction = model(torch.from_numpy(pixels).to(device))
    result, masks = summarize_prediction(prediction)
    # Recheck immutable inputs before committing an answer.
    verify_package(package, manifest_sha256)
    require(sha256_file(image) == metadata['sha256'], 'Input image changed during inference')
    result.update(schema='oa-cxr-single-image-retention-v1', input=metadata, manifest_sha256=manifest_sha256,
                  model_weights_sha256=manifest['weights']['sha256'], gpu_uuid=gpu_uuid,
                  execution='CPU float32 technical check; not formal GPU equivalence' if cpu_technical_check else 'CUDA BF16 autocast',
                  numerical_equivalence_to_batched_exports_verified=False, preprocessing=PREPROCESS)
    with exclusive_writer(output):
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'prediction.json', result)
        files = ['prediction.json']
        if save_masks:
            for i in range(2):
                name = f'predicted_region_{i}.png'
                Image.fromarray(masks[i].astype(np.uint8) * 255).save(output / name)
                files.append(name)
        write_json(output / 'status.json', {'status': 'completed', 'clinical_support_model': False,
                   'output_sha256': {name: sha256_file(output / name) for name in files},
                   'mask_coordinates': '256x256 letterboxed current input; no inverse warp'})
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for option in ('package', 'manifest-sha256', 'image', 'output'):
        p.add_argument('--' + option, required=True)
    device = p.add_mutually_exclusive_group(required=True)
    device.add_argument('--gpu-uuid')
    device.add_argument('--cpu-technical-check', action='store_true')
    p.add_argument('--save-masks', action='store_true')
    a = p.parse_args()
    print(run(a.package, a.manifest_sha256, a.image, a.output, gpu_uuid=a.gpu_uuid,
              cpu_technical_check=a.cpu_technical_check, save_masks=a.save_masks))

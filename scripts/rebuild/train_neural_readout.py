"""New CPU fit/dev-only neural readout experiment, with an immutable source config."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import tabular
from oa_cxr.io import exclusive_writer, load_json, read_jsonl, sha256_file, stable_hash, write_json

SCHEMA = 'oa-cxr-neural-readout-training-v1'
SOURCE_NAMES = {'fit_manifest', 'dev_manifest', 'selected_checkpoint', 'original_dev_cohort', 'dev_predictions',
                'fit_export_status', 'dev_export_status'}
FINDINGS = ('pleural_effusion', 'pneumothorax', 'consolidation')
FIT_SOURCES, DEV_SOURCES, ORIGINAL_DEV_SOURCES = 19086, 2609, 512
SETTINGS = dict(seed=17, epochs=30, batch_size=2048, threads=12, dtype='float32',
                learning_rate=.001, final_learning_rate=.00001, weight_decay=.0001,
                width=256, inner_width=256, blocks=2, dropout=0.)
CANDIDATES = (
    dict(name='direct_linear_l1', activation='linear_clamp', residual_mode='none', loss='l1', beta=.02),
    dict(name='direct_linear_huber002', activation='linear_clamp', residual_mode='none', loss='huber', beta=.02),
    dict(name='old_logit_residual_l1', activation='sigmoid', residual_mode='old_logit', loss='l1', beta=.02))


def require(condition, message):
    if not condition: raise ValueError(message)


def allowed_path(path):
    """Reject recognizable heldout source directories before reading their bytes."""
    path = Path(path).resolve()
    for part in path.parts:
        low = part.lower()
        require(not re.fullmatch(r'(?:labels_|labelled_|export_)?(?:test|calibration|external(?:_[a-z0-9]+)*)', low),
                'Heldout path is prohibited: ' + str(path))
    return path


def configuration(path, digest):
    path = Path(path).resolve(strict=True)
    require(re.fullmatch('[0-9a-f]{64}', digest or '') and sha256_file(path) == digest, 'Config SHA mismatch')
    config = load_json(path)
    require(set(config) == {'schema', 'experiment', 'sources'} and config['schema'] == SCHEMA
        and isinstance(config['experiment'], str) and config['experiment'] and set(config['sources']) == SOURCE_NAMES,
        'Unexpected source-only frozen configuration')
    bindings = {str(path): digest}; sources = {}
    for name, spec in config['sources'].items():
        require(set(spec) == {'path', 'sha256'} and re.fullmatch('[0-9a-f]{64}', spec['sha256']), 'Source SHA required')
        candidate = Path(spec['path'])
        source = allowed_path(candidate if candidate.is_absolute() else path.parent / candidate)
        require(source.is_file() and sha256_file(source) == spec['sha256'], 'Bound source differs: ' + name)
        sources[name] = source; bindings[str(source)] = spec['sha256']
    return config, sources, bindings


def load_split(path, split):
    require(split in ('fit', 'dev'), 'Only fit/dev can be loaded')
    path = allowed_path(path)
    manifest = load_json(path)
    for key in ('samples', 'features'):
        spec = manifest[key]; value = Path(spec['path'])
        allowed_path(value if value.is_absolute() else path.parent / value)
    # Check row split before load_data opens any feature/target NPZ.
    sample_path = Path(manifest['samples']['path'])
    if not sample_path.is_absolute(): sample_path = path.parent / sample_path
    require(sha256_file(sample_path) == manifest['samples']['sha256'], 'Sample source SHA differs')
    rows = read_jsonl(sample_path)
    require(rows and all(row.get('split') == split for row in rows), 'Manifest contains a forbidden or mixed split')
    del rows
    return tabular.load_data(path, targets=True)


def validate_population(data, split, expected_sources):
    rows, X, y = data['rows'], data['X'], data['y']
    require(data['available'].all() and X.dtype == np.float32 and X.shape == (expected_sources * 39, 1253)
        and y is not None and len(y) == len(X), 'Complete float32 1253-column population required')
    names = data['manifest']['feature_names']
    require(len(names) == 1253 and names[-3:] == ['finding_' + f for f in FINDINGS], 'Finding feature order differs')
    expected_names = [f'image_encoder_{i:04d}' for i in range(1024)]
    expected_names += [f'finding_region_{i:03d}' for i in range(64)]
    expected_names += [f'lung_{lung}_region_{i:03d}' for lung in (0, 1) for i in range(64)]
    expected_names += [f'lung_{lung}_{name}' for lung in (0, 1) for name in
        ('soft_mass', 'maximum', 'top_contact', 'bottom_contact', 'left_contact', 'right_contact', 'hard_present')]
    expected_names += ['finding_region_mass', 'lung_0_region_mass', 'lung_1_region_mass', 'regional_branch_used']
    expected_names += [f'finding_embedding_{i:03d}' for i in range(16)] + ['finding_' + f for f in FINDINGS]
    require(names == expected_names, 'Actual pre-Linear feature column contract differs')
    roster = {}; groups = set(); images = set(); source_ids = set()
    for index, row in enumerate(rows):
        require(row.get('split') == split and type(row.get('variant_index')) is int and 0 <= row['variant_index'] < 13
            and row.get('finding') in FINDINGS, 'Incomplete source/variant/finding identity')
        key = (row['dataset'], row['source_id']); entry = roster.setdefault(key, {'queries': set(),
            'image': row['source_image_sha256'], 'group': row['split_group_id']})
        query = (row['variant_index'], row['finding'])
        require(query not in entry['queries'] and entry['image'] == row['source_image_sha256']
            and entry['group'] == row['split_group_id'], 'Duplicate query or inconsistent source identity')
        entry['queries'].add(query)
        expected_input = stable_hash(['rebuild-input-v1', row['dataset'], row['source_id'], row['variant_index']])
        require(row['input_id'] == expected_input and row['row_id'] == stable_hash(
            ['rebuild-query-v1', row['dataset'], row['source_id'], row['variant_index'], row['finding']]), 'Query identity differs')
        finding_index = FINDINGS.index(row['finding'])
        require(np.array_equal(X[index, -3:], np.eye(3, dtype=np.float32)[finding_index]), 'Finding one-hot differs')
        groups.add(row['split_group_id']); images.add(row['source_image_sha256']); source_ids.add(row['source_id'])
    require(len(roster) == len(images) == expected_sources and all(len(item['queries']) == 39 for item in roster.values()),
            'Every unique source must supply all 13 variants and three findings')
    return dict(source_keys=set(roster), source_ids=source_ids, groups=groups, images=images)


def validate_pair(fit, dev, *, fit_count=FIT_SOURCES, dev_count=DEV_SOURCES):
    require(fit['manifest']['feature_names'] == dev['manifest']['feature_names'] and
        fit['manifest']['feature_protocol_sha256'] == dev['manifest']['feature_protocol_sha256'], 'Fit/dev feature protocol differs')
    a, b = validate_population(fit, 'fit', fit_count), validate_population(dev, 'dev', dev_count)
    for field in a:
        require(not a[field] & b[field], 'Fit/dev source/image/group overlap: ' + field)
    return dict(fit_sources=fit_count, dev_sources=dev_count, fit_queries=len(fit['rows']), dev_queries=len(dev['rows']),
                complete_variants_per_source=13, findings_per_variant=3, source_image_group_disjoint=True)


def aligned_dev_scores(path, rows):
    predictions = read_jsonl(path)
    require(len(predictions) == len(rows), 'Incomplete dev replay reference')
    scores = np.empty(len(rows), dtype=np.float32)
    for index, (predicted, row) in enumerate(zip(predictions, rows)):
        require(predicted.get('status') == 'success' and predicted.get('method') == 'direct_anatomy_retention'
            and all(predicted.get(key) == row[key] for key in ('row_id', 'input_id', 'source_id', 'split', 'finding', 'source_image_sha256'))
            and isinstance(predicted.get('score'), (float, int)) and math.isfinite(predicted['score'])
            and 0 <= predicted['score'] <= 1, 'Dev prediction identity or finite-score coverage differs')
        scores[index] = predicted['score']
    return scores


def verify_lineage(manifest_path, data, export_status_path, split, checkpoint_sha, training_protocol_sha, bindings):
    """Bind consumed labelled bytes to completed exports without rereading NPZ."""
    require(split in ('fit', 'dev'), 'Only fit/dev lineage is allowed')
    labelled = allowed_path(manifest_path).parent
    export_status_path = allowed_path(export_status_path)
    require(export_status_path.name == 'status.json', 'Explicit export status path required')
    exported = export_status_path.parent

    def digest(path):
        path = allowed_path(path); key = str(path)
        value = bindings.get(key)
        if value is None: value = sha256_file(path); bindings[key] = value
        return value

    def bound_json(folder, name, receipt):
        path = folder / name
        require(receipt.get('output_sha256', {}).get(name) == digest(path), 'Receipt output SHA differs: ' + str(path))
        return load_json(path)

    label_status_path = labelled / 'status.json'; digest(label_status_path)
    label_status = load_json(label_status_path)
    require(label_status.get('status') == 'completed' and label_status.get('action') == 'attach-labels'
        and label_status.get('queries') == len(data['rows']), 'Require completed full label-attachment receipt')
    for name in ('manifest.json', 'samples.jsonl', 'features.npz'):
        path = labelled / name
        # These exact hashes were already checked by configuration/load_data.
        require(str(path) in bindings and label_status.get('output_sha256', {}).get(name) == bindings[str(path)],
                'Labelled receipt does not bind the consumed file: ' + name)
    for field, filename in (('samples', 'samples.jsonl'), ('features', 'features.npz')):
        path = Path(data['manifest'][field]['path'])
        require((path if path.is_absolute() else labelled / path).resolve() == labelled / filename,
                'Labelled manifest file escapes its completed bundle')
    label_provenance = bound_json(labelled, 'provenance.json', label_status)
    export_status_sha = digest(export_status_path); export_status = load_json(export_status_path)
    require(label_provenance.get('source_export_status_sha256') == export_status_sha
        and label_provenance.get('label_free_features_preserved') is True
        and label_provenance.get('prediction_rerun') is False, 'Labelled provenance refers to another export')
    require(export_status.get('status') == 'completed' and export_status.get('action') == 'export'
        and export_status.get('cache_split') == split and export_status.get('queries') == len(data['rows']),
        'Require completed matching-split export receipt')
    original_manifest = bound_json(exported, 'manifest.json', export_status)
    provenance = bound_json(exported, 'provenance.json', export_status)
    feature_protocol = bound_json(exported, 'feature_protocol.json', export_status)
    protocol_sha = stable_hash({key: value for key, value in feature_protocol.items() if key != 'artifact_sha256'})
    require(feature_protocol.get('artifact_sha256') == protocol_sha == data['manifest']['feature_protocol_sha256']
        == original_manifest.get('feature_protocol_sha256') == label_provenance.get('feature_protocol_sha256')
        == provenance.get('feature_protocol_sha256'), 'Feature protocol artifact identity differs')
    require(feature_protocol.get('model_checkpoint_sha256') == checkpoint_sha
        and feature_protocol.get('training_protocol_sha256') == training_protocol_sha
        and provenance.get('checkpoint_sha256') == checkpoint_sha and provenance.get('cache_split') == split
        and provenance.get('target_values_loaded') is False, 'Export checkpoint/split/label-free provenance differs')
    require(original_manifest.get('feature_names') == data['manifest']['feature_names'] == feature_protocol.get('feature_names')
        and original_manifest.get('label_free_features') is True, 'Original feature roster differs')
    require(original_manifest.get('samples', {}).get('sha256') == export_status.get('output_sha256', {}).get('samples.jsonl')
        == bindings[str(labelled / 'samples.jsonl')], 'Export and labelled query copies differ')
    exported_feature_sha = original_manifest.get('features', {}).get('sha256')
    require(isinstance(exported_feature_sha, str) and re.fullmatch('[0-9a-f]{64}', exported_feature_sha)
        and exported_feature_sha == export_status.get('output_sha256', {}).get('features.npz'),
            'Export feature manifest is not bound to its receipt')
    cache_root = allowed_path(provenance['cache_root']); cache_status_path = cache_root / 'status.json'
    require(digest(cache_status_path) == provenance.get('cache_status_sha256'), 'Prepared cache receipt differs')
    cache_status = load_json(cache_status_path)
    require(cache_status.get('status') == 'completed', 'Prepared cache is incomplete')
    input_bindings = provenance.get('cache_bindings', {}); labels = label_provenance.get('independent_label_bindings', {})
    require(set(input_bindings) == {f'{split}/sources.jsonl', f'{split}/images.npy'}
        and set(labels) == {f'{split}/sources.jsonl', f'{split}/targets.npy', f'{split}/masks.npy'}
        and input_bindings[f'{split}/sources.jsonl'] == labels[f'{split}/sources.jsonl'], 'Independent target/source lineage differs')
    require(all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)
        and cache_status.get('files', {}).get(key) == value for key, value in {**input_bindings, **labels}.items()),
            'Input/label SHA bindings differ from the completed cache')
    return dict(split=split, labelled_status_sha256=bindings[str(label_status_path)], export_status_sha256=export_status_sha,
        cache_root=str(cache_root), cache_status_sha256=provenance['cache_status_sha256'], feature_protocol_sha256=protocol_sha,
        checkpoint_sha256=checkpoint_sha, consumed_feature_sha256=bindings[str(labelled / 'features.npz')],
        verification='completed producer receipts and SHA-bound consumed bytes; cache arrays/export NPZ not reopened')


def select_candidate(receipts):
    require(receipts and all(math.isfinite(item['best_dev_mae']) and type(item['best_epoch']) is int
        and item['best_epoch'] > 0 for item in receipts), 'Finite selected candidate metrics required')
    return min(enumerate(receipts), key=lambda item: (item[1]['best_dev_mae'], item[1]['best_epoch'], item[0]))


def array_metrics(predictions, targets):
    delta = np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64)
    require(delta.ndim == 1 and len(delta) and np.isfinite(delta).all(), 'Finite complete metric denominator required')
    return dict(rows=len(delta), mae=float(np.abs(delta).mean()), rmse=float(np.sqrt(np.square(delta).mean())))


def parameter_snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}


def parameter_delta(initial, model):
    final = dict(model.named_parameters()); changed = 0; squared = 0.; items = {}
    for name, before in initial.items():
        after = final[name].detach().cpu(); a, b = before.numpy(), after.numpy()
        delta = b.astype(np.float64) - a.astype(np.float64)
        count = int(np.count_nonzero(delta)); changed += count; squared += float(np.square(delta).sum())
        items[name] = dict(shape=list(a.shape), changed_elements=count,
            initial_float32_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
            final_float32_sha256=hashlib.sha256(b.tobytes()).hexdigest())
    return dict(changed_elements=changed, difference_l2_norm=math.sqrt(squared), parameters=items,
                actual_updates_established=changed > 0, clinical_effectiveness_established=False)


def evaluate(model, X, y, *, kind, beta, batch_size):
    import torch
    from oa_cxr.rebuild.neural_readout import readout_loss
    model.eval(); predictions = np.empty(len(y), dtype=np.float32); total_loss = 0.
    with torch.inference_mode():
        for start in range(0, len(y), batch_size):
            stop = min(start + batch_size, len(y)); output = model(X[start:stop])
            predictions[start:stop] = output['retention'].cpu().numpy()
            total_loss += float(readout_loss(output, y[start:stop], kind=kind, beta=beta)) * (stop - start)
    return dict(array_metrics(predictions, y.numpy()), loss=total_loss / len(y)), predictions


def train_candidate(model, specification, fit_X, fit_y, dev_X, dev_y, secondary_mask, output, settings, *, protocol_sha256):
    import torch
    from oa_cxr.rebuild.neural_readout import readout_loss
    output.mkdir(exist_ok=False)
    initial = parameter_snapshot(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    per_epoch = math.ceil(len(fit_y) / settings['batch_size']); total_steps = per_epoch * settings['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=settings['final_learning_rate'])
    steps = 0; best = None; history = []
    for epoch in range(1, settings['epochs'] + 1):
        started = time.time(); model.train()
        indices = torch.randperm(len(fit_y), generator=torch.Generator().manual_seed(settings['seed'] + epoch))
        absolute = squared = loss_sum = 0.; seen = 0
        lr_first = optimizer.param_groups[0]['lr']
        for start in range(0, len(indices), settings['batch_size']):
            idx = indices[start:start + settings['batch_size']]; target = fit_y[idx]
            optimizer.zero_grad(set_to_none=True); values = model(fit_X[idx])
            loss = readout_loss(values, target, kind=specification['loss'], beta=specification['beta'])
            require(bool(torch.isfinite(loss)), 'Nonfinite fit loss')
            loss.backward()
            require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad),
                    'Missing/nonfinite trainable gradients')
            optimizer.step(); scheduler.step(); steps += 1
            error = values['retention'].detach().double() - target.double()
            absolute += float(error.abs().sum()); squared += float(error.square().sum())
            loss_sum += float(loss.detach()) * len(idx); seen += len(idx)
        require(seen == len(fit_y) and steps == epoch * per_epoch, 'Incomplete shuffled fit epoch')
        dev_metrics, predictions = evaluate(model, dev_X, dev_y, kind=specification['loss'], beta=specification['beta'],
                                             batch_size=settings['batch_size'])
        record = dict(epoch=epoch, optimizer_steps=steps, optimizer_steps_in_epoch=per_epoch,
            fit_online=dict(rows=seen, loss=loss_sum / seen, mae=absolute / seen, rmse=math.sqrt(squared / seen)),
            dev=dev_metrics, original512dev_secondary=array_metrics(predictions[secondary_mask], dev_y.numpy()[secondary_mask]),
            learning_rate_first_step=lr_first, learning_rate_after_epoch=optimizer.param_groups[0]['lr'],
            elapsed_seconds=time.time() - started)
        if best is None or dev_metrics['mae'] < best['dev']['mae']:
            best = copy.deepcopy(record)
            torch.save({name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}, output / 'best_state.pt')
            write_json(output / 'best_model.json', dict(schema=SCHEMA, candidate=specification, model_config=model.config(),
                best_epoch=epoch, selected_optimizer_steps=steps, dev_mae=dev_metrics['mae'], protocol_sha256=protocol_sha256,
                state_sha256=sha256_file(output / 'best_state.pt'), optimizer_included=False))
        history.append(record); write_json(output / f'epoch_{epoch:02d}.json', record)
        write_json(output / 'status.json', dict(status='running', completed_epochs=epoch, optimizer_steps=steps,
            best_epoch=best['epoch'], best_dev_mae=best['dev']['mae']))
        print(specification['name'], epoch, 'dev_mae', dev_metrics['mae'], 'steps', steps, flush=True)
    evidence_last = parameter_delta(initial, model)
    model.load_state_dict(torch.load(output / 'best_state.pt', map_location='cpu', weights_only=True), strict=True)
    evidence_best = parameter_delta(initial, model)
    require(steps == total_steps and evidence_last['actual_updates_established'] and evidence_best['actual_updates_established'],
            'Expected real parameter updates at selected and final state')
    write_json(output / 'parameter_update_evidence.json', dict(selected=evidence_best, last_epoch=evidence_last,
        optimizer_steps=steps, selected_optimizer_steps=best['optimizer_steps'], encoder_updated=False))
    receipt = dict(status='completed', candidate=specification['name'], completed_epochs=settings['epochs'], optimizer_steps=steps,
        best_epoch=best['epoch'], best_dev_mae=best['dev']['mae'], selection='full fixed dev MAE only; earliest epoch on ties',
        output_sha256={p.name: sha256_file(p) for p in output.iterdir() if p.is_file() and p.name != 'status.json'})
    write_json(output / 'status.json', receipt)
    return receipt


def run(config_path, config_sha256, output):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    config, sources, bindings = configuration(config_path, config_sha256)
    output = Path(output).resolve(); require(not output.exists(), 'Training output must be fresh')
    require(not any(path.is_relative_to(output) or output.is_relative_to(path.parent) for path in sources.values()),
            'Output must be separate from immutable input directories')
    fit, dev = load_split(sources['fit_manifest'], 'fit'), load_split(sources['dev_manifest'], 'dev')
    population = validate_pair(fit, dev, fit_count=FIT_SOURCES, dev_count=DEV_SOURCES)
    bindings.update(fit['bindings']); bindings.update(dev['bindings'])
    cohort = load_json(sources['original_dev_cohort']); subset = cohort.get('source_ids', [])
    require(cohort.get('selection_before_first_epoch') is True and len(subset) == len(set(subset)) == ORIGINAL_DEV_SOURCES
        and set(subset).issubset({row['source_id'] for row in dev['rows']}), 'Original fixed dev source cohort differs')
    subset_ids = set(subset)
    secondary = np.asarray([row['source_id'] in subset_ids for row in dev['rows']])
    require(int(secondary.sum()) == ORIGINAL_DEV_SOURCES * 39, 'Original dev subset query coverage differs')
    reference = aligned_dev_scores(sources['dev_predictions'], dev['rows'])
    import export
    selected = export.selected_checkpoint(sources['selected_checkpoint'].parent)
    require(selected['path'] == sources['selected_checkpoint'] and selected['sha256'] == bindings[str(selected['path'])],
            'Source checkpoint is not the frozen dev-selected original model')
    lineage = {split: verify_lineage(sources[split + '_manifest'], data, sources[split + '_export_status'], split,
        selected['sha256'], selected['protocol_sha256'], bindings) for split, data in (('fit', fit), ('dev', dev))}
    require(all(lineage['fit'][key] == lineage['dev'][key] for key in
        ('cache_root', 'cache_status_sha256', 'feature_protocol_sha256', 'checkpoint_sha256')), 'Fit/dev producer ancestry differs')
    require(sources['dev_predictions'].parent == sources['dev_export_status'].parent
        and load_json(sources['dev_export_status'])['output_sha256'].get('predictions.jsonl') ==
        bindings[str(sources['dev_predictions'])], 'Dev replay reference is not bound to the verified dev export')
    for path in (selected['path'].parent / 'protocol.json',
                 selected['path'].parent / 'status.json', selected['path'].parent / 'best.json'):
        bindings[str(path)] = sha256_file(path)
    import torch
    from oa_cxr.rebuild.neural_readout import NeuralReadout, FrozenRetentionHead
    torch.set_num_threads(SETTINGS['threads']); torch.set_num_interop_threads(1)
    require(str(torch.__version__) == selected['protocol']['torch'], 'Original Torch runtime differs')
    random.seed(SETTINGS['seed']); np.random.seed(SETTINGS['seed']); torch.manual_seed(SETTINGS['seed'])
    payload = torch.load(selected['path'], weights_only=True, map_location='cpu')
    require(payload.get('epoch') == selected['epoch'] and payload.get('protocol_sha256') == selected['protocol_sha256']
        and payload.get('findings') == list(FINDINGS), 'Selected checkpoint payload identity differs')
    old_head = {key.removeprefix('retention_head.'): value.float() for key, value in payload['model_state'].items()
                if key.startswith('retention_head.')}
    del payload
    control = FrozenRetentionHead(1250, 128, 3); control.layers.load_state_dict(old_head, strict=True); control.eval()
    fit_X, fit_y = torch.from_numpy(fit['X']), torch.from_numpy(fit['y']).float()
    dev_X, dev_y = torch.from_numpy(dev['X']), torch.from_numpy(dev['y']).float()
    for name in ('scripts/rebuild/train_neural_readout.py', 'src/oa_cxr/rebuild/neural_readout.py', 'baselines/tabular.py',
                 'src/oa_cxr/io.py', 'scripts/rebuild/export.py', 'scripts/rebuild/train.py', 'src/oa_cxr/rebuild/vision.py'):
        bindings[str(ROOT / name)] = sha256_file(ROOT / name)
    with exclusive_writer(output):
        output.mkdir(parents=True, exist_ok=False)
        state = dict(status='running', schema=SCHEMA, started_at=time.time(), heldout_data_access=False)
        write_json(output / 'status.json', state)
        try:
            protocol = dict(schema=SCHEMA, config_sha256=config_sha256, experiment=config['experiment'], settings=SETTINGS,
                candidates=CANDIDATES, population=population, source_file_sha256=bindings, producer_lineage=lineage,
                runtime_versions={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'scikit-learn')},
                original_checkpoint_sha256=selected['sha256'], original_checkpoint_epoch=selected['epoch'],
                precision='CPU float32, no autocast', selection='lexicographic (full2609dev MAE, selected epoch, declared candidate order)',
                development_protocol_revision='all dev sources primary; original fixed512 cohort secondary only',
                normalizer='fit only, all fit query rows, streaming population mean/std, frozen for dev',
                fit_online_metrics='pre-update predictions across a changing model; not final frozen-model fit metrics',
                compared_same_feature_learners=True, encoder_updated=False, masks_updated=False, clinical_effectiveness_established=False)
            write_json(output / 'protocol.json', protocol); protocol_sha = sha256_file(output / 'protocol.json')
            torch.save(old_head, output / 'original_head_state.pt')
            control_fit, _ = evaluate(control, fit_X, fit_y, kind='l1', beta=.02, batch_size=SETTINGS['batch_size'])
            control_dev, replay = evaluate(control, dev_X, dev_y, kind='l1', beta=.02, batch_size=SETTINGS['batch_size'])
            delta = replay.astype(np.float64) - reference.astype(np.float64)
            write_json(output / 'original_head_control.json', dict(fit=control_fit, dev=control_dev,
                original512dev_secondary=array_metrics(replay[secondary], dev['y'][secondary]),
                bf16_dev_reference=array_metrics(reference, dev['y']), fp32_vs_bf16=dict(rows=len(delta),
                    mean_absolute_difference=float(np.abs(delta).mean()), max_absolute_difference=float(np.abs(delta).max()),
                    rmse=float(np.sqrt(np.square(delta).mean())), bitwise_equal=bool(np.array_equal(replay, reference))),
                original_checkpoint_sha256=selected['sha256'], replay_ignores_appended_finding_onehot=True))
            models = []; moments = None
            for specification in CANDIDATES:
                random.seed(SETTINGS['seed']); np.random.seed(SETTINGS['seed']); torch.manual_seed(SETTINGS['seed'])
                model = NeuralReadout(1253, width=SETTINGS['width'], inner_width=SETTINGS['inner_width'], blocks=SETTINGS['blocks'],
                    dropout=SETTINGS['dropout'], activation=specification['activation'], residual_mode=specification['residual_mode'],
                    retained_head=control if specification['residual_mode'] == 'old_logit' else None)
                if moments is None:
                    model.fit_standardizer((fit_X[start:start + 8192] for start in range(0, len(fit_X), 8192)), split='fit')
                    moments = {key: value.clone() for key, value in model.standardizer.state_dict().items()}
                    torch.save(moments, output / 'fit_statistics.pt')
                else: model.standardizer.load_state_dict(moments, strict=True)
                receipt = train_candidate(model, specification, fit_X, fit_y, dev_X, dev_y, secondary,
                    output / specification['name'], SETTINGS, protocol_sha256=protocol_sha)
                models.append(receipt); del model
                state.update(completed_candidates=[item['candidate'] for item in models]); write_json(output / 'status.json', state)
            index, winner = select_candidate(models)
            selection = dict(candidate=winner['candidate'], candidate_order=index, best_epoch=winner['best_epoch'],
                dev_mae=winner['best_dev_mae'], original_fp32_control_dev_mae=control_dev['mae'],
                improved_vs_original_fp32_control=winner['best_dev_mae'] < control_dev['mae'],
                recommendation=winner['candidate'] if winner['best_dev_mae'] < control_dev['mae'] else 'retain_original_head',
                selection_used_heldout=False, selection_used_original512_secondary=False,
                candidate_status_sha256={item['candidate']: sha256_file(output / item['candidate'] / 'status.json') for item in models})
            write_json(output / 'selection.json', selection)
            tabular.verify(bindings)
            state.update(status='completed', selected_candidate=winner['candidate'], selected_dev_mae=winner['best_dev_mae'],
                output_sha256={str(path.relative_to(output)): sha256_file(path) for path in output.rglob('*')
                    if path.is_file() and path != output / 'status.json'})
        except BaseException as exc:
            state.update(status='failed', error=repr(exc), traceback=traceback.format_exc()); raise
        finally:
            state['finished_at'] = time.time(); write_json(output / 'status.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    print(run(args.config, args.config_sha256, args.output))

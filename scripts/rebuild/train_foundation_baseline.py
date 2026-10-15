"""Train the four fixed new baselines, or predict without loading target values."""
from __future__ import annotations

import argparse
import copy
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
from baselines import tabular
from oa_cxr.io import exclusive_writer, load_json, read_jsonl, sha256_file, write_json, write_jsonl

SCHEMA = 'oa-cxr-foundation-baseline-training-v1'
PREDICTION_SCHEMA = 'oa-cxr-foundation-baseline-predictions-v1'
METHODS = ('biomedclip', 'chexzero', 'dinov2', 'tabm')
FINDINGS = ('pleural_effusion', 'pneumothorax', 'consolidation')
SETTINGS = dict(seed=17, epochs=30, batch_size=2048, learning_rate=.001,
    final_learning_rate=.00001, weight_decay=.0001, dtype='float32', threads=12, microbatch_size=2048)


def method_settings(method):
    # Official TabM optimizer defaults and architecture, with a fixed effective
    # batch shared with the image heads. Microbatches only bound activation RAM.
    return dict(SETTINGS, **(dict(learning_rate=.002, weight_decay=.0003, microbatch_size=256) if method == 'tabm' else {}))


def require(condition, message):
    if not condition: raise ValueError(message)


def bind(path, expected, bindings):
    path = Path(path).resolve(strict=True)
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'Explicit SHA256 required')
    require(sha256_file(path) == expected, 'Bound file differs: ' + str(path))
    bindings[str(path)] = expected
    return path


def configuration(path, expected):
    bindings = {}; path = bind(path, expected, bindings); value = load_json(path)
    require(value.get('schema') == SCHEMA and value.get('method') in METHODS, 'Unsupported training configuration')
    require(set(value) == {'schema', 'method', 'sources', 'expected_sources'}, 'Unexpected configuration fields')
    require(value['expected_sources'] == {'fit': 19086, 'dev': 2609}, 'Fixed full fit/dev population required')
    require(set(value['sources']) == {'fit_manifest', 'dev_manifest'}, 'Fit/dev-only explicit sources required')
    sources = {}
    for name, item in value['sources'].items():
        candidate = Path(item['path'])
        sources[name] = bind(candidate if candidate.is_absolute() else path.parent / candidate, item['sha256'], bindings)
    return value, sources, bindings


def load_training_split(path, split):
    require(split in ('fit', 'dev'), 'Training only accepts fit/development data')
    manifest = load_json(path)
    sample = Path(manifest['samples']['path'])
    if not sample.is_absolute(): sample = Path(path).parent / sample
    require(sha256_file(sample) == manifest['samples']['sha256'], 'Sample hash differs')
    # Reject heldout or mixed rows before opening the feature/target NPZ.
    with sample.open(encoding='utf-8-sig') as stream:
        import json
        count = 0
        for line in stream:
            if line.strip():
                require(json.loads(line).get('split') == split, 'Heldout/mixed data in training manifest')
                count += 1
    require(count > 0, 'Empty training split')
    return tabular.load_data(path, targets=True)


def population(data, split, count):
    rows, X, y = data['rows'], data['X'], data['y']
    names = data['manifest']['feature_names']
    require(X.dtype == np.float32 and X.shape == (count * 39, len(names)) and len(names) >= 4
        and y is not None and len(y) == len(X) and data['available'].all(), 'Full finite available source/query population required')
    require(names[-3:] == ['finding_' + f for f in FINDINGS], 'Finding one-hot column order differs')
    findings = np.asarray([FINDINGS.index(row['finding']) for row in rows], dtype=np.int8)
    require(np.array_equal(X[:, -3:], np.eye(3, dtype=np.float32)[findings]), 'Finding one-hot values differ')
    roster = {}; groups = set(); images = set()
    for row in rows:
        require(row['split'] == split and type(row.get('variant_index')) is int and 0 <= row['variant_index'] < 13,
            'Source split/variant identity differs')
        key = (row['dataset'], row['source_id'])
        entry = roster.setdefault(key, dict(queries=set(), image=row['source_image_sha256'], group=row['split_group_id']))
        query = (row['variant_index'], row['finding'])
        require(query not in entry['queries'] and entry['image'] == row['source_image_sha256']
            and entry['group'] == row['split_group_id'], 'Duplicate query or inconsistent source identity')
        entry['queries'].add(query); groups.add(row['split_group_id']); images.add(row['source_image_sha256'])
    require(len(roster) == len(images) == count and all(len(v['queries']) == 39 for v in roster.values()),
        'Every unique source must provide 13 variants and three findings')
    return dict(source_keys=set(roster), groups=groups, images=images)


def validate_pair(fit, dev, *, fit_count=19086, dev_count=2609):
    require(fit['manifest']['feature_names'] == dev['manifest']['feature_names'] and
        fit['manifest']['feature_protocol_sha256'] == dev['manifest']['feature_protocol_sha256'], 'Fit/dev encoder protocol differs')
    a, b = population(fit, 'fit', fit_count), population(dev, 'dev', dev_count)
    for key in a: require(not a[key] & b[key], 'Fit/dev overlap: ' + key)
    return dict(fit_sources=fit_count, dev_sources=dev_count, fit_queries=len(fit['rows']),
        dev_queries=len(dev['rows']), source_image_group_disjoint=True, variants_per_source=13, findings_per_input=3)


def model_evaluation(model, X, y, batch_size):
    import torch
    from baselines.foundation_regressors import raw_l1_loss
    model.eval(); absolute = squared = loss = 0.0
    with torch.inference_mode():
        for start in range(0, len(y), batch_size):
            target = y[start:start + batch_size]; output = model(X[start:start + batch_size])
            error = output['retention'].double() - target.double()
            absolute += float(error.abs().sum()); squared += float(error.square().sum())
            loss += float(raw_l1_loss(output, target)) * len(target)
    return dict(rows=len(y), mae=absolute / len(y), rmse=math.sqrt(squared / len(y)), raw_l1=loss / len(y))


def train_epochs(model, fit_X, fit_y, dev_X, dev_y, output, protocol_sha256, method, *, settings=SETTINGS):
    """A complete shuffled epoch, cosine per update, earliest minimum full-dev MAE."""
    import torch
    from baselines.foundation_regressors import raw_l1_loss
    output = Path(output); initial = {k: v.detach().cpu().clone() for k, v in model.named_parameters()}
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    per_epoch = math.ceil(len(fit_y) / settings['batch_size']); total_steps = settings['epochs'] * per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=settings['final_learning_rate'])
    best = None; steps = 0; started = time.time()
    for epoch in range(1, settings['epochs'] + 1):
        epoch_started = time.time(); model.train(); seen = 0; loss_sum = 0.0
        indices = torch.randperm(len(fit_y), generator=torch.Generator().manual_seed(settings['seed'] + epoch)).to(fit_X.device)
        for start in range(0, len(indices), settings['batch_size']):
            ids = indices[start:start + settings['batch_size']]; target = fit_y[ids]
            optimizer.zero_grad(set_to_none=True); batch_loss = 0.0
            for micro_start in range(0, len(ids), settings.get('microbatch_size', settings['batch_size'])):
                micro_ids = ids[micro_start:micro_start + settings.get('microbatch_size', settings['batch_size'])]
                loss = raw_l1_loss(model(fit_X[micro_ids]), fit_y[micro_ids])
                require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
                # The last effective batch and last microbatch retain their
                # true sample weights; no dropped/duplicated source queries.
                (loss * (len(micro_ids) / len(ids))).backward()
                batch_loss += float(loss.detach()) * len(micro_ids)
            require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters), 'Missing/nonfinite gradients')
            optimizer.step(); scheduler.step(); steps += 1; seen += len(ids); loss_sum += batch_loss
        require(seen == len(fit_y) and steps == epoch * per_epoch, 'Partial training epoch')
        dev = model_evaluation(model, dev_X, dev_y, settings.get('microbatch_size', settings['batch_size']))
        record = dict(epoch=epoch, optimizer_steps=steps, full_fit_queries_seen=seen, fit_raw_l1=loss_sum / seen,
            dev=dev, learning_rate_after_epoch=optimizer.param_groups[0]['lr'], elapsed_seconds=time.time() - epoch_started)
        if best is None or dev['mae'] < best['dev']['mae']:
            best = copy.deepcopy(record)
            torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, output / 'best_state.pt')
            write_json(output / 'best_model.json', dict(schema=SCHEMA, method=method, model_config=model.config(),
                selected_epoch=epoch, selected_optimizer_steps=steps, dev_mae=dev['mae'], seed=settings['seed'],
                protocol_sha256=protocol_sha256, state_sha256=sha256_file(output / 'best_state.pt')))
        write_json(output / f'epoch_{epoch:02d}.json', record)
        write_json(output / 'progress.json', dict(status='running', method=method, completed_epochs=epoch,
            optimizer_steps=steps, best_epoch=best['epoch'], best_dev_mae=best['dev']['mae']))
        print(method, 'epoch', epoch, 'dev_mae', dev['mae'], 'steps', steps, flush=True)
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, output / 'last_state.pt')
    write_json(output / 'last_model.json', dict(schema=SCHEMA, method=method, model_config=model.config(),
        selected_epoch=settings['epochs'], selected_optimizer_steps=steps, seed=settings['seed'],
        protocol_sha256=protocol_sha256, state_sha256=sha256_file(output / 'last_state.pt')))
    deltas = {}
    for endpoint in ('last', 'best'):
        if endpoint == 'best': model.load_state_dict(torch.load(output / 'best_state.pt', map_location=fit_X.device, weights_only=True))
        deltas[endpoint] = sum(int(torch.count_nonzero(value.detach().cpu() != initial[name])) for name, value in model.named_parameters())
        require(deltas[endpoint] > 0, 'No real parameter updates at ' + endpoint)
    require(steps == total_steps, 'Training update budget differs')
    return dict(completed_epochs=settings['epochs'], optimizer_steps=steps, updates_per_epoch=per_epoch,
        selected_epoch=best['epoch'], selected_optimizer_steps=best['optimizer_steps'], best_dev_mae=best['dev']['mae'],
        parameter_count=sum(p.numel() for p in parameters), changed_parameter_elements=deltas,
        training_wall_seconds=time.time() - started, selection='Full development MAE; earliest epoch on ties')


def run(config_path, config_sha256, output, device='cpu'):
    import torch
    from baselines.foundation_regressors import make_regressor
    output = Path(output).resolve(); config, sources, bindings = configuration(config_path, config_sha256)
    require(not output.exists(), 'Fresh training output required')
    with exclusive_writer(output):
        output.mkdir(parents=True, exist_ok=False)
        state = dict(schema=SCHEMA, status='running', method=config['method'], seed=17, heldout_data_access=False, started_at=time.time())
        write_json(output / 'status.json', state)
        try:
            torch.set_num_threads(SETTINGS['threads']); random.seed(17); np.random.seed(17); torch.manual_seed(17)
            if device.startswith('cuda'):
                require(torch.cuda.is_available(), 'Requested CUDA device unavailable')
                torch.cuda.manual_seed_all(17)
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            fit = load_training_split(sources['fit_manifest'], 'fit'); dev = load_training_split(sources['dev_manifest'], 'dev')
            audit = validate_pair(fit, dev)
            if config['method'] == 'tabm': require(fit['X'].shape[1] == 1253, 'TabM must consume original 1253 features')
            bindings.update(fit['bindings']); bindings.update(dev['bindings'])
            for path in (Path(__file__), ROOT / 'baselines/foundation_regressors.py', ROOT / 'src/oa_cxr/rebuild/neural_readout.py'):
                bindings[str(path.resolve())] = sha256_file(path)
            model = make_regressor(config['method'], fit['X'].shape[1])
            model.fit_standardizer((torch.from_numpy(fit['X'][start:start + 8192]) for start in range(0, len(fit['X']), 8192)), split='fit')
            settings = method_settings(config['method'])
            protocol = dict(schema=SCHEMA, config=config, settings=settings, model_config=model.config(),
                feature_names=fit['manifest']['feature_names'], feature_protocol_sha256=fit['manifest']['feature_protocol_sha256'],
                population=audit, source_file_sha256=bindings, standardizer_fit_only=True, trained_encoder=False,
                loss='Memberwise raw L1 for TabM; raw L1 otherwise; inference average then clamp to [0,1]',
                device=device, capacity_note='Matched 30 full epochs and effective batch2048; TabM official optimizer and architecture differ',
                task_adaptation=True, clinical_support_training=False,
                environment={name: importlib.metadata.version(name) for name in
                    (('torch', 'numpy', 'tabm', 'rtdl_num_embeddings') if config['method'] == 'tabm' else ('torch', 'numpy'))})
            write_json(output / 'protocol.json', protocol)
            model = model.to(device)
            tensors = [torch.from_numpy(value).to(device) for value in (fit['X'], fit['y'], dev['X'], dev['y'])]
            result = train_epochs(model, *tensors, output, sha256_file(output / 'protocol.json'), config['method'], settings=settings)
            tabular.verify(bindings)
            state.update(result, status='completed', target='anatomical_retention', seed=17, source_file_sha256=bindings,
                output_sha256={p.name: sha256_file(p) for p in output.iterdir() if p.is_file() and p.name != 'status.json'})
        except BaseException as exc:
            state.update(status='failed', error=repr(exc), traceback=traceback.format_exc()); raise
        finally:
            state['finished_at'] = time.time(); write_json(output / 'status.json', state)
    return state


def predict(training, training_status_sha256, manifest, manifest_sha256, output, device='cpu'):
    import torch
    from baselines.foundation_regressors import make_regressor
    bindings = {}; training = Path(training).resolve(); output = Path(output).resolve()
    state = load_json(bind(training / 'status.json', training_status_sha256, bindings))
    require(state.get('schema') == SCHEMA and state.get('status') == 'completed' and state.get('heldout_data_access') is False
        and state.get('seed') == 17 and state.get('completed_epochs') == 30, 'Completed full fixed-seed training required')
    for name in ('protocol.json', 'best_model.json', 'best_state.pt'):
        bind(training / name, state['output_sha256'][name], bindings)
    info = load_json(training / 'best_model.json'); protocol = load_json(training / 'protocol.json')
    require(info['state_sha256'] == sha256_file(training / 'best_state.pt') and info['protocol_sha256'] == sha256_file(training / 'protocol.json'), 'Model receipt differs')
    require(info.get('method') == state.get('method') == protocol['config'].get('method')
        and info.get('seed') == 17, 'Trained method/seed differs across receipts')
    data = tabular.load_data(bind(manifest, manifest_sha256, bindings), targets=False)
    require(data['y'] is None and all(r['split'] in ('test', 'external') for r in data['rows']), 'Heldout label-blind prediction required')
    require(data['manifest']['feature_names'] == protocol['feature_names'] and data['manifest']['feature_protocol_sha256'] == protocol['feature_protocol_sha256'], 'Prediction encoder/feature lineage differs')
    bindings.update(data['bindings']); require(not output.exists(), 'Fresh prediction output required')
    torch.set_num_threads(4); model = make_regressor(info['method'], len(protocol['feature_names']))
    require(model.config() == info['model_config'], 'Architecture differs from trained model')
    model.load_state_dict(torch.load(training / 'best_state.pt', map_location='cpu', weights_only=True), strict=True)
    model.to(device).eval(); values = np.full(len(data['rows']), np.nan, dtype=np.float32)
    with exclusive_writer(output):
        output.mkdir(parents=True, exist_ok=False)
        receipt = dict(schema=PREDICTION_SCHEMA, status='running', method=info['method'], seed=17,
            nominal_queries=len(values), target_values_loaded=False, training_status_sha256=training_status_sha256,
            model_state_sha256=info['state_sha256'], source_file_sha256=bindings, started_at=time.time())
        write_json(output / 'status.json', receipt)
        try:
            indices = np.flatnonzero(data['available'])
            with torch.inference_mode():
                batch_size = method_settings(info['method'])['microbatch_size']
                for start in range(0, len(indices), batch_size):
                    ix = indices[start:start + batch_size]; result = model(torch.from_numpy(data['X'][ix]).to(device))['retention'].cpu().numpy()
                    require(result.shape == (len(ix),) and np.isfinite(result).all(), 'Prediction batch invalid')
                    values[ix] = result
            records = [dict(row, method=info['method'], score=float(value) if ok else None,
                status='success' if ok else 'failed', failure_reason=None if ok else 'encoder_features_unavailable')
                for row, value, ok in zip(data['rows'], values, data['available'])]
            tabular.verify(bindings); write_jsonl(output / 'predictions.jsonl', records)
            receipt.update(status='completed' if data['available'].all() else 'completed_with_failures',
                successful_queries=int(data['available'].sum()), failed_queries=int((~data['available']).sum()),
                output_sha256={'predictions.jsonl': sha256_file(output / 'predictions.jsonl')})
        except BaseException as exc:
            receipt.update(status='failed', error=repr(exc), traceback=traceback.format_exc()); raise
        finally:
            receipt['finished_at'] = time.time(); write_json(output / 'status.json', receipt)
    return receipt


def smoke(method, output, device='cpu'):
    """Two production-batch synthetic updates; not a scientific model result."""
    import torch
    from baselines.foundation_regressors import make_regressor
    require(method in METHODS, 'Unknown smoke method'); output = Path(output).resolve()
    require(not output.exists(), 'Fresh technical smoke directory required')
    torch.set_num_threads(4); torch.manual_seed(17)
    dimension = 1253 if method == 'tabm' else (771 if method == 'dinov2' else 515)
    X = torch.randn(4096, dimension); y = (X[:, 0] * .1 + .5).clamp(0, 1)
    dev_X = torch.randn(256, dimension); dev_y = (dev_X[:, 0] * .1 + .5).clamp(0, 1)
    model = make_regressor(method, dimension); model.fit_standardizer(X, split='fit')
    cuda = device.startswith('cuda'); hardware = {'device': device}
    if cuda:
        require(torch.cuda.is_available(), 'Requested CUDA smoke device unavailable')
        torch.cuda.reset_peak_memory_stats(device)
        properties = torch.cuda.get_device_properties(device)
        hardware.update(name=properties.name, uuid=str(getattr(properties, 'uuid', 'unavailable')),
            total_device_memory_bytes=properties.total_memory,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    model.to(device)
    settings = dict(method_settings(method), epochs=1)
    with exclusive_writer(output):
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'protocol.json', dict(technical_smoke_only=True, production_training=False,
            synthetic_rows=4096, synthetic_dev_rows=256, method=method, device=device, settings=settings))
        result = dict(status='running', technical_smoke_only=True, production_training=False, hardware=hardware)
        write_json(output / 'status.json', result)
        try:
            result.update(train_epochs(model, X.to(device), y.to(device), dev_X.to(device), dev_y.to(device),
                output, sha256_file(output / 'protocol.json'), method, settings=settings))
            require(result['optimizer_steps'] == 2, 'Smoke must perform two optimizer updates')
            result.update(status='passed', synthetic_fit_queries=4096, synthetic_dev_queries=256,
                effective_batch_size=settings['batch_size'], microbatch_size=settings['microbatch_size'],
                production_feature_tensor_estimated_bytes=(744354 + 101751) * (dimension + 1) * 4,
                entire_production_feature_population_loaded=False,
                cuda_peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else None,
                cuda_peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(device) if cuda else None,
                output_sha256={p.name: sha256_file(p) for p in output.iterdir() if p.is_file() and p.name != 'status.json'})
        except BaseException as exc:
            result.update(status='failed', error=repr(exc), traceback=traceback.format_exc()); raise
        finally:
            write_json(output / 'status.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('train'); fit.add_argument('--config', required=True); fit.add_argument('--config-sha256', required=True)
    pred = sub.add_parser('predict')
    test = sub.add_parser('smoke'); test.add_argument('--method', choices=METHODS, required=True)
    for name in ('training', 'training-status-sha256', 'manifest', 'manifest-sha256'): pred.add_argument('--' + name, required=True)
    for p in (fit, pred, test): p.add_argument('--output', required=True); p.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.command == 'train': run(args.config, args.config_sha256, args.output, args.device)
    elif args.command == 'predict': predict(args.training, args.training_status_sha256, args.manifest, args.manifest_sha256, args.output, args.device)
    else: smoke(args.method, args.output, args.device)

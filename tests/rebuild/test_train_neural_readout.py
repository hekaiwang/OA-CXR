"""Fit/dev contracts and a real tiny CPU optimization; no production inputs."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('new_neural_readout_training', ROOT / 'scripts/rebuild/train_neural_readout.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def dataset(split, count):
    import export
    dummy = SimpleNamespace(encoder_channels=1024, lung_head=SimpleNamespace(in_channels=64),
        finding_embeddings=SimpleNamespace(embedding_dim=16), retention_head=[SimpleNamespace(in_features=1250)])
    names = export.feature_names(dummy)
    rows = []
    for source in range(count):
        source_id = f'{split}-source-{source}'
        for variant in range(13):
            for finding in m.FINDINGS:
                rows.append(dict(dataset='technical', split=split, source_id=source_id,
                    split_group_id=source_id, source_image_sha256=m.stable_hash(source_id), variant_index=variant,
                    input_id=m.stable_hash(['rebuild-input-v1', 'technical', source_id, variant]), finding=finding,
                    row_id=m.stable_hash(['rebuild-query-v1', 'technical', source_id, variant, finding])))
    X = np.random.default_rng(4).normal(size=(len(rows), 1253)).astype(np.float32)
    X[:, -3:] = np.tile(np.eye(3, dtype=np.float32), (len(rows) // 3, 1))
    return dict(rows=rows, X=X, y=np.linspace(0, 1, len(rows), dtype=np.float32),
        available=np.ones(len(rows), bool), manifest=dict(feature_names=names, feature_protocol_sha256='a' * 64))


def test_complete_all39_queries_and_disjoint_fit_dev():
    value = m.validate_pair(dataset('fit', 2), dataset('dev', 1), fit_count=2, dev_count=1)
    assert value['fit_queries'] == 78 and value['dev_queries'] == 39 and value['source_image_group_disjoint']


@pytest.mark.parametrize('defect', ['split', 'missing', 'duplicate', 'onehot', 'available', 'group_leak', 'sha_leak', 'protocol'])
def test_population_rejects_leakage_missing_or_misaligned_queries(defect):
    fit, dev = dataset('fit', 2), dataset('dev', 1)
    if defect == 'split': dev['rows'][0]['split'] = 'test'
    elif defect == 'missing': dev['X'] = dev['X'][:-1]
    elif defect == 'duplicate': dev['rows'][1] = dict(dev['rows'][0])
    elif defect == 'onehot': dev['X'][0, -3:] = [0, 0, 1]
    elif defect == 'available': dev['available'][0] = False
    elif defect == 'group_leak':
        for row in dev['rows']: row['split_group_id'] = fit['rows'][0]['split_group_id']
    elif defect == 'sha_leak':
        for row in dev['rows']: row['source_image_sha256'] = fit['rows'][0]['source_image_sha256']
    else: dev['manifest']['feature_protocol_sha256'] = 'b' * 64
    with pytest.raises(ValueError): m.validate_pair(fit, dev, fit_count=2, dev_count=1)


@pytest.mark.parametrize('directory', ['test', 'calibration', 'external_nih', 'labels_external_shenzhen', 'export_test'])
def test_heldout_paths_are_refused_before_open(directory):
    with pytest.raises(ValueError, match='Heldout'): m.allowed_path(ROOT / 'never-opened' / directory / 'features.npz')


def test_mixed_split_is_rejected_before_any_feature_array_load(tmp_path, monkeypatch):
    samples = tmp_path / 'samples.jsonl'
    samples.write_text('{"split":"test"}\n')
    manifest = tmp_path / 'manifest.json'
    m.write_json(manifest, dict(samples=dict(path=samples.name, sha256=m.sha256_file(samples)),
                               features=dict(path='never_load_features.npz', sha256='a' * 64)))
    monkeypatch.setattr(m.tabular, 'load_data', lambda *a, **k: pytest.fail('Unexpected feature/target read'))
    with pytest.raises(ValueError, match='forbidden'): m.load_split(manifest, 'fit')


def test_exact_config_sha_and_fixed_source_roster(tmp_path):
    sources = {}
    for index, name in enumerate(sorted(m.SOURCE_NAMES)):
        path = tmp_path / f'source{index}.json'; path.write_text('{}')
        sources[name] = dict(path=str(path), sha256=m.sha256_file(path))
    path = tmp_path / 'config.json'; m.write_json(path, dict(schema=m.SCHEMA, experiment='technical', sources=sources))
    _, actual, _ = m.configuration(path, m.sha256_file(path)); assert set(actual) == m.SOURCE_NAMES
    with pytest.raises(ValueError, match='Config SHA'): m.configuration(path, '0' * 64)
    value = m.load_json(path); value['epochs'] = 1; m.write_json(path, value)
    with pytest.raises(ValueError, match='configuration'): m.configuration(path, m.sha256_file(path))


def test_dev_replay_requires_exact_source_order_and_finite_reference(tmp_path):
    import json
    data = dataset('dev', 1)
    predictions = [dict(row, status='success', method='direct_anatomy_retention', score=.5) for row in data['rows']]
    path = tmp_path / 'predictions.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in predictions))
    assert np.array_equal(m.aligned_dev_scores(path, data['rows']), np.full(39, .5, dtype=np.float32))
    predictions[0], predictions[1] = predictions[1], predictions[0]
    path.write_text(''.join(json.dumps(row) + '\n' for row in predictions))
    with pytest.raises(ValueError, match='identity'): m.aligned_dev_scores(path, data['rows'])


def test_actual_tiny_cpu_training_steps_frozen_normalizer_and_state_roundtrip(tmp_path):
    torch = pytest.importorskip('torch')
    from oa_cxr.rebuild.neural_readout import NeuralReadout
    torch.set_num_threads(2); torch.manual_seed(17)
    fit, dev = dataset('fit', 2), dataset('dev', 1)
    X, y = torch.from_numpy(fit['X']), torch.from_numpy(fit['y'])
    model = NeuralReadout(1253, width=16, inner_width=16, blocks=2)
    model.fit_standardizer(X, split='fit'); mean = model.standardizer.mean.clone()
    settings = {**m.SETTINGS, 'epochs': 2, 'batch_size': 16}
    output = tmp_path / 'candidate'
    receipt = m.train_candidate(model, m.CANDIDATES[0], X, y, torch.from_numpy(dev['X']), torch.from_numpy(dev['y']),
        np.ones(len(dev['y']), bool), output, settings, protocol_sha256='a' * 64)
    assert receipt['status'] == 'completed' and receipt['optimizer_steps'] == 2 * math.ceil(len(y) / 16)
    assert torch.equal(mean, model.standardizer.mean)
    meta = m.load_json(output / 'best_model.json')
    saved = torch.load(output / 'best_state.pt', map_location='cpu', weights_only=True)
    assert all(torch.is_tensor(value) for value in saved.values()) and not meta['optimizer_included']
    restored = NeuralReadout.from_config(meta['model_config']); restored.load_state_dict(saved, strict=True); restored.eval()
    with torch.inference_mode(): assert torch.equal(model(X[:4])['retention'], restored(X[:4])['retention'])
    evidence = m.load_json(output / 'parameter_update_evidence.json')
    assert evidence['selected']['actual_updates_established'] and evidence['last_epoch']['actual_updates_established']
    final = m.load_json(output / 'epoch_02.json')
    assert final['fit_online']['rows'] == 78 and final['dev']['rows'] == 39
    assert final['learning_rate_after_epoch'] == pytest.approx(settings['final_learning_rate'])


def test_candidate_selection_ties_use_earlier_epoch_then_declared_order():
    rows = [dict(best_dev_mae=.05, best_epoch=8), dict(best_dev_mae=.05, best_epoch=3),
            dict(best_dev_mae=.05, best_epoch=3)]
    assert m.select_candidate(rows)[0] == 1
    rows[2]['best_dev_mae'] = .04
    assert m.select_candidate(rows)[0] == 2
    rows[2]['best_dev_mae'] = float('nan')
    with pytest.raises(ValueError): m.select_candidate(rows)


def lineage_fixture(tmp_path, defect=None):
    split = 'fit'; labelled = tmp_path / 'labeled' / split; labelled.mkdir(parents=True)
    exported = tmp_path / 'export' / split; exported.mkdir(parents=True)
    cache = tmp_path / 'cache'; cache.mkdir()
    data = dataset(split, 1)
    cp, training = 'c' * 64, 'd' * 64
    protocol = dict(model_checkpoint_sha256=cp, training_protocol_sha256=training, feature_names=data['manifest']['feature_names'])
    protocol_sha = m.stable_hash(protocol); protocol['artifact_sha256'] = protocol_sha
    data['manifest']['feature_protocol_sha256'] = protocol_sha
    source_sha = m.stable_hash('source-index')
    cache_bindings = {f'{split}/sources.jsonl': source_sha, f'{split}/images.npy': m.stable_hash('image-array')}
    label_bindings = {f'{split}/sources.jsonl': source_sha, f'{split}/targets.npy': m.stable_hash('targets'),
                      f'{split}/masks.npy': m.stable_hash('masks')}
    m.write_json(cache / 'status.json', dict(status='completed', files={**cache_bindings, **label_bindings}))
    m.write_json(exported / 'provenance.json', dict(checkpoint_sha256=cp, cache_split=split,
        target_values_loaded=False, feature_protocol_sha256=protocol_sha, cache_root=str(cache),
        cache_status_sha256=m.sha256_file(cache / 'status.json'), cache_bindings=cache_bindings))
    if defect == 'checkpoint':
        value = m.load_json(exported / 'provenance.json'); value['checkpoint_sha256'] = 'e' * 64
        m.write_json(exported / 'provenance.json', value)
    if defect == 'protocol': protocol['artifact_sha256'] = 'e' * 64
    m.write_json(exported / 'feature_protocol.json', protocol)
    (labelled / 'samples.jsonl').write_text('technical sample identity copy')
    (labelled / 'features.npz').write_bytes(b'already verified input fixture; never rehash in lineage')
    manifest = dict(data['manifest'], label_free_features=True,
        samples=dict(path='samples.jsonl', sha256=m.sha256_file(labelled / 'samples.jsonl')),
        features=dict(path='features.npz', sha256=m.sha256_file(labelled / 'features.npz')))
    data['manifest'] = manifest; m.write_json(labelled / 'manifest.json', manifest)
    original_manifest = copy.deepcopy(manifest); original_manifest['features']['sha256'] = 'f' * 64
    m.write_json(exported / 'manifest.json', original_manifest)
    export_files = {name: m.sha256_file(exported / name) for name in ('provenance.json', 'feature_protocol.json', 'manifest.json')}
    export_files.update({'samples.jsonl': manifest['samples']['sha256'], 'features.npz': 'f' * 64})
    m.write_json(exported / 'status.json', dict(status='completed', action='export', cache_split=split,
                                              queries=len(data['rows']), output_sha256=export_files))
    export_status_sha = m.sha256_file(exported / 'status.json')
    m.write_json(labelled / 'provenance.json', dict(source_export_status_sha256=export_status_sha,
        label_free_features_preserved=True, prediction_rerun=False, feature_protocol_sha256=protocol_sha,
        independent_label_bindings=label_bindings))
    if defect in ('parent', 'target_source'):
        value = m.load_json(labelled / 'provenance.json')
        if defect == 'parent': value['source_export_status_sha256'] = 'b' * 64
        else: value['independent_label_bindings']['fit/targets.npy'] = 'b' * 64
        m.write_json(labelled / 'provenance.json', value)
    label_files = {name: m.sha256_file(labelled / name) for name in ('manifest.json', 'samples.jsonl', 'features.npz', 'provenance.json')}
    m.write_json(labelled / 'status.json', dict(status='failed' if defect == 'incomplete' else 'completed',
        action='attach-labels', queries=len(data['rows']), output_sha256=label_files))
    if defect == 'unbound_provenance':
        value = m.load_json(exported / 'provenance.json'); value['cache_split'] = 'dev'
        m.write_json(exported / 'provenance.json', value)
    bindings = {str(labelled / name): label_files[name] for name in ('manifest.json', 'samples.jsonl', 'features.npz')}
    bindings[str(exported / 'status.json')] = export_status_sha
    return labelled / 'manifest.json', data, exported / 'status.json', split, cp, training, bindings


@pytest.mark.parametrize('defect', [None, 'checkpoint', 'protocol', 'parent', 'incomplete', 'unbound_provenance', 'target_source'])
def test_receipt_lineage_binds_consumed_features_without_rehashing_npz(tmp_path, monkeypatch, defect):
    args = lineage_fixture(tmp_path, defect)
    real_hash = m.sha256_file
    def guarded_hash(path):
        assert Path(path).suffix != '.npz', 'Lineage needlessly reopened a large NPZ'
        return real_hash(path)
    monkeypatch.setattr(m, 'sha256_file', guarded_hash)
    if defect:
        with pytest.raises(ValueError): m.verify_lineage(*args)
    else:
        value = m.verify_lineage(*args)
        assert value['checkpoint_sha256'] == 'c' * 64 and value['split'] == 'fit'
        assert value['cache_root'] == str(tmp_path / 'cache')

"""Meaningful contract, optimization and paired-evaluation checks for new baselines."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/rebuild'))
import train_foundation_baseline as trainer
import compare_foundation_baselines as evaluation
from oa_cxr.io import load_json, write_json, sha256_file


def dataset(split, count=2, dim=7):
    rows = []; X = np.zeros((count * 39, dim), dtype=np.float32)
    for source in range(count):
        for variant in range(13):
            for fi, finding in enumerate(trainer.FINDINGS):
                index = len(rows); X[index, -3 + fi] = 1
                rows.append(dict(row_id=f'{split}-{index}', source_id=f'{split}-{source}',
                    input_id=f'{split}-{source}-{variant}', finding=finding, split=split, dataset='synthetic',
                    split_group_id=f'{split}-{source}', source_image_sha256=f'{split}-{source}', variant_index=variant))
    return dict(rows=rows, X=X, y=np.linspace(0, 1, len(rows), dtype=np.float32), available=np.ones(len(rows), dtype=bool),
        manifest=dict(feature_names=[f'x{i}' for i in range(dim - 3)] + ['finding_' + f for f in trainer.FINDINGS], feature_protocol_sha256='a' * 64))


def test_variable_dimensions_and_disjoint_population():
    for dimension in (515, 771, 1253):
        result = trainer.validate_pair(dataset('fit', dim=dimension), dataset('dev', dim=dimension), fit_count=2, dev_count=2)
        assert result['fit_queries'] == 78 and result['source_image_group_disjoint']


@pytest.mark.parametrize('field', ['source_image_sha256', 'split_group_id'])
def test_fit_dev_overlap_rejected(field):
    fit, dev = dataset('fit'), dataset('dev')
    for row in dev['rows'][:39]: row[field] = fit['rows'][0][field]
    with pytest.raises(ValueError, match='overlap'): trainer.validate_pair(fit, dev, fit_count=2, dev_count=2)


@pytest.mark.parametrize('corruption', ['missing', 'onehot', 'variant', 'unavailable', 'protocol', 'split'])
def test_training_contract_rejects_corruption(corruption):
    fit, dev = dataset('fit'), dataset('dev')
    if corruption == 'missing': fit['X'] = fit['X'][:-1]
    elif corruption == 'onehot': fit['X'][0, -3:] = 0
    elif corruption == 'variant': fit['rows'][0]['variant_index'] = 13
    elif corruption == 'unavailable': fit['available'][0] = False
    elif corruption == 'protocol': dev['manifest']['feature_protocol_sha256'] = 'b' * 64
    else: fit['rows'][0]['split'] = 'test'
    with pytest.raises(ValueError): trainer.validate_pair(fit, dev, fit_count=2, dev_count=2)


def test_config_exact_population_and_hash(tmp_path):
    for split in ('fit', 'dev'): write_json(tmp_path / f'{split}.json', {'test': split})
    config = dict(schema=trainer.SCHEMA, method='tabm', expected_sources={'fit': 19086, 'dev': 2609},
        sources={f'{s}_manifest': dict(path=f'{s}.json', sha256=sha256_file(tmp_path / f'{s}.json')) for s in ('fit', 'dev')})
    path = tmp_path / 'config.json'; write_json(path, config)
    assert trainer.configuration(path, sha256_file(path))[0] == config
    with pytest.raises(ValueError, match='differs'): trainer.configuration(path, '0' * 64)
    config['expected_sources']['fit'] = 100; write_json(path, config)
    with pytest.raises(ValueError, match='population'): trainer.configuration(path, sha256_file(path))


def test_official_tabm_settings_separate_from_matched_visual_heads():
    visual, tabm = trainer.method_settings('dinov2'), trainer.method_settings('tabm')
    assert visual['epochs'] == tabm['epochs'] == 30 and visual['batch_size'] == tabm['batch_size'] == 2048
    assert (tabm['learning_rate'], tabm['weight_decay'], tabm['microbatch_size']) == (.002, .0003, 256)
    assert visual['learning_rate'] == .001 and visual['microbatch_size'] == 2048


def records():
    truth = dataset('test', 1)['rows'][:3]
    return truth, [dict(row, method='biomedclip', status='success', score=.5, failure_reason=None) for row in truth]


@pytest.mark.parametrize('corruption', ['duplicate', 'missing', 'source', 'method', 'nan', 'range', 'failure'])
def test_prediction_alignment_rejects_invalid_receipt(corruption):
    truth, rows = records()
    if corruption == 'duplicate': rows.append(rows[0])
    elif corruption == 'missing': rows.pop()
    elif corruption == 'source': rows[0]['source_id'] = 'another-source'
    elif corruption == 'method': rows[0]['method'] = 'dinov2'
    elif corruption == 'nan': rows[0]['score'] = float('nan')
    elif corruption == 'range': rows[0]['score'] = 1.2
    else: rows[0].update(status='failed', score=None, failure_reason=None)
    with pytest.raises(ValueError): evaluation.align(truth, rows, 'biomedclip')


def test_alignment_orders_by_identity_and_preserves_failure():
    truth, rows = records(); rows[0].update(status='failed', score=None, failure_reason='encoder failed')
    values, ordered = evaluation.align(truth, rows[::-1], 'biomedclip')
    assert np.isnan(values[0]) and ordered[0]['row_id'] == truth[0]['row_id']


def test_common_denominator_includes_all_methods_and_failures():
    truth, rows = records(); y = np.array([.1, .95, .8]); mask = np.ones(3, dtype=bool)
    vectors = {m: y.copy() for m in evaluation.METHODS}; vectors['tabm'][0] = np.nan
    pred = {m: copy.deepcopy(rows) for m in evaluation.METHODS}; pred['tabm'][0]['failure_reason'] = 'empty'
    result, common = evaluation.point_group(truth, y, vectors, pred, mask)
    assert common.tolist() == [False, True, True] and result['common_valid_queries'] == 2
    assert result['coverage']['tabm']['failed_queries'] == 1
    assert result['coverage']['tabm']['automatic_review_for_missing_score'] == 1
    assert all(v['mae'] == 0 for v in result['metrics'].values())


def test_group_bootstrap_zero_error_and_known_paired_difference():
    rows = dataset('test', 2)['rows']; y = np.full(len(rows), .5)
    vectors = {m: y.copy() + (.1 if m != 'oa_cxr' else 0) for m in evaluation.METHODS}
    mask = np.ones(len(rows), dtype=bool)
    result = evaluation.regression_bootstrap(rows, y, vectors, mask, mask, replicates=40)
    assert result['original_groups'] == 2
    assert result['individual']['oa_cxr']['mae']['ci95'] == [0, 0]
    ci = result['paired_oa_cxr_minus_baseline']['tabm']['mae']['ci95']
    assert ci == pytest.approx([-.1, -.1])


def test_macro_has_equal_dataset_weights_and_no_missing_source_ranking():
    datasets = ['a', 'b', 'c', 'd']
    groups = {d: dict(metrics={method: dict(mae=(i + 1) / 10) for method in evaluation.METHODS}) for i, d in enumerate(datasets)}
    result = evaluation.macro_mae(groups, datasets)
    assert result['methods']['oa_cxr']['mae'] == .25 and result['methods']['oa_cxr']['rank'] == 1
    groups['d']['metrics']['tabm']['mae'] = None
    result = evaluation.macro_mae(groups, datasets)
    assert result['methods']['tabm']['mae'] is None and result['methods']['tabm']['rank'] is None


def test_no_report_claims_remains_undefined():
    data = dataset('test'); denominators = [dict(input_id='some', dataset='synthetic', split_group_id='a',
        generation_status='success', eligible_claims=0, possibly_truncated=False)]
    result = evaluation.report_points([], denominators, data['rows'], data['y'],
        {m: data['y'] for m in evaluation.METHODS}, ['synthetic'])
    group = result['groups']['synthetic']
    assert group['nominal_queries'] == 0 and group['metrics']['oa_cxr']['bad_proxy_auroc'] is None
    assert group['ranks']['oa_cxr']['mae'] is None
    assert group['report_denominators']['zero_eligible_reports'] == 1


def test_report_target_or_source_mismatch_rejected():
    data = dataset('test'); row = data['rows'][0]
    claim = dict(row, claim_id='c', retention=.777)
    with pytest.raises(ValueError, match='target'):
        evaluation.report_points([claim], [], data['rows'], data['y'], {m: data['y'] for m in evaluation.METHODS}, ['synthetic'])


def test_identical_visual_readout_and_raw_loss():
    torch = pytest.importorskip('torch')
    from baselines.foundation_regressors import make_regressor, raw_l1_loss
    from oa_cxr.rebuild.neural_readout import NeuralReadout
    torch.manual_seed(17); baseline = make_regressor('biomedclip', 7)
    torch.manual_seed(17); reference = NeuralReadout(7, width=256, inner_width=256, blocks=2, dropout=0)
    X = torch.randn(9, 7); baseline.fit_standardizer(X, split='fit'); reference.fit_standardizer(X, split='fit')
    assert all(torch.equal(baseline.state_dict()[k], v) for k, v in reference.state_dict().items())
    assert torch.equal(baseline(X)['retention'], reference(X)['retention'])
    pred = torch.tensor([[2., 0.], [.5, -.5]], requires_grad=True)
    loss = raw_l1_loss({'loss_prediction': pred}, torch.tensor([1., 0.]))
    assert loss.item() == .75
    loss.backward(); assert torch.count_nonzero(pred.grad) == 4


def test_fit_standardizer_never_updates_on_dev():
    torch = pytest.importorskip('torch')
    from baselines.foundation_regressors import make_regressor
    model = make_regressor('dinov2', 5); fit = torch.randn(11, 5)
    model.fit_standardizer(fit, split='fit'); mean = model.standardizer.mean.clone()
    model(torch.ones(5, 5) * 100)
    assert torch.equal(mean, model.standardizer.mean)
    with pytest.raises(ValueError): model.fit_standardizer(fit, split='dev')


def test_microbatch_weights_and_full_epoch_budget(tmp_path):
    torch = pytest.importorskip('torch')
    from oa_cxr.rebuild.neural_readout import NeuralReadout
    torch.set_num_threads(1); torch.manual_seed(17)
    X = torch.randn(19, 5); y = torch.rand(19); model = NeuralReadout(5, width=8, inner_width=8, blocks=1)
    model.fit_standardizer(X, split='fit')
    settings = dict(trainer.SETTINGS, epochs=2, batch_size=8, microbatch_size=3)
    result = trainer.train_epochs(model, X, y, X[:5], y[:5], tmp_path, 'a' * 64, 'biomedclip', settings=settings)
    assert result['optimizer_steps'] == 6 and result['updates_per_epoch'] == 3
    assert load_json(tmp_path / 'epoch_02.json')['full_fit_queries_seen'] == 19
    assert result['changed_parameter_elements']['best'] > 0 and result['changed_parameter_elements']['last'] > 0
    for endpoint in ('best', 'last'):
        assert load_json(tmp_path / f'{endpoint}_model.json')['state_sha256'] == sha256_file(tmp_path / f'{endpoint}_state.pt')


def test_official_tabm_member_axis_and_architecture():
    torch = pytest.importorskip('torch'); pytest.importorskip('tabm')
    from baselines.foundation_regressors import make_regressor
    torch.set_num_threads(1); model = make_regressor('tabm', 7); X = torch.randn(5, 7)
    model.fit_standardizer(X, split='fit'); model.eval(); output = model(X)
    assert output['loss_prediction'].shape == (5, 32)
    assert torch.allclose(output['retention'], output['loss_prediction'].mean(1).clamp(0, 1))
    assert model.config()['d_block'] == 512 and model.config()['dropout'] == .1

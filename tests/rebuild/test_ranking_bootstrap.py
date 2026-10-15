"""Check weighted statistics against literal cluster resampling and sklearn."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.integrate import trapezoid
from oa_cxr.rebuild.metrics import metrics
from oa_cxr.rebuild.ranking_bootstrap import RankingCache, COVERAGES


@pytest.mark.parametrize('tied', [False, True])
def test_frequency_weights_match_expanded_existing_metrics(tied):
    rng = np.random.default_rng(13)
    y = rng.uniform(size=157)
    groups = rng.integers(0, 17, len(y))
    scores = rng.normal(size=len(y))
    if tied: scores = np.round(scores, 0)
    cache = RankingCache(y, scores, groups, 19)  # includes two zero-row groups
    for _ in range(25):
        weights = np.bincount(rng.integers(0, 19, 19), minlength=19)
        indices = np.repeat(np.arange(len(y)), weights[groups])
        actual = cache.evaluate(weights)
        expected = metrics(y[indices], scores[indices], curve=True)
        values = [expected[k] for k in ('bad_proxy_auroc', 'bad_proxy_auprc', 'aurc', 'excess_aurc')]
        values += [r['conditional_bad_risk'] for r in expected['matched_risk_coverage']]
        np.testing.assert_allclose(actual, values, atol=2e-14, rtol=2e-13)
        assert actual[0] == pytest.approx(roc_auc_score(y[indices] < .9, -scores[indices]))
        assert actual[1] == pytest.approx(average_precision_score(y[indices] < .9, -scores[indices]))


def test_constant_score_ties_are_random_order_expectation():
    y = np.array([.1, 1., 1., .2]); groups = np.arange(4)
    cache = RankingCache(y, np.ones(4), groups, 4)
    values = cache.evaluate(np.ones(4))
    assert values[0] == .5 and values[1] == .5 and values[2] == .5
    np.testing.assert_allclose(values[4:], .5)


@pytest.mark.parametrize('truth,aurc', [([1., 1.], 0.), ([.1, .2], 1.)])
def test_single_class_auc_is_undefined(truth, aurc):
    cache = RankingCache(truth, [.1, .5], [0, 1], 2)
    values = cache.evaluate([1, 1])
    assert np.isnan(values[:2]).all()
    assert values[2] == pytest.approx(aurc)
    assert values[3] == pytest.approx(0)


def test_zero_contributing_groups_remain_undefined():
    cache = RankingCache([.1, 1.], [.2, .4], [0, 1], 3)
    assert np.isnan(cache.evaluate([0, 0, 3])).all()
    with pytest.raises(ValueError): cache.evaluate([.5, 1, 1])


def test_curves_follow_low_retention_positive_direction():
    cache = RankingCache([.1, .2, 1., 1.], [.1, .2, .8, .9], np.arange(4), 4)
    curves = cache.curves(np.ones(4))
    assert cache.evaluate(np.ones(4))[0] == 1
    assert trapezoid(curves['tpr'], curves['fpr']) == 1
    assert curves['risk'][-1] == .5
    assert len(curves['risk']) == 201

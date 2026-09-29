import numpy as np
import pytest
from sklearn.metrics import roc_auc_score, average_precision_score
from oa_cxr.rebuild.metrics import metrics,bootstrap,case_clusters
@pytest.mark.parametrize('seed',range(8))
def test_rank_metrics_match_independent_implementations_with_ties(seed):
    rng=np.random.default_rng(seed);y=rng.uniform(.6,1,160);s=rng.integers(-4,5,160);bad=y<.9
    result=metrics(y,s)
    assert result['bad_proxy_auroc']==pytest.approx(roc_auc_score(bad,-s))
    assert result['bad_proxy_auprc']==pytest.approx(average_precision_score(bad,-s))
    assert result['mae'] is None


def test_perfect_ranking_and_oracle_excess():
    result=metrics([1,.4,1,.6],[1,.1,.9,.2])
    assert result['bad_proxy_auroc']==1 and result['bad_proxy_auprc']==1
    assert result['excess_aurc']==0
    assert result['matched_risk_coverage'][0]['expected_bad_retained']==0


def test_constant_score_ties_do_not_use_labels():
    result=metrics([1,0,1,0],[2]*4)
    assert result['aurc']==.5 and result['bad_proxy_auroc']==.5 and result['bad_proxy_auprc']==.5
    assert result['matched_risk_coverage'][0]['expected_good_lost']==1


@pytest.mark.parametrize('truth',[[],[1,1],[0,0]])
def test_undefined_discrimination_explicit(truth):
    r=metrics(truth,[.5]*len(truth),regression=True)
    assert r['bad_proxy_auroc'] is None and r['bad_proxy_auprc'] is None and r['spearman'] is None


def test_regression_error_and_rank_scale():
    r=metrics([0,.5,1],[.1,.5,.9],regression=True)
    assert r['mae']==pytest.approx(.2/3)
    assert r['rmse']==pytest.approx(np.sqrt(.02/3)) and r['spearman']==1
    with pytest.raises(ValueError):metrics([1],[-1],regression=True)


def test_case_bootstrap_preserves_zero_eligible_cases_and_shared_draws():
    rows=[{'dataset':'d','split_group_id':'a','retention':.5}]
    pool=[('d','a'),('d','b'),('d','c')]
    assert [len(x) for x in case_clusters(rows,pool)]==[1,0,0]
    result=bootstrap(rows,{'ours':[.4],'same':[.4]},set(),'ours',pool,replicates=100,seed=17)
    assert result['original_cases']==3 and result['contributing_cases']==1
    d=result['paired_selected_minus_comparator']['same']['aurc']
    assert d['ci95']==[0,0] and 0<d['undefined_replicates']<100
    with pytest.raises(ValueError):case_clusters(rows,[('d','b')])



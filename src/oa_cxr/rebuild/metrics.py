"""Anatomical-retention metrics with label-blind ties and paired source-group resampling."""
import math
import numpy as np
from scipy.stats import rankdata

COVERAGES=(.5,.6,.7,.8,.9,1.)
METRICS=('mae','rmse','spearman','bad_proxy_auroc','bad_proxy_auprc','aurc','excess_aurc')


def metrics(truth,scores,*,regression=False,curve=True):
    y,s=np.asarray(truth,dtype=float),np.asarray(scores,dtype=float)
    if y.ndim!=1 or s.shape!=y.shape or not np.isfinite(y).all() or not np.isfinite(s).all():
        raise ValueError('Metrics need aligned finite one-dimensional vectors')
    if np.any((y<0)|(y>1)):raise ValueError('Target outside retention range')
    if regression and np.any((s<0)|(s>1)):raise ValueError('Regressor outside retention range')
    n=len(y);bad=y<.9;b=int(bad.sum());g=n-b
    out={k:None for k in METRICS};out.update(rows=n,bad_proxy_rows=b,good_proxy_rows=g,matched_risk_coverage=[])
    if not n:return out
    order=np.argsort(-s,kind='stable');ss=s[order];bb=bad[order].astype(float)
    starts=np.r_[0,np.flatnonzero(ss[1:]!=ss[:-1])+1];ends=np.r_[starts[1:],n]
    lengths=ends-starts;counts=np.add.reduceat(bb,starts);good=lengths-counts
    prior=np.cumsum(counts)-counts
    ranks=np.arange(n)-np.repeat(starts,lengths)+1
    prefix=np.repeat(prior,lengths)+ranks*np.repeat(counts/lengths,lengths)
    oracle=np.maximum(0,np.arange(1,n+1)-g)
    out['aurc']=float(np.mean(prefix/np.arange(1,n+1)))
    out['excess_aurc']=out['aurc']-float(np.mean(oracle/np.arange(1,n+1)))
    if b and g:
        out['bad_proxy_auroc']=float(np.sum(counts*(np.cumsum(good)-good+good/2))/(b*g))
        # AP uses whole score groups, matching sklearn average_precision_score.
        out['bad_proxy_auprc']=float(np.sum(counts[::-1]*np.cumsum(counts[::-1])/np.cumsum(lengths[::-1]))/b)
    if regression:
        out['mae']=float(np.mean(abs(y-s)));out['rmse']=float(np.sqrt(np.mean((y-s)**2)))
        if np.ptp(y)>0 and np.ptp(s)>0:out['spearman']=float(np.corrcoef(rankdata(y),rankdata(s))[0,1])
    if curve:
        for coverage in COVERAGES:
            kept=math.ceil(n*coverage);bad_kept=float(prefix[kept-1]);lost=g-kept+bad_kept
            out['matched_risk_coverage'].append({'target_coverage':coverage,'actual_coverage':kept/n,
                'retained_rows':kept,'expected_bad_retained':bad_kept,'expected_good_lost':max(0.,lost),
                'conditional_bad_risk':bad_kept/kept,'bad_per_original_query':bad_kept/n,
                'good_loss_fraction':max(0.,lost)/g if g else None})
    return out


def case_clusters(records,case_pool):
    cases=sorted(set(case_pool));groups={c:[] for c in cases}
    for i,r in enumerate(records):
        key=(r['dataset'],r['split_group_id'])
        if key not in groups:raise ValueError('Scored case outside original denominator')
        groups[key].append(i)
    return [np.asarray(groups[c],dtype=int) for c in cases]


def interval(values,replicates):
    finite=[float(v) for v in values if v is not None and math.isfinite(v)]
    return {'valid_replicates':len(finite),'undefined_replicates':replicates-len(finite),
        'ci95':[float(x) for x in np.quantile(finite,[.025,.975])] if finite else None}


def bootstrap(records,vectors,regressors,selected,case_pool,*,replicates=1000,seed=17):
    clusters=case_clusters(records,case_pool);ncase=len(clusters)
    if selected not in vectors or replicates<1:raise ValueError('Invalid bootstrap protocol')
    out={'original_cases':ncase,'contributing_cases':sum(bool(len(x)) for x in clusters),
        'replicates':replicates,'seed':seed,'zero_eligible_cases_included':True,
        'interval':'paired percentile 95%; frozen models; no multiplicity correction',
        'direction':'selected minus comparator; negative favors selected for MAE/RMSE/AURC/eAURC',
        'individual':{},'paired_selected_minus_comparator':{}}
    y=np.asarray([r['retention'] for r in records]);vectors={k:np.asarray(v) for k,v in vectors.items()}
    point={k:metrics(y,v,regression=k in regressors,curve=False) for k,v in vectors.items()}
    keys={k:[m for m in METRICS if point[k][m] is not None] for k in vectors}
    absolute={k:{m:[] for m in keys[k]} for k in vectors}
    diff={k:{m:[] for m in keys[k] if m in keys[selected]} for k in vectors if k!=selected}
    if ncase>=2 and len(y):
        rng=np.random.default_rng(seed)
        for _ in range(replicates):
            index=np.concatenate([clusters[j] for j in rng.integers(0,ncase,ncase)])
            estimates={k:metrics(y[index],v[index],regression=k in regressors,curve=False) for k,v in vectors.items()}
            for k in absolute:
                for m in absolute[k]:absolute[k][m].append(estimates[k][m])
            for k in diff:
                for m in diff[k]:
                    a,b=estimates[selected][m],estimates[k][m]
                    diff[k][m].append(a-b if a is not None and b is not None else None)
    for k,mm in absolute.items():
        out['individual'][k]={m:{'point':point[k][m],**interval(v,replicates)} for m,v in mm.items()}
    for k,mm in diff.items():
        out['paired_selected_minus_comparator'][k]={m:{'point':point[selected][m]-point[k][m],**interval(v,replicates)} for m,v in mm.items()}
    return out

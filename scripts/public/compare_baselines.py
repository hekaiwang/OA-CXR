"""Compare aligned baseline scores on a common cohort, with paired group bootstrap."""
from common import *
import argparse
import math
import numpy as np
from oa_cxr.rebuild.metrics import metrics,bootstrap
import export as original


REGRESSORS={'oa_cxr','hgb','ridge','xgboost','catboost','tabm','biomedclip','chexzero','dinov2'}


def align(rows,records):
    lookup={(r['input_id'],r['finding']):i for i,r in enumerate(rows)}
    if len(lookup)!=len(rows):raise ValueError('Duplicate truth queries')
    values=np.full(len(rows),np.nan);seen=set()
    for row in records:
        key=(row['input_id'],row['finding'])
        if key not in lookup or key in seen:raise ValueError('Unknown or duplicate prediction query')
        seen.add(key);truth=rows[lookup[key]]
        for name in ('dataset','source_id','source_image_sha256'):
            if row.get(name)!=truth[name]:raise ValueError('Prediction source identity differs')
        score=row.get('score')
        if score is not None:
            if row.get('status')!='success' or type(score) not in (int,float) or not math.isfinite(score):
                raise ValueError('Invalid successful score')
            values[lookup[key]]=score
        elif row.get('status')=='success':raise ValueError('A successful prediction cannot have a null score')
    if len(seen)!=len(rows):raise ValueError('Missing prediction rows: retain failures explicitly')
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--split',required=True)
    p.add_argument('--prediction',action='append',required=True,metavar='METHOD=JSONL')
    p.add_argument('--reference',required=True);p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.bootstrap<0:raise ValueError('Nonnegative bootstrap count required')
    cache=original.cache_inputs(a.cache,a.split,labels=True)
    rows=list(original.query_records(cache['rows'],a.split))
    y=np.load(cache['paths']['targets.npy'],allow_pickle=False).reshape(-1)
    if not np.isfinite(y).all():raise ValueError('Finite independent targets required')
    vectors={};bindings={}
    for item in a.prediction:
        method,path=item.split('=',1)
        if method in vectors:raise ValueError('Duplicate method')
        path=Path(path);vectors[method]=align(rows,read_jsonl(path));bindings[method]=sha256_file(path)
    if a.reference not in vectors:raise ValueError('Reference predictions are required')
    out=fresh(a.output);groups={}
    for dataset in sorted({r['dataset'] for r in rows}):
        nominal=np.array([r['dataset']==dataset for r in rows]);common=nominal.copy()
        for v in vectors.values():common &= np.isfinite(v)
        if not common.any():raise ValueError('No common-valid queries in '+dataset)
        scores={name:v[common] for name,v in vectors.items()}
        records=[dict(row,retention=float(target)) for row,target,keep in zip(rows,y,common) if keep]
        pool=sorted({(r['dataset'],r['split_group_id']) for r,keep in zip(rows,nominal) if keep})
        summary=dict(nominal_queries=int(nominal.sum()),common_queries=int(common.sum()),
            per_method_success={name:int(np.isfinite(v[nominal]).sum()) for name,v in vectors.items()},
            point_metrics={name:metrics(y[common],v,regression=name in REGRESSORS) for name,v in scores.items()})
        if a.bootstrap:
            summary['bootstrap']=bootstrap(records,scores,REGRESSORS,a.reference,pool,replicates=a.bootstrap,seed=17)
        groups[dataset]=summary
    write_json(out/'metrics.json',dict(schema='oa-cxr-public-baseline-comparison-v1',datasets=groups,
        prediction_sha256=bindings,reference=a.reference,seed=17,
        population='common successful queries; complete nominal and per-method counts retained',
        private_data_or_record_level_outputs_published=False))


if __name__=='__main__':main()

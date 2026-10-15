"""Evaluate fixed generated claim occurrences; retain failed and zero-claim reports."""
from common import *
import argparse
from collections import Counter
import math
import numpy as np
from baselines.reporting import checked_run,report_claims,digest
from oa_cxr.rebuild.metrics import metrics,bootstrap
from baseline_inputs import selected_report_sources
from compare_baselines import align
import export as original


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--split',required=True)
    p.add_argument('--generation',type=Path,required=True)
    p.add_argument('--image-score',action='append',default=[],metavar='METHOD=JSONL')
    p.add_argument('--report-score',action='append',default=[],metavar='METHOD=JSONL')
    p.add_argument('--reference',required=True);p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.bootstrap<0:raise ValueError('Invalid replicate count')
    if a.split!='test' and not a.split.startswith('external'):raise ValueError('Evaluation splits only')
    cache=original.cache_inputs(a.cache,a.split,labels=True)
    targets=np.load(cache['paths']['targets.npy'],allow_pickle=False).reshape(-1)
    truth={(r['input_id'],r['finding']):dict(r,retention=float(y))
           for r,y in zip(original.query_records(cache['rows'],a.split),targets)}
    reports,_,_=checked_run(a.generation,'reports.jsonl','generation')
    selected=selected_report_sources(cache['rows'])
    expected_inputs={r['input_id'] for r in truth.values() if (r['dataset'],r['source_id']) in selected}
    if {r['input']['input_id'] for r in reports}!=expected_inputs:
        raise ValueError('Generated reports must retain the complete fixed selection, including failures')
    records=[];population={};seen=set()
    for report in reports:
        inp=report['input'];key=inp['input_id']
        if key in seen:raise ValueError('Duplicate generated input')
        seen.add(key);source=truth[(key,original.FINDINGS[0])]
        if any(inp[k]!=source[k] for k in ('source_id','source_image_sha256','dataset','split')):
            raise ValueError('Report and independent target source differ')
        population[key]=source
        for claim in report_claims(report):
            row=truth[(key,claim['finding'])]
            records.append(dict(row,claim_id=claim['claim_id'],
                generated_findings_sha256=digest(report['generated_findings'])))
    vectors={};bindings={}
    for is_report,items in [(False,a.image_score),(True,a.report_score)]:
        for item in items:
            method,path=item.split('=',1)
            if method in vectors:raise ValueError('Duplicate method')
            rows=read_jsonl(path);index={}
            if not is_report:align(list(truth.values()),rows)
            for row in rows:
                key=(row['input_id'],row['claim_id'] if is_report else row['finding'])
                if key in index:raise ValueError('Duplicate scored occurrence')
                index[key]=row
            if is_report and set(index)!={(r['input_id'],r['claim_id']) for r in records}:
                raise ValueError('Report scores must match the exact eligible claim roster')
            values=[]
            for row in records:
                key=(row['input_id'],row['claim_id'] if is_report else row['finding'])
                if key not in index:raise ValueError('Missing claim score; represent failures explicitly')
                score=index[key]
                if any(score.get(k)!=row[k] for k in ('source_id','source_image_sha256','dataset')):
                    raise ValueError('Score source identity differs')
                if is_report and score.get('generated_findings_sha256')!=row['generated_findings_sha256']:
                    raise ValueError('Score belongs to another generated report')
                if is_report and score.get('finding')!=row['finding']:
                    raise ValueError('Scored finding differs from the generated claim')
                value=score.get('score')
                if value is not None and (score.get('status')!='success' or type(value) not in (int,float) or not math.isfinite(value)):
                    raise ValueError('Invalid score')
                if value is None and score.get('status')=='success':raise ValueError('Successful null score')
                values.append(np.nan if value is None else value)
            vectors[method]=np.asarray(values,dtype=float);bindings[method]=sha256_file(path)
    if a.reference not in vectors:raise ValueError('Reference scores required')
    out=fresh(a.output);results={}
    for dataset in sorted({r['dataset'] for r in population.values()}):
        nominal=np.array([r['dataset']==dataset for r in records],dtype=bool);common=nominal.copy()
        for score in vectors.values():common &= np.isfinite(score)
        selected=[r for r,keep in zip(records,common) if keep]
        y=np.array([r['retention'] for r in selected]);scores={m:s[common] for m,s in vectors.items()}
        pool=sorted({(r['dataset'],r['split_group_id']) for r in population.values() if r['dataset']==dataset})
        result=dict(reports=sum(r['dataset']==dataset for r in population.values()),eligible_claims=int(nominal.sum()),
            common_scored_claims=int(common.sum()),
            per_method_scored={m:int(np.isfinite(s[nominal]).sum()) for m,s in vectors.items()},
            point_metrics={m:metrics(y,s,regression=False) for m,s in scores.items()})
        if a.bootstrap:result['bootstrap']=bootstrap(selected,scores,set(),a.reference,pool,replicates=a.bootstrap,seed=17)
        results[dataset]=result
    write_json(out/'metrics.json',dict(schema='oa-cxr-public-report-comparison-v1',datasets=results,
        generation_status_sha256=sha256_file(a.generation/'status.json'),score_sha256=bindings,
        generation_counts=dict(Counter(r['status'] for r in reports)),
        full_report_denominator=len(reports),eligible_claims=len(records),
        scope='common scored occurrences for exactly the supplied methods; different common cohorts are not merged',
        clinical_report_correctness_evaluated=False))


if __name__=='__main__':main()

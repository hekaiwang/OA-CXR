"""Fresh common-population comparisons; never mix score units in regression MAE."""
from __future__ import annotations
import argparse
import math
import re
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from oa_cxr.io import load_json, read_jsonl, sha256_file, stable_hash, write_json, exclusive_writer
from oa_cxr.rebuild.metrics import metrics, bootstrap

SELECTED='direct_anatomy_retention'
REGRESSORS={SELECTED,'hgb','xgboost','catboost','ridge'}
VISUAL={'chexpert','autoqc','biovil_t'}
IDENTITY=('input_id','source_id','dataset','split','finding','source_image_sha256')
PRESENTED_IDENTITY=('input_id','source_id','dataset','split','source_image_sha256',
                    'presented_image_path','presented_image_sha256')


def key(row):return row['input_id'],row['finding']


def verified_output(directory,filename):
    root=Path(directory).resolve();state=load_json(root/'status.json')
    if state.get('status') not in ('completed','completed_with_failures'):
        raise ValueError('Cannot evaluate an incomplete stage: '+str(root))
    path=root/filename
    if state.get('output_sha256',{}).get(filename)!=sha256_file(path):
        raise ValueError('Unbound or changed output: '+str(path))
    return path,state


def load_truth(directory):
    sample_path,_=verified_output(directory,'samples.jsonl')
    label_path,_=verified_output(directory,'labels.npz')
    rows=read_jsonl(sample_path)
    if len({key(r) for r in rows})!=len(rows):raise ValueError('Duplicate target identity')
    with np.load(label_path,allow_pickle=False) as z:
        y=z['y'];ids=z['row_ids'].tolist()
    if ids!=[r['row_id'] for r in rows] or y.shape!=(len(rows),):raise ValueError('Target array identity mismatch')
    if not np.isfinite(y).all() or (y<0).any() or (y>1).any():raise ValueError('Invalid target')
    if any(r['split'] not in ('test','external') for r in rows):raise ValueError('Leaderboard requires heldout roles only')
    groups={}
    for row in rows:
        previous=groups.setdefault((row['dataset'],row['split_group_id']),row['split'])
        if previous!=row['split']:raise ValueError('Group crosses evaluation roles')
    return rows,y


def align_scores(truth,predictions,method,canonical=None):
    expected={key(r):r for r in truth};seen={};scores=[]
    for row in predictions:
        k=key(row)
        if k not in expected or k in seen:raise ValueError('Unexpected or duplicate prediction')
        reference=expected[k]
        if any(row.get(name)!=reference[name] for name in IDENTITY):raise ValueError('Prediction/source identity mismatch')
        if row.get('method')!=method:raise ValueError('Method identity mismatch')
        if method in VISUAL:
            if canonical is None or row['input_id'] not in canonical:
                raise ValueError('Visual scores require bound canonical presented inputs')
            if any(row.get(name)!=canonical[row['input_id']][name] for name in PRESENTED_IDENTITY):
                raise ValueError('Visual prediction/presented pixel identity mismatch')
        value=row.get('score');success=row.get('status')=='success'
        if success:
            if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Nonfinite/invalid score')
            if method in REGRESSORS and not 0<=value<=1:raise ValueError('Regressor outside [0,1]')
        elif row.get('status')!='failed' or value is not None or not row.get('failure_reason'):
            raise ValueError('Failure needs null score and explicit reason')
        seen[k]=row
    if set(seen)!=set(expected):raise ValueError('Incomplete nominal prediction denominator')
    for row in truth:
        value=seen[key(row)]['score'];scores.append(np.nan if value is None else value)
    return np.asarray(scores,dtype=float),[seen[key(r)] for r in truth]


def _metadata(directory,name,bindings):
    path,state=verified_output(directory,name)
    bindings[str(path)]=sha256_file(path)
    receipt=path.parent/'status.json';bindings[str(receipt)]=sha256_file(receipt)
    return load_json(path),state


def _manifest_file(root,spec):
    if not isinstance(spec,dict) or not isinstance(spec.get('path'),str) or not re.fullmatch('[0-9a-f]{64}',spec.get('sha256','')):
        raise ValueError('Invalid manifest file binding')
    path=(root/spec['path']).resolve()
    if not path.is_relative_to(root):raise ValueError('Manifest file escapes its artifact directory')
    return path


def verify_common_inputs(labelled,spec,truth):
    """Check accepted stage receipts; do not rescan multi-GB feature/image arrays."""
    if not isinstance(spec.get('presentation_directory'),str) or not spec['presentation_directory']:
        raise ValueError('A bound presentation_directory is required')
    labelled=Path(labelled).resolve()
    own=Path(spec['methods'][SELECTED]['directory']).resolve()
    presentation=Path(spec['presentation_directory']).resolve()
    bindings={}
    own_provenance,own_state=_metadata(own,'provenance.json',bindings)
    own_manifest,_=_metadata(own,'manifest.json',bindings)
    feature_protocol,_=_metadata(own,'feature_protocol.json',bindings)
    label_provenance,label_state=_metadata(labelled,'provenance.json',bindings)
    label_manifest,_=_metadata(labelled,'manifest.json',bindings)
    protocol,presentation_state=_metadata(presentation,'protocol.json',bindings)
    inputs_path,_=verified_output(presentation,'inputs.jsonl')
    bindings[str(inputs_path)]=sha256_file(inputs_path)
    if own_state.get('action')!='export' or label_state.get('action')!='attach-labels':
        raise ValueError('Expected completed model export and independent label attachment')
    if label_provenance.get('source_export_status_sha256')!=sha256_file(own/'status.json'):
        raise ValueError('Labels were attached to a different model export')
    if own_state.get('output_sha256',{}).get('samples.jsonl')!=label_state.get('output_sha256',{}).get('samples.jsonl'):
        raise ValueError('Own export and labelled query identities differ')
    feature_sha=stable_hash({k:v for k,v in feature_protocol.items() if k!='artifact_sha256'})
    if (feature_protocol.get('artifact_sha256')!=feature_sha or
            any(document.get('feature_protocol_sha256')!=feature_sha for document in
                (own_provenance,label_provenance,own_manifest,label_manifest))):
        raise ValueError('Learned feature protocol differs across accepted artifacts')
    for manifest in (own_manifest,label_manifest):
        if (manifest.get('schema')!='oa-cxr-tabular-dataset-v1' or manifest.get('target')!='anatomical_retention'
                or manifest.get('label_free_features') is not True):
            raise ValueError('Invalid label-free feature manifest')
    if own_manifest.get('feature_names')!=label_manifest.get('feature_names'):
        raise ValueError('Feature columns differ between original export and labelled bridge')
    expected_tabular={}
    expected_tabular[str(labelled/'manifest.json')]=sha256_file(labelled/'manifest.json')
    for field,filename in (('samples','samples.jsonl'),('features','features.npz')):
        path=_manifest_file(labelled,label_manifest[field])
        if path!=labelled/filename or label_manifest[field]['sha256']!=label_state.get('output_sha256',{}).get(filename):
            raise ValueError('Labelled manifest and attachment receipt bindings differ')
        expected_tabular[str(path)]=label_manifest[field]['sha256']
        own_path=_manifest_file(own,own_manifest[field])
        if own_path!=own/filename or own_manifest[field]['sha256']!=own_state.get('output_sha256',{}).get(filename):
            raise ValueError('Own export manifest and receipt bindings differ')
    if expected_tabular[str(labelled/'samples.jsonl')]!=sha256_file(labelled/'samples.jsonl'):
        raise ValueError('Labelled samples changed')
    root=Path(own_provenance['cache_root']).resolve();split=own_provenance['cache_split']
    if not re.fullmatch(r'test|external_[a-z0-9_]+',split):
        raise ValueError('Own export must use an explicit heldout cache split')
    if protocol.get('cache_root')!=str(root) or protocol.get('cache_split')!=split:
        raise ValueError('Presentation and own export use different cache root/split')
    cache_status=root/'status.json';cache_protocol=root/'protocol.json';review_path=root/'visual_review.json'
    current_sha=sha256_file(cache_status)
    if (own_provenance.get('cache_status_sha256')!=current_sha or protocol.get('cache_status_sha256')!=current_sha):
        raise ValueError('Presentation and own export use different completed cache status')
    cache_state=load_json(cache_status);review=load_json(review_path);data_protocol=load_json(cache_protocol)
    if (cache_state.get('status')!='completed' or review.get('status')!='passed' or review.get('inspected') is not True
            or review.get('cache_status_sha256')!=current_sha or protocol.get('data_review_sha256')!=sha256_file(review_path)):
        raise ValueError('Cache receipt/actual visual review is no longer valid')
    for relative in ('protocol.json',split+'/images.npy',split+'/sources.jsonl'):
        expected=cache_state.get('files',{}).get(relative)
        if not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected):
            raise ValueError('Missing exact cache file digest')
        if (protocol.get('cache_bindings',{}).get(relative)!=expected or
                protocol.get('source_file_sha256',{}).get(str(root/relative))!=expected):
            raise ValueError('Presentation cache image/source/protocol binding differs')
        if relative!='protocol.json' and own_provenance.get('cache_bindings',{}).get(relative)!=expected:
            raise ValueError('Own export cache image/source binding differs')
    if sha256_file(cache_protocol)!=cache_state['files']['protocol.json']:
        raise ValueError('Cache data protocol changed')
    if feature_protocol.get('input_data_protocol')!=data_protocol.get('data_protocol'):
        raise ValueError('Own model input data protocol differs from presentation')
    for relative in (split+'/sources.jsonl',split+'/targets.npy',split+'/masks.npy'):
        expected=cache_state.get('files',{}).get(relative)
        if (not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected)
                or label_provenance.get('independent_label_bindings',{}).get(relative)!=expected):
            raise ValueError('Independent labels use a different cache binding')
    for path in (cache_status,cache_protocol,review_path,root/split/'sources.jsonl'):
        bindings[str(path)]=sha256_file(path)
    if bindings[str(root/split/'sources.jsonl')]!=cache_state['files'][split+'/sources.jsonl']:
        raise ValueError('Cache source manifest changed')
    canonical={}
    for row in read_jsonl(inputs_path):
        if (set(row)!=set(PRESENTED_IDENTITY) or row['input_id'] in canonical
                or not re.fullmatch('[0-9a-f]{64}',row.get('presented_image_sha256',''))):
            raise ValueError('Invalid or duplicate canonical presented input')
        canonical[row['input_id']]=row
    if (set(canonical)!={row['input_id'] for row in truth} or len(canonical)!=protocol.get('expected_inputs')
            or len(canonical)!=presentation_state.get('processed')):
        raise ValueError('Canonical presentation population differs from truth')
    for row in truth:
        if row.get('cache_split')!=split:
            raise ValueError('Labelled queries name a different cache split')
        if any(row.get(k)!=canonical[row['input_id']][k] for k in PRESENTED_IDENTITY[:5]):
            raise ValueError('Truth and canonical presentation source identity differ')
    return canonical,expected_tabular,bindings,dict(
        cache_root=str(root),cache_split=split,cache_status_sha256=current_sha,
        presentation_inputs_sha256=bindings[str(inputs_path)],canonical_inputs_path=str(inputs_path),
        own_export_status_sha256=sha256_file(own/'status.json'),
        feature_arrays_rehashed=False,cache_image_arrays_rehashed=False,
        verification='receipt SHA bindings from accepted export, label attachment and prediction stages; metadata and prediction files rehashed')


def common_comparison(rows,y,vectors,mask,methods,replicates):
    common=mask.copy()
    for name in methods:common &= np.isfinite(vectors[name])
    records=[{**r,'retention':float(t)} for r,t,keep in zip(rows,y,common) if keep]
    pool=sorted({(r['dataset'],r['split_group_id']) for r,keep in zip(rows,mask) if keep})
    scores={name:vectors[name][common] for name in methods}
    point={name:metrics(y[common],value,regression=name in REGRESSORS) for name,value in scores.items()}
    ranks={}
    for metric,higher in [('mae',False),('rmse',False),('bad_proxy_auroc',True),('bad_proxy_auprc',True),('aurc',False)]:
        values={name:p[metric] for name,p in point.items() if p[metric] is not None}
        ranks[metric]={name:1+sum(other>value if higher else other<value for other in values.values()) for name,value in values.items()}
    intervals=bootstrap(records,scores,REGRESSORS,SELECTED,pool,replicates=replicates,seed=17)
    return {'nominal_queries':int(mask.sum()),'common_valid_queries':int(common.sum()),
            'common_valid_source_images':len({r['source_image_sha256'] for r in records}),
            'common_query_keys_sha256':stable_hash([key(r) for r in records]),
            'conditional_on_common_success':True,'point_metrics':point,'point_estimate_ranks':ranks,'bootstrap':intervals}


def compare(labelled,specification,out,*,replicates=1000):
    if replicates!=1000:raise ValueError('Formal comparison requires fixed1000 bootstrap replicates')
    labelled=Path(labelled);specification=Path(specification);out=Path(out)
    if out.exists():raise FileExistsError('Do not overwrite any evaluation')
    out.mkdir(parents=True);state={'status':'validating','started_at':time.time(),'groups_completed':[]}
    write_json(out/'status.json',state)
    try:
        rows,y=load_truth(labelled);spec=load_json(specification)
        if spec.get('schema')!='oa-cxr-large-comparison-v1' or set(spec.get('methods',{}))!=REGRESSORS|VISUAL:
            raise ValueError('Full eight-method roster required, no internal V1/V2/V3')
        canonical,tabular_bindings,bindings,input_audit=verify_common_inputs(labelled,spec,rows)
        vectors={};records={}
        bindings[str(specification.resolve())]=sha256_file(specification)
        implementation={str(p.resolve()):sha256_file(p) for p in
                        (Path(__file__),ROOT/'src/oa_cxr/rebuild/metrics.py',ROOT/'src/oa_cxr/io.py')}
        bindings.update(implementation)
        for method,item in spec['methods'].items():
            path,receipt=verified_output(item['directory'],item['filename'])
            if method in REGRESSORS-{SELECTED}:
                if (receipt.get('schema')!='oa-cxr-rebuilt-baseline-predictions-v1' or
                        receipt.get('method')!=method or receipt.get('source_file_sha256')!=tabular_bindings):
                    raise ValueError('Tabular prediction receipt does not bind the exact labelled feature bridge')
            elif method in VISUAL:
                if (receipt.get('schema')!='oa-cxr-visual-baseline-run-v1' or receipt.get('method')!=method or
                        receipt.get('source_file_sha256',{}).get(input_audit['canonical_inputs_path'])!=
                        input_audit['presentation_inputs_sha256']):
                    raise ValueError('Visual prediction receipt does not bind the canonical input manifest')
            vectors[method],records[method]=align_scores(rows,read_jsonl(path),method,canonical)
            bindings[str(path)]=sha256_file(path);bindings[str(path.parent/'status.json')]=sha256_file(path.parent/'status.json')
        for name in ('status.json','samples.jsonl','labels.npz'):
            path=labelled/name;bindings[str(path.resolve())]=sha256_file(path)
        state.update(status='running',input_identity_audit=input_audit);write_json(out/'status.json',state)
        groups={}
        for dataset,split in sorted({(r['dataset'],r['split']) for r in rows}):
            mask=np.asarray([r['dataset']==dataset and r['split']==split for r in rows]);coverage={}
            for method,vector in vectors.items():
                available=mask & np.isfinite(vector);failed=mask & ~np.isfinite(vector)
                reasons={}
                for row,keep in zip(records[method],failed):
                    if keep:reasons[row['failure_reason']]=reasons.get(row['failure_reason'],0)+1
                coverage[method]={'nominal_queries':int(mask.sum()),'scored_queries':int(available.sum()),
                    'failed_queries':int(failed.sum()),'query_coverage':float(available.sum()/mask.sum()),
                    'automatic_review_missing_score_fraction':float(failed.sum()/mask.sum()),
                    'failure_reasons':reasons,'own_available_subset_metrics':metrics(y[available],vector[available],regression=method in REGRESSORS)}
            group_name=dataset+'/'+split
            groups[group_name]={'nominal_source_images':len({r['source_image_sha256'] for r,keep in zip(rows,mask) if keep}),
                'nominal_inputs':len({r['input_id'] for r,keep in zip(rows,mask) if keep}),
                'nominal_queries':int(mask.sum()),'coverage':coverage,
                'five_regressors':common_comparison(rows,y,vectors,mask,sorted(REGRESSORS),replicates),
                'eight_score_methods':common_comparison(rows,y,vectors,mask,sorted(vectors),replicates)}
            state['groups_completed'].append(group_name);write_json(out/'status.json',state)
        result={'schema':'oa-cxr-large-comparison-result-v1','groups':groups,'source_file_sha256':bindings,
            'input_identity_audit':input_audit,'implementation_sha256':implementation,
            'clinical_effectiveness_evaluated':False,'old_result_reuse':False,
            'interpretation':'MAE primary among five regressors; AUROC/AUPRC/AURC among eight scoring methods on common valid queries. Per-method available metrics are descriptive, not matched rankings.',
            'bootstrap_unit':'conservative split_group_id, not verified patient identity for COVIDQU/Kermany',
            'no_posthoc_overall_champion_score':True}
        for path,digest in bindings.items():
            if sha256_file(path)!=digest:raise ValueError('Bound artifact changed during comparison')
        write_json(out/'comparison.json',result)
        state.update(status='completed',output_sha256={'comparison.json':sha256_file(out/'comparison.json')})
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        state['finished_at']=time.time();write_json(out/'status.json',state)
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--labelled',type=Path,required=True)
    p.add_argument('--specification',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with exclusive_writer(a.output):compare(a.labelled,a.specification,a.output)

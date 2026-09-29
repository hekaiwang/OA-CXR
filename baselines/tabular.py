"""Train NEW same-feature CPU controls; never import historical experiment runs.

Run from the repository root with ``python -m baselines.tabular --help``.
Manifest and NPZ contracts are documented in baselines/MANIFEST_SCHEMA.md.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import re

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SCHEMA = 'oa-cxr-tabular-dataset-v1'
MODEL_SCHEMA = 'oa-cxr-rebuilt-tabular-model-v1'
PREDICTION_SCHEMA = 'oa-cxr-rebuilt-baseline-predictions-v1'
METHODS = ('hgb', 'xgboost', 'catboost', 'ridge')
FINDINGS = {'pleural_effusion', 'pneumothorax', 'consolidation'}
SPLITS = {'fit', 'dev', 'calibration', 'test', 'external'}
MIN_FIT_SOURCES = 10000


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def signed(value):
    return {**value, 'artifact_sha256': digest({k:v for k,v in value.items() if k != 'artifact_sha256'})}


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temp, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def fresh_output(path):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = Path(str(path)+'.lock')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        path.mkdir(exist_ok=False)
        yield path
    finally:
        lock.unlink(missing_ok=True)


def _bound_path(manifest_path, specification, override=None):
    require(isinstance(specification, dict) and isinstance(specification.get('path'), str)
            and re.fullmatch('[0-9a-f]{64}', specification.get('sha256', '')) is not None, 'Invalid bound file')
    path = Path(override if override is not None else specification['path'])
    if not path.is_absolute():
        path = manifest_path.parent/path
    path = path.resolve()
    require(sha(path) == specification['sha256'], 'Manifest file hash mismatch: '+str(path))
    return path


def load_data(manifest_path, features_npz=None, *, targets=False):
    """Prediction never indexes/loads NPZ y, including when y is present."""
    path = Path(manifest_path).resolve(); manifest = read(path)
    require(manifest.get('schema') == SCHEMA and manifest.get('target') == 'anatomical_retention'
            and manifest.get('label_free_features') is True, 'Unsupported target or feature contract')
    names = manifest.get('feature_names')
    require(isinstance(names, list) and names and all(isinstance(x,str) and x for x in names)
            and len(set(names)) == len(names), 'Unique ordered feature names required')
    require(re.fullmatch('[0-9a-f]{64}', manifest.get('feature_protocol_sha256','')) is not None,
            'Feature protocol hash required')
    sample_path = _bound_path(path, manifest['samples'])
    feature_path = _bound_path(path, manifest['features'], features_npz)
    rows = [json.loads(line) for line in sample_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    require(rows, 'Nonempty samples required')
    ids, grouping, source_splits, image_splits, query_keys = set(), {}, {}, {}, set()
    for row in rows:
        require(all(isinstance(row.get(k),str) and row[k] for k in
                    ('row_id','input_id','source_id','split_group_id','dataset','split','finding','source_image_sha256')),
                'Missing sample identity')
        require(row['split'] in SPLITS and row['finding'] in FINDINGS
                and re.fullmatch('[0-9a-f]{64}',row['source_image_sha256']) is not None, 'Invalid split/finding/source hash')
        require(row['row_id'] not in ids, 'Duplicate row_id'); ids.add(row['row_id'])
        query = (row['dataset'],row['input_id'],row['finding'])
        require(query not in query_keys, 'Duplicate input/finding query'); query_keys.add(query)
        for mapping,key in ((grouping,row['split_group_id']),
                            (source_splits,(row['dataset'],row['source_id'])),
                            (image_splits,row['source_image_sha256'])):
            require(key not in mapping or mapping[key] == row['split'], 'Source/group leakage across splits')
            mapping[key] = row['split']
    with np.load(feature_path, allow_pickle=False) as blob:
        require({'X','row_ids'} <= set(blob.files) and set(blob.files) <= {'X','row_ids','y','available'},
                'Unexpected NPZ arrays')
        X = np.asarray(blob['X']); row_ids = blob['row_ids']
        require(X.dtype.kind == 'f' and X.shape == (len(rows),len(names)), 'Invalid feature shape/dtype')
        require(row_ids.dtype.kind == 'U' and row_ids.tolist() == [r['row_id'] for r in rows], 'NPZ row alignment differs')
        available = np.asarray(blob['available']) if 'available' in blob else np.ones(len(rows),dtype=bool)
        require(available.dtype == np.dtype(bool) and available.shape == (len(rows),), 'Invalid availability array')
        require(np.isfinite(X[available]).all(), 'Nonfinite available features')
        y = np.asarray(blob['y']) if targets and 'y' in blob else None
        if targets:
            require(y is not None and y.dtype.kind == 'f' and y.shape == (len(rows),)
                    and np.isfinite(y).all() and ((y >= 0) & (y <= 1)).all(), 'Finite targets in [0,1] required')
    bindings = {str(path):sha(path), str(sample_path):sha(sample_path), str(feature_path):sha(feature_path)}
    return dict(manifest=manifest,rows=rows,X=X,y=y,available=available,bindings=bindings)


def config(method, path=None):
    require(method in METHODS, 'Unknown new tabular baseline')
    path = Path(path) if path else ROOT/method/'config.json'
    value = read(path)
    require(value.get('method') == method and value.get('schema') == 'oa-cxr-baseline-config-v1'
            and value.get('training') == 'new_fit_only' and value.get('minimum_fit_source_images') == MIN_FIT_SOURCES
            and value.get('hyperparameter_search') is False, 'Invalid fixed training specification')
    expected = read(ROOT/method/'config.json')
    require(value == expected, 'Baseline parameters must match committed fixed configuration')
    return value, {str(path.resolve()):sha(path)}


def estimator(method, parameters):
    if method == 'hgb':
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(**parameters)
    if method == 'xgboost':
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError('Install the xgboost baseline dependency; no substitute estimator is used') from exc
        return XGBRegressor(**parameters)
    if method == 'catboost':
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise RuntimeError('Install the catboost baseline dependency; no substitute estimator is used') from exc
        return CatBoostRegressor(**parameters)
    if method == 'ridge':
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(),Ridge(**parameters))
    raise ValueError('Unknown estimator')


def verify(bindings):
    require(all(sha(p) == value for p,value in bindings.items()), 'Bound input changed during execution')


def versions(method):
    return {package:item['version'] for package,item in version_evidence(method).items()}


def version_evidence(method):
    packages = ['numpy','scikit-learn','joblib'] + ([method] if method in ('xgboost','catboost') else [])
    result={}
    for package in packages:
        candidates=(package,'xgboost-cpu') if package=='xgboost' else (package,)
        for distribution in candidates:
            try:
                result[package]=dict(version=version(distribution),source='installed_distribution_metadata',
                                     distribution=distribution)
                break
            except PackageNotFoundError:
                pass
        if package in result:continue
        if package not in ('xgboost','catboost'):raise PackageNotFoundError(package)
        module=importlib.import_module(package);value=getattr(module,'__version__',None)
        require(isinstance(value,str) and re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+(?:[a-zA-Z0-9.+_-]*)',value) is not None,
                'Imported baseline has no verifiable module version')
        result[package]=dict(version=value,source='module.__version__; distribution metadata unavailable',
                             distribution=None,module_file=str(Path(module.__file__).resolve()))
    return result


def fit(method, manifest, features_npz, output, config_path=None):
    with fresh_output(output) as out:
        state = dict(schema=MODEL_SCHEMA,status='running',method=method,started_at=now(),production_training=True)
        write(out/'status.json',state)
        try:
            spec, bindings = config(method,config_path)
            data = load_data(manifest,features_npz,targets=True)
            require(all(r['split'] in {'fit','dev'} for r in data['rows']), 'Fit manifest must not contain calibration/test/external rows')
            fit_mask = np.array([r['split']=='fit' for r in data['rows']])
            usable = fit_mask & data['available']
            train_rows = [r for r,keep in zip(data['rows'],usable) if keep]
            source_shas = {r['source_image_sha256'] for r in train_rows}
            require(len(source_shas) >= MIN_FIT_SOURCES, 'At least 10000 usable distinct fit source images required')
            bindings.update(data['bindings']); bindings[str(Path(__file__).resolve())]=sha(__file__)
            with threadpool_limits(limits=4):
                model = estimator(method,spec['parameters']).fit(data['X'][usable],data['y'][usable])
            joblib.dump(model,out/'model.joblib',compress=3)
            verify(bindings)
            train_population = [r for r in data['rows'] if r['split']=='fit']
            model_manifest = signed(dict(schema=MODEL_SCHEMA,method=method,feature_names=data['manifest']['feature_names'],
                feature_protocol_sha256=data['manifest']['feature_protocol_sha256'],parameters=spec['parameters'],
                source_file_sha256=bindings,model_sha256=sha(out/'model.joblib'),environment=versions(method),
                environment_evidence=version_evidence(method),
                fit_source_images=len(source_shas),fit_rows=int(usable.sum()),fit_attempted_rows=int(fit_mask.sum()),
                fit_unavailable_rows=int((fit_mask & ~data['available']).sum()),dev_used_for_training=False,
                heldout_used_for_training=False,hyperparameter_search=False,clinical_support_labels=False,
                training_group_hashes=sorted({digest(r['split_group_id']) for r in train_population}),
                training_source_sha256=sorted({r['source_image_sha256'] for r in train_population}),
                prediction_clip=[0,1],legacy_checkpoint_imported=False))
            write(out/'model_manifest.json',model_manifest)
            state.update(status='completed',model_artifact_sha256=model_manifest['artifact_sha256'],
                         fit_source_images=len(source_shas),fit_rows=int(usable.sum()),
                         output_sha256={n:sha(out/n) for n in ('model.joblib','model_manifest.json')})
        except BaseException as exc:
            state.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
        finally:
            state['finished_at']=now();write(out/'status.json',state)
        return state


def load_model(checkpoint):
    path = Path(checkpoint).resolve(); state=read(path/'status.json'); meta=read(path/'model_manifest.json')
    require(state.get('status')=='completed' and meta.get('schema')==MODEL_SCHEMA
            and meta==signed(meta) and meta.get('method') in METHODS
            and meta.get('legacy_checkpoint_imported') is False and meta.get('fit_source_images',0)>=MIN_FIT_SOURCES,
            'Only completed NEW >=10000-source baseline checkpoints are accepted')
    require(state.get('model_artifact_sha256')==meta['artifact_sha256']
            and state.get('output_sha256',{}).get('model_manifest.json')==sha(path/'model_manifest.json')
            and state.get('output_sha256',{}).get('model.joblib')==meta['model_sha256']==sha(path/'model.joblib'),
            'Checkpoint bytes changed')
    require(meta['environment']==versions(meta['method']), 'Checkpoint package versions differ; reproduce its environment')
    return joblib.load(path/'model.joblib'),meta


def predict(checkpoint, manifest, features_npz, output):
    with fresh_output(output) as out:
        state=dict(schema=PREDICTION_SCHEMA,status='running',started_at=now(),target_values_loaded=False)
        write(out/'status.json',state)
        try:
            model,meta=load_model(checkpoint);data=load_data(manifest,features_npz,targets=False)
            require(meta['feature_names']==data['manifest']['feature_names']
                    and meta['feature_protocol_sha256']==data['manifest']['feature_protocol_sha256'], 'Feature column/protocol mismatch')
            seen_groups=set(meta['training_group_hashes']);seen_sources=set(meta['training_source_sha256'])
            require(all(r['split']=='fit' or (digest(r['split_group_id']) not in seen_groups
                        and r['source_image_sha256'] not in seen_sources) for r in data['rows']), 'Evaluation overlaps training source/group')
            available=data['available'];pred=np.empty(0)
            if available.any():
                with threadpool_limits(limits=4):pred=np.asarray(model.predict(data['X'][available]),dtype=float)
                require(pred.shape==(int(available.sum()),) and np.isfinite(pred).all(), 'Invalid prediction before clipping')
                pred=np.clip(pred,0,1)
            records=[];cursor=0
            for row,valid in zip(data['rows'],available):
                score=float(pred[cursor]) if valid else None;cursor+=int(valid)
                records.append({**row,'method':meta['method'],'score':score,'status':'success' if valid else 'failed',
                                'failure_reason':None if valid else 'upstream_features_unavailable'})
            with (out/'predictions.jsonl').open('x',encoding='utf-8') as stream:
                for record in records:stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            verify(data['bindings'])
            state.update(status='completed',method=meta['method'],model_artifact_sha256=meta['artifact_sha256'],
                source_file_sha256=data['bindings'],attempted_rows=len(records),scored_rows=int(available.sum()),
                unavailable_rows=int((~available).sum()),output_sha256={'predictions.jsonl':sha(out/'predictions.jsonl')})
        except BaseException as exc:
            state.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
        finally:
            state['finished_at']=now();write(out/'status.json',state)
        return state


def metrics(y, scores):
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score,roc_auc_score
    y=np.asarray(y,dtype=float);s=np.asarray(scores,dtype=float)
    if not len(y):return {k:None for k in ('mae','rmse','spearman','bad_auroc','bad_auprc','aurc','excess_aurc')}
    bad=y<.9;order=np.argsort(-s,kind='stable');sorted_s=s[order];b=bad[order].astype(float)
    starts=np.r_[0,np.flatnonzero(sorted_s[1:]!=sorted_s[:-1])+1];ends=np.r_[starts[1:],len(s)]
    lengths=ends-starts;counts=np.add.reduceat(b,starts);prior=np.cumsum(counts)-counts
    within=np.arange(len(s))-np.repeat(starts,lengths)+1
    expected=np.repeat(prior,lengths)+within*np.repeat(counts/lengths,lengths)
    aurc=float(np.mean(expected/np.arange(1,len(s)+1)))
    oracle=float(np.mean(np.cumsum(np.sort(b))/np.arange(1,len(s)+1)))
    two_classes=len(np.unique(bad))==2
    return dict(mae=float(np.abs(y-s).mean()),rmse=float(np.sqrt(np.square(y-s).mean())),
        spearman=float(spearmanr(y,s).statistic) if len(np.unique(y))>1 and len(np.unique(s))>1 else None,
        bad_auroc=float(roc_auc_score(bad,-s)) if two_classes else None,
        bad_auprc=float(average_precision_score(bad,-s)) if two_classes else None,
        aurc=aurc,excess_aurc=aurc-oracle)


def evaluate(manifest, features_npz, predictions, output):
    """Single-method point metrics; paired multicandidate inference belongs to the shared project evaluator."""
    with fresh_output(output) as out:
        state=dict(status='running',started_at=now(),schema='oa-cxr-baseline-evaluation-v1')
        write(out/'status.json',state)
        try:
            data=load_data(manifest,features_npz,targets=True);p=Path(predictions).resolve();receipt=read(p/'status.json')
            require(receipt.get('schema')==PREDICTION_SCHEMA and receipt.get('status')=='completed'
                    and receipt.get('method') in METHODS and receipt['source_file_sha256']==data['bindings']
                    and receipt['output_sha256']['predictions.jsonl']==sha(p/'predictions.jsonl'), 'Predictions are not bound to evaluation inputs')
            rows=[json.loads(line) for line in (p/'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
            require(len(rows)==len(data['rows']), 'Complete prediction denominator required')
            for prediction,row,available in zip(rows,data['rows'],data['available']):
                require(all(prediction.get(k)==v for k,v in row.items()) and prediction.get('method')==receipt['method'], 'Prediction identity/order mismatch')
                score=prediction.get('score')
                require((available and prediction.get('status')=='success' and type(score) in (float,int)
                         and math.isfinite(score) and 0<=score<=1)
                        or (not available and prediction.get('status')=='failed' and score is None), 'Failure score contract mismatch')
            groups={}
            for dataset,split in sorted({(r['dataset'],r['split']) for r in rows}):
                mask=np.array([r['dataset']==dataset and r['split']==split for r in rows]);common=mask&data['available']
                groups[dataset+'/'+split]=dict(attempted_rows=int(mask.sum()),available_rows=int(common.sum()),
                    unavailable_rows=int((mask&~data['available']).sum()),coverage=float(common.sum()/mask.sum()),
                    source_images=len({r['source_image_sha256'] for r,keep in zip(rows,mask) if keep}),
                    metrics=metrics(data['y'][common],[r['score'] for r,keep in zip(rows,common) if keep]))
            result=dict(method=receipt['method'],groups=groups,comparison_scope='per-method available subset; not a common-method leaderboard',
                uncertainty_computed=False,paired_bootstrap_required_for_comparisons=True,clinical_effectiveness_evaluated=False,
                source_file_sha256={**data['bindings'],str(p/'status.json'):sha(p/'status.json'),str(p/'predictions.jsonl'):sha(p/'predictions.jsonl')})
            verify(result['source_file_sha256']);write(out/'metrics.json',result)
            state.update(status='completed',output_sha256={'metrics.json':sha(out/'metrics.json')})
        except BaseException as exc:
            state.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
        finally:
            state['finished_at']=now();write(out/'status.json',state)
        return state


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    for action in ('fit','predict','evaluate'):
        p=sub.add_parser(action);p.add_argument('--manifest',type=Path,required=True)
        p.add_argument('--features-npz',type=Path);p.add_argument('--output',type=Path,required=True)
        if action=='fit':p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--config',type=Path)
        elif action=='predict':p.add_argument('--checkpoint',type=Path,required=True)
        else:p.add_argument('--predictions',type=Path,required=True)
    a=parser.parse_args()
    if a.action=='fit':result=fit(a.method,a.manifest,a.features_npz,a.output,a.config)
    elif a.action=='predict':result=predict(a.checkpoint,a.manifest,a.features_npz,a.output)
    else:result=evaluate(a.manifest,a.features_npz,a.predictions,a.output)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()

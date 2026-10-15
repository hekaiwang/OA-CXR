"""New-data baseline contracts; synthetic fixtures are not scientific results."""
from pathlib import Path
import importlib.util
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from baselines import tabular as b


def test_optional_overlay_version_records_actual_module_without_inventing_metadata(monkeypatch):
    original=b.version
    def version(name):
        if name in ('xgboost','xgboost-cpu'):raise b.PackageNotFoundError(name)
        return original(name)
    monkeypatch.setattr(b,'version',version)
    monkeypatch.setattr(b.importlib,'import_module',lambda name:SimpleNamespace(__version__='3.0.5',__file__=__file__))
    assert b.versions('xgboost')['xgboost']=='3.0.5'
    assert 'metadata unavailable' in b.version_evidence('xgboost')['xgboost']['source']


@pytest.mark.parametrize('installed,expected',[
    ({'xgboost':'3.0.5','xgboost-cpu':'3.0.4'},'xgboost'),
    ({'xgboost-cpu':'3.0.5'},'xgboost-cpu'),
])
def test_xgboost_cpu_distribution_metadata_precedes_module_fallback(monkeypatch,installed,expected):
    original=b.version
    def version(name):
        if name in ('xgboost','xgboost-cpu'):
            if name not in installed:raise b.PackageNotFoundError(name)
            return installed[name]
        return original(name)
    monkeypatch.setattr(b,'version',version)
    monkeypatch.setattr(b.importlib,'import_module',lambda name:pytest.fail('Metadata exists; module fallback must not run'))
    evidence=b.version_evidence('xgboost')['xgboost']
    assert evidence=={'version':'3.0.5','source':'installed_distribution_metadata','distribution':expected}
    assert b.versions('xgboost')['xgboost']=='3.0.5'


def dataset(folder,n=10,*,split='fit',offset=0,unavailable=False,with_targets=True):
    folder.mkdir(parents=True)
    rows=[dict(row_id=f'r{i+offset}',input_id=f'i{i+offset}',source_id=f's{i+offset}',
        split_group_id=f'g{i+offset}',dataset='synthetic',split=split,finding='consolidation',
        source_image_sha256=b.digest(['synthetic-source',i+offset])) for i in range(n)]
    X=np.linspace(0,1,n*2,dtype=np.float32).reshape(n,2)
    available=np.ones(n,dtype=bool)
    if unavailable:available[-1]=False;X[-1]=np.nan
    y=np.clip(.2+.6*np.nan_to_num(X[:,0]),0,1).astype(np.float32)
    arrays=dict(X=X,row_ids=np.asarray([r['row_id'] for r in rows]),available=available)
    if with_targets:arrays['y']=y
    np.savez(folder/'features.npz',**arrays)
    (folder/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    manifest=dict(schema=b.SCHEMA,target='anatomical_retention',label_free_features=True,
        feature_protocol_sha256=b.digest('synthetic-feature-protocol'),feature_names=['one','two'],
        samples=dict(path='samples.jsonl',sha256=b.sha(folder/'samples.jsonl')),
        features=dict(path='features.npz',sha256=b.sha(folder/'features.npz')))
    b.write(folder/'manifest.json',manifest)
    return folder/'manifest.json'


def replace_rows(manifest,mutate):
    m=b.read(manifest);p=manifest.parent/m['samples']['path']
    rows=[json.loads(x) for x in p.read_text().splitlines()];mutate(rows)
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    m['samples']['sha256']=b.sha(p);b.write(manifest,m)


def test_registry_has_all_fifteen_external_methods_and_real_entrypoints():
    r=b.read(b.ROOT/'registry.json')
    expected={'hgb','xgboost','catboost','ridge','chexpert','autoqc','biovil_t','maira_logprob','sentence_logprob','token_entropy','scuq_radgraph',
              'tabm','biomedclip','chexzero','dinov2'}
    assert {m['id'] for m in r['methods']}==expected and len(r['methods'])==15
    for m in r['methods']:
        assert m['internal_version'] is False
        assert m['full_original_paper_reproduction'] is False
        assert (b.ROOT.parent/m['entrypoint']).is_file()


@pytest.mark.parametrize('method',b.METHODS)
def test_fixed_cpu_configs(method):
    c,_=b.config(method)
    assert c['training']=='new_fit_only' and c['minimum_fit_source_images']==10000
    if method=='hgb':assert c['parameters']['early_stopping'] is False
    if method=='xgboost':assert c['parameters']['device']=='cpu'
    if method=='catboost':assert c['parameters']['task_type']=='CPU'


@pytest.mark.parametrize('fault',['group','image','source','duplicate'])
def test_cross_split_and_duplicate_identity_rejected(tmp_path,fault):
    m=dataset(tmp_path/'data')
    def mutate(rows):
        rows[1]['split']='test'
        if fault=='group':rows[1]['split_group_id']=rows[0]['split_group_id']
        elif fault=='image':rows[1]['source_image_sha256']=rows[0]['source_image_sha256']
        elif fault=='source':rows[1]['source_id']=rows[0]['source_id']
        else:rows[1]['row_id']=rows[0]['row_id']
    replace_rows(m,mutate)
    with pytest.raises(ValueError):b.load_data(m)


def test_unaligned_npz_and_unbound_bytes_rejected(tmp_path):
    m=dataset(tmp_path/'data');p=m.parent/'features.npz'
    with np.load(p) as z:arrays={k:z[k] for k in z.files}
    arrays['row_ids']=arrays['row_ids'][::-1];np.savez(p,**arrays)
    with pytest.raises(ValueError,match='hash mismatch'):b.load_data(m)
    meta=b.read(m);meta['features']['sha256']=b.sha(p);b.write(m,meta)
    with pytest.raises(ValueError,match='alignment'):b.load_data(m)


def test_prediction_load_does_not_require_or_validate_target_values(tmp_path):
    m=dataset(tmp_path/'data',with_targets=False)
    assert b.load_data(m,targets=False)['y'] is None
    with pytest.raises(ValueError,match='targets'):b.load_data(m,targets=True)
    p=m.parent/'features.npz'
    with np.load(p) as z:arrays={k:z[k] for k in z.files}
    arrays['y']=np.full(10,np.nan);np.savez(p,**arrays)
    meta=b.read(m);meta['features']['sha256']=b.sha(p);b.write(m,meta)
    assert b.load_data(m,targets=False)['y'] is None


def test_small_or_heldout_training_fails_and_preserves_receipt(tmp_path):
    m=dataset(tmp_path/'data')
    with pytest.raises(ValueError,match='10000'):b.fit('ridge',m,None,tmp_path/'small')
    assert b.read(tmp_path/'small/status.json')['status']=='failed'
    with pytest.raises(FileExistsError):b.fit('ridge',m,None,tmp_path/'small')
    test=dataset(tmp_path/'test',split='test')
    with pytest.raises(ValueError,match='must not contain'):b.fit('ridge',test,None,tmp_path/'heldout')


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    root=tmp_path_factory.mktemp('new_baseline')
    train=dataset(root/'train',n=10000)
    state=b.fit('ridge',train,None,root/'checkpoint')
    assert state['status']=='completed' and state['fit_source_images']==10000
    return root


def test_real_new_fit_predict_evaluate_with_full_failure_denominator(trained):
    root=trained;m=dataset(root/'evaluation',n=11,split='test',offset=20000,unavailable=True)
    state=b.predict(root/'checkpoint',m,None,root/'predictions')
    assert state['scored_rows']==10 and state['unavailable_rows']==1 and state['target_values_loaded'] is False
    assert b.evaluate(m,None,root/'predictions',root/'metrics')['status']=='completed'
    result=b.read(root/'metrics/metrics.json');group=result['groups']['synthetic/test']
    assert group['attempted_rows']==11 and group['available_rows']==10 and group['unavailable_rows']==1
    assert result['uncertainty_computed'] is False
    rows=[json.loads(x) for x in (root/'predictions/predictions.jsonl').read_text().splitlines()]
    assert rows[-1]['score'] is None and rows[-1]['status']=='failed'


def test_checkpoint_train_source_cannot_be_relabelled_test(trained):
    m=dataset(trained/'leak',split='test')
    with pytest.raises(ValueError,match='overlaps'):b.predict(trained/'checkpoint',m,None,trained/'leaked_predictions')


def test_wrong_feature_protocol_fails_before_prediction(trained):
    m=dataset(trained/'changed_features',split='test',offset=30000)
    value=b.read(m);value['feature_protocol_sha256']=b.digest('changed');b.write(m,value)
    with pytest.raises(ValueError,match='Feature column/protocol'):b.predict(trained/'checkpoint',m,None,trained/'wrong_protocol')


def test_nonregression_or_old_checkpoint_not_accepted(tmp_path):
    out=tmp_path/'old';out.mkdir()
    b.write(out/'status.json',{'status':'completed'})
    b.write(out/'model_manifest.json',{'schema':'v3','method':'ridge'})
    with pytest.raises(ValueError,match='NEW'):b.load_model(out)


def test_hash_tampering_stops_checkpoint_load(trained,tmp_path):
    import shutil
    copied=tmp_path/'copy';shutil.copytree(trained/'checkpoint',copied)
    with (copied/'model.joblib').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='bytes changed'):b.load_model(copied)


def test_hgb_real_estimator_fit_uses_fixed_cpu_parameters():
    c,_=b.config('hgb');model=b.estimator('hgb',c['parameters'])
    X=np.arange(100,dtype=float).reshape(50,2)/100;y=X[:,0]
    model.fit(X,y);prediction=model.predict(X)
    assert np.isfinite(prediction).all() and prediction.shape==(50,)


@pytest.mark.parametrize('method',['xgboost','catboost'])
def test_optional_packages_never_silently_substitute(method):
    c,_=b.config(method)
    if importlib.util.find_spec(method) is None:
        with pytest.raises(RuntimeError,match='dependency'):b.estimator(method,c['parameters'])
    else:
        model=b.estimator(method,c['parameters']);assert method in type(model).__module__


def test_tie_aware_aurc_and_single_class_nulls():
    result=b.metrics([0,1],[.5,.5])
    assert result['aurc']==pytest.approx(.5)
    assert b.metrics([1,1],[.5,.7])['bad_auroc'] is None
    assert b.metrics([],[])['mae'] is None

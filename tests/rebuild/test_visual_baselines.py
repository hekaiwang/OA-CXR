"""Synthetic technical contracts; these fixtures are not experiment results."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from baselines import visual as v
from baselines.autoqc.run import prepare_image as niqe_image,vector_scores
from baselines.chexpert.run import prepare_image as chex_image,tensor_state


def fixture(folder,n=3):
    folder.mkdir(exist_ok=True);rows=[]
    for i in range(n):
        p=folder/f'{i}.png';Image.fromarray(np.tile(np.arange(224,dtype=np.uint8),(224,1))).save(p)
        rows.append(dict(input_id=f'i{i}',source_id=f's{i}',dataset='synthetic',split='test',source_image_sha256=v.digest(i),
                         presented_image_path=str(p.resolve()),presented_image_sha256=v.sha(p)))
    manifest=folder/'input.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return manifest,rows


def mock_backend(monkeypatch,engine):
    monkeypatch.setattr(v,'verified_assets',lambda *a:({},{},Path('.'),{}))
    monkeypatch.setattr(v,'backend',lambda *a:engine)


class Engine:
    def prepare(self,row):return row
    def infer(self,rows):return [{f:{'score':.5} for f in v.FINDINGS} for _ in rows]


def records(out):return [json.loads(r) for r in (out/'scores.jsonl').read_text().splitlines()]


def test_canonical_rejects_labels_duplicate_sources_and_extra_fields(tmp_path):
    manifest,rows=fixture(tmp_path)
    rows[0]['label']=0;manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='label-free'):v.inputs(manifest,3)
    del rows[0]['label'];rows[1].update(source_image_sha256=rows[0]['source_image_sha256'],split='calibration')
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='crosses splits'):v.inputs(manifest,3)


def test_complete_scores_and_hash_failure_null(tmp_path,monkeypatch):
    manifest,rows=fixture(tmp_path/'input');Path(rows[-1]['presented_image_path']).write_bytes(b'bad')
    mock_backend(monkeypatch,Engine());out=tmp_path/'output'
    status=v.score('chexpert',manifest,out,expected_inputs=3,device='cpu')
    assert status['status']=='completed_with_failures' and status['scored_inputs']==2 and status['failed_inputs']==1
    rr=records(out);assert len(rr)==9 and all(r['score'] is None for r in rr[-3:])
    assert status['attempted_inputs']==3 and status['unattempted_inputs']==0


def test_backend_setup_failure_keeps_full_population(tmp_path,monkeypatch):
    manifest,_=fixture(tmp_path/'input')
    monkeypatch.setattr(v,'verified_assets',lambda *a:(_ for _ in ()).throw(ValueError('asset changed')))
    out=tmp_path/'out';status=v.score('chexpert',manifest,out,expected_inputs=3,device='cpu')
    assert status['status']=='failed' and status['unattempted_inputs']==3
    assert len(records(out))==9 and all(r['score'] is None and not r['attempted'] for r in records(out))


def test_batch_failure_has_no_retry_preserves_remaining(tmp_path,monkeypatch):
    manifest,_=fixture(tmp_path/'input',19)
    class Broken(Engine):
        def infer(self,rows):raise RuntimeError('simulated OOM')
    mock_backend(monkeypatch,Broken());out=tmp_path/'out'
    status=v.score('chexpert',manifest,out,expected_inputs=19,device='cpu')
    assert status['status']=='failed' and status['failed_inputs']==16 and status['unattempted_inputs']==3
    assert len(records(out))==57 and status['output_inputs']==19


def test_chexpert_batch32_records_actual_batch_and_preserves_tail(tmp_path,monkeypatch):
    manifest,_=fixture(tmp_path/'input',35)
    class Measured(Engine):
        sizes=[]
        def infer(self,rows):
            self.sizes.append(len(rows));return super().infer(rows)
    engine=Measured();mock_backend(monkeypatch,engine);out=tmp_path/'out'
    status=v.score('chexpert',manifest,out,expected_inputs=35,device='cpu',batch_size=32)
    assert status['status']=='completed' and status['batch_size']==32
    assert engine.sizes==[32,3] and len(records(out))==105
    assert v.read(out/'protocol.json')['batch_size']==32


def test_chexpert_batch32_failure_keeps_every_original_input(tmp_path,monkeypatch):
    manifest,_=fixture(tmp_path/'input',35)
    class Broken(Engine):
        def infer(self,rows):raise RuntimeError('simulated batch32 failure')
    mock_backend(monkeypatch,Broken());out=tmp_path/'out'
    state=v.score('chexpert',manifest,out,expected_inputs=35,device='cpu',batch_size=32)
    assert state['failed_inputs']==32 and state['unattempted_inputs']==3 and state['output_inputs']==35
    assert len(records(out))==105 and all(r['score'] is None for r in records(out))


@pytest.mark.parametrize('method,batch',[('biovil_t',32),('autoqc',32),('chexpert',64),('chexpert',128),('chexpert',True)])
def test_batch_amendment_does_not_relax_other_methods(tmp_path,method,batch):
    with pytest.raises(ValueError,match='batch16'):
        v.score(method,tmp_path/'missing',tmp_path/'out',expected_inputs=1,device='cpu',batch_size=batch)
    assert not (tmp_path/'out').exists()


def test_preview_failure_preserves_remaining_and_close_error(tmp_path,monkeypatch):
    manifest,_=fixture(tmp_path/'input',19)
    class Broken(Engine):
        def preview(self,*a):raise ValueError('preview write failed')
        def close(self):raise RuntimeError('close failed')
    mock_backend(monkeypatch,Broken());out=tmp_path/'out'
    status=v.score('chexpert',manifest,out,expected_inputs=19,device='cpu')
    assert status['scored_inputs']==16 and status['unattempted_inputs']==3 and status['status']=='failed'
    assert len(records(out))==57 and 'close_error' in status


def test_finding_contract_nonfinite_and_identity_overwrite_rejected():
    item={f:{'score':.5} for f in v.FINDINGS};item[v.FINDINGS[0]]['extra']=float('nan')
    with pytest.raises(ValueError):v.successful({},'chexpert',item)
    item[v.FINDINGS[0]]={'score':.5,'source_id':'injected'}
    with pytest.raises(ValueError,match='identity'):v.successful({},'chexpert',item)


def test_output_is_never_overwritten_or_competed(tmp_path,monkeypatch):
    manifest,_=fixture(tmp_path/'input');mock_backend(monkeypatch,Engine());out=tmp_path/'out'
    v.score('chexpert',manifest,out,expected_inputs=3,device='cpu');original=v.sha(out/'scores.jsonl')
    with pytest.raises(FileExistsError):v.score('chexpert',manifest,out,expected_inputs=3,device='cpu')
    assert v.sha(out/'scores.jsonl')==original
    lock=Path(str(tmp_path/'locked')+'.lock');lock.write_text('123')
    with pytest.raises(FileExistsError):v.score('chexpert',manifest,tmp_path/'locked',expected_inputs=3,device='cpu')
    assert lock.read_text()=='123'


def test_chexpert_preprocessing_and_tensor_only_state(tmp_path):
    p=tmp_path/'x.png';Image.fromarray(np.array([[0,255],[0,255]],dtype='uint8')).save(p)
    x=chex_image(p);assert x.shape==(1,224,224) and x.dtype==np.float32
    assert x[0,0,0]==-1024 and x[0,0,-1]==1024
    torch=SimpleNamespace(is_tensor=lambda x:isinstance(x,np.ndarray));array=np.array([1])
    assert set(tensor_state({'module.weight':array},torch))=={'weight'}
    with pytest.raises(ValueError):tensor_state({'weight':array,'module.bias':array},torch)
    with pytest.raises(ValueError):tensor_state({'model':object()},torch)


def test_autoqc_preprocessing_and_distance_contract(tmp_path):
    manifest,rows=fixture(tmp_path)
    assert niqe_image(rows[0]['presented_image_path']).shape==(224,224)
    assert vector_scores(np.arange(37))['score']==-36
    for bad in (np.ones(36),np.r_[np.ones(36),-1],np.r_[np.ones(36),np.nan]):
        with pytest.raises(ValueError):vector_scores(bad)
    p=tmp_path/'flat.png';Image.new('L',(224,224),0).save(p)
    with pytest.raises(ValueError,match='Constant'):niqe_image(p)
    Image.new('L',(224,192),0).save(p)
    with pytest.raises(ValueError,match='square'):niqe_image(p)


def test_asset_roster_bytes_and_hash_binding(tmp_path):
    folder=tmp_path/'assets';folder.mkdir();asset=folder/'weight';asset.write_bytes(b'official')
    meta=dict(schema='oa-cxr-baseline-assets-v1',method='chexpert',legacy_results_included=False,asset_root='assets',
              files={'weight':dict(bytes=8,sha256=v.sha(asset))})
    v.write(tmp_path/'assets_manifest.json',meta)
    config=dict(schema='oa-cxr-baseline-config-v1',method='chexpert',full_original_paper_reproduction=False,
                asset_root='assets',assets_manifest=dict(path='assets_manifest.json',sha256=v.sha(tmp_path/'assets_manifest.json')))
    v.write(tmp_path/'config.json',config)
    assert v.verified_assets('chexpert',tmp_path/'config.json',tmp_path)[2]==folder
    asset.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='byte mismatch'):v.verified_assets('chexpert',tmp_path/'config.json',tmp_path)


def test_gpu_requires_uuid_and_refuses_occupied_device(monkeypatch):
    with pytest.raises(ValueError,match='UUID'):v.select_gpu('2')
    uuid='GPU-3a93e1ab-3211-7961-e56d-4b7c19a7cb39'
    monkeypatch.setattr(v.subprocess,'check_output',lambda argv,**kw:f'{uuid}, A40, 500, 0' if '-i' in argv else '')
    with pytest.raises(RuntimeError,match='occupied'):v.select_gpu(uuid)


def test_adapter_code_does_not_import_historical_scripts_or_runs():
    for method in v.METHODS:
        text=(v.ROOT/method/'run.py').read_text()
        assert 'oa_cxr.' not in text and 'runs/' not in text and 'from scripts' not in text

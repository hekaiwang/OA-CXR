"""Technical report identity and denominator tests; no actual medical results."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from baselines import reporting as r
from baselines.scuq_radgraph.run import sentence_index,sample_group
from baselines.visual import digest,sha,read,write

TEXT='No pleural effusion or pneumothorax. No consolidation.'


def row(i=0):
    return dict(input_id=f'technical{i}',source_id=f'source{i}',dataset='synthetic_technical',split='calibration',
                source_image_sha256=digest(['source',i]),presented_image_path='/technical/input.png',presented_image_sha256=digest(['pixels',i]))


def token_record(i=0,status='success'):
    lp=[-.2]*len(TEXT);ent=[.9]*len(TEXT);ids=list(range(len(TEXT)));offsets=[(i,i+1) for i in range(len(TEXT))]
    return dict(input=row(i),status=status,generation_status='success',generated_findings=TEXT,attempted=True,
        possibly_truncated=False,mean_token_log_probability=-.2,
        claims=r.claim_token_scores(TEXT,ids,offsets,lp,ent) if status=='success' else [])


def fixture_tokens(path,rows):
    path.mkdir();r.write_jsonl(path/'token_reports.jsonl',rows);r.write_jsonl(path/'denominators.jsonl',r.denominator(rows))
    write(path/'protocol.json',{'schema':r.SCHEMA,'stage':'tokens'})
    write(path/'status.json',dict(schema=r.SCHEMA,stage='tokens',status='completed_with_failures',input_count=len(rows),
        output_sha256={p:sha(path/p) for p in ('token_reports.jsonl','denominators.jsonl','protocol.json')}))
    return path


def test_generic_bridge_has_only_actual_frontal_and_no_metadata():
    b=r.generic_row(row());assert b['local_image_paths']=={'frontal':'/technical/input.png'} and b['allowed_input_metadata']=={}
    assert b['partition']=='calibration' and 'labels' not in b
    bad=row();bad['label']=0
    with pytest.raises(ValueError):r.generic_row(bad)


def test_token_scores_sentence_scope_and_invalid_offsets():
    value=token_record();assert len(value['claims'])==3
    assert all(x['sentence_mean_log_probability']==pytest.approx(-.2) and x['sentence_negative_mean_token_entropy']==pytest.approx(-.9) for x in value['claims'])
    with pytest.raises(ValueError,match='fully covered'):r.token_positions('No effusion.',[(0,2)],0,12)
    with pytest.raises(ValueError,match='distribution'):r.claim_token_scores(TEXT,[0],[(0,len(TEXT))],[.2],[1])


def test_token_failure_keeps_generation_claim_denominator():
    value=token_record(status='failed');d=r.denominator([value])[0]
    assert d['eligible_claims']==3 and d['generation_status']=='success' and d['status']=='failed'


@pytest.mark.parametrize('method',r.METHODS[:3])
def test_independent_three_score_adapters_preserve_failed_claims(tmp_path,method):
    tokenrun=fixture_tokens(tmp_path/'tokens',[token_record(),token_record(1,'failed')]);out=tmp_path/method
    r.score_tokens_method(method,tokenrun,out)
    rows=[json.loads(s) for s in (out/'scores.jsonl').read_text().splitlines()]
    assert len(rows)==6 and all(x['method']==method for x in rows)
    assert sum(x['score'] is None for x in rows)==3
    assert read(out/'status.json')['eligible_claims']==6 and read(out/'status.json')['full_input_count']==2
    assert read(out/'status.json')['status']=='completed_with_failures'


def test_changed_or_historical_report_bundle_rejected(tmp_path):
    path=fixture_tokens(tmp_path/'tokens',[token_record()]);(path/'token_reports.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='changed'):r.checked_run(path,'token_reports.jsonl','tokens')
    state=read(path/'status.json');state['schema']='historical-cache';write(path/'status.json',state)
    with pytest.raises(ValueError,match='new report stage'):r.checked_run(path,'token_reports.jsonl','tokens')


def test_scientific_no_eligible_reports_do_not_invent_findings(tmp_path):
    record=token_record();record.update(generated_findings='Lungs are clear.',claims=[])
    folder=fixture_tokens(tmp_path/'tokens',[record]);out=tmp_path/'method'
    r.score_tokens_method('sentence_logprob',folder,out)
    assert (out/'scores.jsonl').read_text()=='' and read(out/'status.json')['eligible_claims']==0
    assert read(out/'status.json')['full_input_count']==1


def test_scuq_claim_mapping_and_exact_ten_nontruncated_samples():
    claims=r.extract_claims(TEXT);sentences=TEXT.split('. ')
    assert sentence_index(TEXT,sentences,claims[0])==0 and sentence_index(TEXT,sentences,claims[-1])==1
    samples=dict(samples=[dict(sample_index=i,status='success',possibly_truncated=False,generated_findings=TEXT) for i in range(10)])
    assert len(sample_group(samples))==10
    samples['samples'][-1]['possibly_truncated']=True
    with pytest.raises(ValueError,match='truncated'):sample_group(samples)
    samples['samples']=samples['samples'][:9]
    with pytest.raises(ValueError,match='ten'):sample_group(samples)


def test_generation_failure_saves_complete_unattempted_population(tmp_path,monkeypatch):
    rows=[row(i) for i in range(18)];manifest=tmp_path/'inputs.jsonl'
    for item in rows:
        p=tmp_path/(item['input_id']+'.png');p.write_bytes(b'technical fixture');item.update(presented_image_path=str(p),presented_image_sha256=sha(p))
    r.write_jsonl(manifest,rows)
    config=dict(backend='maira',model_source='/not-loaded',model_revision='a'*40,processor_revision='a'*40,
        generation=dict(do_sample=False,max_new_tokens=512,use_cache=True,num_beams=1),seed=17,dtype='bfloat16',device='cuda:0',local_files_only=True)
    monkeypatch.setattr(r,'maira_config',lambda *a:(config,{}));monkeypatch.setattr(r,'select_gpu',lambda x:{})
    monkeypatch.setattr(r,'runtime_device',lambda *a:{})
    class Broken:
        def __init__(self,config):self.torch=object()
        def generate_batch(self,*a):raise RuntimeError('synthetic OOM')
    monkeypatch.setattr(r,'BatchedMairaBackend',Broken);out=tmp_path/'out'
    with pytest.raises(RuntimeError,match='OOM'):r.generate(manifest,out,18,'technical-id')
    actual=[json.loads(x) for x in (out/'reports.jsonl').read_text().splitlines()]
    assert len(actual)==18 and sum(x['status']=='failed' for x in actual)==16 and sum(x['status']=='unattempted' for x in actual)==2
    assert read(out/'status.json')['status']=='failed'


def test_existing_output_and_lock_are_never_overwritten(tmp_path):
    folder=tmp_path/'out';folder.mkdir();(folder/'keep').write_text('keep')
    with pytest.raises(FileExistsError):
        with r.fresh_run(folder,'tokens'):pass
    assert (folder/'keep').read_text()=='keep'


def test_no_old_controller_or_supplement_dependencies():
    for p in [Path(r.__file__),Path(r.__file__).parent/'scuq_radgraph/run.py']:
        text=p.read_text();assert 'supplement_v3' not in text and 'runs/' not in text and 'from scripts' not in text

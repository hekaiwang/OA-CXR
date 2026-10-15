"""Public release integration checks using synthetic data, never private records."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import pytest
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/public'))
from common import sha256_file
from prepare_private import prepare,validate_manifest
from compare_baselines import align


def synthetic_cohort(folder):
    folder.mkdir()
    pixels=np.tile(np.arange(96,dtype=np.uint8),(112,1))*2
    Image.fromarray(pixels).save(folder/'image.png')
    for name,box in [('left',(8,8,42,102)),('right',(54,8,88,102))]:
        mask=Image.new('L',(96,112));ImageDraw.Draw(mask).ellipse(box,fill=255)
        mask.save(folder/(name+'.png'))
    row=dict(source_id='case_'+'1'*16,group_id='group_'+'2'*16)
    for key,name in [('image','image'),('image_left_mask','left'),('image_right_mask','right')]:
        row[key]=name+'.png';row[key+'_sha256']=sha256_file(folder/(name+'.png'))
    value=dict(schema='oa-cxr-restricted-cohort-v1',dataset='synthetic_only',
        annotation_type='senior_physician_manual_lung_masks',samples=[row])
    path=folder/'manifest.json';path.write_text(json.dumps(value));return path


def test_local_manual_mask_interface_produces_shared_cache(tmp_path):
    # Synthetic masks exercise the declared manual-mask interface, not provenance verification.
    path=synthetic_cohort(tmp_path/'fixture');out=tmp_path/'cache'
    prepare(path,out)
    import export
    cached=export.cache_inputs(out,'external_private',labels=False)
    assert len(cached['rows'])==1 and not cached['target_values_loaded']
    targets=np.load(out/'external_private/targets.npy')
    assert targets.shape==(1,13,3) and np.isfinite(targets).all()
    np.testing.assert_array_equal(targets[0,0],np.ones(3))
    assert ((targets>=0)&(targets<=1)).all()


@pytest.mark.parametrize('problem',['extra_metadata','outside_path','stale_mask'])
def test_private_interface_rejects_metadata_paths_and_changed_annotations(tmp_path,problem):
    path=synthetic_cohort(tmp_path/'fixture');value=json.loads(path.read_text())
    if problem=='extra_metadata':value['samples'][0]['patient_name']='synthetic prohibited field'
    elif problem=='outside_path':value['samples'][0]['image']='../elsewhere.png'
    else:(path.parent/'left.png').write_bytes(b'changed annotation')
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):validate_manifest(path)


def test_comparison_requires_complete_bound_query_roster():
    row=dict(input_id='i',finding='pleural_effusion',dataset='synthetic',source_id='s',source_image_sha256='a'*64)
    assert np.isnan(align([row],[dict(row,score=None,status='failed')])[0])
    for records in [[],[dict(row,score=.5,status='success',source_id='other')],
                    [dict(row,score=None,status='success')],
                    [dict(row,score=.5,status='success')]*2]:
        with pytest.raises(ValueError):align([row],records)


def test_asset_reverification_detects_change_even_with_reused_metadata(tmp_path,monkeypatch):
    from baselines import asset_verification as a
    path=tmp_path/'tensor';path.write_bytes(b'original')
    receipt=a.verify_file(path,sha256_file(path),8)
    expected=receipt['sha256'];fingerprint=receipt['fingerprint']
    path.write_bytes(b'modified')
    monkeypatch.setattr(a,'fingerprint',lambda _path:fingerprint)
    with pytest.raises(ValueError,match='SHA differs'):a.verify_file(path,expected,8,receipt)


@pytest.mark.parametrize('problem',[None,'dropped_report','extra_claim','wrong_finding','changed_report','missing_image_score'])
def test_report_comparison_preserves_population_and_bound_claims(tmp_path,monkeypatch,problem):
    import export
    import compare_report_baselines as comparison
    from baselines import reporting as reporting
    from oa_cxr.io import write_jsonl
    from common import write_json
    path=synthetic_cohort(tmp_path/'fixture');cache=tmp_path/'cache';prepare(path,cache)
    sources=export.cache_inputs(cache,'external_private',labels=True)['rows']
    queries=list(export.query_records(sources,'external_private'))
    reports=[]
    for i,row in enumerate(queries[::3]):
        reports.append(dict(input={k:row[k] for k in ('input_id','source_id','source_image_sha256','dataset','split')},
            status='failed' if i==2 else 'success',
            generated_findings='No pleural effusion or pneumothorax. No consolidation.' if i==0 else 'Lungs are clear.'))
    claim_scores=[]
    for report in reports:
        for claim in reporting.report_claims(report):
            claim_scores.append(dict(report['input'],claim_id=claim['claim_id'],finding=claim['finding'],
                generated_findings_sha256=reporting.digest(report['generated_findings']),score=-.5,status='success'))
    # A scoring failure remains an eligible claim in the nominal population.
    claim_scores[-1].update(score=None,status='failed')
    image_scores=[dict(row,score=.8,status='success') for row in queries]
    if problem=='dropped_report':reports.pop()
    elif problem=='extra_claim':claim_scores.append(dict(claim_scores[0],claim_id='unknown'))
    elif problem=='wrong_finding':claim_scores[0]['finding']='unknown'
    elif problem=='changed_report':claim_scores[0]['generated_findings_sha256']='0'*64
    elif problem=='missing_image_score':image_scores.pop()
    generation=tmp_path/'generation';generation.mkdir()
    write_jsonl(generation/'reports.jsonl',reports)
    write_jsonl(generation/'denominators.jsonl',reporting.denominator(reports))
    write_json(generation/'protocol.json',dict(schema=reporting.SCHEMA,stage='generation'))
    write_json(generation/'status.json',dict(schema=reporting.SCHEMA,stage='generation',
        status='completed_with_failures',input_count=len(reports),
        output_sha256={name:sha256_file(generation/name) for name in ('reports.jsonl','denominators.jsonl','protocol.json')}))
    write_jsonl(tmp_path/'claims.jsonl',claim_scores);write_jsonl(tmp_path/'images.jsonl',image_scores)
    monkeypatch.setattr(sys,'argv',['compare_report_baselines.py','--cache',str(cache),'--split','external_private',
        '--generation',str(generation),'--report-score','sentence_logprob='+str(tmp_path/'claims.jsonl'),
        '--image-score','oa_cxr='+str(tmp_path/'images.jsonl'),'--reference','oa_cxr',
        '--bootstrap','5','--output',str(tmp_path/'comparison')])
    if problem:
        with pytest.raises(ValueError):comparison.main()
    else:
        comparison.main();value=json.loads((tmp_path/'comparison/metrics.json').read_text())
        result=value['datasets']['synthetic_only']
        assert value['full_report_denominator']==13 and value['generation_counts']['failed']==1
        assert result['eligible_claims']==3 and result['common_scored_claims']==2
        assert result['per_method_scored']==dict(oa_cxr=3,sentence_logprob=2)
        assert result['bootstrap']['original_cases']==1

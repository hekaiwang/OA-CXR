import importlib.util
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/public'))
from common import frozen_rows,checked,sha256_file,write_json
from oa_cxr.rebuild.cache_validation import validation_receipt


def test_published_cohorts_are_complete_and_disjoint():
    rows=frozen_rows();expected=dict(fit=19086,dev=2609,calibration=2651,test=2792)
    for split,n in expected.items():assert sum(r['split']==split for r in rows)==n
    for key in ('source_id','source_image_sha256','split_group_id'):
        groups={s:{r[key] for r in rows if r['split']==s} for s in expected}
        for a in groups:
            for b in groups:
                if a!=b:assert not groups[a]&groups[b]
    assert len(frozen_rows('external'))==6013


def test_same_size_rewrite_is_detected_without_timestamp_assumptions(tmp_path):
    p=tmp_path/'weights';p.write_bytes(b'original');digest=sha256_file(p)
    checked(p,digest);p.write_bytes(b'modified')
    with pytest.raises(ValueError,match='SHA256'):checked(p,digest)


def test_automated_receipt_is_bound_and_never_claims_visual_inspection(tmp_path):
    write_json(tmp_path/'status.json',dict(status='completed'))
    receipt=dict(schema='oa-cxr-public-cache-validation-v1',status='passed',mode='automated_integrity',
        source_and_annotation_sha256_checked=True,cache_status_sha256=sha256_file(tmp_path/'status.json'),
        visual_or_clinical_review_claimed=False)
    write_json(tmp_path/'validation.json',receipt)
    assert validation_receipt(tmp_path)==tmp_path/'validation.json'
    write_json(tmp_path/'status.json',dict(status='failed'))
    with pytest.raises(ValueError):validation_receipt(tmp_path)


def test_public_training_entrypoint_can_plan_without_historical_paths():
    import subprocess
    result=subprocess.run([sys.executable,str(ROOT/'scripts/public/train_pipeline.py'),'--help'],capture_output=True,text=True)
    assert result.returncode==0
    assert '--cache' in result.stdout and '--initial' in result.stdout

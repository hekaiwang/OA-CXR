"""Full SHA256 verification on every call, with explicit file identity receipts.

Previous receipts are retained for provenance only; they never replace hashing.
"""
from pathlib import Path
import os
from baselines.visual import read,write,sha,require,now

SCHEMA='oa-cxr-full-sha-metadata-cache-v1'


def fingerprint(path):
    s=Path(path).stat()
    return dict(size=s.st_size,inode=s.st_ino,device=s.st_dev,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def verify_file(path,expected_sha,expected_size,previous=None,*,force_full=False):
    path=Path(path).resolve();before=fingerprint(path)
    require(before['size']==expected_size,'Asset byte size differs: '+str(path))
    require(sha(path)==expected_sha,'Asset SHA differs: '+str(path))
    require(fingerprint(path)==before,'Asset changed while hashing: '+str(path))
    return dict(path=str(path),sha256=expected_sha,fingerprint=before,full_sha_verified_at=now(),checked_at=now(),
                verification_mode='full_sha256')


def verify_manifest(folder,manifest,receipt_path,manifest_sha,*,force_full=False):
    receipt_path=Path(receipt_path);previous={}
    if receipt_path.exists():
        value=read(receipt_path)
        if value.get('schema')==SCHEMA and value.get('manifest_sha256')==manifest_sha:previous=value.get('files',{})
    files={}
    for name,item in manifest['files'].items():
        p=(Path(folder)/name).resolve();require(p.is_relative_to(Path(folder).resolve()),'Asset leaves official root')
        files[name]=verify_file(p,item['sha256'],item['bytes'],previous.get(name),force_full=force_full)
    receipt_path.parent.mkdir(parents=True,exist_ok=True)
    value=dict(schema=SCHEMA,manifest_sha256=manifest_sha,files=files,checked_at=now(),
        policy='full SHA256 on every verification; no metadata-only acceptance')
    write(receipt_path,value);return value

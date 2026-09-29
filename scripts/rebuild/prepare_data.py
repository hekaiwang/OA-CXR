"""Audit an unused development pool and cache a new large-data experiment.

Historical identities are retained solely to prevent duplicate development/test
exposure. No old model, prediction, performance result, or threshold is loaded.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import os
import re
from pathlib import Path
import sys
import time
import traceback
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))

import numpy as np
from PIL import Image, ImageDraw

from oa_cxr.io import load_json, read_jsonl, sha256_file, stable_hash, write_json, write_jsonl, exclusive_writer
from oa_cxr.rebuild.data import (DATA_PROTOCOL, PROTOCOL_SHA256, fingerprint, NearIndex,
                                load_masks, variants, retention_targets, render_variant)

PREFIX='Lung Segmentation Data/Lung Segmentation Data/'
ARCHIVE_SHA='2a91b372cdd104d05d472f79dc446dc9bb7e3ccc26f513cf971f64cabc151333'
SEED='oa-cxr-large-training:17:'
VERSION='large-data-independent-development-v1'
_ARCHIVE=None


def _init_archive(path):
    global _ARCHIVE
    _ARCHIVE=zipfile.ZipFile(path)


def source_id(name):return 'covidqu-'+hashlib.sha256(name.encode()).hexdigest()[:24]


def choose_partition(row):
    h=int(hashlib.sha256((SEED+row['source_id']).encode()).hexdigest(),16)
    if row['official_split']=='Train':return 'test' if h%10==0 else 'fit'
    if row['official_split']=='Val':return 'dev' if h%2==0 else 'calibration'
    raise ValueError('Only previously unused Train/Val enters this development plan')


def kermany_group(row):
    """Conservative filename-family grouping, not validated patient identity."""
    stem=Path(row['original_filename']).stem
    match=re.match(r'(person\d+)_',stem)
    if match:return 'kermany-family:'+match.group(1)
    match=re.match(r'((?:NORMAL2-)?IM-\d+)-',stem)
    if match:return 'kermany-family:'+match.group(1)
    return 'kermany-image:'+row['source_image_sha256']


def kermany_partitions(rows):
    groups={}
    for row in rows:groups.setdefault(kermany_group(row),[]).append(row)
    result={}
    for group,members in groups.items():
        official={r['official_split'].lower() for r in members}
        if not official<={'train','val','test'}:raise ValueError('Unknown Kermany release partition')
        if 'test' in official:split='test'
        elif 'val' in official:split='dev'
        else:
            remainder=int(hashlib.sha256((SEED+group).encode()).hexdigest(),16)%10
            split='dev' if remainder==0 else 'calibration' if remainder==1 else 'fit'
        result[group]=split
    return result


def verify_annotation_bytes(row):
    if row['source_kind']=='union_mask':
        pairs=[(row['mask_path'],row['mask_sha256'])]
    elif row['source_kind']=='paired_masks':
        pairs=[(row[side+'_mask_path'],row['manual_mask_sha256'][side]) for side in ('left','right')]
    else:raise ValueError('Unsupported annotation format')
    if any(sha256_file(path)!=digest for path,digest in pairs):raise ValueError('Annotation bytes changed')


def _stage_kermany(row):
    row=dict(row)
    try:
        raw=Path(row['source_image_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['source_image_sha256']:raise ValueError('Image bytes changed')
        verify_annotation_bytes(row);fp=fingerprint(raw);masks=load_masks(row,tuple(fp['size']))
        retention_targets(masks,[b for _,b in variants(tuple(fp['size']))])
        row.update(source_id=row['subject_id'],fingerprint=fp,split_group_id=kermany_group(row),
                   patient_independence_verified=False,
                   historical_exposure='Previously evaluated exploratory cohort; new fit/dev/test partition, not blind confirmation')
        return {'row':row}
    except (ValueError,OSError,KeyError) as exc:
        return {'excluded':row,'reason':type(exc).__name__+': '+str(exc)}


def _stage_one(args):
    archive,name,rawdir=args
    row={'source_id':source_id(name),'dataset':'covidqu_v7','official_split':name[len(PREFIX):].split('/')[0],
         'official_class':name[len(PREFIX):].split('/')[1],'archive_member':name,
         'source_kind':'union_mask','annotation_type':'published human-machine lung mask',
         'patient_id':None,'patient_independence_verified':False}
    try:
        if _ARCHIVE is None:
            with zipfile.ZipFile(archive) as z:
                raw=z.read(name);mask=z.read(name.replace('/images/','/lung masks/'))
        else:
            raw=_ARCHIVE.read(name);mask=_ARCHIVE.read(name.replace('/images/','/lung masks/'))
        imagepath=Path(rawdir)/(row['source_id']+'.png');maskpath=Path(rawdir)/(row['source_id']+'_mask.png')
        for path,data in ((imagepath,raw),(maskpath,mask)):
            if path.exists():
                if path.read_bytes()!=data:raise ValueError('Existing raw file differs')
            else:path.write_bytes(data)
        row.update(source_image_path=str(imagepath),source_image_sha256=hashlib.sha256(raw).hexdigest(),
                   mask_path=str(maskpath),mask_sha256=hashlib.sha256(mask).hexdigest())
        fp=fingerprint(raw)
        masks=load_masks(row,tuple(fp['size']))
        targets=retention_targets(masks,[box for _,box in variants(tuple(fp['size']))])
        assert targets.shape==(13,3)
        row.update(fingerprint=fp,split_group_id='image-'+fp['decoded_sha256'])
        return {'row':row}
    except (ValueError,OSError,KeyError) as exc:
        return {'excluded':row,'reason':type(exc).__name__+': '+str(exc)}


def audit(archive,historical,out,workers):
    if out.exists():raise FileExistsError('Use a fresh audit output; do not overwrite partial attempts')
    if sha256_file(archive)!=ARCHIVE_SHA:raise ValueError('Pinned data archive differs')
    history=read_jsonl(historical)
    if not history:raise ValueError('Historical source identities required before rebuilding')
    out.mkdir(parents=True);(out/'raw').mkdir()
    protocol={'version':VERSION,'seed':SEED,'archive_sha256':ARCHIVE_SHA,
        'data_protocol':DATA_PROTOCOL,'data_protocol_sha256':PROTOCOL_SHA256,
        'historical_catalog_sha256':sha256_file(historical),
        'cohort_order':'Historical other sources protected against Kermany; canonical Kermany then unused COVIDQU Val/Train. All history protected against COVIDQU.',
        'duplicates':'exclude bytes/RGB/gray256 exact or dhash256<=8 AND gray32 MAE<=12/255; retain all audit rows',
        'partition_rule':'Train hash modulo10==0 new image-level test else fit; Val parity0 dev else calibration',
        'kermany_partition_rule':'filename family grouped; any official Test member protects whole family in test; Val family dev; remaining hashmod10=0 dev,1 calibration,else fit',
        'kermany_scope':'previously seen cohort repartitioned for pediatric training; reserved test is exploratory, not new blind confirmation',
        'minimum_unique_fit_sources':10000,'model_parameters_selected_from_test':False,
        'new_test_scope':'previously unused images, not verified independent patients or independent of foundation pretraining',
        'clinical_support_labels':False,'implementation_sha256':{
            relative:sha256_file(ROOT/relative) for relative in
            ('scripts/rebuild/prepare_data.py','src/oa_cxr/rebuild/data.py','src/oa_cxr/anatomy_geometry.py')}}
    write_json(out/'protocol.json',protocol)
    state={'status':'running','stage':'historical_identity_index','started_at':time.time(),'processed':0}
    write_json(out/'status.json',state)
    index=NearIndex();k_protected=NearIndex();seen=set();historical_rows=[]
    for row in history:
        path=row.get('source_image_path')
        if not path or path in seen:continue
        seen.add(path);p=Path(path)
        fp=fingerprint(p.read_bytes())
        expected=row.get('source_image_sha256')
        if expected and fp['file_sha256']!=expected:raise ValueError('Historical source hash differs: '+path)
        entry={**fp,'identity':'historical:'+str(p),'role':'historical','dataset':row.get('dataset')}
        index.add(entry);historical_rows.append({k:v for k,v in entry.items() if k!='thumbnail_hex'})
        if row.get('dataset')!='kermany_v7':k_protected.add(entry)
        if len(seen)%200==0:
            state.update(historical_indexed=len(seen));write_json(out/'status.json',state)
    write_jsonl(out/'historical_identity_index.jsonl',historical_rows)
    with zipfile.ZipFile(archive) as z:
        names=[n for split in ('Val','Train') for n in sorted(z.namelist())
               if n.startswith(PREFIX+split+'/') and '/images/' in n and n.endswith('.png')]
    if len(names)!=27132:raise ValueError('Expected official Train21715+Val5417')
    write_json(out/'nominal_sources.json',{'archive_members':names,'count':len(names)})
    krows=[r for r in history if r.get('dataset')=='kermany_v7' and
        r.get('exposure_provenance',{}).get('source_manifest')=='runs/reviewed_large_external_sources_20260922/kermany/sources.jsonl']
    if len(krows)!=4547:raise ValueError('Expected4547 canonical pediatric source identities')
    ksplit=kermany_partitions(krows)
    accepted=[];excluded=[];state.update(stage='pediatric_annotation_and_overlap_audit',nominal=len(names)+len(krows))
    write_json(out/'status.json',state)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,result in enumerate(pool.map(_stage_kermany,sorted(krows,key=lambda r:r['subject_id']),chunksize=4),1):
            if 'excluded' in result:excluded.append(result)
            else:
                row=result['row'];matches=k_protected.matches(row['fingerprint'])
                if matches:
                    excluded.append({'excluded':{k:v for k,v in row.items() if k!='fingerprint'},
                        'reason':'exact_or_near_duplicate','matches':[k_protected.rows[j]['identity'] for j in matches]})
                else:
                    row['split']=ksplit[row['split_group_id']];accepted.append(row)
                    k_protected.add({**row['fingerprint'],'identity':row['source_id'],'role':row['split']})
            if i%100==0:
                state.update(processed=i,accepted=len(accepted),excluded=len(excluded));write_json(out/'status.json',state)
    state.update(stage='covidqu_annotation_and_overlap_audit')
    write_json(out/'status.json',state)
    with ProcessPoolExecutor(max_workers=workers,initializer=_init_archive,initargs=(str(archive),)) as pool:
        tasks=((str(archive),name,str(out/'raw')) for name in names)
        for i,result in enumerate(pool.map(_stage_one,tasks,chunksize=16),1):
            if 'excluded' in result:excluded.append(result)
            else:
                row=result['row'];fp=row['fingerprint'];matches=index.matches(fp)
                if matches:
                    excluded.append({'excluded':{k:v for k,v in row.items() if k!='fingerprint'},
                        'reason':'exact_or_near_duplicate','matches':[index.rows[j]['identity'] for j in matches]})
                else:
                    row['split']=choose_partition(row);row['subject_id']=row['source_id']
                    accepted.append(row)
                    index.add({**fp,'identity':row['source_id'],'role':row['split']})
            if i%100==0:
                state.update(processed=i+len(krows),accepted=len(accepted),excluded=len(excluded),split_counts=dict(Counter(r['split'] for r in accepted)))
                write_json(out/'status.json',state);print(json.dumps(state),flush=True)
    write_jsonl(out/'sources.jsonl',accepted);write_jsonl(out/'excluded.jsonl',excluded)
    counts=Counter(r['split'] for r in accepted)
    if counts['fit']<10000:
        state.update(status='insufficient_training_sources',split_counts=dict(counts))
        write_json(out/'status.json',state);raise ValueError('Fewer than10000 unique eligible fit sources; do not alter split to chase results')
    if not all(counts[k]>0 for k in ('dev','calibration','test')):raise ValueError('Empty declared partition')
    group_splits={}
    for row in accepted:
        previous=group_splits.setdefault(row['split_group_id'],row['split'])
        if previous!=row['split']:raise ValueError('Source group crosses declared partitions')
    preview=[]
    for dataset in ('covidqu_v7','kermany_v7'):
        for split in ('fit','dev','calibration','test'):
            preview.extend(sorted((r for r in accepted if r['split']==split and r['dataset']==dataset),
                                  key=lambda r:stable_hash(r['source_id']))[:2])
    sheet=Image.new('RGB',(1024,((len(preview)+3)//4)*280),'white');draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(preview):
        with Image.open(row['source_image_path']) as im:
            image=im.convert('L');m=load_masks(row,im.size)
            x,y=render_variant(image,m,(0,0,im.width,im.height))
        rgb=np.repeat(x[...,None],3,-1);union=y.any(0)
        rgb[union]=(rgb[union]*.65+np.array([0,180,40])*.35).astype(np.uint8)
        xx,yy=i%4*256,i//4*280;sheet.paste(Image.fromarray(rgb),(xx,yy));draw.text((xx+4,yy+258),row['dataset']+' '+row['split']+' '+row['source_id'][-6:],fill='black')
    sheet.save(out/'source_contact_sheet.png');write_json(out/'preview_sources.json',preview)
    state.update(status='audited_pending_visual_review',stage='audit_complete',processed=len(names)+len(krows),
        accepted=len(accepted),excluded=len(excluded),split_counts=dict(counts),finished_at=time.time(),
        by_dataset_split=dict(Counter(r['dataset']+':'+r['split'] for r in accepted)),
        files={n:sha256_file(out/n) for n in ('protocol.json','sources.jsonl','excluded.jsonl','source_contact_sheet.png','preview_sources.json')})
    state['artifact_sha256']=stable_hash(state);write_json(out/'status.json',state)
    return state


def _render_one(row):
    if sha256_file(row['source_image_path'])!=row['source_image_sha256']:raise ValueError('Image bytes changed')
    verify_annotation_bytes(row)
    with Image.open(row['source_image_path']) as im:
        image=im.convert('L');masks=load_masks(row,im.size)
    plan=variants(image.size);y=retention_targets(masks,[b for _,b in plan])
    rendered=[render_variant(image,masks,b) for _,b in plan]
    images=np.stack([x for x,_ in rendered]);masks=np.stack([m for _,m in rendered])
    packed=np.packbits(masks.reshape(13,2,-1),axis=-1)
    return images,packed,y


def cache(audit_dir,out,workers):
    if out.exists():raise FileExistsError('Cache requires fresh output directory')
    state=load_json(audit_dir/'status.json');review=load_json(audit_dir/'visual_review.json')
    if (state.get('status')!='audited_pending_visual_review' or review.get('status')!='passed'
            or review.get('source_contact_sheet_sha256')!=sha256_file(audit_dir/'source_contact_sheet.png')
            or review.get('audit_status_sha256')!=sha256_file(audit_dir/'status.json')):
        raise ValueError('Actual bound source technical review required')
    for name,digest in state['files'].items():
        if sha256_file(audit_dir/name)!=digest:raise ValueError('Audited bytes changed')
    rows=read_jsonl(audit_dir/'sources.jsonl');out.mkdir(parents=True)
    protocol={**load_json(audit_dir/'protocol.json'),'audit_artifact_sha256':state['artifact_sha256'],
        'audit_sources_sha256':sha256_file(audit_dir/'sources.jsonl'),'variants':13,'findings':list(DATA_PROTOCOL['findings']),
        'training_unique_source_minimum':10000}
    write_json(out/'protocol.json',protocol)
    status={'status':'running','stage':'cache','started_at':time.time(),'files':{},'splits':{}}
    write_json(out/'status.json',status)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for split in ('fit','dev','calibration','test'):
            selected=sorted((r for r in rows if r['split']==split),key=lambda r:r['source_id'])
            p=out/split;p.mkdir();n=len(selected)
            image_array=np.lib.format.open_memmap(p/'images.npy',mode='w+',dtype=np.uint8,shape=(n,13,256,256))
            mask_array=np.lib.format.open_memmap(p/'masks.npy',mode='w+',dtype=np.uint8,shape=(n,13,2,8192))
            target_array=np.lib.format.open_memmap(p/'targets.npy',mode='w+',dtype=np.float32,shape=(n,13,3))
            for i,(x,m,y) in enumerate(pool.map(_render_one,selected,chunksize=4)):
                image_array[i]=x;mask_array[i]=m;target_array[i]=y
                if (i+1)%100==0:
                    status.update(current_split=split,split_processed=i+1,split_total=n);write_json(out/'status.json',status)
            image_array.flush();mask_array.flush();target_array.flush();del image_array,mask_array,target_array
            sources=[{**{k:v for k,v in r.items() if k!='fingerprint'},'row_index':i} for i,r in enumerate(selected)]
            write_jsonl(p/'sources.jsonl',sources)
            status['splits'][split]={'sources':n,'inputs':n*13,'targets':n*39}
            for name in ('images.npy','masks.npy','targets.npy','sources.jsonl'):
                status['files'][split+'/'+name]=sha256_file(p/name)
            write_json(out/'status.json',status)
    status['files']['protocol.json']=sha256_file(out/'protocol.json')
    status.update(status='completed',finished_at=time.time(),source_review_sha256=sha256_file(audit_dir/'visual_review.json'),
                  cache_visual_review_pending=True,clinical_support_labels=False)
    status['artifact_sha256']=stable_hash(status);write_json(out/'status.json',status)
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('audit','cache'))
    p.add_argument('--archive',type=Path,default=ROOT/'dataset/covidqu_v3/covidqu-v7.zip')
    p.add_argument('--historical',type=Path,default=ROOT/'dataset/catalog/source_identity.jsonl')
    p.add_argument('--audit',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT/'dataset') or out==ROOT/'dataset':raise ValueError('Output must be a dataset child')
    if not 1<=a.workers<=8:raise ValueError('Use1..8 CPU workers')
    with exclusive_writer(out):
        if out.exists():raise FileExistsError('Do not overwrite an existing attempt')
        try:
            result=audit(a.archive.resolve(),a.historical.resolve(),out,a.workers) if a.action=='audit' else cache(a.audit.resolve(),out,a.workers)
        except Exception as exc:
            if out.exists():
                state=load_json(out/'status.json') if (out/'status.json').exists() else {}
                state.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),finished_at=time.time())
                write_json(out/'status.json',state)
            raise
    print(json.dumps(result),flush=True)

"""Prepare an authorized local cohort with physician-drawn lung masks; no upload."""
from common import *
import argparse
import re
import numpy as np
from PIL import Image
from oa_cxr.rebuild.data import DATA_PROTOCOL,variants,retention_targets,render_variant
from oa_cxr.io import write_jsonl

FIELDS={'source_id','group_id','image','image_sha256','image_left_mask','image_left_mask_sha256',
        'image_right_mask','image_right_mask_sha256'}


def validate_manifest(path):
    path=Path(path).resolve();value=load_json(path)
    if set(value)!={'schema','dataset','annotation_type','samples'} or value['schema']!='oa-cxr-restricted-cohort-v1':
        raise ValueError('Use the documented restricted-cohort schema; clinical metadata is not accepted')
    if value['annotation_type']!='senior_physician_manual_lung_masks':raise ValueError('Physician manual masks required')
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,63}',value['dataset']):raise ValueError('Use a non-identifying dataset alias')
    if not isinstance(value['samples'],list) or not value['samples']:raise ValueError('Nonempty samples required')
    seen=set();images=set();prepared=[]
    for item in value['samples']:
        if set(item)!=FIELDS:raise ValueError('Unexpected sample fields; no clinical text or patient metadata is accepted')
        for name,prefix in [('source_id','case'),('group_id','group')]:
            if not re.fullmatch(prefix+r'_[0-9a-f]{16,64}',item[name]):
                raise ValueError('Use random non-identifying case/group aliases; keep the mapping private')
        if item['source_id'] in seen or item['image_sha256'] in images:raise ValueError('Duplicate source image')
        seen.add(item['source_id']);images.add(item['image_sha256']);record=dict(item)
        for name in ('image','image_left_mask','image_right_mask'):
            relative=Path(item[name])
            if relative.is_absolute():raise ValueError('Manifest paths must be relative to its local directory')
            file=(path.parent/relative).resolve()
            if not file.is_relative_to(path.parent):raise ValueError('Input path leaves the local cohort directory')
            if not re.fullmatch('[0-9a-f]{64}',item[name+'_sha256']):raise ValueError('Explicit source/mask SHA256 required')
            checked(file,item[name+'_sha256']);record[name]=file
        prepared.append(record)
    return value,prepared


def prepare(manifest,output):
    value,records=validate_manifest(manifest);out=fresh(output);split='external_private'
    folder=out/split;folder.mkdir();n=len(records)
    write_json(out/'protocol.json',dict(schema='oa-cxr-local-private-cache-v1',data_protocol=DATA_PROTOCOL,
        findings=DATA_PROTOCOL['findings'],variants=13,dataset=value['dataset'],
        annotation_type=value['annotation_type'],annotation_provenance='author-supplied declaration',
        manifest_sha256=sha256_file(manifest),data_public=False,model_training=False,
        input_requirement='de-identified source images and aligned manual masks supplied locally; no anonymization is performed'))
    state=dict(status='running',files={},splits={},clinical_support_labels=False,data_public=False)
    write_json(out/'status.json',state)
    try:
        arrays=[np.lib.format.open_memmap(folder/name,mode='w+',dtype=dtype,shape=shape)
            for name,dtype,shape in [('images.npy',np.uint8,(n,13,256,256)),
                ('masks.npy',np.uint8,(n,13,2,8192)),('targets.npy',np.float32,(n,13,3))]]
        rows=[]
        for i,row in enumerate(records):
            for key in ('image','image_left_mask','image_right_mask'):checked(row[key],row[key+'_sha256'])
            with Image.open(row['image']) as source:
                if getattr(source,'n_frames',1)!=1 or source.getexif().get(274,1)!=1:
                    raise ValueError('Resolve multi-frame images and orientation before preparing the cohort')
                image=source.convert('L')
            masks=[]
            for key in ('image_left_mask','image_right_mask'):
                with Image.open(row[key]) as mask:
                    if mask.size!=image.size or mask.mode not in ('1','L'):raise ValueError('Native-size binary lung masks required')
                    raw=np.asarray(mask)
                    if not set(np.unique(raw)).issubset({0,1,255}):raise ValueError('Binary masks required')
                    masks.append(raw>0)
            if not all(m.any() for m in masks) or (masks[0]&masks[1]).any():raise ValueError('Two nonempty disjoint manual masks required')
            if np.nonzero(masks[0])[1].mean()>=np.nonzero(masks[1])[1].mean():
                raise ValueError('Use image-coordinate left/right mask order, not clinical laterality')
            masks=np.stack(masks)
            boxes=[box for _,box in variants(image.size)]
            targets=retention_targets(masks,boxes)
            transformed=[render_variant(image,masks,box) for box in boxes]
            arrays[0][i]=np.stack([x for x,_ in transformed])
            arrays[1][i]=np.packbits(np.stack([m for _,m in transformed]).reshape(13,2,-1),axis=-1)
            arrays[2][i]=targets
            rows.append(dict(source_id=row['source_id'],split_group_id=row['group_id'],dataset=value['dataset'],
                split=split,row_index=i,source_image_sha256=row['image_sha256'],source_image_path=str(row['image'])))
        for array in arrays:array.flush()
        del arrays
        write_jsonl(folder/'sources.jsonl',rows)
        state['splits'][split]=dict(sources=n,inputs=n*13,targets=n*39)
        state['files']={str(p.relative_to(out)):sha256_file(p) for p in folder.iterdir() if p.is_file()}
        state['files']['protocol.json']=sha256_file(out/'protocol.json');state['status']='completed'
    except BaseException as exc:state.update(status='failed',error=str(exc));raise
    finally:write_json(out/'status.json',state)
    write_json(out/'validation.json',dict(schema='oa-cxr-public-cache-validation-v1',status='passed',
        mode='automated_integrity',source_and_annotation_sha256_checked=True,
        cache_status_sha256=sha256_file(out/'status.json'),visual_or_clinical_review_claimed=False))
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.manifest,args.output)

"""Prepare the frozen NIH/CheXmask and Shenzhen evaluation cohorts."""
from common import *
import argparse
import csv
import importlib.util
import numpy as np
from PIL import Image

# Avoid the similarly named historical helper on scripts/rebuild's import path.
spec=importlib.util.spec_from_file_location('public_prepare_data',Path(__file__).with_name('prepare_data.py'))
prepare=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=prepare
spec.loader.exec_module(prepare)


def decode_rle(text,height,width):
    runs=np.fromstring(text,sep=' ',dtype=np.int64)
    if len(runs)%2:raise ValueError('Malformed CheXmask RLE')
    mask=np.zeros(height*width,np.uint8)
    for start,length in zip(runs[::2],runs[1::2]):
        if start<1 or length<1 or start-1+length>len(mask):raise ValueError('Out-of-bounds RLE')
        mask[start-1:start-1+length]=255
    return mask.reshape(height,width)


def lookup(root):
    result={}
    for p in Path(root).rglob('*.png'):result.setdefault(p.name,[]).append(p)
    return result


def resolve(index,name,digest):
    for path in index.get(name,[]):
        if sha256_file(path)==digest:return path.resolve()
    raise ValueError('Missing external file matching SHA: '+name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nih-images',type=Path)
    p.add_argument('--chexmask-csv',type=Path)
    p.add_argument('--shenzhen-images',type=Path)
    p.add_argument('--shenzhen-masks',type=Path)
    p.add_argument('--split',choices=['external_nih','external_shenzhen'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--smoke-sources',type=int)
    a=p.parse_args();rows=[r for r in frozen_rows('external') if r['split']==a.split]
    if a.smoke_sources:rows=rows[:a.smoke_sources]
    out=fresh(a.output)
    if a.split=='external_nih':
        if not a.nih_images or not a.chexmask_csv:p.error('NIH images and CheXmask CSV required')
        images=lookup(a.nih_images);wanted={r['original_filename']:r for r in rows};masks=out/'masks';masks.mkdir()
        found=set()
        with a.chexmask_csv.open(newline='') as stream:
            for record in csv.DictReader(stream):
                name=record['Image Index']
                if name not in wanted:continue
                row=wanted[name];h,w=int(record['Height']),int(record['Width'])
                # Channels follow image-x position, not the anatomical label name.
                candidates=[decode_rle(record[k],h,w) for k in ('Left Lung','Right Lung')]
                candidates.sort(key=lambda m:float(np.nonzero(m)[1].mean()))
                for side,mask in zip(('left','right'),candidates):
                    target=masks/row['mask_filenames'][side];Image.fromarray(mask).save(target)
                    checked(target,row['manual_mask_sha256'][side]);row[side+'_mask_path']=str(target)
                row['source_image_path']=str(resolve(images,name,row['source_image_sha256']));found.add(name)
        if found!=set(wanted):raise ValueError('Required CheXmask annotations missing')
    else:
        if not a.shenzhen_images or not a.shenzhen_masks:p.error('Shenzhen images and masks required')
        images=lookup(a.shenzhen_images);masks=lookup(a.shenzhen_masks)
        for row in rows:
            row['source_image_path']=str(resolve(images,row['image_filename'],row['source_image_sha256']))
            row['mask_path']=str(resolve(masks,row['mask_filename'],row['mask_sha256']))
    print(json.dumps(prepare.cache_rows(rows,out/'cache',a.workers,manifest_name='external'),indent=2))


if __name__=='__main__':main()

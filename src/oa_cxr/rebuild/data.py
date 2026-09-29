"""Image-only model inputs; independent masks are used solely for supervision.

No empty predicted-mask exclusion is present in this data contract. Source-mask
technical defects are audited before splitting, never by a model's test score.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label

from oa_cxr.anatomy_geometry import prepare_manual_regions
from oa_cxr.io import stable_hash

FINDINGS = ('pleural_effusion', 'pneumothorax', 'consolidation')
IMAGE_SIZE = 256
DATA_PROTOCOL = {
    'version': 'large-data-cxr-supervision-v1', 'findings': list(FINDINGS),
    'image_size': IMAGE_SIZE, 'image_transform': 'crop then preserve-aspect letterbox256; bilinear image; nearest masks',
    'model_input': 'current image only; no source ID, original image, mask, target, or crop parameters',
    'normalization': '(gray/255*2-1)*1024',
    'mask_channels': 'image-x ordered annotation regions; not clinical laterality',
    'target': 'minimum per-lung native-annotation ROI fraction inside crop',
    'regions': {'pleural_effusion': 'bottom25% bbox', 'pneumothorax': 'inner2% bbox-short-edge rim', 'consolidation': 'whole annotated visible lung'},
    'clinical_support_labels': False,
}
PROTOCOL_SHA256 = stable_hash(DATA_PROTOCOL)


def fingerprint(raw: bytes):
    with Image.open(io.BytesIO(raw)) as im:
        if getattr(im, 'n_frames', 1) != 1 or im.getexif().get(274, 1) != 1:
            raise ValueError('Unsupported frames/orientation')
        rgb = im.convert('RGB'); gray = rgb.convert('L')
        a = np.asarray(gray.resize((17, 16), Image.Resampling.BILINEAR))
        return {'file_sha256': hashlib.sha256(raw).hexdigest(),
            'decoded_sha256': stable_hash({'size': rgb.size, 'rgb': hashlib.sha256(rgb.tobytes()).hexdigest()}),
            'gray256_sha256': hashlib.sha256(gray.resize((256,256), Image.Resampling.BICUBIC).tobytes()).hexdigest(),
            'dhash256': np.packbits(a[:,1:] > a[:,:-1]).tobytes().hex(),
            'thumbnail_hex': gray.resize((32,32), Image.Resampling.BILINEAR).tobytes().hex(),
            'size': list(rgb.size)}


class NearIndex:
    """Exact pigeonhole candidates for 256-bit Hamming<=8, followed by pixel check."""
    def __init__(self):
        self.rows = []; self.buckets = [{} for _ in range(9)]
        self.exact = {k:{} for k in ('file_sha256','decoded_sha256','gray256_sha256')}

    @staticmethod
    def bands(value):
        return [(value >> (i*28)) & ((1 << (32 if i == 8 else 28))-1) for i in range(9)]

    def matches(self, row):
        candidates = set()
        for key, mapping in self.exact.items(): candidates.update(mapping.get(row[key], []))
        value = int(row['dhash256'], 16)
        for b, bucket in zip(self.bands(value), self.buckets): candidates.update(bucket.get(b, []))
        thumb = np.frombuffer(bytes.fromhex(row['thumbnail_hex']), dtype=np.uint8).astype(np.int16)
        result=[]
        for i in sorted(candidates):
            old=self.rows[i]
            exact=any(old[k]==row[k] for k in self.exact)
            if exact or ((int(old['dhash256'],16)^value).bit_count() <= 8 and
                np.abs(thumb-np.frombuffer(bytes.fromhex(old['thumbnail_hex']),dtype=np.uint8).astype(np.int16)).mean() <= 12):
                result.append(i)
        return result

    def add(self,row):
        i=len(self.rows);self.rows.append(row)
        for key, mapping in self.exact.items():mapping.setdefault(row[key],[]).append(i)
        for b,bucket in zip(self.bands(int(row['dhash256'],16)),self.buckets):bucket.setdefault(b,[]).append(i)
        return i


def split_union(binary):
    if binary.ndim != 2 or binary.dtype != np.bool_ or not binary.any():
        raise ValueError('Nonempty 2D annotation mask required')
    regions,n = label(binary, structure=np.ones((3,3),dtype=np.uint8))
    areas=np.bincount(regions.ravel());order=sorted(range(1,n+1),key=lambda i:(-areas[i],i))
    if len(order)<2:raise ValueError('Annotation needs two separately identifiable lung regions')
    total=int(binary.sum())
    if any(areas[i]*20 < total for i in order[:2]) or sum(areas[i] for i in order[2:])*1000 > total:
        raise ValueError('Annotation component areas outside fixed technical rule')
    centers={i:np.argwhere(regions==i).mean(axis=0) for i in order}
    main=sorted(order[:2],key=lambda i:(centers[i][1],centers[i][0],i))
    assignment={i:j for j,i in enumerate(main)}
    for i in order[2:]:assignment[i]=int(np.argmin([np.square(centers[i]-centers[m]).sum() for m in main]))
    masks=np.stack([np.isin(regions,[i for i,j in assignment.items() if j==side]) for side in (0,1)])
    if np.any(masks[0]&masks[1]) or not np.array_equal(masks.any(0),binary):
        raise ValueError('Annotation decomposition changed pixels')
    return masks


def load_masks(row, size):
    if row['source_kind']=='union_mask':
        with Image.open(row['mask_path']) as im:
            if im.size!=size:raise ValueError('Annotation/image dimensions differ')
            if im.mode not in ('1','L'):raise ValueError('Only explicit binary/grayscale annotation files accepted')
            a=np.asarray(im.convert('L'))
        if not set(np.unique(a)).issubset({0,1,255}):raise ValueError('Nonbinary annotation')
        return split_union(a>0)
    if row['source_kind']=='paired_masks':
        masks=[]
        for key in ('left_mask_path','right_mask_path'):
            with Image.open(row[key]) as im:
                if im.size!=size:raise ValueError('Annotation/image dimensions differ')
                if im.mode not in ('1','L'):raise ValueError('Only explicit binary/grayscale annotation files accepted')
                a=np.asarray(im.convert('L'))
            if not set(np.unique(a)).issubset({0,1,255}):raise ValueError('Nonbinary annotation')
            masks.append(a>0)
        masks=np.stack(masks)
        if not masks[0].any() or not masks[1].any() or np.any(masks[0]&masks[1]):
            raise ValueError('Empty or overlapping source annotation')
        if np.argwhere(masks[0])[:,1].mean()>np.argwhere(masks[1])[:,1].mean():masks=masks[::-1].copy()
        return masks
    raise ValueError('Unsupported source annotation kind')


def variants(size):
    w,h=size;rows=[('original',(0,0,w,h))]
    for direction in ('top','bottom','left','right'):
        for fraction in (.1,.2):
            dx,dy=round(fraction*w),round(fraction*h)
            box=(dx if direction=='left' else 0,dy if direction=='top' else 0,
                 w-dx if direction=='right' else w,h-dy if direction=='bottom' else h)
            rows.append((f'{direction}_{int(fraction*100)}',box))
    dx,dy=round(.15*w),round(.15*h)
    for vertical in ('top','bottom'):
        for horizontal in ('left','right'):
            rows.append((vertical+'_'+horizontal+'_15',(dx if horizontal=='left' else 0,
                dy if vertical=='top' else 0,w-dx if horizontal=='right' else w,h-dy if vertical=='bottom' else h)))
    return rows


def _area(mask,box):
    x0,y0,x1,y1=box;h,w=mask.shape
    wx=np.maximum(0,np.minimum(np.arange(w)+1,x1)-np.maximum(np.arange(w),x0))
    wy=np.maximum(0,np.minimum(np.arange(h)+1,y1)-np.maximum(np.arange(h),y0))
    return float(np.sum(mask*wy[:,None]*wx[None,:]))


def _integral_area(integral,box):
    """Exact continuous unit-pixel area, including fractional ROI edges."""
    h,w=np.array(integral.shape)-1
    x0,y0,x1,y1=box
    if x1<=x0 or y1<=y0:return 0.
    def prefix(x,y):
        x=float(np.clip(x,0,w));y=float(np.clip(y,0,h))
        ix,iy=int(x),int(y);jx,jy=min(ix+1,w),min(iy+1,h)
        fx,fy=x-ix,y-iy
        return ((1-fx)*(1-fy)*integral[iy,ix]+fx*(1-fy)*integral[iy,jx]
                +(1-fx)*fy*integral[jy,ix]+fx*fy*integral[jy,jx])
    return float(prefix(x1,y1)-prefix(x0,y1)-prefix(x1,y0)+prefix(x0,y0))


def retention_targets(masks,boxes):
    regions=prepare_manual_regions(masks[0],masks[1]).regions
    integrals=[np.pad(r.mask.astype(np.float64).cumsum(0).cumsum(1),((1,0),(1,0))) for r in regions]
    result=[]
    for box in boxes:
        per={f:[] for f in FINDINGS}
        for region,integral in zip(regions,integrals):
            b=(max(box[0],region.box[0]),max(box[1],region.box[1]),min(box[2],region.box[2]),min(box[3],region.box[3]))
            per[region.finding].append(_integral_area(integral,b)/region.source_area)
        result.append([min(per[f]) for f in FINDINGS])
    result=np.asarray(result,dtype=np.float32)
    if not np.isfinite(result).all() or (result<0).any() or (result>1+1e-6).any():raise ValueError('Invalid independent target')
    return np.clip(result,0,1)


def letterbox(image, size=IMAGE_SIZE, *, mask=False):
    w,h=image.size;scale=size/max(w,h);nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    output=Image.new('L',(size,size),0)
    image=image.convert('L').resize((nw,nh),Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR)
    output.paste(image,((size-nw)//2,(size-nh)//2))
    return np.asarray(output,dtype=np.uint8)


def render_variant(image,masks,box,size=IMAGE_SIZE):
    if (len(box)!=4 or any(type(v) is not int for v in box) or
        not (0<=box[0]<box[2]<=image.width and 0<=box[1]<box[3]<=image.height)):
        raise ValueError('Crop must be an integer nonempty box within the source image')
    if masks.shape!=(2,image.height,image.width):raise ValueError('Source masks not aligned with source image')
    x=letterbox(image.crop(box),size)
    y=np.stack([letterbox(Image.fromarray(m.astype(np.uint8)*255).crop(box),size,mask=True)>0 for m in masks])
    return x,y

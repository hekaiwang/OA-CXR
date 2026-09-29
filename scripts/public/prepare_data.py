"""Rebuild the frozen main cohort from official archives and released V7 masks.

The published split manifest is the output of the original deduplication audit.
This command replays that membership, not a new split or a new audit decision.
"""
from common import *
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import shutil
import zipfile
import numpy as np
from oa_cxr.rebuild.data import DATA_PROTOCOL, load_masks, variants, retention_targets, render_variant
from PIL import Image


def render(row):
    checked(row['source_image_path'], row['source_image_sha256'])
    if row['source_kind'] == 'union_mask':
        checked(row['mask_path'], row['mask_sha256'])
    else:
        for side in ('left', 'right'):
            checked(row[side + '_mask_path'], row['manual_mask_sha256'][side])
    with Image.open(row['source_image_path']) as image:
        masks = load_masks(row, image.size)
        plan = variants(image.size)
        targets = retention_targets(masks, [box for _, box in plan])
        transformed = [render_variant(image.convert('L'), masks, box) for _, box in plan]
    images = np.stack([x for x, _ in transformed])
    packed = np.packbits(np.stack([m for _, m in transformed]).reshape(13, 2, -1), axis=-1)
    if not np.isfinite(targets).all() or (targets < 0).any() or (targets > 1).any():
        raise ValueError('Invalid geometric target')
    return images, packed, targets


def member_bytes(archive, candidates, expected):
    for member in candidates:
        raw = archive.read(member)
        if hashlib.sha256(raw).hexdigest() == expected:
            return raw
    raise ValueError('No archive member matches the published SHA256')


def cache_rows(rows, out, workers, *, manifest_name='main'):
    out = fresh(out)
    write_json(out / 'protocol.json', dict(schema='oa-cxr-public-cache-v1',
        data_protocol=DATA_PROTOCOL, findings=DATA_PROTOCOL['findings'], variants=13,
        membership='published frozen audit membership; no new split selection',
        manifest_sha256=sha256_file(ROOT / 'reproducibility' / (manifest_name + '_sources.jsonl.gz'))))
    status = dict(status='running', files={}, splits={}, clinical_support_labels=False)
    write_json(out / 'status.json', status)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for split in sorted({r['split'] for r in rows}):
            selected = sorted((r for r in rows if r['split'] == split), key=lambda r: r['source_id'])
            folder = out / split; folder.mkdir(); n = len(selected)
            arrays = [np.lib.format.open_memmap(folder / name, mode='w+', dtype=dtype, shape=shape)
                      for name, dtype, shape in [('images.npy', np.uint8, (n,13,256,256)),
                          ('masks.npy', np.uint8, (n,13,2,8192)), ('targets.npy', np.float32, (n,13,3))]]
            for i, values in enumerate(pool.map(render, selected, chunksize=4)):
                for array, value in zip(arrays, values): array[i] = value
                if (i + 1) % 500 == 0: print(split, i + 1, '/', n, flush=True)
            for array in arrays: array.flush()
            del arrays
            with (folder / 'sources.jsonl').open('w') as stream:
                for i, row in enumerate(selected):
                    stream.write(json.dumps(dict(row, row_index=i), sort_keys=True) + '\n')
            status['splits'][split] = dict(sources=n, inputs=n*13, targets=n*39)
            for name in ('images.npy','masks.npy','targets.npy','sources.jsonl'):
                status['files'][split+'/'+name] = sha256_file(folder/name)
            write_json(out/'status.json', status)
    status['files']['protocol.json'] = sha256_file(out/'protocol.json')
    status['status'] = 'completed'; write_json(out/'status.json', status)
    write_json(out/'validation.json', dict(schema='oa-cxr-public-cache-validation-v1',
        status='passed', mode='automated_integrity', source_and_annotation_sha256_checked=True,
        cache_status_sha256=sha256_file(out/'status.json'),
        visual_or_clinical_review_claimed=False))
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--covidqu-archive', type=Path, required=True)
    p.add_argument('--kermany-archive', type=Path, required=True)
    p.add_argument('--kermany-masks', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--splits', nargs='+', choices=['fit','dev','calibration','test'], default=['fit','dev','calibration','test'])
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--smoke-per-dataset', type=int, help='Technical subset only; cannot satisfy formal training counts')
    args = p.parse_args()
    rows = [r for r in frozen_rows() if r['split'] in args.splits]
    if args.smoke_per_dataset:
        groups = {}
        for r in rows:
            key = (r['dataset'],r['split']); groups.setdefault(key,[])
            if len(groups[key]) < args.smoke_per_dataset: groups[key].append(r)
        rows = [r for group in groups.values() for r in group]
    out = fresh(args.output); raw_dir = out/'raw'; raw_dir.mkdir()
    with zipfile.ZipFile(args.covidqu_archive) as covid, zipfile.ZipFile(args.kermany_archive) as kermany:
        by_name = {}
        for name in kermany.namelist():
            if not name.startswith('__MACOSX/') and not Path(name).name.startswith('._'):
                by_name.setdefault(Path(name).name,[]).append(name)
        for row in rows:
            image = raw_dir/(row['source_id'] + Path(row['image_filename']).suffix)
            if row['dataset'] == 'covidqu_v7':
                raw = member_bytes(covid,[row['archive_member']],row['source_image_sha256'])
                mask = raw_dir/(row['source_id']+'_mask.png')
                mask.write_bytes(member_bytes(covid,[row['archive_member'].replace('/images/','/lung masks/')],row['mask_sha256']))
                row['mask_path'] = str(mask)
            else:
                raw = member_bytes(kermany,by_name.get(row['original_filename'],[]),row['source_image_sha256'])
                for side in ('left','right'):
                    mask = checked(args.kermany_masks/row['mask_filenames'][side],row['manual_mask_sha256'][side])
                    row[side+'_mask_path'] = str(mask.resolve())
            image.write_bytes(raw); row['source_image_path'] = str(image)
    print(json.dumps(cache_rows(rows,out/'cache',args.workers),indent=2))


if __name__ == '__main__': main()

"""Materialize identical PNG presentations for visual and report baselines."""
from common import *
import argparse
import hashlib
import numpy as np
from PIL import Image
from oa_cxr.io import write_jsonl
import export as original


def selected_report_sources(rows):
    selected=set()
    for dataset in sorted({r['dataset'] for r in rows}):
        pool=sorted((r for r in rows if r['dataset']==dataset),
            key=lambda r:hashlib.sha256(('report-rebuild:17:'+r['source_id']).encode()).hexdigest())
        selected.update((r['dataset'],r['source_id']) for r in pool[:128])
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--split',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cache=original.cache_inputs(a.cache,a.split,labels=False)
    if a.split!='test' and not a.split.startswith('external'):raise ValueError('Evaluation splits only')
    out=fresh(a.output);(out/'images').mkdir()
    images=np.load(cache['paths']['images.npy'],mmap_mode='r',allow_pickle=False)
    selected=selected_report_sources(cache['rows'])
    rows=[]
    for query in original.query_records(cache['rows'],a.split):
        if query['finding']!='pleural_effusion':continue
        pixels=images[query['source_row_index'],query['variant_index']]
        file=out/'images'/(query['input_id']+'.png');Image.fromarray(pixels).save(file)
        with Image.open(file) as saved:
            if not np.array_equal(np.asarray(saved),pixels):raise ValueError('PNG round-trip changed input')
        rows.append({**{k:query[k] for k in ['input_id','source_id','dataset','split','source_image_sha256']},
            'presented_image_path':str(file),'presented_image_sha256':sha256_file(file)})
    report=[r for r in rows if (r['dataset'],r['source_id']) in selected]
    write_jsonl(out/'inputs.jsonl',rows);write_jsonl(out/'report_inputs.jsonl',report)
    write_json(out/'status.json',dict(status='completed',labels_read=False,inputs=len(rows),
        report_inputs=len(report),report_sources=len(selected),selection_before_generation=True,
        selection='first 128 sources per dataset by SHA256(report-rebuild:17:source_id)',
        output_sha256={n:sha256_file(out/n) for n in ['inputs.jsonl','report_inputs.jsonl']}))


if __name__=='__main__':main()

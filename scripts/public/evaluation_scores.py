"""Convert the public OA evaluation artifact to the common baseline score format."""
from common import *
import argparse
import numpy as np
from oa_cxr.io import write_jsonl
import export as original


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--split',required=True)
    p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cache=original.cache_inputs(a.cache,a.split,labels=True)
    with np.load(a.evaluation/'predictions.npz',allow_pickle=False) as bundle:
        if bundle['source_id'].tolist()!=[r['source_id'] for r in cache['rows']] or bundle['dataset'].tolist()!=[r['dataset'] for r in cache['rows']]:
            raise ValueError('OA evaluation source ordering differs')
        truth=np.load(cache['paths']['targets.npy'],allow_pickle=False)
        if not np.array_equal(bundle['target'],truth):raise ValueError('OA evaluation target binding differs')
        values=bundle['prediction']
        if values.shape!=truth.shape or not np.isfinite(values).all():raise ValueError('Incomplete OA predictions')
        records=[dict(row,method='oa_cxr',score=float(score),status='success',failure_reason=None)
                 for row,score in zip(original.query_records(cache['rows'],a.split),values.reshape(-1))]
    out=fresh(a.output);write_jsonl(out/'predictions.jsonl',records)
    write_json(out/'status.json',dict(status='completed',method='oa_cxr',
        evaluation_sha256=sha256_file(a.evaluation/'predictions.npz'),
        output_sha256={'predictions.jsonl':sha256_file(out/'predictions.jsonl')}))


if __name__=='__main__':main()

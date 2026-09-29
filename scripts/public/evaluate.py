"""Evaluate the public OA model on a prepared split, preserving all queries."""
from common import *
import argparse
import numpy as np
from bundle import device_setup,load_bundle
from oa_cxr.rebuild.metrics import metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--split',choices=['dev','calibration','test','external_nih','external_shenzhen'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu-uuid')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--bootstrap',type=int,default=1000,help='Source/family bootstrap replicates for MAE and RMSE; 0 disables')
    a=p.parse_args()
    if a.batch_size<1 or a.bootstrap<0:p.error('Invalid batch size/bootstrap count')
    device=device_setup(a.gpu_uuid)
    import torch
    from oa_cxr.rebuild.cache_validation import validation_receipt
    torch.set_num_threads(4);validation_receipt(a.cache)
    status=load_json(a.cache/'status.json')
    for name in ('sources.jsonl','images.npy','targets.npy'):
        checked(a.cache/a.split/name,status['files'][a.split+'/'+name])
    rows=read_jsonl(a.cache/a.split/'sources.jsonl')
    images=np.load(a.cache/a.split/'images.npy',mmap_mode='r',allow_pickle=False)
    targets=np.load(a.cache/a.split/'targets.npy',allow_pickle=False)
    if images.shape!=(len(rows),13,256,256) or targets.shape!=(len(rows),13,3):raise ValueError('Invalid cache shapes')
    adapter,meta=load_bundle(a.weights,device);out=fresh(a.output)
    values=np.empty_like(targets);flat=images.reshape(-1,256,256);predictions=values.reshape(-1,3)
    with torch.inference_mode():
        for start in range(0,len(flat),a.batch_size):
            pixels=np.array(flat[start:start+a.batch_size],dtype=np.float32)[:,None]
            pixels=(pixels/255*2-1)*1024
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
                result=adapter(torch.from_numpy(pixels).to(device))['retention']
            predictions[start:start+len(pixels)]=result.cpu().numpy()
    if not np.isfinite(values).all():raise ValueError('Nonfinite predictions; no failed query is silently dropped')
    np.savez_compressed(out/'predictions.npz',prediction=values,target=targets,
        source_id=np.asarray([r['source_id'] for r in rows]),dataset=np.asarray([r['dataset'] for r in rows]),
        split_group_id=np.asarray([r['split_group_id'] for r in rows]))
    result=dict(schema='oa-cxr-public-evaluation-v1',split=a.split,source_images=len(rows),queries=values.size,
        model_config_sha256=sha256_file(a.weights/'config.json'),datasets={},clinical_support_labels=False,
        execution='CUDA BF16 image / FP32 readout' if device=='cuda' else 'CPU FP32',
        bootstrap_unit='source image except Kermany filename family; fixed model, not training-seed uncertainty')
    for dataset in sorted({r['dataset'] for r in rows}):
        idx=np.array([i for i,r in enumerate(rows) if r['dataset']==dataset]);y=targets[idx];s=values[idx]
        summary=metrics(y.ravel(),s.ravel(),regression=True)
        summary['source_images']=len(idx)
        if a.bootstrap:
            # Original image-ranking protocol uses source images for NIH/COVIDQU,
            # conservative filename families for Kermany, source images for Shenzhen.
            groups={}
            for i in idx:
                key=rows[i]['split_group_id'] if dataset=='kermany_v7' else rows[i]['source_id']
                groups.setdefault(key,[]).append(i)
            groups=[np.asarray(groups[k]) for k in sorted(groups)]
            abs_sum=np.array([np.abs(values[g].astype(float)-targets[g]).sum() for g in groups])
            sq_sum=np.array([np.square(values[g].astype(float)-targets[g]).sum() for g in groups])
            counts=np.array([len(g)*39 for g in groups]);rng=np.random.default_rng(17);mae=[];rmse=[]
            for _ in range(a.bootstrap):
                draw=rng.integers(0,len(groups),size=len(groups));den=counts[draw].sum()
                mae.append(abs_sum[draw].sum()/den);rmse.append(np.sqrt(sq_sum[draw].sum()/den))
            summary['bootstrap']=dict(replicates=a.bootstrap,seed=17,groups=len(groups),
                mae_ci95=np.quantile(mae,[.025,.975]).tolist(),rmse_ci95=np.quantile(rmse,[.025,.975]).tolist(),
                exact_historical_resample_stream_claimed=False)
        result['datasets'][dataset]=summary
    result['macro_mae']=float(np.mean([v['mae'] for v in result['datasets'].values()]))
    write_json(out/'metrics.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()

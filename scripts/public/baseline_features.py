"""Export label-free OA/frozen-encoder features, then attach targets separately."""
from common import *
import argparse
import numpy as np
from oa_cxr.io import stable_hash, write_jsonl
from bundle import device_setup, load_bundle
import export as original


def export_features(a):
    device = device_setup(a.gpu_uuid)
    import torch
    torch.set_num_threads(4)
    cache = original.cache_inputs(a.cache, a.split, labels=False)
    if device == 'cuda' and not torch.cuda.is_bf16_supported():
        raise ValueError('The reference GPU export requires BF16 support')
    batch = 64 if a.method == 'oa' else 128
    if a.method == 'oa':
        if a.weights is None: raise ValueError('--weights is required for OA features')
        adapter, metadata = load_bundle(a.weights, device)
        model = adapter.base
        names = original.feature_names(model)
        identity = metadata['files']['image_state.pt']['sha256']
    else:
        from baselines import foundation_encoders as enc
        if a.assets is None: raise ValueError('--assets is required for a frozen encoder')
        assets = enc.verify_assets(a.assets, sha256_file(a.assets), a.method)
        model = enc.load_encoder(assets).to(device)
        names = enc.feature_names(a.method, original.FINDINGS)
        identity = assets['sha256']
    protocol = dict(schema='oa-cxr-public-baseline-features-v1', method=a.method,
        weights_sha256=identity, feature_names=names, batch_size=batch,
        precision='cuda-bfloat16' if device == 'cuda' else 'cpu-float32',
        targets_or_masks_read=False, data_protocol=cache['protocol']['data_protocol'],
        implementation_sha256=sha256_file(__file__))
    protocol_sha = stable_hash(protocol)
    rows = list(original.query_records(cache['rows'], a.split))
    images = np.load(cache['paths']['images.npy'], mmap_mode='r', allow_pickle=False).reshape(-1,256,256)
    out = fresh(a.output)
    state = dict(status='running', action='public-baseline-features', method=a.method,
        sources=len(cache['rows']), inputs=len(images), queries=len(rows), target_values_loaded=False)
    write_json(out/'status.json', state)
    captured = []
    hook = None
    try:
        write_json(out/'feature_protocol.json', protocol)
        write_jsonl(out/'samples.jsonl', rows)
        X = np.lib.format.open_memmap(out/'feature_work.npy', mode='w+', dtype=np.float32,
                                      shape=(len(rows),len(names)))
        if a.method == 'oa':
            hook = model.retention_head[0].register_forward_pre_hook(
                lambda _module, values: captured.append(values[0].detach().float()))
        with torch.inference_mode():
            for start in range(0,len(images),batch):
                presented = images[start:start+batch]
                if a.method == 'oa':
                    x = torch.from_numpy((np.array(presented,dtype=np.float32)[:,None]/255*2-1)*1024)
                else:
                    x = torch.from_numpy(np.stack([enc.preprocess_image(image,a.method) for image in presented]))
                captured.clear()
                with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
                    if a.method == 'oa': model(x.to(device))
                    else: features = enc.encode_batch(model,x.to(device),a.method).cpu().numpy()
                if a.method == 'oa':
                    if len(captured)!=1: raise ValueError('Expected one actual pre-Linear feature tensor')
                    head = captured[0].cpu().numpy()
                    onehot = np.broadcast_to(np.eye(3,dtype=np.float32),(len(presented),3,3))
                    features = np.concatenate((head,onehot),axis=-1).reshape(-1,len(names))
                else: features = enc.query_features(features)
                if features.shape!=(len(presented)*3,len(names)) or not np.isfinite(features).all():
                    raise ValueError('Invalid encoder features; no failed query is silently dropped')
                X[start*3:(start+len(presented))*3] = features
        X.flush()
        np.savez(out/'features.npz',X=X,row_ids=np.asarray([r['row_id'] for r in rows],dtype='U64'),
                 available=np.ones(len(rows),dtype=bool))
        del X
        (out/'feature_work.npy').unlink()
        write_json(out/'manifest.json',original.make_manifest(out/'samples.jsonl',out/'features.npz',names,protocol_sha))
        state.update(status='completed', feature_protocol_sha256=protocol_sha,
            output_sha256={n:sha256_file(out/n) for n in ('features.npz','samples.jsonl','manifest.json','feature_protocol.json')})
    except BaseException as exc:
        state.update(status='failed',error=str(exc));raise
    finally:
        if hook is not None:hook.remove()
        write_json(out/'status.json',state)


def attach(a):
    from baselines.tabular import load_data
    source = Path(a.features).resolve()
    status = load_json(source/'status.json')
    if status.get('status')!='completed':raise ValueError('Completed feature export required')
    for name,digest in status['output_sha256'].items():checked(source/name,digest)
    data = load_data(source/'manifest.json',targets=False)
    cache = original.cache_inputs(a.cache,a.split,labels=True)
    expected = list(original.query_records(cache['rows'],a.split))
    if data['rows']!=expected:raise ValueError('Cache and feature query identities differ')
    y = np.load(cache['paths']['targets.npy'],allow_pickle=False).reshape(-1)
    if not np.isfinite(y).all() or (y<0).any() or (y>1).any():raise ValueError('Invalid independent target')
    out = fresh(a.output)
    write_jsonl(out/'samples.jsonl',expected)
    np.savez(out/'features.npz',X=data['X'],y=y,available=data['available'],
             row_ids=np.asarray([r['row_id'] for r in expected],dtype='U64'))
    write_json(out/'manifest.json',original.make_manifest(out/'samples.jsonl',out/'features.npz',
        data['manifest']['feature_names'],data['manifest']['feature_protocol_sha256']))
    write_json(out/'status.json',dict(status='completed',action='attach-independent-targets',
        source_status_sha256=sha256_file(source/'status.json'),
        output_sha256={n:sha256_file(out/n) for n in ('features.npz','samples.jsonl','manifest.json')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('export');p.add_argument('--method',choices=['oa','biomedclip','chexzero','dinov2'],required=True)
    p.add_argument('--weights',type=Path);p.add_argument('--assets',type=Path);p.add_argument('--gpu-uuid')
    p=sub.add_parser('attach-labels');p.add_argument('--features',type=Path,required=True)
    for p in sub.choices.values():
        p.add_argument('--cache',type=Path,required=True);p.add_argument('--split',required=True)
        p.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    (export_features if a.action=='export' else attach)(a)


if __name__=='__main__':main()

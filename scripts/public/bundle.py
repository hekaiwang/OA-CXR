"""Portable, tensor-only OA-CXR deployment. No historical run directories."""
from common import *
import os
import shutil


def device_setup(gpu_uuid=None):
    if gpu_uuid:
        import train
        train.check_gpu(gpu_uuid)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid or ''
    return 'cuda' if gpu_uuid else 'cpu'


def load_bundle(folder, device='cpu'):
    import torch
    from predict_image import model_from_state
    from oa_cxr.rebuild.neural_readout import NeuralReadout, NeuralReadoutImageAdapter
    folder = Path(folder).resolve()
    meta = load_json(folder/'config.json')
    if meta.get('schema') != 'oa-cxr-public-model-v1': raise ValueError('Unsupported model bundle')
    for name in ('image_state.pt','readout_state.pt'):
        checked(folder/name,meta['files'][name]['sha256'])
    for name,digest in meta['implementation_sha256'].items(): checked(ROOT/name,digest)
    base = model_from_state(torch.load(folder/'image_state.pt',map_location='cpu',weights_only=True))
    readout = NeuralReadout.from_config(meta['readout_config'])
    readout.load_state_dict(torch.load(folder/'readout_state.pt',map_location='cpu',weights_only=True),strict=True)
    if not bool(readout.standardizer.fitted): raise ValueError('Missing fit-only normalization')
    return NeuralReadoutImageAdapter(base,readout).eval().to(device),meta


def bundle(image_state, readout_dir, output, *, provenance):
    import torch
    output = fresh(output)
    selection = load_json(Path(readout_dir)/'selection.json')
    candidate = Path(readout_dir)/selection['candidate']
    metadata = load_json(candidate/'best_model.json')
    checked(candidate/'best_state.pt',metadata['state_sha256'])
    shutil.copyfile(image_state,output/'image_state.pt')
    shutil.copyfile(candidate/'best_state.pt',output/'readout_state.pt')
    meta = dict(schema='oa-cxr-public-model-v1',findings=['pleural_effusion','pneumothorax','consolidation'],
        readout_config=metadata['model_config'],selection=selection,provenance=provenance,
        preprocessing='grayscale; aspect-preserving bilinear letterbox256; (gray/255*2-1)*1024',
        clinical_support_model=False,files={},implementation_sha256={})
    for name in ('image_state.pt','readout_state.pt'):
        meta['files'][name]=dict(bytes=(output/name).stat().st_size,sha256=sha256_file(output/name))
    for name in ('src/oa_cxr/rebuild/vision.py','src/oa_cxr/rebuild/neural_readout.py',
                 'src/oa_cxr/rebuild/data.py','src/oa_cxr/anatomy_geometry.py','scripts/rebuild/predict_image.py'):
        meta['implementation_sha256'][name]=sha256_file(ROOT/name)
    write_json(output/'config.json',meta)
    load_bundle(output)
    return meta

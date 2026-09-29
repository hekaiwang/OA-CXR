"""Fetch the exact official XRV CheXpert initialization and convert to tensors."""
from common import *
import argparse
from collections import OrderedDict
from urllib.request import urlopen
import shutil

URL='https://github.com/mlmed/torchxrayvision/releases/download/v1/chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt'
SHA='1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,help='Optional already-downloaded official original; same SHA required')
    a=p.parse_args();out=fresh(a.output);original=out/'official.pt'
    if a.source:shutil.copyfile(checked(a.source,SHA),original)
    else:
        with urlopen(URL,timeout=120) as response,original.open('wb') as stream:shutil.copyfileobj(response,stream)
    checked(original,SHA)
    import torch
    import torchxrayvision as xrv
    if torch.__version__.split('+')[0]!='2.6.0':raise ValueError('Use Torch 2.6.0 for the pinned legacy conversion')
    # This historical object is unpickled ONLY after checking the fixed official SHA.
    value=torch.load(original,map_location='cpu',weights_only=False)
    if isinstance(value,torch.nn.Module):
        for module in value.modules():
            if not hasattr(module,'_non_persistent_buffers_set'):module._non_persistent_buffers_set=set()
        value=value.state_dict()
    if not isinstance(value,dict) or not value or any(not torch.is_tensor(v) for v in value.values()):
        raise ValueError('Expected tensor state')
    prefixed=[k.startswith('module.') for k in value]
    if any(prefixed) and not all(prefixed):raise ValueError('Mixed tensor names')
    state=OrderedDict((k[7:] if all(prefixed) else k,v.detach().cpu()) for k,v in value.items())
    xrv.models.DenseNet(weights=None,op_threshs=None,apply_sigmoid=False).load_state_dict(state,strict=True)
    target=out/'initial_state.pt';torch.save(state,target)
    loaded=torch.load(target,map_location='cpu',weights_only=True)
    if not all(torch.equal(v,loaded[k]) for k,v in state.items()):raise ValueError('Conversion changed tensors')
    receipt=dict(source_url=URL,source_sha256=SHA,tensor_sha256=sha256_file(target),tensor_values_verified=True)
    write_json(out/'conversion.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

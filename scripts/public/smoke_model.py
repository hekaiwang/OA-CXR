"""Synthetic CPU functionality check for a downloaded model, not an accuracy test."""
from common import *
import argparse
from bundle import device_setup,load_bundle


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--weights',type=Path,required=True);a=p.parse_args()
    device_setup()
    import torch
    torch.set_num_threads(2);adapter,meta=load_bundle(a.weights)
    generator=torch.Generator().manual_seed(17)
    pixels=(torch.rand(2,1,256,256,generator=generator)*2-1)*1024
    with torch.inference_mode():result=adapter(pixels)
    scores=result['retention']
    if scores.shape!=(2,3) or not torch.isfinite(scores).all() or (scores<0).any() or (scores>1).any():
        raise ValueError('Invalid synthetic inference output')
    print(json.dumps(dict(status='passed',mode='synthetic_cpu_functionality_only',shape=list(scores.shape),
        model_config_sha256=sha256_file(a.weights/'config.json'))))


if __name__=='__main__':main()

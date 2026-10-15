"""Predict from a completed baseline training run without reading targets."""
from common import *
import argparse
from baselines import tabular
import train_foundation_baseline as neural


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu-uuid')
    a=p.parse_args();state=load_json(a.training/'status.json')
    if state.get('method') in tabular.METHODS:
        if a.gpu_uuid:raise ValueError('Tabular prediction uses CPU')
        tabular.predict(a.training,a.manifest,None,a.output)
    else:
        from bundle import device_setup
        device=device_setup(a.gpu_uuid)
        neural.predict(a.training,sha256_file(a.training/'status.json'),a.manifest,
                       sha256_file(a.manifest),a.output,device=device)


if __name__=='__main__':main()

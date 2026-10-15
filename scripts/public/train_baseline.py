"""Train any of the eight regression baselines using explicit fit/dev manifests."""
from common import *
import argparse
import tempfile
from baselines import tabular
import train_foundation_baseline as neural


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=list(tabular.METHODS)+list(neural.METHODS),required=True)
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--dev',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu-uuid')
    a=p.parse_args()
    # Reject held-out rows before loading the feature/target arrays.
    manifest=load_json(a.fit);sample=Path(manifest['samples']['path'])
    if not sample.is_absolute():sample=a.fit.parent/sample
    checked(sample,manifest['samples']['sha256'])
    if any(r['split']!='fit' for r in read_jsonl(sample)):raise ValueError('Fit-only manifest required')
    if a.method in tabular.METHODS:
        if a.gpu_uuid:raise ValueError('These four fixed baselines train on CPU')
        tabular.fit(a.method,a.fit,None,a.output)
    else:
        if a.dev is None:raise ValueError('--dev is required for neural checkpoint selection')
        from bundle import device_setup
        device=device_setup(a.gpu_uuid)
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'config.json'
            write_json(config,dict(schema=neural.SCHEMA,method=a.method,
                sources={k:dict(path=str(v.resolve()),sha256=sha256_file(v))
                    for k,v in [('fit_manifest',a.fit),('dev_manifest',a.dev)]},
                expected_sources=dict(fit=19086,dev=2609)))
            neural.run(config,sha256_file(config),a.output,device=device)


if __name__=='__main__':main()

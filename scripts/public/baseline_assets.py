"""Fetch pinned upstream assets and create local configurations; never mirror vendor weights."""
from common import *
import argparse
import shutil
import subprocess
import urllib.request
import assets as legacy

FOUNDATION=('biomedclip','chexzero','dinov2')


def foundation(method,output):
    manifest=load_json(ROOT/'baselines'/method/'asset_manifest.json')
    out=fresh(output)
    upstream=manifest.get('implementation_upstream',manifest['upstream'])
    subprocess.run(['git','clone','--filter=blob:none','--no-checkout',upstream['url'],str(out/'upstream')],check=True)
    subprocess.run(['git','-C',str(out/'upstream'),'checkout','--detach',upstream['commit']],check=True)
    weights=out/'checkpoints/pretrained';weights.mkdir(parents=True)
    if method=='biomedclip':
        from huggingface_hub import hf_hub_download
        repository=manifest['upstream']['url'].split('huggingface.co/')[1]
        for name in ('open_clip_pytorch_model.bin','open_clip_config.json','README.md'):
            hf_hub_download(repository,name,revision=manifest['upstream']['revision'],local_dir=weights)
    elif method=='chexzero':
        import gdown
        result=gdown.download(id=manifest['checkpoint_file_id'],
            output=str(out/manifest['loader']['weights']),use_cookies=False,quiet=False)
        if result is None:raise RuntimeError('Official CheXzero checkpoint download failed')
    else:
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'
        with urllib.request.urlopen(url,timeout=120) as source,(out/manifest['loader']['weights']).open('xb') as dest:
            shutil.copyfileobj(source,dest)
    # Includes every pinned upstream Python file, configuration and weight.
    for record in manifest['files']:
        file=checked(out/record['path'],record['sha256'])
        if file.stat().st_size!=record['bytes']:raise ValueError('Asset length mismatch')
    write_json(out/'asset_manifest.json',manifest)
    from baselines.foundation_encoders import verify_assets
    verify_assets(out/'asset_manifest.json',sha256_file(out/'asset_manifest.json'),method)
    return out/'asset_manifest.json'


def standard(method,output):
    out=fresh(output)
    actual='maira_shared' if method=='maira_logprob' else method
    manifest,binding=legacy.bound_manifest(actual)
    legacy.download(manifest,binding,out/'download')
    staged=out/'download/candidate_manifest.json'
    if method=='chexpert':
        legacy.convert_chexpert(manifest,binding,out/'download/files'/legacy.LEGACY_FILE,out/'converted')
        staged=out/'converted/candidate_manifest.json'
    name='generation_config.json' if method=='maira_logprob' else 'config.json'
    config=load_json(ROOT/'baselines'/method/name)
    config['assets_manifest']=dict(path=str(staged.relative_to(out)),sha256=sha256_file(staged))
    if 'asset_root' in config:config['asset_root']=load_json(staged)['asset_root']
    if method=='scuq_radgraph':
        source,binding=legacy.bound_manifest('scuq_source')
        legacy.download(source,binding,out/'source')
        source_path=out/'source/candidate_manifest.json'
        config['source_manifest']=dict(path=str(source_path.relative_to(out)),sha256=sha256_file(source_path))
    write_json(out/name,config)
    return out/name


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=list(FOUNDATION)+['chexpert','autoqc','biovil_t','maira_logprob','scuq_radgraph'],required=True)
    p.add_argument('--output',type=Path,required=True,help='Fresh directory under this clone: asset_staging/NAME')
    a=p.parse_args();a.output=a.output.resolve()
    if not a.output.is_relative_to(ROOT/'asset_staging'):
        raise ValueError('Use a fresh output under this clone\'s asset_staging directory')
    config=(foundation if a.method in FOUNDATION else standard)(a.method,a.output)
    print(json.dumps(dict(status='verified',method=a.method,configuration=str(config),sha256=sha256_file(config))))


if __name__=='__main__':main()

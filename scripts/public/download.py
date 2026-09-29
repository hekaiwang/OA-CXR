"""Download immutable public model/annotation artifacts; verify every SHA256."""
from common import *
import argparse
import shutil
import tarfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('weights'))
    p.add_argument('--with-training-masks',action='store_true')
    a=p.parse_args();out=fresh(a.output)
    from huggingface_hub import hf_hub_download
    spec=load_json(ROOT/'configs/public_assets.json')
    names=[n for n in spec['files'] if n.startswith('main/')]
    if a.with_training_masks:names.append('reproduction/kermany-v7-masks.tar.gz')
    for name in names:
        downloaded=hf_hub_download(spec['repo_id'],name,revision=spec['revision'],token=False)
        checked(downloaded,spec['files'][name]['sha256'])
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(downloaded,target)
        checked(target,spec['files'][name]['sha256'])
        print('Verified',name,flush=True)
    if a.with_training_masks:
        with tarfile.open(out/'reproduction/kermany-v7-masks.tar.gz') as archive:
            for member in archive:
                path=Path(member.name)
                if (not member.isfile() or path.is_absolute() or len(path.parts)!=2
                        or path.parts[0]!='kermany_masks' or '..' in path.parts or path.suffix!='.png'):
                    raise ValueError('Unexpected annotation archive member')
                target=out/path;target.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as source,target.open('xb') as destination:
                    shutil.copyfileobj(source,destination)
    write_json(out/'download_receipt.json',dict(repo_id=spec['repo_id'],revision=spec['revision'],verified_files=names))


if __name__=='__main__':main()

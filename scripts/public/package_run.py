"""Package completed image/readout training without repeating optimization."""
from common import *
import argparse
from bundle import bundle


def package_run(run_dir,output):
    import torch
    import export
    run_dir=Path(run_dir).resolve()
    selected=export.selected_checkpoint(run_dir/'image')
    readout=run_dir/'readout';status=load_json(readout/'status.json')
    if status.get('status')!='completed':raise ValueError('Readout fitting is incomplete')
    for name,digest in status['output_sha256'].items():
        path=(readout/name).resolve()
        if not path.is_relative_to(readout):raise ValueError('Output path outside training directory')
        checked(path,digest)
    protocol=load_json(readout/'protocol.json')
    if protocol['original_checkpoint_sha256']!=selected['sha256']:
        raise ValueError('Readout and image training do not match')
    payload=torch.load(selected['path'],map_location='cpu',weights_only=True)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        state=Path(tmp)/'image_state.pt';torch.save(payload['model_state'],state)
        return bundle(state,readout,output,provenance=dict(seed=17,
            image_checkpoint_sha256=selected['sha256'],image_epoch=selected['epoch'],
            image_training_status_sha256=sha256_file(run_dir/'image/status.json'),
            readout_status_sha256=sha256_file(readout/'status.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();package_run(a.run,a.output)

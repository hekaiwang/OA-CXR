"""Run the paper's 13-epoch image stage and three 30-epoch readout candidates.

Only fit/dev enter this pipeline. Each stage writes fresh outputs and SHA receipts.
Use a BF16-capable GPU with approximately 24 GiB free memory at batch 256.
"""
from common import *
import argparse
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();a.cache=a.cache.resolve();a.initial=a.initial.resolve();out=fresh(a.output)
    state={'status':'running','completed_stages':[],'heldout_data_access':False}
    def run(stage,script,args):
        command=[sys.executable,str(ROOT/'scripts/rebuild'/script),*map(str,args)]
        write_json(out/(stage+'_command.json'),dict(argv=command))
        with (out/(stage+'.log')).open('w') as log:
            print('Running',stage,flush=True)
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        state['completed_stages'].append(stage);write_json(out/'status.json',state)
    try:
        initial_sha=sha256_file(a.initial)
        common=['--data',a.cache,'--checkpoint',a.initial,'--checkpoint-sha256',initial_sha,
                '--gpu-uuid',a.gpu_uuid,'--batch-size',256,'--workers',a.workers]
        run('image_smoke','train.py',['--action','smoke',*common,'--output',out/'smoke'])
        run('image_train','train.py',['--action','train',*common,'--smoke-dir',out/'smoke','--output',out/'image'])
        for split in ('fit','dev'):
            run('export_'+split,'export.py',['--action','export','--data',a.cache,'--split',split,
                '--train-dir',out/'image','--gpu-uuid',a.gpu_uuid,'--batch-size',64,
                '--workers',a.workers,'--output',out/'export'/split])
            run('labels_'+split,'export.py',['--action','attach-labels','--data',a.cache,'--split',split,
                '--export-dir',out/'export'/split,'--output',out/'labeled'/split])
        best=load_json(out/'image/best.json')
        paths=dict(fit_manifest=out/'labeled/fit/manifest.json',dev_manifest=out/'labeled/dev/manifest.json',
            selected_checkpoint=out/'image'/best['checkpoint'],original_dev_cohort=out/'image/dev_cohort.json',
            dev_predictions=out/'export/dev/predictions.jsonl',fit_export_status=out/'export/fit/status.json',
            dev_export_status=out/'export/dev/status.json')
        config=out/'readout_config.json'
        write_json(config,dict(schema='oa-cxr-neural-readout-training-v1',experiment='public-reproduction',
            sources={k:dict(path=str(v),sha256=sha256_file(v)) for k,v in paths.items()}))
        run('readout_train','train_neural_readout.py',['--config',config,'--config-sha256',sha256_file(config),'--output',out/'readout'])
        from package_run import package_run
        package_run(out,out/'deploy')
        state['status']='completed'
    except BaseException as exc:
        state.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:write_json(out/'status.json',state)


if __name__=='__main__':main()

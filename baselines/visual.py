"""Label-free, complete-denominator execution shared by visual baselines.

Only standard-library imports here: AutoQC can run in its isolated CPU env.
No historical experiment outputs or source labels are consumed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parent
METHODS = ('chexpert','autoqc','biovil_t')
FINDINGS = ('pleural_effusion','pneumothorax','consolidation')
FIELDS = {'input_id','source_id','dataset','split','source_image_sha256','presented_image_path','presented_image_sha256'}
SCHEMA = 'oa-cxr-visual-baseline-run-v1'


def now():return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def write(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)


def require(condition,message):
    if not condition:raise ValueError(message)


def verified_assets(method,config_path=None,project_root=PROJECT):
    require(method in METHODS,'Unknown visual baseline')
    cp=Path(config_path or ROOT/method/'config.json').resolve();config=read(cp)
    require(config.get('method')==method and config.get('schema')=='oa-cxr-baseline-config-v1'
            and config.get('full_original_paper_reproduction') is False,'Unexpected baseline configuration')
    spec=config.get('assets_manifest',{});mp=(cp.parent/spec.get('path','')).resolve()
    require(mp.is_file() and spec.get('sha256')==sha(mp),'Asset manifest does not match config SHA')
    assets=read(mp)
    require(assets.get('schema')=='oa-cxr-baseline-assets-v1' and assets.get('method')==method
            and assets.get('legacy_results_included') is False,'Only official asset metadata may be consumed')
    project_root=Path(project_root).resolve();folder=(project_root/config['asset_root']).resolve()
    require(folder.is_relative_to(project_root) and str(config['asset_root'])==assets['asset_root'], 'Asset root changed or leaves project')
    bindings={str(cp):sha(cp),str(mp):sha(mp)}
    require(isinstance(assets.get('files'),dict) and assets['files'],'Empty official asset roster')
    for name,item in assets['files'].items():
        path=(folder/name).resolve()
        require(path.is_relative_to(folder) and path.is_file() and path.stat().st_size==item['bytes']
                and sha(path)==item['sha256'],'Official asset byte mismatch: '+name)
        bindings[str(path)]=item['sha256']
    return config,assets,folder,bindings


def inputs(path,expected_inputs):
    path=Path(path).resolve();rows=[];ids=set();sources={};images={}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():continue
        row=json.loads(line)
        require(isinstance(row,dict) and set(row)==FIELDS,'Canonical label-free input fields required; labels/extra columns prohibited')
        require(all(isinstance(v,str) and v for v in row.values()),'Input identity fields must be nonempty strings')
        require(row['split'] in ('fit','dev','calibration','test','external') and row['input_id'] not in ids,'Invalid split or duplicate input')
        for k in ('source_image_sha256','presented_image_sha256'):
            require(re.fullmatch('[0-9a-f]{64}',row[k]) is not None,'Invalid source/presented image SHA')
        for table,key in ((sources,(row['dataset'],row['source_id'])),(images,row['source_image_sha256'])):
            require(key not in table or table[key]==row['split'],'Source image crosses splits');table[key]=row['split']
        ids.add(row['input_id']);p=Path(row['presented_image_path'])
        row['presented_image_path']=str((path.parent/p).resolve() if not p.is_absolute() else p.resolve())
        rows.append(row)
    require(type(expected_inputs) is int and expected_inputs>0 and len(rows)==expected_inputs,'Full expected input population differs')
    return rows


def select_gpu(uuid):
    require(isinstance(uuid,str) and re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',uuid) is not None,
            'Exactly one full GPU UUID is required')
    raw=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    fields=[x.strip() for x in raw.split(',')]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    require(len(fields)==4 and fields[0]==uuid,'GPU identity differs')
    if int(fields[2])>256 or int(fields[3])>5 or any(line.split(',')[0].strip()==uuid for line in apps.splitlines()):
        raise RuntimeError('Selected GPU is occupied; no task will be stopped')
    os.environ['CUDA_VISIBLE_DEVICES']=uuid
    return dict(gpu_uuid=uuid,snapshot=raw)


def runtime_device(torch,device,uuid):
    if device=='cpu':return dict(device='cpu',gpu_uuid=None)
    require(device=='cuda:0' and torch.cuda.is_available() and torch.cuda.device_count()==1,'Single masked CUDA device required')
    props=torch.cuda.get_device_properties(0);actual=getattr(props,'uuid',None)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==uuid and (actual is None or str(actual).lower().removeprefix('gpu-')==uuid.lower().removeprefix('gpu-')),
            'CUDA runtime UUID disagrees with selected GPU')
    return dict(device=device,gpu_uuid=uuid,gpu_name=props.name,torch=str(torch.__version__))


def backend(method,folder,config,device,gpu_uuid,workers):
    module=importlib.import_module('baselines.'+method+'.run')
    return module.Backend(folder,config,device=device,gpu_uuid=gpu_uuid,workers=workers)


def failure(row,method,reason,*,attempted):
    return [{**row,'finding':f,'method':method,'status':'failed','score':None,
             'failure_reason':reason,'attempted':attempted} for f in FINDINGS]


def successful(row,method,values):
    require(isinstance(values,dict) and set(values)==set(FINDINGS),'Exactly three finding scores required')
    result=[]
    for finding in FINDINGS:
        item=values[finding]
        json.dumps(item,allow_nan=False)
        require(isinstance(item,dict) and type(item.get('score')) in (float,int)
                and math.isfinite(item['score']),'Finite nonboolean score required')
        require(not set(item)&(FIELDS|{'finding','method','status','attempted','failure_reason'}),'Scorer attempted to overwrite identity')
        if method=='chexpert':require(0<=item['score']<=1,'CheXpert absence score out of range')
        if method=='autoqc':require(item['score']<=0,'AutoQC distance direction differs')
        result.append({**row,**item,'finding':finding,'method':method,'status':'success','failure_reason':None,'attempted':True})
    return result


def score(method,inputs_path,output,*,expected_inputs,config_path=None,project_root=PROJECT,
          device='cuda:0',gpu_uuid=None,batch_size=16,workers=4):
    allowed_batches=(16,32) if method=='chexpert' else (16,)
    require(type(batch_size) is int and batch_size in allowed_batches,
            'CheXpert permits batch16/32; AutoQC and BioViL-T remain batch16')
    require(type(workers) is int and 1<=workers<=16,'Use 1..16 CPU workers')
    require(device in ('cpu','cuda:0'),'Only CPU or single masked cuda:0 supported')
    if method=='autoqc':require(device=='cpu' and gpu_uuid is None,'AutoQC is CPU only')
    out=Path(output).resolve();out.parent.mkdir(parents=True,exist_ok=True);lock=Path(str(out)+'.lock')
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(fd,str(os.getpid()).encode());os.close(fd);out.mkdir(exist_ok=False)
        state=dict(schema=SCHEMA,status='validating',method=method,started_at=now(),expected_inputs=expected_inputs,
                   batch_size=batch_size,
                   attempted_inputs=0,scored_inputs=0,failed_inputs=0,unattempted_inputs=0,
                   labels_loaded=False,source_images_read=False,clinical_effectiveness_evaluated=False)
        write(out/'status.json',state);engine=None;roster=[];bindings={};processed=0;emitted=0
        try:
            manifest=Path(inputs_path).resolve();roster=inputs(manifest,expected_inputs)
            bindings[str(manifest)]=sha(manifest)
            config,assets,folder,asset_bindings=verified_assets(method,config_path,project_root)
            bindings.update(asset_bindings)
            bindings[str(Path(__file__).resolve())]=sha(__file__)
            run_file=ROOT/method/'run.py';bindings[str(run_file)]=sha(run_file)
            execution=select_gpu(gpu_uuid) if device=='cuda:0' else {'device':'cpu','gpu_uuid':None}
            if device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
            engine=backend(method,folder,config,device,gpu_uuid,workers)
            protocol=dict(schema='oa-cxr-visual-baseline-protocol-v1',method=method,config=config,assets=assets,
                execution={**execution,**getattr(engine,'execution',{})},backend=getattr(engine,'protocol',{}),
                batch_size=batch_size,workers=workers,source_file_sha256=bindings,labels_loaded=False,
                all_original_inputs_preserved=True,full_original_paper_reproduction=False)
            write(out/'protocol.json',protocol);state.update(status='running',input_count=len(roster),execution=protocol['execution'])
            write(out/'status.json',state)
            with (out/'scores.jsonl').open('x',encoding='utf-8') as stream:
                def emit(records):
                    nonlocal emitted
                    payload=''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in records)
                    stream.write(payload)
                    stream.flush()
                    emitted+=1
                for start in range(0,len(roster),batch_size):
                    batch=roster[start:start+batch_size];prepared=[];valid=[];answers={}
                    for row in batch:
                        try:
                            require(sha(row['presented_image_path'])==row['presented_image_sha256'],'Presented image hash mismatch')
                            prepared.append(engine.prepare(row));valid.append(row)
                        except Exception as exc:
                            answers[row['input_id']]=failure(row,method,f'input_preparation: {type(exc).__name__}: {exc}',attempted=True)
                    try:
                        inferred=engine.infer(prepared) if prepared else []
                        require(len(inferred)==len(valid),'Backend changed batch population')
                        for row,value in zip(valid,inferred):
                            try:
                                require(sha(row['presented_image_path'])==row['presented_image_sha256'],'Presented image changed during inference')
                                if isinstance(value,dict) and '_failure' in value:raise ValueError(value['_failure'])
                                answers[row['input_id']]=successful(row,method,value)
                            except Exception as exc:
                                answers[row['input_id']]=failure(row,method,f'input_scoring: {type(exc).__name__}: {exc}',attempted=True)
                    except BaseException as exc:
                        for row in batch:
                            answers[row['input_id']]=failure(row,method,f'batch_inference: {type(exc).__name__}: {exc}',attempted=True)
                        for row in batch:emit(answers[row['input_id']])
                        state['failed_inputs']+=len(batch);state['attempted_inputs']+=len(batch);processed+=len(batch)
                        for row in roster[processed:]:emit(failure(row,method,'not_attempted_after_batch_failure',attempted=False))
                        state['unattempted_inputs']=len(roster)-processed
                        raise
                    for row in batch:
                        result=answers[row['input_id']];emit(result)
                        state['scored_inputs' if result[0]['status']=='success' else 'failed_inputs']+=1
                    state['attempted_inputs']+=len(batch);processed+=len(batch)
                    if hasattr(engine,'preview') and not (out/'technical_preview').exists() and prepared:
                        engine.preview(prepared[0],valid[0],out/'technical_preview')
                    write(out/'status.json',state)
            require(all(sha(p)==h for p,h in bindings.items()),'Bound input/assets changed during scoring')
            state.update(status='completed' if not state['failed_inputs'] else 'completed_with_failures',
                         scored_queries=3*state['scored_inputs'],failed_queries=3*state['failed_inputs'],
                         output_sha256={'scores.jsonl':sha(out/'scores.jsonl'),'protocol.json':sha(out/'protocol.json')})
            if not state['scored_inputs']:state['status']='failed'
        except BaseException as exc:
            state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            if roster and emitted<len(roster):
                with (out/'scores.jsonl').open('a',encoding='utf-8') as stream:
                    for row in roster[emitted:]:
                        for record in failure(row,method,'not_attempted_after_run_failure',attempted=False):
                            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                state['unattempted_inputs']+=len(roster)-emitted
                emitted=len(roster)
            if (out/'scores.jsonl').exists():state['output_sha256']={'scores.jsonl':sha(out/'scores.jsonl')}
        finally:
            if engine is not None and hasattr(engine,'close'):
                try:engine.close()
                except BaseException as exc:state.update(status='failed',close_error=f'{type(exc).__name__}: {exc}')
            state.update(finished_at=now(),input_count=len(roster),source_file_sha256=bindings)
            state.update(scored_queries=3*state['scored_inputs'],failed_queries=3*state['failed_inputs'],
                         unattempted_queries=3*state['unattempted_inputs'],output_inputs=emitted)
            if roster and (state['scored_inputs']+state['failed_inputs']+state['unattempted_inputs']!=len(roster) or emitted!=len(roster)):
                state.update(status='failed',accounting_error='Incomplete output partition')
            write(out/'status.json',state)
        return state
    finally:lock.unlink(missing_ok=True)


def cli(method):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--expected-inputs',type=int);parser.add_argument('--config',type=Path,default=ROOT/method/'config.json')
    parser.add_argument('--project-root',type=Path,default=PROJECT);parser.add_argument('--verify-assets-only',action='store_true')
    parser.add_argument('--gpu-uuid');parser.add_argument('--device',choices=['cpu','cuda:0'],default='cpu' if method=='autoqc' else 'cuda:0')
    parser.add_argument('--batch-size',type=int,default=16);parser.add_argument('--workers',type=int,default=4)
    a=parser.parse_args()
    if a.verify_assets_only:
        _,assets,_,_=verified_assets(method,a.config,a.project_root)
        print(json.dumps({'status':'verified','method':method,'asset_files':len(assets['files'])}));return 0
    if a.inputs is None or a.output is None or a.expected_inputs is None:parser.error('--inputs, --output and --expected-inputs are required')
    value=score(method,a.inputs,a.output,expected_inputs=a.expected_inputs,config_path=a.config,project_root=a.project_root,
                device=a.device,gpu_uuid=a.gpu_uuid,batch_size=a.batch_size,workers=a.workers)
    print(json.dumps(value,ensure_ascii=False));return {'completed':0,'completed_with_failures':2}.get(value['status'],1)

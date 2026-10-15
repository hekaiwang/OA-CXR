"""New canonical-input MAIRA generation, token uncertainty and fixed sampling.

Uses the audited generic processor implementations, never historical runs or
version-specific experiment controllers. GPU execution is explicit and offline.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from baselines.visual import PROJECT,ROOT,FIELDS,inputs,read,write,sha,digest,require,now,select_gpu,runtime_device
from baselines.asset_verification import verify_manifest,verify_file

sys.path.insert(0,str(PROJECT/'src'))
from oa_cxr.claims import extract_claims
from oa_cxr.generation import MairaBackend,validate_config
from oa_cxr.batched_generation import BatchedMairaBackend,batch_config,verify_batch_artifacts
from oa_cxr.report_likelihood import MairaReportLikelihoodBackend

SCHEMA='oa-cxr-new-report-run-v1'
METHODS=('maira_logprob','sentence_logprob','token_entropy','scuq_radgraph')
_ASSET_VERIFICATIONS={}


def write_jsonl(path,rows):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8');os.replace(temp,path)


def generic_row(row):
    """Map only label-free current PNG identity into the generic processor API."""
    require(set(row)==FIELDS,'Strict canonical seven-field input required')
    return dict(input_id=row['input_id'],subject_id=digest([row['dataset'],row['source_id']]),study_id=row['input_id'],
        dataset=row['dataset'],partition=row['split'],local_image_paths={'frontal':row['presented_image_path']},
        image_sha256={'frontal':row['presented_image_sha256']},allowed_input_metadata={},
        preprocessing_version='new-canonical-presented-png-v1')


def code_bindings():
    result={str(Path(__file__).resolve()):sha(__file__),str(ROOT/'visual.py'):sha(ROOT/'visual.py'),
            str(ROOT/'asset_verification.py'):sha(ROOT/'asset_verification.py')}
    for name in ('claims','generation','batched_generation','report_likelihood','io'):
        p=Path(importlib.import_module('oa_cxr.'+name).__file__).resolve();result[str(p)]=sha(p)
    return result


def verify(bindings):
    for path,value in bindings.items():
        cached=_ASSET_VERIFICATIONS.get(path)
        if cached:
            _ASSET_VERIFICATIONS[path]=verify_file(path,value,cached['fingerprint']['size'],cached,
                force_full=os.environ.get('OA_CXR_FORCE_FULL_ASSET_SHA')=='1')
        else:require(sha(path)==value,'Bound file changed: '+str(path))


def verification_summary():
    return dict(policy='full SHA256 on every asset verification',
        files=copy.deepcopy(_ASSET_VERIFICATIONS),fresh_full_sha_for_every_stage=True)


def asset_bindings(manifest_path,project_root=PROJECT):
    manifest_path=Path(manifest_path).resolve();value=read(manifest_path);project_root=Path(project_root).resolve()
    folder=(project_root/value['asset_root']).resolve()
    require(folder.is_relative_to(project_root) and value.get('legacy_results_included') is False,'Only project-local official assets allowed')
    result={str(manifest_path):sha(manifest_path)}
    require(value.get('files'),'Empty asset manifest')
    receipt=verify_manifest(folder,value,manifest_path.parent/'verification_receipts'/(sha(manifest_path)+'.json'),sha(manifest_path),
                            force_full=os.environ.get('OA_CXR_FORCE_FULL_ASSET_SHA')=='1')
    for item in receipt['files'].values():
        result[item['path']]=item['sha256'];_ASSET_VERIFICATIONS[item['path']]=item
    return value,folder,result


def maira_config(path=ROOT/'maira_logprob/generation_config.json',project_root=PROJECT):
    path=Path(path).resolve();c=read(path);spec=c['assets_manifest'];manifest=(path.parent/spec['path']).resolve()
    require(sha(manifest)==spec['sha256'],'MAIRA asset manifest changed')
    assets,folder,bindings=asset_bindings(manifest,project_root)
    require(assets['revision']==c['model_revision']==c['processor_revision'],'MAIRA immutable revision differs')
    require(c['dtype']=='bfloat16' and c['device']=='cuda:0' and c['local_files_only'] is True
            and c['batch_size']==16 and c['seed']==17,'Fixed MAIRA execution protocol changed')
    require(c['generation']==dict(do_sample=False,max_new_tokens=512,use_cache=True,num_beams=1),'Greedy max512 protocol changed')
    config={k:copy.deepcopy(c[k]) for k in ('backend','model_revision','processor_revision','device','dtype','seed','local_files_only','generation')}
    config['model_source']=str(folder);validate_config(config);bindings[str(path)]=sha(path);bindings.update(code_bindings())
    return config,bindings


@contextmanager
def fresh_run(output,stage):
    out=Path(output).resolve();out.parent.mkdir(parents=True,exist_ok=True);lock=Path(str(out)+'.lock')
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(fd,str(os.getpid()).encode());os.close(fd);out.mkdir(exist_ok=False)
        state=dict(schema=SCHEMA,stage=stage,status='validating',started_at=now(),labels_read=False,clinical_effectiveness_evaluated=False)
        write(out/'status.json',state)
        try:yield out,state
        except BaseException as exc:
            state.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
        finally:state['finished_at']=now();write(out/'status.json',state)
    finally:lock.unlink(missing_ok=True)


def report_claims(row):
    if row['status']!='success':return []
    return [c for c in extract_claims(row['generated_findings']) if c['eligible']]


def denominator(rows):
    return [dict(input=r['input'],generation_status=r.get('generation_status',r['status']),
                 status=r['status'],eligible_claims=len(report_claims({**r,'status':r.get('generation_status',r['status'])})),possibly_truncated=r.get('possibly_truncated'),
                 attempted=r.get('attempted',False),error=r.get('error')) for r in rows]


def finish(out,state,rows,filename,bindings):
    counts=dict(Counter(r['status'] for r in rows))
    write_jsonl(out/filename,rows);write_jsonl(out/'denominators.jsonl',denominator(rows))
    state.update(status='completed' if all(r['status'] in ('success','not_applicable') for r in rows) else 'completed_with_failures',
                 input_count=len(rows),counts=counts,eligible_claims=sum(d['eligible_claims'] for d in denominator(rows)),
                 source_file_sha256=bindings,output_sha256={p:sha(out/p) for p in (filename,'denominators.jsonl','protocol.json')})
    verify(bindings)
    state['asset_verification']=verification_summary()


def checked_run(folder,filename,stage):
    folder=Path(folder).resolve();state=read(folder/'status.json')
    require(state.get('schema')==SCHEMA and state.get('stage')==stage and state.get('status') in ('completed','completed_with_failures'),
            'Require finished new report stage; historical results are unsupported')
    for name,h in state['output_sha256'].items():require(sha(folder/name)==h,'Report bundle changed')
    rows=[json.loads(line) for line in (folder/filename).read_text(encoding='utf-8').splitlines() if line.strip()]
    require(len(rows)==state['input_count'] and len({r['input']['input_id'] for r in rows})==len(rows),'Report population differs')
    protocol=read(folder/'protocol.json')
    return rows,protocol,{str(folder/name):sha(folder/name) for name in (filename,'status.json','protocol.json','denominators.jsonl')}


def generate(manifest,output,expected_inputs,gpu_uuid,config_path=ROOT/'maira_logprob/generation_config.json',project_root=PROJECT):
    roster=inputs(manifest,expected_inputs)
    rows=[dict(input=r,status='unattempted',attempted=False,generated_findings=None,possibly_truncated=None,error=None) for r in roster]
    with fresh_run(output,'generation') as (out,state):
        try:
            config,bindings=maira_config(config_path,project_root);bindings[str(Path(manifest).resolve())]=sha(manifest)
            config=batch_config(config,16);execution=select_gpu(gpu_uuid);backend=BatchedMairaBackend(config)
            execution.update(runtime_device(backend.torch,'cuda:0',gpu_uuid))
            write(out/'protocol.json',dict(schema=SCHEMA,stage='generation',config=config,execution=execution,
                canonical_inputs_sha256=sha(manifest),source_file_sha256=bindings,batch_size=16,labels_read=False,
                asset_verification=verification_summary(),
                selection='externally frozen full canonical manifest; no output-dependent resampling'))
            for start in range(0,len(rows),16):
                group=rows[start:start+16];valid=[]
                for r in group:
                    r['attempted']=True
                    try:
                        require(sha(r['input']['presented_image_path'])==r['input']['presented_image_sha256'],'presented image SHA mismatch')
                        valid.append(r)
                    except Exception as exc:r.update(status='failed',error=f'input: {type(exc).__name__}: {exc}')
                if valid:
                    try:
                        values=backend.generate_batch([generic_row(r['input']) for r in valid],
                                [out/'audit'/digest(r['input']['input_id']) for r in valid])
                        require(len(values)==len(valid),'Generation batch count changed')
                        for r,value in zip(valid,values):
                            if isinstance(value,Exception):r.update(status='failed',error=str(value));continue
                            r.update(value,backend='maira',status='failed' if value.get('error_status') else ('truncated' if value['possibly_truncated'] else 'success'))
                            require(sha(r['input']['presented_image_path'])==r['input']['presented_image_sha256'],'Image changed during generation')
                            if r['status']=='success':verify_batch_artifacts(r)
                    except BaseException as exc:
                        for r in valid:r.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                        raise
                write_jsonl(out/'reports.jsonl',rows);state.update(status='running',processed_inputs=min(start+16,len(rows)))
                write(out/'status.json',state)
            finish(out,state,rows,'reports.jsonl',bindings)
        finally:
            write_jsonl(out/'reports.jsonl',rows);write_jsonl(out/'denominators.jsonl',denominator(rows))


def token_positions(text,offsets,start,end):
    require(type(start) is int and type(end) is int and 0<=start<end<=len(text),'Invalid sentence span')
    covered=np.zeros(end-start,dtype=bool);positions=[]
    for i,(a,b) in enumerate(offsets):
        a,b=int(a),int(b);require(0<=a<=b<=len(text),'Invalid token offset')
        if b>start and a<end:positions.append(i);covered[max(a,start)-start:min(b,end)-start]=True
    require(positions and all(covered[i] or c.isspace() for i,c in enumerate(text[start:end])),'Sentence not fully covered by token offsets')
    return positions


def claim_token_scores(text,ids,offsets,lp,ent):
    lp,ent=np.asarray(lp),np.asarray(ent)
    require(len(ids)==len(offsets)==len(lp)==len(ent) and len(ids)>0,'Token alignment differs')
    require(np.isfinite(lp).all() and np.isfinite(ent).all() and np.all(lp<=1e-6) and np.all(ent>=-1e-6),'Invalid token distribution')
    rows=[]
    for claim in extract_claims(text):
        if not claim['eligible']:continue
        pos=token_positions(text,offsets,claim['sentence_start'],claim['sentence_end'])
        rows.append({**claim,'sentence_mean_log_probability':float(lp[pos].mean()),
                     'sentence_negative_mean_token_entropy':-float(ent[pos].mean()),'scored_token_count':len(pos)})
    return rows


class TokenBackend(MairaReportLikelihoodBackend):
    def score_tokens(self,reports,dirs):
        captured=[]
        def hook(module,args,output):
            require(not captured,'Unexpected multiple teacher-forcing forwards');torch=self.torch
            for i,directory in enumerate(dirs):
                with np.load(directory/'teacher_forcing_tokens.npz',allow_pickle=False) as z:labels=z['labels']
                positions=np.flatnonzero(labels!=-100);ids=labels[positions];lp=[];ent=[]
                for start in range(0,len(ids),64):
                    at=torch.as_tensor(positions[start:start+64]-1,device=self.device);target=torch.as_tensor(ids[start:start+64],device=self.device)
                    ld=torch.log_softmax(output.logits[i,at].float(),dim=-1)
                    lp.extend(ld.gather(1,target[:,None]).squeeze(1).cpu().tolist());ent.extend((-(ld.exp()*ld).sum(-1)).cpu().tolist())
                captured.append((ids.tolist(),lp,ent))
        handle=self.model.register_forward_hook(hook)
        try:results=self.score_batch(reports,dirs)
        finally:handle.remove()
        require(len(captured)==len(reports),'Missing teacher-forced token statistics')
        for item,directory,result,(ids,lp,ent) in zip(reports,dirs,results,captured):
            text=item['cache']['generated_findings'];encoded=self.processor.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
            require(encoded['input_ids']==ids and np.isclose(np.mean(lp),result['score'],atol=1e-6,rtol=1e-6),'Canonical scored token sequence differs')
            result['claims']=claim_token_scores(text,ids,encoded['offset_mapping'],lp,ent)
            path=directory/'token_statistics.npz';np.savez(path,token_ids=ids,offsets=encoded['offset_mapping'],log_probability=lp,entropy=ent)
            result.update(token_statistics_path=str(path.resolve()),token_statistics_sha256=sha(path))
        return results


def tokens(generation,output,gpu_uuid,config_path=ROOT/'maira_logprob/generation_config.json',project_root=PROJECT):
    original,protocol,sources=checked_run(generation,'reports.jsonl','generation')
    rows=[{**r,'generation_status':r['status'],'status':'unattempted' if r['status']=='success' else r['status'],
           'attempted':False,'mean_token_log_probability':None,'claims':[]} for r in original]
    with fresh_run(output,'tokens') as (out,state):
        try:
            config,bindings=maira_config(config_path,project_root);bindings.update(sources)
            require(batch_config(config,16)==protocol['config'],'Token scoring model/generation protocol differs')
            execution=select_gpu(gpu_uuid);backend=TokenBackend(config)
            execution.update(runtime_device(backend.torch,'cuda:0',gpu_uuid))
            write(out/'protocol.json',dict(schema=SCHEMA,stage='tokens',config=config,execution=execution,source_file_sha256=bindings,
                asset_verification=verification_summary(),
                batch_size=16,score='canonical report suffix only; no prompt/PAD/EOS; mean token logP and negative entropy',
                generation_protocol_sha256=sha(Path(generation)/'protocol.json'),labels_read=False))
            pending=[r for r in rows if r['generation_status']=='success']
            for start in range(0,len(pending),16):
                group=pending[start:start+16]
                for r in group:r['attempted']=True
                try:
                    items=[dict(row=generic_row(r['input']),cache=r) for r in group]
                    result=backend.score_tokens(items,[out/'audit'/digest(r['input']['input_id']) for r in group])
                    require(len(result)==len(group),'Teacher-forcing population changed')
                    for r,value in zip(group,result):r.update(value,status='success')
                except BaseException as exc:
                    for r in group:r.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                    raise
                write_jsonl(out/'token_reports.jsonl',rows);state.update(status='running',processed_reports=min(start+16,len(pending)))
                write(out/'status.json',state)
            finish(out,state,rows,'token_reports.jsonl',bindings)
        finally:write_jsonl(out/'token_reports.jsonl',rows);write_jsonl(out/'denominators.jsonl',denominator(rows))


def samples(generation,output,gpu_uuid,config_path=ROOT/'maira_logprob/generation_config.json',project_root=PROJECT):
    original,protocol,sources=checked_run(generation,'reports.jsonl','generation')
    rows=[{**r,'generation_status':r['status'],'status':'unattempted' if report_claims(r) else 'not_applicable',
           'attempted':False,'samples':[],'eligible_original_claims':report_claims(r)} for r in original]
    with fresh_run(output,'samples') as (out,state):
        try:
            config,bindings=maira_config(config_path,project_root);bindings.update(sources)
            require(batch_config(config,16)==protocol['config'],'Sampling model differs from greedy reports')
            config['generation']=dict(do_sample=True,max_new_tokens=512,use_cache=True,num_beams=1,temperature=1.0,top_p=1.0,top_k=0)
            execution=select_gpu(gpu_uuid);backend=MairaBackend(config)
            execution.update(runtime_device(backend.torch,'cuda:0',gpu_uuid))
            write(out/'protocol.json',dict(schema=SCHEMA,stage='samples',config=config,execution=execution,source_file_sha256=bindings,
                asset_verification=verification_summary(),
                samples_per_eligible_report=10,seed_base=20260922,labels_read=False,
                selection='all successful nontruncated greedy reports with eligible parser target negatives; no outcome labels'))
            for r in rows:
                if r['status']=='not_applicable':continue
                r['attempted']=True
                try:
                    for index in range(10):
                        seed=(20260922+index+int(digest(r['input']['input_id'])[:8],16))%(2**31-1)
                        backend.config['seed']=seed
                        result=backend.generate(generic_row(r['input']),out/'audit'/digest(r['input']['input_id'])/str(index))
                        for key in ('tensor_sha256','presented_image_sha256'):
                            require(result['processor_audit'][key]==r['processor_audit'][key],'Sampled report saw different pixels')
                        require(sha(r['input']['presented_image_path'])==r['input']['presented_image_sha256'],'Input changed')
                        r['samples'].append(dict(sample_index=index,seed=seed,status='truncated' if result['possibly_truncated'] else 'success',**result))
                    r['status']='success' if all(x['status']=='success' for x in r['samples']) else 'truncated'
                except BaseException as exc:
                    r.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
                finally:
                    write_jsonl(out/'samples.jsonl',rows);state.update(status='running',sampled_reports=sum(bool(x['samples']) for x in rows))
                    write(out/'status.json',state)
            finish(out,state,rows,'samples.jsonl',bindings)
        finally:write_jsonl(out/'samples.jsonl',rows);write_jsonl(out/'denominators.jsonl',denominator(rows))


def score_tokens_method(method,token_run,output):
    require(method in METHODS[:3],'Expected a token-based report method')
    rows,protocol,bindings=checked_run(token_run,'token_reports.jsonl','tokens');bindings.update(code_bindings())
    cp=ROOT/method/'config.json';config=read(cp);bindings[str(cp)]=sha(cp)
    field={'maira_logprob':'mean_token_log_probability','sentence_logprob':'sentence_mean_log_probability',
           'token_entropy':'sentence_negative_mean_token_entropy'}[method]
    scores=[]
    with fresh_run(output,method) as (out,state):
        write(out/'protocol.json',dict(method=method,score_field=field,source_file_sha256=bindings,config=config,
            claim_scope='original eligible negative occurrences; absent declarations are not invented',clinical_support_probability=False))
        for row in rows:
            claims=report_claims({**row,'status':row['generation_status']})
            by_id={c['claim_id']:c for c in row.get('claims',[])}
            if row['status']=='success':require(set(by_id)=={c['claim_id'] for c in claims},'Scored claim population changed')
            for claim in claims:
                score=(row[field] if method=='maira_logprob' else by_id[claim['claim_id']][field]) if row['status']=='success' else None
                require(score is None or (np.isfinite(score) and score<=1e-6),'Invalid confidence score')
                scores.append({**row['input'],**claim,'method':method,'status':'success' if score is not None else 'failed',
                    'score':score,'failure_reason':None if score is not None else row['status'],
                    'generated_findings_sha256':digest(row['generated_findings'])})
        write_jsonl(out/'scores.jsonl',scores);write_jsonl(out/'denominators.jsonl',denominator(rows));verify(bindings)
        state.update(status='completed' if all(r['score'] is not None for r in scores) else 'completed_with_failures',
            full_input_count=len(rows),eligible_claims=len(scores),scored_claims=sum(r['score'] is not None for r in scores),
            generation_counts=dict(Counter(r['generation_status'] for r in rows)),token_counts=dict(Counter(r['status'] for r in rows)),
            output_sha256={p:sha(out/p) for p in ('scores.jsonl','denominators.jsonl','protocol.json')})


def score_cli(method):
    p=argparse.ArgumentParser();p.add_argument('--token-run',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();score_tokens_method(method,a.token_run,a.output)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('generate','tokens','samples'))
    p.add_argument('--inputs',type=Path);p.add_argument('--expected-inputs',type=int);p.add_argument('--generation',type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--config',type=Path,default=ROOT/'maira_logprob/generation_config.json');p.add_argument('--project-root',type=Path,default=PROJECT)
    a=p.parse_args()
    if a.stage=='generate':
        if a.inputs is None or a.expected_inputs is None:p.error('generate needs --inputs and --expected-inputs')
        generate(a.inputs,a.output,a.expected_inputs,a.gpu_uuid,a.config,a.project_root)
    else:
        if a.generation is None:p.error('tokens/samples needs --generation')
        globals()[a.stage](a.generation,a.output,a.gpu_uuid,a.config,a.project_root)


if __name__=='__main__':main()

"""Pinned official SCUQ on ten new samples and the same greedy report claims."""
import argparse
from collections import Counter
import math
import os
from pathlib import Path
import sys
from baselines.visual import ROOT,PROJECT,read,sha,digest,require,write,select_gpu,runtime_device
from baselines.reporting import checked_run,asset_bindings,code_bindings,fresh_run,write_jsonl,report_claims,denominator,verify,verification_summary


def sentence_index(text,sentences,claim):
    require(sentences==[s for s in text.split('. ') if s.strip()],'Official sentence list changed')
    spans=[];cursor=0
    for part in text.split('. '):
        if part.strip():spans.append((cursor,cursor+len(part)))
        cursor+=len(part)+2
    positions=[i for i,(a,b) in enumerate(spans) if a<=claim['start'] and claim['end']<=b]
    require(len(positions)==1,'Cannot uniquely map original claim to official SCUQ sentence')
    return positions[0]


def sample_group(row):
    samples=row['samples']
    require(len(samples)==10 and {s['sample_index'] for s in samples}==set(range(10)),'Exactly ten unique samples required')
    require(all(s['status']=='success' and s['possibly_truncated'] is False and isinstance(s['generated_findings'],str)
                and s['generated_findings'].strip() for s in samples),'Incomplete or truncated sample set')
    return [s['generated_findings'] for s in sorted(samples,key=lambda s:s['sample_index'])]


def score(generation,samples,output,gpu_uuid,project_root=PROJECT,config_path=None):
    originals,gp,bindings=checked_run(generation,'reports.jsonl','generation')
    sampled,sp,source=checked_run(samples,'samples.jsonl','samples');bindings.update(source);bindings.update(code_bindings())
    bindings[str(Path(__file__).resolve())]=sha(__file__)
    require(sp.get('samples_per_eligible_report')==10,'Sampling protocol budget changed')
    a={r['input']['input_id']:r for r in sampled};scores=[];per_report={}
    require(set(a)=={r['input']['input_id'] for r in originals},'Sample population differs')
    for original in originals:
        key=original['input']['input_id'];s=a[key]
        require(original['input']==s['input'] and original['generated_findings']==s['generated_findings']
                and original['status']==s['generation_status'],'Samples belong to a different greedy report')
        claims=report_claims(original)
        per_report[key]=[]
        for claim in claims:
            record={**original['input'],**claim,'method':'scuq_radgraph','score':None,'status':'failed',
                    'failure_reason':'not_attempted','attempted':False,'generated_findings_sha256':digest(original['generated_findings'])}
            scores.append(record);per_report[key].append(record)
    with fresh_run(output,'scuq_radgraph') as (out,state):
        try:
            cp=Path(config_path).resolve() if config_path else ROOT/'scuq_radgraph/config.json'
            config=read(cp);bindings[str(cp)]=sha(cp)
            for name in ('assets_manifest','source_manifest'):
                spec=config[name];p=(cp.parent/spec['path']).resolve()
                require(sha(p)==spec['sha256'],'SCUQ bound inventory changed')
                _,folder,b=asset_bindings(p,project_root);bindings.update(b)
                if name=='assets_manifest':radgraph=folder
                else:upstream=folder
            os.environ.update(HF_HOME=str(radgraph/'hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            execution=select_gpu(gpu_uuid)
            sys.path.insert(0,str(upstream))
            import torch
            from radgraph import F1RadGraph
            from scuq import SentenceUncertaintyScorer,ReportSample
            import scuq.uq.sentence_uq as official
            require(Path(official.__file__).resolve().is_relative_to(upstream.resolve()),'SCUQ import is not the pinned upstream source')
            torch.set_num_threads(4);execution.update(runtime_device(torch,'cuda:0',gpu_uuid))
            scorer=SentenceUncertaintyScorer()
            scorer._f1radgraph=F1RadGraph(reward_level='complete',model_type='radgraph',cuda=0,
                                         model_cache_dir=str(radgraph),tokenizer_cache_dir=str(radgraph/'hf/hub'))
            write(out/'protocol.json',dict(method='scuq_radgraph',samples=10,config=config,execution=execution,
                asset_verification=verification_summary(),
                source_file_sha256=bindings,upstream_sentence_scoring_sha256=sha(official.__file__),
                scope='same original eligible negative claim occurrences; all input denominators retained',
                original_radgraph_explicitly_selected=True,full_original_paper_reproduction=False))
            for original in originals:
                key=original['input']['input_id'];group=per_report[key]
                if not group:continue
                for r in group:r['attempted']=True
                try:texts=sample_group(a[key])
                except ValueError as exc:
                    for r in group:r['failure_reason']=str(exc)
                    continue
                try:
                    with torch.inference_mode():result=scorer.score(ReportSample(study_id=key,original_report=original['generated_findings'],sampled_reports=texts))
                    for r in group:
                        index=sentence_index(original['generated_findings'],result.sentences,r);value=1-float(result.uncertainty_scores[index])
                        require(math.isfinite(value) and 0<=value<=1,'Invalid official SCUQ consistency score')
                        r.update(status='success',score=value,failure_reason=None,sentence_index=index)
                except BaseException as exc:
                    for r in group:r.update(status='failed',score=None,failure_reason=f'{type(exc).__name__}: {exc}')
                    raise
                write_jsonl(out/'scores.jsonl',scores);state.update(status='running',scored_claims=sum(r['score'] is not None for r in scores))
                write(out/'status.json',state)
            verify(bindings);write_jsonl(out/'scores.jsonl',scores);write_jsonl(out/'denominators.jsonl',denominator(sampled))
            state.update(status='completed' if all(r['score'] is not None for r in scores) else 'completed_with_failures',
                full_input_count=len(originals),eligible_report_count=sum(bool(v) for v in per_report.values()),eligible_claims=len(scores),
                scored_claims=sum(r['score'] is not None for r in scores),generation_counts=dict(Counter(r['status'] for r in originals)),
                sampling_counts=dict(Counter(r['status'] for r in sampled)),
                asset_verification=verification_summary(),
                output_sha256={p:sha(out/p) for p in ('scores.jsonl','denominators.jsonl','protocol.json')})
        finally:write_jsonl(out/'scores.jsonl',scores);write_jsonl(out/'denominators.jsonl',denominator(sampled))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--generation',type=Path,required=True)
    p.add_argument('--samples',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--config',type=Path);p.add_argument('--project-root',type=Path,default=PROJECT)
    a=p.parse_args();score(a.generation,a.samples,a.output,a.gpu_uuid,a.project_root,a.config)


if __name__=='__main__':main()

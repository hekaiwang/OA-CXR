"""Versioned fixed-budget sampling continuation; never retry a failed sample slot.

The original reporting/generation implementations remain unchanged. This entry
also supports fresh cohorts and records processor parse failures as missing.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image

from baselines import reporting as r

VERSION = 'fixed-slot-sampling-recovery-v1'
EVIDENCE_SCHEMA = 'oa-cxr-sampling-parse-failure-evidence-v1'
SAMPLING = dict(do_sample=True, max_new_tokens=512, use_cache=True, num_beams=1,
                temperature=1.0, top_p=1.0, top_k=0)
OLD_FILES = ('status.json', 'protocol.json', 'samples.jsonl', 'denominators.jsonl')


def seed_for(input_id, index):
    r.require(type(index) is int and 0 <= index < 10, 'Fixed sample index must be 0..9')
    return (20260922 + index + int(r.digest(input_id)[:8], 16)) % (2**31 - 1)


def fresh_rows(original):
    return [{**copy.deepcopy(row), 'generation_status': row['status'],
             'status': 'unattempted' if r.report_claims(row) else 'not_applicable',
             'attempted': False, 'samples': [], 'eligible_original_claims': r.report_claims(row)} for row in original]


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def audit_bindings(audit, original_audit, directory):
    """Verify sample's actual tensor/PNG and the audit JSON, not just its label."""
    directory = Path(directory).resolve(strict=True)
    r.require(isinstance(audit, dict) and all(audit.get(k) == original_audit.get(k)
        for k in ('tensor_sha256', 'presented_image_sha256')), 'Sample processor pixels differ from original report')
    paths = {'tensor': (audit.get('tensor_path'), audit.get('tensor_sha256'))}
    images, hashes = audit.get('presented_image_paths', {}), audit.get('presented_image_sha256', {})
    r.require(images and images.keys() == hashes.keys(), 'Missing sample processor PNG identity')
    paths.update({view: (path, hashes[view]) for view, path in images.items()})
    bindings = {}
    for value, digest in paths.values():
        r.require(isinstance(value, str) and isinstance(digest, str), 'Missing sample audit file/SHA')
        path = Path(value).resolve(strict=True)
        r.require(path.is_relative_to(directory) and path.is_file() and r.sha(path) == digest,
                  'Sample audit path/SHA differs')
        bindings[str(path)] = digest
    path = directory / 'processor_audit.json'
    r.require(r.read(path) == audit, 'Sample audit JSON differs from recorded audit')
    bindings[str(path)] = r.sha(path)
    return bindings


def import_failed(original, generation, config, failed_run, evidence_path):
    """Accept only one sealed, traceback-confirmed historical parser failure."""
    failed = Path(failed_run).resolve(strict=True)
    evidence_path = Path(evidence_path).resolve(strict=True)
    evidence = r.read(evidence_path)
    r.require(evidence.get('schema') == EVIDENCE_SCHEMA and
        Path(evidence.get('failed_run', '')).resolve() == failed and
        Path(evidence.get('generation_run', '')).resolve() == Path(generation).resolve() and
        isinstance(evidence.get('observed_by'), str) and evidence['observed_by'].strip(), 'Failure evidence identity differs')
    r.require(not Path(str(failed) + '.lock').exists(), 'Old failed run still has an active/stale writer lock')
    expected = {str(failed / name): r.sha(failed / name) for name in OLD_FILES}
    r.require(evidence.get('source_file_sha256') == expected, 'Old failure bundle differs from sealed evidence')
    trace = evidence.get('traceback', {})
    trace_path = Path(trace.get('path', '')).resolve(strict=True)
    r.require(trace_path.is_file() and r.sha(trace_path) == trace.get('sha256'), 'Failure traceback SHA differs')
    trace_text = trace_path.read_text(encoding='utf-8', errors='replace')
    r.require('convert_output_to_plaintext_or_grounded_sequence' in trace_text and 'AssertionError' in trace_text,
              'Historical processor AssertionError is not demonstrated by the bound traceback')
    expected.update({str(evidence_path): r.sha(evidence_path), str(trace_path): trace['sha256']})
    state, protocol = r.read(failed / 'status.json'), r.read(failed / 'protocol.json')
    r.require(state.get('schema') == r.SCHEMA and state.get('stage') == 'samples' and
        state.get('status') == 'failed' and state.get('error', '').startswith('AssertionError:'),
        'Expected the original failed sample stage with processor AssertionError')
    r.require(protocol.get('config') == config and protocol.get('samples_per_eligible_report') == 10
        and protocol.get('seed_base') == 20260922 and protocol.get('labels_read') is False,
        'Historical sampling model/seed/budget protocol differs')
    r.require(all(protocol.get('source_file_sha256', {}).get(str(Path(generation).resolve() / name)) ==
        r.sha(Path(generation) / name) for name in ('reports.jsonl', 'status.json', 'protocol.json', 'denominators.jsonl')),
        'Historical sampler is not bound to the same completed greedy generation')
    # Current maira_config verifies official assets; these bindings additionally
    # establish that the old sampler used the same immutable inputs and code.
    r.verify(protocol['source_file_sha256'])
    expected.update(protocol['source_file_sha256'])
    rows = read_rows(failed / 'samples.jsonl')
    r.require(len(rows) == len(original) and [x['input'] for x in rows] == [x['input'] for x in original],
              'Historical sample population/order changed')
    r.require(read_rows(failed / 'denominators.jsonl') == r.denominator(rows), 'Historical sample denominators differ')
    failure = evidence.get('failure', {})
    r.require(failure.get('stage') == 'processor_parse' and failure.get('exception_type') == 'AssertionError'
        and failure.get('raw_completion_saved') is False, 'Only evidenced historical parse failure is recoverable')
    r.require(sum(x['status'] == 'failed' for x in rows) == 1, 'Expected exactly one historical failed report')
    reused, audits, failure_seen, historical_slot = 0, {}, False, None
    for original_row, row in zip(original, rows):
        r.require(row.get('generation_status') == original_row['status'] and
            row.get('eligible_original_claims') == r.report_claims(original_row) and
            all(row.get(k) == v for k, v in original_row.items() if k not in ('status', 'attempted', 'error')),
            'Historical samples belong to different greedy report contents')
        eligible = bool(r.report_claims(original_row)); samples = row.get('samples')
        r.require(isinstance(samples, list) and len(samples) <= 10, 'Invalid historical sample slots')
        if not eligible:
            r.require(row['status'] == 'not_applicable' and not samples and row.get('attempted') is False,
                      'Ineligible report received samples')
            continue
        identifier = row['input']['input_id']
        for index, sample in enumerate(samples):
            r.require(type(sample.get('sample_index')) is int and sample['sample_index'] == index
                and sample.get('seed') == seed_for(identifier, index) and sample.get('generation_calls') == 1
                and sample.get('status') in ('success', 'truncated') and sample.get('synthetic') is False
                and isinstance(sample.get('generated_findings'), str) and sample['generated_findings'].strip()
                and sample.get('possibly_truncated') is (sample['status'] == 'truncated'),
                'Historical sample index/seed/completion status differs')
            directory = failed / 'audit' / r.digest(identifier) / str(index)
            audits.update(audit_bindings(sample['processor_audit'], original_row['processor_audit'], directory))
        if row['status'] == 'failed':
            index = len(samples)
            r.require(not failure_seen and index < 10 and row.get('attempted') is True
                and row.get('error', '').startswith('AssertionError:') and failure.get('input_id') == identifier
                and failure.get('sample_index') == index and failure.get('seed') == seed_for(identifier, index),
                'Historical failed slot identity differs')
            directory = failed / 'audit' / r.digest(identifier) / str(index)
            audit = r.read(directory / 'processor_audit.json')
            audits.update(audit_bindings(audit, original_row['processor_audit'], directory))
            historical_slot = dict(input_id=identifier, sample_index=index, seed=seed_for(identifier, index))
            samples.append(dict(sample_index=index, seed=seed_for(identifier, index), status='failed',
                generated_findings=None, decoded_completion=None, completion_tokens=None, prompt_tokens=None,
                possibly_truncated=None, processor_audit=audit, generation_calls=1, synthetic=False,
                failure_stage='processor_parse', exception_type='AssertionError',
                error='Historical processor parse failed; raw completion was not persisted and is unavailable',
                raw_completion_recoverable=False, failure_evidence_sha256=r.sha(evidence_path)))
            failure_seen = True
        elif failure_seen:
            r.require(row['status'] == 'unattempted' and not samples and row.get('attempted') is False,
                      'Unexpected work after historical fatal slot')
        else:
            r.require(len(samples) == 10 and row.get('attempted') is True and row['status'] ==
                ('success' if all(s['status'] == 'success' for s in samples) else 'truncated'),
                'Incomplete report before historical failure')
        reused += len(samples) - int(row['status'] == 'failed')
    r.require(failure_seen, 'Missing evidenced historical failure')
    expected.update(audits)
    return rows, expected, audits, dict(parent_run=str(failed), parent_files_sha256=evidence['source_file_sha256'],
        failure_evidence_sha256=r.sha(evidence_path), reused_sample_count=reused,
        historical_failed_slot=historical_slot, raw_completion_recoverable=False,
        original_execution=protocol.get('execution'), no_resampling=True)


class ParseSafeMairaBackend(r.MairaBackend):
    """Identical singleton inputs/RNG/generate kwargs; persist before parsing."""
    def generate(self, row, audit_dir):
        torch = self.torch; seed = int(self.config['seed'])
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        images = {}
        for view, path in row['local_image_paths'].items():
            with Image.open(path) as im: images[view] = im.convert('RGB')
        metadata = row.get('allowed_input_metadata', {})
        inputs = self.processor.format_and_preprocess_reporting_input(
            current_frontal=images['frontal'], current_lateral=images.get('lateral'), prior_frontal=None, prior_report=None,
            indication=metadata.get('indication'), technique=metadata.get('technique'), comparison=metadata.get('comparison'),
            get_grounding=False, return_tensors='pt')
        inputs = inputs.to(self.device)
        if self.config['dtype'] != 'float32':
            for key, value in inputs.items():
                if hasattr(value, 'is_floating_point') and value.is_floating_point():
                    inputs[key] = value.to(dtype=getattr(torch, self.config['dtype']))
        audit = self._audit(inputs, row, audit_dir)
        torch.cuda.synchronize(self.device); torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode(): output = self.model.generate(**inputs, **self.config['generation'])
        torch.cuda.synchronize(self.device)
        prefix_len = inputs['input_ids'].shape[-1]
        token_path = audit_dir / 'completion_tokens.npy'
        np.save(token_path, output[0, prefix_len:].detach().cpu().numpy(), allow_pickle=False)
        # Decode/I/O/GPU failures remain fatal; only the explicit parser boundary
        # below becomes a missing scientific score.
        raw_text = self.processor.decode(output[0, prefix_len:], skip_special_tokens=True).lstrip()
        text_path = audit_dir / 'decoded_completion.json'; r.write(text_path, {'decoded_completion': raw_text})
        value = dict(generated_findings=None, decoded_completion=raw_text, processor_audit=audit,
            generation_calls=1, synthetic=False, prompt_tokens=int(prefix_len),
            completion_tokens=int(output.shape[-1] - prefix_len),
            possibly_truncated=int(output.shape[-1] - prefix_len) >= self.config['generation']['max_new_tokens'],
            peak_memory_bytes=int(torch.cuda.max_memory_allocated(self.device)),
            peak_reserved_memory_bytes=int(torch.cuda.max_memory_reserved(self.device)),
            completion_artifacts={str(token_path.resolve()): r.sha(token_path), str(text_path.resolve()): r.sha(text_path)})
        try:
            findings = self.processor.convert_output_to_plaintext_or_grounded_sequence(raw_text)
            r.require(isinstance(findings, str) and findings.strip(), 'Expected nonempty ungrounded Findings string')
        except (AssertionError, ValueError, TypeError, IndexError) as exc:
            return dict(value, status='failed', failure_stage='processor_parse', exception_type=type(exc).__name__,
                        error='Processor completion parsing failed; raw completion and tokens preserved')
        return dict(value, generated_findings=findings, status='truncated' if value['possibly_truncated'] else 'success')


def samples(generation, output, gpu_uuid, config_path=r.ROOT / 'maira_logprob/generation_config.json',
            project_root=r.PROJECT, *, failed_run=None, failure_evidence=None, backend_factory=None):
    r.require((failed_run is None) == (failure_evidence is None), 'Recovery requires both --failed-run and --failure-evidence')
    original, greedy_protocol, bindings = r.checked_run(generation, 'reports.jsonl', 'generation')
    config, bound = r.maira_config(config_path, project_root); bindings.update(bound)
    r.require(r.batch_config(config, 16) == greedy_protocol['config'], 'Sampling model differs from original greedy reports')
    config['generation'] = copy.deepcopy(SAMPLING)
    bindings[str(Path(__file__).resolve())] = r.sha(__file__)
    rows, artifacts, recovery = fresh_rows(original), {}, dict(parent_run=None, reused_sample_count=0,
        historical_failed_slot=None, no_resampling=True)
    if failed_run is not None:
        rows, imported, artifacts, recovery = import_failed(original, generation, config, failed_run, failure_evidence)
        bindings.update(imported)
    # Fail before loading a model if any current presented pixels changed.
    for row in rows:
        r.require(r.sha(row['input']['presented_image_path']) == row['input']['presented_image_sha256'], 'Current image SHA differs')
    with r.fresh_run(output, 'samples') as (out, state):
        try:
            execution = r.select_gpu(gpu_uuid)
            backend = (backend_factory or ParseSafeMairaBackend)(config)
            execution.update(r.runtime_device(backend.torch, 'cuda:0', gpu_uuid))
            write_protocol = dict(schema=r.SCHEMA, stage='samples', config=config, execution=execution,
                source_file_sha256=bindings, samples_per_eligible_report=10, seed_base=20260922, labels_read=False,
                selection='all successful nontruncated greedy reports with eligible parser target negatives; no outcome labels',
                recovery_version=VERSION, recovery=recovery, resumed_gpu_uuid=gpu_uuid,
                numerical_equivalence_to_original_hardware_verified=False,
                parser_failure_policy='Keep fixed failed slot; no replacement/resampling; incomplete set yields missing SCUQ',
                asset_verification=r.verification_summary())
            r.write(out / 'protocol.json', write_protocol)
            r.write_jsonl(out / 'samples.jsonl', rows)
            calls = 0
            for row in rows:
                if not row['eligible_original_claims']: continue
                row['attempted'] = True
                identifier = row['input']['input_id']
                for index in range(len(row['samples']), 10):
                    seed = seed_for(identifier, index); directory = out / 'audit' / r.digest(identifier) / str(index)
                    intent = dict(input_id=identifier, sample_index=index, seed=seed, status='attempt_intent',
                                  no_automatic_retry=True)
                    r.write(out / 'attempt_intent.json', intent)
                    state.update(status='running', active_slot=intent, new_generation_calls=calls)
                    r.write(out / 'status.json', state)
                    r.require(r.sha(row['input']['presented_image_path']) == row['input']['presented_image_sha256'], 'Input changed')
                    backend.config['seed'] = seed
                    try:
                        result = backend.generate(r.generic_row(row['input']), directory); calls += 1
                    except BaseException:
                        row.update(status='failed', error='Fatal sampling infrastructure/decode failure; active slot must not be automatically retried')
                        raise
                    artifacts.update(audit_bindings(result['processor_audit'], row['processor_audit'], directory))
                    r.require(r.sha(row['input']['presented_image_path']) == row['input']['presented_image_sha256'], 'Input changed')
                    r.require(result.get('status') in ('success', 'truncated', 'failed'), 'Missing explicit sample outcome')
                    if result['status'] == 'failed':
                        r.require(result.get('failure_stage') == 'processor_parse' and result.get('generated_findings') is None,
                                  'Only processor parse failures may continue fixed sampling')
                    else:
                        r.require(isinstance(result.get('generated_findings'), str) and result['generated_findings'].strip()
                            and result.get('possibly_truncated') is (result['status'] == 'truncated'), 'Invalid parsed sample')
                    r.require(result.get('generation_calls') == 1 and result.get('synthetic') is False,
                              'Expected exactly one real generation call per new slot')
                    for path, digest in result['completion_artifacts'].items():
                        r.require(Path(path).resolve().is_relative_to(directory.resolve()) and r.sha(path) == digest,
                                  'New raw completion evidence differs')
                        artifacts[path] = digest
                    row['samples'].append(dict(sample_index=index, seed=seed, **result))
                    r.write_jsonl(out / 'samples.jsonl', rows)
                    r.write(out / 'attempt_intent.json', {**intent, 'status': 'committed'})
                row['status'] = ('failed' if any(x['status'] == 'failed' for x in row['samples']) else
                                 ('truncated' if any(x['status'] == 'truncated' for x in row['samples']) else 'success'))
                row['error'] = 'One or more fixed sample slots failed; no replacement was drawn' if row['status'] == 'failed' else None
                state.update(status='running', active_slot=None, new_generation_calls=calls,
                    complete_slot_reports=sum(len(x['samples']) == 10 for x in rows))
                r.write_jsonl(out / 'samples.jsonl', rows); r.write(out / 'status.json', state)
            r.require(all(len(row['samples']) == 10 for row in rows if row['eligible_original_claims']), 'Missing fixed sample slots')
            r.verify(artifacts)
            r.write(out / 'sample_audit_manifest.json', {'schema': VERSION, 'artifact_sha256': artifacts})
            r.finish(out, state, rows, 'samples.jsonl', bindings)
            state['output_sha256']['sample_audit_manifest.json'] = r.sha(out / 'sample_audit_manifest.json')
            if (out / 'attempt_intent.json').exists():
                state['output_sha256']['attempt_intent.json'] = r.sha(out / 'attempt_intent.json')
            state.update(recovery_version=VERSION, reused_sample_count=recovery['reused_sample_count'],
                         historical_failed_slots=int(recovery['historical_failed_slot'] is not None), new_generation_calls=calls,
                         fixed_sample_slots=sum(len(row['samples']) for row in rows))
        finally:
            r.write_jsonl(out / 'samples.jsonl', rows)
            r.write_jsonl(out / 'denominators.jsonl', r.denominator(rows))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True); parser.add_argument('--failed-run', type=Path)
    parser.add_argument('--failure-evidence', type=Path)
    parser.add_argument('--config', type=Path, default=r.ROOT / 'maira_logprob/generation_config.json')
    parser.add_argument('--project-root', type=Path, default=r.PROJECT)
    args = parser.parse_args()
    print(samples(args.generation, args.output, args.gpu_uuid, args.config, args.project_root,
                  failed_run=args.failed_run, failure_evidence=args.failure_evidence))


if __name__ == '__main__': main()

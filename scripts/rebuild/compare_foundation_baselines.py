"""Five-regressor comparison on the fixed four-source image/report populations.

All new methods are task adaptations. Internal OA-CXR versions are not counted
as independent baselines. Predictions are loaded only from completed bound
receipts, with full nominal denominators and paired source-group resampling.
"""
from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare
from train_foundation_baseline import bind, require, PREDICTION_SCHEMA
from baselines import tabular
from oa_cxr.io import exclusive_writer, load_json, read_jsonl, sha256_file, stable_hash, write_json
from oa_cxr.rebuild.metrics import metrics, interval, bootstrap

SCHEMA = 'oa-cxr-foundation-baseline-comparison-v1'
METHODS = ('oa_cxr', 'biomedclip', 'chexzero', 'dinov2', 'tabm')
IDENTITY = ('row_id', 'input_id', 'source_id', 'source_image_sha256', 'dataset', 'split',
            'split_group_id', 'finding', 'variant_index', 'cache_split')


def align(truth, records, method):
    lookup = {row['row_id']: row for row in truth}; mapped = {}
    require(len(lookup) == len(truth), 'Duplicate truth identity')
    for row in records:
        key = row.get('row_id')
        require(key in lookup and key not in mapped, 'Unknown/duplicate prediction row')
        require(all(row.get(k) == lookup[key].get(k) for k in IDENTITY), 'Prediction source/query identity differs')
        if method != 'oa_cxr': require(row.get('method') == method, 'Prediction method differs')
        if row.get('status') == 'success':
            require(type(row.get('score')) in (int, float) and math.isfinite(row['score']) and 0 <= row['score'] <= 1,
                'Successful regression score must be finite within [0,1]')
        else:
            require(row.get('status') == 'failed' and row.get('score') is None and row.get('failure_reason'),
                'Failure must be explicit with null score and reason')
        mapped[key] = row
    require(set(mapped) == set(lookup), 'Incomplete nominal prediction denominator')
    ordered = [mapped[row['row_id']] for row in truth]
    return np.array([row['score'] if row['score'] is not None else np.nan for row in ordered], dtype=float), ordered


def rank_points(points):
    result = {method: {} for method in points}
    for metric in ('mae', 'rmse', 'bad_proxy_auroc', 'bad_proxy_auprc', 'aurc'):
        valid = {method: value[metric] for method, value in points.items() if value[metric] is not None}
        larger = metric in ('bad_proxy_auroc', 'bad_proxy_auprc')
        for method in points:
            result[method][metric] = None if method not in valid else 1 + sum(
                score > valid[method] if larger else score < valid[method] for score in valid.values())
    return result


def macro_mae(groups, datasets):
    values = {}
    for method in METHODS:
        per_dataset = {dataset: groups[dataset]['metrics'][method]['mae'] for dataset in datasets}
        mean = float(np.mean(list(per_dataset.values()))) if all(v is not None for v in per_dataset.values()) else None
        values[method] = dict(mae=mean, dataset_maes=per_dataset)
    for method, value in values.items():
        value['rank'] = None if value['mae'] is None else 1 + sum(
            item['mae'] is not None and item['mae'] < value['mae'] for item in values.values())
    return dict(metric='equal_weight_mean_of_four_dataset_MAE', datasets=datasets,
        dataset_weights={dataset: 1 / len(datasets) for dataset in datasets}, methods=values,
        missing_dataset_policy='Undefined if any of four dataset MAEs is undefined',
        bootstrap_interval=None, interval_scope='Individual dataset paired intervals reported separately')


def point_group(rows, y, vectors, records, mask):
    common = mask.copy()
    for values in vectors.values(): common &= np.isfinite(values)
    points = {method: metrics(y[common], values[common], regression=True) for method, values in vectors.items()}
    coverage = {}
    for method, values in vectors.items():
        good = mask & np.isfinite(values); bad = mask & ~good
        coverage[method] = dict(nominal_queries=int(mask.sum()), scored_queries=int(good.sum()), failed_queries=int(bad.sum()),
            coverage=float(good.sum() / mask.sum()) if mask.any() else None,
            failure_reasons=dict(Counter(records[method][int(i)].get('failure_reason', 'missing_score') for i in np.flatnonzero(bad))),
            automatic_review_for_missing_score=int(bad.sum()),
            own_available_metrics=metrics(y[good], values[good], regression=True) if not np.array_equal(good, common) else points[method])
    selected = [row for row, ok in zip(rows, mask) if ok]
    return dict(nominal_queries=int(mask.sum()), nominal_inputs=len({r['input_id'] for r in selected}),
        nominal_source_images=len({r['source_image_sha256'] for r in selected}),
        common_valid_queries=int(common.sum()), common_query_keys_sha256=stable_hash([r['row_id'] for r, ok in zip(rows, common) if ok]),
        conditional_on_all_five_methods_available=True, metrics=points, ranks=rank_points(points), coverage=coverage), common


def regression_bootstrap(rows, y, vectors, nominal, common, *, replicates=1000, seed=17):
    """Query-weighted statistics from paired source/group draws; exact and fast."""
    pool = sorted({(r['dataset'], r['split_group_id']) for r, ok in zip(rows, nominal) if ok})
    lookup = {key: i for i, key in enumerate(pool)}; index = np.flatnonzero(common)
    groups = np.asarray([lookup[(rows[i]['dataset'], rows[i]['split_group_id'])] for i in index], dtype=int)
    count = np.bincount(groups, minlength=len(pool)).astype(float)
    draws = {method: {metric: np.full(replicates, np.nan) for metric in ('mae', 'rmse')} for method in vectors}
    if len(pool) >= 2 and len(index):
        stats = {}
        for method, values in vectors.items():
            error = values[index] - y[index]
            stats[method] = (np.bincount(groups, weights=abs(error), minlength=len(pool)),
                             np.bincount(groups, weights=error ** 2, minlength=len(pool)))
        rng = np.random.default_rng(seed)
        for iteration in range(replicates):
            weights = np.bincount(rng.integers(0, len(pool), len(pool)), minlength=len(pool)); n = float(weights @ count)
            if n == 0: continue
            for method, (absolute, squared) in stats.items():
                draws[method]['mae'][iteration] = float(weights @ absolute / n)
                draws[method]['rmse'][iteration] = float(np.sqrt(weights @ squared / n))
    points = {method: metrics(y[common], values[common], regression=True, curve=False) for method, values in vectors.items()}
    result = dict(replicates=replicates, seed=seed, original_groups=len(pool), contributing_groups=int((count > 0).sum()),
        unit='dataset + split_group_id', all_variants_and_findings_resampled_together=True, zero_scored_groups_included=True,
        direction='OA-CXR minus comparator; negative favors OA-CXR for MAE/RMSE', individual={}, paired_oa_cxr_minus_baseline={})
    for method in vectors:
        result['individual'][method] = {metric: dict(point=points[method][metric], **interval(draws[method][metric], replicates)) for metric in ('mae', 'rmse')}
        if method != 'oa_cxr':
            result['paired_oa_cxr_minus_baseline'][method] = {metric: dict(
                point=points['oa_cxr'][metric] - points[method][metric] if points['oa_cxr'][metric] is not None and points[method][metric] is not None else None,
                **interval(draws['oa_cxr'][metric] - draws[method][metric], replicates)) for metric in ('mae', 'rmse')}
    return result


def load_report_sources(sources, bindings):
    claims, denominators = [], []
    for source in sources:
        status = load_json(bind(source['status']['path'], source['status']['sha256'], bindings))
        require(status.get('status') == 'completed', 'Fixed report evaluation source incomplete')
        for name, destination in (('claims', claims), ('denominators', denominators)):
            path = bind(source[name]['path'], source[name]['sha256'], bindings)
            require(status.get('output_sha256', {}).get(path.name) == source[name]['sha256'], 'Report receipt does not bind data')
            destination.extend(read_jsonl(path))
    eligible = [r for r in claims if r.get('eligible') is True]
    require(len(eligible) == 784 and len(denominators) == 6656, 'Fixed 784 eligible claims and 6656 report inputs required')
    require(len({r['input_id'] for r in denominators}) == len(denominators), 'Duplicate report input')
    counts = Counter(r['input_id'] for r in eligible)
    require(all(r.get('eligible_claims') == counts[r['input_id']] for r in denominators), 'Claim/denominator partition differs')
    require(set(counts) <= {r['input_id'] for r in denominators}, 'Claim outside nominal report roster')
    return eligible, denominators


def report_points(claims, denominators, rows, y, vectors, datasets):
    lookup = {(r['input_id'], r['finding']): i for i, r in enumerate(rows)}
    require(len(lookup) == len(rows), 'Nonunique image/finding identity')
    mapped = []; report_rows = []; seen = set()
    for claim in claims:
        key = (claim['input_id'], claim['finding']); require(key in lookup, 'Report claim lacks image prediction')
        index = lookup[key]; source = rows[index]; occurrence = (claim['input_id'], claim['claim_id'])
        require(occurrence not in seen, 'Duplicate claim occurrence'); seen.add(occurrence)
        require(all(claim.get(k) == source.get(k) for k in ('dataset', 'split', 'source_id', 'split_group_id')),
            'Report claim source identity differs')
        require(claim['retention'] == float(y[index]), 'Report target differs from independent image truth')
        mapped.append(index); report_rows.append(dict(source, row_id=stable_hash(occurrence), retention=float(y[index])))
    labels = y[mapped]; selected_vectors = {method: values[mapped] for method, values in vectors.items()}
    records = {method: [dict(row, failure_reason=None if np.isfinite(value) else 'image_score_unavailable')
        for row, value in zip(report_rows, values)] for method, values in selected_vectors.items()}
    result = dict(eligible_claims=len(claims), nominal_report_inputs=len(denominators), fixed_threshold=.9,
        original_report_keys_sha256=stable_hash([r['input_id'] for r in denominators]),
        original_claim_keys_sha256=stable_hash(sorted(seen)), groups={},
        scope='Frozen eligible-claim scoring and decision audit; no new generation or text editing', clinical_support_evaluated=False)
    for dataset in datasets:
        mask = np.asarray([r['dataset'] == dataset for r in report_rows], dtype=bool)
        points, common = point_group(report_rows, labels, selected_vectors, records, mask)
        pool = {(r['dataset'], r['split_group_id']) for r in denominators if r['dataset'] == dataset}
        rows_used = [r for r, ok in zip(report_rows, common) if ok]
        points['paired_bootstrap'] = bootstrap(rows_used, {m: s[common] for m, s in selected_vectors.items()}, set(vectors),
            'oa_cxr', pool, replicates=1000, seed=17)
        points['decisions'] = {}
        for method, values in selected_vectors.items():
            valid = mask & np.isfinite(values); bad = labels < .9; reviewed = values < .9
            points['decisions'][method] = dict(eligible_claims=int(mask.sum()), scored_claims=int(valid.sum()),
                low_coverage_incorrectly_retained=int((valid & bad & ~reviewed).sum()),
                high_coverage_incorrectly_reviewed=int((valid & ~bad & reviewed).sum()),
                reviewed_scored_claims=int((valid & reviewed).sum()), missing_score_review=int((mask & ~valid).sum()))
        original = [r for r in denominators if r['dataset'] == dataset]
        points['report_denominators'] = dict(nominal_inputs=len(original),
            generation_status_counts=dict(Counter(r['generation_status'] for r in original)),
            possibly_truncated=sum(bool(r.get('possibly_truncated')) for r in original),
            zero_eligible_reports=sum(r['eligible_claims'] == 0 for r in original))
        result['groups'][dataset] = points
    return result


def run(config, config_sha256, output):
    bindings = {}; spec = load_json(bind(config, config_sha256, bindings)); output = Path(output).resolve()
    require(spec.get('schema') == 'oa-cxr-foundation-baseline-comparison-plan-v1' and spec.get('methods') == list(METHODS), 'Fixed comparison roster required')
    datasets = spec['datasets']; require(len(datasets) == len(set(datasets)) == 4, 'Four dataset groups required')
    require(not output.exists(), 'Fresh comparison directory required')
    with exclusive_writer(output):
        output.mkdir(parents=True, exist_ok=False); state = dict(schema=SCHEMA, status='running', started_at=time.time())
        write_json(output / 'status.json', state)
        try:
            rows, targets = [], []; pieces = {m: [] for m in METHODS}; records = {m: [] for m in METHODS}; ancestry = {}
            for item in spec['splits']:
                truth_dir = Path(item['truth_directory']).resolve()
                bind(truth_dir / 'status.json', item['truth_status_sha256'], bindings)
                truth, y = compare.load_truth(truth_dir)
                for name in ('samples.jsonl', 'labels.npz'): bindings[str(truth_dir / name)] = sha256_file(truth_dir / name)
                require(set(item['predictions']) == set(METHODS), 'Missing baseline prediction in split')
                for method, source in item['predictions'].items():
                    directory = Path(source['directory']).resolve(); status = load_json(bind(directory / 'status.json', source['status_sha256'], bindings))
                    require(status.get('status') in ('completed', 'completed_with_failures') and status.get('target_values_loaded') is False,
                        'Completed label-blind prediction required')
                    if method != 'oa_cxr': require(status.get('schema') == PREDICTION_SCHEMA and status.get('method') == method, 'Unexpected prediction schema/method')
                    path = bind(directory / 'predictions.jsonl', status['output_sha256']['predictions.jsonl'], bindings)
                    vector, ordered = align(truth, read_jsonl(path), method)
                    success = int(np.isfinite(vector).sum())
                    require(status.get('nominal_queries') == len(truth) and status.get('successful_queries', status.get('completed_scored_queries')) == success
                        and status.get('failed_queries') == len(truth) - success, 'Prediction denominator receipt differs')
                    if method != 'oa_cxr':
                        identity = (status['training_status_sha256'], status['model_state_sha256'])
                        require(method not in ancestry or ancestry[method] == identity, 'Baseline checkpoint changed between sources')
                        ancestry[method] = identity
                    pieces[method].append(vector); records[method].extend(ordered)
                rows.extend(truth); targets.append(y)
            require(len(rows) == 343395 and len({r['row_id'] for r in rows}) == len(rows), 'Full unique 343395 query population required')
            require(len({r['source_image_sha256'] for r in rows}) == 8805 and len({r['input_id'] for r in rows}) == 114465,
                '8805 source images and 114465 presented inputs required')
            require(set(r['dataset'] for r in rows) == set(datasets), 'Dataset roster differs')
            y = np.concatenate(targets); vectors = {m: np.concatenate(v) for m, v in pieces.items()}
            result = dict(schema=SCHEMA, methods=list(METHODS), reference='oa_cxr',
                nominal_queries=len(rows), nominal_source_images=8805, nominal_inputs=114465,
                groups={}, subgroups={}, source_file_sha256=bindings, baseline_model_identity=ancestry,
                task_adaptations=True, selected_internal_versions_counted_as_external_baselines=False,
                clinical_effectiveness_evaluated=False, ranking_metrics='Image ranking point estimates; report ranking includes paired bootstrap',
                target='Anatomical retention proxy', model_or_threshold_selection_during_evaluation=False)
            for dataset in datasets:
                mask = np.asarray([r['dataset'] == dataset for r in rows], dtype=bool)
                points, common = point_group(rows, y, vectors, records, mask)
                points['paired_regression_bootstrap'] = regression_bootstrap(rows, y, vectors, mask, common)
                result['groups'][dataset] = points; subgroup = {'finding': {}, 'variant': {}, 'finding_by_variant': {}}
                for finding in sorted(tabular.FINDINGS):
                    subset = mask & np.asarray([r['finding'] == finding for r in rows])
                    subgroup['finding'][finding] = point_group(rows, y, vectors, records, subset)[0]
                for variant in range(13):
                    subset = mask & np.asarray([r['variant_index'] == variant for r in rows])
                    subgroup['variant'][str(variant)] = point_group(rows, y, vectors, records, subset)[0]
                    for finding in sorted(tabular.FINDINGS):
                        both = subset & np.asarray([r['finding'] == finding for r in rows])
                        subgroup['finding_by_variant'][finding + '/' + str(variant)] = point_group(rows, y, vectors, records, both)[0]
                result['subgroups'][dataset] = subgroup
                print('completed image metrics', dataset, flush=True)
            result['primary_macro'] = macro_mae(result['groups'], datasets)
            claims, denominators = load_report_sources(spec['report_sources'], bindings)
            result['report_claims'] = report_points(claims, denominators, rows, y, vectors, datasets)
            tabular.verify(bindings)
            result['implementation_sha256'] = {str(Path(__file__).resolve()): sha256_file(__file__),
                str(ROOT / 'src/oa_cxr/rebuild/metrics.py'): sha256_file(ROOT / 'src/oa_cxr/rebuild/metrics.py')}
            write_json(output / 'comparison.json', result)
            state.update(status='completed', output_sha256={'comparison.json': sha256_file(output / 'comparison.json')})
        except BaseException as exc:
            state.update(status='failed', error=repr(exc), traceback=traceback.format_exc()); raise
        finally:
            state['finished_at'] = time.time(); write_json(output / 'status.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'config-sha256', 'output'): parser.add_argument('--' + name, required=True)
    args = parser.parse_args(); run(args.config, args.config_sha256, args.output)

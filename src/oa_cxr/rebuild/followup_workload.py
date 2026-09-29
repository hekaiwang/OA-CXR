"""Matched review budgets for retention-proxy events, not clinical errors.

Lower confidence is reviewed first. Exact ties receive the same fractional
selection probability, so labels never break ties. Bootstrap weights represent
source/group multiplicities, not independently resampled claims or variants.
"""
from __future__ import annotations

import numpy as np

METHODS = ("oa_finding", "oa_whole_lung", "maira_logprob", "sentence_logprob")
BUDGETS = (.05, .10, .20, .30, .50)
EVENT_THRESHOLD = .9
METRICS = ("bad_recall", "risk_among_unreviewed", "unnecessary_fraction_of_reviews",
           "high_coverage_review_rate", "all_report_review_fraction")


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


class BudgetRanking:
    """Precompute score tie groups; apply paired resampling weights repeatedly."""
    def __init__(self, scores, bad, *, mandatory_missing):
        self.scores, self.bad = np.asarray(scores, float), np.asarray(bad, bool)
        if self.scores.ndim != 1 or self.bad.shape != self.scores.shape or np.isinf(self.scores).any():
            raise ValueError("aligned 1D finite-or-NaN confidence and event vectors required")
        self.missing = np.isnan(self.scores)
        if self.missing.any() and not mandatory_missing:
            raise ValueError("common-valid analysis may not include missing scores")
        valid = ~self.missing
        self.valid_indices = np.flatnonzero(valid)
        self.unique_scores, self.inverse = np.unique(self.scores[valid], return_inverse=True)

    def curves(self, weights=None, *, budgets=BUDGETS, all_report_denominator=None):
        if weights is None: weights = np.ones(len(self.bad), float)
        weights = np.asarray(weights, float)
        if weights.shape != self.bad.shape or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("aligned nonnegative finite multiplicities required")
        if any(not np.isfinite(b) or b < 0 or b > 1 for b in budgets):
            raise ValueError("review budgets must be in [0,1]")
        total = float(weights.sum()); bad = float(weights @ self.bad)
        missing = float(weights[self.missing].sum()); missing_bad = float(weights[self.missing] @ self.bad[self.missing])
        w = weights[self.valid_indices]
        tie_weight = np.bincount(self.inverse, weights=w, minlength=len(self.unique_scores)).astype(float)
        tie_bad = np.bincount(self.inverse, weights=w * self.bad[self.valid_indices], minlength=len(self.unique_scores)).astype(float)
        before = np.cumsum(tie_weight) - tie_weight
        result = []
        for budget in budgets:
            quota = max(0., float(budget) * total - missing)
            probability = np.zeros_like(tie_weight)
            np.divide(np.clip(quota - before, 0, tie_weight), tie_weight, out=probability, where=tie_weight > 0)
            reviewed = missing + float(probability @ tie_weight)
            reviewed_bad = missing_bad + float(probability @ tie_bad)
            unreviewed, unreviewed_bad = max(0., total - reviewed), max(0., bad - reviewed_bad)
            unnecessary = max(0., reviewed - reviewed_bad)
            result.append(dict(requested_review_fraction=float(budget), eligible_units=total, low_retention_units=bad,
                mandatory_missing_reviews=missing, reviewed_units=reviewed, reviewed_fraction=_ratio(reviewed, total),
                budget_excess_units=max(0., reviewed - float(budget) * total), reviewed_low_retention=reviewed_bad,
                bad_recall=_ratio(reviewed_bad, bad), unreviewed_units=unreviewed, unreviewed_low_retention=unreviewed_bad,
                risk_among_unreviewed=_ratio(unreviewed_bad, unreviewed), unnecessary_reviews=unnecessary,
                unnecessary_fraction_of_reviews=_ratio(unnecessary, reviewed),
                high_coverage_review_rate=_ratio(unnecessary, total - bad),
                all_report_review_fraction=_ratio(reviewed, all_report_denominator) if all_report_denominator is not None else None))
        return result


def aggregate_reports(claim_units):
    """Any-low-retention event, minimum claim confidence; missing propagates."""
    groups = {}
    for row in claim_units:
        groups.setdefault(row["input_id"], []).append(row)
    reports = []
    for input_id, rows in groups.items():
        first = rows[0]
        if any(any(r[key] != first[key] for key in ("dataset", "split_group_id", "source_id")) for r in rows):
            raise ValueError("claims in one report have inconsistent source identity")
        scores = {}
        for method in METHODS:
            values = [r["scores"][method] for r in rows]
            scores[method] = min(values) if all(np.isfinite(v) for v in values) else float("nan")
        reports.append(dict(input_id=input_id, dataset=first["dataset"], split_group_id=first["split_group_id"],
                            source_id=first["source_id"], bad=any(r["bad"] for r in rows), scores=scores,
                            eligible_claims=len(rows)))
    return reports


def _interval(values):
    values = np.asarray(values, float); valid = values[np.isfinite(values)]
    return dict(lower=float(np.quantile(valid, .025)) if len(valid) else None,
                upper=float(np.quantile(valid, .975)) if len(valid) else None,
                valid_replicates=len(valid), undefined_replicates=len(values) - len(valid))


def analyze_group(claim_units, denominators, *, replicates=1000, seed=17, budgets=BUDGETS):
    """Paired, dataset-stratified source-group bootstrap including zero claims.

    Report budget denominator is the number of reports with eligible claims;
    all-report workload additionally divides by every fixed generation input.
    For common-valid report comparison, retain whole reports only when all their
    claims have all four scores. Never drop an inconvenient constituent claim.
    """
    if replicates < 1: raise ValueError("positive bootstrap replicates required")
    pool = sorted({(r["dataset"], r["split_group_id"]) for r in denominators})
    lookup = {key: i for i, key in enumerate(pool)}
    report_counts = np.bincount([lookup[(r["dataset"], r["split_group_id"])] for r in denominators], minlength=len(pool)).astype(float)
    sources = {(r["dataset"], r["source_id"]) for r in denominators}
    report_units = aggregate_reports(claim_units)
    levels = {"claim": claim_units, "report": report_units}
    result = dict(full_report_inputs=len(denominators), original_sources=len(sources), original_source_groups=len(pool),
                  eligible_claims=len(claim_units), eligible_reports=len(report_units),
                  zero_eligible_reports=len(denominators) - len(report_units),
                  bootstrap_replicates=replicates, bootstrap_seed=seed,
                  bootstrap_unit="dataset + split_group_id, stratified within dataset; zero-eligible groups included",
                  proxy_event="original finding-specific relative retention < 0.9; not clinical error",
                  report_aggregation="minimum constituent confidence (maximum risk); any constituent proxy event",
                  tie_policy="fractional expected selection identical within each exact score tie",
                  levels={})
    prepared = []
    for level, units in levels.items():
        common = np.asarray([all(np.isfinite(r["scores"][m]) for m in METHODS) for r in units], bool)
        result["levels"][level] = dict(nominal_eligible_units=len(units), common_valid_units=int(common.sum()),
            unavailable_by_method={m: sum(not np.isfinite(r["scores"][m]) for r in units) for m in METHODS}, analyses={})
        for name, mask in (("common_valid", common), ("full_failure_first", np.ones(len(units), bool))):
            selected = [r for r, active in zip(units, mask) if active]
            groups = np.asarray([lookup[(r["dataset"], r["split_group_id"])] for r in selected], int)
            bad = np.asarray([r["bad"] for r in selected], bool)
            rankers = {m: BudgetRanking([r["scores"][m] for r in selected], bad,
                                        mandatory_missing=name == "full_failure_first") for m in METHODS}
            point = {m: ranker.curves(budgets=budgets, all_report_denominator=len(denominators) if level == "report" else None)
                     for m, ranker in rankers.items()}
            draws = {m: np.full((replicates, len(budgets), len(METRICS)), np.nan) for m in METHODS}
            analysis = dict(eligible_units=len(selected), contributing_source_groups=len(set(groups)),
                low_retention_units=int(bad.sum()), methods=point,
                workload_denominator="eligible claim occurrences" if level == "claim" else "eligible reports; full generation inputs additionally reported",
                missing_policy="none; same jointly valid units" if name == "common_valid" else "mandatory review before scored units, even when this exceeds requested budget")
            result["levels"][level]["analyses"][name] = analysis
            prepared.append((level, groups, rankers, draws, analysis))
    strata = [np.asarray([i for i, (d, _) in enumerate(pool) if d == dataset], int) for dataset in sorted({d for d, _ in pool})]
    rng = np.random.default_rng(seed)
    for iteration in range(replicates):
        group_weights = np.zeros(len(pool), int)
        for indices in strata:
            group_weights += np.bincount(rng.choice(indices, size=len(indices), replace=True), minlength=len(pool))
        all_reports = float(group_weights @ report_counts)
        for level, groups, rankers, draws, _ in prepared:
            weights = group_weights[groups]
            for method, ranker in rankers.items():
                curves = ranker.curves(weights, budgets=budgets, all_report_denominator=all_reports if level == "report" else None)
                draws[method][iteration] = [[np.nan if row[k] is None else row[k] for k in METRICS] for row in curves]
    for level, groups, rankers, draws, analysis in prepared:
        analysis["paired_oa_finding_minus_comparator"] = {}
        for method in METHODS:
            for b, row in enumerate(analysis["methods"][method]):
                row["intervals"] = {metric: _interval(draws[method][:, b, k]) for k, metric in enumerate(METRICS)}
            if method == "oa_finding": continue
            differences = []
            for b, budget in enumerate(budgets):
                item = dict(requested_review_fraction=float(budget), metrics={})
                for k, metric in enumerate(METRICS):
                    a, other = analysis["methods"]["oa_finding"][b][metric], analysis["methods"][method][b][metric]
                    item["metrics"][metric] = dict(point=a - other if a is not None and other is not None else None,
                        **_interval(draws["oa_finding"][:, b, k] - draws[method][:, b, k]))
                differences.append(item)
            analysis["paired_oa_finding_minus_comparator"][method] = differences
    return result

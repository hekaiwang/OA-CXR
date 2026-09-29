"""Fixed calibration-only logistic combination and matched review workloads."""
from __future__ import annotations

from collections import Counter
import warnings
import numpy as np
from .followup_workload import BudgetRanking, BUDGETS, METRICS, EVENT_THRESHOLD, _interval

METHODS = ("oa_finding", "maira_logprob", "calibrated_combo", "oa_whole_lung")
MODEL_SETTINGS = dict(features=["1-oa_finding", "-maira_logprob"], C=1.0, penalty="l2", solver="lbfgs",
    max_iter=1000, tol=1e-4, random_state=17, class_weight=None, fit_intercept=True,
    weighting="inverse complete eligible claim count per dataset/source group; normalized mean1",
    standardization="same sample-weighted mean/population variance; exactly zero variance scale1")


def risk_features(oa, logprob):
    oa, logprob = np.asarray(oa, float), np.asarray(logprob, float)
    if oa.ndim != 1 or logprob.shape != oa.shape or np.isinf(oa).any() or np.isinf(logprob).any():
        raise ValueError("aligned finite-or-NaN one-dimensional features required")
    if ((oa[np.isfinite(oa)] < 0) | (oa[np.isfinite(oa)] > 1)).any() or (logprob[np.isfinite(logprob)] > 1e-7).any():
        raise ValueError("OA in [0,1] and mean log probability <=0 required")
    return np.column_stack((1.0 - oa, -logprob))


def group_weights(groups):
    groups = [tuple(g) for g in groups]
    counts = Counter(groups)
    if not groups or any(len(g) != 2 or not all(isinstance(x, str) and x for x in g) for g in groups):
        raise ValueError("nonempty dataset/source-group identities required")
    weights = np.asarray([1.0 / counts[g] for g in groups], float)
    return weights / weights.mean()


def fit_combination(x, bad, groups):
    """Caller supplies only verified independent calibration rows, never test."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    x, bad = np.asarray(x, float), np.asarray(bad)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all() or bad.shape != (len(x),) or len(groups) != len(x):
        raise ValueError("two complete calibration features and aligned labels/groups required")
    if set(np.unique(bad).tolist()) != {False, True}:
        raise ValueError("calibration must contain both fixed proxy-event classes")
    weights = group_weights(groups)
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    scale = np.where(variance == 0, 1.0, np.sqrt(variance))
    model = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, tol=1e-4,
        random_state=17, class_weight=None, fit_intercept=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit((x - mean) / scale, bad.astype(int), sample_weight=weights)
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise ValueError("nonfinite logistic fit")
    return dict(schema="oa-cxr-fixed-calibration-logistic-v1", settings=MODEL_SETTINGS,
        mean=mean.tolist(), population_variance=variance.tolist(), scale=scale.tolist(),
        coefficients=model.coef_[0].tolist(), intercept=float(model.intercept_[0]),
        classes=model.classes_.tolist(), optimizer_iterations=int(model.n_iter_[0]),
        fit_claims=len(x), fit_groups=len(set(tuple(g) for g in groups)),
        low_retention_claims=int(bad.sum()), high_retention_claims=int((~bad.astype(bool)).sum()),
        sample_weight_sum=float(weights.sum()), sample_weight_mean=float(weights.mean()),
        calibration_only=True, test_rows_used_for_fit=0)


def predict_confidence(model, x):
    from scipy.special import expit
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] != 2 or np.isinf(x).any():
        raise ValueError("finite-or-NaN two-feature matrix required")
    if model.get("settings") != MODEL_SETTINGS or model.get("classes") != [0, 1] or model.get("calibration_only") is not True:
        raise ValueError("fixed logistic model contract differs")
    mean, scale, coef = (np.asarray(model[key], float) for key in ("mean", "scale", "coefficients"))
    if any(v.shape != (2,) or not np.isfinite(v).all() for v in (mean, scale, coef)) or (scale <= 0).any() or not np.isfinite(model["intercept"]):
        raise ValueError("invalid logistic parameter state")
    valid = np.isfinite(x).all(axis=1)
    confidence = np.full(len(x), np.nan)
    confidence[valid] = 1.0 - expit(((x[valid] - mean) / scale) @ coef + model["intercept"])
    return confidence


def aggregate_reports(claim_units):
    grouped = {}
    for row in claim_units:
        grouped.setdefault(row["input_id"], []).append(row)
    output = []
    for input_id, rows in grouped.items():
        first = rows[0]
        if any(any(r[k] != first[k] for k in ("dataset", "split_group_id", "source_id")) for r in rows):
            raise ValueError("report claim identities disagree")
        scores = {}
        for method in METHODS:
            values = [r["scores"][method] for r in rows]
            scores[method] = min(values) if np.isfinite(values).all() else float("nan")
        output.append(dict(input_id=input_id, source_id=first["source_id"], dataset=first["dataset"],
            split_group_id=first["split_group_id"], bad=any(r["bad"] for r in rows), scores=scores,
            eligible_claims=len(rows)))
    return output


def analyze_group(claims, denominators, *, replicates=1000, seed=17):
    """Report-primary paired bootstrap; all zero-eligible groups stay in pool."""
    if replicates < 1:
        raise ValueError("positive bootstrap replicates required")
    pool = sorted({(r["dataset"], r["split_group_id"]) for r in denominators})
    lookup = {key: i for i, key in enumerate(pool)}
    report_counts = np.bincount([lookup[(r["dataset"], r["split_group_id"])] for r in denominators], minlength=len(pool))
    reports = aggregate_reports(claims)
    result = dict(full_report_inputs=len(denominators), original_sources=len({(r["dataset"], r["source_id"]) for r in denominators}),
        original_source_groups=len(pool), eligible_claims=len(claims), eligible_reports=len(reports),
        zero_eligible_reports=len(denominators)-len(reports), bootstrap_replicates=replicates, bootstrap_seed=seed,
        primary_level="report", secondary_level="claim", event="original finding retention <0.9; not clinical error",
        report_aggregation="min confidence / max risk; any original finding low-retention event",
        ci="individual paired source/group percentile95 intervals; not multiplicity adjusted", levels={})
    prepared = []
    for level, units in (("report", reports), ("claim", claims)):
        common = np.asarray([all(np.isfinite(r["scores"][m]) for m in METHODS) for r in units], bool)
        result["levels"][level] = dict(nominal_eligible_units=len(units), common_valid_units=int(common.sum()),
            unavailable_by_method={m:sum(not np.isfinite(r["scores"][m]) for r in units) for m in METHODS}, analyses={})
        for name, mask in (("common_valid", common), ("full_failure_first", np.ones(len(units), bool))):
            selected = [r for r, keep in zip(units, mask) if keep]
            indices = np.asarray([lookup[(r["dataset"], r["split_group_id"])] for r in selected], int)
            bad = np.asarray([r["bad"] for r in selected], bool)
            rankers = {m:BudgetRanking([r["scores"][m] for r in selected], bad, mandatory_missing=name=="full_failure_first") for m in METHODS}
            point = {m:r.curves(all_report_denominator=len(denominators) if level=="report" else None) for m,r in rankers.items()}
            draws = {m:np.full((replicates,len(BUDGETS),len(METRICS)),np.nan) for m in METHODS}
            section = dict(eligible_units=len(selected), contributing_source_groups=len(set(indices)),
                low_retention_units=int(bad.sum()), methods=point,
                missing_policy="missing mandated review before ranked scores even beyond budget" if name=="full_failure_first" else "same jointly valid full units")
            result["levels"][level]["analyses"][name] = section
            prepared.append((level,indices,rankers,draws,section))
    strata = [np.asarray([i for i,(d,_) in enumerate(pool) if d==dataset],int) for dataset in sorted({d for d,_ in pool})]
    rng = np.random.default_rng(seed)
    for iteration in range(replicates):
        group_weights = np.zeros(len(pool), int)
        for indices in strata:
            group_weights += np.bincount(rng.choice(indices,size=len(indices),replace=True),minlength=len(pool))
        total_reports = float(group_weights @ report_counts)
        for level,indices,rankers,draws,_ in prepared:
            for method,ranker in rankers.items():
                curves = ranker.curves(group_weights[indices],all_report_denominator=total_reports if level=="report" else None)
                draws[method][iteration] = [[np.nan if row[k] is None else row[k] for k in METRICS] for row in curves]
    for _,_,_,draws,section in prepared:
        for m in METHODS:
            for b,row in enumerate(section["methods"][m]):
                row["intervals"] = {metric:_interval(draws[m][:,b,k]) for k,metric in enumerate(METRICS)}
        section["paired_differences"] = {}
        for reference in ("oa_finding","calibrated_combo"):
            section["paired_differences"][reference] = {}
            for other in METHODS:
                if other==reference: continue
                rows = []
                for b,budget in enumerate(BUDGETS):
                    row = dict(requested_review_fraction=budget,metrics={})
                    for k,metric in enumerate(METRICS):
                        a,c = section["methods"][reference][b][metric],section["methods"][other][b][metric]
                        row["metrics"][metric] = dict(point=None if a is None or c is None else a-c,
                            **_interval(draws[reference][:,b,k]-draws[other][:,b,k]))
                    rows.append(row)
                section["paired_differences"][reference][other] = rows
    return result

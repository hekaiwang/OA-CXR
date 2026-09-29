"""Fixed five-model challenge analysis; geometry ground truth is evaluation only.

Bootstrap resamples source groups using group sufficient statistics. No target,
prediction, or subgroup is used to select a model or tune a threshold.
"""
from __future__ import annotations
from collections import Counter
import numpy as np
from oa_cxr.io import stable_hash
from oa_cxr.rebuild.metrics import metrics, interval

METHODS = ("oa_cxr_frozen", "tabm_oa_features", "densenet_global",
           "predicted_mask_geometry_hgb", "oa_cxr_robust_candidate")
DATASETS = ("covidqu_v7", "kermany_v7", "nih_chexmask", "shenzhen")
FINDINGS = ("pleural_effusion", "pneumothorax", "consolidation")
FAMILIES = ("anchor", "unseen_single", "unseen_asymmetric", "background", "photometric", "padding")
BINS = ("all", "[0,.5)", "[.5,.9)", "[.9,1]")
METRICS = ("mae", "rmse", "bad_proxy_auroc", "bad_proxy_auprc", "aurc")


def require(condition, message):
    if not condition: raise ValueError(message)


def validate_targets(targets, valid):
    require(valid.ndim == 2 and valid.shape[1] == 21 and valid.dtype == np.bool_, "valid shape/dtype differs")
    require(targets.shape == valid.shape + (3,) and targets.dtype.kind == "f", "target shape/dtype differs")
    require(np.isfinite(targets[valid]).all() and np.isnan(targets[~valid]).all(), "GT missing partition differs")
    require(np.all((targets[valid] >= 0) & (targets[valid] <= 1)), "GT outside retention range")


def validate_predictions(scores, available, states, valid):
    require(scores.shape == valid.shape + (3,) and scores.dtype.kind == "f", "score shape/dtype differs")
    require(available.shape == valid.shape and available.dtype == np.bool_, "availability shape/dtype differs")
    require(states.shape == valid.shape and states.dtype == np.uint8, "input-state shape/dtype differs")
    require(np.isin(states, (0, 1, 2, 3, 4, 5)).all(), "unknown input state")
    require(np.array_equal(available, states == 3), "availability/input-state mismatch")
    require(np.array_equal(valid, np.isin(states, (2, 3, 4, 5))), "construction/state partition mismatch")
    require(np.isfinite(scores[available]).all() and np.isnan(scores[~available]).all(),
            "available scores must be finite; unavailable scores must all be NaN")
    require(np.all((scores[available] >= 0) & (scores[available] <= 1)), "score outside retention range")
    return dict(successful_inputs=int(available.sum()), successful_queries=int(available.sum()) * 3,
                failed_inference_inputs=int((states == 4).sum()), unattempted_eligible_inputs=int((states == 2).sum()),
                upstream_unavailable_eligible_inputs=int((states == 5).sum()),
                construction_ineligible_inputs=int((states == 0).sum()), construction_failed_inputs=int((states == 1).sum()))


def validate_records(sources, inputs, specification, valid, source_rows_sha256=None):
    require(len(sources) == len(valid) and len(inputs) == valid.size, "record count mismatch")
    require(len({r["source_id"] for r in sources}) == len(sources), "duplicate source identity")
    variants = specification["variants"]
    require(len(variants) == 21 and specification["findings"] == list(FINDINGS), "fixed variant/region roster differs")
    require({v["family"] for v in variants} == set(FAMILIES), "family roster differs")
    if source_rows_sha256 is not None:
        require(stable_hash([{k: v for k, v in r.items() if k != "challenge_source_index"} for r in sources]) == source_rows_sha256,
                "source manifest identity differs")
    ids = set()
    for i, source in enumerate(sources):
        require(source["challenge_source_index"] == i and source["dataset"] in DATASETS, "source row ordering differs")
        for j, variant in enumerate(variants):
            row = inputs[i * 21 + j]
            require(all(row[k] == source[k] for k in ("source_id", "dataset", "split_group_id", "split")), "input source identity differs")
            require((row["source_index"], row["flat_index"], row["variant_index"]) == (i, i*21+j, j), "input row ordering differs")
            require(row["variant"] == variant["name"] and row["family"] == variant["family"], "variant identity differs")
            anchor = "left_30" if variant["name"].startswith("left30_") else "top_30" if variant["name"].startswith("top30_") else "original"
            require(row["anchor_variant"] == anchor, "recorded anchor differs from fixed construction")
            require(row["input_id"] not in ids, "duplicate input id"); ids.add(row["input_id"])
            require(row["status"] in ("eligible", "construction_ineligible", "failed"), "unknown construction state")
            require(bool(valid[i,j]) == (row["status"] == "eligible"), "input eligibility differs")
    return dict(construction_status=dict(Counter(r["status"] for r in inputs)),
                construction_reasons=dict(Counter(str(r.get("reason")) for r in inputs if r["status"] != "eligible")),
                effective_transform_false_eligible=sum(r["status"] == "eligible" and r.get("effective_transform") is False and r["variant"] != "original" for r in inputs))


def verify_geometry(targets, valid, inputs, specification):
    """Reject invalid ground-truth challenge assumptions before reading model metrics."""
    names = [v["name"] for v in specification["variants"]]; lookup = {n: i for i, n in enumerate(names)}
    tolerance = 1e-6; invariant_pairs = []; checked = 0
    for j, item in enumerate(specification["variants"]):
        if item["family"] not in ("background", "photometric", "padding"): continue
        anchors = {inputs[i*21+j]["anchor_variant"] for i in range(len(valid))}
        require(len(anchors) == 1, "anchor mapping changes by source")
        anchor = lookup[next(iter(anchors))]; eligible = valid[:,j] & valid[:,anchor]
        require(np.all(np.abs(targets[eligible,j]-targets[eligible,anchor]) <= tolerance), "GT geometry invariance failed")
        checked += int(eligible.sum()) * 3; invariant_pairs.append((anchor, j, item["family"]))
    expected = [[a,b] for side in ("top", "bottom", "left", "right") for a,b in (("original", side+"_05"), (side+"_05", side+"_30"))]
    require(specification["nested_pairs"] == expected, "fixed eight nested pairs differ")
    nested_pairs = []; nested_checked = 0
    for less, more in expected:
        a,b = lookup[less],lookup[more]; eligible = valid[:,a] & valid[:,b]
        require(np.all(targets[eligible,b] <= targets[eligible,a] + tolerance), "GT nested monotonicity failed")
        for i in np.flatnonzero(eligible):
            outer = inputs[i*21+a]["native_crop_box_xyxy"]; inner = inputs[i*21+b]["native_crop_box_xyxy"]
            require(outer[0] <= inner[0] <= inner[2] <= outer[2] and outer[1] <= inner[1] <= inner[3] <= outer[3],
                    "GT native nested boxes are not contained")
        nested_checked += int(eligible.sum()) * 3; nested_pairs.append((a,b,less+" -> "+more))
    return invariant_pairs, nested_pairs, dict(status="passed", invariance_target_pairs_checked=checked,
        nested_target_pairs_checked=nested_checked, tolerance=tolerance, clinical_invariance_claimed=False)


def group_weights(group_count, replicates=1000, seed=17):
    require(group_count > 0 and replicates > 0, "positive bootstrap population required")
    rng=np.random.default_rng(seed); weights=np.empty((replicates,group_count), dtype=np.float64)
    for i in range(replicates): weights[i]=np.bincount(rng.integers(0,group_count,group_count),minlength=group_count)
    return weights


def regression_draws(y, scores, group_ids, weights, keep):
    """One group scan, then matrix multiplication; includes zero-row groups."""
    count=np.bincount(group_ids[keep],minlength=weights.shape[1]).astype(float)
    error=scores[keep]-y[keep]
    sufficient=np.column_stack((count,
        np.bincount(group_ids[keep],weights=np.abs(error),minlength=len(count)),
        np.bincount(group_ids[keep],weights=error**2,minlength=len(count))))
    sampled=weights@sufficient
    draws=np.full((len(weights),2),np.nan)
    np.divide(sampled[:,1:],sampled[:,0,None],out=draws,where=sampled[:,0,None]>0)
    draws[:,1]=np.sqrt(draws[:,1]); return draws


def _points(y,s):
    raw=metrics(y,s,regression=True,curve=False)
    return {k:raw[k] for k in (*METRICS,"rows","bad_proxy_rows","good_proxy_rows")}


def review_accounting(y,s,eligible):
    available=eligible & np.isfinite(s); missing=eligible & ~available; retained=available & (s>=.9)
    reviewed=eligible & ~retained; bad=eligible & (y<.9); good=eligible & ~bad
    n=int(eligible.sum()); retained_n=int(retained.sum()); reviewed_n=int(reviewed.sum())
    return dict(eligible_queries=n, available_queries=int(available.sum()), automatic_review_for_missing_score=int(missing.sum()),
        retained_queries=retained_n, reviewed_queries=reviewed_n,
        score_coverage=float(available.sum()/n) if n else None, retained_coverage=retained_n/n if n else None,
        reviewed_fraction=reviewed_n/n if n else None, bad_retained=int((retained&bad).sum()),
        good_reviewed=int((reviewed&good).sum()), bad_reviewed=int((reviewed&bad).sum()),
        risk_among_retained=float((retained&bad).sum()/retained_n) if retained_n else None,
        recall_bad_for_review=float((reviewed&bad).sum()/bad.sum()) if bad.any() else None,
        review_is_geometric_coverage_proxy=True)


def scope_result(y,vectors,eligible,nominal_scope,group_ids,weights,common_all):
    eligible=eligible & nominal_scope; common=eligible & common_all
    result=dict(nominal_queries=int(nominal_scope.sum()), gt_eligible_queries=int(eligible.sum()),
        construction_unavailable_queries=int((nominal_scope & ~np.isfinite(y)).sum()), common_valid_queries=int(common.sum()),
        common_missing_queries=int(eligible.sum()-common.sum()), own_available={}, common={}, full_eligible_review={},
        paired_oa_frozen_minus_comparator={}, bootstrap=dict(replicates=len(weights),source_groups=weights.shape[1],
            zero_eligible_groups_included=True,seed=17,metrics=["mae","rmse"],ranking_intervals=False))
    draws={}
    for method,s in vectors.items():
        own=eligible & np.isfinite(s); point=_points(y[own],s[own]); commonpoint=_points(y[common],s[common])
        d=regression_draws(y,s,group_ids,weights,common); draws[method]=d
        own_d=d if np.array_equal(own,common) else regression_draws(y,s,group_ids,weights,own)
        result["own_available"][method]=dict(metrics=point,missing_queries=int(eligible.sum()-own.sum()),
            regression_ci={k:interval(own_d[:,j],len(weights)) for j,k in enumerate(("mae","rmse"))})
        result["common"][method]=dict(metrics=commonpoint,regression_ci={k:interval(d[:,j],len(weights)) for j,k in enumerate(("mae","rmse"))})
        result["full_eligible_review"][method]=review_accounting(y,s,eligible)
    ref=METHODS[0]
    for method in METHODS[1:]:
        result["paired_oa_frozen_minus_comparator"][method]={k:dict(point=(result["common"][ref]["metrics"][k]-result["common"][method]["metrics"][k]) if common.any() else None,
            **interval(draws[ref][:,j]-draws[method][:,j],len(weights))) for j,k in enumerate(("mae","rmse"))}
    return result


def analyze_regression(sources,inputs,targets,valid,scores,*,replicates=1000):
    require(tuple(scores)==METHODS,"all five declared methods required in fixed order")
    output=[]; macro_values={m:[] for m in METHODS}; source_groups={}
    for dataset in DATASETS:
        ix=np.asarray([i for i,s in enumerate(sources) if s["dataset"]==dataset],int)
        require(len(ix)>0,"all four declared datasets required")
        keys=sorted({sources[i]["split_group_id"] for i in ix}); lookup={k:i for i,k in enumerate(keys)}
        group=np.repeat([lookup[sources[i]["split_group_id"]] for i in ix],63); source_groups[dataset]=len(keys)
        weights=group_weights(len(keys),replicates=replicates)
        y=targets[ix].ravel().astype(float); eligible=np.repeat(valid[ix].ravel(),3)
        vectors={m:np.asarray(s[ix],float).ravel() for m,s in scores.items()}
        common=np.logical_and.reduce([np.isfinite(v) for v in vectors.values()])
        families=np.repeat(np.asarray([inputs[i*21+j]["family"] for i in ix for j in range(21)]),3)
        regions=np.tile(np.arange(3),len(ix)*21)
        for family in ("all",*FAMILIES):
            for region in ("all",*FINDINGS):
                base=np.ones(len(y),bool)
                if family!="all":base &= families==family
                if region!="all":base &= regions==FINDINGS.index(region)
                for bin_name in BINS:
                    subset=base.copy()
                    if bin_name=="[0,.5)":subset &= (y>=0)&(y<.5)
                    elif bin_name=="[.5,.9)":subset &= (y>=.5)&(y<.9)
                    elif bin_name=="[.9,1]":subset &= (y>=.9)&(y<=1)
                    r=scope_result(y,vectors,eligible,subset,group,weights,common)
                    r.update(dataset=dataset,family=family,region=region,retention_bin=bin_name,
                        base_nominal_queries=int(base.sum()),unknown_retention_bin_queries=int((base&~eligible).sum()))
                    output.append(r)
                    if family==region==bin_name=="all":
                        for m in METHODS:macro_values[m].append(r["common"][m]["metrics"]["mae"])
    macro={m:float(np.mean(v)) if all(x is not None for x in v) else None for m,v in macro_values.items()}
    ranks={m:1+sum(v is not None and v<macro[m] for v in macro.values()) if macro[m] is not None else None for m in METHODS}
    return dict(scopes=output,source_groups=source_groups,
        four_source_macro_mae=dict(values=macro,ranks=ranks,interval=None,scope="all five methods common valid; equal weight across four datasets"))


def pair_metrics(delta, left, right,kind):
    if kind=="invariance":
        return dict(signed_score_shift=float(np.mean(delta)) if len(delta) else None,
            absolute_score_shift=float(np.mean(abs(delta))) if len(delta) else None,
            review_threshold_flip_fraction=float(np.mean((left<.9)!=(right<.9))) if len(delta) else None)
    violation=delta>1e-6
    return dict(violation_fraction=float(np.mean(violation)) if len(delta) else None,
        mean_positive_increase=float(np.mean(np.maximum(delta,0))) if len(delta) else None,
        mean_increase_given_violation=float(np.mean(delta[violation])) if violation.any() else None,
        violations=int(violation.sum()))


def analyze_pairs(sources,valid,scores,pairs,kind):
    output=[]
    names=sorted({p[2] for p in pairs})
    for dataset in DATASETS:
        indices=np.asarray([i for i,s in enumerate(sources) if s["dataset"]==dataset])
        for group in ("all",*names):
            selected=[p for p in pairs if group=="all" or p[2]==group]
            for region in ("all",*FINDINGS):
                ri=list(range(3)) if region=="all" else [FINDINGS.index(region)]
                eligible=np.concatenate([np.repeat(valid[indices,a]&valid[indices,b],len(ri)) for a,b,_ in selected])
                vectors={m:(np.concatenate([arr[indices,a][:,ri].ravel() for a,b,_ in selected]),
                            np.concatenate([arr[indices,b][:,ri].ravel() for a,b,_ in selected])) for m,arr in scores.items()}
                common=eligible & np.logical_and.reduce([np.isfinite(a)&np.isfinite(b) for a,b in vectors.values()])
                row=dict(dataset=dataset,group=group,region=region,nominal_target_pairs=len(eligible),
                    gt_eligible_target_pairs=int(eligible.sum()),construction_unavailable_target_pairs=int((~eligible).sum()),
                    common_valid_target_pairs=int(common.sum()),intervals=False,methods={})
                for m,(left,right) in vectors.items():
                    own=eligible&np.isfinite(left)&np.isfinite(right)
                    row["methods"][m]=dict(own_valid_target_pairs=int(own.sum()),missing_eligible_target_pairs=int((eligible&~own).sum()),
                        own=pair_metrics(right[own]-left[own],left[own],right[own],kind),
                        common=pair_metrics(right[common]-left[common],left[common],right[common],kind))
                output.append(row)
    return output

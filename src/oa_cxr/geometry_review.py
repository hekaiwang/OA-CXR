"""Independent geometry-proxy review wording; the frozen editor is unchanged.

This module applies already-made decisions. It neither estimates anatomical
retention nor decides whether a clinical negative assertion is supportable.
"""
from __future__ import annotations

import hashlib
import math
import re

from .claims import _analyze
from .editing import _join_items, apply_edits as legacy_apply_edits

EDITOR_VERSION = "geometry-review-v1"
TEMPLATE = ("Relevant anatomical coverage may be incomplete; the negative statement "
            "about {phrase} has been withheld for image review.")
SCOPE = "Geometric-proxy experiment selected image review; clinical support is unverified."
TRIGGERS = frozenset({"predicted_anatomical_retention", "non_retention_ranking_budget",
                      "true_anatomical_retention_oracle", "missing_geometry_automatic_review"})


def text_sha256(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Report text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def review_text(phrase: str) -> str:
    """Call with a noun phrase returned by the unchanged conservative parser."""
    if not isinstance(phrase, str) or not phrase.strip():
        raise ValueError("A validated nonempty noun phrase is required")
    return TEMPLATE.format(phrase=phrase)


def _decisions(text, decisions):
    if not isinstance(decisions, list):
        raise TypeError("Decisions must be a list")
    claims, groups = _analyze(text)
    known = {c["claim_id"]: c for c in claims}
    normalized, selected, seen = [], {}, set()
    for d in decisions:
        if not isinstance(d, dict) or not isinstance(d.get("claim_id"), str):
            raise ValueError("Invalid decision identity")
        identifier = d["claim_id"]
        if identifier in seen or identifier not in known:
            raise ValueError("Duplicate, unknown or stale decision")
        seen.add(identifier)
        claim = known[identifier]
        if any(d.get(k) != v for k, v in claim.items()):
            raise ValueError("Claim metadata differs from the unchanged source report")
        if not claim["eligible"] or claim["polarity"] != "negative":
            raise ValueError("Only eligible explicit-negative decisions are accepted")
        if type(d.get("abstain")) is not bool or not isinstance(d.get("reason"), str) or not d["reason"].strip():
            raise ValueError("A frozen boolean decision and nonempty reason are required")
        score = d.get("proxy_retention_score", d.get("support_score"))
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))
                                  or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Proxy retention score must be None or finite in [0,1]")
        legacy = d.get("support_score")
        if legacy is not None and (isinstance(legacy, bool) or not isinstance(legacy, (int, float))
                                   or not math.isfinite(legacy) or not 0 <= legacy <= 1 or legacy != score):
            raise ValueError("Conflicting or invalid legacy score")
        trigger = d.get("trigger", "predicted_anatomical_retention")
        if not isinstance(trigger, str) or trigger not in TRIGGERS:
            raise ValueError("Unknown decision trigger/source")
        if trigger != "predicted_anatomical_retention" and score is not None:
            raise ValueError("Only a predicted-retention trigger can expose proxy_retention_score")
        normalized.append({**claim, "abstain": d["abstain"], "support_score": None, "reason": d["reason"]})
        if d["abstain"]:
            selected[claim["start"]] = (claim, score, d["reason"], trigger)
    # Reuse the frozen validation path, without monkeypatching its template or
    # changing its bytes. Every requested decision must survive its checks.
    old = legacy_apply_edits(text, normalized)
    wanted = {c["claim_id"] for c, _, _, _ in selected.values()}
    if old["skipped_edits"] or {e["claim_id"] for e in old["edits"]} != wanted:
        raise ValueError("Frozen editor did not accept exactly the declared decisions")
    return groups, selected


def _render(text, groups, selected):
    patches, edits = [], []
    for group in groups:
        removals = [item for item in group.items if item.target_start in selected]
        if not removals:
            continue
        remaining = [item.text for item in group.items if item.target_start not in selected]
        parts = []
        if remaining:
            if group.postfix:
                retained = "No " + _join_items(remaining)
            elif group.prefix.lower().startswith("neither"):
                retained = "No " + _join_items(remaining) + group.suffix
            else:
                retained = group.prefix + _join_items(remaining) + group.suffix
            if len(remaining) == 1 and group.suffix:
                plural = bool(re.search(r"\b(?:effusions|pneumothoraces|consolidations|fractures|abnormalities)$", remaining[0], re.I))
                retained = re.sub(r"\s+(?:is|are)\s+(?=seen|identified|present|evident|detected|visualized)",
                                  " are " if plural else " is ", retained, flags=re.I)
            parts.append(retained)
        for item in removals:
            statement = review_text(item.text)
            parts.append(statement[:-1] if not parts else statement[0].lower() + statement[1:-1])
        replacement = "; ".join(parts)
        patches.append((group.start, group.end, replacement))
        for item in removals:
            claim, score, reason, trigger = selected[item.target_start]
            edits.append({"claim_id": claim["claim_id"], "finding": claim["finding"],
                          "original_span": [claim["start"], claim["end"]],
                          "original_claim": claim["claim_text"], "action": "abstain",
                          "new_text": review_text(item.text), "support_score": None,
                          "proxy_retention_score": score, "reason": reason,
                          "replacement_span": [group.start, group.end],
                          "original_group": text[group.start:group.end], "replacement_text": replacement,
                          "editor_version": EDITOR_VERSION, "trigger": trigger,
                          "clinical_support_labels": False})
    cursor, chunks, unchanged = 0, [], 0
    for start, end, replacement in sorted(patches):
        if start < cursor:
            raise ValueError("Overlapping negative groups; no output produced")
        gap = text[cursor:start]
        chunks.extend((gap, replacement))
        unchanged += len(gap.encode("utf-8"))
        cursor = end
    chunks.append(text[cursor:])
    unchanged += len(text[cursor:].encode("utf-8"))
    return "".join(chunks), edits, {"outside_edited_groups_byte_identical": True,
        "outside_target_sentences_byte_identical": True, "changed_negative_groups": len(patches),
        "unchanged_utf8_bytes_outside_patches": unchanged}


def apply_geometry_review(text: str, decisions: list[dict]) -> dict:
    """Replay frozen decisions with geometry-only wording, failing on any invalid row.

    Decisions contain the complete unchanged ``extract_claims`` record plus
    ``abstain``, ``reason`` and optional ``proxy_retention_score``. Legacy
    ``support_score`` is accepted only as the same geometry score, never a
    clinical support probability. Partial decision lists are allowed here;
    the offline full-roster exporter requires one decision per eligible claim.
    Optional ``trigger`` must belong to TRIGGERS. Ranking-only, truth-oracle and
    missing-geometry triggers must leave proxy_retention_score null.
    """
    original_digest = text_sha256(text)
    groups, selected = _decisions(text, decisions)
    edited, edits, audit = _render(text, groups, selected)
    return {"edited_findings": edited, "edits": edits, "skipped_edits": [],
            "editor_version": EDITOR_VERSION, "clinical_support_labels": False, "scope": SCOPE,
            "original_findings_utf8_sha256": original_digest,
            "edited_findings_utf8_sha256": text_sha256(edited), "preservation_audit": audit}


def audit_geometry_review(original: str, result: dict, eligible_claims: list[dict] | None = None) -> dict:
    """Revalidate edits, noun lists, source bytes and the exact new template."""
    claims, _ = _analyze(original)
    eligible = [c for c in claims if c["eligible"]]
    if eligible_claims is not None and eligible_claims != eligible:
        raise ValueError("Audit claims differ from the unchanged source parser")
    known = {c["claim_id"]: c for c in eligible}
    if not isinstance(result, dict) or not isinstance(result.get("edits"), list):
        raise ValueError("Invalid geometry review output")
    decisions = []
    for e in result["edits"]:
        if not isinstance(e, dict) or e.get("claim_id") not in known:
            raise ValueError("Edited unknown or ineligible claim")
        decisions.append({**known[e["claim_id"]], "abstain": True,
                          "proxy_retention_score": e.get("proxy_retention_score"), "reason": e.get("reason"),
                          "trigger": e.get("trigger")})
    expected = apply_geometry_review(original, decisions)
    if result != expected:
        raise ValueError("Geometry-review result differs from exact source-bound replay")
    return {**expected["preservation_audit"], "editor_version": EDITOR_VERSION,
            "original_findings_utf8_sha256": text_sha256(original),
            "edited_findings_utf8_sha256": text_sha256(expected["edited_findings"]),
            "clinical_support_labels": False}

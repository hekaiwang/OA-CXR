"""Deterministic local abstention with revalidated spans and a complete audit."""

from __future__ import annotations

import math
import re
from collections import Counter

from .claims import _analyze


EDITOR_VERSION = "local-abstention-v1"


def _join_items(items: list[str], connector: str = "or") -> str:
    if len(items) < 2:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} {connector} {items[1]}"
    return ", ".join(items[:-1]) + f", {connector} " + items[-1]


def abstention_text(phrase: str) -> str:
    """Fixed uncertainty statement; ``phrase`` comes only from validated text."""
    return f"The current image does not support confident exclusion of {phrase}."


def apply_edits(text: str, decisions: list[dict]) -> dict:
    """Apply requested abstentions, preserving all other report content.

    Each decision contains the complete record from :func:`extract_claims`,
    ``abstain`` (bool), ``support_score`` (float or None), and ``reason`` (str).
    Decision spans/IDs/metadata are revalidated against the current report.
    Stale, duplicate, ambiguous, and invalid requests are audited and skipped.
    The only reconstruction occurs inside validated negative noun lists. Each
    unedited noun phrase is copied verbatim; unrelated clauses remain verbatim.
    """
    if not isinstance(decisions, list):
        raise TypeError("decisions must be a list")
    claims, groups = _analyze(text)
    by_id = {claim["claim_id"]: claim for claim in claims}
    counts = Counter(item.get("claim_id") for item in decisions if isinstance(item, dict) and isinstance(item.get("claim_id"), str))
    selected = {}
    skipped = []

    def skip(decision: object, reason: str) -> None:
        skipped.append({
            "claim_id": decision.get("claim_id") if isinstance(decision, dict) else None,
            "reason": reason,
        })

    for decision in decisions:
        if not isinstance(decision, dict):
            skip(decision, "invalid_decision")
            continue
        if type(decision.get("abstain")) is not bool:
            skip(decision, "invalid_abstain")
            continue
        if not decision["abstain"]:
            continue
        claim_id = decision.get("claim_id")
        if not isinstance(claim_id, str) or claim_id not in by_id:
            skip(decision, "unknown_or_stale_claim")
            continue
        if counts[claim_id] != 1:
            skip(decision, "duplicate_decision")
            continue
        claim = by_id[claim_id]
        if any(decision.get(key) != value for key, value in claim.items()):
            skip(decision, "claim_metadata_mismatch")
            continue
        if not claim["eligible"] or claim["polarity"] != "negative":
            skip(decision, claim["skip_reason"] or "ineligible_claim")
            continue
        score = decision.get("support_score")
        if score is not None and (isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1):
            skip(decision, "invalid_support_score")
            continue
        if not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
            skip(decision, "missing_reason")
            continue
        selected[claim["start"]] = (claim, decision)

    edits = []
    patches = []
    for group in groups:
        removals = [item for item in group.items if item.target_start in selected]
        if not removals:
            continue
        remaining = [item.text for item in group.items if item.target_start not in selected]
        parts = []
        if remaining:
            if group.postfix:
                # Normalizing a negative noun list avoids carrying incorrect
                # plural verb agreement after removing a conjunction.
                retained = "No " + _join_items(remaining)
            elif group.prefix.lower().startswith("neither"):
                retained = "No " + _join_items(remaining) + group.suffix
            else:
                retained = group.prefix + _join_items(remaining) + group.suffix
            # A surviving single noun may need copula agreement. This changes
            # grammar only and does not drop any non-target noun phrase.
            if len(remaining) == 1 and group.suffix:
                plural = bool(re.search(r"\b(?:effusions|pneumothoraces|consolidations|fractures|abnormalities)$", remaining[0], re.IGNORECASE))
                retained = re.sub(r"\s+(?:is|are)\s+(?=seen|identified|present|evident|detected|visualized)", " are " if plural else " is ", retained, flags=re.IGNORECASE)
            parts.append(retained)
        for item in removals:
            statement = abstention_text(item.text)
            parts.append(statement[:-1] if not parts else statement[0].lower() + statement[1:-1])
        replacement = "; ".join(parts)
        # Keep the original final punctuation/delimiter outside this patch.
        # Fragments without punctuation remain fragments in the output.
        patches.append((group.start, group.end, replacement))
        for item in removals:
            claim, decision = selected[item.target_start]
            edits.append({
                "claim_id": claim["claim_id"],
                "finding": claim["finding"],
                "original_span": [claim["start"], claim["end"]],
                "original_claim": claim["claim_text"],
                "action": "abstain",
                "new_text": abstention_text(item.text),
                "support_score": decision.get("support_score"),
                "reason": decision["reason"],
                "replacement_span": [group.start, group.end],
                "original_group": text[group.start:group.end],
                "replacement_text": replacement,
                "editor_version": EDITOR_VERSION,
            })
    edited = text
    previous_start = len(text) + 1
    for start, end, replacement in sorted(patches, reverse=True):
        if end > previous_start:
            raise RuntimeError("Overlapping edit groups; no output produced")
        edited = edited[:start] + replacement + edited[end:]
        previous_start = start
    return {"edited_findings": edited, "edits": edits, "skipped_edits": skipped}

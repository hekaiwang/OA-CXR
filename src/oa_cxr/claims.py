"""Conservative, auditable extraction of a deliberately small CXR vocabulary.

This is a rule-based research parser, not a clinical NLP model. Every lexical
target mention is returned, but automatic editing is restricted to validated
explicit-negative noun lists. Unknown syntax is retained with a skip reason.
Offsets are Python string offsets into the unmodified input (not byte offsets).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


PARSER_VERSION = "negative-claims-v2"
FINDINGS = ("pleural_effusion", "pneumothorax", "consolidation")

# Bare "effusion" and "opacity" are deliberately not aliases: their anatomical
# or diagnostic meaning can differ from the three registered targets.
_TARGET = re.compile(
    r"\b(?P<pleural_effusion>pleural\s+effusions?)\b"
    r"|\b(?P<pneumothorax>pneumothora(?:x|ces))\b"
    r"|\b(?P<consolidation>consolidations?)\b",
    re.IGNORECASE,
)
_PREFIX = re.compile(
    r"^(?:(?:there\s+(?:is|are)\s+)?no\s+"
    r"(?:(?:radiographic\s+)?evidence\s+of\s+)?"
    r"|without\s+|negative\s+for\s+|neither\s+)", re.IGNORECASE,
)
_SUFFIX = re.compile(
    r"\s+(?:is|are)\s+(?:seen|identified|present|evident|detected|visualized)$",
    re.IGNORECASE,
)
_POST_NEGATIVE = re.compile(
    r"\s+(?:is|are)\s+(?:absent|not\s+(?:seen|identified|present|evident|"
    r"detected|visualized))$", re.IGNORECASE,
)
_UNCERTAIN = re.compile(
    r"\b(?:cannot|can\s+not|can't|could\s+not|unable\s+to)\s+"
    r"(?:exclude|be\s+excluded|rule\s+out|be\s+ruled\s+out|assess|evaluate)\b|\b(?:possible|possibly|probable|"
    r"probably|questionable|equivocal|suspected|suspicious|may|might|could)\b|"
    r"\b(?:difficult|hard)\s+to\s+exclude\b|\bnot\s+excluded\b|"
    r"\b(?:limited|suboptimal|obscured)\b|"
    r"\bdoes\s+not\s+support\s+confident\s+exclusion\s+of\b", re.IGNORECASE,
)
_HISTORY = re.compile(r"\b(?:history\s+of|historical|previous(?:ly)?|prior)\b", re.IGNORECASE)
_RESOLVED = re.compile(r"\b(?:resolved|resolution|resolving|cleared)\b", re.IGNORECASE)
_COMPARISON = re.compile(
    r"\b(?:interval|unchanged|stable|increased|decreased|improved|improving|"
    r"worsened|worsening|persistent|persisting|compared|comparison|new)\b|"
    r"\bno\s+(?:significant\s+)?change\b", re.IGNORECASE,
)
_QUALIFIED = re.compile(
    r"\b(?:large|small|moderate|trace|tiny|significant|sizable|sizeable|"
    r"substantial|definite|definitely|convincing|obvious|gross|acute)\b",
    re.IGNORECASE,
)
_CLAUSE_BREAK = re.compile(r";|,?\s+\b(?:but|however|yet)\b\s*", re.IGNORECASE)
_LIST_BREAK = re.compile(r"\s*,\s*(?:(?:and|or|nor)\s+)?|\s+(?:and|or|nor)\s+", re.IGNORECASE)
_LOCATION_WORDS = {
    "left", "right", "bilateral", "bilaterally", "apical", "biapical",
    "basal", "basilar", "bibasilar", "upper", "lower", "mid", "middle",
    "lung", "lungs", "zone", "zones", "lobe", "lobes",
}
_TARGET_MODIFIERS = {
    "pleural_effusion": _LOCATION_WORDS,
    "pneumothorax": _LOCATION_WORDS,
    "consolidation": _LOCATION_WORDS | {"focal", "airspace", "lobar"},
}
_NON_TARGET = re.compile(
    r"(?:(?:left|right|bilateral|bibasilar|basilar|focal)\s+)*"
    r"(?:pulmonary\s+edema|edema|atelectasis|pulmonary\s+vascular\s+congestion|"
    r"vascular\s+congestion|pulmonary\s+congestion|fractures?|"
    r"osseous\s+abnormalit(?:y|ies))", re.IGNORECASE,
)


@dataclass(frozen=True)
class _Item:
    start: int
    end: int
    text: str
    target_start: int | None


@dataclass(frozen=True)
class _NegativeGroup:
    start: int
    end: int
    prefix: str
    suffix: str
    items: tuple[_Item, ...]
    postfix: bool = False


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _sentences(text: str) -> list[tuple[int, int]]:
    """Punctuation/newline segmentation; punctuation is kept in sentence spans."""
    result = []
    start = 0
    for boundary in re.finditer(r"[.!?]+|\r?\n", text):
        end = boundary.end() if text[boundary.start()] in ".!?" else boundary.start()
        a, b = _trim(text, start, end)
        if a < b:
            result.append((a, b))
        start = boundary.end()
    a, b = _trim(text, start, len(text))
    if a < b:
        result.append((a, b))
    return result


def _clauses(text: str, start: int, end: int) -> list[tuple[int, int]]:
    while end > start and text[end - 1] in ".!?":
        end -= 1
    result = []
    cursor = start
    for boundary in _CLAUSE_BREAK.finditer(text, start, end):
        a, b = _trim(text, cursor, boundary.start())
        if a < b:
            result.append((a, b))
        cursor = boundary.end()
    a, b = _trim(text, cursor, end)
    if a < b:
        result.append((a, b))
    return result


def _side_and_region(phrase: str) -> tuple[str, str]:
    words = set(re.findall(r"[a-z]+", phrase.lower()))
    if words & {"bilateral", "bilaterally", "bibasilar", "biapical"} or {"left", "right"} <= words:
        side = "bilateral"
    elif "left" in words:
        side = "left"
    elif "right" in words:
        side = "right"
    else:
        side = "unspecified"
    regions = []
    if words & {"apical", "biapical"}:
        regions.append("apical")
    if words & {"basal", "basilar", "bibasilar"}:
        regions.append("basilar")
    for region in ("upper", "middle", "lower"):
        if region in words or (region == "middle" and "mid" in words):
            regions.append(region)
    return side, "+".join(regions) or "unspecified"


def _item(text: str, start: int, end: int) -> _Item | None:
    start, end = _trim(text, start, end)
    if start == end:
        return None
    phrase = text[start:end]
    targets = list(_TARGET.finditer(phrase))
    if len(targets) == 1:
        target = targets[0]
        # Modifiers must precede the noun. Relative clauses, verbs, postposed
        # anatomy, parentheses, and non-registered modifiers are unvalidated.
        if phrase[target.end():].strip():
            return None
        modifier = phrase[:target.start()]
        if re.search(r"[^A-Za-z\s]", modifier):
            return None
        words = set(modifier.lower().split())
        if not words <= _TARGET_MODIFIERS[target.lastgroup]:
            return None
        return _Item(start, end, phrase, start + target.start())
    if not targets and _NON_TARGET.fullmatch(phrase):
        return _Item(start, end, phrase, None)
    return None


def _parse_negative_group(text: str, start: int, end: int) -> tuple[_NegativeGroup | None, str | None]:
    clause = text[start:end]
    prefix_match = _PREFIX.match(clause)
    postfix = False
    suffix = ""
    body_start = start
    body_end = end
    if prefix_match:
        prefix = prefix_match.group()
        body_start += prefix_match.end()
        suffix_match = _SUFFIX.search(text[body_start:end])
        if suffix_match:
            body_end = body_start + suffix_match.start()
            suffix = text[body_end:end]
    else:
        suffix_match = _POST_NEGATIVE.search(clause)
        if not suffix_match:
            return None, None
        prefix = ""
        postfix = True
        body_end = start + suffix_match.start()
        suffix = text[body_end:end]
    items = []
    cursor = body_start
    delimiters = list(_LIST_BREAK.finditer(text, body_start, body_end))
    # With a postfix predicate, "X or Y is absent" does not establish that
    # both X and Y are absent. This differs from the explicit "No X or Y"
    # exclusion list; do not distribute that predicate over a disjunction.
    if postfix and any(re.search(r"\b(?:or|nor)\b", delimiter.group(), re.IGNORECASE) for delimiter in delimiters):
        return None, "ambiguous_scope"
    if not postfix and any(re.search(r"\band\b", delimiter.group(), re.IGNORECASE) for delimiter in delimiters):
        return None, "ambiguous_scope"
    # "No X and Y is present" can mean a negative X and a positive Y.
    # A terminal predicate is shared only in the explicit exclusion-list
    # construction "no X or Y is seen"; ambiguous commas/and are not edited.
    if not postfix and suffix and delimiters and not re.search(r"\b(?:or|nor)\b", delimiters[-1].group(), re.IGNORECASE):
        return None, "ambiguous_scope"
    for delimiter in delimiters:
        item = _item(text, cursor, delimiter.start())
        if item is None:
            return None, "ambiguous_scope"
        items.append(item)
        cursor = delimiter.end()
    item = _item(text, cursor, body_end)
    if item is None:
        return None, "ambiguous_scope"
    items.append(item)
    if prefix.lower().startswith("neither") and not re.search(r"\bnor\b", clause, re.IGNORECASE):
        return None, "ambiguous_scope"
    if re.search(r"\b(?:and|or)\b", clause, re.IGNORECASE) and re.search(r"\bnor\b", clause, re.IGNORECASE):
        return None, "ambiguous_scope"
    # A leading anatomical modifier could be shared by later conjunctions.
    # Do not silently broaden "no left X or Y" to an unqualified negative Y.
    for index, item in enumerate(items[:-1]):
        if _side_and_region(item.text) != ("unspecified", "unspecified"):
            for later in items[index + 1:]:
                if _side_and_region(later.text) == ("unspecified", "unspecified"):
                    return None, "ambiguous_modifier_scope"
        # "No focal consolidation or atelectasis" may share "focal".
        # Removing consolidation must not broaden the other negative fact.
        for modifier in ("focal", "lobar", "airspace"):
            if re.search(rf"\b{modifier}\b", item.text, re.IGNORECASE):
                for later in items[index + 1:]:
                    compatible = r"\b(?:consolidation|consolidations|atelectasis|edema)\b" if modifier == "focal" else r"\b(?:consolidation|consolidations|atelectasis)\b"
                    if re.search(compatible, later.text, re.IGNORECASE) and not re.search(rf"\b{modifier}\b", later.text, re.IGNORECASE):
                        return None, "ambiguous_modifier_scope"
    return _NegativeGroup(start, end, prefix, suffix, tuple(items), postfix), None


def _claim_id(text: str, start: int, end: int, finding: str) -> str:
    payload = f"{PARSER_VERSION}\0{text}\0{start}:{end}:{finding}".encode("utf-8")
    return "claim_" + hashlib.sha256(payload).hexdigest()[:24]


def _analyze(text: str) -> tuple[list[dict], list[_NegativeGroup]]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    claims = []
    groups = []
    for sentence_start, sentence_end in _sentences(text):
        sentence = text[sentence_start:sentence_end]
        for clause_start, clause_end in _clauses(text, sentence_start, sentence_end):
            clause = text[clause_start:clause_end]
            mentions = list(_TARGET.finditer(text, clause_start, clause_end))
            if not mentions:
                continue
            group, parse_reason = _parse_negative_group(text, clause_start, clause_end)
            # Temporal and uncertainty cues are conservatively sentence-wide:
            # clauses can refer to each other even across conjunctions.
            if _HISTORY.search(sentence):
                hazard = "history"
            elif _RESOLVED.search(sentence):
                hazard = "resolved"
            elif _COMPARISON.search(sentence):
                hazard = "comparison"
            elif _UNCERTAIN.search(sentence):
                hazard = "uncertain"
            elif _QUALIFIED.search(clause):
                hazard = "qualified_assertion"
            else:
                hazard = None
            if group is not None and hazard is None:
                groups.append(group)
            for mention in mentions:
                if _UNCERTAIN.search(clause):
                    polarity = "uncertain"
                elif group is not None:
                    polarity = "negative"
                elif _COMPARISON.search(clause) or _RESOLVED.search(clause):
                    polarity = "uncertain"
                elif _PREFIX.match(clause) or _POST_NEGATIVE.search(clause):
                    # A leading negative cue with an unvalidated scope cannot
                    # safely label every later noun as a negative assertion.
                    polarity = "uncertain"
                else:
                    polarity = "positive"
                eligible = group is not None and hazard is None
                reason = None if eligible else hazard or parse_reason or (
                    "not_explicit_negative" if polarity == "positive" else "ambiguous_scope"
                )
                # Only use this item, not another finding's laterality. For
                # rejected clauses, inspect the immediately preceding segment.
                target_item = next((item for item in group.items if item.target_start == mention.start()), None) if group else None
                if target_item:
                    local_phrase = target_item.text
                else:
                    preceding = text[clause_start:mention.start()]
                    preceding = re.split(r",|\b(?:and|or|nor)\b", preceding, flags=re.IGNORECASE)[-1]
                    local_phrase = preceding + mention.group()
                side, region = _side_and_region(local_phrase)
                claims.append({
                    "claim_id": _claim_id(text, mention.start(), mention.end(), mention.lastgroup),
                    "finding": mention.lastgroup,
                    "start": mention.start(),
                    "end": mention.end(),
                    "claim_text": mention.group(),
                    "sentence_start": sentence_start,
                    "sentence_end": sentence_end,
                    "sentence_text": sentence,
                    "polarity": polarity,
                    "eligible": eligible,
                    "skip_reason": reason,
                    "laterality": side,
                    "region": region,
                })
    return claims, groups


def extract_claims(text: str) -> list[dict]:
    """Return every registered finding mention and its editing eligibility.

    An empty result means no registered lexical mention, never a normal exam.
    A ``negative`` polarity does not imply eligibility: unknown syntax,
    comparisons, uncertainty, and qualified exclusions are explicitly skipped.
    Bare effusion/opacity aliases and general clinical assertion resolution are
    outside this parser's deliberately restricted vocabulary.
    """
    return _analyze(text)[0]

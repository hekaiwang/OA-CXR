"""Clinical-language safety cases; no models, datasets, or GPU are needed."""

import unittest

from oa_cxr.claims import extract_claims
from oa_cxr.editing import apply_edits


def decisions_for(text, findings=None):
    return [dict(claim, abstain=findings is None or claim["finding"] in findings,
                 support_score=0.15, reason="Test support score below fixed threshold")
            for claim in extract_claims(text)]


class ClaimsTests(unittest.TestCase):
    def test_compound_negation_is_two_distinct_claims(self):
        text = "No pleural effusion or pneumothorax."
        claims = extract_claims(text)
        self.assertEqual([x["finding"] for x in claims], ["pleural_effusion", "pneumothorax"])
        self.assertTrue(all(x["eligible"] and x["polarity"] == "negative" for x in claims))
        for claim in claims:
            self.assertEqual(text[claim["start"]:claim["end"]], claim["claim_text"])
            self.assertEqual(text[claim["sentence_start"]:claim["sentence_end"]], claim["sentence_text"])
        self.assertNotEqual(claims[0]["claim_id"], claims[1]["claim_id"])
        self.assertEqual(claims, extract_claims(text))

    def test_prefix_suffix_and_non_target_negative(self):
        for text in [
            "There is no evidence of consolidation, pulmonary edema, pleural effusion, or pneumothorax.",
            "No focal consolidation, pleural effusion, or pneumothorax.",
            "No pleural effusions or pneumothorax are identified.",
            "Neither pleural effusion nor pneumothorax is seen.",
            "Pleural effusion and pneumothorax are absent.",
            "Without pleural effusion or pneumothorax.",
        ]:
            with self.subTest(text=text):
                self.assertTrue(all(x["eligible"] for x in extract_claims(text)))

    def test_laterality_and_region_do_not_leak_between_items(self):
        text = "No left pleural effusion or right apical pneumothorax. No bibasilar consolidation."
        claims = extract_claims(text)
        self.assertEqual([(x["laterality"], x["region"]) for x in claims],
                         [("left", "unspecified"), ("right", "apical"), ("bilateral", "basilar")])
        self.assertTrue(all(x["eligible"] for x in claims))

    def test_preserve_difficult_mentions_as_ineligible(self):
        cases = {
            "History of pneumothorax.": "history",
            "The pleural effusion has resolved.": "resolved",
            "Cannot exclude consolidation.": "uncertain",
            "Pleural effusion is not excluded.": "uncertain",
            "No interval change in pleural effusion.": "comparison",
            "No new pneumothorax.": "comparison",
            "No significant pleural effusion.": "qualified_assertion",
            "No large pneumothorax.": "qualified_assertion",
            "No pleural effusion with adjacent consolidation.": "ambiguous_scope",
            "No left pleural effusion or pneumothorax.": "ambiguous_modifier_scope",
            "No focal consolidation or atelectasis.": "ambiguous_modifier_scope",
            "No airspace consolidation or atelectasis.": "ambiguous_modifier_scope",
            "No pleural effusion and pneumothorax.": "ambiguous_scope",
            "No pleural effusion or pneumothorax and a left basilar opacity is present.": "ambiguous_scope",
            "Pneumothorax is present.": "not_explicit_negative",
        }
        for text, reason in cases.items():
            with self.subTest(text=text):
                claims = extract_claims(text)
                self.assertGreater(len(claims), 0)
                self.assertTrue(all(not x["eligible"] for x in claims))
                self.assertTrue(all(x["skip_reason"] == reason for x in claims))
                self.assertEqual(apply_edits(text, decisions_for(text))["edited_findings"], text)

    def test_uncertainty_is_not_explicit_negative(self):
        for text in ["Cannot exclude pneumothorax.", "Pneumothorax cannot be excluded.", "Possible pleural effusion.", "Consolidation may be present."]:
            self.assertEqual(extract_claims(text)[0]["polarity"], "uncertain")

    def test_postfix_disjunction_does_not_establish_each_negative(self):
        for text in ["Pleural effusion or pneumothorax is absent.",
                     "Pleural effusion or pneumothorax is not seen.",
                     "Consolidation, pleural effusion, or pneumothorax are absent."]:
            with self.subTest(text=text):
                claims = extract_claims(text)
                self.assertTrue(claims)
                self.assertTrue(all(not claim["eligible"] for claim in claims))
                self.assertTrue(all(claim["skip_reason"] == "ambiguous_scope" for claim in claims))
                self.assertEqual(apply_edits(text, decisions_for(text))["edited_findings"], text)

    def test_ambiguous_negative_scope_does_not_label_positive_as_negative(self):
        for text in ["No pleural effusion and consolidation is present.",
                     "No pleural effusion, consolidation is present.",
                     "No pleural effusion or pneumothorax, edema is present."]:
            with self.subTest(text=text):
                claims = extract_claims(text)
                self.assertTrue(all(not x["eligible"] for x in claims))
                self.assertTrue(all(x["polarity"] == "uncertain" for x in claims))
                self.assertEqual(apply_edits(text, decisions_for(text))["edited_findings"], text)

    def test_negation_modifiers_and_temporal_states_are_not_dropped(self):
        for text in ["No large pleural effusion.", "No acute consolidation.",
                     "No definite pneumothorax.", "No trace pleural effusion.",
                     "No change in pneumothorax.", "Previously seen pneumothorax has resolved.",
                     "No pneumothorax is visible in the imaged right apex.",
                     "No pleural effusion, although consolidation cannot be excluded.",
                     "No right or left pleural effusion."]:
            with self.subTest(text=text):
                claims = extract_claims(text)
                self.assertTrue(claims)
                self.assertTrue(all(not x["eligible"] for x in claims))
                self.assertEqual(apply_edits(text, decisions_for(text))["edited_findings"], text)

    def test_positive_mentions_and_offsets_with_unicode(self):
        text = "  FINDINGS:\n  α Small left pleural effusion.\nNo pneumothorax!  "
        claims = extract_claims(text)
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0]["polarity"], "positive")
        self.assertEqual(claims[0]["laterality"], "left")
        self.assertFalse(claims[0]["eligible"])
        for claim in claims:
            self.assertEqual(text[claim["start"]:claim["end"]], claim["claim_text"])
        self.assertTrue(claims[1]["eligible"])

    def test_no_vocabulary_match_is_not_a_normal_exam(self):
        self.assertEqual(extract_claims("No pericardial effusion. Edema is present."), [])
        self.assertEqual(extract_claims("No effusion or opacity."), [])
        self.assertEqual(extract_claims(""), [])
        with self.assertRaises(TypeError):
            extract_claims(None)


class EditingTests(unittest.TestCase):
    def test_partial_compound_edit_keeps_other_explicit_negative(self):
        text = "No pleural effusion or pneumothorax."
        result = apply_edits(text, decisions_for(text, {"pleural_effusion"}))
        self.assertEqual(result["edited_findings"],
                         "No pneumothorax; the current image does not support confident exclusion of pleural effusion.")
        self.assertEqual(len(result["edits"]), 1)
        self.assertEqual(result["skipped_edits"], [])
        self.assertEqual(result["edits"][0]["original_claim"], "pleural effusion")

    def test_full_compound_edit_removes_both_negative_assertions(self):
        text = "No pleural effusion or pneumothorax."
        result = apply_edits(text, decisions_for(text))
        self.assertEqual(len(result["edits"]), 2)
        self.assertNotIn("No pleural", result["edited_findings"])
        self.assertNotIn("No pneumothorax", result["edited_findings"])
        self.assertIn("exclusion of pleural effusion", result["edited_findings"])
        self.assertIn("exclusion of pneumothorax", result["edited_findings"])
        reparsed = extract_claims(result["edited_findings"])
        self.assertTrue(all(x["polarity"] == "uncertain" and not x["eligible"] for x in reparsed))

    def test_non_target_negative_and_positive_sentences_are_preserved(self):
        positive = "Cardiomegaly and right basilar atelectasis are present."
        text = positive + "\nNo consolidation, pulmonary edema, pleural effusion, or pneumothorax.\nThe left chest tube is unchanged."
        result = apply_edits(text, decisions_for(text, {"pleural_effusion", "pneumothorax"}))
        self.assertTrue(result["edited_findings"].startswith(positive + "\n"))
        self.assertTrue(result["edited_findings"].endswith("\nThe left chest tube is unchanged."))
        self.assertIn("No consolidation or pulmonary edema;", result["edited_findings"])
        self.assertEqual(len(result["edits"]), 2)

    def test_mixed_positive_clause_verbatim_preservation(self):
        for delimiter in [", but ", "; "]:
            positive = "left basilar atelectasis is present"
            text = "No pleural effusion or pneumothorax" + delimiter + positive + "."
            result = apply_edits(text, decisions_for(text, {"pleural_effusion"}))
            self.assertEqual(len(result["edits"]), 1)
            self.assertIn("No pneumothorax;", result["edited_findings"])
            self.assertTrue(result["edited_findings"].endswith(delimiter + positive + "."))

    def test_target_positive_in_same_sentence_is_not_edited(self):
        text = "No pneumothorax, but small left pleural effusion is present."
        original = extract_claims(text)
        self.assertTrue(original[0]["eligible"])
        self.assertFalse(original[1]["eligible"])
        self.assertEqual(original[1]["polarity"], "positive")
        result = apply_edits(text, decisions_for(text))
        self.assertEqual([x["finding"] for x in result["edits"]], ["pneumothorax"])
        self.assertTrue(result["edited_findings"].endswith(", but small left pleural effusion is present."))

    def test_non_target_tail_predicate_is_never_deleted(self):
        for noun in ["edema", "atelectasis", "pulmonary vascular congestion"]:
            text = "No pleural effusion or pneumothorax, " + noun + " is present."
            result = apply_edits(text, decisions_for(text))
            self.assertEqual(result["edits"], [])
            self.assertEqual(result["edited_findings"], text)

    def test_laterality_and_modifiers_are_in_abstention(self):
        text = "No left pleural effusion or right apical pneumothorax."
        result = apply_edits(text, decisions_for(text, {"pneumothorax"}))
        self.assertIn("No left pleural effusion;", result["edited_findings"])
        self.assertIn("exclusion of right apical pneumothorax", result["edited_findings"])

    def test_no_decisions_and_supported_decisions_leave_exact_text(self):
        text = "\n No pleural effusion or pneumothorax.  "
        self.assertEqual(apply_edits(text, [])["edited_findings"], text)
        self.assertEqual(apply_edits(text, decisions_for(text, set()))["edited_findings"], text)

    def test_stale_or_modified_metadata_cannot_edit_report(self):
        text = "No pneumothorax."
        original = decisions_for(text)
        result = apply_edits("No pneumothorax. Cardiomegaly.", original)
        self.assertEqual(result["edits"], [])
        self.assertEqual(result["skipped_edits"][0]["reason"], "unknown_or_stale_claim")
        for key, value in [("start", 0), ("claim_text", "pleural effusion"), ("laterality", "right"), ("eligible", False)]:
            with self.subTest(key=key):
                decision = dict(original[0], **{key: value})
                result = apply_edits(text, [decision])
                self.assertEqual(result["edited_findings"], text)
                self.assertEqual(result["skipped_edits"][0]["reason"], "claim_metadata_mismatch")

    def test_duplicate_and_invalid_decisions_are_skipped(self):
        text = "No pneumothorax."
        decision = decisions_for(text)[0]
        result = apply_edits(text, [decision, decision])
        self.assertEqual(result["edited_findings"], text)
        self.assertTrue(all(x["reason"] == "duplicate_decision" for x in result["skipped_edits"]))
        for fields in [{"support_score": float("nan")}, {"support_score": 2}, {"support_score": True}, {"abstain": "true"}, {"reason": ""}]:
            with self.subTest(fields=fields):
                result = apply_edits(text, [dict(decision, **fields)])
                self.assertEqual(result["edits"], [])
                self.assertEqual(result["edited_findings"], text)

    def test_postfix_negative_has_safe_partial_edit(self):
        text = "Pleural effusion and pneumothorax are absent."
        result = apply_edits(text, decisions_for(text, {"pleural_effusion"}))
        self.assertTrue(result["edited_findings"].startswith("No pneumothorax;"))
        self.assertEqual(len(result["edits"]), 1)

    def test_audit_spans_reconstruct_report(self):
        text = "No pneumothorax. Heart is enlarged. No pleural effusion."
        result = apply_edits(text, decisions_for(text))
        reconstructed = text
        patches = {(tuple(x["replacement_span"]), x["replacement_text"]) for x in result["edits"]}
        for (start, end), replacement in sorted(patches, reverse=True):
            reconstructed = reconstructed[:start] + replacement + reconstructed[end:]
        self.assertEqual(reconstructed, result["edited_findings"])
        self.assertIn(". Heart is enlarged. ", reconstructed)


if __name__ == "__main__":
    unittest.main()

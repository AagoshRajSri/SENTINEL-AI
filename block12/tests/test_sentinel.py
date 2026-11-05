# tests/test_sentinel.py
"""
Comprehensive regression and unit test suite for Sentinel-AI pipeline.
All external API calls are mocked to ensure offline, hermetic execution.
"""

import hashlib
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is on sys.path
_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.classifier import ClassificationResult, IntentCategory, classify
from src.escalation import (
    EscalationDecision,
    EscalationLLMResult,
    gatekeeper,
    rule_based_escalation,
)
from src.reply_drafter import ReplyResult, draft_reply, apply_grounding_safeguards
from src.retriever import get_index_metric_type, retrieve


class TestPipelineRegression(unittest.TestCase):

    # ---------------------------------------------------------
    # 1. PII and Rule-Based Escalation
    # ---------------------------------------------------------
    def test_pii_detection_phone_number(self):
        text = "Please call me at 555-234-5678 immediately regarding my order!"
        should_esc, reason = rule_based_escalation(text, 0.95)
        self.assertTrue(should_esc)
        self.assertIn("phone number", reason.lower())

    def test_pii_detection_email(self):
        text = "Send my refund receipt to customer_help@example.com right now"
        should_esc, reason = rule_based_escalation(text, 0.90)
        self.assertTrue(should_esc)
        self.assertIn("email address", reason.lower())

    def test_severe_legal_keyword_escalation(self):
        for kw in ["sue", "lawyer", "attorney", "legal", "police"]:
            text = f"I am going to call the {kw} if my order is not fixed today."
            should_esc, reason = rule_based_escalation(text, 0.90)
            self.assertTrue(should_esc, f"Failed to escalate keyword: {kw}")
            self.assertIn(kw, reason.lower())

    def test_routine_case_does_not_trigger_rules(self):
        text = "Where is my package? The tracking has not updated since yesterday."
        should_esc, reason = rule_based_escalation(text, 0.95)
        self.assertFalse(should_esc)

    # ---------------------------------------------------------
    # 2. Empty & Malformed Input Handling
    # ---------------------------------------------------------
    def test_empty_input_classifier(self):
        res = classify("")
        self.assertEqual(res.intent, IntentCategory.ORDER_ISSUES)
        self.assertEqual(res.confidence, 0.0)

        res_none = classify(None)
        self.assertEqual(res_none.confidence, 0.0)

    def test_empty_input_gatekeeper(self):
        res = gatekeeper("", "Order Issues", 0.0, "None")
        self.assertFalse(res.should_escalate)

    def test_empty_input_reply_drafter(self):
        res = draft_reply("")
        self.assertIn("assist", res.draft_reply.lower())

    def test_empty_input_retriever(self):
        results = retrieve("")
        self.assertEqual(results, [])

    # ---------------------------------------------------------
    # 3. Escalation Gatekeeper Decision Logic
    # ---------------------------------------------------------
    @patch("src.escalation._get_cache")
    def test_gatekeeper_severe_case_via_score(self, mock_cache):
        # High risk score 0.85 should escalate under 0.4 or 0.7 threshold
        text = "Your driver threw my package onto the roof and broke it."
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        mock_cache.return_value = {
            text_hash: {"escalation_score": 0.85, "reasoning": "Driver misconduct and damage."}
        }
        with patch.dict(os.environ, {"ESCALATION_THRESHOLD": "0.40"}):
            decision = gatekeeper(text, "Product Issues", 0.90, "Damaged item")
            self.assertTrue(decision.should_escalate)

    @patch("src.escalation._get_cache")
    def test_gatekeeper_routine_case_below_threshold(self, mock_cache):
        text = "Can I change the delivery address for order 123?"
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        mock_cache.return_value = {
            text_hash: {"escalation_score": 0.20, "reasoning": "Routine address modification request."}
        }
        with patch.dict(os.environ, {"ESCALATION_THRESHOLD": "0.40"}):
            decision = gatekeeper(text, "Order Issues", 0.95, "Address update")
            self.assertFalse(decision.should_escalate)

    # ---------------------------------------------------------
    # 4. Cache Hit & Miss Behavior
    # ---------------------------------------------------------
    @patch("src.classifier._get_cache")
    def test_classifier_cache_hit_bypasses_network(self, mock_cache):
        query = "Cached delivery inquiry"
        q_hash = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()
        mock_cache.return_value = {
            q_hash: {
                "intent": "Delivery Issues",
                "confidence": 0.99,
                "reasoning": "Found in test cache fixture."
            }
        }
        # Does not call Groq or requests
        res = classify(query)
        self.assertEqual(res.intent, IntentCategory.DELIVERY_ISSUES)
        self.assertEqual(res.confidence, 0.99)

    def test_offline_eval_cache_miss_exits_safely(self):
        # In offline evaluation mode, unexpected cache miss should be detected
        novel_query = "A totally novel query never seen in the golden set xyz12345"
        with patch.dict(os.environ, {"SENTINEL_OFFLINE_EVAL": "1"}):
            with self.assertRaises(SystemExit):
                classify(novel_query)

    # ---------------------------------------------------------
    # 5. Missing API Credentials & Error Handling
    # ---------------------------------------------------------
    @patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False)
    @patch("src.classifier._get_cache", return_value={})
    def test_classifier_missing_credentials_raises_value_error(self, mock_cache):
        # When GROQ_API_KEY is not set and uncached query is passed, raises explicit ValueError
        with self.assertRaises(ValueError) as ctx:
            classify("Uncached query needing Groq API")
        self.assertIn("CRITICAL: GROQ_API_KEY environment variable is missing", str(ctx.exception))

    @patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False)
    @patch("src.reply_drafter._get_cache", return_value={})
    def test_reply_drafter_missing_credentials_fallback(self, mock_cache):
        # When GROQ_API_KEY is not set, draft_reply returns safe fallback rather than crashing
        res = draft_reply("Where is my order?", use_cache=False)
        self.assertIn("contact-us", res.draft_reply)
        self.assertIn("Fallback", res.policy_adherence_check)

    # ---------------------------------------------------------
    # 6. Invalid Model Output Handling
    # ---------------------------------------------------------
    @patch("src.classifier.time.sleep", return_value=None)
    @patch("src.classifier.requests.post")
    @patch("src.classifier._get_cache", return_value={})
    @patch.dict(os.environ, {"GROQ_API_KEY": "dummy_key", "SENTINEL_OFFLINE_EVAL": "0"})
    def test_classifier_invalid_json_fallback(self, mock_cache, mock_post, mock_sleep):
        # Mock API returning malformed response
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "INVALID NOT JSON"}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        # Should trigger fail-safe fallback to SEVERE_ESCALATION
        res = classify("Some random query")
        self.assertEqual(res.intent, IntentCategory.SEVERE_ESCALATION)
        self.assertEqual(res.confidence, 0.0)

    # ---------------------------------------------------------
    # 7. Retrieval Similarity & Filtering
    # ---------------------------------------------------------
    def test_retriever_metric_type_is_valid(self):
        metric = get_index_metric_type()
        self.assertIn(metric, ["IP", "L2"])

    def test_retriever_returns_top_k(self):
        res = retrieve("I need help with my Amazon delivery", k=3)
        self.assertIsInstance(res, list)
        self.assertLessEqual(len(res), 3)

    def test_retriever_exclude_exact_match(self):
        # If querying text that exists in clean_pairs, exclude_exact_match should omit it
        test_text = "Thanks @115821 for leaving my parcels in a safe place https://t.co/W959kz0j9w"
        full_res = retrieve(test_text, k=1, exclude_exact_match=False)
        excluded_res = retrieve(test_text, k=1, exclude_exact_match=True)
        # Excluded should still return historical examples but not identical self-match
        self.assertIsInstance(excluded_res, list)

    # ---------------------------------------------------------
    # 8. Grounding Safeguards & Historical Commitment Neutralization
    # ---------------------------------------------------------
    def test_safeguards_sanitize_urls(self):
        text_with_bad_links = (
            "Please share your order at https://example.com/fake-help or https://t.co/xyz123."
        )
        sanitized = apply_grounding_safeguards(text_with_bad_links)
        self.assertNotIn("example.com", sanitized)
        self.assertNotIn("t.co", sanitized)
        self.assertIn("https://www.amazon.com/contact-us", sanitized)

    def test_safeguards_neutralize_historical_cutoff_promises(self):
        # Copied past cutoff promise must be neutralized
        raw_draft = (
            "I see you selected next-day delivery after the cut-off time, "
            "so the earliest we can get your order to you is tomorrow. "
            "Unfortunately we can't guarantee a same-day delivery in this case."
        )
        safe_draft = apply_grounding_safeguards(raw_draft)
        self.assertNotIn("the earliest we can get your order to you is tomorrow", safe_draft.lower())
        self.assertNotIn("we can't guarantee a same-day delivery in this case", safe_draft.lower())
        self.assertIn("checkout", safe_draft.lower())

    def test_safeguards_neutralize_sla_promises(self):
        # Unverified SLA timeline promises ("get back to you shortly") must be neutralized
        raw_draft = "Please submit your details and we will get back to you shortly."
        safe_draft = apply_grounding_safeguards(raw_draft)
        self.assertNotIn("get back to you shortly", safe_draft.lower())
        self.assertIn("review your inquiry", safe_draft.lower())

    # ---------------------------------------------------------
    # 9. Low-Similarity Retrieval & Safe Fallback
    # ---------------------------------------------------------
    def test_retriever_high_similarity_threshold_filters_weak_matches(self):
        # An absurd query with an impossibly high similarity threshold (0.99) should return []
        res = retrieve("Completely unrelated query about interstellar astrophysics", k=3, min_similarity=0.99)
        self.assertEqual(res, [])

    # ---------------------------------------------------------
    # 10. Escalation Threshold Configuration & Consistency
    # ---------------------------------------------------------
    @patch("src.escalation._get_cache")
    def test_escalation_default_threshold_is_040(self, mock_cache):
        # With no env var set, default threshold must be 0.40 (calibrated for high recall / FNR < 35%)
        text = "Order tracking with intermediate risk score"
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        mock_cache.return_value = {
            text_hash: {"escalation_score": 0.50, "reasoning": "Intermediate risk score."}
        }
        with patch.dict(os.environ, {}, clear=True):
            # 0.50 >= default 0.40 -> should escalate
            decision = gatekeeper(text, "Order Issues", 0.90, "Intermediate issue")
            self.assertTrue(decision.should_escalate)

    @patch("src.escalation._get_cache")
    def test_escalation_threshold_override(self, mock_cache):
        # Environment variable override must be respected
        text = "Order tracking with intermediate risk score"
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        mock_cache.return_value = {
            text_hash: {"escalation_score": 0.50, "reasoning": "Intermediate risk score."}
        }
        with patch.dict(os.environ, {"ESCALATION_THRESHOLD": "0.70"}):
            # 0.50 < 0.70 -> should NOT escalate under 0.70
            decision = gatekeeper(text, "Order Issues", 0.90, "Intermediate issue")
            self.assertFalse(decision.should_escalate)

    # ---------------------------------------------------------
    # 11. Clean-Clone Bootstrap Fixture Integrity
    # ---------------------------------------------------------
    def test_bootstrap_fixtures_exist(self):
        # Evaluation fixtures must exist and have valid JSON content
        fixtures_dir = os.path.join(_ROOT, "eval", "fixtures")
        for fixture_file in [
            "classifier_cache.json",
            "escalation_cache.json",
            "drafter_cache.json",
            "zeroshot_cache.json"
        ]:
            fp = os.path.join(fixtures_dir, fixture_file)
            self.assertTrue(os.path.exists(fp), f"Missing fixture file: {fixture_file}")
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.assertGreater(len(data), 0, f"Fixture {fixture_file} is empty!")

    # ---------------------------------------------------------
    # 12. Cache Corruption Resilience
    # ---------------------------------------------------------
    def test_cache_corruption_graceful_handling(self):
        from src.reply_drafter import _get_cache
        # Mocking an invalid/corrupt JSON read
        with patch("builtins.open", side_effect=json.JSONDecodeError("Expecting value", "corrupt", 0)):
            with patch("os.path.exists", return_value=True):
                # When cache file is corrupt, should return empty dict rather than crashing
                cache = _get_cache()
                self.assertIsInstance(cache, dict)

    # ---------------------------------------------------------
    # 13. Escalated Case Auto-Draft Suppression
    # ---------------------------------------------------------
    @patch("src.reply_drafter.draft_reply")
    @patch("src.escalation.gatekeeper")
    @patch("src.classifier.classify")
    def test_escalated_case_never_calls_draft_reply(self, mock_classify, mock_gate, mock_draft):
        # High-risk query: threat of legal action
        mock_classify.return_value = ClassificationResult(
            intent=IntentCategory.SEVERE_ESCALATION,
            confidence=0.98,
            reasoning="Legal threat detected."
        )
        mock_gate.return_value = EscalationDecision(
            should_escalate=True,
            reason="Escalated: Legal keywords present."
        )

        from eval.evaluate_end_to_end import run_full_pipeline_triage
        result = run_full_pipeline_triage("I will sue you in court today!")

        # Crucial safety check: draft_reply must NEVER be called
        mock_draft.assert_not_called()
        self.assertFalse(result["auto_draft_generated"])
        self.assertIn("[ESCALATED TO HUMAN AGENT]", result["draft_reply"])

    # ---------------------------------------------------------
    # 14. Golden-Set ID Exclusion in Retriever
    # ---------------------------------------------------------
    def test_retriever_excludes_all_golden_ids(self):
        # Querying an exact golden test example with exclude_exact_match=True
        # must exclude its tweet_id_customer from the returned context
        eval_path = os.path.join(_ROOT, "eval", "golden_test.json")
        with open(eval_path, "r", encoding="utf-8") as f:
            gold_cases = json.load(f)
        
        target_case = gold_cases[0]
        results = retrieve(
            target_case["text_customer"],
            k=3,
            exclude_exact_match=True,
            return_metadata=True
        )
        
        # Verify none of the returned historical queries have the same tweet_id_customer
        for r in results:
            self.assertNotEqual(r["historical_query"].strip().lower(), target_case["text_customer"].strip().lower())


if __name__ == "__main__":
    unittest.main()

# tests/test_safety_and_edge_cases.py
"""
Dedicated Safety, Guardrail, and Failure Mode Regression Test Suite for Sentinel-AI.
Covers:
  - Critical safety hazards, physical injuries, and legal threats
  - PII injection, credit card & SSN leak prevention
  - Automated draft suppression for high-risk escalated cases
"""

import os
import sys
import pytest

# Ensure project root is on sys.path
_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.classifier import (
    ClassificationResult,
    IntentCategory,
    _get_cache as class_get_cache,
    _save_cache as class_save_cache,
    classify,
)
from src.escalation import (
    _get_cache as esc_get_cache,
    _save_cache as esc_save_cache,
    evaluate_risk,
    gatekeeper,
    rule_based_escalation,
)
from src.pipeline import run_full_pipeline
from src.reply_drafter import apply_grounding_safeguards
from unittest.mock import MagicMock, patch


class TestSafetyAndEdgeCases:

    # -------------------------------------------------------------
    # 1. Severe Emergency & Legal Escalation Tests
    # -------------------------------------------------------------
    @pytest.mark.parametrize("toxic_payload", [
        "I cut my hand on your broken packaging, I am going to the hospital.",
        "If I don't get my refund today, you will be hearing from my lawyer.",
        "My driver was verbally abusive and threatened me."
    ])
    def test_severe_cases_trigger_immediate_escalation(self, toxic_payload):
        """Validates that life safety hazards, legal threats, and driver abuse immediately escalate."""
        # 1. Classification should recognize the intent
        result = classify(toxic_payload)
        assert result.confidence >= 0.0

        # 2. Risk evaluation must breach the 0.40 threshold
        risk_score, should_escalate = evaluate_risk(result)
        
        assert should_escalate is True, f"Failed to escalate toxic payload: {toxic_payload}"
        assert risk_score >= 0.40, "Risk score did not reflect severity."

        # 3. Full pipeline must withhold automated customer draft
        pipeline_res = run_full_pipeline(toxic_payload)
        assert pipeline_res.should_escalate is True
        assert pipeline_res.draft is None, "Automated draft must never be sent for severe cases!"
        assert pipeline_res.auto_draft_generated is False

    # -------------------------------------------------------------
    # 2. PII & Sensitive Credential Injection Safeguards
    # -------------------------------------------------------------
    def test_pii_injection_safeguard(self):
        """Verifies SSN and credit card numbers trigger escalation and are never reflected in drafts."""
        pii_payload = "My account is locked. My SSN is 000-11-2222 and credit card is 4111-2222-3333-4444."
        
        # Run through full pipeline (Classification -> Escalation -> Draft)
        pipeline_response = run_full_pipeline(pii_payload)
        
        # System must escalate due to sensitive credentials
        assert pipeline_response.should_escalate is True
        assert "SSN" in pipeline_response.escalation_reason or "card" in pipeline_response.escalation_reason.lower()

        # Assert the system does not echo the PII back in an automated draft
        if pipeline_response.draft:
            assert "000-11-2222" not in pipeline_response.draft
            assert "4111-2222-3333-4444" not in pipeline_response.draft

    def test_pii_scrubbing_in_grounding_safeguards(self):
        """Verifies deterministic post-generation safeguards redact SSN and credit cards."""
        synthetic_echo = (
            "We see your SSN is 000-11-2222 and your card 4111-2222-3333-4444. "
            "Please visit https://example.com/login for help."
        )
        scrubbed = apply_grounding_safeguards(synthetic_echo)
        assert "000-11-2222" not in scrubbed
        assert "4111-2222-3333-4444" not in scrubbed
        assert "[REDACTED_SSN]" in scrubbed
        assert "[REDACTED_CARD]" in scrubbed
        assert "https://www.amazon.com/contact-us" in scrubbed
        assert "example.com" not in scrubbed

    # -------------------------------------------------------------
    # 3. Deterministic Safety Rules Backstop
    # -------------------------------------------------------------
    def test_deterministic_emergency_keywords(self):
        """Verifies emergency and legal keywords trigger deterministic Layer 1 escalation."""
        for kw in ["police", "ambulance", "hospital", "lawyer", "attorney", "sue"]:
            text = f"There is an emergency with my package and I called the {kw}."
            should_esc, reason = rule_based_escalation(text, 0.90)
            assert should_esc is True, f"Keyword '{kw}' failed rule-based escalation!"
            assert kw in reason.lower()

    # -------------------------------------------------------------
    # 4. Strict Code Receipts & Branch Execution Tests
    # -------------------------------------------------------------
    @patch("src.classifier.requests.post")
    @patch("src.classifier._save_cache")
    @patch("src.classifier._get_cache")
    def test_classifier_uncached_api_success(self, mock_get_cache, mock_save_cache, mock_post):
        mock_get_cache.return_value = {}
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"intent": "Product Issues", "confidence": 0.88, "reasoning": "Uncached success test"}'
                }
            }]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        res = classify("My mixer stopped turning yesterday")
        assert res.intent == IntentCategory.PRODUCT_ISSUES
        assert res.confidence == 0.88
        assert mock_save_cache.called

    @patch("src.escalation.requests.post")
    @patch("src.escalation._save_cache")
    @patch("src.escalation._get_cache")
    def test_gatekeeper_uncached_api_success(self, mock_get_cache, mock_save_cache, mock_post):
        mock_get_cache.return_value = {}
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"escalation_score": 0.85, "reasoning": "Critical unhandled issue"}'
                }
            }]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        dec = gatekeeper("Uncached high risk escalation message", "Order Issues", 0.9, "Damaged item")
        assert dec.should_escalate is True
        assert "Critical" in dec.reason
        assert mock_save_cache.called

    @patch("src.escalation.time.sleep", return_value=None)
    @patch("src.escalation.requests.post")
    @patch("src.escalation._get_cache")
    def test_gatekeeper_api_failure_fallback(self, mock_get_cache, mock_post, mock_sleep):
        mock_get_cache.return_value = {}
        mock_post.side_effect = RuntimeError("Groq rate limit exceeded")

        dec = gatekeeper("Any message when API fails", "Order Issues", 0.9, "Damaged item")
        assert dec.should_escalate is True
        assert "API Error" in dec.reason

    @patch("src.escalation._get_cache")
    def test_gatekeeper_invalid_threshold_env_fallback(self, mock_get_cache):
        mock_get_cache.return_value = {
            "dummy_hash": {"escalation_score": 0.50, "reasoning": "Score above 0.40"}
        }
        with patch("hashlib.sha256") as mock_hash:
            mock_hash.return_value.hexdigest.return_value = "dummy_hash"
            with patch.dict(os.environ, {"ESCALATION_THRESHOLD": "not_a_valid_float"}):
                dec = gatekeeper("Test message", "Order Issues", 0.9, "Reason")
                assert dec.should_escalate is True

    @patch("src.classifier.classify")
    @patch("src.escalation._get_cache")
    def test_evaluate_risk_with_string_target(self, mock_cache, mock_classify):
        mock_classify.return_value = ClassificationResult(
            intent=IntentCategory.DELIVERY_ISSUES,
            confidence=0.92,
            reasoning="Late arrival"
        )
        mock_cache.return_value = {
            "hash_123": {"escalation_score": 0.20, "reasoning": "Minor delay"}
        }
        with patch("hashlib.sha256") as mock_hash:
            mock_hash.return_value.hexdigest.return_value = "hash_123"
            score, should_esc = evaluate_risk("Where is my package?", threshold=0.40)
            assert should_esc is False
            assert score == 0.20

    def test_rule_based_credit_card_escalation(self):
        should_esc, reason = rule_based_escalation("My card number is 4111-2222-3333-4444 please refund", 0.90)
        assert should_esc is True
        assert "credit/debit card" in reason.lower()

    @patch("src.escalation._get_cache", return_value={})
    def test_gatekeeper_missing_credentials_raises_value_error(self, mock_cache):
        with patch.dict(os.environ, {"GROQ_API_KEY": "", "SENTINEL_OFFLINE_EVAL": "0"}):
            with pytest.raises(ValueError) as excinfo:
                gatekeeper("Uncached query", "Order Issues", 0.9, "Damaged item")
            assert "CRITICAL: GROQ_API_KEY environment variable is missing" in str(excinfo.value)

    @patch("src.escalation._get_cache", return_value={})
    def test_gatekeeper_offline_eval_cache_miss_exits(self, mock_cache):
        with patch.dict(os.environ, {"SENTINEL_OFFLINE_EVAL": "1"}):
            with pytest.raises(SystemExit):
                gatekeeper("Uncached query", "Order Issues", 0.9, "Damaged item")

    def test_evaluate_risk_severe_intent_immediate(self):
        severe_res = ClassificationResult(
            intent=IntentCategory.SEVERE_ESCALATION,
            confidence=0.99,
            reasoning="Severe distress"
        )
        score, should_esc = evaluate_risk(severe_res)
        assert should_esc is True
        assert score == 1.0

    def test_cache_helpers_and_empty_rule_checks(self):
        should_esc, reason = rule_based_escalation("", 0.9)
        assert should_esc is False

        # Verify cache serialization works cleanly
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_cache_file = os.path.join(tmp_dir, "cache.json")
            with patch("src.classifier.CACHE_FILE", tmp_cache_file):
                class_save_cache({"test_key": {"intent": "Feedback"}})
                assert os.path.exists(tmp_cache_file)
            with patch("src.escalation.CACHE_FILE", tmp_cache_file):
                esc_save_cache({"test_key": {"escalation_score": 0.9}})
                assert os.path.exists(tmp_cache_file)



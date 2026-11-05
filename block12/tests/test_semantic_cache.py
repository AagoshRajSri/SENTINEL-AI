# tests/test_semantic_cache.py
"""
Unit and Integration Tests for Sentinel-AI Semantic Cache (src/cache.py).
Validates exact hash matching, cosine similarity thresholding, cache miss behavior,
and graceful offline fallback.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.cache import SemanticCache, triage_with_semantic_cache
from src.pipeline import PipelineResponse


class TestSemanticCache(unittest.TestCase):

    def setUp(self):
        # Create an in-memory/test cache instance
        self.cache = SemanticCache(redis_url=None)  # Uses local in-memory fallback
        self.cache.clear()

        self.sample_response = PipelineResponse(
            intent="Delivery Issues",
            confidence=0.98,
            reasoning="Customer asking for delayed package update.",
            escalation_score=0.10,
            should_escalate=False,
            escalation_reason="Passed escalation checks.",
            draft="Hi! Please contact us at https://www.amazon.com/contact-us so we can locate your order.",
            policy_adherence="Empathetic, verified official link.",
            auto_draft_generated=True,
            retrieved_context_count=3
        )

    def test_store_and_exact_hash_hit(self):
        query = "Where is my delayed package?"
        self.cache.store(query, self.sample_response)

        # Lookup exact same query
        result = self.cache.lookup(query)
        self.assertIsNotNone(result)
        cached_resp, sim, matched_q = result
        self.assertEqual(cached_resp.intent, "Delivery Issues")
        self.assertEqual(matched_q, query)
        self.assertGreaterEqual(sim, 0.99)
        self.assertEqual(self.cache.hits, 1)

    def test_semantic_similarity_hit_above_threshold(self):
        q1 = "Where is my delayed package?"
        # Minor variation: punctuation and capitalization
        q2 = "where is my delayed package"

        self.cache.store(q1, self.sample_response)
        result = self.cache.lookup(q2, threshold=0.95)

        self.assertIsNotNone(result)
        cached_resp, sim, matched_q = result
        self.assertEqual(cached_resp.intent, "Delivery Issues")
        self.assertGreaterEqual(sim, 0.95)
        self.assertEqual(matched_q, q1)

    def test_cache_miss_for_unrelated_query(self):
        q1 = "Where is my delayed package?"
        q_unrelated = "How do I cancel my Amazon Prime subscription?"

        self.cache.store(q1, self.sample_response)
        result = self.cache.lookup(q_unrelated, threshold=0.95)

        self.assertIsNone(result)
        self.assertEqual(self.cache.misses, 1)

    def test_empty_or_whitespace_query_returns_none(self):
        self.assertIsNone(self.cache.lookup(""))
        self.assertIsNone(self.cache.lookup("   "))

    def test_triage_with_semantic_cache_wrapper(self):
        query = "Where is my package?"
        with patch("src.cache.run_full_pipeline", return_value=self.sample_response) as mock_pipeline:
            # First call: Cache Miss -> executes pipeline
            resp1, is_hit1, sim1, lat1 = triage_with_semantic_cache(query, cache=self.cache)
            self.assertFalse(is_hit1)
            mock_pipeline.assert_called_once()

            # Second call: Cache Hit -> skips pipeline entirely
            mock_pipeline.reset_mock()
            resp2, is_hit2, sim2, lat2 = triage_with_semantic_cache(query, cache=self.cache)
            self.assertTrue(is_hit2)
            self.assertGreaterEqual(sim2, 0.99)
            mock_pipeline.assert_not_called()


if __name__ == "__main__":
    unittest.main()

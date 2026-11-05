# tests/test_kafka_pipeline.py
"""
Hermetic Unit Tests for Sentinel-AI Kafka Ingestion and Dispatch Pipeline.
Validates event serialization, schema compliance, and at-least-once commit semantics
with mocked Kafka brokers (no live broker required).
"""

import asyncio
import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.kafka_producer import SentinelKafkaProducer
from src.kafka_consumer import SentinelKafkaConsumer
from src.pipeline import PipelineResponse


class TestKafkaPipeline(unittest.IsolatedAsyncioTestCase):

    async def test_producer_event_serialization_and_schema(self):
        """Verifies producer formats PipelineResponse into valid JSON event payload."""
        producer = SentinelKafkaProducer(bootstrap_servers="mock:9092")
        mock_aiokafka_producer = AsyncMock()
        mock_aiokafka_producer.send_and_wait.return_value = MagicMock(
            topic="processed_triage_events",
            partition=1,
            offset=42
        )
        producer._producer = mock_aiokafka_producer

        sample_pipeline_res = PipelineResponse(
            intent="Delivery Issues",
            confidence=0.98,
            reasoning="Customer asking for delayed package update.",
            escalation_score=0.10,
            should_escalate=False,
            escalation_reason="Passed escalation checks.",
            draft="Hi! Please contact us at https://www.amazon.com/contact-us",
            policy_adherence="Valid contact link, empathetic tone.",
            auto_draft_generated=True,
            route="⚡ AUTO-DRAFTED"
        )

        event = await producer.publish_triage_result(
            ticket_id="ticket_test_101",
            inbound_text="Where is my delayed package?",
            pipeline_response=sample_pipeline_res,
            processing_latency_ms=12.5,
            topic="processed_triage_events"
        )

        self.assertEqual(event["ticket_id"], "ticket_test_101")
        self.assertEqual(event["intent"], "Delivery Issues")
        self.assertFalse(event["should_escalate"])
        self.assertTrue(event["auto_draft_generated"])
        self.assertEqual(event["processing_latency_ms"], 12.5)
        self.assertIn("_kafka", event)
        self.assertEqual(event["_kafka"]["partition"], 1)
        self.assertEqual(event["_kafka"]["offset"], 42)

        # Assert correct topic and key passed to Kafka client
        mock_aiokafka_producer.send_and_wait.assert_awaited_once()
        call_kwargs = mock_aiokafka_producer.send_and_wait.call_args.kwargs
        self.assertEqual(call_kwargs["topic"], "processed_triage_events")
        self.assertEqual(call_kwargs["key"], "ticket_test_101")

    async def test_consumer_single_ticket_escalation_routing(self):
        """Verifies consumer processes high-risk message and routes to human queue."""
        consumer = SentinelKafkaConsumer(bootstrap_servers="mock:9092")
        mock_producer = AsyncMock()
        mock_producer.publish_triage_result.return_value = {"ticket_id": "ticket_esc_01"}
        consumer.producer = mock_producer

        raw_msg = MagicMock(
            key="ticket_esc_01",
            value={"text": "I will sue Amazon in small claims court today!"},
            topic="inbound_support_tickets",
            partition=0,
            offset=10
        )

        with patch("src.kafka_consumer.run_full_pipeline") as mock_pipeline:
            mock_pipeline.return_value = PipelineResponse(
                intent="Severe Escalation",
                confidence=0.99,
                reasoning="Legal threat keyword 'sue' detected.",
                escalation_score=1.0,
                should_escalate=True,
                escalation_reason="Escalated: Legal keywords present.",
                draft=None,
                policy_adherence="Safety Gate: High-risk query withheld.",
                auto_draft_generated=False,
                route="🚨 ESCALATED (Human Queue)"
            )

            result = await consumer._process_single_ticket(raw_msg)
            self.assertEqual(result["ticket_id"], "ticket_esc_01")
            self.assertEqual(consumer.total_dispatched, 1)
            self.assertEqual(consumer.total_escalated, 1)
            self.assertEqual(consumer.total_auto_drafted, 0)

    async def test_consumer_at_least_once_semantics_on_failure(self):
        """Verifies consumer does NOT commit offsets when processing raises an exception."""
        consumer = SentinelKafkaConsumer(bootstrap_servers="mock:9092", batch_size=1)
        mock_consumer = AsyncMock()
        consumer.consumer = mock_consumer
        consumer.producer = AsyncMock()
        consumer.running = True

        raw_msg = MagicMock(
            key="bad_ticket",
            value={"text": "Malformed test payload"},
            topic="inbound_support_tickets",
            partition=0,
            offset=5
        )

        # Return a batch with 1 message, then empty/cancel
        tp = MagicMock(topic="inbound_support_tickets", partition=0)
        mock_consumer.getmany.side_effect = [
            {tp: [raw_msg]},
            {}
        ]

        with patch.object(consumer, "_process_single_ticket", side_effect=RuntimeError("Transient pipeline crash")):
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                # Run one loop iteration and stop
                async def stop_after_error(*args):
                    consumer.running = False
                mock_sleep.side_effect = stop_after_error

                await consumer.run_loop(max_messages=1)

            # CRITICAL AT-LEAST-ONCE ASSERTION: commit() must NEVER be called upon failure
            mock_consumer.commit.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()

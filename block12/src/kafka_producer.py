# src/kafka_producer.py
"""
Asynchronous Kafka Dispatch Producer for Sentinel-AI.
Serializes PipelineResponse triage outcomes and publishes to processed_triage_events.
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, Optional
from aiokafka import AIOKafkaProducer

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
PROCESSED_TOPIC = os.getenv("KAFKA_PROCESSED_TOPIC", "processed_triage_events")


class SentinelKafkaProducer:
    """Production async producer managing connection pooling and JSON event serialization."""

    def __init__(self, bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS):
        self.bootstrap_servers = bootstrap_servers
        self._producer: Optional[AIOKafkaProducer] = None

    async def start(self):
        if self._producer is None:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",  # Strongest durability guarantee
                retry_backoff_ms=250,
            )
            await self._producer.start()

    async def stop(self):
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None

    async def publish_triage_result(
        self,
        ticket_id: str,
        inbound_text: str,
        pipeline_response: Any,
        processing_latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        topic: str = PROCESSED_TOPIC
    ) -> Dict[str, Any]:
        """Publishes the finalized PipelineResponse event to Kafka."""
        if self._producer is None:
            await self.start()

        event_payload = {
            "ticket_id": ticket_id,
            "text": inbound_text,
            "intent": getattr(pipeline_response, "intent", "Unknown"),
            "confidence": getattr(pipeline_response, "confidence", 0.0),
            "reasoning": getattr(pipeline_response, "reasoning", ""),
            "escalation_score": getattr(pipeline_response, "escalation_score", 0.0),
            "should_escalate": getattr(pipeline_response, "should_escalate", False),
            "escalation_reason": getattr(pipeline_response, "escalation_reason", ""),
            "draft": getattr(pipeline_response, "draft", None),
            "policy_adherence": getattr(pipeline_response, "policy_adherence", None),
            "auto_draft_generated": getattr(pipeline_response, "auto_draft_generated", False),
            "processing_latency_ms": round(processing_latency_ms, 2),
            "timestamp": time.time(),
            "metadata": metadata or {},
        }

        record_metadata = await self._producer.send_and_wait(
            topic=topic,
            key=ticket_id,
            value=event_payload
        )

        event_payload["_kafka"] = {
            "topic": record_metadata.topic,
            "partition": record_metadata.partition,
            "offset": record_metadata.offset,
        }
        return event_payload

    async def publish_raw_ticket(self, topic: str, ticket_id: str, payload: Dict[str, Any]):
        """Publishes an inbound ticket into the ingestion topic."""
        if self._producer is None:
            await self.start()
        return await self._producer.send_and_wait(
            topic=topic,
            key=ticket_id,
            value=payload
        )

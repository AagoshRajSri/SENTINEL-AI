# src/kafka_consumer.py
"""
Asynchronous Kafka Ingestion Consumer Daemon for Sentinel-AI.
Consumes batches of inbound customer support tickets, executes the two-pass
Sentinel pipeline concurrently via asyncio.gather, dispatches routing events
to processed_triage_events, and commits offsets with at-least-once semantics.
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv()

from aiokafka import AIOKafkaConsumer
from src.kafka_producer import SentinelKafkaProducer
from src.pipeline import run_full_pipeline, PipelineResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [SentinelKafkaConsumer] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SentinelKafkaConsumer")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
INBOUND_TOPIC = os.getenv("KAFKA_INBOUND_TOPIC", "inbound_support_tickets")
PROCESSED_TOPIC = os.getenv("KAFKA_PROCESSED_TOPIC", "processed_triage_events")
CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "sentinel-triage-workers")


class SentinelKafkaConsumer:
    """Enterprise event-driven consumer daemon with at-least-once manual offset commit."""

    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
        inbound_topic: str = INBOUND_TOPIC,
        processed_topic: str = PROCESSED_TOPIC,
        group_id: str = CONSUMER_GROUP,
        batch_size: int = 10,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.inbound_topic = inbound_topic
        self.processed_topic = processed_topic
        self.group_id = group_id
        self.batch_size = batch_size

        self.consumer: Optional[AIOKafkaConsumer] = None
        self.producer: Optional[SentinelKafkaProducer] = None
        self.running = False

        # Live telemetry counters
        self.total_consumed = 0
        self.total_dispatched = 0
        self.total_escalated = 0
        self.total_auto_drafted = 0
        self.total_latencies_ms: List[float] = []

    async def start(self):
        logger.info(f"Connecting to Kafka brokers at: {self.bootstrap_servers}")
        logger.info(f"Subscribing to inbound topic: '{self.inbound_topic}' (Group: '{self.group_id}')")

        self.consumer = AIOKafkaConsumer(
            self.inbound_topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            enable_auto_commit=False,  # Manual commits enforce at-least-once delivery
            auto_offset_reset="earliest",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            max_poll_records=self.batch_size,
        )

        self.producer = SentinelKafkaProducer(bootstrap_servers=self.bootstrap_servers)
        await self.consumer.start()
        await self.producer.start()
        self.running = True
        logger.info("Kafka consumer daemon initialized successfully. Ready to process events.")

    async def stop(self):
        logger.info("Initiating graceful shutdown of Kafka consumer and producer...")
        self.running = False
        if self.consumer is not None:
            await self.consumer.stop()
            self.consumer = None
        if self.producer is not None:
            await self.producer.stop()
            self.producer = None
        logger.info("Shutdown complete.")

    async def _process_single_ticket(self, raw_message) -> Dict[str, Any]:
        """Runs the two-pass pipeline on one message in a worker thread and dispatches outcome."""
        payload = raw_message.value
        ticket_id = (
            raw_message.key
            or (payload.get("ticket_id") if isinstance(payload, dict) else None)
            or f"ticket_{raw_message.partition}_{raw_message.offset}"
        )

        text = payload.get("text", "") if isinstance(payload, dict) else str(payload)
        t0 = time.perf_counter()

        # Non-blocking async execution offloaded to threadpool
        pipeline_res: PipelineResponse = await asyncio.to_thread(run_full_pipeline, text)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Dispatch finalized decision to output topic
        dispatched_event = await self.producer.publish_triage_result(
            ticket_id=ticket_id,
            inbound_text=text,
            pipeline_response=pipeline_res,
            processing_latency_ms=latency_ms,
            metadata={
                "inbound_topic": raw_message.topic,
                "inbound_partition": raw_message.partition,
                "inbound_offset": raw_message.offset,
            },
            topic=self.processed_topic
        )

        # Update telemetry
        self.total_dispatched += 1
        self.total_latencies_ms.append(latency_ms)
        if pipeline_res.should_escalate:
            self.total_escalated += 1
        if pipeline_res.auto_draft_generated:
            self.total_auto_drafted += 1

        route_flag = "🚨 ESCALATED (Human Queue)" if pipeline_res.should_escalate else "⚡ AUTO-DRAFTED"
        logger.info(
            f"Processed [{ticket_id}] in {latency_ms:.1f}ms -> {pipeline_res.intent} | {route_flag}"
        )
        return dispatched_event

    async def run_loop(self, max_messages: Optional[int] = None):
        """Main batch-processing polling loop."""
        if not self.running:
            await self.start()

        logger.info(f"Starting consumer poll loop (batch_size={self.batch_size})...")

        try:
            while self.running:
                # Fetch batch across assigned partitions
                batch_dict = await self.consumer.getmany(
                    timeout_ms=1000,
                    max_records=self.batch_size
                )

                if not batch_dict:
                    if max_messages and self.total_consumed >= max_messages:
                        logger.info(f"Reached target message limit ({max_messages}). Exiting loop.")
                        break
                    await asyncio.sleep(0.05)
                    continue

                for tp, messages in batch_dict.items():
                    if not messages:
                        continue

                    batch_count = len(messages)
                    self.total_consumed += batch_count
                    logger.info(f"Fetched batch of {batch_count} messages from {tp.topic}-{tp.partition}")

                    try:
                        # Process batch concurrently using asyncio.gather for peak throughput
                        await asyncio.gather(
                            *(self._process_single_ticket(msg) for msg in messages)
                        )

                        # Manual offset commit — executed ONLY after all records are safely dispatched
                        await self.consumer.commit()
                        logger.info(f"Successfully committed offsets for {batch_count} messages on {tp.topic}-{tp.partition}")

                    except Exception as exc:
                        # Strict At-Least-Once Delivery: Do NOT commit offsets on crash
                        logger.error(
                            f"Transient failure processing batch on {tp.topic}-{tp.partition}: {exc}. "
                            f"Offsets will NOT be committed. Retrying batch on next poll."
                        )
                        await asyncio.sleep(2.0)

                if max_messages and self.total_consumed >= max_messages:
                    logger.info(f"Completed processing of {self.total_consumed} messages. Target reached.")
                    break

        except asyncio.CancelledError:
            logger.info("Consumer poll loop received cancel signal.")
        finally:
            await self.stop()

    def print_summary(self):
        """Displays execution telemetry and throughput statistics."""
        avg_lat = (sum(self.total_latencies_ms) / len(self.total_latencies_ms)) if self.total_latencies_ms else 0.0
        print("\n" + "=" * 80)
        print("SENTINEL-AI: KAFKA CONSUMER THROUGHPUT & PROCESSING SUMMARY")
        print("=" * 80)
        print(f"Total Inbound Messages Consumed : {self.total_consumed}")
        print(f"Total Events Dispatched         : {self.total_dispatched}")
        print(f"Human Escalations Routed        : {self.total_escalated}")
        print(f"Safe Automated Drafts Delivered : {self.total_auto_drafted}")
        print(f"Average Pipeline Latency        : {avg_lat:.2f} ms")
        print(f"Delivery Semantics Guaranteed   : At-Least-Once (Manual Offset Commits)")
        print("=" * 80 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="Sentinel-AI Kafka Ingestion Consumer")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch poll size (default: 10)")
    parser.add_argument("--max-messages", type=int, default=None, help="Stop after processing N messages (default: unbounded)")
    parser.add_argument("--bootstrap-servers", type=str, default=KAFKA_BOOTSTRAP_SERVERS, help="Kafka broker host:port")
    args = parser.parse_args()

    daemon = SentinelKafkaConsumer(
        bootstrap_servers=args.bootstrap_servers,
        batch_size=args.batch_size
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(daemon.stop()))
        except (NotImplementedError, AttributeError):
            pass  # Windows event loop fallback

    try:
        await daemon.run_loop(max_messages=args.max_messages)
    finally:
        daemon.print_summary()


if __name__ == "__main__":
    asyncio.run(main())

"""Kafka specialist agent for event streaming.

Handles message publishing, consumption, and stream processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from .base import SpecialistAgent, SpecialistConfig, SpecialistCapability


@dataclass
class KafkaConfig(SpecialistConfig):
    """Kafka-specific configuration."""

    bootstrap_servers: str = "localhost:9092"
    client_id: str = "agentic-workflows"
    group_id: str = "agentic-workflows-group"
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = True
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: str | None = None
    sasl_username: str | None = None
    sasl_password: str | None = None


class KafkaAgent(SpecialistAgent):
    """Specialist agent for Apache Kafka.

    Capabilities:
    - Message publishing
    - Topic subscription
    - Consumer group management
    - Stream processing
    """

    def __init__(self, config: KafkaConfig | None = None, **kwargs):
        self.kafka_config = config or KafkaConfig()
        super().__init__(config=self.kafka_config, **kwargs)

        self._producer = None
        self._consumers: dict[str, Any] = {}
        self._admin = None

        self.register_handler("publish", self._publish)
        self.register_handler("subscribe", self._subscribe)
        self.register_handler("create_topic", self._create_topic)
        self.register_handler("list_topics", self._list_topics)
        self.register_handler("get_topic_info", self._get_topic_info)

    @property
    def capabilities(self) -> list[SpecialistCapability]:
        return [
            SpecialistCapability.EVENT_STREAMING,
            SpecialistCapability.MESSAGE_QUEUE,
            SpecialistCapability.PUBSUB,
        ]

    @property
    def service_name(self) -> str:
        return "Kafka"

    async def _connect(self) -> None:
        """Connect to Kafka cluster."""
        try:
            from aiokafka import AIOKafkaProducer
            from kafka.admin import KafkaAdminClient

            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.kafka_config.bootstrap_servers,
                client_id=self.kafka_config.client_id,
            )
            await self._producer.start()

            self._admin = KafkaAdminClient(
                bootstrap_servers=self.kafka_config.bootstrap_servers,
                client_id=f"{self.kafka_config.client_id}-admin",
            )

        except ImportError:
            self.logger.warning("aiokafka/kafka-python not installed")

    async def _disconnect(self) -> None:
        """Disconnect from Kafka."""
        if self._producer:
            await self._producer.stop()
            self._producer = None

        for consumer in self._consumers.values():
            await consumer.stop()
        self._consumers.clear()

        if self._admin:
            self._admin.close()
            self._admin = None

    async def _health_check(self) -> bool:
        """Check Kafka health."""
        if self._admin is None:
            return False
        try:
            self._admin.list_topics()
            return True
        except Exception:
            return False

    async def _publish(
        self,
        topic: str,
        value: bytes | str | dict,
        key: bytes | str | None = None,
        headers: dict[str, str] | None = None,
        partition: int | None = None,
    ) -> dict[str, Any]:
        """Publish a message to a topic.

        Args:
            topic: Topic name.
            value: Message value.
            key: Optional message key.
            headers: Optional headers.
            partition: Optional partition.

        Returns:
            Publish result with offset.
        """
        if self._producer is None:
            return {"error": "Producer not connected"}

        import json

        if isinstance(value, dict):
            value = json.dumps(value).encode()
        elif isinstance(value, str):
            value = value.encode()

        if isinstance(key, str):
            key = key.encode()

        header_list = [(k, v.encode()) for k, v in (headers or {}).items()]

        result = await self._producer.send_and_wait(
            topic,
            value=value,
            key=key,
            headers=header_list if header_list else None,
            partition=partition,
        )

        return {
            "topic": result.topic,
            "partition": result.partition,
            "offset": result.offset,
        }

    async def _subscribe(
        self,
        topics: list[str],
        callback: Callable[[dict], None] | None = None,
        group_id: str | None = None,
    ) -> str:
        """Subscribe to topics.

        Args:
            topics: List of topics.
            callback: Message callback.
            group_id: Consumer group ID.

        Returns:
            Consumer ID.
        """
        try:
            from aiokafka import AIOKafkaConsumer
        except ImportError:
            return "error: aiokafka not installed"

        consumer_id = f"consumer-{len(self._consumers)}"

        consumer = AIOKafkaConsumer(
            *topics,
            bootstrap_servers=self.kafka_config.bootstrap_servers,
            group_id=group_id or self.kafka_config.group_id,
            auto_offset_reset=self.kafka_config.auto_offset_reset,
            enable_auto_commit=self.kafka_config.enable_auto_commit,
        )

        await consumer.start()
        self._consumers[consumer_id] = consumer

        return consumer_id

    async def consume(
        self,
        consumer_id: str,
        timeout_ms: int = 1000,
        max_messages: int = 100,
    ) -> list[dict[str, Any]]:
        """Consume messages from a consumer.

        Args:
            consumer_id: Consumer ID from subscribe.
            timeout_ms: Poll timeout.
            max_messages: Maximum messages to return.

        Returns:
            List of messages.
        """
        consumer = self._consumers.get(consumer_id)
        if not consumer:
            return []

        messages = []
        async for msg in consumer:
            messages.append({
                "topic": msg.topic,
                "partition": msg.partition,
                "offset": msg.offset,
                "key": msg.key.decode() if msg.key else None,
                "value": msg.value,
                "timestamp": msg.timestamp,
                "headers": dict(msg.headers) if msg.headers else {},
            })
            if len(messages) >= max_messages:
                break

        return messages

    async def _create_topic(
        self,
        name: str,
        partitions: int = 1,
        replication_factor: int = 1,
        config: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create a new topic.

        Args:
            name: Topic name.
            partitions: Number of partitions.
            replication_factor: Replication factor.
            config: Topic configuration.

        Returns:
            Creation result.
        """
        if self._admin is None:
            return {"error": "Admin not connected"}

        from kafka.admin import NewTopic

        topic = NewTopic(
            name=name,
            num_partitions=partitions,
            replication_factor=replication_factor,
            topic_configs=config,
        )

        try:
            self._admin.create_topics([topic])
            return {"topic": name, "created": True}
        except Exception as e:
            return {"topic": name, "created": False, "error": str(e)}

    async def _list_topics(self) -> list[str]:
        """List all topics."""
        if self._admin is None:
            return []
        return list(self._admin.list_topics())

    async def _get_topic_info(self, topic: str) -> dict[str, Any]:
        """Get topic metadata."""
        if self._admin is None:
            return {"error": "Admin not connected"}

        try:
            metadata = self._admin.describe_topics([topic])
            if metadata:
                info = metadata[0]
                return {
                    "topic": info.get("topic"),
                    "partitions": len(info.get("partitions", [])),
                    "is_internal": info.get("is_internal", False),
                }
            return {"topic": topic, "error": "Not found"}
        except Exception as e:
            return {"topic": topic, "error": str(e)}

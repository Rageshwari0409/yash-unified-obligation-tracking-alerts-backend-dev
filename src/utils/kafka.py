"""Kafka utilities for logging and messaging."""

import os
import json
import logging
import atexit
import threading  # Import the threading module
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Union
from uuid import uuid4

try:
    from kafka import KafkaProducer
    from kafka.errors import NoBrokersAvailable

    KAFKA_INSTALLED = True
except ImportError:
    KAFKA_INSTALLED = False

    class NoBrokersAvailable(Exception):
        pass


logger = logging.getLogger(__name__)


class KafkaLogger:
    """
    A resilient, production-grade logger that sends payloads asynchronously.
    It initializes its connection to Kafka lazily and is failsafe, meaning
    application startup and requests will not fail if Kafka is unavailable.
    """

    def __init__(self):
        self.producer = None
        self._lock = threading.Lock()
        self.topic = os.getenv("KAFKA_TOPIC_NAME", "llm-token-usage")
        atexit.register(self.close)
 
    def _initialize_producer(self) -> bool:
        """
        Initializes the KafkaProducer. This method is called internally and
        is protected by a lock to ensure it's thread-safe.
        Returns True on success, False on failure.
        """
        if not KAFKA_INSTALLED:
            logger.critical(
                "Dependency 'kafka-python' is not installed. Kafka logging is disabled."
            )
            return False
 
        bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
 
        if not bootstrap_servers or not self.topic:
            logger.critical(
                "KAFKA_BOOTSTRAP_SERVERS or KAFKA_TOPIC_NAME is not set. Kafka logging disabled."
            )
            return False
       
        try:
            logger.info(
                f"Attempting to initialize KafkaProducer and connect to {bootstrap_servers}..."
            )
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers.split(","),
                security_protocol="SSL",
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                retries=5,
                request_timeout_ms=30000,
                acks="all",
                # A small buffer time to allow the producer to batch messages even under low load
                linger_ms=5,
            )
            logger.info(
                f"KafkaProducer connected successfully. Logging to topic '{self.topic}'."
            )
            return True
        except (NoBrokersAvailable, Exception) as e:
            # Catch ANY exception during initialization.
            logger.critical(
                f"FATAL: Could not initialize KafkaProducer. Logging will be disabled. Error: {e}",
                exc_info=True,
            )
            self.producer = None  # Ensure producer is None on failure
            return False
 
    def _on_send_success(self, record_metadata):
        """Callback for successful message sends."""
        logging.debug(
            f"Message delivered to topic '{record_metadata.topic}' partition {record_metadata.partition}"
        )

    def _on_send_error(self, excp):
        """Callback for failed message sends."""
        logger.error(
            f"Error sending message to Kafka in the background: {excp}", exc_info=excp
        )

    def log(self, data: dict):
        """
        Sends a log asynchronously. If the producer is not initialized, it will
        attempt to do so. This operation will not block the caller or raise an
        exception if Kafka is unavailable.
        """
        if not self.producer:
            with self._lock:
                if not self.producer and not self._initialize_producer(): 
                        logger.warning("Kafka producer is not available. Message not sent.")
                        return
       
        if self.producer:
            try:
                print("\n--- [KAFKA PAYLOAD DEBUG] ---")
                print(json.dumps(data, indent=2))
                print("-----------------------------\n")
            except Exception as e:
                print(f"--- [KAFKA PAYLOAD DEBUG] FAILED TO PRINT PAYLOAD: {e} ---")
 
            try:
                future = self.producer.send(self.topic, value=data)
                future.add_callback(self._on_send_success)
                future.add_errback(self._on_send_error)
            except Exception as e:
                logger.error(f"Error while queuing message for Kafka: {e}", exc_info=True)
 
    def close(self):
        """Flushes buffered messages and closes the producer during graceful shutdown."""
        if self.producer:
            logger.info("Flushing remaining messages and closing Kafka producer...")
            try:
                self.producer.flush(timeout=10)
            except Exception as e:
                logger.error(f"Error flushing messages to Kafka: {e}", exc_info=True)
            finally:
                self.producer.close()
                logger.info("Kafka producer closed.")


class KafkaEventLogger:
    """
    Simplified Event Logger for real-time user visibility.
    Sends minimal event structure: encrypted_payload, timestamp, message only.
    """

    def __init__(self):
        self.producer = None
        self._lock = threading.Lock()
        self.topic = "agent-event-notification"
        self.agent_name = os.getenv("AGENT_NAME", "OBLIGATION_TRACKING_ALERTS_AGENT")
        self.server_name = os.getenv("SERVER_NAME", "OBLIGATION_TRACKING_ALERTS_BACKEND")

    def _initialize_producer(self) -> bool:
        """Initialize KafkaProducer for event logging."""
        if not KAFKA_INSTALLED:
            logger.warning("kafka-python not installed. Event logging disabled.")
            return False

        bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        if not bootstrap_servers:
            logger.warning("KAFKA_BOOTSTRAP_SERVERS not set. Event logging disabled.")
            return False

        try:
            logger.info(f"Initializing Event Logger for topic '{self.topic}'...")
            producer_config = {
                "bootstrap_servers": bootstrap_servers.split(","),
                "value_serializer": lambda v: json.dumps(v, default=str).encode(
                    "utf-8"
                ),
                "key_serializer": lambda k: k.encode("utf-8") if k else None,
                "retries": 3,
                "request_timeout_ms": 15000,
                "acks": 1,
                "linger_ms": 10,
                "batch_size": 16384,
            }
            if os.getenv("KAFKA_USE_SSL", "true").lower() == "true":
                producer_config["security_protocol"] = "SSL"

            self.producer = KafkaProducer(**producer_config)
            logger.info(f"Event Logger connected successfully. Topic: '{self.topic}'")
            return True
        except Exception as e:
            logger.error(f"Could not initialize Event Logger: {e}", exc_info=True)
            self.producer = None
            return False

    def _extract_user_context_from_request(self, auth_token: str = None) -> dict:
        """Extract encrypted_payload from auth token for user context."""
        user_context = {"encrypted_payload": "N/A"}

        if not auth_token:
            return user_context

        try:
            # Split JWT from encrypted payload using custom separator
            CUSTOM_TOKEN_SEPARATOR = "$YashUnified2025$"
            if CUSTOM_TOKEN_SEPARATOR in auth_token:
                _, encrypted_payload = auth_token.split(CUSTOM_TOKEN_SEPARATOR, 1)
                user_context["encrypted_payload"] = encrypted_payload
            else:
                # Fallback: use a mock encrypted payload derived from token
                jwt_part = auth_token
                if jwt_part.lower().startswith("bearer "):
                    jwt_part = jwt_part[7:]

                # Create mock encrypted payload from JWT (for testing/fallback)
                user_context["encrypted_payload"] = (
                    f"mock-encrypted-payload-{jwt_part[-10:]}"
                )

        except Exception as e:
            logger.debug(f"Error extracting user context from token: {e}")

        return user_context

    def _create_base_event(self, message: str, auth_token: str = None) -> dict:
        """Create simplified base event with only essential fields."""
        from datetime import datetime, timezone

        user_context = self._extract_user_context_from_request(auth_token)

        return {
            "encrypted_payload": user_context["encrypted_payload"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
            # "kafka_topic_name": self.topic,
            "kafka_topic_name": "agent-event-notification",
            "server_name": self.server_name,
            "type": "agent-event"
        }

    def _send_event(self, event: dict):
        """Send event to Kafka topic."""
        if not self.producer:
            with self._lock:
                if not self.producer:
                    self._initialize_producer()

        if not self.producer:
            return

        try:
            # Enhanced terminal logging for developer visibility
            print("\n--- [KAFKA EVENT DEBUG] ---")
            print(json.dumps(event, indent=2))
            print("-----------------------------\n")

            self.producer.send(self.topic, value=event)
        except Exception as e:
            logger.error(f"Error sending event to Kafka: {e}")

    def log_event(self, message: str, auth_token: str = None):
        """Log a general event."""
        event = self._create_base_event(message, auth_token)
        self._send_event(event)

    def log_progress(self, message: str, percent: int = None, auth_token: str = None):
        """Log a progress event."""
        if percent is not None:
            message = f"{message} ({percent}%)"
        event = self._create_base_event(message, auth_token)
        self._send_event(event)

    def log_llm_interaction(self, message: str, auth_token: str = None):
        """Log an LLM interaction event."""
        event = self._create_base_event(message, auth_token)
        self._send_event(event)

    def log_error(
        self, message: str, error_details: str = None, auth_token: str = None
    ):
        """Log an error event."""
        if error_details:
            message = f"{message}: {error_details}"
        event = self._create_base_event(message, auth_token)
        self._send_event(event)

    def log_success(self, message: str, auth_token: str = None):
        """Log a success event."""
        event = self._create_base_event(message, auth_token)
        self._send_event(event)

    def close(self):
        """Close the event logger producer."""
        if self.producer:
            logger.info("Closing Event Logger Kafka producer...")
            self.producer.flush(timeout=5)
            self.producer.close()
            logger.info("Event Logger closed.")


# Factory functions to create logger instances
def create_event_logger() -> KafkaEventLogger:
    """Create a new event logger instance."""
    return KafkaEventLogger()


# Global Kafka logger instance
kafka_logger = KafkaLogger()
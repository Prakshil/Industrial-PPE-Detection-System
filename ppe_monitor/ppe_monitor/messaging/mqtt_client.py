"""
MQTT client for publishing confirmed violation events.
Publishes to topic: ppe/<zone_id>/violations
"""

from __future__ import annotations

import json
import logging
from typing import Any

import paho.mqtt.client as mqtt

from ppe_monitor.compliance.worker_state import ViolationEvent

log = logging.getLogger(__name__)


class MQTTClient:
    """Thin wrapper around paho-mqtt for PPE event publishing."""

    BASE_TOPIC = "ppe"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 1883,
        client_id: str = "ppe_edge_node",
        username: str | None = None,
        password: str | None = None,
        qos: int = 1,
    ):
        self._qos = qos
        # Support both paho-mqtt v2.x and v1.x client initialization
        try:
            callback_api = getattr(mqtt, "CallbackAPIVersion", None)
            if callback_api is not None:
                self._client = mqtt.Client(callback_api.VERSION1, client_id=client_id)
            else:
                self._client = mqtt.Client(client_id=client_id)
        except Exception:
            self._client = mqtt.Client(client_id=client_id)

        if username:
            self._client.username_pw_set(username, password)

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._is_connected = False

        try:
            self._client.connect(host, port, keepalive=60)
            self._client.loop_start()
        except Exception as exc:
            self._is_connected = False
            log.warning("MQTT broker unavailable (%s). Events will be stored locally.", exc)

    def _on_connect(self, client, userdata, flags, rc, *args):
        if rc == 0:
            self._is_connected = True
            log.info("MQTT connected.")
        else:
            self._is_connected = False
            log.warning("MQTT connection failed with rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc, *args):
        self._is_connected = False
        log.warning("MQTT disconnected (rc=%d). Will retry.", rc)

    def publish(self, event: ViolationEvent) -> None:
        if not getattr(self, "_is_connected", False):
            return  # Do not block or log when MQTT broker is not running
        topic = f"{self.BASE_TOPIC}/{event.zone_id}/violations"
        payload: dict[str, Any] = {
            "track_id": event.track_id,
            "zone_id": event.zone_id,
            "missing_ppe": event.missing_ppe,
            "timestamp": event.timestamp,
            "frame": event.frame_number,
        }
        try:
            result = self._client.publish(topic, json.dumps(payload), qos=self._qos)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                log.warning("MQTT publish failed: rc=%d", result.rc)
        except Exception as exc:
            log.warning("MQTT publish exception: %s", exc)

    def stop(self) -> None:
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass

"""
pi/azure_sender.py
--------------------
Optional: Send sensor readings and risk scores to Azure IoT Hub.

This module is NOT required for the core project.
It is included for the 'high-mark enhancement' (Section 22 of the plan).

To enable:
  1. Set "azure": {"enabled": true} in pi/config.json.
  2. Fill in the "connection_string" field with your Azure IoT Hub device
     connection string.
  3. Install: pip install azure-iot-device

Usage (called from sensor_reader.py or inference.py):
    from pi.azure_sender import AzureSender
    sender = AzureSender()
    sender.send(payload_dict)
"""

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.json"
if not CONFIG_PATH.exists():
    CONFIG_PATH = Path(__file__).parent / "config.example.json"

with open(CONFIG_PATH) as f:
    _CONFIG = json.load(f)

AZURE_CFG = _CONFIG.get("azure", {})
ENABLED   = AZURE_CFG.get("enabled", False)
CONN_STR  = AZURE_CFG.get("connection_string", "")


class AzureSender:
    """
    Thin wrapper around azure-iot-device IoTHubDeviceClient.
    Falls back to no-op logging when disabled or library not installed.
    """

    def __init__(self):
        self._client = None
        if not ENABLED:
            log.info("Azure IoT Hub: disabled in config (azure.enabled=false)")
            return
        if not CONN_STR or CONN_STR.startswith("YOUR_"):
            log.warning("Azure IoT Hub: no valid connection string in config.json")
            return
        try:
            from azure.iot.device import IoTHubDeviceClient, Message  # noqa: F401
            self._client = IoTHubDeviceClient.create_from_connection_string(CONN_STR)
            self._client.connect()
            log.info("Azure IoT Hub: connected")
        except ImportError:
            log.warning(
                "azure-iot-device not installed. "
                "Install with: pip install azure-iot-device"
            )
        except Exception as e:
            log.error(f"Azure IoT Hub connection failed: {e}")

    def send(self, payload: dict) -> bool:
        """
        Send a JSON payload to Azure IoT Hub.

        Parameters
        ----------
        payload : dict   Any serialisable dictionary (reading + risk score).

        Returns
        -------
        bool   True if sent successfully, False otherwise.
        """
        if self._client is None:
            log.debug(f"Azure (disabled): {payload}")
            return False
        try:
            from azure.iot.device import Message
            msg = Message(json.dumps(payload))
            msg.content_encoding  = "utf-8"
            msg.content_type      = "application/json"
            self._client.send_message(msg)
            log.info(f"Azure IoT Hub: sent message for {payload.get('fridge_id', '?')}")
            return True
        except Exception as e:
            log.error(f"Azure send failed: {e}")
            return False

    def disconnect(self):
        if self._client:
            self._client.disconnect()
            log.info("Azure IoT Hub: disconnected")

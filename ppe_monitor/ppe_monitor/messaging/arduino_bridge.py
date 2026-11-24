"""
Arduino Serial Bridge for Physical Traffic Light, Buzzer & Relay Actuation.
Sends real-time safety status commands to Arduino/ESP32 over USB Serial.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

try:
    import serial
except ImportError:
    serial = None


class ArduinoBridge:
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self._last_state = None
        self._last_sent_time = 0.0
        self.buzzer_active_low = True
        self._lock = threading.Lock()
        self._last_reconnect_attempt = 0.0

        if port and serial:
            self._connect()
        elif port and not serial:
            print("[ARDUINO/ESP WARNING] 'pyserial' is not installed. Run: pip install pyserial")

    def _connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            print(f"[ARDUINO/ESP] Successfully connected to serial port on {self.port}")
            self._last_state = None
            self.send_idle()
        except Exception as e:
            print(f"[ARDUINO/ESP WARNING] Could not open serial port {self.port} (ESP32 not attached or in use): {e}")
            self.ser = None

    def _write(self, msg: str):
        with self._lock:
            # Auto-reconnect if serial was dropped or closed, rate-limited to avoid I/O lag when hardware is not plugged in
            if not self.ser or not self.ser.is_open:
                now = time.time()
                if self.port and serial and (now - self._last_reconnect_attempt > 5.0):
                    self._last_reconnect_attempt = now
                    try:
                        self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
                        print(f"[ARDUINO/ESP] Reconnected serial port on {self.port}")
                    except Exception:
                        return
                else:
                    return

            try:
                self.ser.write((msg + "\n").encode())
                self.ser.flush()
            except Exception as e:
                print(f"[ARDUINO/ESP ERROR] Write failed: {e}. Resetting serial handle.")
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None

    def send_idle(self):
        now = time.time()
        if self._last_state != "IDLE" or (now - self._last_sent_time > 3.0):
            self._write("STATE:IDLE")
            self._last_state = "IDLE"
            self._last_sent_time = now

    def send_compliant(self):
        now = time.time()
        if self._last_state != "COMPLIANT" or (now - self._last_sent_time > 3.0):
            self._write("STATE:COMPLIANT")
            self._last_state = "COMPLIANT"
            self._last_sent_time = now

    def send_pending(self):
        now = time.time()
        if self._last_state != "PENDING" or (now - self._last_sent_time > 2.0):
            self._write("STATE:PENDING")
            self._last_state = "PENDING"
            self._last_sent_time = now

    def send_violation(self, track_id: int, missing_item: str):
        v_state = f"VIOLATION:{track_id}:{missing_item}"
        # The firmware deliberately sounds for a bounded interval. Re-sending a
        # continuing violation resets that timer and creates an unsafe siren.
        if self._last_state != v_state:
            msg = f"STATE:VIOLATION:{track_id}:{missing_item}"
            self._write(msg)
            self._last_state = v_state
            self._last_sent_time = time.time()

    def set_buzzer_polarity(self, active_low: bool):
        """Sends command to ESP32 to configure Active-Low vs Active-High buzzer module polarity."""
        cmd = "POLARITY:LOW" if active_low else "POLARITY:HIGH"
        self._write(cmd)
        self.buzzer_active_low = active_low

    def close(self):
        with self._lock:
            if self.ser and self.ser.is_open:
                try:
                    self.ser.write(b"STATE:IDLE\n")
                    self.ser.flush()
                    self.ser.close()
                except Exception:
                    pass
            self.ser = None

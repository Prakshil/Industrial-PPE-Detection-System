"""
main.py — Runs the full PPE monitoring pipeline.

Usage:
    python main.py --source 0                        # webcam
    python main.py --source rtsp://cam1/stream       # IP camera
    python main.py --source video.mp4                # video file
    python main.py --source 0 --zone main_entry --save-evidence
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure this directory is in sys.path so inner ppe_monitor is discoverable
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ppe_monitor.compliance.engine import ComplianceEngine
from ppe_monitor.compliance.policy_loader import PolicyLoader
from ppe_monitor.compliance.worker_state import ComplianceStatus
from ppe_monitor.detector.yolo_detector import PPEDetector
from ppe_monitor.display.visualizer import Visualizer
from ppe_monitor.messaging.mqtt_client import MQTTClient
from ppe_monitor.messaging.arduino_bridge import ArduinoBridge
from ppe_monitor.persistence.database import EventDatabase
from ppe_monitor.tracker.association import associate_ppe_to_workers
from ppe_monitor.tracker.bytetrack_wrapper import PersonTracker


def save_evidence(
    frame: np.ndarray,
    event,
    evidence_dir: Path,
) -> str:
    """Crops and saves the person region as evidence for a violation."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = int(event.timestamp)
    filename = f"track{event.track_id}_{event.missing_ppe}_{ts}.jpg"
    path = evidence_dir / filename
    cv2.imwrite(str(path), frame)
    return str(path)


def run(
    source: str,
    model_path: str,
    zone_id: str,
    config_path: str = "config/zones.yaml",
    mqtt_host: str = "localhost",
    mqtt_port: int = 1883,
    db_path: str = "events.db",
    save_evidence_flag: bool = False,
    display: bool = False,
    max_fps: int = 25,
    serial_port: str | None = None,
    required_ppe: str | None = None,
    conf_threshold: float = 0.25,
) -> None:

    # ── Component initialisation ──────────────────────────────────────────────
    cfg_p = Path(config_path)
    if not cfg_p.is_file() and (_HERE / config_path).is_file():
        config_path = str(_HERE / config_path)
    print(f"[INFO] Loading model: {model_path} (conf={conf_threshold})")
    detector  = PPEDetector(model_path, conf_threshold=conf_threshold)
    tracker   = PersonTracker()
    policy    = PolicyLoader(config_path)
    engine    = ComplianceEngine(policy)
    db        = EventDatabase(db_path)
    mqtt      = MQTTClient(mqtt_host, mqtt_port)
    arduino   = ArduinoBridge(serial_port)
    viz       = Visualizer()

    body_regions = policy.get_body_regions()
    zone_policy  = policy.get_policy(zone_id)

    if required_ppe:
        zone_policy.required_ppe = [p.strip().lower() for p in required_ppe.split(",") if p.strip()]

    print(f"[INFO] Zone: {zone_id} | Required PPE: {zone_policy.required_ppe}")
    print(f"[INFO] Opening source: {source}")

    if source.isdigit():
        src_id = int(source)
        if sys.platform.startswith("win"):
            # DirectShow prevents Windows Media Foundation (MSMF -1072873822) camera locks
            cap = cv2.VideoCapture(src_id, cv2.CAP_DSHOW)
            if not cap.isOpened() or not cap.read()[0]:
                cap.release()
                cap = cv2.VideoCapture(src_id)
        else:
            cap = cv2.VideoCapture(src_id)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}. Ensure no other application (Zoom, browser, camera app) is using the webcam.")

    # Optimize camera resolution for high FPS and low latency
    if source.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_idx = 0
    consecutive_empty_frames = 0
    frame_timestamps = []
    fps_display = 25.0

    try:
        while True:
            t_loop_start = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_empty_frames += 1
                if consecutive_empty_frames > 25:
                    print("[INFO] Camera stream ended or disconnected.")
                    break
                time.sleep(0.04)
                continue
            consecutive_empty_frames = 0

            # Rolling FPS measurement over last 15 frames
            frame_timestamps.append(t_loop_start)
            if len(frame_timestamps) > 15:
                frame_timestamps.pop(0)
            if len(frame_timestamps) >= 2:
                fps_display = (len(frame_timestamps) - 1) / max(frame_timestamps[-1] - frame_timestamps[0], 1e-4)

            frame_idx += 1

            # ── 1. Detect ─────────────────────────────────────────────────────
            persons_raw, ppe_items = detector.detect(frame)

            # ── 2. Track persons ──────────────────────────────────────────────
            tracked_persons = tracker.update(persons_raw)

            # ── 3. Spatial association ────────────────────────────────────────
            associations = associate_ppe_to_workers(
                tracked_persons, ppe_items, region_map=body_regions
            )

            # ── 4. Compliance evaluation ──────────────────────────────────────
            events = engine.update(associations, zone_id)

            # ── 5. Handle confirmed violation events ──────────────────────────
            if events:
                for event in events:
                    print(
                        f"[VIOLATION] frame={frame_idx} track={event.track_id} "
                        f"zone={event.zone_id} missing={event.missing_ppe}"
                    )
                    evidence_path = None
                    if save_evidence_flag:
                        evidence_path = save_evidence(
                            frame, event, Path("evidence")
                        )
                    db.log_event(event, evidence_path)
                    mqtt.publish(event)

            # Hardware Actuation strictly reflects current active compliance state
            violating_workers = [ws for ws in engine.worker_states.values() if ws.status == ComplianceStatus.VIOLATION]
            pending_workers = [ws for ws in engine.worker_states.values() if ws.status == ComplianceStatus.PENDING]

            if violating_workers:
                v_ws = violating_workers[0]
                missing_name = next(iter(v_ws.confirmed_violations), "helmet")
                arduino.send_violation(v_ws.track_id, str(missing_name))
            elif pending_workers:
                arduino.send_pending()
            elif tracked_persons:
                arduino.send_compliant()
            else:
                arduino.send_idle()

            # ── 6. Display ────────────────────────────────────────────────────
            if display:
                annotated = viz.draw(
                    frame,
                    tracked_persons,
                    ppe_items,
                    associations,
                    engine.worker_states,
                )
                # Clean, smooth FPS overlay
                cv2.putText(
                    annotated, f"FPS: {fps_display:.1f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )
                cv2.imshow("PPE Monitor", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Quit requested.")
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        mqtt.stop()
        arduino.close()
        db.close()
        print("[INFO] Pipeline stopped cleanly.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Industrial PPE Monitor")
    ap.add_argument("--source",         default="0",
                    help="Video source: 0=webcam, RTSP URL, or file path")
    ap.add_argument("--model",          default="runs/ppe_6class_60eps/ultra/ultra/weights/best.pt",
                    help="Path to trained YOLOv8 weights (.pt)")
    ap.add_argument("--zone",           default="main_entry",
                    help="Active zone ID (must match zones.yaml)")
    ap.add_argument("--config",         default="config/zones.yaml")
    ap.add_argument("--mqtt-host",      default="localhost")
    ap.add_argument("--mqtt-port",      type=int, default=1883)
    ap.add_argument("--serial-port",    default=None,
                    help="Arduino COM port (e.g., COM3 on Windows or /dev/ttyACM0 on Linux)")
    ap.add_argument("--required-ppe",   default=None,
                    help="Override required PPE (e.g., 'helmet' to test helmet only without flagging missing boots/vest)")
    ap.add_argument("--conf",           type=float, default=0.20,
                    help="YOLO detection confidence threshold (default: 0.20)")
    ap.add_argument("--db",             default="events.db")
    ap.add_argument("--save-evidence",  action="store_true",
                    help="Save a JPEG crop for each confirmed violation")
    ap.add_argument("--display",        action="store_true",
                    help="Enable separate OpenCV window (default: False, integrated in web dashboard)")
    ap.add_argument("--max-fps",        type=int, default=25)
    args = ap.parse_args()

    run(
        source=args.source,
        model_path=args.model,
        zone_id=args.zone,
        config_path=args.config,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        serial_port=args.serial_port,
        required_ppe=args.required_ppe,
        conf_threshold=args.conf,
        db_path=args.db,
        save_evidence_flag=args.save_evidence,
        display=args.display,
        max_fps=args.max_fps,
    )


if __name__ == "__main__":
    main()

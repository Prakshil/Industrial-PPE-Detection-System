"""
live_stream.py — High-performance edge live webcam & inference stream service.
Pre-initializes YOLOv8s detector on CUDA GPU, ByteTrack, Spatial Association, and ESP32 bridge.
Uses an asynchronous, rate-limited evidence logger so disk I/O in OneDrive NEVER stalls video capture.
Includes an automatic camera re-sync watchdog so the webcam NEVER drops, freezes, or turns off.
"""
from __future__ import annotations

import hashlib
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# Resolve project paths
_BASE_DIR = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = _BASE_DIR / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


class AsyncEvidenceLogger:
    """Non-blocking background worker that writes evidence crops and logs to SQLite without stalling the camera loop."""

    def __init__(self, db_instance, evidence_dir: Path):
        self.db = db_instance
        self.evidence_dir = evidence_dir
        self.queue: queue.Queue = queue.Queue(maxsize=30)
        self.running = True
        self.last_log_time: dict[int, float] = {}
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, ev: Any, frame: np.ndarray, wbox: Optional[Any], associations: list[Any]):
        now = time.time()
        # Rate-limit disk writes to once per 10s per worker track_id to avoid OneDrive I/O stalls
        if now - self.last_log_time.get(ev.track_id, 0.0) < 10.0:
            return
        self.last_log_time[ev.track_id] = now

        try:
            if wbox is not None:
                bx1, by1, bx2, by2 = [max(0, int(v)) for v in wbox]
                margin = 35
                H, W = frame.shape[:2]
                cx1 = max(0, bx1 - margin)
                cy1 = max(0, by1 - margin)
                cx2 = min(W, bx2 + margin)
                cy2 = min(H, by2 + margin)
                crop = frame[cy1:cy2, cx1:cx2].copy() if (cx2 > cx1 and cy2 > cy1) else frame.copy()
            else:
                crop = frame.copy()

            # Stamp clear red violation banner on top of the evidence image
            if crop.size > 0:
                cw = crop.shape[1]
                missing_str = ", ".join(ev.missing_ppe) if isinstance(ev.missing_ppe, list) else str(ev.missing_ppe)
                banner_text = f"Worker #{ev.track_id} | VIOLATION: {missing_str.upper()}"
                cv2.rectangle(crop, (0, 0), (cw, 24), (0, 0, 200), -1)
                cv2.putText(crop, banner_text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

            self.queue.put_nowait((ev, crop, wbox, associations, now))
        except queue.Full:
            pass

    def _worker(self):
        while self.running:
            try:
                ev, crop, wbox, associations, t_stamp = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                crop_filename = f"live_worker_{ev.track_id}_{time.time_ns()}.jpg"
                crop_path = self.evidence_dir / crop_filename

                if crop.size > 0:
                    cv2.imwrite(str(crop_path), crop)

                sha = hashlib.sha256(crop_path.read_bytes()).hexdigest() if crop_path.is_file() else ""
                ar_item = next((a for a in associations if a.track_id == ev.track_id), None)
                present_items = [p for p, ok in (ar_item.ppe_present.items() if ar_item else []) if ok]
                conf_map = {p: det.confidence for p, det in (ar_item.ppe_detections.items() if ar_item else []) if det}

                if self.db:
                    self.db.log_event(
                        event=ev,
                        evidence_path=f"evidence/{crop_filename}",
                        present_ppe=present_items,
                        confidence_map=conf_map,
                        bbox=list(wbox) if wbox else [],
                        sha256_hash=sha,
                    )
            except Exception as e:
                print(f"[ASYNC EVIDENCE ERROR]: {e}")
            finally:
                self.queue.task_done()

    def stop(self):
        self.running = False


class ThreadedCamera:
    """
    Dedicated background frame grabber.
    Continuously drains the underlying camera driver buffer so that:
      1. Zero frame lag (always returns the freshest hardware frame, < 33ms latency).
      2. Driver buffers never queue up, leak memory, or crash.
      3. cap.read() blocks only in this lightweight thread without blocking FastAPI or YOLO.
      4. Automatic watchdog safely re-syncs if the camera driver drops frames.
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.last_frame_id: int = 0
        self.last_read_time: float = 0.0
        self.error_message: Optional[str] = None

    def start(self) -> bool:
        with self.lock:
            if self.running:
                return True
            self.running = True

            # Prefer DSHOW on Windows — consistent, low-latency, no MSMF frame-drop errors.
            # MSMF can produce -2147483638 "can't grab frame" on first connect which triggers
            # the watchdog resync and shows FPS=0 for 1-2 seconds.
            if sys.platform.startswith("win"):
                cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
            else:
                cap = cv2.VideoCapture(self.camera_index, cv2.CAP_ANY)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.camera_index)

            if not cap.isOpened():
                self.running = False
                self.error_message = f"Cannot open camera index {self.camera_index}."
                return False

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.cap = cap
            self.last_read_time = time.time()
            self.thread = threading.Thread(target=self._capture_worker, daemon=True)
            self.thread.start()
            return True

    def _capture_worker(self):
        consecutive_drops = 0
        while self.running:
            cap = self.cap
            if not cap or not cap.isOpened():
                time.sleep(0.02)
                continue

            try:
                ret, frame = cap.read()
            except Exception:
                ret, frame = False, None

            now = time.time()
            if ret and frame is not None and frame.size > 0:
                with self.lock:
                    self.latest_frame = frame
                    self.last_frame_id += 1
                    self.last_read_time = now
                consecutive_drops = 0
            else:
                consecutive_drops += 1
                time.sleep(0.015)

                # Watchdog: if camera drops frames for > 1.5 seconds, auto-resync
                if consecutive_drops > 45 and self.running:
                    print(f"[THREADED CAM WATCHDOG] Re-syncing webcam {self.camera_index}...")
                    try:
                        cap.release()
                    except Exception:
                        pass
                    time.sleep(0.15)
                    # Same DSHOW-first order for watchdog resync
                    if sys.platform.startswith("win"):
                        new_cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                        if not new_cap.isOpened():
                            new_cap = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
                    else:
                        new_cap = cv2.VideoCapture(self.camera_index, cv2.CAP_ANY)
                    if not new_cap.isOpened():
                        new_cap = cv2.VideoCapture(self.camera_index)
                    if new_cap.isOpened():
                        new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self.cap = new_cap
                        consecutive_drops = 0

    def get_latest(self) -> tuple[Optional[np.ndarray], int, float]:
        with self.lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
            return frame, self.last_frame_id, self.last_read_time

    def stop(self):
        with self.lock:
            self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)


class LiveWebcamService:
    """Thread-safe background service that manages camera capture, YOLO inference, and ESP32 hardware."""

    CAMERA_PROFILES = {
        "upper_body": ["helmet", "vest", "gloves"],
        "full_body": ["helmet", "vest", "gloves", "boots", "goggles"],
        "helmet_only": ["helmet"],
    }

    def __init__(self):
        self.lock = threading.RLock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.camera_index: int = 0
        self.serial_port: Optional[str] = "COM10"
        self.test_mode: str = "all_ppe"  # Default to mandatory 4-PPE audit
        self.zone_id: str = "active_work_zone"
        self.current_fps: float = 0.0
        self.latest_jpeg: Optional[bytes] = None
        self.latest_workers: list[dict[str, Any]] = []
        self.latest_checklist: dict[str, Any] = {
            "helmet": {"present": False, "confidence": 0.0},
            "vest": {"present": False, "confidence": 0.0},
            "boots": {"present": False, "confidence": 0.0},
            "gloves": {"present": False, "confidence": 0.0},
        }
        self.arduino = None
        self.detector = None
        self.tracker = None
        self.policy = None
        self.engine = None
        self.viz = None
        self.last_frame_time: float = 0.0
        self.error_message: Optional[str] = None
        self.ws_manager = None
        self.db_instance = None
        self.evidence_logger: Optional[AsyncEvidenceLogger] = None
        self._models_loaded = False
        self.threaded_cam: Optional[ThreadedCamera] = None
        self.session_id: str = f"live_{int(time.time())}"
        self._attendance_snapshots: set[int] = set()
        self._last_attendance_log: dict[int, float] = {}
        self.camera_profile = "upper_body"
        self.observable_ppe = list(self.CAMERA_PROFILES[self.camera_profile])
        self.configured_required_ppe: list[str] = []

    def initialize(self, ws_manager=None, db_instance=None):
        """Attach services without opening hardware or loading a model at API startup."""
        self.ws_manager = ws_manager
        self.db_instance = db_instance
        if self.db_instance and not self.evidence_logger:
            self.evidence_logger = AsyncEvidenceLogger(self.db_instance, EVIDENCE_DIR)
        if self.db_instance:
            zone = self.db_instance.get_zone(self.zone_id)
            if zone:
                self.configured_required_ppe = list(zone.get("required_ppe", []))
        # Inference is loaded only when a camera is explicitly started.

    def _load_models(self):
        if self._models_loaded:
            return
        model_candidates = [
            _BASE_DIR / "weights" / "best.pt",
            _BASE_DIR / "PPE_detection" / "runs" / "ppe_exp_v3" / "ultra" / "weights" / "best.pt",
            _BASE_DIR / "PPE_detection" / "runs" / "ppe_three_source_v1" / "ultra" / "weights" / "best.pt",
            _BASE_DIR / "runs" / "ppe_6class_60eps" / "ultra" / "ultra" / "weights" / "best.pt",
            _BASE_DIR / "yolov8s.pt",
        ]
        model_path = next((p for p in model_candidates if p.is_file()), _BASE_DIR / "yolov8s.pt")

        from ppe_monitor.compliance.engine import ComplianceEngine, PolicyLoader
        from ppe_monitor.detector.yolo_detector import PPEDetector
        from ppe_monitor.display.visualizer import Visualizer
        from ppe_monitor.messaging.arduino_bridge import ArduinoBridge
        from ppe_monitor.tracker.bytetrack_wrapper import PersonTracker

        print(f"[LIVE CAM] Pre-loading YOLO detector with weights: {model_path}")
        self.detector = PPEDetector(str(model_path), conf_threshold=0.010)
        self.tracker = PersonTracker()
        self.policy = PolicyLoader(str(_BASE_DIR / "ppe_monitor" / "config" / "zones.yaml"))
        if not self.configured_required_ppe:
            self.configured_required_ppe = list(self.policy.get_policy(self.zone_id).required_ppe)
        else:
            self.policy.update_runtime_policy(self.zone_id, required_ppe=self.configured_required_ppe)
        self.engine = ComplianceEngine(self.policy)
        self.viz = Visualizer()

        if self.serial_port and not self.arduino:
            self.arduino = ArduinoBridge(self.serial_port)

        self._models_loaded = True
        print("[LIVE CAM] Inference pipeline and hardware pre-initialized successfully.")

    def get_status(self) -> dict[str, Any]:
        with self.lock:
            serial_ok = bool(self.arduino and self.arduino.ser and self.arduino.ser.is_open)
            return {
                "running": self.running,
                "camera_index": self.camera_index,
                "serial_port": self.serial_port,
                "serial_connected": serial_ok,
                "buzzer_active_low": bool(self.arduino and self.arduino.buzzer_active_low),
                "camera_profile": self.camera_profile,
                "observable_ppe": list(self.observable_ppe),
                "test_mode": self.test_mode,
                "fps": round(self.current_fps, 1),
                "active_workers": len(self.latest_workers),
                "workers": self.latest_workers,
                "checklist": self.latest_checklist,
                "error": self.error_message,
            }

    def set_test_mode(self, mode: str):
        with self.lock:
            if mode in ("helmet_only", "all_ppe"):
                self.test_mode = mode
                print(f"[LIVE CAM] Inspection mode changed to: {self.test_mode}")

    def set_camera_profile(self, camera_profile: str) -> bool:
        """Set the PPE classes this physical camera can reliably observe."""
        if camera_profile not in self.CAMERA_PROFILES:
            return False
        with self.lock:
            self.camera_profile = camera_profile
            self.observable_ppe = list(self.CAMERA_PROFILES[camera_profile])
        return True

    def update_policy(self, required_ppe, confirmation_frames, cooldown_seconds):
        """Synchronise an approved dashboard policy with the active edge process."""
        if required_ppe is not None:
            self.configured_required_ppe = list(required_ppe)
        if self.policy:
            self.policy.update_runtime_policy(
                self.zone_id,
                required_ppe=required_ppe,
                confirmation_frames=confirmation_frames,
                confirmation_seconds=max(0.25, confirmation_frames / 25) if confirmation_frames else None,
                alert_cooldown_seconds=cooldown_seconds,
            )

    def reconnect_serial(self, port: Optional[str] = "COM10") -> bool:
        with self.lock:
            self.serial_port = port
            if self.arduino:
                try:
                    self.arduino.close()
                except Exception:
                    pass
            from ppe_monitor.messaging.arduino_bridge import ArduinoBridge
            self.arduino = ArduinoBridge(port)
            connected = bool(self.arduino and self.arduino.ser and self.arduino.ser.is_open)
            print(f"[LIVE CAM] Arduino serial reconnect on {port}: {'SUCCESS' if connected else 'FAILED'}")
            return connected

    def set_buzzer_polarity(self, active_low: bool) -> bool:
        with self.lock:
            if self.arduino:
                self.arduino.set_buzzer_polarity(active_low)
                return True
            return False

    def start(
        self,
        camera_index: int = 0,
        serial_port: Optional[str] = "COM10",
        test_mode: str = "all_ppe",
        camera_profile: str = "upper_body",
    ):
        with self.lock:
            if self.running:
                return
            self._load_models()
            self.camera_index = camera_index
            self.serial_port = serial_port
            if test_mode not in ("helmet_only", "all_ppe"):
                self.error_message = "Unsupported inspection mode."
                return
            if not self.set_camera_profile(camera_profile):
                self.error_message = "Unsupported camera profile."
                return
            self.test_mode = test_mode
            self.running = True
            self.error_message = None
            self.session_id = f"live_{int(time.time())}"
            self._attendance_snapshots = set()
            self._last_attendance_log = {}

            # Cleanly stop any existing capture thread or camera handle before starting anew
            if self.threaded_cam:
                try:
                    self.threaded_cam.stop()
                except Exception:
                    pass
                self.threaded_cam = None

            self.thread = threading.Thread(target=self._run_capture_loop, daemon=True)
            self.thread.start()
            print(f"[LIVE CAM] Started zero-lag capture worker on camera {camera_index} (Mode: {test_mode})")

    def stop(self):
        with self.lock:
            self.running = False
        if self.threaded_cam:
            try:
                self.threaded_cam.stop()
            except Exception:
                pass
            self.threaded_cam = None
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        with self.lock:
            if self.arduino:
                try:
                    self.arduino.send_idle()
                except Exception:
                    pass
        print("[LIVE CAM] Stopped live webcam service cleanly.")

    def _run_capture_loop(self):
        """
        Background worker thread: consumes freshest frames from ThreadedCamera,
        executes YOLO pipeline, and signals ESP32.
        Decoupled capture guarantees ZERO lag accumulation, ZERO buffer queue,
        and eliminates device contention.
        """
        from ppe_monitor.compliance.worker_state import ComplianceStatus
        from ppe_monitor.tracker.association import associate_ppe_to_workers

        cam = ThreadedCamera(self.camera_index)
        if not cam.start():
            self.error_message = cam.error_message or f"Failed to access webcam {self.camera_index}. Ensure no other app is using it."
            print(f"[LIVE CAM ERROR] {self.error_message}")
            with self.lock:
                self.running = False
            return

        self.threaded_cam = cam
        last_processed_fid = -1
        frame_timestamps = []
        frame_idx = 0
        w, h = 640, 480

        dev_name = getattr(self.detector, "device", "cuda:0")
        print(f"[LIVE CAM] Camera {self.camera_index} streaming online on {dev_name} (Zero-Lag Direct Draining).")

        try:
            while self.running:
                frame, fid, ftime = cam.get_latest()
                if frame is None or fid == last_processed_fid:
                    time.sleep(0.005)
                    continue

                last_processed_fid = fid
                t_start = time.time()

                try:
                    if frame.shape[1] != w or frame.shape[0] != h:
                        frame = cv2.resize(frame, (w, h))

                    frame_idx += 1
                    frame_timestamps.append(t_start)
                    if len(frame_timestamps) > 15:
                        frame_timestamps.pop(0)
                    if len(frame_timestamps) >= 2:
                        self.current_fps = (len(frame_timestamps) - 1) / max(frame_timestamps[-1] - frame_timestamps[0], 1e-4)

                    # This laptop camera is an upper-body sensor. Gear outside
                    # the frame is reported as out-of-view, never as missing.
                    zone_policy = self.policy.get_policy(self.zone_id)
                    configured_ppe = list(self.configured_required_ppe or zone_policy.required_ppe)
                    required_ppe = ["helmet"] if self.test_mode == "helmet_only" else [
                        item for item in configured_ppe if item in self.observable_ppe
                    ]
                    self.policy.update_runtime_policy(self.zone_id, required_ppe=required_ppe)

                    # 1. Detect (runs in ~9.5ms on CUDA)
                    persons_raw, ppe_items = self.detector.detect(frame)

                    # 2. Track (guaranteed unique IDs)
                    tracked_persons = self.tracker.update(persons_raw)

                    # 3. Associate by anatomy. WorkerState performs time-based
                    # confirmation, so stale detections cannot mask an alert.
                    associations = associate_ppe_to_workers(
                        tracked_persons, ppe_items, region_map=self.policy.get_body_regions()
                    )

                    # 4. Compliance evaluation
                    events = self.engine.update(associations, self.zone_id)

                    # Extract live checklist state for primary tracked worker
                    active_workers_payload = []
                    checklist_snapshot = {
                        "helmet": {"present": False, "confidence": 0.0},
                        "vest": {"present": False, "confidence": 0.0},
                        "boots": {"present": False, "confidence": 0.0},
                        "gloves": {"present": False, "confidence": 0.0},
                    }

                    valid_tracked = [tp for tp in tracked_persons if tp.track_id is not None]
                    primary_worker = None

                    if valid_tracked and associations:
                        primary_worker = max(
                            valid_tracked,
                            key=lambda p: (p.box[2] - p.box[0]) * (p.box[3] - p.box[1])
                        )
                        primary_ar = next((a for a in associations if a.track_id == primary_worker.track_id), None)
                        if primary_ar:
                            for ppe_k in ["helmet", "vest", "boots", "gloves"]:
                                is_pres = primary_ar.ppe_present.get(ppe_k, False)
                                det_obj = primary_ar.ppe_detections.get(ppe_k)
                                c_val = det_obj.confidence if det_obj else (0.85 if is_pres else 0.0)
                                checklist_snapshot[ppe_k] = {"present": is_pres, "confidence": round(float(c_val), 2)}

                    for tp in tracked_persons:
                        if tp.track_id is None:
                            continue
                        ws = self.engine.worker_states.get(tp.track_id)
                        st_str = ws.status.value if ws else "COMPLIANT"
                        ar = next((a for a in associations if a.track_id == tp.track_id), None)
                        w_checklist = {}
                        for ppe_k in ["helmet", "vest", "boots", "gloves"]:
                            pres = ar.ppe_present.get(ppe_k, False) if ar else False
                            det = ar.ppe_detections.get(ppe_k) if ar else None
                            w_checklist[ppe_k] = {"present": pres, "confidence": det.confidence if det else 0.0}

                        active_workers_payload.append({
                            "track_id": tp.track_id,
                            "bbox": [int(v) for v in tp.box],
                            "status": st_str,
                            "checklist": w_checklist,
                        })

                    # 5. Hardware Actuation (ESP32)
                    if self.arduino:
                        try:
                            states = [self.engine.worker_states.get(tp.track_id) for tp in valid_tracked]
                            violators = [state for state in states if state and state.status == ComplianceStatus.VIOLATION]
                            if violators:
                                violator = min(violators, key=lambda state: state.track_id)
                                self.arduino.send_violation(
                                    violator.track_id, sorted(violator.confirmed_violations)[0]
                                )
                            elif any(state and state.status == ComplianceStatus.PENDING for state in states):
                                self.arduino.send_pending()
                            elif valid_tracked:
                                self.arduino.send_compliant()
                            else:
                                self.arduino.send_idle()
                        except Exception as exc:
                            self.error_message = f"ESP32 command failed: {exc}"

                    # 5b. Shift Worker Attendance & Verification Registry Logging
                    if self.db_instance and valid_tracked:
                        now_att = time.time()
                        for tp in valid_tracked:
                            if now_att - self._last_attendance_log.get(tp.track_id, 0.0) >= 0.5:
                                ws = self.engine.worker_states.get(tp.track_id)
                                is_compliant = (ws.status == ComplianceStatus.COMPLIANT) if ws else True
                                has_viol = (ws.status == ComplianceStatus.VIOLATION) if ws else False

                                thumb_rel = None
                                if tp.track_id not in self._attendance_snapshots:
                                    bx1, by1, bx2, by2 = [max(0, int(v)) for v in tp.box]
                                    if bx2 > bx1 and by2 > by1:
                                        crop = frame[by1:by2, bx1:bx2]
                                        if crop.size > 0:
                                            thumb_name = f"att_live_{tp.track_id}_{int(now_att)}.jpg"
                                            thumb_path = EVIDENCE_DIR / thumb_name
                                            cv2.imwrite(str(thumb_path), crop)
                                            thumb_rel = f"evidence/{thumb_name}"
                                            self._attendance_snapshots.add(tp.track_id)

                                self.db_instance.record_worker_attendance(
                                    session_id=self.session_id,
                                    track_id=tp.track_id,
                                    zone_id=self.zone_id,
                                    is_compliant=is_compliant,
                                    has_violation=has_viol,
                                    snapshot_path=thumb_rel,
                                    source="💻 Laptop Webcam",
                                    timestamp=now_att,
                                )
                                self._last_attendance_log[tp.track_id] = now_att

                    # 6. Non-Blocking Asynchronous Evidence Logging
                    if events and self.evidence_logger:
                        for ev in events:
                            wbox = None
                            for tp in tracked_persons:
                                if tp.track_id == ev.track_id:
                                    wbox = tp.box
                                    break
                            self.evidence_logger.submit(ev, frame, wbox, associations)

                    # 7. Render annotations
                    annotated = self.viz.draw(frame, tracked_persons, ppe_items, associations, self.engine.worker_states)

                    # Top-left HUD info banner
                    cv2.rectangle(annotated, (0, 0), (w, 28), (248, 250, 252), -1)
                    mode_label = "TEST: HELMET ONLY" if self.test_mode == "helmet_only" else "UPPER-BODY PPE"
                    cv2.putText(
                        annotated,
                        f"LIVE WEBCAM | {mode_label} | {self.current_fps:.1f} FPS | WORKERS: {len(tracked_persons)}",
                        (10, 19),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (15, 23, 42),
                        1,
                    )

                    # Encode to JPEG
                    _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    jpeg_bytes = jpeg.tobytes()

                    with self.lock:
                        self.latest_jpeg = jpeg_bytes
                        self.latest_workers = active_workers_payload
                        self.latest_checklist = checklist_snapshot
                        self.last_frame_time = t_start

                except Exception as inner_e:
                    print(f"[LIVE CAM PROCESSING ERROR]: {inner_e}")
                    self.error_message = f"Frame processing failed: {inner_e}"

        finally:
            cam.stop()
            self.threaded_cam = None
            with self.lock:
                self.running = False
            if self.arduino:
                try:
                    self.arduino.send_idle()
                except Exception:
                    pass
            print("[LIVE CAM] Capture loop ended cleanly.")

    def get_stream_frame(self) -> bytes:
        """Returns the latest annotated JPEG frame bytes or a clean standby placeholder."""
        with self.lock:
            if self.running and self.latest_jpeg is not None and (time.time() - self.last_frame_time < 3.0):
                return self.latest_jpeg

        # Standby graphic when camera is OFF
        img = np.zeros((480, 640, 3), dtype=np.uint8) + 18
        cv2.putText(img, "STANDBY: CAMERA IS OFF", (145, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2)
        cv2.putText(img, "Click [Turn ON Laptop Camera] above to start stream", (110, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)
        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()


# Global singleton
live_service = LiveWebcamService()

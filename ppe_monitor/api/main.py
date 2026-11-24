"""
FastAPI central industrial safety platform backend.
Provides real-time WebSockets telemetry, RESTful triage endpoints,
video streaming, and offline-edge database synchronization.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

# Ensure both repository root and ppe_monitor package directory are in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
for _p in [str(_REPO_ROOT), str(_PACKAGE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hashlib
import html
import shutil

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

try:
    from ppe_monitor.persistence.database import EventDatabase
except (ModuleNotFoundError, ImportError):
    if str(_PACKAGE_DIR / "ppe_monitor") not in sys.path:
        sys.path.insert(0, str(_PACKAGE_DIR / "ppe_monitor"))
    from persistence.database import EventDatabase

# Path configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "events.db"
EVIDENCE_DIR = BASE_DIR / "evidence"
UPLOAD_DIR = BASE_DIR / "uploads"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_VIDEO_PATH = EVIDENCE_DIR / "latest_processed_feed.mp4"

# Global database instance
db = EventDatabase(DB_PATH)

try:
    from ppe_monitor.api.live_stream import live_service
except ImportError:
    from api.live_stream import live_service


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict[str, Any]):
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        for d in dead:
            if d in self.active_connections:
                self.active_connections.remove(d)


manager = ConnectionManager()
_last_live_telemetry_log = 0.0
_event_loop: asyncio.AbstractEventLoop | None = None  # stored at startup for cross-thread use


# Background telemetry broadcaster: emits real live edge data from active webcam or video stream
async def telemetry_broadcaster():
    """Pushes real-time worker detections and 4-item checklists to WebSocket clients."""
    global _last_live_telemetry_log, _event_loop
    _event_loop = asyncio.get_event_loop()  # capture for video stream thread
    while True:
        try:
            now = time.time()
            # Priority: video stream active → use video stream telemetry
            if video_stream_service.running or (
                video_stream_service.done
                and not video_stream_service.error
                and video_stream_service.frames_processed > 0
            ):
                vs = video_stream_service.get_status()
                workers = vs.get("workers", [])
                compliant_c = sum(1 for w in workers if w.get("status") == "COMPLIANT")
                violation_c = sum(1 for w in workers if w.get("status") == "VIOLATION")
                payload = {
                    "type": "telemetry_tick",
                    "timestamp": now,
                    "zone_id": live_service.zone_id,
                    "fps": vs.get("fps", 0.0),
                    "active_workers": len(workers),
                    "compliant_count": compliant_c,
                    "violation_count": violation_c,
                    "workers": workers,
                    "checklist": vs.get("checklist", {}),
                    "test_mode": "all_ppe",
                    "serial_connected": bool(
                        live_service.arduino
                        and live_service.arduino.ser
                        and live_service.arduino.ser.is_open
                    ),
                    "source": "video_stream",
                    "frames_processed": vs.get("frames_processed", 0),
                    "total_frames": vs.get("total_frames", 0),
                    "done": vs.get("done", False),
                }
            elif live_service.running:
                status = live_service.get_status()
                workers = status.get("workers", [])
                compliant_c = sum(1 for w in workers if w.get("status") == "COMPLIANT")
                violation_c = sum(1 for w in workers if w.get("status") == "VIOLATION")
                payload = {
                    "type": "telemetry_tick",
                    "timestamp": now,
                    "zone_id": live_service.zone_id,
                    "fps": status.get("fps", 0.0),
                    "active_workers": len(workers),
                    "compliant_count": compliant_c,
                    "violation_count": violation_c,
                    "workers": workers,
                    "checklist": status.get("checklist", {}),
                    "test_mode": status.get("test_mode", "all_ppe"),
                    "serial_connected": status.get("serial_connected", False),
                    "error": status.get("error"),
                    "camera_profile": status.get("camera_profile", "upper_body"),
                    "observable_ppe": status.get("observable_ppe", []),
                }
                if now - _last_live_telemetry_log >= 5.0:
                    db.log_telemetry(
                        zone_id=live_service.zone_id,
                        camera_id="cam_live",
                        active_workers=len(workers),
                        compliant_count=compliant_c,
                        violation_count=violation_c,
                        fps=status.get("fps", 0.0),
                    )
                    _last_live_telemetry_log = now
            else:
                payload = {
                    "type": "telemetry_tick",
                    "timestamp": now,
                    "zone_id": "active_work_zone",
                    "fps": 0.0,
                    "active_workers": 0,
                    "compliant_count": 0,
                    "violation_count": 0,
                    "workers": [],
                    "checklist": {
                        "helmet": {"present": False, "confidence": 0.0},
                        "vest": {"present": False, "confidence": 0.0},
                        "boots": {"present": False, "confidence": 0.0},
                        "gloves": {"present": False, "confidence": 0.0},
                    },
                    "test_mode": live_service.test_mode,
                    "serial_connected": bool(
                        live_service.arduino
                        and live_service.arduino.ser
                        and live_service.arduino.ser.is_open
                    ),
                }
            try:
                payload["attendance_summary"] = db.get_attendance_summary()
            except Exception:
                pass
            await manager.broadcast(payload)
        except Exception as exc:
            print(f"[TELEMETRY ERROR] {exc}")
        await asyncio.sleep(0.4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    live_service.initialize(ws_manager=manager, db_instance=db)
    try:
        db.register_or_update_camera(
            camera_id="cam_live",
            zone_id="active_work_zone",
            name="💻 Live Webcam (Direct Edge)",
            source_url="webcam://0",
            is_online=1,
        )
    except Exception as e:
        print(f"[WARN] Could not register cam_live: {e}")

    task = asyncio.create_task(telemetry_broadcaster())
    yield
    task.cancel()
    live_service.stop()


app = FastAPI(
    title="Industrial PPE Safety Intelligence API",
    description="Backend API for edge monitoring, real-time telemetry, and safety compliance dashboard",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Request Schemas ──────────────────────────────────────────────────
class ViolationActionRequest(BaseModel):
    action: str = Field(..., description="Action: CONFIRMED, DISMISSED, or FALSE_POSITIVE")
    reviewed_by: str = Field("Safety Officer", description="Name/badge of reviewer")
    notes: str = Field("", description="Review notes or justification")


class PolicyUpdateRequest(BaseModel):
    required_ppe: list[str] | None = None
    n_frame_confirm: int | None = Field(default=None, ge=1, le=300)
    alert_cooldown_seconds: int | None = Field(default=None, ge=1, le=3600)


class EdgeEventSyncItem(BaseModel):
    track_id: int
    zone_id: str
    missing_ppe: list[str] | str
    present_ppe: list[str] = []
    confidence_map: dict[str, float] = {}
    bbox: list[int] = []
    timestamp: float
    frame_number: int
    evidence_path: str = ""
    evidence_sha256: str = ""


class EdgeSyncBatch(BaseModel):
    edge_node: str = "edge_gateway_01"
    events: list[EdgeEventSyncItem]


class LiveCameraControlRequest(BaseModel):
    camera_index: int = 0
    serial_port: str = "COM10"
    mode: str = "all_ppe"
    camera_profile: str = "upper_body"


class LiveCameraModeRequest(BaseModel):
    mode: str = "all_ppe"


class LiveCameraProfileRequest(BaseModel):
    camera_profile: str


class LiveCameraSerialRequest(BaseModel):
    port: str = "COM10"


class BuzzerPolarityRequest(BaseModel):
    active_low: bool = False


# ── API Endpoints ─────────────────────────────────────────────────────────────
@app.get("/api/v1/health", summary="System Health & Status")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "Industrial PPE Safety Gateway",
        "version": "2.0.0",
        "database": "SQLite WAL active",
        "timestamp": time.time(),
    }


@app.get("/api/v1/camera/live/status", summary="Get status of live webcam edge service")
def get_live_camera_status() -> dict[str, Any]:
    return live_service.get_status()


@app.post("/api/v1/camera/live/start", summary="Start live webcam capture & YOLO edge inference")
def start_live_camera(req: LiveCameraControlRequest | None = None) -> dict[str, Any]:
    idx = req.camera_index if req else 0
    port = req.serial_port if req else "COM10"
    mode = req.mode if req else "helmet_only"
    profile = req.camera_profile if req else "upper_body"
    live_service.start(camera_index=idx, serial_port=port, test_mode=mode, camera_profile=profile)
    db.register_or_update_camera(
        camera_id="cam_live",
        zone_id=live_service.zone_id,
        name="💻 Live Camera",
        source_url=f"webcam://{idx}",
        is_online=1,
        camera_profile=profile,
    )
    return {"status": "started", "details": live_service.get_status()}


@app.post("/api/v1/camera/live/stop", summary="Stop live webcam capture")
def stop_live_camera() -> dict[str, Any]:
    live_service.stop()
    return {"status": "stopped", "details": live_service.get_status()}


@app.post("/api/v1/camera/live/mode", summary="Set inspection mode (helmet_only vs all_ppe)")
def set_live_camera_mode(req: LiveCameraModeRequest) -> dict[str, Any]:
    if req.mode not in {"helmet_only", "all_ppe"}:
        raise HTTPException(status_code=422, detail="Mode must be 'helmet_only' or 'all_ppe'.")
    live_service.set_test_mode(req.mode)
    return {"status": "updated", "mode": live_service.test_mode}


@app.post("/api/v1/camera/live/profile", summary="Set physical camera framing profile")
def set_live_camera_profile(req: LiveCameraProfileRequest) -> dict[str, Any]:
    if not live_service.set_camera_profile(req.camera_profile):
        raise HTTPException(status_code=422, detail="Unsupported camera profile.")
    return {"status": "updated", "camera_profile": live_service.camera_profile}


@app.post("/api/v1/camera/live/reconnect-serial", summary="Reconnect ESP32 on specified COM port")
def reconnect_live_serial(req: LiveCameraSerialRequest) -> dict[str, Any]:
    ok = live_service.reconnect_serial(req.port)
    return {"status": "connected" if ok else "failed", "port": req.port, "is_open": ok}


@app.post(
    "/api/v1/camera/live/buzzer-polarity",
    summary="Set ESP32 buzzer active-low vs active-high polarity",
)
def set_live_buzzer_polarity(req: BuzzerPolarityRequest) -> dict[str, Any]:
    ok = live_service.set_buzzer_polarity(req.active_low)
    return {"status": "ok" if ok else "failed", "active_low": req.active_low}


@app.get("/api/v1/zones", summary="List all safety zones and policies")
def get_zones() -> list[dict[str, Any]]:
    return db.get_zones()


@app.post("/api/v1/zones/{zone_id}/policy", summary="Update zone safety policy")
def update_zone_policy(zone_id: str, req: PolicyUpdateRequest) -> dict[str, Any]:
    supported = {"helmet", "vest", "boots", "gloves", "goggles"}
    if req.required_ppe is not None and (
        not req.required_ppe or any(item not in supported for item in req.required_ppe)
    ):
        raise HTTPException(status_code=422, detail="Select one or more supported PPE classes.")
    ok = db.update_zone_policy(
        zone_id=zone_id,
        required_ppe=req.required_ppe,
        n_frame_confirm=req.n_frame_confirm,
        alert_cooldown_seconds=req.alert_cooldown_seconds,
    )
    if not ok:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found")
    if zone_id == live_service.zone_id:
        live_service.update_policy(
            req.required_ppe, req.n_frame_confirm, req.alert_cooldown_seconds
        )
    return {"status": "updated", "zone": db.get_zone(zone_id)}


@app.get("/api/v1/cameras", summary="List registered edge cameras")
def get_cameras() -> list[dict[str, Any]]:
    return [
        {
            "camera_id": "cam_live",
            "zone_id": "active_work_zone",
            "name": "Live Laptop Webcam",
            "source_url": "webcam://0",
            "is_online": 1 if live_service.running else 0,
        }
    ]


@app.get("/api/v1/violations", summary="List filterable violation events")
def get_violations(
    zone_id: str | None = Query("all", description="Filter by zone ID"),
    status: str | None = Query(
        "all", description="Filter by status (UNREVIEWED, CONFIRMED, DISMISSED, FALSE_POSITIVE)"
    ),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[dict[str, Any]]:
    return db.get_violations(zone_id=zone_id, status=status, limit=limit, offset=offset)


@app.get("/api/v1/violations/{violation_id}", summary="Get detailed violation by ID")
def get_violation(violation_id: int) -> dict[str, Any]:
    v = db.get_violation_by_id(violation_id)
    if not v:
        raise HTTPException(status_code=404, detail="Violation record not found")
    return v


@app.patch("/api/v1/violations/{violation_id}/action", summary="Triage violation status")
def update_violation_action(violation_id: int, req: ViolationActionRequest) -> dict[str, Any]:
    ok = db.update_violation_status(
        violation_id=violation_id,
        status=req.action,
        reviewed_by=req.reviewed_by,
        review_notes=req.notes,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Violation record not found or invalid action")
    return {"status": "success", "violation_id": violation_id, "new_status": req.action.upper()}


@app.delete("/api/v1/violations/{violation_id}", summary="Delete a single violation record")
def delete_violation_record(violation_id: int) -> dict[str, Any]:
    ok = db.delete_violation(violation_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Violation record not found")
    return {"status": "success", "deleted_id": violation_id}


@app.delete("/api/v1/violations", summary="Clear all violation records from ledger")
def clear_all_violation_records() -> dict[str, Any]:
    count = db.clear_all_violations()
    return {"status": "success", "cleared_count": count}


@app.get("/api/v1/telemetry", summary="Aggregated safety metrics and KPI summary")
def get_telemetry_summary(hours: int = Query(24, ge=1, le=168)) -> dict[str, Any]:
    return db.get_telemetry_summary(hours=hours)


@app.post("/api/v1/edge/sync", summary="Batch ingestion from offline edge nodes")
def sync_edge_events(batch: EdgeSyncBatch) -> dict[str, Any]:
    count = 0
    for item in batch.events:
        missing_list = (
            item.missing_ppe if isinstance(item.missing_ppe, list) else [item.missing_ppe]
        )
        dummy_event = type(
            "DummyEvent",
            (),
            {
                "track_id": item.track_id,
                "zone_id": item.zone_id,
                "missing_ppe": missing_list,
                "timestamp": item.timestamp,
                "frame_number": item.frame_number,
            },
        )()
        db.log_event(
            event=dummy_event,
            evidence_path=item.evidence_path,
            present_ppe=item.present_ppe,
            confidence_map=item.confidence_map,
            bbox=item.bbox,
            sha256_hash=item.evidence_sha256,
        )
        count += 1
    return {"status": "synced", "received_records": count, "edge_node": batch.edge_node}


@app.get(
    "/api/v1/export/audit", summary="Export compliance audit records (CSV / Excel / PDF / JSON)"
)
def export_audit_log(
    format: str = Query("csv", pattern="^(csv|json|excel|xlsx|pdf)$"),
    zone_id: str | None = "all",
) -> Response:
    violations = db.get_violations(zone_id=zone_id, limit=1000)
    if format == "json":
        return Response(content=json.dumps(violations, indent=2), media_type="application/json")

    if format == "pdf":
        try:
            from ppe_monitor.api.pdf_report import generate_compliance_audit_pdf
        except ImportError:
            from api.pdf_report import generate_compliance_audit_pdf
        pdf_bytes = generate_compliance_audit_pdf(violations, zone_id=zone_id)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=ppe_safety_audit_{int(time.time())}.pdf"
            },
        )

    if format in ("excel", "xlsx"):
        # Generate an HTML-based Microsoft Excel spreadsheet with formatted column widths and styles
        rows_html = []
        for v in violations:
            ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(v["timestamp"]))
            missing = ", ".join(v.get("missing_ppe", []))
            present = ", ".join(v.get("present_ppe", []))
            status = v.get("review_status", "UNREVIEWED")
            status_style = (
                "background-color:#fee2e2;color:#991b1b;font-weight:bold;text-align:center;"
                if status == "CONFIRMED"
                else (
                    "background-color:#fef3c7;color:#92400e;font-weight:bold;text-align:center;"
                    if status == "UNREVIEWED"
                    else "background-color:#f1f5f9;color:#334155;text-align:center;"
                )
            )
            rows_html.append(f"""
            <tr>
              <td style="text-align:center;font-weight:bold;">{v["id"]}</td>
              <td style="font-family:Consolas,monospace;font-size:9pt;">{html.escape(str(v.get("event_uuid", "")))}</td>
              <td style="vnd.ms-excel.numberformat:yyyy-mm-dd hh:mm:ss;white-space:nowrap;text-align:center;">{ts_str}</td>
              <td style="font-weight:600;">{html.escape(str(v.get("zone_id", "")).replace("_", " ").upper())}</td>
              <td style="text-align:center;font-weight:bold;">{v.get("track_id", "")}</td>
              <td style="color:#b91c1c;font-weight:bold;">{html.escape(missing.upper())}</td>
              <td style="color:#15803d;">{html.escape(present)}</td>
              <td style="{status_style}">{status}</td>
              <td>{html.escape(str(v.get("reviewed_by", "") or "Pending Review"))}</td>
              <td>{html.escape(str(v.get("review_notes", "") or ""))}</td>
              <td style="font-family:Consolas,monospace;font-size:9pt;">{html.escape(str(v.get("evidence_sha256", "")))}</td>
            </tr>
            """)

        excel_content = f"""<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:x="urn:schemas-microsoft-com:office:excel" xmlns="http://www.w3.org/TR/REC-html40">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<!--[if gte mso 9]><xml><x:ExcelWorkbook><x:ExcelWorksheets><x:ExcelWorksheet><x:Name>PPE Safety Audit</x:Name><x:WorksheetOptions><x:DisplayGridlines/></x:WorksheetOptions></x:ExcelWorksheet></x:ExcelWorksheets></x:ExcelWorkbook></xml><![endif]-->
<style>
  body {{ font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 10pt; }}
  table {{ border-collapse: collapse; }}
  th {{ background-color: #1e3a8a; color: #ffffff; font-weight: bold; padding: 10px 14px; border: 1px solid #cbd5e1; font-size: 10pt; }}
  td {{ padding: 8px 12px; border: 1px solid #cbd5e1; vertical-align: middle; }}
  tr:nth-child(even) {{ background-color: #f8fafc; }}
</style>
</head>
<body>
  <h2 style="color:#0f172a;margin-bottom:4px;">OSHA &amp; ISO 45001 Industrial Safety Compliance Audit Ledger</h2>
  <p style="color:#64748b;font-size:9pt;margin-top:0px;margin-bottom:12px;">Exported: {time.strftime('%Y-%m-%d %H:%M:%S')} Local | Plant: Safety Intelligence Sentinel</p>
  <table>
    <colgroup>
      <col style="width: 65px;">
      <col style="width: 270px;">
      <col style="width: 160px;">
      <col style="width: 180px;">
      <col style="width: 85px;">
      <col style="width: 170px;">
      <col style="width: 200px;">
      <col style="width: 120px;">
      <col style="width: 180px;">
      <col style="width: 320px;">
      <col style="width: 300px;">
    </colgroup>
    <thead>
      <tr>
        <th>Event ID</th>
        <th>Audit UUID</th>
        <th>Timestamp</th>
        <th>Zone</th>
        <th>Worker ID</th>
        <th>Missing PPE</th>
        <th>Verified PPE</th>
        <th>Review Status</th>
        <th>Reviewed By</th>
        <th>Audit Notes &amp; Citations</th>
        <th>SHA-256 Verification Hash</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>"""
        return Response(
            content=excel_content,
            media_type="application/vnd.ms-excel",
            headers={
                "Content-Disposition": f"attachment; filename=ppe_safety_audit_{int(time.time())}.xls"
            },
        )

    # Standard CSV with UTF-8 BOM for Microsoft Excel on Windows
    output = io.StringIO()
    output.write("\ufeff")  # Byte Order Mark for Excel
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(
        [
            "Event ID",
            "UUID",
            "Timestamp (Local)",
            "Zone ID",
            "Worker ID",
            "Missing PPE",
            "Present PPE",
            "Review Status",
            "Reviewed By",
            "Audit Notes",
            "SHA-256 Hash",
        ]
    )
    for v in violations:
        writer.writerow(
            [
                v["id"],
                v.get("event_uuid", ""),
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(v["timestamp"])),
                v["zone_id"],
                v["track_id"],
                ", ".join(v.get("missing_ppe", [])),
                ", ".join(v.get("present_ppe", [])),
                v.get("review_status", "UNREVIEWED"),
                v.get("reviewed_by", "") or "Unreviewed",
                v.get("review_notes", "") or "None",
                v.get("evidence_sha256", ""),
            ]
        )
    return Response(
        content=output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename=ppe_safety_audit_{int(time.time())}.csv"
        },
    )


# ── Worker Shift Attendance & Registry Endpoints ──────────────────────────────
@app.get("/api/v1/attendance", summary="List worker shift attendance and compliance registry")
def get_attendance(
    session_id: str | None = Query("all", description="Filter by session ID"),
    status: str | None = Query("all", description="Filter by status (PASSED or FLAGGED)"),
    zone_id: str | None = Query("all", description="Filter by zone ID"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    records = db.get_attendance(
        session_id=session_id, status=status, zone_id=zone_id, limit=limit, offset=offset
    )
    summary = db.get_attendance_summary()
    return {"attendance": records, "summary": summary}


@app.delete("/api/v1/attendance", summary="Clear all worker shift attendance records")
def clear_attendance_records() -> dict[str, Any]:
    cnt = db.clear_attendance()
    return {"status": "success", "cleared_count": cnt}


@app.get("/api/v1/export/attendance", summary="Export attendance roster (CSV / Excel / JSON)")
def export_attendance_roster(
    format: str = Query("csv", pattern="^(csv|json|excel|xlsx)$"),
    session_id: str | None = "all",
    status: str | None = "all",
    zone_id: str | None = "all",
) -> Response:
    records = db.get_attendance(session_id=session_id, status=status, zone_id=zone_id, limit=2000)
    if format == "json":
        return Response(content=json.dumps(records, indent=2), media_type="application/json")

    if format in ("excel", "xlsx"):
        rows_html = []
        for r in records:
            in_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["first_seen"]))
            out_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["last_seen"]))
            st = r.get("compliance_status", "PASSED")
            st_style = (
                "background-color:#dcfce7;color:#166534;font-weight:bold;text-align:center;"
                if st == "PASSED"
                else "background-color:#fee2e2;color:#991b1b;font-weight:bold;text-align:center;"
            )
            dwell_s = r.get("dwell_seconds", 0.0)
            dwell_fmt = (
                f"{int(dwell_s // 60)}m {int(dwell_s % 60)}s"
                if dwell_s >= 60
                else f"{dwell_s:.1f}s"
            )
            rows_html.append(f"""
            <tr>
              <td style="text-align:center;font-weight:bold;">#{r.get("track_id", "")}</td>
              <td style="white-space:nowrap;text-align:center;">{in_str}</td>
              <td style="white-space:nowrap;text-align:center;">{out_str}</td>
              <td style="text-align:center;font-weight:600;">{dwell_fmt}</td>
              <td style="{st_style}">{st}</td>
              <td style="text-align:center;font-weight:bold;">{r.get("verification_score", 100.0)}%</td>
              <td style="text-align:center;">{r.get("violation_count", 0)}</td>
              <td style="font-weight:600;">{html.escape(str(r.get("zone_id", "")).replace("_", " ").upper())}</td>
              <td>{html.escape(str(r.get("source", "")))}</td>
              <td style="font-family:Consolas,monospace;font-size:8.5pt;">{html.escape(str(r.get("session_id", "")))}</td>
            </tr>
            """)

        excel_content = f"""<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:x="urn:schemas-microsoft-com:office:excel" xmlns="http://www.w3.org/TR/REC-html40">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<style>
  body {{ font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 10pt; }}
  table {{ border-collapse: collapse; }}
  th {{ background-color: #1e3a8a; color: #ffffff; font-weight: bold; padding: 10px 14px; border: 1px solid #cbd5e1; font-size: 10pt; }}
  td {{ padding: 8px 12px; border: 1px solid #cbd5e1; vertical-align: middle; }}
  tr:nth-child(even) {{ background-color: #f8fafc; }}
</style>
</head>
<body>
  <h2 style="color:#0f172a;margin-bottom:4px;">OSHA &amp; ISO 45001 Workforce Shift Attendance &amp; PPE Compliance Roster</h2>
  <p style="color:#64748b;font-size:9pt;margin-top:0px;margin-bottom:12px;">Exported: {time.strftime('%Y-%m-%d %H:%M:%S')} Local | System: IPSIP Safety Command Sentinel</p>
  <table>
    <thead>
      <tr>
        <th>Worker ID</th>
        <th>First Seen (In-Time)</th>
        <th>Last Seen (Out-Time)</th>
        <th>Dwell Duration</th>
        <th>Compliance Status</th>
        <th>PPE Score</th>
        <th>Violations</th>
        <th>Zone</th>
        <th>Feed Source</th>
        <th>Session ID</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>"""
        return Response(
            content=excel_content,
            media_type="application/vnd.ms-excel",
            headers={
                "Content-Disposition": f"attachment; filename=worker_attendance_roster_{int(time.time())}.xls"
            },
        )

    output = io.StringIO()
    output.write("\ufeff")
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(
        [
            "Worker ID",
            "First Seen (Local)",
            "Last Seen (Local)",
            "Dwell Time (s)",
            "Compliance Status",
            "PPE Verification Score (%)",
            "Violation Count",
            "Zone ID",
            "Source Feed",
            "Session ID",
        ]
    )
    for r in records:
        writer.writerow(
            [
                r["track_id"],
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["first_seen"])),
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["last_seen"])),
                r.get("dwell_seconds", 0.0),
                r.get("compliance_status", "PASSED"),
                r.get("verification_score", 100.0),
                r.get("violation_count", 0),
                r.get("zone_id", ""),
                r.get("source", ""),
                r.get("session_id", ""),
            ]
        )
    return Response(
        content=output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename=worker_attendance_roster_{int(time.time())}.csv"
        },
    )


@app.get("/api/v1/videos/sample-list", summary="List preloaded industrial CCTV demo clips")
def get_sample_videos() -> list[dict[str, Any]]:
    samples = [
        {
            "filename": "group_workers_helmets_vests_4s.mp4",
            "title": "👷 Group Inspection (Helmets & Vests)",
            "description": "Multi-worker bay with workers wearing hard hats & high-visibility vests",
            "duration": "4s",
            "category": "Multi-Worker Assembly",
        },
        {
            "filename": "indoor_worker_safety_gear_10s.mp4",
            "title": "🏭 Indoor Manufacturing Technician",
            "description": "Plant floor technician operating manufacturing equipment with full safety gear",
            "duration": "10s",
            "category": "Indoor Plant",
        },
        {
            "filename": "worker_rebar_helmet_vest_17s.mp4",
            "title": "🏗️ Heavy Construction / Rebar Worker",
            "description": "Heavy machinery construction site worker wearing helmet and safety vest",
            "duration": "17s",
            "category": "Heavy Machinery",
        },
        {
            "filename": "test_cctv.mp4",
            "title": "📹 Plant CCTV Surveillance Feed",
            "description": "Fixed overhead plant CCTV monitoring personnel entry walkway",
            "duration": "8s",
            "category": "CCTV Surveillance",
        },
    ]
    existing = []
    for s in samples:
        fpath = UPLOAD_DIR / s["filename"]
        if fpath.is_file():
            s["size_bytes"] = fpath.stat().st_size
            s["size_mb"] = round(fpath.stat().st_size / (1024 * 1024), 2)
            existing.append(s)
    return existing


@app.get("/api/v1/evidence/{filename}", summary="Serve cropped evidence snapshot")
def get_evidence_image(filename: str):
    path = EVIDENCE_DIR / filename
    if not path.is_file():
        # Generate dynamic placeholder evidence crop if missing
        img = np.zeros((300, 200, 3), dtype=np.uint8) + 240
        cv2.putText(img, "EVIDENCE", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 200), 2)
        cv2.putText(img, filename[:15], (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
        _, buf = cv2.imencode(".jpg", img)
        return Response(content=buf.tobytes(), media_type="image/jpeg")
    return FileResponse(str(path), media_type="image/jpeg")


# ── Dynamic Camera MJPEG Stream Endpoint ──────────────────────────────────────
@app.get("/api/v1/stream/feed", summary="Live camera MJPEG stream with dynamic bounding boxes")
async def video_feed(
    request: Request, zone: str = "active_work_zone", camera_id: str | None = None
):
    """Streams the real live edge webcam feed directly into the dashboard."""

    async def frame_generator():
        while True:
            if await request.is_disconnected():
                break
            jpeg_bytes = live_service.get_stream_frame()
            length = len(jpeg_bytes)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(length).encode("ascii")
                + b"\r\n\r\n"
                + jpeg_bytes
                + b"\r\n"
            )
            await asyncio.sleep(0.033)  # ~30 FPS smooth real-time delivery

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/v1/video/processed", summary="Serve the most recently annotated uploaded video")
def get_processed_video() -> FileResponse:
    if not PROCESSED_VIDEO_PATH.is_file():
        raise HTTPException(status_code=404, detail="No processed video is available yet.")
    return FileResponse(str(PROCESSED_VIDEO_PATH), media_type="video/mp4")


# ── Video Upload & Automated Pipeline Inspection Endpoint ───────────────────────
_PIPELINE_LOCK = asyncio.Lock()
_PIPELINE = None


def get_ppe_pipeline():
    """Return a shared model with fresh stateful components for each upload."""
    global _PIPELINE
    if _PIPELINE is None:
        model_candidates = [
            BASE_DIR / "weights" / "best.pt",
            BASE_DIR / "PPE_detection" / "runs" / "ppe_exp_v3" / "ultra" / "weights" / "best.pt",
            BASE_DIR
            / "PPE_detection"
            / "runs"
            / "ppe_three_source_v1"
            / "ultra"
            / "weights"
            / "best.pt",
            BASE_DIR / "runs" / "ppe_6class_60eps" / "ultra" / "ultra" / "weights" / "best.pt",
            BASE_DIR / "yolov8s.pt",
        ]
        model_path = next((p for p in model_candidates if p.is_file()), BASE_DIR / "yolov8s.pt")
        from ppe_monitor.compliance.policy_loader import PolicyLoader
        from ppe_monitor.detector.yolo_detector import PPEDetector
        from ppe_monitor.display.visualizer import Visualizer

        detector = PPEDetector(str(model_path))
        policy = PolicyLoader(str(BASE_DIR / "ppe_monitor" / "config" / "zones.yaml"))
        viz = Visualizer()
        _PIPELINE = (detector, policy, viz)

    detector, policy, viz = _PIPELINE
    from ppe_monitor.compliance.engine import ComplianceEngine
    from ppe_monitor.tracker.bytetrack_wrapper import PersonTracker

    # Policies edited in the dashboard must also govern offline analysis.
    return detector, PersonTracker(), ComplianceEngine(policy), viz, policy


@app.post(
    "/api/v1/video/upload-and-process",
    summary="Upload custom video feed and run automated PPE inspection",
)
async def upload_and_process_video(
    file: UploadFile = File(...),
    zone_id: str = Form("active_work_zone"),
    max_frames: int = Form(99999),
):
    """Uploads a video, executes the YOLOv8s PPE pipeline, records violations & evidence, and saves an annotated video feed."""
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        raise HTTPException(
            status_code=400, detail=f"Unsupported format '{ext}'. Upload .mp4, .avi, or .mov"
        )

    saved_path = UPLOAD_DIR / (file.filename or "uploaded_video.mp4")
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cap = cv2.VideoCapture(str(saved_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Unable to decode uploaded video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    async with _PIPELINE_LOCK:
        try:
            from ppe_monitor.tracker.association import associate_ppe_to_workers

            detector, tracker, engine, viz, policy = get_ppe_pipeline()
            zone = db.get_zone(zone_id)
            if not zone:
                raise HTTPException(status_code=404, detail=f"Unknown zone '{zone_id}'.")
            policy.update_runtime_policy(
                zone_id,
                required_ppe=zone["required_ppe"],
                confirmation_frames=zone["n_frame_confirm"],
                confirmation_seconds=max(0.25, zone["n_frame_confirm"] / 25),
                alert_cooldown_seconds=zone["alert_cooldown_seconds"],
            )

            PROCESSED_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = None

            frame_idx = 0
            violations_logged = []
            workers_seen = set()
            t_start = time.time()
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            video_fps = max(float(video_fps), 1.0)
            session_id = f"batch_{Path(saved_path).stem}_{int(time.time())}"
            attendance_snapshots = set()

            while frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame_idx += 1

                # Resize to standard 640x480 for fast edge inference and consistent display
                frame_resized = cv2.resize(frame, (640, 480))

                # 1. Detection
                persons_raw, ppe_items = detector.detect(frame_resized)

                # 2. Tracking
                tracked_persons = tracker.update(persons_raw)
                for tp in tracked_persons:
                    if tp.track_id is not None:
                        workers_seen.add(tp.track_id)

                # 3. Spatial Association
                associations = associate_ppe_to_workers(
                    tracked_persons, ppe_items, region_map=policy.get_body_regions()
                )

                # 4. Compliance Evaluation
                # Use source-video time, not the much faster CPU inference time.
                events = engine.update(associations, zone_id, now=frame_idx / video_fps)

                # 5. Handle confirmed violation events
                now = time.time()
                for ev in events:
                    crop_filename = f"upload_worker_{ev.track_id}_{frame_idx}_{time.time_ns()}.jpg"
                    crop_path = EVIDENCE_DIR / crop_filename

                    wbox = None
                    for tp in tracked_persons:
                        if tp.track_id == ev.track_id:
                            wbox = tp.box
                            break

                    if wbox is not None:
                        bx1, by1, bx2, by2 = [max(0, int(v)) for v in wbox]
                        crop = frame_resized[by1:by2, bx1:bx2]
                        if crop.size > 0:
                            cv2.imwrite(str(crop_path), crop)
                        else:
                            cv2.imwrite(str(crop_path), frame_resized)
                    else:
                        cv2.imwrite(str(crop_path), frame_resized)

                    sha = (
                        hashlib.sha256(crop_path.read_bytes()).hexdigest()
                        if crop_path.is_file()
                        else ""
                    )
                    ar_item = next((a for a in associations if a.track_id == ev.track_id), None)
                    present_items = [
                        p for p, ok in (ar_item.ppe_present.items() if ar_item else []) if ok
                    ]
                    conf_map = {
                        p: det.confidence
                        for p, det in (ar_item.ppe_detections.items() if ar_item else [])
                        if det
                    }

                    row_id = db.log_event(
                        event=ev,
                        evidence_path=f"evidence/{crop_filename}",
                        present_ppe=present_items,
                        confidence_map=conf_map,
                        bbox=list(wbox) if wbox else [],
                        sha256_hash=sha,
                    )
                    violations_logged.append(
                        {
                            "id": row_id,
                            "track_id": ev.track_id,
                            "zone_id": zone_id,
                            "missing_ppe": ev.missing_ppe,
                            "frame": frame_idx,
                        }
                    )

                # 5b. Shift Worker Attendance Logging
                valid_tracked = [tp for tp in tracked_persons if tp.track_id is not None]
                if valid_tracked and (frame_idx % 3 == 0):
                    from ppe_monitor.compliance.worker_state import ComplianceStatus

                    for tp in valid_tracked:
                        ws = engine.worker_states.get(tp.track_id)
                        is_compliant = (ws.status == ComplianceStatus.COMPLIANT) if ws else True
                        has_viol = (ws.status == ComplianceStatus.VIOLATION) if ws else False

                        thumb_rel = None
                        if tp.track_id not in attendance_snapshots:
                            bx1, by1, bx2, by2 = [max(0, int(v)) for v in tp.box]
                            if bx2 > bx1 and by2 > by1:
                                crop = frame_resized[by1:by2, bx1:bx2]
                                if crop.size > 0:
                                    thumb_name = f"att_batch_{tp.track_id}_{frame_idx}_{int(time.time_ns())}.jpg"
                                    thumb_path = EVIDENCE_DIR / thumb_name
                                    cv2.imwrite(str(thumb_path), crop)
                                    thumb_rel = f"evidence/{thumb_name}"
                                    attendance_snapshots.add(tp.track_id)

                        db.record_worker_attendance(
                            session_id=session_id,
                            track_id=tp.track_id,
                            zone_id=zone_id,
                            is_compliant=is_compliant,
                            has_violation=has_viol,
                            snapshot_path=thumb_rel,
                            source=file.filename or "uploaded_video.mp4",
                            timestamp=now,
                        )

                # 6. Annotate frame
                annotated = viz.draw(
                    frame_resized, tracked_persons, ppe_items, associations, engine.worker_states
                )
                cv2.rectangle(annotated, (0, 0), (640, 28), (248, 250, 252), -1)
                cv2.putText(
                    annotated,
                    f"UPLOADED FEED: {file.filename[:18]} | FRAME {frame_idx} | {zone_id.upper()}",
                    (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (15, 23, 42),
                    1,
                )

                if writer is None:
                    writer = cv2.VideoWriter(str(PROCESSED_VIDEO_PATH), fourcc, 25.0, (640, 480))
                writer.write(annotated)

            if writer is not None:
                writer.release()
            cap.release()

            elapsed = max(time.time() - t_start, 0.001)
            eff_fps = frame_idx / elapsed

            # Log telemetry
            db.log_telemetry(
                zone_id=zone_id,
                camera_id="cam_uploaded",
                active_workers=len(workers_seen) or 1,
                compliant_count=max(0, len(workers_seen) - len(violations_logged)),
                violation_count=len(violations_logged),
                fps=round(eff_fps, 1),
            )

            # Register as active camera
            db.register_or_update_camera(
                camera_id="cam_uploaded",
                zone_id=zone_id,
                name=f"Uploaded: {file.filename[:16]}",
                source_url=f"file://evidence/latest_processed_feed.mp4",
                is_online=1,
            )

            return {
                "status": "success",
                "message": f"Processed {frame_idx} frames successfully. Found {len(violations_logged)} violation events.",
                "camera_id": "cam_uploaded",
                "zone_id": zone_id,
                "frames_processed": frame_idx,
                "total_video_frames": total_frames,
                "violations_detected": len(violations_logged),
                "unique_workers": len(workers_seen),
                "processing_fps": round(eff_fps, 1),
            }

        except Exception as e:
            if cap:
                cap.release()
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


# ── Video Stream Processing Service ───────────────────────────────────────────
class VideoStreamService:
    """
    Runs the full PPE pipeline on an uploaded video file — frame by frame —
    streaming annotated MJPEG frames exactly like the live webcam.
    Also fires ESP32 commands and broadcasts WebSocket telemetry in real time.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.latest_jpeg: Optional[bytes] = None
        self.last_frame_time: float = 0.0
        self.current_fps: float = 0.0
        self.latest_checklist: dict[str, Any] = {
            "helmet": {"present": False, "confidence": 0.0},
            "vest": {"present": False, "confidence": 0.0},
            "boots": {"present": False, "confidence": 0.0},
            "gloves": {"present": False, "confidence": 0.0},
        }
        self.latest_workers: list[dict[str, Any]] = []
        self.active_workers_seen: set = set()
        self.violations_logged: int = 0
        self.frames_processed: int = 0
        self.total_frames: int = 0
        self.filename: str = ""
        self.error: Optional[str] = None
        self.done: bool = False

    def start(self, video_path: str, zone_id: str, arduino=None, ws_manager=None, db=None):
        with self.lock:
            if self.running:
                return
            # Reset state
            self.running = True
            self.done = False
            self.error = None
            self.latest_jpeg = None
            self.latest_workers = []
            self.active_workers_seen = set()
            self.violations_logged = 0
            self.frames_processed = 0
            self.filename = Path(video_path).name
            self.latest_checklist = {
                k: {"present": False, "confidence": 0.0}
                for k in ["helmet", "vest", "boots", "gloves"]
            }
        self.thread = threading.Thread(
            target=self._run, args=(video_path, zone_id, arduino, ws_manager, db), daemon=True
        )
        self.thread.start()

    def stop(self):
        with self.lock:
            self.running = False

    def get_stream_frame(self) -> bytes:
        with self.lock:
            if self.latest_jpeg is not None and (time.time() - self.last_frame_time < 5.0):
                return self.latest_jpeg
        # Standby / done placeholder
        label = (
            "PROCESSING COMPLETE"
            if self.done
            else ("ERROR: " + (self.error or "")) if self.error else "STARTING..."
        )
        img = np.zeros((480, 640, 3), dtype=np.uint8) + 18
        cv2.putText(img, label, (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (200, 200, 200), 2)
        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()

    def get_status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "running": self.running,
                "done": self.done,
                "error": self.error,
                "filename": self.filename,
                "frames_processed": self.frames_processed,
                "total_frames": self.total_frames,
                "fps": round(self.current_fps, 1),
                "workers": self.latest_workers,
                "checklist": self.latest_checklist,
                "unique_workers": len(self.active_workers_seen),
                "violations_logged": self.violations_logged,
            }

    def _run(self, video_path: str, zone_id: str, arduino, ws_manager, db):
        import threading as _threading

        from ppe_monitor.compliance.engine import ComplianceEngine
        from ppe_monitor.compliance.policy_loader import PolicyLoader
        from ppe_monitor.compliance.worker_state import ComplianceStatus
        from ppe_monitor.detector.yolo_detector import PPEDetector
        from ppe_monitor.display.visualizer import Visualizer
        from ppe_monitor.tracker.association import associate_ppe_to_workers
        from ppe_monitor.tracker.bytetrack_wrapper import PersonTracker

        cap = None
        try:
            model_candidates = [
                BASE_DIR / "weights" / "best.pt",
                BASE_DIR
                / "PPE_detection"
                / "runs"
                / "ppe_exp_v3"
                / "ultra"
                / "weights"
                / "best.pt",
                BASE_DIR
                / "PPE_detection"
                / "runs"
                / "ppe_three_source_v1"
                / "ultra"
                / "weights"
                / "best.pt",
                BASE_DIR / "runs" / "ppe_6class_60eps" / "ultra" / "ultra" / "weights" / "best.pt",
                BASE_DIR / "yolov8s.pt",
            ]
            model_path = next((p for p in model_candidates if p.is_file()), BASE_DIR / "yolov8s.pt")

            detector = PPEDetector(str(model_path), conf_threshold=0.010)
            tracker = PersonTracker()
            policy = PolicyLoader(str(BASE_DIR / "ppe_monitor" / "config" / "zones.yaml"))
            engine = ComplianceEngine(policy)
            viz = Visualizer()

            if db:
                zone_data = db.get_zone(zone_id)
                if zone_data:
                    policy.update_runtime_policy(
                        zone_id,
                        required_ppe=zone_data["required_ppe"],
                        confirmation_frames=zone_data["n_frame_confirm"],
                        confirmation_seconds=max(0.25, zone_data["n_frame_confirm"] / 25),
                        alert_cooldown_seconds=zone_data["alert_cooldown_seconds"],
                    )

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                with self.lock:
                    self.error = f"Cannot open video: {video_path}"
                    self.running = False
                    self.done = True
                return

            video_fps = max(float(cap.get(cv2.CAP_PROP_FPS) or 25.0), 1.0)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            with self.lock:
                self.total_frames = total
            session_id = f"video_{Path(video_path).stem}_{int(time.time())}"
            attendance_snapshots = set()

            frame_idx = 0
            frame_timestamps: list[float] = []
            w, h = 640, 480

            while self.running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1
                t_start = time.time()
                frame = cv2.resize(frame, (w, h))

                # FPS tracking
                frame_timestamps.append(t_start)
                if len(frame_timestamps) > 15:
                    frame_timestamps.pop(0)
                fps_now = (
                    (len(frame_timestamps) - 1)
                    / max(frame_timestamps[-1] - frame_timestamps[0], 1e-4)
                    if len(frame_timestamps) >= 2
                    else 0.0
                )

                # 1. Detect
                persons_raw, ppe_items = detector.detect(frame)
                # 2. Track
                tracked_persons = tracker.update(persons_raw)
                for tp in tracked_persons:
                    if tp.track_id is not None:
                        self.active_workers_seen.add(tp.track_id)
                # 3. Associate
                associations = associate_ppe_to_workers(
                    tracked_persons, ppe_items, region_map=policy.get_body_regions()
                )
                # 4. Compliance
                events = engine.update(associations, zone_id, now=frame_idx / video_fps)

                # 5. ESP32 — identical logic as live webcam
                if arduino:
                    try:
                        valid_tracked = [tp for tp in tracked_persons if tp.track_id is not None]
                        states = [engine.worker_states.get(tp.track_id) for tp in valid_tracked]
                        violators = [
                            s for s in states if s and s.status == ComplianceStatus.VIOLATION
                        ]
                        if violators:
                            violator = min(violators, key=lambda s: s.track_id)
                            arduino.send_violation(
                                violator.track_id, sorted(violator.confirmed_violations)[0]
                            )
                        elif any(s and s.status == ComplianceStatus.PENDING for s in states):
                            arduino.send_pending()
                        elif valid_tracked:
                            arduino.send_compliant()
                        else:
                            arduino.send_idle()
                    except Exception:
                        pass

                # 6. DB logging
                now_wall = time.time()
                violations_in_frame = 0
                for ev in events:
                    violations_in_frame += 1
                    crop_filename = (
                        f"videostream_worker_{ev.track_id}_{frame_idx}_{time.time_ns()}.jpg"
                    )
                    crop_path = EVIDENCE_DIR / crop_filename
                    wbox = next(
                        (tp.box for tp in tracked_persons if tp.track_id == ev.track_id), None
                    )
                    if wbox is not None:
                        bx1, by1, bx2, by2 = [max(0, int(v)) for v in wbox]
                        crop = frame[by1:by2, bx1:bx2]
                        cv2.imwrite(str(crop_path), crop if crop.size > 0 else frame)
                    else:
                        cv2.imwrite(str(crop_path), frame)
                    sha = (
                        hashlib.sha256(crop_path.read_bytes()).hexdigest()
                        if crop_path.is_file()
                        else ""
                    )
                    ar_item = next((a for a in associations if a.track_id == ev.track_id), None)
                    present_items = [
                        p for p, ok in (ar_item.ppe_present.items() if ar_item else []) if ok
                    ]
                    conf_map = {
                        p: det.confidence
                        for p, det in (ar_item.ppe_detections.items() if ar_item else [])
                        if det
                    }
                    if db:
                        db.log_event(
                            event=ev,
                            evidence_path=f"evidence/{crop_filename}",
                            present_ppe=present_items,
                            confidence_map=conf_map,
                            bbox=list(wbox) if wbox else [],
                            sha256_hash=sha,
                        )

                # 6b. Shift Worker Attendance & Verification Registry
                valid_tracked = [tp for tp in tracked_persons if tp.track_id is not None]
                if db and valid_tracked and (frame_idx % 3 == 0):
                    for tp in valid_tracked:
                        ws = engine.worker_states.get(tp.track_id)
                        is_compliant = (ws.status == ComplianceStatus.COMPLIANT) if ws else True
                        has_viol = (ws.status == ComplianceStatus.VIOLATION) if ws else False

                        thumb_rel = None
                        if tp.track_id not in attendance_snapshots:
                            bx1, by1, bx2, by2 = [max(0, int(v)) for v in tp.box]
                            if bx2 > bx1 and by2 > by1:
                                crop = frame[by1:by2, bx1:bx2]
                                if crop.size > 0:
                                    thumb_name = f"att_video_{tp.track_id}_{frame_idx}_{int(time.time_ns())}.jpg"
                                    thumb_path = EVIDENCE_DIR / thumb_name
                                    cv2.imwrite(str(thumb_path), crop)
                                    thumb_rel = f"evidence/{thumb_name}"
                                    attendance_snapshots.add(tp.track_id)

                        db.record_worker_attendance(
                            session_id=session_id,
                            track_id=tp.track_id,
                            zone_id=zone_id,
                            is_compliant=is_compliant,
                            has_violation=has_viol,
                            snapshot_path=thumb_rel,
                            source=Path(video_path).name,
                            timestamp=now_wall,
                        )

                # 7. Build live checklist (primary/largest worker)
                checklist_snap = {
                    k: {"present": False, "confidence": 0.0}
                    for k in ["helmet", "vest", "boots", "gloves"]
                }
                workers_payload = []
                valid_tracked = [tp for tp in tracked_persons if tp.track_id is not None]
                if valid_tracked and associations:
                    primary = max(
                        valid_tracked, key=lambda p: (p.box[2] - p.box[0]) * (p.box[3] - p.box[1])
                    )
                    primary_ar = next(
                        (a for a in associations if a.track_id == primary.track_id), None
                    )
                    if primary_ar:
                        for k in ["helmet", "vest", "boots", "gloves"]:
                            pres = primary_ar.ppe_present.get(k, False)
                            det = primary_ar.ppe_detections.get(k)
                            checklist_snap[k] = {
                                "present": pres,
                                "confidence": round(
                                    float(det.confidence if det else (0.85 if pres else 0.0)), 2
                                ),
                            }
                for tp in valid_tracked:
                    ws = engine.worker_states.get(tp.track_id)
                    ar = next((a for a in associations if a.track_id == tp.track_id), None)
                    w_cl = {}
                    for k in ["helmet", "vest", "boots", "gloves"]:
                        pres = ar.ppe_present.get(k, False) if ar else False
                        det = ar.ppe_detections.get(k) if ar else None
                        w_cl[k] = {"present": pres, "confidence": det.confidence if det else 0.0}
                    workers_payload.append(
                        {
                            "track_id": tp.track_id,
                            "bbox": [int(v) for v in tp.box],
                            "status": ws.status.value if ws else "COMPLIANT",
                            "checklist": w_cl,
                        }
                    )

                # 8. Annotate frame
                annotated = viz.draw(
                    frame, tracked_persons, ppe_items, associations, engine.worker_states
                )
                cv2.rectangle(annotated, (0, 0), (w, 28), (248, 250, 252), -1)
                cv2.putText(
                    annotated,
                    f"VIDEO FEED: {self.filename[:20]} | FRAME {frame_idx}/{total} | {fps_now:.1f} FPS | WORKERS: {len(valid_tracked)}",
                    (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (15, 23, 42),
                    1,
                )

                # 9. Encode JPEG
                _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
                jpeg_bytes = jpeg.tobytes()

                # 10. Update shared state
                with self.lock:
                    self.latest_jpeg = jpeg_bytes
                    self.last_frame_time = t_start
                    self.current_fps = fps_now
                    self.latest_checklist = checklist_snap
                    self.latest_workers = workers_payload
                    self.frames_processed = frame_idx
                    self.violations_logged += violations_in_frame

                # 11. Broadcast WebSocket telemetry (same as live webcam)
                if ws_manager:
                    compliant_c = sum(1 for w in workers_payload if w.get("status") == "COMPLIANT")
                    violation_c = sum(1 for w in workers_payload if w.get("status") == "VIOLATION")
                    try:
                        import asyncio as _asyncio

                        loop = _event_loop or _asyncio.get_event_loop()
                        if loop.is_running():
                            _asyncio.run_coroutine_threadsafe(
                                ws_manager.broadcast(
                                    {
                                        "type": "telemetry_tick",
                                        "timestamp": now_wall,
                                        "zone_id": zone_id,
                                        "fps": round(fps_now, 1),
                                        "active_workers": len(workers_payload),
                                        "compliant_count": compliant_c,
                                        "violation_count": violation_c,
                                        "workers": workers_payload,
                                        "checklist": checklist_snap,
                                        "test_mode": "all_ppe",
                                        "serial_connected": bool(
                                            arduino and arduino.ser and arduino.ser.is_open
                                        ),
                                        "source": "video_stream",
                                    }
                                ),
                                loop,
                            )
                    except Exception:
                        pass

                # 12. Log DB telemetry every ~2 seconds
                if db and frame_idx % max(1, int(video_fps * 2)) == 0:
                    db.log_telemetry(
                        zone_id=zone_id,
                        camera_id="cam_videostream",
                        active_workers=len(workers_payload) or 1,
                        compliant_count=max(
                            0, len(workers_payload) - (1 if violations_in_frame > 0 else 0)
                        ),
                        violation_count=1 if violations_in_frame > 0 else 0,
                        fps=round(fps_now, 1),
                    )

                # Throttle to ~real-time playback speed (don't burn the CPU)
                elapsed = time.time() - t_start
                target_delay = 1.0 / video_fps
                if elapsed < target_delay:
                    time.sleep(target_delay - elapsed)

        except Exception as exc:
            with self.lock:
                self.error = str(exc)
            print(f"[VIDEO STREAM ERROR] {exc}")
        finally:
            if cap:
                cap.release()
            # ESP32 idle on completion
            if arduino:
                try:
                    arduino.send_idle()
                except Exception:
                    pass
            with self.lock:
                self.running = False
                self.done = True
            # Register camera
            if db:
                db.register_or_update_camera(
                    camera_id="cam_videostream",
                    zone_id=zone_id,
                    name=f"Video: {Path(video_path).name[:20]}",
                    source_url=f"file://uploads/{Path(video_path).name}",
                    is_online=0,
                )
            print(
                f"[VIDEO STREAM] Finished processing {video_path} — {self.frames_processed} frames, {self.violations_logged} violations."
            )


# Global video stream service singleton
video_stream_service = VideoStreamService()


@app.post(
    "/api/v1/video/start-stream",
    summary="Start real-time MJPEG stream processing of an uploaded video file",
)
async def start_video_stream(
    file: UploadFile | None = File(None),
    sample_filename: str | None = Form(None),
    zone_id: str = Form("active_work_zone"),
):
    """
    Accepts a video upload or a preloaded sample clip, saves it (if uploaded),
    then immediately starts streaming annotated MJPEG frames exactly like the live webcam —
    including ESP32 actuation, WebSocket telemetry and SQLite shift attendance logging.
    """
    if sample_filename:
        saved_path = UPLOAD_DIR / sample_filename
        if not saved_path.is_file():
            raise HTTPException(
                status_code=404, detail=f"Sample video '{sample_filename}' not found."
            )
        chosen_filename = sample_filename
    elif file is not None and getattr(file, "filename", None):
        ext = Path(file.filename or "video.mp4").suffix.lower()
        if ext not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            raise HTTPException(status_code=400, detail=f"Unsupported format '{ext}'.")

        saved_path = UPLOAD_DIR / file.filename
        with open(saved_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        chosen_filename = file.filename
    else:
        raise HTTPException(
            status_code=400, detail="Provide an uploaded video file or a preloaded sample clip."
        )

    # Verify zone
    zone = db.get_zone(zone_id)
    if not zone:
        raise HTTPException(status_code=404, detail=f"Unknown zone '{zone_id}'.")

    # Stop any in-progress stream before starting new one
    video_stream_service.stop()
    await asyncio.sleep(0.3)

    # Start background processing thread
    video_stream_service.start(
        video_path=str(saved_path),
        zone_id=zone_id,
        arduino=live_service.arduino,  # Share the same ESP32 bridge
        ws_manager=manager,
        db=db,
    )

    return {
        "status": "started",
        "filename": chosen_filename,
        "zone_id": zone_id,
        "stream_url": "/api/v1/video/stream-feed",
    }


@app.get(
    "/api/v1/video/stream-feed",
    summary="MJPEG stream of the video being processed with live bounding boxes",
)
async def video_stream_feed(request: Request):
    """Streams annotated JPEG frames from the currently-processing uploaded video."""

    async def frame_generator():
        while True:
            if await request.is_disconnected():
                break
            jpeg_bytes = video_stream_service.get_stream_frame()
            length = len(jpeg_bytes)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(length).encode("ascii")
                + b"\r\n\r\n"
                + jpeg_bytes
                + b"\r\n"
            )
            await asyncio.sleep(0.033)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get(
    "/api/v1/video/stream-status", summary="Get live status of the video stream processing job"
)
def get_video_stream_status() -> dict[str, Any]:
    return video_stream_service.get_status()


@app.post("/api/v1/video/stream-stop", summary="Stop the video stream processing job")
def stop_video_stream() -> dict[str, Any]:
    video_stream_service.stop()
    return {"status": "stopped"}


# ── Real-Time Telemetry WebSocket ─────────────────────────────────────────────
@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; client can send ping/commands
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


# ── Serve Built Frontend Dashboard (Production Distribution) ───────────────────
from fastapi.staticfiles import StaticFiles

FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

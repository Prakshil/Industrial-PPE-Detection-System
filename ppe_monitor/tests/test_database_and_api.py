"""
Comprehensive test suite for SQLite EventDatabase and FastAPI Backend.
Verifies WAL mode, transactions, schema integrity, and API endpoints.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_PM = Path(__file__).resolve().parent.parent
for p in [str(_ROOT), str(_PM)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from api.main import app
from fastapi.testclient import TestClient

from ppe_monitor.compliance.worker_state import ViolationEvent
from ppe_monitor.persistence.database import EventDatabase


def test_sqlite_wal_and_schema():
    """Verify SQLite WAL mode, table creation, and initial seed data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        db = EventDatabase(db_path)
        # Check WAL mode
        cur = db._conn.execute("PRAGMA journal_mode;")
        mode = cur.fetchone()[0]
        assert mode.lower() == "wal", f"Expected WAL mode, got {mode}"

        # Verify zones seeded
        zones = db.get_zones()
        assert len(zones) >= 3
        zone_ids = [z["zone_id"] for z in zones]
        assert "main_entry" in zone_ids
        assert "active_work_zone" in zone_ids
        assert "heavy_machinery_zone" in zone_ids

        # Test log_event and retrieval
        ev = ViolationEvent(
            track_id=99,
            zone_id="main_entry",
            missing_ppe=["helmet"],
            timestamp=time.time(),
            frame_number=100,
        )
        row_id = db.log_event(ev, evidence_path="test_crop.jpg", present_ppe=["vest", "boots"])
        assert row_id > 0

        # Verify retrieval
        v = db.get_violation_by_id(row_id)
        assert v is not None
        assert v["track_id"] == 99
        assert "helmet" in v["missing_ppe"]

        # Test triage update
        ok = db.update_violation_status(
            row_id, "CONFIRMED", reviewed_by="Inspector Smith", review_notes="Verified on CCTV"
        )
        assert ok is True
        v_updated = db.get_violation_by_id(row_id)
        assert v_updated["review_status"] == "CONFIRMED"
        assert v_updated["reviewed_by"] == "Inspector Smith"

        # Test delete violation
        assert db.delete_violation(row_id) is True
        assert db.get_violation_by_id(row_id) is None

        # Test zone policy update
        ok = db.update_zone_policy(
            "main_entry", required_ppe=["helmet", "vest", "boots", "gloves"], n_frame_confirm=7
        )
        assert ok is True
        z = db.get_zone("main_entry")
        assert "gloves" in z["required_ppe"]
        assert z["n_frame_confirm"] == 7

        # Test telemetry summary
        summary = db.get_telemetry_summary(hours=24)
        assert "compliance_rate_percent" in summary
        assert "violations_by_ppe" in summary
        assert "total_violations" in summary

        db.close()
    finally:
        for ext in ["", "-wal", "-shm"]:
            p = Path(db_path + ext)
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass


def test_fastapi_rest_endpoints():
    """Verify all REST API endpoints using FastAPI TestClient."""
    client = TestClient(app)

    # 1. Health check
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"

    # 2. Zones list
    res = client.get("/api/v1/zones")
    assert res.status_code == 200
    zones = res.json()
    assert len(zones) >= 3

    # 3. Policy update
    res = client.post(
        "/api/v1/zones/main_entry/policy",
        json={
            "required_ppe": ["helmet", "vest"],
            "n_frame_confirm": 4,
            "alert_cooldown_seconds": 20,
        },
    )
    assert res.status_code == 200
    assert res.json()["zone"]["n_frame_confirm"] == 4

    # 4. Cameras list
    res = client.get("/api/v1/cameras")
    assert res.status_code == 200
    assert len(res.json()) >= 1
    assert res.json()[0]["camera_id"] == "cam_live"

    # 5. Violations list
    from ppe_monitor.api.main import db
    from ppe_monitor.compliance.worker_state import ViolationEvent

    db.log_event(
        event=ViolationEvent(
            track_id=99,
            zone_id="active_work_zone",
            missing_ppe=["helmet"],
            timestamp=time.time(),
            frame_number=10,
        ),
        evidence_path="evidence/test.jpg",
    )

    res = client.get("/api/v1/violations?zone_id=all&status=all&limit=10")
    assert res.status_code == 200
    violations = res.json()
    assert isinstance(violations, list)
    assert len(violations) > 0
    target_id = violations[0]["id"]

    # 6. Triage patch
    res = client.patch(
        f"/api/v1/violations/{target_id}/action",
        json={
            "action": "CONFIRMED",
            "reviewed_by": "Safety Officer John",
            "notes": "Verified worker entering without helmet",
        },
    )
    assert res.status_code == 200
    assert res.json()["new_status"] == "CONFIRMED"

    # 7. Telemetry KPI summary
    res = client.get("/api/v1/telemetry?hours=24")
    assert res.status_code == 200
    telemetry = res.json()
    assert "compliance_rate_percent" in telemetry
    assert "violations_by_ppe" in telemetry

    # 8. Edge sync batch
    sync_payload = {
        "edge_node": "edge_cam_test",
        "events": [
            {
                "track_id": 88,
                "zone_id": "active_work_zone",
                "missing_ppe": ["boots"],
                "present_ppe": ["helmet", "vest"],
                "confidence_map": {"helmet": 0.88, "vest": 0.85},
                "bbox": [100, 100, 200, 400],
                "timestamp": time.time(),
                "frame_number": 450,
                "evidence_path": "evidence/sync_test.jpg",
                "evidence_sha256": "abcdef123456",
            }
        ],
    }
    res = client.post("/api/v1/edge/sync", json=sync_payload)
    assert res.status_code == 200
    assert res.json()["received_records"] == 1

    # 9. CSV and PDF Audit log export
    res = client.get("/api/v1/export/audit?format=csv")
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]
    assert "Event ID" in res.text
    assert "Missing PPE" in res.text

    # PDF Audit export verification
    res_pdf = client.get("/api/v1/export/audit?format=pdf")
    assert res_pdf.status_code == 200
    assert "application/pdf" in res_pdf.headers["content-type"]
    assert res_pdf.content.startswith(b"%PDF-"), "Generated file is not a valid PDF"
    assert len(res_pdf.content) > 1000

    # Live camera edge service endpoints
    res_cam = client.get("/api/v1/camera/live/status")
    assert res_cam.status_code == 200
    assert "test_mode" in res_cam.json()

    res_mode = client.post("/api/v1/camera/live/mode", json={"mode": "helmet_only"})
    assert res_mode.status_code == 200
    assert res_mode.json()["mode"] == "helmet_only"

    # 10. Evidence image serving
    res = client.get("/api/v1/evidence/test_nonexistent.jpg")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"

    # Clean up test event from database so tests don't leave test artifacts
    from api.main import db as live_db

    live_db._conn.execute("DELETE FROM violation_events WHERE evidence_sha256 = 'abcdef123456'")
    live_db._conn.commit()

    print("ALL DATABASE AND FASTAPI API TESTS PASSED!")


if __name__ == "__main__":
    test_sqlite_wal_and_schema()
    test_fastapi_rest_endpoints()

"""
Local SQLite event database — offline-first, WAL mode.
Stores all confirmed violation events with evidence metadata, camera catalogs,
zone policies, and time-series telemetry.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from ppe_monitor.compliance.worker_state import ViolationEvent

_SCHEMA = """
-- Zone policy configurations
CREATE TABLE IF NOT EXISTS zones (
    zone_id                TEXT PRIMARY KEY,
    name                   TEXT NOT NULL,
    risk_level             TEXT NOT NULL, -- LOW, MEDIUM, HIGH, EXTREME
    required_ppe           TEXT NOT NULL, -- JSON array of strings
    n_frame_confirm        INTEGER DEFAULT 35,
    alert_cooldown_seconds INTEGER DEFAULT 30,
    is_active              INTEGER DEFAULT 1,
    created_at             REAL DEFAULT (strftime('%s', 'now'))
);

-- Edge camera nodes directory
CREATE TABLE IF NOT EXISTS cameras (
    camera_id              TEXT PRIMARY KEY,
    zone_id                TEXT NOT NULL,
    name                   TEXT NOT NULL,
    source_url             TEXT NOT NULL,
    camera_profile         TEXT NOT NULL DEFAULT 'upper_body',
    is_online              INTEGER DEFAULT 1,
    last_ping              REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY(zone_id) REFERENCES zones(zone_id)
);

-- Confirmed safety violation events
CREATE TABLE IF NOT EXISTS violation_events (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    event_uuid             TEXT UNIQUE NOT NULL,
    track_id               INTEGER NOT NULL,
    zone_id                TEXT NOT NULL,
    missing_ppe            TEXT NOT NULL,          -- JSON string or comma-separated
    present_ppe            TEXT DEFAULT '[]',      -- JSON array of present items
    confidence_map         TEXT DEFAULT '{}',      -- JSON dict of class confidences
    bbox                   TEXT DEFAULT '[]',      -- JSON [x1, y1, x2, y2]
    timestamp              REAL NOT NULL,
    frame_number           INTEGER NOT NULL,
    evidence_path          TEXT,
    evidence_sha256        TEXT,
    review_status          TEXT DEFAULT 'UNREVIEWED', -- UNREVIEWED, CONFIRMED, DISMISSED, FALSE_POSITIVE
    reviewed_by            TEXT,
    review_notes           TEXT,
    synced_to_cloud        INTEGER DEFAULT 0,
    created_at             REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY(zone_id) REFERENCES zones(zone_id)
);

-- Time-series telemetry ticks for operational monitoring
CREATE TABLE IF NOT EXISTS telemetry_ticks (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp              REAL NOT NULL,
    zone_id                TEXT NOT NULL,
    camera_id              TEXT NOT NULL,
    active_workers         INTEGER NOT NULL,
    compliant_count        INTEGER NOT NULL,
    violation_count        INTEGER NOT NULL,
    fps                    REAL NOT NULL
);

-- Shift Worker Attendance & Verification Registry
CREATE TABLE IF NOT EXISTS worker_attendance (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id             TEXT NOT NULL,
    track_id               INTEGER NOT NULL,
    zone_id                TEXT NOT NULL,
    first_seen             REAL NOT NULL,
    last_seen              REAL NOT NULL,
    total_frames           INTEGER DEFAULT 1,
    compliant_frames       INTEGER DEFAULT 0,
    violation_count        INTEGER DEFAULT 0,
    compliance_status      TEXT NOT NULL DEFAULT 'PASSED',
    verification_score     REAL DEFAULT 100.0,
    snapshot_path          TEXT,
    source                 TEXT DEFAULT 'webcam',
    created_at             REAL DEFAULT (strftime('%s', 'now')),
    UNIQUE(session_id, track_id, zone_id) ON CONFLICT REPLACE
);

-- Indices for rapid queries in control room
CREATE INDEX IF NOT EXISTS idx_zone_time
    ON violation_events (zone_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_review_status
    ON violation_events (review_status);

CREATE INDEX IF NOT EXISTS idx_unsynced
    ON violation_events (synced_to_cloud)
    WHERE synced_to_cloud = 0;

CREATE INDEX IF NOT EXISTS idx_telemetry_time
    ON telemetry_ticks (timestamp);

CREATE INDEX IF NOT EXISTS idx_att_session
    ON worker_attendance (session_id);

CREATE INDEX IF NOT EXISTS idx_att_status
    ON worker_attendance (compliance_status);

CREATE INDEX IF NOT EXISTS idx_att_last_seen
    ON worker_attendance (last_seen DESC);
"""


class EventDatabase:
    """Thread-safe SQLite wrapper for PPE violation events and telemetry."""

    def __init__(self, db_path: str | Path = "events.db"):
        self._path = str(db_path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._conn.executescript(_SCHEMA)
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(cameras)")}
        if "camera_profile" not in columns:
            self._conn.execute(
                "ALTER TABLE cameras ADD COLUMN camera_profile TEXT NOT NULL DEFAULT 'upper_body'"
            )
        self._conn.commit()
        self.seed_initial_data()

    def seed_initial_data(self) -> None:
        """Seeds default zones if table is empty. Zero mock data."""
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) as cnt FROM zones")
        if cur.fetchone()["cnt"] == 0:
            initial_zones = [
                ("main_entry", "Main Entry Gate", "LOW", json.dumps(["helmet", "vest", "boots"]), 25, 30),
                ("active_work_zone", "Assembly & Welding Bay", "HIGH", json.dumps(["helmet", "vest", "gloves", "boots"]), 25, 30),
                ("heavy_machinery_zone", "Stamping & Heavy Crane Bay", "EXTREME", json.dumps(["helmet", "vest", "gloves", "boots", "goggles"]), 25, 15),
            ]
            cur.executemany(
                """INSERT INTO zones (zone_id, name, risk_level, required_ppe, n_frame_confirm, alert_cooldown_seconds)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                initial_zones,
            )
            self._conn.commit()

    def log_event(
        self,
        event: ViolationEvent,
        evidence_path: str | None = None,
        present_ppe: list[str] | None = None,
        confidence_map: dict[str, float] | None = None,
        bbox: list[int] | None = None,
        sha256_hash: str | None = None,
    ) -> int:
        """Insert a confirmed violation event. Returns the new row id."""
        event_uuid = str(uuid.uuid4())
        missing_json = json.dumps(event.missing_ppe if isinstance(event.missing_ppe, list) else [event.missing_ppe])
        present_json = json.dumps(present_ppe or [])
        conf_json = json.dumps(confidence_map or {})
        bbox_json = json.dumps(bbox or [])

        if sha256_hash is None and evidence_path and Path(evidence_path).is_file():
            try:
                sha256_hash = hashlib.sha256(Path(evidence_path).read_bytes()).hexdigest()
            except Exception:
                sha256_hash = ""

        with self._lock:
            cur = self._conn.execute(
                """
            INSERT INTO violation_events
                (event_uuid, track_id, zone_id, missing_ppe, present_ppe,
                 confidence_map, bbox, timestamp, frame_number, evidence_path,
                 evidence_sha256, review_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'UNREVIEWED')
            """,
                (
                    event_uuid,
                    event.track_id,
                    event.zone_id,
                    missing_json,
                    present_json,
                    conf_json,
                    bbox_json,
                    event.timestamp,
                    event.frame_number,
                    evidence_path,
                    sha256_hash or "",
                ),
            )
            self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_violations(
        self,
        zone_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query violations with optional filters."""
        query = "SELECT * FROM violation_events WHERE 1=1"
        params: list[Any] = []
        if zone_id and zone_id != "all":
            query += " AND zone_id = ?"
            params.append(zone_id)
        if status and status != "all":
            query += " AND review_status = ?"
            params.append(status.upper())
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self._conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["missing_ppe"] = json.loads(d["missing_ppe"])
            except Exception:
                d["missing_ppe"] = [d["missing_ppe"]]
            try:
                d["present_ppe"] = json.loads(d["present_ppe"])
            except Exception:
                d["present_ppe"] = []
            try:
                d["confidence_map"] = json.loads(d["confidence_map"])
            except Exception:
                d["confidence_map"] = {}
            try:
                d["bbox"] = json.loads(d["bbox"])
            except Exception:
                d["bbox"] = []
            results.append(d)
        return results

    def get_violation_by_id(self, violation_id: int) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM violation_events WHERE id = ?", (violation_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["missing_ppe"] = json.loads(d["missing_ppe"])
        except Exception:
            d["missing_ppe"] = [d["missing_ppe"]]
        try:
            d["present_ppe"] = json.loads(d["present_ppe"])
        except Exception:
            d["present_ppe"] = []
        try:
            d["confidence_map"] = json.loads(d["confidence_map"])
        except Exception:
            d["confidence_map"] = {}
        try:
            d["bbox"] = json.loads(d["bbox"])
        except Exception:
            d["bbox"] = []
        return d

    def update_violation_status(
        self,
        violation_id: int,
        status: str,
        reviewed_by: str = "Operator",
        review_notes: str = "",
    ) -> bool:
        """Update review status (CONFIRMED, DISMISSED, FALSE_POSITIVE)."""
        valid_statuses = {"UNREVIEWED", "CONFIRMED", "DISMISSED", "FALSE_POSITIVE"}
        st = status.upper()
        if st not in valid_statuses:
            return False
        cur = self._conn.execute(
            """
            UPDATE violation_events
            SET review_status = ?, reviewed_by = ?, review_notes = ?
            WHERE id = ?
            """,
            (st, reviewed_by, review_notes, violation_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_violation(self, violation_id: int) -> bool:
        """Deletes a single violation event by ID."""
        cur = self._conn.execute("DELETE FROM violation_events WHERE id = ?", (violation_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def clear_all_violations(self) -> int:
        """Deletes all violation records from the database. Returns count of deleted records."""
        cur = self._conn.execute("DELETE FROM violation_events")
        self._conn.commit()
        return cur.rowcount

    def get_zones(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM zones ORDER BY created_at ASC").fetchall()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["required_ppe"] = json.loads(d["required_ppe"])
            except Exception:
                d["required_ppe"] = []
            results.append(d)
        return results

    def get_zone(self, zone_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM zones WHERE zone_id = ?", (zone_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["required_ppe"] = json.loads(d["required_ppe"])
        except Exception:
            d["required_ppe"] = []
        return d

    def update_zone_policy(
        self,
        zone_id: str,
        required_ppe: list[str] | None = None,
        n_frame_confirm: int | None = None,
        alert_cooldown_seconds: int | None = None,
    ) -> bool:
        updates = []
        params = []
        if required_ppe is not None:
            updates.append("required_ppe = ?")
            params.append(json.dumps(required_ppe))
        if n_frame_confirm is not None:
            updates.append("n_frame_confirm = ?")
            params.append(int(n_frame_confirm))
        if alert_cooldown_seconds is not None:
            updates.append("alert_cooldown_seconds = ?")
            params.append(int(alert_cooldown_seconds))

        if not updates:
            return False
        params.append(zone_id)
        cur = self._conn.execute(
            f"UPDATE zones SET {', '.join(updates)} WHERE zone_id = ?",
            params,
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get_cameras(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM cameras ORDER BY camera_id ASC").fetchall()
        return [dict(r) for r in rows]

    def register_or_update_camera(
        self,
        camera_id: str,
        zone_id: str,
        name: str,
        source_url: str,
        is_online: int = 1,
        camera_profile: str = "upper_body",
    ) -> None:
        """Register a new camera feed or update an existing one."""
        with self._lock:
            self._conn.execute(
                """
            INSERT INTO cameras (camera_id, zone_id, name, source_url, is_online, last_ping, camera_profile)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(camera_id) DO UPDATE SET
                zone_id = excluded.zone_id,
                name = excluded.name,
                source_url = excluded.source_url,
                is_online = excluded.is_online,
                last_ping = excluded.last_ping,
                camera_profile = excluded.camera_profile
            """,
            (camera_id, zone_id, name, source_url, is_online, time.time(), camera_profile),
        )
        self._conn.commit()

    def log_telemetry(
        self,
        zone_id: str,
        camera_id: str,
        active_workers: int,
        compliant_count: int,
        violation_count: int,
        fps: float,
        timestamp: float | None = None,
    ) -> None:
        t = timestamp or time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO telemetry_ticks
                    (timestamp, zone_id, camera_id, active_workers, compliant_count, violation_count, fps)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (t, zone_id, camera_id, active_workers, compliant_count, violation_count, fps),
            )
            self._conn.commit()

    def get_telemetry_summary(self, hours: int = 24) -> dict[str, Any]:
        since = time.time() - hours * 3600
        cur = self._conn.cursor()

        # Violation counts by missing item
        cur.execute("SELECT missing_ppe FROM violation_events WHERE timestamp >= ?", (since,))
        missing_counts: dict[str, int] = {}
        for r in cur.fetchall():
            try:
                items = json.loads(r["missing_ppe"])
            except Exception:
                items = [r["missing_ppe"]]
            for item in items:
                missing_counts[item] = missing_counts.get(item, 0) + 1

        # Violation counts by zone
        cur.execute(
            """
            SELECT zone_id, COUNT(*) as count
            FROM violation_events
            WHERE timestamp >= ?
            GROUP BY zone_id
            """,
            (since,),
        )
        zone_counts = {r["zone_id"]: r["count"] for r in cur.fetchall()}

        # Active workers and total violations
        cur.execute(
            """
            SELECT AVG(active_workers) as avg_workers, AVG(compliant_count) as avg_compliant,
                   AVG(violation_count) as avg_violations, AVG(fps) as avg_fps
            FROM telemetry_ticks
            WHERE timestamp >= ?
            """,
            (since,),
        )
        telemetry_avg = cur.fetchone()

        raw_workers = telemetry_avg["avg_workers"] if telemetry_avg else None
        raw_compliant = telemetry_avg["avg_compliant"] if telemetry_avg else None
        raw_fps = telemetry_avg["avg_fps"] if telemetry_avg else None

        avg_workers = round(raw_workers, 1) if raw_workers is not None else 0.0
        avg_comp = round(raw_compliant, 1) if raw_compliant is not None else 0.0
        compliance_pct = round((avg_comp / max(avg_workers, 1)) * 100, 1) if avg_workers > 0 else 100.0
        avg_fps = round(raw_fps, 1) if raw_fps is not None else 0.0

        return {
            "time_window_hours": hours,
            "compliance_rate_percent": compliance_pct,
            "average_active_workers": int(round(avg_workers)),
            "average_fps": avg_fps,
            "violations_by_ppe": missing_counts,
            "violations_by_zone": zone_counts,
            "total_violations": sum(zone_counts.values()),
        }

    def get_unsynced(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT id, event_uuid, track_id, zone_id, missing_ppe,
                   present_ppe, confidence_map, bbox, timestamp,
                   frame_number, evidence_path, evidence_sha256
            FROM violation_events
            WHERE synced_to_cloud = 0
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["missing_ppe"] = json.loads(d["missing_ppe"])
            except Exception:
                d["missing_ppe"] = [d["missing_ppe"]]
            results.append(d)
        return results

    def mark_synced(self, event_ids: list[int]) -> None:
        if not event_ids:
            return
        self._conn.execute(
            f"UPDATE violation_events SET synced_to_cloud=1 "
            f"WHERE id IN ({','.join('?' * len(event_ids))})",
            event_ids,
        )
        self._conn.commit()

    def record_worker_attendance(
        self,
        session_id: str,
        track_id: int,
        zone_id: str,
        is_compliant: bool,
        has_violation: bool = False,
        snapshot_path: str | None = None,
        source: str = "webcam",
        timestamp: float | None = None,
    ) -> None:
        """Upsert a worker's shift attendance log, updating first/last seen and compliance ratio."""
        now = timestamp or time.time()
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, first_seen, total_frames, compliant_frames, violation_count, snapshot_path
                FROM worker_attendance
                WHERE session_id = ? AND track_id = ? AND zone_id = ?
                """,
                (session_id, track_id, zone_id),
            ).fetchone()

            if row is None:
                total_frames = 1
                comp_frames = 1 if is_compliant else 0
                viol_count = 1 if has_violation else 0
                status = "FLAGGED" if viol_count > 0 else "PASSED"
                score = 100.0 if is_compliant else 0.0
                self._conn.execute(
                    """
                    INSERT INTO worker_attendance
                        (session_id, track_id, zone_id, first_seen, last_seen,
                         total_frames, compliant_frames, violation_count,
                         compliance_status, verification_score, snapshot_path, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        track_id,
                        zone_id,
                        now,
                        now,
                        total_frames,
                        comp_frames,
                        viol_count,
                        status,
                        score,
                        snapshot_path or "",
                        source,
                    ),
                )
            else:
                total_frames = row["total_frames"] + 1
                comp_frames = row["compliant_frames"] + (1 if is_compliant else 0)
                viol_count = row["violation_count"] + (1 if has_violation else 0)
                status = "FLAGGED" if viol_count > 0 else "PASSED"
                score = round((comp_frames / max(1, total_frames)) * 100.0, 1)
                snap = snapshot_path or row["snapshot_path"] or ""

                self._conn.execute(
                    """
                    UPDATE worker_attendance
                    SET last_seen = ?,
                        total_frames = ?,
                        compliant_frames = ?,
                        violation_count = ?,
                        compliance_status = ?,
                        verification_score = ?,
                        snapshot_path = CASE WHEN ? != '' THEN ? ELSE snapshot_path END
                    WHERE id = ?
                    """,
                    (
                        now,
                        total_frames,
                        comp_frames,
                        viol_count,
                        status,
                        score,
                        snap,
                        snap,
                        row["id"],
                    ),
                )
            self._conn.commit()

    def get_attendance(
        self,
        session_id: str | None = None,
        status: str | None = None,
        zone_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query worker attendance records with calculated dwell time."""
        query = "SELECT * FROM worker_attendance WHERE 1=1"
        params: list[Any] = []
        if session_id and session_id != "all":
            query += " AND session_id = ?"
            params.append(session_id)
        if status and status != "all":
            query += " AND compliance_status = ?"
            params.append(status.upper())
        if zone_id and zone_id != "all":
            query += " AND zone_id = ?"
            params.append(zone_id)
        query += " ORDER BY last_seen DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self._conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            dwell = max(0.0, d["last_seen"] - d["first_seen"])
            d["dwell_seconds"] = round(dwell, 1)
            results.append(d)
        return results

    def get_attendance_summary(self) -> dict[str, Any]:
        """Aggregate shift attendance statistics."""
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) as total FROM worker_attendance")
        total = cur.fetchone()["total"]

        cur.execute("SELECT COUNT(*) as passed FROM worker_attendance WHERE compliance_status = 'PASSED'")
        passed = cur.fetchone()["passed"]

        flagged = total - passed
        pass_rate = round((passed / max(1, total)) * 100.0, 1) if total > 0 else 100.0

        return {
            "total_workers": total,
            "passed_workers": passed,
            "flagged_workers": flagged,
            "pass_rate_percent": pass_rate,
        }

    def clear_attendance(self) -> int:
        """Clear all worker attendance records."""
        cur = self._conn.execute("DELETE FROM worker_attendance")
        self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        self._conn.close()

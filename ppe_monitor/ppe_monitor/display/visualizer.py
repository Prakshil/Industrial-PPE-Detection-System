"""
Visualizer — draws tracked workers, associated PPE status and violation alerts
on each frame using OpenCV.
"""
from __future__ import annotations

import cv2
import numpy as np

from ppe_monitor.compliance.worker_state import ComplianceStatus, WorkerState
from ppe_monitor.detector.yolo_detector import RawDetection
from ppe_monitor.tracker.association import AssociationResult

# Colors: BGR
_GREEN  = (0,  200,   0)
_RED    = (0,   0,  220)
_YELLOW = (0, 200,  200)
_WHITE  = (255, 255, 255)
_BLACK  = (0,   0,    0)
_ORANGE = (0, 140,  255)

_STATUS_COLOR = {
    ComplianceStatus.COMPLIANT: _GREEN,
    ComplianceStatus.PENDING:   _YELLOW,
    ComplianceStatus.VIOLATION: _RED,
}


def _draw_filled_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float = 0.55,
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y + baseline), color, -1)
    cv2.putText(frame, text, (x + 2, y), font, font_scale, _WHITE, thickness,
                cv2.LINE_AA)


class Visualizer:
    """Draws the complete annotated frame for the local HMI display."""

    def __init__(self, show_ppe_boxes: bool = True):
        self.show_ppe_boxes = show_ppe_boxes

    def draw(
        self,
        frame: np.ndarray,
        tracked_persons: list[RawDetection],
        ppe_detections: list[RawDetection],
        associations: list[AssociationResult],
        worker_states: dict[int, WorkerState],
    ) -> np.ndarray:
        out = frame.copy()

        # Build lookup: track_id -> AssociationResult
        ar_by_id = {ar.track_id: ar for ar in associations}

        # Draw PPE item boxes (grey, thin)
        if self.show_ppe_boxes:
            for det in ppe_detections:
                x1, y1, x2, y2 = (int(v) for v in det.box)
                cv2.rectangle(out, (x1, y1), (x2, y2), (180, 180, 180), 1)
                _draw_filled_label(out, f"{det.class_name} {det.confidence:.2f}",
                                   (x1, y1 - 2), (120, 120, 120), 0.40, 1)

        # Draw each tracked person (strictly deduplicate so no track_id is ever drawn twice)
        drawn_tids: set[int] = set()
        for person in tracked_persons:
            if person.track_id is None or person.track_id in drawn_tids:
                continue
            drawn_tids.add(person.track_id)
            tid = person.track_id
            x1, y1, x2, y2 = (int(v) for v in person.box)
            ws = worker_states.get(tid)
            status = ws.status if ws else ComplianceStatus.COMPLIANT
            color = _STATUS_COLOR[status]

            # Person bounding box
            thickness = 3 if status == ComplianceStatus.VIOLATION else 2
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            # Prominent status header
            if status == ComplianceStatus.COMPLIANT:
                _draw_filled_label(out, f"[OK] COMPLIANT - ID:{tid}", (x1, y1 - 4), _GREEN, 0.52, 2)
            elif status == ComplianceStatus.PENDING:
                _draw_filled_label(out, f"VERIFYING - ID:{tid}", (x1, y1 - 4), _YELLOW, 0.50, 1)
            else:
                _draw_filled_label(out, f"ID:{tid}", (x1, y1 - 4), _RED, 0.50, 1)

            # PPE status panel (clamped inside frame bounds so it never clips off-screen)
            if ws:
                ar = ar_by_id.get(tid)
                if ar:
                    # Explicit 4-parameter ordering: helmet, vest, boots, gloves
                    display_ppe_order = ["helmet", "vest", "boots", "gloves"]
                    extra_items = [p for p in ar.ppe_present if p not in display_ppe_order]
                    all_display_items = display_ppe_order + extra_items

                    # Calculate total height needed; clamp inside frame bounds
                    item_height = 18
                    total_panel_h = len(all_display_items) * item_height
                    if y2 + total_panel_h + 10 <= out.shape[0]:
                        panel_y = y2 + 18
                    else:
                        # Draw neatly inside person box or clamped near bottom
                        panel_y = max(y1 + 45, out.shape[0] - total_panel_h - 10)

                    # Clamp horizontally to prevent text clipping
                    lbl_x = max(10, min(x1, out.shape[1] - 230))

                    for ppe in all_display_items:
                        present = ar.ppe_present.get(ppe, False)
                        is_required = ppe in ws.required_ppe

                        if present:
                            label_color = _GREEN
                            tick = "[OK]"
                            status_desc = "WORN"
                            counter_str = ""
                        else:
                            if is_required:
                                label_color = _RED
                                tick = "[X]"
                                status_desc = "MISSING"
                                count = ws.missing_counters.get(ppe, 0)
                                counter_str = f"({count}/{ws.confirmation_frames})" if count > 0 else ""
                            else:
                                label_color = (140, 140, 140)
                                tick = "[-]"
                                status_desc = "NOT AUDITED"
                                counter_str = ""

                        label = f"{tick} {ppe.upper()} {status_desc} {counter_str}".strip()
                        _draw_filled_label(out, label, (lbl_x, panel_y), label_color, 0.42, 1)
                        panel_y += item_height

            # Large VIOLATION overlay (deterministic fixed class ordering)
            if status == ComplianceStatus.VIOLATION and ws and ws.confirmed_violations:
                _priority = {"helmet": 0, "vest": 1, "boots": 2, "gloves": 3, "goggles": 4}
                active_violations = [v for v in ws.confirmed_violations if v in ws.required_ppe]
                if active_violations:
                    sorted_violations = sorted(
                        active_violations,
                        key=lambda v: _priority.get(v.lower(), 99),
                    )
                    alert_text = "VIOLATION: " + ", ".join(v.upper() for v in sorted_violations)
                    _draw_filled_label(out, alert_text, (x1, y1 - 24), _RED, 0.55, 2)

        # Frame-level summary top-left
        n_violation = sum(
            1 for ws in worker_states.values()
            if ws.status == ComplianceStatus.VIOLATION
        )
        summary = f"Workers: {len(tracked_persons)}  Violations: {n_violation}"
        _draw_filled_label(out, summary, (10, 28), _BLACK, 0.65, 2)

        # Top-right 4-Item PPE Status Checklist HUD
        if tracked_persons and associations:
            primary_ar = associations[0]
            hud_x = max(10, out.shape[1] - 180)
            hud_y = 24
            _draw_filled_label(out, "PPE CHECKLIST (4-ITEMS)", (hud_x, hud_y), _BLACK, 0.42, 1)
            hud_y += 18
            for ppe in ["helmet", "vest", "boots", "gloves"]:
                present = primary_ar.ppe_present.get(ppe, False)
                color = _GREEN if present else _RED
                tag = f"[OK] {ppe.upper()}" if present else f"[X]  {ppe.upper()}"
                _draw_filled_label(out, tag, (hud_x, hud_y), color, 0.40, 1)
                hud_y += 18

        return out


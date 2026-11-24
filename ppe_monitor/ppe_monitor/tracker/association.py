"""
Spatial Association Engine — Anatomical Ratio Partitioning.

Maps each tracked person's bounding box into approximate body regions,
then checks whether detected PPE items fall within the correct region.
A helmet near a worker's feet does NOT count as a worn helmet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ppe_monitor.detector.yolo_detector import RawDetection

# Default body region ratios (top of person box = 0.0, bottom = 1.0)
# Format: (ratio_from_top, ratio_to_top)
# Accommodates both full-body industrial feeds and laptop webcam upper-body framing
DEFAULT_BODY_REGIONS: dict[str, tuple[float, float]] = {
    "helmet": (-0.35, 0.65),
    "goggles": (0.00, 0.40),
    "vest": (0.15, 0.80),
    "gloves": (0.35, 0.95),
    "boots": (0.65, 1.10),
}


@dataclass
class AssociationResult:
    """PPE association status for a single tracked worker."""

    track_id: int
    person_box: tuple[float, float, float, float]
    ppe_present: dict[str, bool] = field(default_factory=dict)
    ppe_detections: dict[str, Optional[RawDetection]] = field(default_factory=dict)


def _centroid_in_region(
    ppe_box: tuple[float, float, float, float],
    region: tuple[float, float, float, float],
) -> bool:
    """True if the centroid of ppe_box falls inside region (x1,y1,x2,y2)."""
    cx = (ppe_box[0] + ppe_box[2]) / 2
    cy = (ppe_box[1] + ppe_box[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def _box_iou(a: tuple, b: tuple) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _get_body_region(
    person_box: tuple[float, float, float, float],
    ppe_class: str,
    region_map: dict[str, tuple[float, float]],
) -> tuple[float, float, float, float]:
    """
    Compute the pixel bounding box of a body region
    given the person's bounding box, with horizontal safety margin
    to handle head tilt and helmet width.
    """
    x1, y1, x2, y2 = person_box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    ratio_top, ratio_bot = region_map[ppe_class]
    margin_x = 0.15 * w
    return (x1 - margin_x, y1 + ratio_top * h, x2 + margin_x, y1 + ratio_bot * h)


def associate_ppe_to_workers(
    tracked_persons: list[RawDetection],
    ppe_detections: list[RawDetection],
    region_map: dict[str, tuple[float, float]] | None = None,
    iou_threshold: float = 0.20,
) -> list[AssociationResult]:
    """
    Core spatial association algorithm.

    For each tracked person:
      1. Compute the approximate body region for each PPE class.
      2. Check all detections of that class.
      3. A PPE item counts only when its centroid is inside its expected
         anatomical region.  IoU is retained as a secondary guard, not a
         substitute for location.
      4. Among valid candidates, keep the highest-confidence one.
    """
    if region_map is None:
        region_map = DEFAULT_BODY_REGIONS

    results: list[AssociationResult] = []

    for person in tracked_persons:
        if person.track_id is None:
            continue

        ar = AssociationResult(
            track_id=person.track_id,
            person_box=person.box,
        )

        pw = max(1.0, person.box[2] - person.box[0])
        ph = max(1.0, person.box[3] - person.box[1])

        for ppe_class, _ in region_map.items():
            if ppe_class not in region_map:
                continue
            body_region = _get_body_region(person.box, ppe_class, region_map)

            # Collect all detections of this PPE class
            candidates = [d for d in ppe_detections if d.class_name.lower() == ppe_class.lower()]

            best_det: Optional[RawDetection] = None
            best_conf = 0.0

            for cand in candidates:
                cand_cx = (cand.box[0] + cand.box[2]) / 2
                cand_cy = (cand.box[1] + cand.box[3]) / 2

                centroid_ok = _centroid_in_region(cand.box, body_region)
                iou_ok = _box_iou(cand.box, body_region) >= iou_threshold

                # Region IoU alone is unsafe: a large region can overlap an
                # unrelated object. Require the PPE centre to be anatomically
                # plausible; it is the decisive association criterion.
                if centroid_ok:
                    if cand.confidence > best_conf:
                        best_conf = cand.confidence
                        best_det = cand

            ar.ppe_present[ppe_class] = best_det is not None
            ar.ppe_detections[ppe_class] = best_det

        results.append(ar)

    return results

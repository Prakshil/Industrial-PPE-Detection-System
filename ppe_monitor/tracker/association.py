"""
PPE-to-Person Association Logic (Phase 5)

Maps detected PPE items to tracked persons by checking whether the PPE
centroid falls inside the corresponding body region of the person bounding box.

Body regions are defined as fractions of the person bounding box height:
- head:    top 20%
- face:    5%-30% from top
- torso:   20%-70%
- hands:   50%-80%
- feet:    bottom 20%

This is a direct port of the reference implementation in tests/test_compliance.py.
"""

from typing import Dict, List, Tuple

BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)
PPEDetection = Dict[str, float]  # {"class": str, "cx": float, "cy": float}
PPEStatus = Dict[str, bool]


# PPE class -> body region mapping
PPE_REGION_MAP: Dict[str, str] = {
    "helmet": "head",
    "goggles": "face",
    "mask": "face",
    "vest": "torso",
    "gloves": "hands",
    "boots": "feet",
    "harness": "torso",
    "face_shield": "face",
    "earmuff": "head",
    "safety_suit": "torso",
}


def _compute_body_regions(person_bbox: BBox) -> Dict[str, BBox]:
    """
    Compute body regions as fractions of person bounding box.
    Returns dict of region_name -> (x1, y1, x2, y2).
    """
    px1, py1, px2, py2 = person_bbox
    ph = py2 - py1

    return {
        "head": (px1, py1, px2, py1 + ph * 0.20),
        "face": (px1, py1 + ph * 0.05, px2, py1 + ph * 0.30),
        "torso": (px1, py1 + ph * 0.20, px2, py1 + ph * 0.70),
        "hands": (px1, py1 + ph * 0.50, px2, py1 + ph * 0.80),
        "feet": (px1, py1 + ph * 0.80, px2, py2),
    }


def _point_in_box(cx: float, cy: float, box: BBox) -> bool:
    """Check if point (cx, cy) is inside box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    return x1 <= cx <= x2 and y1 <= cy <= y2


def associate_ppe(person_bbox: BBox, ppe_detections: List[PPEDetection]) -> PPEStatus:
    """
    Associate PPE detections to a person by body-region containment.

    Args:
        person_bbox: (x1, y1, x2, y2) of the tracked person
        ppe_detections: List of {"class": str, "cx": float, "cy": float}

    Returns:
        Dict mapping PPE class -> True/False (detected in correct region)
    """
    regions = _compute_body_regions(person_bbox)

    # Initialize all known PPE classes as False
    status: PPEStatus = {k: False for k in PPE_REGION_MAP}

    for det in ppe_detections:
        cls = det.get("class")
        cx = det.get("cx")
        cy = det.get("cy")

        if cls not in PPE_REGION_MAP:
            # Unknown class — ignore (not an error)
            continue

        if cx is None or cy is None:
            continue

        region_name = PPE_REGION_MAP[cls]
        region_box = regions[region_name]

        if _point_in_box(cx, cy, region_box):
            status[cls] = True

    return status


def associate_ppe_multi(
    person_bboxes: List[BBox], ppe_detections: List[PPEDetection]
) -> List[PPEStatus]:
    """
    Associate PPE detections to multiple persons.
    Each PPE detection is assigned to the first person whose body region contains it.
    """
    results: List[PPEStatus] = []
    used_detections = set()

    for person_bbox in person_bboxes:
        status = {k: False for k in PPE_REGION_MAP}
        regions = _compute_body_regions(person_bbox)

        for i, det in enumerate(ppe_detections):
            if i in used_detections:
                continue

            cls = det.get("class")
            cx = det.get("cx")
            cy = det.get("cy")

            if cls not in PPE_REGION_MAP or cx is None or cy is None:
                continue

            region_name = PPE_REGION_MAP[cls]
            region_box = regions[region_name]

            if _point_in_box(cx, cy, region_box):
                status[cls] = True
                used_detections.add(i)

        results.append(status)

    return results

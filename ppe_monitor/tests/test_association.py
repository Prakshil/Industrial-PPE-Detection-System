"""
Unit test for Spatial Association Engine (Anatomical Ratio Partitioning).
Verifies:
  1. A helmet on the head is correctly associated.
  2. A helmet near the feet is REJECTED (does NOT count as worn).
  3. Boots at the feet and vest on the torso are correctly associated.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppe_monitor.detector.yolo_detector import RawDetection
from ppe_monitor.tracker.association import (
    DEFAULT_BODY_REGIONS,
    associate_ppe_to_workers,
)


def test_spatial_association():
    # Person box: x1=100, y1=100, x2=300, y2=500 (height = 400)
    # Head region (0.00 to 0.25): y: 100 to 200
    # Torso region (0.20 to 0.70): y: 180 to 380
    # Feet region (0.75 to 1.00): y: 400 to 500
    person = RawDetection(
        class_name="Person",
        confidence=0.92,
        box=(100.0, 100.0, 300.0, 500.0),
        track_id=1,
    )

    # 1. Helmet on head
    head_helmet = RawDetection(
        class_name="helmet",
        confidence=0.88,
        box=(150.0, 110.0, 250.0, 170.0),  # centroid y = 140 (head)
    )

    # 2. Vest on torso
    torso_vest = RawDetection(
        class_name="vest",
        confidence=0.85,
        box=(120.0, 200.0, 280.0, 360.0),  # centroid y = 280 (torso)
    )

    # 3. Boots on feet
    feet_boots = RawDetection(
        class_name="boots",
        confidence=0.82,
        box=(130.0, 420.0, 270.0, 490.0),  # centroid y = 455 (feet)
    )

    # Scenario 1: Fully equipped worker
    results = associate_ppe_to_workers([person], [head_helmet, torso_vest, feet_boots])
    assert len(results) == 1
    ar = results[0]
    assert ar.ppe_present["helmet"] is True, "Helmet on head should be associated"
    assert ar.ppe_present["vest"] is True, "Vest on torso should be associated"
    assert ar.ppe_present["boots"] is True, "Boots on feet should be associated"
    assert ar.ppe_present["gloves"] is False, "Gloves should be missing"
    assert ar.ppe_present["goggles"] is False, "Goggles should be missing"
    print("[PASS] Test 1: Compliant PPE correctly associated to head, torso, feet.")

    # Scenario 2: Negative test - Helmet placed at feet (e.g. dropped on ground)
    dropped_helmet = RawDetection(
        class_name="helmet",
        confidence=0.89,
        box=(150.0, 430.0, 250.0, 480.0),  # centroid y = 455 (feet region!)
    )
    results_dropped = associate_ppe_to_workers([person], [dropped_helmet])
    ar_dropped = results_dropped[0]
    assert (
        ar_dropped.ppe_present["helmet"] is False
    ), "Helmet near feet must NOT be counted as worn helmet!"
    print("[PASS] Test 2: Helmet near feet correctly REJECTED by Anatomical Ratio Partitioning.")

    print("\nALL SPATIAL ASSOCIATION TESTS PASSED!")


if __name__ == "__main__":
    test_spatial_association()

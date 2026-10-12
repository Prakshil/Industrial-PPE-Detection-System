"""
Unit tests for the PPE Compliance Engine
Phase 1 baseline — tests core logic used in Phases 5 & 6

Run with:  pytest tests/ -v
"""

# ================================================================
# Tests for: PPE → Person Association Logic (Phase 5)
# ================================================================


def associate_ppe(person_bbox, ppe_detections):
    """
    Inline copy of the association logic for unit testing.
    person_bbox = (x1, y1, x2, y2)
    Returns dict of {ppe_class: True/False}
    """
    px1, py1, px2, py2 = person_bbox
    ph = py2 - py1

    regions = {
        "head": (px1, py1, px2, py1 + ph * 0.20),
        "face": (px1, py1 + ph * 0.05, px2, py1 + ph * 0.30),
        "torso": (px1, py1 + ph * 0.20, px2, py1 + ph * 0.70),
        "hands": (px1, py1 + ph * 0.50, px2, py1 + ph * 0.80),
        "feet": (px1, py1 + ph * 0.80, px2, py2),
    }

    ppe_map = {
        "helmet": "head",
        "goggles": "face",
        "mask": "face",
        "vest": "torso",
        "gloves": "hands",
        "boots": "feet",
        "harness": "torso",
    }

    status = {k: False for k in ppe_map}
    for det in ppe_detections:
        cls, cx, cy = det["class"], det["cx"], det["cy"]
        if cls in ppe_map:
            r = regions[ppe_map[cls]]
            if r[0] <= cx <= r[2] and r[1] <= cy <= r[3]:
                status[cls] = True

    return status


class TestAssociatePPE:

    def test_helmet_in_head_region_detected(self):
        """Helmet centroid in head region → True"""
        person = (100, 0, 300, 400)  # height=400 → head region y: 0–80
        helmet_det = [{"class": "helmet", "cx": 200, "cy": 40}]
        result = associate_ppe(person, helmet_det)
        assert result["helmet"] is True

    def test_helmet_below_head_region_ignored(self):
        """Helmet centroid below head region → False (e.g., held in hand)"""
        person = (100, 0, 300, 400)  # head region y: 0–80
        helmet_det = [{"class": "helmet", "cx": 200, "cy": 300}]
        result = associate_ppe(person, helmet_det)
        assert result["helmet"] is False

    def test_vest_in_torso_region_detected(self):
        """Vest centroid in torso region → True"""
        person = (0, 0, 100, 400)  # height=400 → torso y: 80–280
        vest_det = [{"class": "vest", "cx": 50, "cy": 180}]
        result = associate_ppe(person, vest_det)
        assert result["vest"] is True

    def test_boots_in_feet_region_detected(self):
        """Boots centroid in feet region → True"""
        person = (0, 0, 100, 500)  # height=500 → feet y: 400–500
        boots_det = [{"class": "boots", "cx": 50, "cy": 460}]
        result = associate_ppe(person, boots_det)
        assert result["boots"] is True

    def test_multiple_ppe_items_all_detected(self):
        """Multiple PPE items, all in correct regions"""
        person = (0, 0, 200, 500)
        dets = [
            {"class": "helmet", "cx": 100, "cy": 30},  # head: 0–100
            {"class": "vest", "cx": 100, "cy": 200},  # torso: 100–350
            {"class": "gloves", "cx": 100, "cy": 300},  # hands: 250–400
            {"class": "boots", "cx": 100, "cy": 450},  # feet: 400–500
        ]
        result = associate_ppe(person, dets)
        assert result["helmet"] is True
        assert result["vest"] is True
        assert result["gloves"] is True
        assert result["boots"] is True

    def test_empty_detection_list(self):
        """No detections → all False"""
        person = (0, 0, 200, 500)
        result = associate_ppe(person, [])
        assert all(v is False for v in result.values())

    def test_unknown_ppe_class_ignored(self):
        """Unknown class → ignored, no KeyError"""
        person = (0, 0, 200, 500)
        dets = [{"class": "unknown_gear", "cx": 100, "cy": 100}]
        result = associate_ppe(person, dets)  # should not raise
        assert "unknown_gear" not in result


# ================================================================
# Tests for: Compliance Engine (Phase 6)
# ================================================================


def evaluate_compliance(worker_ppe_status, required_ppe):
    """
    Inline copy of the compliance engine for unit testing.
    required_ppe: list of class names required for the zone.
    Returns {'compliant': bool, 'missing': list}
    """
    missing = [item for item in required_ppe if not worker_ppe_status.get(item, False)]
    return {"compliant": len(missing) == 0, "missing": missing}


class TestEvaluateCompliance:

    def test_fully_compliant_worker(self):
        """All required PPE present → compliant"""
        status = {"helmet": True, "vest": True, "gloves": True, "boots": True}
        result = evaluate_compliance(status, ["helmet", "vest", "gloves", "boots"])
        assert result["compliant"] is True
        assert result["missing"] == []

    def test_missing_helmet_non_compliant(self):
        """Helmet absent → non-compliant, helmet in missing list"""
        status = {"helmet": False, "vest": True, "gloves": True, "boots": True}
        result = evaluate_compliance(status, ["helmet", "vest", "gloves", "boots"])
        assert result["compliant"] is False
        assert "helmet" in result["missing"]
        assert len(result["missing"]) == 1

    def test_multiple_missing_items(self):
        """Multiple missing → all listed"""
        status = {"helmet": False, "vest": True, "gloves": False, "boots": False}
        result = evaluate_compliance(status, ["helmet", "vest", "gloves", "boots"])
        assert result["compliant"] is False
        assert set(result["missing"]) == {"helmet", "gloves", "boots"}

    def test_zone_requires_only_helmet_and_vest(self):
        """Zone only requires helmet + vest; gloves absence is irrelevant"""
        status = {"helmet": True, "vest": True, "gloves": False}
        result = evaluate_compliance(status, ["helmet", "vest"])
        assert result["compliant"] is True
        assert result["missing"] == []

    def test_all_ppe_absent(self):
        """Worker with no PPE → all required items in missing"""
        status = {"helmet": False, "vest": False, "gloves": False, "boots": False}
        result = evaluate_compliance(status, ["helmet", "vest", "gloves", "boots"])
        assert result["compliant"] is False
        assert set(result["missing"]) == {"helmet", "vest", "gloves", "boots"}

    def test_no_ppe_required(self):
        """Empty required list → always compliant"""
        status = {"helmet": False}
        result = evaluate_compliance(status, [])
        assert result["compliant"] is True
        assert result["missing"] == []

    def test_vest_missing_via_inference_rule(self):
        """
        'no_vest' class doesn't exist in dataset.
        Vest absence is caught by evaluate_compliance when vest is required
        but not detected — not via a special case.
        """
        status = {"helmet": True, "vest": False, "gloves": True, "boots": True}
        result = evaluate_compliance(status, ["helmet", "vest", "gloves", "boots"])
        assert result["compliant"] is False
        assert "vest" in result["missing"]


# ================================================================
# Tests for: Temporal Confirmation Logic (Phase 6)
# ================================================================


class WorkerState:
    """Inline copy for testing."""

    def __init__(self, confirm_threshold=5, cooldown_seconds=10):
        self.violation_frames = {}
        self.CONFIRM_THRESHOLD = confirm_threshold
        self.COOLDOWN_SECONDS = cooldown_seconds
        self.triggered_alerts = []

    def update(self, track_id, compliance_result, timestamp=0):
        if not compliance_result["compliant"]:
            key = (track_id, tuple(sorted(compliance_result["missing"])))
            self.violation_frames[key] = self.violation_frames.get(key, 0) + 1
            if self.violation_frames[key] == self.CONFIRM_THRESHOLD:
                self.triggered_alerts.append((track_id, compliance_result["missing"], timestamp))
        else:
            for key in list(self.violation_frames):
                if key[0] == track_id:
                    del self.violation_frames[key]


class TestWorkerState:

    def test_alert_not_triggered_before_threshold(self):
        """Below N frames → no alert"""
        ws = WorkerState(confirm_threshold=5)
        for _ in range(4):
            ws.update(1, {"compliant": False, "missing": ["helmet"]})
        assert len(ws.triggered_alerts) == 0

    def test_alert_triggered_exactly_at_threshold(self):
        """Exactly N frames → alert fires once"""
        ws = WorkerState(confirm_threshold=5)
        for _ in range(5):
            ws.update(1, {"compliant": False, "missing": ["helmet"]})
        assert len(ws.triggered_alerts) == 1
        assert ws.triggered_alerts[0][0] == 1

    def test_alert_not_triggered_twice_for_same_violation(self):
        """N+1 frames → still only 1 alert (fires at threshold, not every frame)"""
        ws = WorkerState(confirm_threshold=5)
        for _ in range(10):
            ws.update(1, {"compliant": False, "missing": ["helmet"]})
        assert len(ws.triggered_alerts) == 1

    def test_violation_counter_resets_on_compliance(self):
        """Worker becomes compliant → counter resets → no alert"""
        ws = WorkerState(confirm_threshold=5)
        for _ in range(4):
            ws.update(1, {"compliant": False, "missing": ["helmet"]})
        # Worker puts on helmet
        ws.update(1, {"compliant": True, "missing": []})
        # Frames reset; 4 more non-compliant frames don't trigger
        for _ in range(4):
            ws.update(1, {"compliant": False, "missing": ["helmet"]})
        assert len(ws.triggered_alerts) == 0

    def test_different_workers_tracked_independently(self):
        """Two workers — each has independent counters"""
        ws = WorkerState(confirm_threshold=3)
        # Worker 1: 3 frames → alert
        for _ in range(3):
            ws.update(1, {"compliant": False, "missing": ["helmet"]})
        # Worker 2: only 2 frames → no alert
        for _ in range(2):
            ws.update(2, {"compliant": False, "missing": ["vest"]})
        assert len(ws.triggered_alerts) == 1
        assert ws.triggered_alerts[0][0] == 1  # only worker 1

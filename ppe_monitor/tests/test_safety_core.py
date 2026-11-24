"""Safety-critical unit tests that require neither a camera nor a YOLO model."""

from ppe_monitor.compliance.worker_state import ComplianceStatus, WorkerState
from ppe_monitor.detector.yolo_detector import RawDetection
from ppe_monitor.tracker.bytetrack_wrapper import PersonTracker


def test_confirmation_uses_elapsed_time_not_fps():
    state = WorkerState(
        track_id=1,
        zone_id="zone",
        required_ppe=["helmet"],
        confirmation_frames=25,
        cooldown_seconds=30,
        confirmation_seconds=1.0,
        pending_seconds=0.25,
    )
    assert not state.update({"helmet": False}, 1, now=100.0)
    assert state.status == ComplianceStatus.COMPLIANT
    assert not state.update({"helmet": False}, 2, now=100.5)
    assert state.status == ComplianceStatus.PENDING
    events = state.update({"helmet": False}, 3, now=101.01)
    assert len(events) == 1
    assert state.status == ComplianceStatus.VIOLATION


def test_visible_ppe_clears_violation_immediately():
    state = WorkerState(1, "zone", ["helmet"], 1, 30, confirmation_seconds=0.5)
    state.update({"helmet": False}, 1, now=10.0)
    state.update({"helmet": False}, 2, now=10.6)
    state.update({"helmet": True}, 3, now=10.7)
    assert state.status == ComplianceStatus.COMPLIANT


def test_new_track_is_not_aged_on_its_creation_frame():
    tracker = PersonTracker()
    person = RawDetection("Person", 0.9, (100.0, 100.0, 250.0, 400.0))
    tracked = tracker.update([person])
    assert tracked[0].track_id == 1
    assert tracker.tracks[0].lost_frames == 0

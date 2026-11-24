"""
Compliance Engine — orchestrates WorkerState objects across all tracked workers.
Workers that disappear from the scene have their state cleaned up after a timeout.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ppe_monitor.compliance.policy_loader import PolicyLoader
from ppe_monitor.compliance.worker_state import ComplianceStatus, ViolationEvent
from ppe_monitor.compliance.worker_state import WorkerState as ProductionWorkerState
from ppe_monitor.tracker.association import AssociationResult

PPEStatus = dict[str, bool]
ComplianceResult = dict[str, object]  # {"compliant": bool, "missing": list[str]}


def evaluate_compliance(worker_ppe_status: PPEStatus, required_ppe: list[str]) -> ComplianceResult:
    """
    Check if worker has all required PPE items.

    Args:
        worker_ppe_status: Dict from associate_ppe() mapping PPE class -> True/False
        required_ppe: List of PPE class names required for the zone

    Returns:
        {"compliant": bool, "missing": list[str]}
    """
    missing = [item for item in required_ppe if not worker_ppe_status.get(item, False)]
    return {"compliant": len(missing) == 0, "missing": missing}


@dataclass
class WorkerState:
    """
    Legacy worker state for unit test compatibility (tests/test_compliance.py).
    Tracks violation state per worker for temporal confirmation.
    Only triggers an alert after N consecutive non-compliant frames.
    """

    confirm_threshold: int = 5
    cooldown_seconds: int = 10
    violation_frames: dict[tuple[int, tuple[str, ...]], int] = field(default_factory=dict)
    triggered_alerts: list[tuple[int, list[str], float]] = field(default_factory=list)
    last_alert_time: dict[tuple[int, tuple[str, ...]], float] = field(default_factory=dict)

    def update(
        self, track_id: int, compliance_result: ComplianceResult, timestamp: float = 0.0
    ) -> bool:
        """
        Update worker state with latest compliance result.

        Args:
            track_id: ByteTrack track ID
            compliance_result: Output from evaluate_compliance()
            timestamp: Current timestamp (seconds)

        Returns:
            True if a new alert was triggered this frame, False otherwise
        """
        missing = tuple(sorted(compliance_result.get("missing", [])))
        key = (track_id, missing)

        if not compliance_result.get("compliant", True):
            self.violation_frames[key] = self.violation_frames.get(key, 0) + 1
            if self.violation_frames[key] == self.confirm_threshold:
                self.triggered_alerts.append((track_id, list(missing), timestamp))
                return True
        else:
            for k in list(self.violation_frames):
                if k[0] == track_id:
                    del self.violation_frames[k]

        return False

    def get_alerts(self) -> list[tuple[int, list[str], float]]:
        return self.triggered_alerts.copy()

    def clear_alerts(self) -> None:
        self.triggered_alerts.clear()
        self.last_alert_time.clear()


class ZonePolicy:
    """
    Zone-based PPE requirements for tests/test_compliance.py.
    """

    def __init__(self, zones: dict[str, list[str]]):
        self.zones = zones

    def required_ppe(self, zone_name: str) -> list[str]:
        return self.zones.get(zone_name, [])

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ZonePolicy":
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data.get("zones", {}))


def evaluate_worker_zone(
    worker_ppe_status: PPEStatus, zone_policy: ZonePolicy, zone_name: str
) -> ComplianceResult:
    required = zone_policy.required_ppe(zone_name)
    return evaluate_compliance(worker_ppe_status, required)


class ComplianceEngine:
    """
    Central engine that:
      1. Maintains one WorkerState per active track_id.
      2. Feeds each AssociationResult into the corresponding WorkerState.
      3. Returns newly confirmed ViolationEvents each frame.
      4. Prunes states for workers who have left the scene.
    """

    TRACK_TIMEOUT_SECONDS = 10  # remove state if track unseen for this long

    def __init__(self, policy_loader: PolicyLoader):
        self._policy = policy_loader
        self._worker_states: dict[int, ProductionWorkerState] = {}
        self._last_seen: dict[int, float] = {}
        self._frame_number: int = 0

    @property
    def worker_states(self) -> dict[int, ProductionWorkerState]:
        return dict(self._worker_states)

    def update(
        self,
        associations: list[AssociationResult],
        zone_id: str,
        now: float | None = None,
    ) -> list[ViolationEvent]:
        """
        Process one frame's association results for a given zone.

        Returns all new ViolationEvents from this frame.
        """
        self._frame_number += 1
        policy = self._policy.get_policy(zone_id)
        now = time.time() if now is None else now
        all_events: list[ViolationEvent] = []
        active_ids: set[int] = set()

        for ar in associations:
            tid = ar.track_id
            active_ids.add(tid)
            self._last_seen[tid] = now

            # Create a new WorkerState if this track_id is new
            if tid not in self._worker_states:
                self._worker_states[tid] = ProductionWorkerState(
                    track_id=tid,
                    zone_id=zone_id,
                    required_ppe=list(policy.required_ppe),
                    confirmation_frames=policy.confirmation_frames,
                    cooldown_seconds=policy.alert_cooldown_seconds,
                    confirmation_seconds=policy.confirmation_seconds,
                )
            else:
                # Dynamically sync policy if inspection mode changed (e.g. helmet_only <-> all)
                ws = self._worker_states[tid]
                curr_req = list(policy.required_ppe)
                if ws.required_ppe != curr_req:
                    ws.required_ppe = curr_req
                ws.confirmation_frames = policy.confirmation_frames
                ws.confirmation_seconds = policy.confirmation_seconds
                ws.cooldown_seconds = policy.alert_cooldown_seconds

            # Feed the association result to the worker's state
            events = self._worker_states[tid].update(
                ppe_present=ar.ppe_present,
                frame_number=self._frame_number,
                now=now,
            )
            all_events.extend(events)

        # Prune stale worker states
        stale = [
            tid for tid, ts in self._last_seen.items() if now - ts > self.TRACK_TIMEOUT_SECONDS
        ]
        for tid in stale:
            self._worker_states.pop(tid, None)
            self._last_seen.pop(tid, None)

        return all_events

    def get_status(self, track_id: int) -> Optional[ComplianceStatus]:
        ws = self._worker_states.get(track_id)
        return ws.status if ws else None

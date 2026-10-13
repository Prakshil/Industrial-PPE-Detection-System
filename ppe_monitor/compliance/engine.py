"""
Compliance Engine (Phase 6)

Evaluates whether a worker meets the PPE requirements for their zone.
Implements temporal confirmation to prevent single-frame flicker alerts.

Reference implementations from tests/test_compliance.py.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

PPEStatus = Dict[str, bool]
ComplianceResult = Dict[str, object]  # {"compliant": bool, "missing": List[str]}


def evaluate_compliance(worker_ppe_status: PPEStatus, required_ppe: List[str]) -> ComplianceResult:
    """
    Check if worker has all required PPE items.

    Args:
        worker_ppe_status: Dict from associate_ppe() mapping PPE class -> True/False
        required_ppe: List of PPE class names required for the zone

    Returns:
        {"compliant": bool, "missing": List[str]}
    """
    missing = [item for item in required_ppe if not worker_ppe_status.get(item, False)]
    return {"compliant": len(missing) == 0, "missing": missing}


@dataclass
class WorkerState:
    """
    Tracks violation state per worker for temporal confirmation.
    Only triggers an alert after N consecutive non-compliant frames.
    """

    confirm_threshold: int = 5
    cooldown_seconds: int = 10
    violation_frames: Dict[Tuple[int, Tuple[str, ...]], int] = field(default_factory=dict)
    triggered_alerts: List[Tuple[int, List[str], float]] = field(default_factory=list)
    last_alert_time: Dict[Tuple[int, Tuple[str, ...]], float] = field(default_factory=dict)

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
            # Violation ongoing — increment frame counter
            self.violation_frames[key] = self.violation_frames.get(key, 0) + 1

            # Fire alert ONLY when frame count exactly equals threshold (not >=)
            # This matches the original test behavior: one alert per violation sequence
            if self.violation_frames[key] == self.confirm_threshold:
                self.triggered_alerts.append((track_id, list(missing), timestamp))
                return True

        else:
            # Worker compliant — reset violation counter for this track
            for k in list(self.violation_frames):
                if k[0] == track_id:
                    del self.violation_frames[k]

        return False

    def get_alerts(self) -> List[Tuple[int, List[str], float]]:
        """Return all triggered alerts (track_id, missing_items, timestamp)."""
        return self.triggered_alerts.copy()

    def clear_alerts(self) -> None:
        """Clear alert history (e.g., at shift end)."""
        self.triggered_alerts.clear()
        self.last_alert_time.clear()


class ZonePolicy:
    """
    Zone-based PPE requirements.
    Loaded from YAML config (see master plan §6).
    """

    def __init__(self, zones: Dict[str, List[str]]):
        self.zones = zones

    def required_ppe(self, zone_name: str) -> List[str]:
        """Return required PPE list for a zone, or empty list if unknown."""
        return self.zones.get(zone_name, [])

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ZonePolicy":
        """Load zone policy from YAML file."""
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data.get("zones", {}))


def evaluate_worker_zone(
    worker_ppe_status: PPEStatus, zone_policy: ZonePolicy, zone_name: str
) -> ComplianceResult:
    """
    Convenience function: evaluate compliance for a worker in a specific zone.
    """
    required = zone_policy.required_ppe(zone_name)
    return evaluate_compliance(worker_ppe_status, required)

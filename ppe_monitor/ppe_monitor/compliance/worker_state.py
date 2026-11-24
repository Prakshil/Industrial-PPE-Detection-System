"""
WorkerState — per-worker PPE compliance state with temporal confirmation.

The key design:
  - Each missing-PPE class has a counter that increments every frame it is absent.
  - When the counter reaches N (confirmation_frames), the violation is confirmed.
  - If the PPE appears again, the counter resets to 0 and the violation is cleared.
  - A cooldown prevents repeated alerts for the same active violation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    PENDING = "pending"  # violation suspected but not yet confirmed
    VIOLATION = "violation"  # confirmed persistent violation


@dataclass
class ViolationEvent:
    track_id: int
    zone_id: str
    missing_ppe: str | list[str]
    timestamp: float
    frame_number: int


@dataclass
class WorkerState:
    """Mutable state for one tracked worker."""

    track_id: int
    zone_id: str
    required_ppe: list[str]
    confirmation_frames: int  # retained for audit compatibility
    cooldown_seconds: int
    confirmation_seconds: float = 2.0
    pending_seconds: float = 0.5

    # Internal counters
    _missing_counters: dict[str, int] = field(default_factory=dict)
    _detected_counters: dict[str, int] = field(default_factory=dict)
    _confirmed_violations: set[str] = field(default_factory=set)
    _last_alert_time: dict[str, float] = field(default_factory=dict)
    _last_logged_violations: set[str] = field(default_factory=set)
    _last_logged_time: float = 0.0
    _first_seen: float = field(default_factory=time.time)
    _last_updated: float = field(default_factory=time.time)
    _frame_count: int = 0
    _missing_since: dict[str, float] = field(default_factory=dict)

    def update(
        self,
        ppe_present: dict[str, bool],
        frame_number: int,
        now: float | None = None,
    ) -> list[ViolationEvent]:
        """
        Update this worker's state for one frame.

        Args:
            ppe_present: {ppe_class: True/False} from the spatial engine.
            frame_number: Current frame index (for event metadata).

        Returns:
            List of newly confirmed ViolationEvents (may be empty).
        """
        now = time.time() if now is None else now
        self._last_updated = now
        self._frame_count += 1
        new_events: list[ViolationEvent] = []

        # Prune internal collections to only include currently active required_ppe
        req_set = set(self.required_ppe)
        self._confirmed_violations = {p for p in self._confirmed_violations if p in req_set}
        self._missing_counters = {p: c for p, c in self._missing_counters.items() if p in req_set}
        self._detected_counters = {p: c for p, c in self._detected_counters.items() if p in req_set}

        for ppe in self.required_ppe:
            is_present = ppe_present.get(ppe, False)

            if is_present:
                self._detected_counters[ppe] = self._detected_counters.get(ppe, 0) + 1
                self._missing_counters[ppe] = 0
                self._confirmed_violations.discard(ppe)
                self._missing_since.pop(ppe, None)
            else:
                self._detected_counters[ppe] = 0
                self._missing_counters[ppe] = self._missing_counters.get(ppe, 0) + 1
                self._missing_since.setdefault(ppe, now)

                if (now - self._missing_since[ppe] >= self.confirmation_seconds) and (
                    self._missing_counters[ppe] >= min(self.confirmation_frames, 2)
                ):
                    self._confirmed_violations.add(ppe)

        # Emit unified violation event containing ALL currently violated PPE items for this worker
        confirmed_set = self._confirmed_violations & req_set
        if confirmed_set:
            time_since_last = now - self._last_logged_time
            # Emit event when violations first appear, change, or after cooldown expires
            if (
                confirmed_set != self._last_logged_violations
                or time_since_last >= self.cooldown_seconds
            ):
                self._last_logged_violations = set(confirmed_set)
                self._last_logged_time = now
                new_events.append(
                    ViolationEvent(
                        track_id=self.track_id,
                        zone_id=self.zone_id,
                        missing_ppe=sorted(list(confirmed_set)),
                        timestamp=now,
                        frame_number=frame_number,
                    )
                )
        else:
            self._last_logged_violations = set()

        return new_events

    @property
    def status(self) -> ComplianceStatus:
        req_set = set(self.required_ppe)
        active_violations = [p for p in self._confirmed_violations if p in req_set]
        if active_violations:
            return ComplianceStatus.VIOLATION
        if any(
            self._last_updated - self._missing_since.get(p, self._last_updated)
            >= self.pending_seconds
            for p in self.required_ppe
        ):
            return ComplianceStatus.PENDING
        return ComplianceStatus.COMPLIANT

    @property
    def missing_counters(self) -> dict[str, int]:
        return dict(self._missing_counters)

    @property
    def confirmed_violations(self) -> set[str]:
        return set(self._confirmed_violations)

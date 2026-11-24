"""
Compliance Engine — orchestrates WorkerState objects across all tracked workers.
Workers that disappear from the scene have their state cleaned up after a timeout.
"""
from __future__ import annotations

import time
from typing import Optional

from ppe_monitor.compliance.policy_loader import PolicyLoader
from ppe_monitor.compliance.worker_state import (
    ComplianceStatus,
    ViolationEvent,
    WorkerState,
)
from ppe_monitor.tracker.association import AssociationResult


class ComplianceEngine:
    """
    Central engine that:
      1. Maintains one WorkerState per active track_id.
      2. Feeds each AssociationResult into the corresponding WorkerState.
      3. Returns newly confirmed ViolationEvents each frame.
      4. Prunes states for workers who have left the scene.
    """

    TRACK_TIMEOUT_SECONDS = 10   # remove state if track unseen for this long

    def __init__(self, policy_loader: PolicyLoader):
        self._policy = policy_loader
        self._worker_states: dict[int, WorkerState] = {}
        self._last_seen: dict[int, float] = {}
        self._frame_number: int = 0

    @property
    def worker_states(self) -> dict[int, WorkerState]:
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
                self._worker_states[tid] = WorkerState(
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
            tid for tid, ts in self._last_seen.items()
            if now - ts > self.TRACK_TIMEOUT_SECONDS
        ]
        for tid in stale:
            self._worker_states.pop(tid, None)
            self._last_seen.pop(tid, None)

        return all_events

    def get_status(self, track_id: int) -> Optional[ComplianceStatus]:
        ws = self._worker_states.get(track_id)
        return ws.status if ws else None


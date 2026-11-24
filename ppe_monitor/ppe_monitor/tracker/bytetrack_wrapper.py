"""
ByteTrack / Hybrid IoU-Spatial Tracker for Person Detections.
Assigns rock-solid persistent track_ids to workers across frames.
Combines IoU with normalized centroid proximity to eliminate track ID jitter
when workers lean, tilt heads, or adjust PPE in front of the camera.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from ppe_monitor.detector.yolo_detector import RawDetection


def _compute_iou(
    boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]
) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    if interArea == 0.0:
        return 0.0

    boxAArea = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return float(iou)


def _compute_similarity(
    boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]
) -> float:
    """
    Computes hybrid affinity score combining IoU with normalized centroid distance.
    Prevents ID resets when a worker moves quickly or leans forward/backward.
    """
    iou = _compute_iou(boxA, boxB)

    cxA = (boxA[0] + boxA[2]) / 2.0
    cyA = (boxA[1] + boxA[3]) / 2.0
    cxB = (boxB[0] + boxB[2]) / 2.0
    cyB = (boxB[1] + boxB[3]) / 2.0

    wA = max(20.0, boxA[2] - boxA[0])
    hA = max(20.0, boxA[3] - boxA[1])

    # Normalized spatial displacement relative to bounding box scale
    norm_dx = abs(cxA - cxB) / wA
    norm_dy = abs(cyA - cyB) / hA
    norm_dist = math.sqrt(norm_dx**2 + norm_dy**2)

    # Convert distance to a proximity score [0, 1]
    prox_score = max(0.0, 1.0 - norm_dist / 1.6)

    if iou > 0.05:
        return float(0.60 * iou + 0.40 * prox_score)
    elif norm_dist < 0.85:
        return float(0.35 * prox_score)
    return 0.0


class Track:
    def __init__(self, track_id: int, box: Tuple[float, float, float, float]):
        self.track_id = track_id
        self.box = box
        self.lost_frames = 0
        self.hits = 1

    def update(self, box: Tuple[float, float, float, float]):
        self.box = box
        self.lost_frames = 0
        self.hits += 1


class PersonTracker:
    """
    Robust Person Tracker for continuous multi-object tracking.
    Assigns persistent track_id across consecutive frames using IoU & spatial proximity.
    """

    def __init__(
        self,
        track_thresh: float = 0.25,
        track_buffer: int = 45,  # Holds track alive for ~1.8s across temporary occlusions
        match_thresh: float = 0.15,
        **kwargs,
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.next_id = 1
        self.tracks: list[Track] = []

    def update(self, persons: list[RawDetection]) -> list[RawDetection]:
        """
        Assigns track_id to each person detection.
        Returns the same list with track_id populated.
        """
        if not persons:
            for t in self.tracks:
                t.lost_frames += 1
            self.tracks = [t for t in self.tracks if t.lost_frames <= self.track_buffer]
            return []

        # Filter persons above detection threshold
        valid_persons = [p for p in persons if p.confidence >= self.track_thresh]
        if not valid_persons:
            for t in self.tracks:
                t.lost_frames += 1
            self.tracks = [t for t in self.tracks if t.lost_frames <= self.track_buffer]
            return persons

        # Single-worker fast-path: if exactly 1 track and 1 detection in front of camera, lock track_id
        if len(self.tracks) == 1 and len(valid_persons) == 1:
            t = self.tracks[0]
            p = valid_persons[0]
            sim = _compute_similarity(t.box, p.box)
            if sim >= 0.10:
                t.update(p.box)
                p.track_id = t.track_id
                return valid_persons

        # Keep the pre-update list separate. New tracks are appended below and
        # must never be aged as though they existed at frame start.
        existing_tracks = list(self.tracks)
        matched_tracks = set()
        matched_detections = set()

        if self.tracks:
            # Build hybrid similarity matrix
            cost_matrix = np.zeros((len(existing_tracks), len(valid_persons)), dtype=np.float32)
            for i, t in enumerate(existing_tracks):
                for j, p in enumerate(valid_persons):
                    cost_matrix[i, j] = _compute_similarity(t.box, p.box)

            # Greedy matching starting from highest affinity
            while True:
                if cost_matrix.size == 0:
                    break
                max_idx = np.unravel_index(np.argmax(cost_matrix), cost_matrix.shape)
                max_sim = cost_matrix[max_idx]

                if max_sim < self.match_thresh:
                    break

                t_idx, p_idx = int(max_idx[0]), int(max_idx[1])
                matched_tracks.add(t_idx)
                matched_detections.add(p_idx)

                # Assign ID and update
                existing_tracks[t_idx].update(valid_persons[p_idx].box)
                valid_persons[p_idx].track_id = existing_tracks[t_idx].track_id

                # Zero out row and column to prevent reuse
                cost_matrix[t_idx, :] = -1.0
                cost_matrix[:, p_idx] = -1.0

        # Create new tracks for unmatched detections
        for j, p in enumerate(valid_persons):
            if j not in matched_detections:
                p.track_id = self.next_id
                self.tracks.append(Track(self.next_id, p.box))
                self.next_id += 1

        # Age unmatched existing tracks
        for i, t in enumerate(existing_tracks):
            if i not in matched_tracks:
                t.lost_frames += 1

        # Remove dead tracks that exceeded track_buffer
        self.tracks = [t for t in self.tracks if t.lost_frames <= self.track_buffer]

        # Ensure track_id uniqueness across returned persons in this frame
        used_ids = set()
        deduped = []
        for p in valid_persons:
            if p.track_id is not None:
                if p.track_id in used_ids:
                    p.track_id = self.next_id
                    self.tracks.append(Track(self.next_id, p.box))
                    self.next_id += 1
                used_ids.add(p.track_id)
            deduped.append(p)

        return deduped

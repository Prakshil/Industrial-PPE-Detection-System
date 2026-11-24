"""
YOLOv8 detection wrapper.
Returns detections split into persons and PPE items separately.
Optimized for zero-lag real-time GPU/CPU edge inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class RawDetection:
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    track_id: Optional[int] = None


def _box_iou(a: tuple, b: tuple) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _box_containment(a: tuple, b: tuple) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    min_area = min(
        max(1e-6, (a[2] - a[0]) * (a[3] - a[1])), max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    )
    return inter / min_area if min_area > 0 else 0.0


class PPEDetector:
    """Wraps YOLOv8 inference and splits detections into persons vs PPE."""

    PERSON_CLASS = "Person"
    PPE_CLASSES = {"helmet", "gloves", "vest", "boots", "goggles"}

    # Calibrated thresholds for accurate detection with robustness against lighting/angles
    CONF_PERSON = 0.35
    CONF_HELMET = 0.25
    CONF_VEST = 0.30
    CONF_GLOVES = 0.30
    CONF_BOOTS = 0.35
    CONF_OTHER_PPE = 0.30

    MIN_PERSON_WIDTH = 70
    MIN_PERSON_HEIGHT = 100
    MIN_PERSON_AREA = 10000

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.20,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
    ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        try:
            self.model.to(self.device)
        except Exception:
            self.device = "cpu"
            self.model.to("cpu")

        # Use 640 on GPU (~9ms), 480 on CPU for fast zero-lag execution
        self.imgsz = imgsz if self.device.startswith("cuda") else min(imgsz, 480)
        self.conf = max(0.05, conf_threshold)
        if conf_threshold:
            self.CONF_HELMET = min(self.CONF_HELMET, max(0.15, conf_threshold))
        self.iou = iou_threshold
        print(
            f"[YOLO DETECTOR] Initialized on {self.device} (imgsz={self.imgsz}, base_conf={self.conf}, helmet_conf={self.CONF_HELMET})"
        )

    def detect(self, frame: np.ndarray) -> tuple[list[RawDetection], list[RawDetection]]:
        """
        Returns (persons, ppe_items).
        """
        fh, fw = frame.shape[:2]
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        persons: list[RawDetection] = []
        ppe_items: list[RawDetection] = []

        if results.boxes is None or len(results.boxes) == 0:
            return persons, ppe_items

        boxes_xyxy = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cid in zip(boxes_xyxy, confs, class_ids):
            name = self.model.names[cid]
            c_val = float(conf)
            bx1, by1, bx2, by2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            bw = max(0.0, bx2 - bx1)
            bh = max(0.0, by2 - by1)
            barea = bw * bh

            det = RawDetection(
                class_name=name,
                confidence=c_val,
                box=(bx1, by1, bx2, by2),
            )

            if name.lower() == self.PERSON_CLASS.lower():
                # Filter out spurious tiny background noise
                if c_val < self.CONF_PERSON:
                    continue
                if (
                    bw < self.MIN_PERSON_WIDTH
                    or bh < self.MIN_PERSON_HEIGHT
                    or barea < self.MIN_PERSON_AREA
                ):
                    continue
                # Reject extreme border slivers
                if (bx1 <= 4 or bx2 >= fw - 4) and bw < 80:
                    continue

                det.class_name = self.PERSON_CLASS
                persons.append(det)

            elif name in self.PPE_CLASSES or name.lower() in self.PPE_CLASSES:
                ppe_name = name.lower()
                det.class_name = ppe_name

                # Per-class sensitivity thresholds
                if ppe_name == "helmet":
                    if c_val < self.CONF_HELMET:
                        continue
                elif ppe_name == "vest":
                    if c_val < self.CONF_VEST:
                        continue
                elif ppe_name == "gloves":
                    if c_val < self.CONF_GLOVES:
                        continue
                elif ppe_name == "boots":
                    if c_val < self.CONF_BOOTS:
                        continue
                else:
                    if c_val < self.CONF_OTHER_PPE:
                        continue

                ppe_items.append(det)

        # Apply strict NMS and containment suppression to eliminate duplicate/ghost boxes on the same user
        if len(persons) > 1:
            persons.sort(key=lambda p: p.confidence, reverse=True)
            suppressed = []
            for p in persons:
                keep = True
                for s in suppressed:
                    if _box_iou(p.box, s.box) > 0.20 or _box_containment(p.box, s.box) > 0.50:
                        keep = False
                        break
                if keep:
                    suppressed.append(p)
            persons = suppressed

        return persons, ppe_items

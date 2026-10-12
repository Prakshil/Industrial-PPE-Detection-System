"""Run the repeatable parts of Phase 3 real-world validation.

Phase 3 is diagnostic only: it runs the Phase 2 YOLOv8s baseline with ByteTrack
on the three approved site clips, evaluates the held-out test split, and
reviews hard negatives. It does not implement PPE-to-person association, the
compliance engine, or any later-phase behaviour.

Two reporting corrections from the master plan are applied here:

* A single global detector confidence is used for inference, but the plan
  specifies per-class confidence thresholds. Those thresholds are therefore
  applied host-side to the returned boxes before they are written out, and can
  be enabled with ``--filter-per-class``.
* The aggregate mAP50 hides the fact that the PPE-present classes carry the
  usable signal while the ``no_*`` violation classes are noise. When a dataset
  YAML is supplied the report records both the aggregate metric and the
  PPE-present / violation group means so the gap is explicit.

The hard-negative CSV written by this script only ever contains raw model
predictions. The human-reviewed labels live in a separate file so that
re-running the validator can never overwrite reviewed evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

FAILURE_LOG_FIELDS = [
    "video_file",
    "frame_number",
    "timestamp_sec",
    "issue_type",
    "predicted_class",
    "expected_class",
    "confidence",
    "notes",
]

PREDICTION_FIELDS = [
    "image_file",
    "detection_count",
    "max_confidence",
    "detected_classes",
    "false_positive",
    "notes",
]

# Per-class confidence thresholds from the master plan (Section 6). The detector
# exposes one global confidence, so these are applied host-side as a filter.
DEFAULT_PER_CLASS_CONF: dict[str, float] = {
    "Person": 0.50,
    "person": 0.50,
    "helmet": 0.50,
    "no_helmet": 0.50,
    "vest": 0.45,
    "gloves": 0.40,
    "no_gloves": 0.40,
    "boots": 0.40,
    "no_boots": 0.40,
    "goggles": 0.35,
    "no_goggle": 0.35,
}

# Class groups used for the dual-metric report. Names are matched case-insensitively
# so both the frozen ``Person`` label and a lowercase ``person`` label are covered.
PPE_PRESENT_CLASSES = {"helmet", "vest", "gloves", "boots", "goggles", "person"}
VIOLATION_CLASSES = {"no_helmet", "no_goggle", "no_gloves", "no_boots"}

GLOBAL_INFERENCE_CONF = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/yolov8s_baseline_best.pt"))
    parser.add_argument(
        "--videos", type=Path, default=Path("data/site_videos"), help="Video directory"
    )
    parser.add_argument(
        "--hard-negatives",
        type=Path,
        default=Path("data/hard_negatives"),
        help="Hard-negative image directory",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Ultralytics dataset YAML used for the test split validation",
    )
    parser.add_argument("--output", type=Path, default=Path("docs/phase3_metrics.json"))
    parser.add_argument(
        "--review-output",
        type=Path,
        default=Path("docs/phase3_hard_negative_predictions.csv"),
        help="Raw predictions destination (human review file is never overwritten)",
    )
    parser.add_argument(
        "--per-class-conf",
        type=Path,
        help="Optional JSON file mapping class name to confidence threshold",
    )
    parser.add_argument(
        "--filter-per-class",
        action="store_true",
        help="Apply per-class thresholds host-side before writing predictions",
    )
    return parser.parse_args()


def load_per_class_conf(path: Path | None) -> dict[str, float]:
    thresholds = dict(DEFAULT_PER_CLASS_CONF)
    if path is None:
        return thresholds
    if not path.is_file():
        raise SystemExit(f"Per-class confidence file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name, value in payload.items():
        thresholds[str(name)] = float(value)
    return thresholds


def require_videos(path: Path) -> list[Path]:
    if not path.is_dir():
        raise SystemExit(f"Site video directory does not exist: {path}")
    videos = [path / f"clip{number}.mp4" for number in range(1, 4)]
    missing = [str(video) for video in videos if not video.is_file()]
    if missing:
        raise SystemExit(
            "Phase 3 requires clip1.mp4, clip2.mp4, and clip3.mp4: " + ", ".join(missing)
        )
    return videos


def filter_detections(
    boxes: Any,
    names: dict[int, str],
    thresholds: dict[str, float],
    apply_filter: bool,
) -> list[tuple[str, float]]:
    """Return the ``(class_name, confidence)`` pairs that survive filtering."""
    if boxes is None or len(boxes) == 0:
        return []
    kept: list[tuple[str, float]] = []
    for cls_value, conf_value in zip(boxes.cls, boxes.conf):
        class_name = names.get(int(cls_value), str(int(cls_value)))
        confidence = float(conf_value)
        if not apply_filter or confidence >= thresholds.get(class_name, 0.0):
            kept.append((class_name, confidence))
    return kept


def run_tracking(model: Any, videos: list[Path]) -> list[dict[str, str]]:
    tracking_runs = []
    for video in videos:
        results = model.track(
            source=str(video),
            tracker="bytetrack.yaml",
            persist=True,
            conf=GLOBAL_INFERENCE_CONF,
            iou=0.5,
            save=True,
            project="runs",
            name="track",
            exist_ok=True,
        )
        output = "runs/track"
        try:
            if results and getattr(results[0], "save_dir", None):
                output = str(Path(results[0].save_dir) / f"{video.stem}.avi")
        except (AttributeError, IndexError, TypeError):
            output = "runs/track"
        tracking_runs.append({"video": str(video), "output": output})
    return tracking_runs


def run_hard_negative_review(
    model: Any,
    images: list[Path],
    output_path: Path,
    thresholds: dict[str, float],
    apply_filter: bool,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = {int(key): str(value) for key, value in model.names.items()}
    detections = 0
    rows = []
    for image in images:
        results = model.predict(source=str(image), conf=GLOBAL_INFERENCE_CONF, verbose=False)
        boxes = results[0].boxes if results else None
        kept = filter_detections(boxes, names, thresholds, apply_filter)
        count = len(kept)
        max_confidence = ""
        if kept:
            max_confidence = f"{max(confidence for _, confidence in kept):.6f}"
        detected_classes = ";".join(sorted({name for name, _ in kept}))
        detections += int(count > 0)
        rows.append(
            {
                "image_file": str(image),
                "detection_count": count,
                "max_confidence": max_confidence,
                "detected_classes": detected_classes,
                "false_positive": "",
                "notes": "",
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return {"images_reviewed": len(images), "images_with_detections": detections}


def per_class_ap50(metrics: Any, names: dict[int, str]) -> dict[str, float]:
    """Extract per-class AP50 when the installed Ultralytics version exposes it."""
    box = getattr(metrics, "box", None)
    ap50 = getattr(box, "ap50", None) if box is not None else None
    if ap50 is None:
        return {}
    class_indices = getattr(box, "ap_class_index", None)
    if class_indices is None:
        class_indices = list(range(len(list(ap50))))
    result: dict[str, float] = {}
    for position, class_index in enumerate(class_indices):
        result[names.get(int(class_index), str(class_index))] = float(ap50[position])
    return result


def group_mean(values: dict[str, float], group: set[str]) -> float | None:
    selected = [value for name, value in values.items() if name.lower() in group]
    return sum(selected) / len(selected) if selected else None


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(
            f"Model not found: {args.model}. Download the Phase 2 YOLOv8s best.pt there first."
        )

    videos = require_videos(args.videos)
    thresholds = load_per_class_conf(args.per_class_conf)

    hard_negative_images = []
    if args.hard_negatives.is_dir():
        hard_negative_images = sorted(
            item
            for item in args.hard_negatives.iterdir()
            if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from error

    model = YOLO(str(args.model))
    names = {int(key): str(value) for key, value in model.names.items()}
    report: dict[str, Any] = {
        "model": str(args.model),
        "class_names": list(model.names.values()),
        "per_class_confidence": thresholds if args.filter_per_class else None,
        "tracking_runs": run_tracking(model, videos),
    }

    if args.data:
        if not args.data.is_file():
            raise SystemExit(f"Dataset YAML not found: {args.data}")
        metrics = model.val(data=str(args.data), split="test")
        ap50 = per_class_ap50(metrics, names)
        validation: dict[str, Any] = {
            "map50": float(metrics.box.map50),
            "map50_95_per_class": [float(value) for value in metrics.box.maps],
        }
        if ap50:
            validation["map50_per_class"] = ap50
            validation["ppe_present_map50"] = group_mean(ap50, PPE_PRESENT_CLASSES)
            validation["violation_map50"] = group_mean(ap50, VIOLATION_CLASSES)
        report["validation"] = validation

    if hard_negative_images:
        report["hard_negative_review"] = run_hard_negative_review(
            model,
            hard_negative_images,
            args.review_output,
            thresholds,
            args.filter_per_class,
        )
    else:
        report["hard_negative_review"] = {
            "images_reviewed": 0,
            "images_with_detections": 0,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

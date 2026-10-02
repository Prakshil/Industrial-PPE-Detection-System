"""Run the repeatable parts of Phase 3 real-world validation."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("models/yolov8s_baseline_best.pt")
    )
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
        default=Path("docs/phase3_hard_negative_review.csv"),
    )
    return parser.parse_args()


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


def run_tracking(model: Any, videos: list[Path]) -> list[dict[str, str]]:
    tracking_runs = []
    for video in videos:
        model.track(
            source=str(video),
            tracker="bytetrack.yaml",
            persist=True,
            conf=0.35,
            iou=0.5,
            save=True,
            project="runs",
            name="track",
            exist_ok=True,
        )
        tracking_runs.append({"video": str(video), "output": "runs/track"})
    return tracking_runs


def run_hard_negative_review(
    model: Any, images: list[Path], output_path: Path
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detections = 0
    rows = []
    for image in images:
        results = model.predict(source=str(image), conf=0.35, verbose=False)
        boxes = results[0].boxes if results else None
        count = len(boxes) if boxes is not None else 0
        max_confidence = ""
        if boxes is not None and boxes.conf is not None and len(boxes.conf):
            max_confidence = f"{float(boxes.conf.max().item()):.6f}"
        detections += int(count > 0)
        rows.append(
            {
                "image_file": str(image),
                "detection_count": count,
                "max_confidence": max_confidence,
                "false_positive": "",
                "notes": "",
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["image_file"])
        writer.writeheader()
        writer.writerows(rows)

    return {"images_reviewed": len(images), "images_with_detections": detections}


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(
            f"Model not found: {args.model}. Download the Phase 2 YOLOv8s best.pt there first."
        )

    videos = require_videos(args.videos)

    hard_negative_images = []
    if args.hard_negatives.is_dir():
        hard_negative_images = sorted(
            item
            for item in args.hard_negatives.iterdir()
            if item.is_file()
            and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from error

    model = YOLO(str(args.model))
    report: dict[str, Any] = {"model": str(args.model), "tracking_runs": run_tracking(model, videos)}

    if args.data:
        if not args.data.is_file():
            raise SystemExit(f"Dataset YAML not found: {args.data}")
        metrics = model.val(data=str(args.data), split="test")
        report["validation"] = {
            "map50": float(metrics.box.map50),
            "map50_95_per_class": [float(value) for value in metrics.box.maps],
            "class_names": list(model.names.values()),
        }

    if hard_negative_images:
        report["hard_negative_review"] = run_hard_negative_review(
            model, hard_negative_images, args.review_output
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

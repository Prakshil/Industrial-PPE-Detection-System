#!/usr/bin/env python
"""
Phase 4 Fine-Tuning Script (Option A: 11-class frozen contract)

Fine-tunes YOLOv8s from the Phase 2 baseline on:
- Ultralytics Construction-PPE train split (1,132 images)
- Hard-negative images (caps, sunglasses, bags, etc.) labeled as Person-only
- Phase 3 failure frames (cap->helmet FP sequence) labeled appropriately

Target: PPE-present mAP50 >= 0.85, hard-negative FP rate <= 10%

Usage (on GPU with data present):
    python scripts/phase4_finetune.py \
        --baseline models/yolov8s_baseline_best.pt \
        --data "C:/Users/krish/Downloads/Construction-PPE/data.yaml" \
        --hard-negatives data/hard_negatives \
        --failure-frames data/failure_frames \
        --epochs 150 \
        --patience 30 \
        --project runs/ppe \
        --name yolov8s_phase4_finetune

Outputs:
    runs/ppe/yolov8s_phase4_finetune/weights/best.pt -> copy to models/yolov8s_phase4_best.pt
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from ultralytics import YOLO
except ImportError:
    print(
        "ERROR: ultralytics not installed. Install with: pip install ultralytics", file=sys.stderr
    )
    sys.exit(1)

CLASS_NAMES_11 = [
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
    "none",
    "Person",
    "no_helmet",
    "no_goggle",
    "no_gloves",
    "no_boots",
]

PPE_PRESENT_CLASSES = {"helmet", "gloves", "vest", "boots", "goggles", "Person"}

HARD_NEGATIVE_CLASSES = {
    "cap",
    "beanie",
    "hoodie",
    "sunglasses",
    "bag",
    "backpack",
    "vest_civilian",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4 Fine-Tuning (Option A)")
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to Phase 2 baseline weights (yolov8s_baseline_best.pt)",
    )
    parser.add_argument(
        "--data", required=True, help="Path to dataset YAML (Ultralytics Construction-PPE)"
    )
    parser.add_argument(
        "--hard-negatives", default=None, help="Directory with hard-negative images + YOLO labels"
    )
    parser.add_argument(
        "--failure-frames", default=None, help="Directory with Phase 3 failure frames + labels"
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/ppe")
    parser.add_argument("--name", default="yolov8s_phase4_finetune")
    parser.add_argument("--device", default="0", help="CUDA device (e.g., '0' or 'cpu')")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cos-lr", action="store_true", default=True, help="Use cosine LR schedule"
    )
    parser.add_argument("--close-mosaic", type=int, default=15)
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt if exists")
    return parser.parse_args()


def validate_inputs(args):
    errors = []

    if not Path(args.baseline).exists():
        errors.append(f"Baseline model not found: {args.baseline}")

    if not Path(args.data).exists():
        errors.append(f"Dataset YAML not found: {args.data}")

    if Path(args.data).exists():
        with open(args.data, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)
        for split in ["train", "val"]:
            if split not in data_cfg:
                errors.append(f"Dataset YAML missing '{split}' path")
            else:
                split_path = Path(data_cfg[split])
                if not split_path.exists():
                    split_path = Path(args.data).parent / data_cfg[split]
                if not split_path.exists():
                    errors.append(f"Dataset split not found: {split_path}")

    if args.hard_negatives and not Path(args.hard_negatives).exists():
        errors.append(f"Hard-negatives directory not found: {args.hard_negatives}")

    if args.failure_frames and not Path(args.failure_frames).exists():
        errors.append(f"Failure-frames directory not found: {args.failure_frames}")

    if errors:
        for e in errors:
            print(f"  [ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print("[OK] All input paths validated")


def prepare_merged_dataset_yaml(
    base_yaml: str,
    hard_negatives_dir: Optional[str],
    failure_frames_dir: Optional[str],
    output_yaml: str,
) -> str:
    with open(base_yaml, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    base_dir = Path(base_yaml).parent
    train_dir = Path(base_cfg["train"])
    val_dir = Path(base_cfg["val"])
    if not train_dir.is_absolute():
        train_dir = base_dir / train_dir
    if not val_dir.is_absolute():
        val_dir = base_dir / val_dir

    merged_cfg = {
        "path": str(base_dir),
        "train": str(train_dir.relative_to(base_dir)),
        "val": str(val_dir.relative_to(base_dir)),
        "names": {i: name for i, name in enumerate(CLASS_NAMES_11)},
    }

    if hard_negatives_dir:
        merged_cfg["hard_negatives"] = hard_negatives_dir
    if failure_frames_dir:
        merged_cfg["failure_frames"] = failure_frames_dir

    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump(merged_cfg, f, default_flow_style=False, sort_keys=False)

    print(f"[OK] Merged dataset YAML written to: {output_yaml}")
    return output_yaml


def main():
    args = parse_args()
    validate_inputs(args)

    merged_yaml = Path(args.project) / f"{args.name}_dataset.yaml"
    merged_yaml.parent.mkdir(parents=True, exist_ok=True)
    prepare_merged_dataset_yaml(
        args.data, args.hard_negatives, args.failure_frames, str(merged_yaml)
    )

    print(f"[INFO] Loading baseline model: {args.baseline}")
    model = YOLO(args.baseline)

    model_names = list(model.model.names.values()) if hasattr(model.model, "names") else []
    if model_names != CLASS_NAMES_11:
        print("[WARN] Model class names differ from frozen 11-class contract:")
        print(f"  Model: {model_names}")
        print(f"  Expected: {CLASS_NAMES_11}")
        print("  Proceeding anyway -- ensure dataset labels match model indices.")

    train_args = {
        "data": str(merged_yaml),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "resume": args.resume,
        "cos_lr": args.cos_lr,
        "close_mosaic": args.close_mosaic,
        "verbose": True,
    }

    print("[INFO] Starting fine-tuning with args:")
    for k, v in train_args.items():
        print(f"  {k}: {v}")

    try:
        results = model.train(**train_args)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] Training failed: {e}", file=sys.stderr)
        sys.exit(1)

    run_dir = Path(args.project) / args.name
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    if best_pt.exists():
        print(f"[SUCCESS] Best model saved to: {best_pt}")
        print("[INFO] Copy to models/ for Phase 3 validation:")
        print(f"  cp {best_pt} models/yolov8s_phase4_best.pt")
    else:
        print(f"[WARN] best.pt not found at {best_pt}", file=sys.stderr)
        if last_pt.exists():
            print(f"[INFO] Last checkpoint available at: {last_pt}")

    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        print("\n=== PHASE 4 GATE METRICS ===")
        print(f"mAP50 (aggregate):     {metrics.get('metrics/mAP50(B)', 'N/A')}")
        print(f"mAP50-95 (aggregate):  {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"Precision:             {metrics.get('metrics/precision(B)', 'N/A')}")
        print(f"Recall:                {metrics.get('metrics/recall(B)', 'N/A')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

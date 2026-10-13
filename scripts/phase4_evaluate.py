#!/usr/bin/env python
"""
Phase 4 Gate Evaluation Script

Runs the Phase 3 validator on the Phase 4 fine-tuned model and checks
the ADR-001 gate criteria:
- PPE-present average mAP50 >= 0.85
- Hard-negative false-positive rate <= 10%

Usage:
    python scripts/phase4_evaluate.py \
        --model models/yolov8s_phase4_best.pt \
        --data "C:/Users/krish/Downloads/Construction-PPE/data.yaml" \
        --hard-negatives data/hard_negatives \
        --output docs/phase4_gate_report.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PPE_PRESENT_MAP50_THRESHOLD = 0.85
HARD_NEGATIVE_FP_RATE_THRESHOLD = 0.10

PPE_PRESENT_CLASSES = {"helmet", "gloves", "vest", "boots", "goggles", "Person"}
VIOLATION_CLASSES = {"no_helmet", "no_goggle", "no_gloves", "no_boots", "none"}


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4 Gate Evaluation")
    parser.add_argument("--model", required=True, help="Path to Phase 4 fine-tuned model (best.pt)")
    parser.add_argument("--data", required=True, help="Path to dataset YAML")
    parser.add_argument(
        "--hard-negatives", required=True, help="Directory with hard-negative images"
    )
    parser.add_argument(
        "--output", default="docs/phase4_gate_report.json", help="Output JSON report"
    )
    parser.add_argument(
        "--phase3-validator", default="scripts/phase3_validate.py", help="Path to Phase 3 validator"
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Inference confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="Tracking IoU threshold")
    return parser.parse_args()


def run_phase3_validator(
    model: str, data: str, validator_path: str, conf: float, iou: float
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        validator_path,
        "--model",
        model,
        "--data",
        data,
        "--conf",
        str(conf),
        "--iou",
        str(iou),
        "--split",
        "test",
    ]

    print(f"[INFO] Running Phase 3 validator: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    if result.returncode != 0:
        print(f"[ERROR] Phase 3 validator failed (exit code {result.returncode})", file=sys.stderr)
        print(f"STDOUT:\n{result.stdout}", file=sys.stderr)
        print(f"STDERR:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    metrics_path = Path("docs/phase3_metrics.json")
    if not metrics_path.exists():
        print(f"[ERROR] Validator did not produce {metrics_path}", file=sys.stderr)
        sys.exit(1)

    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_ppe_present_map50(metrics: Dict[str, Any]) -> float:
    validation = metrics.get("validation", {})
    map50_95_per_class = validation.get("map50_95_per_class", [])
    class_names = validation.get("class_names", [])

    if not map50_95_per_class or not class_names:
        return 0.0

    ppe_indices = [i for i, name in enumerate(class_names) if name in PPE_PRESENT_CLASSES]
    if not ppe_indices:
        return 0.0

    ppe_scores = [map50_95_per_class[i] for i in ppe_indices if i < len(map50_95_per_class)]
    return sum(ppe_scores) / len(ppe_scores) if ppe_scores else 0.0


def run_hard_negative_evaluation(
    model: str, hard_negatives_dir: str, conf: float
) -> Dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed", file=sys.stderr)
        sys.exit(1)

    model_obj = YOLO(model)
    hn_dir = Path(hard_negatives_dir)

    if not hn_dir.exists():
        print(f"[ERROR] Hard-negatives directory not found: {hn_dir}", file=sys.stderr)
        sys.exit(1)

    image_files = (
        list(hn_dir.glob("*.jpg")) + list(hn_dir.glob("*.png")) + list(hn_dir.glob("*.jpeg"))
    )
    if not image_files:
        print(f"[ERROR] No images found in {hn_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Evaluating {len(image_files)} hard-negative images...")

    fp_count = 0
    total = 0

    for img_path in image_files:
        total += 1
        results = model_obj(str(img_path), conf=conf, verbose=False)

        has_ppe_fp = False
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                cls_name = model_obj.names[cls_id]
                if cls_name in PPE_PRESENT_CLASSES:
                    has_ppe_fp = True
                    break
            if has_ppe_fp:
                break

        if has_ppe_fp:
            fp_count += 1

    fp_rate = fp_count / total if total > 0 else 0.0
    print(f"[INFO] Hard-negative evaluation: {fp_count}/{total} FP = {fp_rate:.2%}")

    return {
        "images_evaluated": total,
        "false_positives": fp_count,
        "fp_rate": fp_rate,
        "threshold": conf,
    }


def main():
    args = parse_args()

    for path, label in [
        (args.model, "Model"),
        (args.data, "Dataset YAML"),
        (args.hard_negatives, "Hard negatives"),
    ]:
        if not Path(path).exists():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    if not Path(args.phase3_validator).exists():
        print(f"[ERROR] Phase 3 validator not found: {args.phase3_validator}", file=sys.stderr)
        sys.exit(1)

    print("\n=== PHASE 3 VALIDATION (Test Split) ===")
    metrics = run_phase3_validator(
        args.model, args.data, args.phase3_validator, args.conf, args.iou
    )

    ppe_present_map50 = compute_ppe_present_map50(metrics)
    print(f"[INFO] PPE-present avg mAP50-95 (proxy): {ppe_present_map50:.4f}")

    print("\n=== HARD-NEGATIVE EVALUATION ===")
    hn_results = run_hard_negative_evaluation(args.model, args.hard_negatives, args.conf)

    ppe_pass = ppe_present_map50 >= PPE_PRESENT_MAP50_THRESHOLD
    fp_pass = hn_results["fp_rate"] <= HARD_NEGATIVE_FP_RATE_THRESHOLD
    overall_pass = ppe_pass and fp_pass

    print("\n=== PHASE 4 GATE RESULTS ===")
    print(
        f"PPE-present mAP50-95:  {ppe_present_map50:.4f}  "
        f"(threshold: {PPE_PRESENT_MAP50_THRESHOLD})  "
        f"{'PASS' if ppe_pass else 'FAIL'}"
    )
    print(
        f"Hard-negative FP rate: {hn_results['fp_rate']:.2%}  "
        f"(threshold: {HARD_NEGATIVE_FP_RATE_THRESHOLD:.0%})  "
        f"{'PASS' if fp_pass else 'FAIL'}"
    )
    print(
        f"OVERALL:               "
        f"{'PASS - Proceed to Phase 5' if overall_pass else 'FAIL - Iterate Phase 4'}"
    )

    report = {
        "model": args.model,
        "dataset_yaml": args.data,
        "gate_thresholds": {
            "ppe_present_map50_threshold": PPE_PRESENT_MAP50_THRESHOLD,
            "hard_negative_fp_rate_threshold": HARD_NEGATIVE_FP_RATE_THRESHOLD,
        },
        "ppe_present": {"map50_95_proxy": ppe_present_map50, "pass": ppe_pass},
        "hard_negative": hn_results,
        "hard_negative_pass": fp_pass,
        "overall_pass": overall_pass,
        "recommendation": (
            "Proceed to Phase 5" if overall_pass else "Collect more hard negatives / retrain"
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Gate report saved to: {output_path}")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()

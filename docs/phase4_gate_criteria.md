# Phase 4 Gate Criteria (Option A: 11-Class Frozen)

## Source
Derived from **ADR-001** (Violation Detection via Spatial Inference) and master plan Phase 4 requirements.

---

## Gate Metrics (Both Must Pass)

| Metric | Baseline (Phase 3) | Phase 4 Target | Measurement Method |
|---|---|---|---|
| **PPE-present avg mAP50** | 0.8328 (YOLOv8s) | **≥ 0.85** | Phase 3 validator on test split → average mAP50-95 over {helmet, gloves, vest, boots, goggles, Person} |
| **Hard-negative FP rate** | 21.43% (6/28) | **≤ 10%** | `phase4_evaluate.py` on `data/hard_negatives/` at `conf=0.35` |

> **Note:** Aggregate mAP50 is **not a gate** (per ADR-001). It is reported for transparency but does not block Phase 5.

---

## Evaluation Procedure

### 1. PPE-Present mAP50
```powershell
python scripts/phase3_validate.py `
  --model models/yolov8s_phase4_best.pt `
  --data "C:/Users/krish/Downloads/Construction-PPE/data.yaml" `
  --split test `
  --conf 0.35 `
  --iou 0.5
```
- Reads `docs/phase3_metrics.json` → `validation.map50_95_per_class` + `validation.class_names`
- Averages scores for indices matching PPE-present classes

### 2. Hard-Negative FP Rate
```powershell
python scripts/phase4_evaluate.py `
  --model models/yolov8s_phase4_best.pt `
  --data "C:/Users/krish/Downloads/Construction-PPE/data.yaml" `
  --hard-negatives data/hard_negatives `
  --output docs/phase4_gate_report.json
```
- Runs inference on every image in `data/hard_negatives/`
- Counts images with ≥1 PPE-present detection at `conf=0.35`
- FP rate = `false_positives / total_images`

---

## Decision Matrix

| PPE-present mAP50 | Hard-negative FP Rate | Decision |
|---|---|---|
| ≥ 0.85 | ≤ 10% | **PASS** → Proceed to Phase 5 |
| ≥ 0.85 | > 10% | **FAIL** → Add more hard negatives, retrain |
| < 0.85 | ≤ 10% | **FAIL** → Check for overfitting / data issues, retrain |
| < 0.85 | > 10% | **FAIL** → Major revision needed |

---

## Output Artifacts

| File | Description |
|---|---|
| `docs/phase4_gate_report.json` | Machine-readable gate result (PASS/FAIL + metrics) |
| `docs/phase3_metrics.json` | Updated by Phase 3 validator (test split metrics) |
| `models/yolov8s_phase4_best.pt` | Fine-tuned model (copy from `runs/ppe/.../weights/best.pt`) |

---

## Example `phase4_gate_report.json`

```json
{
  "model": "models/yolov8s_phase4_best.pt",
  "dataset_yaml": "C:/Users/krish/Downloads/Construction-PPE/data.yaml",
  "gate_thresholds": {
    "ppe_present_map50_threshold": 0.85,
    "hard_negative_fp_rate_threshold": 0.1
  },
  "ppe_present": {
    "map50_95_proxy": 0.8672,
    "pass": true
  },
  "hard_negative": {
    "images_evaluated": 32,
    "false_positives": 2,
    "fp_rate": 0.0625,
    "threshold": 0.35
  },
  "hard_negative_pass": true,
  "overall_pass": true,
  "recommendation": "Proceed to Phase 5"
}
```

---

## Remediation Checklist (If FAIL)

### If PPE-present mAP50 < 0.85
- [ ] Verify training didn't overfit (check val loss curves)
- [ ] Ensure hard negatives aren't diluting PPE signal (reduce count, increase PPE augmentation)
- [ ] Increase epochs / reduce patience / adjust LR
- [ ] Verify dataset labels correct (no mislabeled PPE boxes)

### If Hard-negative FP Rate > 10%
- [ ] Add more hard-negative images of the failing categories
- [ ] Verify hard-negative labels have **only** `Person` class (no PPE boxes)
- [ ] Increase `close_mosaic` (last 15-20 epochs no mosaic helps single-frame accuracy)
- [ ] Try lower `conf` during training (e.g., `conf=0.001` in train args) so model learns to suppress low-conf FPs

---

## Traceability

- **ADR-001** §Scope Impact: "Phase 4 fine-tuning scope changes: target PPE-present mAP ↑ and hard-negative FP ↓"
- **Master Plan** §7 Phase 4: "Hard-negative test passed; per-class recall acceptable"
- **README** §Roadmap: "Phase 4: collect approved site images, extract logged failure frames, freeze a class contract... fine-tune a versioned candidate"

---

*This gate is evaluated by `scripts/phase4_evaluate.py`. Run after `scripts/phase4_finetune.py` completes.*
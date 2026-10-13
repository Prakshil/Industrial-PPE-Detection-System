# ADR-001: Violation Detection via Spatial Inference (Not `no_X` Classification)

## Status
Proposed — awaiting review

## Context
The baseline YOLOv8 models (YOLOv8n mAP50 0.5568, YOLOv8s mAP50 0.5714) fail the 0.70 aggregate gate **solely because violation classes (`no_helmet`, `no_goggle`, `no_gloves`, `no_boots`) are severely underrepresented** in the Ultralytics Construction-PPE dataset (4–20 validation instances each). PPE-present classes average 0.81–0.83 AP50.

Phase 3 diagnostic validation confirms:
- Test mAP50: 0.5109
- Hard-negative FP rate: 21.43% (caps, sunglasses, bags misclassified as PPE)
- Violation-class mAP50-95: 0.0023–0.0441 (effectively noise)

Collecting sufficient `no_X` training data would require thousands of site images per violation class with consent, annotation, and deduplication — a multi-month effort.

## Decision
**Violation detection will be implemented as spatial inference in the Phase 6 Compliance Engine, not as a classification task.**

The detector will only output **PPE-present classes + Person**. For each tracked person, the compliance engine checks whether required PPE items fall inside the corresponding body region (head, face, torso, hands, feet). Absence → violation.

```python
# Pseudocode — actual implementation in ppe_monitor/compliance/engine.py
def evaluate_compliance(worker_ppe_status: dict, required_ppe: list[str]) -> dict:
    missing = [item for item in required_ppe if not worker_ppe_status.get(item, False)]
    return {"compliant": len(missing) == 0, "missing": missing}
```

Body-region association logic (Phase 5) already exists as tested reference implementations in `tests/test_compliance.py`.

## Consequences
### Positive
- Zero dependency on `no_X` training data.
- Deterministic, auditable logic — no model drift on violation classes.
- Works with any PPE-present recall the detector achieves (currently ≥0.81).
- Aligns with master plan §5 "Approach A — Inference Rule" and `Project_Status_Master_Report.md` §3.

### Negative
- Requires reliable Person tracking + PPE-to-person association (Phase 5).
- Single-frame occlusions can cause false violations → mitigated by Phase 6 temporal confirmation (`WorkerState` with 5-frame threshold, already tested).
- `no_X` classes remain in the frozen 11-class contract but are unused; they will be dropped if/when the class contract is updated to the 16-class unified list (master plan §5 Phase 4).

## Scope Impact
- **Phase 4 fine-tuning scope changes**: target PPE-present mAP ↑ and hard-negative FP ↓ (caps, sunglasses, bags). Do NOT collect/annotate `no_X` images for training.
- **Phase 5/6 can proceed now** — logic is pure Python, testable against existing Phase 3 detections, no GPU needed.
- **Class contract**: keep frozen 11-class until explicit approval for 16-class merge.

## Alternatives Considered
| Alternative | Rejected because |
|---|---|
| Collect `no_X` site data (Approach B) | Months of consent/annotation; not feasible without site access |
| Keep `no_X` classes and accept low recall | Unacceptable for safety system; false negatives = missed violations |
| Train a separate "absence" classifier | Same data problem; adds model complexity |

## References
- `docs/phase3_metrics.json` — per-class mAP50-95
- `docs/phase3_review_summary.md` — hard-negative FP analysis
- `Project_Status_Master_Report.md` §3 — architectural fix
- `PPE_Master_Implementation_Plan_K.md` §5 Approach A, §7 Phase 6
- `tests/test_compliance.py` — reference implementations for `associate_ppe`, `evaluate_compliance`, `WorkerState`
# Industrial PPE Detection System — Complete Work Handoff

This document is the single source of truth for handing the project over to the next
engineer. It records everything done so far (Phases 1–3 fixes, ADR-001, Phase 5/6 logic,
Phase 4 scaffolding), the current honest results, what is still blocked, and a detailed,
prioritized plan for the next phases — including the Roboflow dataset merge, synthetic data
generation, and real-world image collection.

Read this top to bottom before touching code. Where a claim is uncertain, it is marked as
uncertain. Do not upgrade any status to "done" without evidence in the repo.

---

## 0. TL;DR — Where We Are

- Phases 1–3 defects were inventoried (D1–D15), fixed, verified, and committed
  (`77ebf67 feat:issue fixes in 1-3 phases`). Phase 3 was merged earlier (`e8619a8`).
- The detector works on obvious PPE but produces unacceptable false positives on ordinary
  headwear (caps, beanies) and delivery gear. The explicit `no_*` violation classes are
  near-useless (e.g. `no_boots` mAP50-95 = 0.0023).
- **Architectural decision (ADR-001):** stop relying on `no_*` classifier outputs. Detect
  only PPE-present classes + `Person`, then infer violations by spatial association
  (PPE-to-body-region) in a compliance engine. This is why violation detection does not need
  `no_*` training data.
- Phase 5 association logic and Phase 6 compliance engine are **implemented as pure Python
  logic with unit tests** (23 tests pass). They are NOT wired into a live video pipeline yet.
- Phase 4 fine-tuning and gate-evaluation **scripts are scaffolded** but have **never been
  run** — no dataset, no weights, and `ultralytics` is not installed locally.
- **No improved metrics exist and none may be invented.** Improved numbers require a real GPU
  training run on real data. This handoff refuses to fabricate them.

**Honest verdict:** the project is at "diagnosed and architected, not yet retrained." The next
person's first real milestone is a reproducible Phase 4 retrain that beats the baseline on the
frozen test set AND on the hard-negative protocol.

---

## 1. Project Context

The long-term intended pipeline is:

```text
Camera/video
  -> YOLO PPE detection
  -> worker tracking (ByteTrack)
  -> PPE-to-person association
  -> zone compliance rules
  -> temporal confirmation
  -> RFID attendance
  -> MQTT/ESP32 safety notification
  -> SQLite event storage
  -> dashboard and edge deployment
```

The system is **not deployment-approved** and must never be used as the only safety control.
It is a research/diagnostic prototype.

---

## 2. Environment & Constraints (read before running anything)

- Repo: `C:\Users\Prakshil\OneDrive\Desktop\PPE\Ppe_detection`
- Remote: `https://github.com/Prakshil/Industrial-PPE-Detection-System.git`
- Default branch: `main`
- Platform: Windows, Python 3.12.1, RTX 4060 Laptop GPU present.
- **Not installed locally:** `ultralytics`. No local `models/` weights, no dataset, no
  `data/site_videos/*.mp4`, no `data/hard_negatives/`. All are Git-ignored.
- Verified tooling: pandas 2.2.2, numpy 1.26.4, matplotlib 3.8.4, black 26.5.1,
  isort 5.13.2, pytest 8.3.4, flake8 7.3.0.
- Windows console is cp1252; scripts call `sys.stdout.reconfigure(encoding="utf-8")`.
  Without this, non-ASCII prints raise `UnicodeEncodeError`.
- PowerShell has no heredoc. Do **not** append to files with `echo "" >>` — it introduced
  null bytes and broke parsing before. Use the write/edit tools instead.
- Notebook JSON convention: `indent=2`, `ensure_ascii=False`, **no trailing newline**.
- CI lint scope is `ppe_monitor/ tests/ scripts/` only (NOT `notebooks/`); line length 100;
  black/isort are blocking, flake8 is exit-zero (non-blocking).
- **Honest-positioning rule:** never fabricate improved metrics. Improved numbers must come
  from a real training run.

---

## 3. Data Contract (Frozen)

The current detector uses the official Ultralytics Construction-PPE dataset, downloaded from:

```text
https://ultralytics.com/assets/construction-ppe.zip
```

Exactly **11 classes**, order preserved everywhere:

```text
0  helmet
1  gloves
2  vest
3  boots
4  goggles
5  none
6  Person
7  no_helmet
8  no_goggle
9  no_gloves
10 no_boots
```

Split: `train 1,132 / val 143 / test 141`. The local YAML lives outside Git at
`C:\Users\krish\Downloads\Construction-PPE\data.yaml`.

The "PPE-present" set used for reporting is:
`{helmet, gloves, vest, boots, goggles, Person}`.
The "violation" set is `{no_helmet, no_goggle, no_gloves, no_boots}` (plus the odd `none`).

---

## 4. Phase 1–2 (original work, before us)

Two YOLOv8 baselines were trained on a Google Colab T4. Reported metrics:

| Metric | YOLOv8n | YOLOv8s |
| --- | ---: | ---: |
| Epochs | 55 | 93 |
| Aggregate mAP50 | 0.5568 | 0.5714 |
| mAP50-95 | 0.2573 | 0.2650 |
| Precision | 0.7313 | 0.6738 |
| Recall | 0.5157 | 0.5390 |
| F1 | 0.6049 | 0.5989 |
| PPE-present avg AP50 | 0.8106 | 0.8328 |
| Violation avg AP50 | 0.1811 | 0.1388 |
| Speed | 98.6 FPS (T4) | ~50 FPS (T4) |

The aggregate mAP50 target was `0.70`; neither baseline reached it. Best epoch: YOLOv8n 44,
YOLOv8s 45. Root cause of the low aggregate: severe class imbalance, driven almost entirely
by the `no_*` violation classes.

---

## 5. Phase 3 (diagnostic validation — completed)

Model restored locally at `models/yolov8s_baseline_best.pt`:

```text
File size: 22,533,354 bytes
SHA-256:   198E210B7689E89B4FBD1059185766B9532297BC91541107F1AB502F19148507
```

Registry/provenance: `docs/model_registry.md`.

### 5.1 Validator

`scripts/phase3_validate.py` requires the model + exactly three videos
(`data/site_videos/clip1.mp4`, `clip2.mp4`, `clip3.mp4`), runs ByteTrack at `conf=0.35`,
`iou=0.5`, validates on the dataset `test` split, evaluates `data/hard_negatives/`, and writes
machine-readable reports under `docs/`. It now uses per-class confidence thresholds
(`DEFAULT_PER_CLASS_CONF`: Person/helmet/no_helmet 0.50, vest 0.45,
gloves/no_gloves/boots/no_boots 0.40, goggles/no_goggle 0.35), writes non-destructive
predictions to `docs/phase3_hard_negative_predictions.csv` (never overwrites the human
`docs/phase3_hard_negative_review.csv`), reports both `map50` and `map50_95_per_class`, plus
`ppe_present_map50` and `violation_map50`.

Run command:

```powershell
python scripts\phase3_validate.py `
  --model models\yolov8s_baseline_best.pt `
  --data "C:\Users\krish\Downloads\Construction-PPE\data.yaml"
```

### 5.2 Test results

Aggregate test mAP50: **0.5109177849**. Per-class mAP50-95 (verified 11-class order):

| Class | mAP50-95 |
| --- | ---: |
| helmet | 0.4895 |
| gloves | 0.3850 |
| vest | 0.5005 |
| boots | 0.3875 |
| goggles | 0.3034 |
| none | 0.1136 |
| Person | 0.4262 |
| no_helmet | 0.0441 |
| no_goggle | 0.0410 |
| no_gloves | 0.0315 |
| no_boots | 0.0023 |

Full JSON: `docs/phase3_metrics.json`.

### 5.3 Video observations

`docs/phase3_failure_log.csv` holds 16 valid rows — one continuous false-positive sequence in
`clip2.mp4` (frames 350–541), NOT 16 independent incidents. A left-cropped person wearing a
maroon baseball cap is repeatedly predicted `helmet`. The same frames also contain a center
person correctly labelled `no_helmet` on a knit beanie; these are different people.

### 5.4 Hard-negative review

28 manually reviewed images in `data/hard_negatives/`, table at
`docs/phase3_hard_negative_review.csv`. Rules: a correct `Person` prediction is not a PPE
false positive; `none`/`no_*` are not counted as PPE-present FPs; only visibly wrong
PPE-present predictions count as `yes`.

```text
Reviewed: 28
Confirmed wrong PPE predictions: 6
False-positive rate: 6 / 28 = 21.43%  (at conf=0.35)
```

The six errors: ordinary headwear, sunglasses/dark clothing, a cap+backpack combo, a yellow
backpack/outerwear shot, and a visor/cap shot classified as PPE.

### 5.5 Phase 3 verdict

The baseline detects obvious PPE but confuses ordinary headwear and delivery gear with PPE,
and its explicit `no_*` violation classes are effectively broken. This justifies targeted data
+ hard-negative fine-tuning, and motivated ADR-001.

---

## 6. Defect Fixes (D1–D15) — committed in `77ebf67`

All of the following were fixed and verified (black/isort/flake8 clean, 19 tests at the time):

- `scripts/phase3_validate.py` — per-class thresholds, non-destructive prediction output, dual
  metrics, real `save_dir` recorded for tracking outputs.
- `notebooks/analyze_phase1.py` — `data_path()` resolves `notebooks/results/` with legacy
  fallback; UTF-8 stdout; dynamic values; added `fig2_diagnosis.png` + `fig3_top15_epochs.png`.
- `notebooks/analyze_phase1_final.py` — `results_path()`; correct reads; UTF-8; gate resolved
  (`aggregate_gate_passed`, `proceed_to_phase3: True` + `proceed_basis`).
- `notebooks/01_yolo_baseline.ipynb` — Cell 6 `EPOCHS_TOTAL=150`, `PATIENCE=30`, `SEED=42`,
  `training_is_finished()`; Cells 10/12 use the helper; Cell 22 gate reconciled.
- Docs reconciled: `README.md` (predictions-file path, per-class note, Python 3.10–3.12),
  `docs/phase3_metrics.json` (tracking outputs → `runs/detect/runs/track/clipN.avi`, D7),
  `PROJECT_WORK_HANDOFF.md`, `Project_Status_Master_Report.md` (phase numbering, 12 figures,
  scope note), `phase1_analysis_report.md`, `project_analysis.md` (SUPERSEDED banners),
  `tests/README.md`, `.gitignore` (`.pytest_cache/`), `.github/workflows/pr-check.yml`
  (Python 3.10/3.11/3.12, lint `scripts/`, lightweight deps, real YAML validation).
- `notebooks/results/phase1_summary.json` regenerated (D15 resolved).

Verify the whole repo with:

```powershell
black --check ppe_monitor/ tests/ scripts/ --line-length 100
isort --check-only ppe_monitor/ tests/ scripts/ --profile black
flake8 ppe_monitor/ tests/ scripts/ --count --exit-zero --max-complexity=12 --max-line-length=100 --statistics
python -m pytest tests/ -q
python notebooks\analyze_phase1.py
python notebooks\analyze_phase1_final.py
```

---

## 7. ADR-001 — Violation Detection via Spatial Inference

File: `docs/adr-001-violation-detection-via-spatial-inference.md` (status: Proposed —
awaiting review).

Decision: the detector emits only PPE-present classes + `Person`. Violations are derived by
checking whether a required PPE item has a matching detection inside the corresponding body
region of each tracked person, over a temporal window. Consequences:

- `no_*` classes are no longer a training target (they stay in the frozen 11-class contract
  only for backward compatibility with the baseline model).
- The Phase 4 objective becomes: raise **PPE-present** mAP while lowering false positives —
  NOT "fix `no_boots`".
- Violation quality is measured end-to-end (association + temporal confirmation), not by a
  per-class AP on `no_*`.

---

## 8. Phase 5 / Phase 6 — Implemented Logic (no live pipeline yet)

These are pure-Python modules with unit tests. They are NOT connected to a running video
pipeline.

### 8.1 Phase 5 — Association

`ppe_monitor/tracker/association.py`:

- `associate_ppe(detections, person_bbox, required_ppe)` → `Dict[str, bool]`.
- `associate_ppe_multi(detections, persons)` → per-person status.
- `PPE_REGION_MAP` and body regions as fractions of person bbox height:
  head top 20%, face 5–30%, torso 20–70%, hands 50–80%, feet bottom 20%.
- `_point_in_box` checks whether a PPE centroid falls in the region.

### 8.2 Phase 6 — Compliance engine

`ppe_monitor/compliance/engine.py`:

- `evaluate_compliance(worker_ppe_status, required_ppe)` → `{"compliant", "missing"}`.
- `WorkerState` — temporal confirmation. Fires an alert **only when the consecutive
  violation frame count equals `confirm_threshold`** (not `>=`); resets counters on
  compliance; supports `cooldown_seconds`; tracks per-`(track_id, missing-set)` keys.
- `ZonePolicy` — per-zone required PPE, loadable via `.from_yaml`.
- `evaluate_worker_zone(worker_ppe_status, zone_policy)`.

Exports: `ppe_monitor/tracker/__init__.py`, `ppe_monitor/compliance/__init__.py`.

### 8.3 Tests

`tests/test_compliance.py` now imports from the production modules (not inline copies) and
includes `TestAssociatePPE`, `TestEvaluateCompliance`, `TestWorkerState`, `TestZonePolicy`,
`TestAssociationComplianceIntegration`. **23 tests pass.**

---

## 9. Phase 4 — Scaffolding Only (never executed)

Files exist but have not been run because there is no dataset/weights locally and
`ultralytics` is not installed.

- `scripts/phase4_finetune.py` — fine-tunes YOLOv8s from the Phase 2 baseline on the
  Ultralytics train split + `data/hard_negatives/` + extracted failure frames.
- `scripts/phase4_evaluate.py` — runs the Phase 3 validator protocol + hard-negative
  evaluation and emits `docs/phase4_gate_report.json`. Uses PPE-present mAP50-95 as a proxy
  for the mAP50 gate.
- `docs/phase4_hard_negative_collection_guide.md` — what to collect and how to annotate.
- `docs/phase4_gate_criteria.md` — the exit gate.

### 9.1 Phase 4 gate (both must pass)

| Metric | Baseline | Target |
| --- | ---: | ---: |
| PPE-present avg mAP50 | 0.8328 | **≥ 0.85** |
| Hard-negative FP rate | 21.43% | **≤ 10%** |

Measured at `conf=0.35`. Aggregate mAP50 is NOT a gate. Do not call Phase 4 done until both
hold on the frozen test split + the hard-negative protocol, with a human review.

Intended run sequence (when data + GPU are available):

```powershell
python scripts\phase4_finetune.py `
  --baseline models\yolov8s_baseline_best.pt `
  --data "C:\Users\krish\Downloads\Construction-PPE\data.yaml" `
  --hard-negatives data\hard_negatives `
  --failure-frames data\failure_frames `
  --epochs 150 --patience 30

Copy-Item runs\ppe\yolov8s_phase4_finetune\weights\best.pt models\yolov8s_phase4_best.pt

python scripts\phase4_evaluate.py `
  --model models\yolov8s_phase4_best.pt `
  --data "C:\Users\krish\Downloads\Construction-PPE\data.yaml" `
  --hard-negatives data\hard_negatives `
  --output docs\phase4_gate_report.json
```

---

## 10. Current Honest Results Summary

| Item | Value | Status |
| --- | --- | --- |
| YOLOv8s baseline test mAP50 | 0.5109 | measured |
| PPE-present avg AP50 (baseline) | 0.8328 | measured |
| Hard-negative FP rate (baseline) | 21.43% | measured |
| `no_boots` mAP50-95 | 0.0023 | measured, effectively broken |
| Phase 4 model | — | NOT trained |
| Improved metrics | — | DO NOT INVENT |
| Phase 5/6 logic | implemented + unit-tested | no live pipeline |
| Live end-to-end system | — | not built |

---

## 11. What Is NOT Done

- Phase 4 fine-tuning run and its gate report
- Any dataset merge / remap (Roboflow)
- Synthetic data generation
- Real-world site image collection (consent + annotation)
- Wiring Phase 5/6 into a live video pipeline
- RFID attendance, SQLite event store, MQTT/ESP32, dashboard, edge deployment
- End-to-end replay test matrix, live pilot

The `ppe_monitor/` folders beyond the two Phase 5/6 modules are still scaffolding. Do not
describe them as production modules.

---

## 12. Next Phases — Detailed Plan (priority order)

### 12.1 Immediate unblockers (do first)

1. **Recover local assets from the original owner (`krish`):** the dataset
   (`data.yaml` + images/labels), the exact baseline weights, and especially
   `data/site_videos/clip1-3.mp4` and `data/hard_negatives/`. These are the highest-value
   assets and are already Git-ignored.
2. **Install the ML stack** and pin versions in a lockfile:
   `pip install ultralytics`, then record the exact version (baseline was trained on
   Ultralytics `8.4.143`, validation on `8.4.171` — there is no lockfile today).
3. **Extract the 16 failure frames** (clip2.mp4 frames 350–541) into `data/failure_frames/`
   with a manifest recording source clip, frame number, timestamp, and image hash. Keep this
   metadata in `docs/`, keep the media out of Git.

### 12.2 Phase 4 — retrain and gate

4. Collect hard negatives per `docs/phase4_hard_negative_collection_guide.md`.
5. Deduplicate, then split by camera/session (never by neighboring frames).
6. Run `scripts/phase4_finetune.py`, then `scripts/phase4_evaluate.py`.
7. Accept only if the gate in §9.1 passes; otherwise iterate or collect more data.
8. Record the final model hash in `docs/model_registry.md`.

### 12.3 Dataset Merge — Roboflow (Phase 4 expansion)

This is the "16-class unified" idea from the master plan. **Read the honest verdict before
committing effort.**

Proposed 16-class unified list:

| ID | Class | Source | Notes |
| --- | --- | --- | --- |
| 0 | helmet | Both | |
| 1 | no_helmet | Both | |
| 2 | vest | Both | no `no_vest` in either source |
| 3 | gloves | Both | |
| 4 | no_gloves | Ultralytics only | |
| 5 | boots | Both | |
| 6 | no_boots | Ultralytics only | |
| 7 | goggles | Both | |
| 8 | no_goggle | Ultralytics only | |
| 9 | mask | Roboflow | new |
| 10 | safety_suit | Roboflow | new (coverall) |
| 11 | face_shield | Roboflow | new |
| 12 | earmuff | Roboflow | new |
| 13 | no_earmuff | Roboflow | new |
| 14 | harness | Roboflow | new |
| 15 | person | Both | rename `Person` → `person` |

Source: Roboflow "PPE Detection (Projelerim)" ~6,550 images. Drop Ultralytics class `none`.

Three strategies, honestly rated:

| Strategy | What | Effort | Verdict |
| --- | --- | --- | --- |
| **A. Keep 11-class frozen** | Fine-tune on Ultralytics + hard negatives only | Low | **Recommended now** |
| **B. Full 16-class merge** | Download Roboflow, remap, dedup, annotate 100–300 site images, retrain | High (3–6 wks) | Only with Roboflow access + site access + annotation capacity |
| **C. Selective 12-class** | 11-class + `mask`, `face_shield`, `harness`, `safety_suit` (PPE-present only) | Medium | Reasonable middle path |

**Honest verdict on B/C:** robotic merging adds *volume*, not *value*, for the classes you
already have. The genuinely new classes (`mask`, `face_shield`, `harness`, `safety_suit`)
require **site-specific images anyway** — Roboflow's labels often will not match your camera
geometry, PPE brands, and lighting. Also note ADR-001 removes the need for `no_*` data, so
merging Roboflow's `no_*`/`no_earmuff` labels buys little.

**Do the merge only when all are true:** (a) you hold Roboflow access/licence, (b) you have
100+ consented site images, (c) you have annotation capacity, (d) you actually need the new
PPE types. If you proceed, write a dedicated remap+dedup script and split by session.

**Licensing:** verify each Roboflow Universe dataset's licence before training. Prefer
CC-BY-4.0 / CC0 / MIT. Record the source URL + licence for every external image source in
`docs/model_registry.md`.

### 12.4 Synthetic Dataset Generation

Zero legal risk, perfect labels, controllable domain. Recommended if real site images are
unavailable.

- Use Blender (+ Python) with MakeHuman/MB-Lab human meshes, PPE meshes (helmet, vest,
  gloves, boots, goggles), and hard-negative meshes (baseball cap, beanie, sunglasses,
  backpack).
- Use site background images as environment maps; randomize pose, camera angle, lighting,
  occlusion, and PPE/hard-negative combinations.
- Auto-generate YOLO labels from 3D→2D projection — no manual annotation.
- Pays off after roughly 200+ images; best used to bootstrap hard negatives and the new PPE
  classes, then fine-tune on a small set of real images.

Not yet built. A generator script is a candidate deliverable.

### 12.5 Real-World Image Collection (preferred over scraping)

Scraping is **not recommended and was declined**: copyright, privacy/GDPR, ToS, and zero
labels make it legally risky and low value. Legitimate alternatives:

| Source | Notes |
| --- | --- |
| Public CC-BY/CC0 datasets (Roboflow Universe, Kaggle Hard Hat Workers, CHV) | Downloadable in YOLO format; verify licence |
| Your own Phase 3 footage (clip1–3) + failure frames | Real site, real failure mode, already available |
| Consented site photos | 50–100 images at the entry gate across times of day; signed release + face blur |
| Staged hard negatives | Workers deliberately wearing caps/beanies/sunglasses; explicit consent |

Record consent/source/hash metadata outside Git. Blur faces before committing anything.

### 12.6 Later phases (not scoped yet)

- Wire Phase 5/6 into a live video pipeline (detect → track → associate → comply → confirm).
- RFID attendance, SQLite event schema, MQTT/ESP32 notification, FastAPI dashboard.
- Edge deployment (Raspberry Pi / Hailo) and a full end-to-end replay test matrix.

---

## 13. Files to Show the Next Person First

1. `README.md` — project overview and current status.
2. This file — the complete handoff.
3. `docs/adr-001-violation-detection-via-spatial-inference.md` — the key architectural
   decision.
4. `docs/phase3_review_summary.md` — Phase 3 results and limitations.
5. `docs/phase3_metrics.json` — machine-readable test metrics and class order.
6. `docs/phase3_failure_log.csv` — exact video observations.
7. `docs/phase3_hard_negative_review.csv` — image-by-image review decisions.
8. `docs/model_registry.md` — model hash, dataset path, class contract.
9. `docs/phase4_gate_criteria.md` + `docs/phase4_hard_negative_collection_guide.md`.
10. `scripts/phase3_validate.py`, `scripts/phase4_finetune.py`, `scripts/phase4_evaluate.py`.
11. `ppe_monitor/tracker/association.py`, `ppe_monitor/compliance/engine.py`,
    `tests/test_compliance.py`.

---

## 14. Bottom Line

Phase 1–2 established the baselines; Phase 3 diagnosed their failure honestly; the defect
sweep made the repo reproducible and CI-clean; ADR-001 redirected violation detection away
from the broken `no_*` classes toward spatial inference; Phase 5/6 association and compliance
logic is implemented and unit-tested; Phase 4 is scaffolded and gated but **not yet run**.

The single most valuable next action is to obtain the real site assets from `krish`, install
Ultralytics, and run a versioned Phase 4 retrain that clears the gate in §9.1 with a labeled
hard-negative set. Everything after that (Roboflow merge, synthetic data, extra PPE classes)
only makes sense once that first real improvement exists.

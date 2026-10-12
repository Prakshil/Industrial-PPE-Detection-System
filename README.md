# Industrial PPE Detection System

YOLOv8-based personal protective equipment detection for construction and industrial safety monitoring. The long-term design combines detection, worker tracking, PPE-to-person association, compliance rules, attendance, MQTT/IoT integration, and edge deployment.

This repository currently contains the Phase 1-2 baseline work and Phase 3 diagnostic validation evidence. It is not a deployment-ready safety system.

## Current Status

| Area | Status |
| --- | --- |
| Phase 1-2 baseline training | Complete in the original project work |
| Phase 3 real-world diagnostic validation | Evidence recorded; baseline weaknesses identified |
| Phase 4 site data expansion | Not started |
| Phases 5-12 production pipeline | Not started |
| Deployment approval | Not approved |

The current baseline is useful for research diagnostics only. It must not be used as the sole safety control or as a production gate decision.

## What Has Been Done

### Original Phase 1-2 work

The original project work trained YOLOv8n and YOLOv8s on the Ultralytics Construction-PPE dataset using Google Colab and a T4 GPU. It also added the initial project structure, CI workflow, analysis notebooks, baseline metrics, and compliance-logic tests.

Verified baseline results:

| Metric | YOLOv8n | YOLOv8s |
| --- | ---: | ---: |
| Aggregate mAP50 | 0.5568 | 0.5714 |
| mAP50-95 | 0.2573 | 0.2650 |
| Precision | 0.7313 | 0.6738 |
| Recall | 0.5157 | 0.5390 |
| PPE-present average AP50 | 0.8106 | 0.8328 |

The aggregate 0.70 target was not met. YOLOv8s was selected for Phase 3 diagnostic validation; YOLOv8n remains the planned edge-speed candidate, not an approved deployment model.

### Phase 3 work

The Phase 3 validator runs the YOLOv8s model with ByteTrack on exactly three clips, evaluates the official dataset test split, and reviews hard-negative images.

Recorded results:

- Test mAP50: `0.5109`
- Hard negatives reviewed: `28`
- Confirmed hard-negative false positives: `6/28` (`21.43%`)
- Main observed weakness: ordinary caps, beanies, and delivery gear can be classified as PPE
- Annotated outputs: `runs/detect/runs/track/`

The failure log currently contains 16 frame-level observations from one continuous cap false-positive sequence. These are not 16 independent incidents. PPE-person association and ID-switch adjudication are not implemented in the Phase 3 validator.

See:

- [Phase 3 review summary](docs/phase3_review_summary.md)
- [Phase 3 metrics](docs/phase3_metrics.json)
- [Failure log](docs/phase3_failure_log.csv)
- [Hard-negative review](docs/phase3_hard_negative_review.csv)
- [Model registry](docs/model_registry.md)

## Dataset

The baseline uses the official Ultralytics Construction-PPE dataset. The verified class contract is:

```text
helmet, gloves, vest, boots, goggles, none, Person,
no_helmet, no_goggle, no_gloves, no_boots
```

The local dataset used for Phase 3 was kept outside Git:

```text
C:\Users\krish\Downloads\Construction-PPE\data.yaml
```

It contains `train` (1,132 images), `val` (143 images), and `test` (141 images) splits. Use the real YAML path for validation; do not create a replacement YAML or mix in another dataset without documenting the class mapping.

## Repository Layout

```text
.
|-- .github/workflows/       CI checks
|-- docs/                    Phase 3 registry, metrics, logs, and review
|-- notebooks/               Baseline training and analysis notebooks
|-- ppe_monitor/             Package structure for later pipeline phases
|-- scripts/                 Phase 3 validation tooling
|-- tests/                   Compliance-logic tests
|-- requirements.txt         Python dependencies
```

Raw videos, hard-negative images, datasets, model weights, and generated `runs/` outputs are intentionally excluded from Git by `.gitignore`.

## Setup

Use Python 3.10–3.12 (the CI matrix covers 3.10, 3.11, and 3.12). Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The Phase 3 model is stored locally as:

```text
models\yolov8s_baseline_best.pt
```

Model provenance and SHA-256 are recorded in [docs/model_registry.md](docs/model_registry.md). Do not commit the weight file.

## Phase 3 Inputs

Keep exactly these three videos in the repository's ignored local data folder:

```text
data\site_videos\clip1.mp4
data\site_videos\clip2.mp4
data\site_videos\clip3.mp4
```

Keep 20-30 manually reviewed hard-negative images in:

```text
data\hard_negatives\
```

Review candidate media manually. A search result or filename is not evidence that an image contains the required PPE scenario.

## Run Phase 3 Validation

Run from the repository root:

```powershell
python scripts\phase3_validate.py `
  --model models\yolov8s_baseline_best.pt `
  --data "C:\Users\krish\Downloads\Construction-PPE\data.yaml"
```

The validator:

- runs ByteTrack with `conf=0.35` and `iou=0.5`;
- processes `clip1.mp4`, `clip2.mp4`, and `clip3.mp4`;
- writes annotated video under `runs/`;
- evaluates the requested dataset split using the real YAML;
- preserves the model class-name order in `docs/phase3_metrics.json`;
- writes raw hard-negative predictions to `docs/phase3_hard_negative_predictions.csv` (it never overwrites the human-reviewed `docs/phase3_hard_negative_review.csv`); and
- reports both aggregate and per-class metrics, plus separate group means for PPE-present and violation classes when per-class AP is available.

Optional per-class confidence filtering is available through `--filter-per-class` (with `--per-class-conf`); the defaults are tuned per class and are documented in `scripts/phase3_validate.py`.

Human review is still required. Do not treat detection counts as false-positive counts, and do not infer PPE-person association from detector output alone.

## Tests and CI

Run the current unit tests locally:

```powershell
pytest tests/ -v
```

The GitHub Actions workflow checks formatting with Black, import sorting with isort, flake8 diagnostics, pytest, and YAML configuration syntax. The current compliance tests contain temporary inline specifications for logic planned for later phases.

## Roadmap

1. Phase 4: collect approved site images, extract logged failure frames, freeze a class contract, deduplicate by session, and fine-tune a versioned candidate.
2. Phase 5: implement tested worker tracking and PPE-to-person association with explicit ambiguous and unknown states.
3. Phase 6: add zone policy and temporal compliance logic without one-frame safety decisions.
4. Phases 7-12: add RFID attendance, offline-first MQTT/SQLite integration, edge benchmarking, local dashboard, replay testing, and a controlled pilot.

Do not begin later phases from aggregate metrics alone. Each phase requires its documented evidence gate and human review.

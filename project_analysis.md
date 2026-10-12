# Industrial PPE Detection System — Full Project Analysis
### Research Grade · September 2026

> **PARTIALLY SUPERSEDED.** The literature review, competitive positioning, and CI/CD guidance remain useful. The Phase 1 status and results sections below were written against an **intermediate 88-epoch run** and are outdated. Final baseline results: YOLOv8n (55 epochs, best epoch 44, mAP50 0.5568) and YOLOv8s (93 epochs, best epoch 45, mAP50 0.5714). Authoritative files: `notebooks/results/phase1_metrics.json`, `notebooks/results/phase1_summary.json`, and `Project_Status_Master_Report.md`. Phase 3 diagnostic validation is also complete; see `docs/phase3_review_summary.md`.

---

## 1. Project Context Summary

| Field | Detail |
|---|---|
| **Official Title** | Real-Time Industrial PPE Detection System: YOLOv8 Deep Learning with IoT-Based Safety Monitoring |
| **Current Phase** | Phase 1-2 baseline complete (YOLOv8n 55 ep, YOLOv8s 93 ep); Phase 3 diagnostic validation complete |
| **Training Platform** | Google Colab · Tesla T4 GPU · CUDA 12.8 |
| **Framework** | Ultralytics 8.4.143 · PyTorch 2.11.0 · Python 3.13 |
| **Target Edge Hardware** | Raspberry Pi 5 (8 GB RAM) + Hailo AI HAT+ (13–26 TOPS NPU) |
| **Target IoT** | ESP32 via MQTT → Gate/Buzzer/LED control |
| **Phases Total** | 12 phases (~16–20 weeks) |
| **Publication Goal** | Research paper with end-to-end validated system metrics |

### System Unique Value Proposition (vs. Existing Papers)

This project is **not just a YOLO detector** — it is a 5-layer integrated pipeline that no single existing paper combines:

```
YOLOv8 Detection
   → ByteTrack Worker Tracking
      → Body-Region PPE–Person Association
         → Zone-Aware Compliance Engine (5-frame temporal confirmation)
            → RFID Identity + Attendance Gating
               → IoT Physical Enforcement (ESP32/MQTT/Gate)
                  → Offline-First SQLite + AWS Dashboard
```

No reviewed paper combines all of: NPU-accelerated edge inference + temporal PPE association + compliance-gated attendance + RFID identity + IoT physical actuation.

---

## 2. Phase 1 Results — YOLOv8n Baseline (88 epochs) *(superseded — see banner)*

> The training ran on **Google Colab T4 GPU**. The `results.csv` shows 88 epochs completed before the notebook was stopped (likely manually — patience=20 means auto-stop would have triggered if plateau was sustained).

### Best Metrics Snapshot (last recorded epoch = 88)

| Metric | Epoch 88 Value | Trend |
|---|---|---|
| **mAP50** | 0.5554 | ↑ Still improving |
| **mAP50-95** | 0.2653 | ↑ Improving |
| **Precision** | 0.5707 | ~Stable (fluctuating 0.5–0.7) |
| **Recall** | 0.5498 | ↑ Improving |
| **Val Box Loss** | 1.8326 | ↓ Slowly declining |
| **Val Cls Loss** | 1.0994 | ↓ Declining (from 3.26 → 1.10) |

### Peak mAP50 Observed (across all 88 epochs)

| Epoch | Best mAP50 | Notes |
|---|---|---|
| 59 | **0.5757** | Highest recorded mAP50 |
| 68 | 0.5706 | Second peak |
| 78 | 0.5736 | Third peak |
| 84 | 0.5673 | Post-peak stabilizing |

> **Key observation:** mAP50 ~0.555–0.575 at epoch 88, still slowly improving. The model was stopped prematurely. Full 150-epoch run is recommended to push toward the **≥0.70 gate** required by the plan.

### Learning Curve Interpretation

- Classification loss dropped dramatically: `3.26 → 1.10` (66% reduction) ✅
- Box loss declining steadily: `1.89 → 1.83` (small but consistent) ⚠️ slow
- **Precision is volatile** (0.46–0.77 range) → expected at this stage; class imbalance effect
- **mAP50 has NOT plateaued** — training should be resumed/rerun to completion

### Inference on Val Images (Observed)
```
Image 1: 1 helmet, 2 gloves, 1 vest, 2 boots, 1 Person  ← Full PPE scenario
Image 2: 1 none, 1 Person, 1 no_helmet                   ← Missing helmet detected ✅
Image 3: 2 nones, 1 Person, 1 no_gloves                  ← Missing gloves detected ✅
```
**Speed:** 4.0ms inference on T4 (≈250 FPS throughput)

---

## 3. Literature Review — Papers You Need to Beat

### Category 1: Pure YOLO PPE Detection (What Everyone Has Done)

| Paper / Source | Model | Dataset | mAP50 | Notes |
|---|---|---|---|---|
| CHV benchmark (various, 2023-24) | YOLOv8x | CHV (helmets + vests) | **0.929** | Large model, limited classes (2) |
| Tandfonline 2024 | YOLOv8l | Construction (multi-class) | ~0.90–0.95 | No edge deployment |
| IJARCCE 2024 | YOLOv8n | Roboflow PPE | ~0.90+ | No IoT integration |
| Edge on Jetson Xavier NX (KAIST 2025) | YOLOv8 + Coord. Attention | Custom | **0.9252** | Edge only, no attendance/IoT |
| EC-3.org 2024 | YOLOv8 variants | Construction | 90-96% range | Academic, no deployment |

> **Gap these papers leave:** None include PPE→Person spatial association, compliance engine, attendance system, IoT actuation, or edge NPU deployment in one system.

### Category 2: PPE + Tracking (Partial Systems)

| Paper / Source | What It Does | What's Missing |
|---|---|---|
| ByteTrack integration papers (2024-25) | YOLO + ByteTrack for worker ID | No PPE compliance engine |
| Roboflow pipeline guides | Detection + tracking | No IoT, no attendance |
| AAU.dk 2024 | Edge AI + IoT monitoring | No attendance, no RFID |
| IJCERT 2024 | PPE + tracking on Raspberry Pi | No NPU, no compliance gate |

### Category 3: PPE + Attendance (Adjacent)

| Paper / Source | What It Does | What's Missing |
|---|---|---|
| IJIRT 2024 (Node-RED/Firebase) | PPE + facial attendance | No NPU edge, no temporal compliance |
| IJCAONLINE | Face recognition + PPE | No ByteTrack, no zone policy |
| IJRASET 2026 | YOLO11 + Raspberry Pi | No compliance engine, no RFID |

### Category 4: Edge AI on Hailo / NPU (Very Thin Literature)

| Paper | What It Does |
|---|---|
| ResearchGate 2025–26 (RPi5 + Hailo-8L) | Shows >100 FPS on Hailo — but no full PPE pipeline |
| MDPI 2026 | Vision Transformers on RPi — PPE only, no IoT enforcement |

---

## 4. Competitive Positioning — Your Differentiators

### What Existing Papers Achieve
- Detection mAP50: 0.85–0.96 (on 2-5 class limited datasets)
- Edge inference: ~12-40 FPS on Raspberry Pi CPU
- No paper combines: NPU + temporal compliance + RFID + attendance + IoT gate

### What Your System Must Achieve to Outcompete

| Dimension | State-of-Art Benchmark | Your Target | How to Beat It |
|---|---|---|---|
| **mAP50** | 0.929 (YOLOv8x, 2 class) | **≥ 0.80** on 11-class dataset | Harder task (11 classes) — your mAP is directly comparable only on multi-class datasets |
| **Edge FPS** | ~12-15 FPS (RPi CPU) | **≥ 15 FPS** (Hailo NPU, INT8) | Hailo-8L NPU vs CPU-only — this alone is publishable |
| **System completeness** | Detection only | **Full 5-layer pipeline** | No paper has your full stack |
| **Attendance accuracy** | Not measured in any paper | **≥ 95% end-to-end accuracy** | Novel metric not in literature |
| **Alert latency** | Not measured | **< 2s detection→IoT action** | Novel KPI |
| **Offline resilience** | Not addressed | **Offline-first SQLite queue** | Novel contribution |

### Your 3 Unique Research Contributions (for the paper)

1. **PPE-Gated Attendance System:** First published system where compliance controls attendance — worker cannot check in without full PPE. All existing systems decouple detection from attendance.
2. **Temporal Body-Region PPE→Person Association at Edge:** The `associate_ppe()` + `WorkerState` 5-frame confirmation running on Hailo-8L NPU — no published paper implements this on such cost-effective hardware (~₹15,000).
3. **End-to-End Valid Entry Accuracy (EVEA) Metric:** A new composite metric combining: correct person ID + correct PPE status + correct compliance decision + correct attendance record — no existing paper defines or measures this.

---

## 5. What Your Phase 1 Results Tell You

### Status: superseded — both baselines completed

The intermediate issues listed below were resolved when training was resumed and the YOLOv8s experiment was completed. Final aggregate mAP50 did not reach the 0.70 gate (YOLOv8n 0.5568, YOLOv8s 0.5714); the root cause is violation-class data scarcity, and the architectural fix is the spatial compliance engine rather than `no_X` classification. See `Project_Status_Master_Report.md` and `docs/phase3_review_summary.md`.

### Original intermediate assessment (88-epoch run)

| Issue | Detail | Action |
|---|---|---|
| **Training stopped at 88/150 epochs** | mAP50 ~0.575, still improving | Re-run to full 150 epochs or until patience triggers |
| **mAP50 below 0.70 gate** | Currently 0.555–0.576 | Continue training; target is achievable by epoch 120-130 based on trend |
| **Precision volatile** | 0.46–0.77 across epochs | Normal for multi-class with imbalance; stabilizes later |
| **No confusion matrix saved** | Plots not found in notebook | Run full training → will auto-generate |
| **Val artifacts missing** | `results.png` not found | Expected — run was in Colab, artifacts weren't persisted locally |

### Projection: If Training Completes

Based on the loss curve trend (cls loss: 3.26 → 1.10, still dropping), the full 150-epoch run should achieve:
- **mAP50: 0.65–0.75** (estimated)
- **Precision: 0.68–0.75** (stabilized)
- **Recall: 0.60–0.68**

This would **meet the ≥0.70 Phase 2 gate** to proceed to Phase 3.

---

## 6. CI/CD Pipeline Analysis

### Current Pipeline: `.github/workflows/pr-check.yml`

**What it does:**
- Triggers on PR and push to `main`/`master`
- Python matrix: 3.10, 3.11, 3.12
- Installs only lightweight lint/test tools (not the full GPU ML stack)
- Linting: `black`, `isort`, `flake8` across `ppe_monitor/`, `tests/`, and `scripts/`
- Validates workflow YAML files
- Testing: `pytest tests/ -v`

**What it's still missing for a research project:**
- No ML-specific checks (model file validation, metric thresholds)
- No notebook linting (`nbqa`)

---

## 7. GitHub CI/CD Setup — Step-by-Step

### Prerequisites Checklist

1. **Git installed** locally → `git --version`
2. **GitHub account** with a repository created (public or private)
3. **Your project directory** is already at `c:\Users\Prakshil\OneDrive\Desktop\PPE\Ppe_detection`

---

### Step 1: Initialize Git (if not already done)

```powershell
cd "c:\Users\Prakshil\OneDrive\Desktop\PPE\Ppe_detection"
git init
git branch -M main
```

---

### Step 2: Create `.gitignore`

Create this file at the project root to exclude large model files and datasets:

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
.env
venv/
.venv/

# Jupyter
.ipynb_checkpoints/
*.ipynb         # exclude notebooks from CI if desired (or keep them)

# Model weights (too large for Git)
*.pt
*.hef
*.onnx
*.ncnn

# Dataset directories (use Roboflow links instead)
datasets/
runs/

# Logs and artifacts
*.log
*.csv

# IDE
.vscode/
.idea/
```

> **Note:** Keep `notebooks/results.csv` tracked (small file, important for reproducibility). Only exclude `runs/` (large training artifacts).

---

### Step 3: Create `requirements.txt`

```text
ultralytics>=8.4.0
opencv-python>=4.8.0
paho-mqtt>=2.0.0
fastapi>=0.110.0
uvicorn>=0.29.0
sqlalchemy>=2.0.0
pydantic>=2.0.0
matplotlib>=3.7.0
pandas>=2.0.0
seaborn>=0.13.0
pytest>=8.0.0
black>=24.0.0
isort>=5.13.0
flake8>=7.0.0
PyYAML>=6.0.1
```

---

### Step 4: Update the CI Workflow

**Edit** [`.github/workflows/pr-check.yml`](file:///c:/Users/Prakshil/OneDrive/Desktop/PPE/Ppe_detection/.github/workflows/pr-check.yml) — uncomment requirements install and pytest:

```yaml
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
```

And uncomment pytest:
```yaml
    - name: Test with pytest
      run: |
        pytest tests/ -v --tb=short
```

---

### Step 5: Create Initial Commit

```powershell
git add .gitignore requirements.txt .github/ notebooks/results.csv ppe_monitor/ PPE_Master_Implementation_Plan_K.md
git commit -m "feat: initial project structure — Phase 1 baseline + CI/CD pipeline"
```

---

### Step 6: Connect to GitHub

```powershell
# Replace with your actual GitHub username and repo name
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
git push -u origin main
```

---

### Step 7: Verify the Pipeline Runs

1. Go to your repository on GitHub
2. Click **Actions** tab
3. You should see **PR Check** workflow triggered by the push
4. Verify all 3 Python version jobs pass (green checkmarks)

---

### Step 8: Configure Branch Protection (Research Best Practice)

1. GitHub → Settings → Branches
2. Add rule for `main`:
   - ✅ **Require status checks to pass** → select `build-and-test`
   - ✅ **Require branches to be up to date**
   - ✅ **Require pull request reviews before merging** (optional for solo research)
3. Click **Save changes**

This means: no code can go to `main` unless CI passes — important for reproducible research.

---

### Step 9: Add GitHub Secrets for Future Phases

For cloud features (AWS, Roboflow API), add secrets:

1. GitHub → Settings → Secrets and variables → Actions
2. Add:
   - `ROBOFLOW_API_KEY` — for dataset downloads in CI
   - `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — for cloud sync tests

---

### Step 10: Add a README Badge

Add to your `README.md`:

```markdown
![CI](https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>/actions/workflows/pr-check.yml/badge.svg)
```

---

## 8. Recommended Immediate Next Actions

### Priority 1 — Complete Phase 1 Training

```python
# Re-run in Colab with full 150 epochs
# Make sure to DOWNLOAD the results BEFORE closing:
import shutil
shutil.copy("runs/ppe/yolov8n_baseline/results.csv", "notebooks/results.csv")
shutil.copy("runs/ppe/yolov8n_baseline/best.pt", "notebooks/yolov8n_best.pt")
```

### Priority 2 — Run YOLOv8s Baseline (Experiment 2)

Train `yolov8s.pt` with identical config — gives you the accuracy vs. speed tradeoff table for the paper.

### Priority 3 — Enable `requirements.txt` in CI

Activate the commented-out line in `pr-check.yml` so CI actually installs your packages.

### Priority 4 — Write `tests/` directory

Even basic unit tests for the compliance engine (`evaluate_compliance()`, `associate_ppe()`) will:
- Validate your logic is correct
- Make the paper's claims reproducible
- Enable CI to catch regressions in Phase 5–8

### Priority 5 — Document per-class mAP

When training completes, extract per-class AP from the model:
```python
results = model.val(data="construction-ppe.yaml")
print(results.box.ap_class_index)  # per-class AP values
```
This is **essential for the paper** — `goggles` and `boots` will likely be the weakest classes.

---

## 9. Research Evaluation Table (Fill As You Complete Phases)

| Model | Dataset | mAP50 | Precision | Recall | mAP50-95 | FPS (Device) | Status |
|---|---|---|---|---|---|---|---|
| YOLOv8n | Construction-PPE | 0.5568 | 0.7313 | 0.5157 | 0.2573 | 98.6 (T4) | ✅ Complete (55 ep, best 44) |
| YOLOv8s | Construction-PPE | 0.5714 | 0.6738 | 0.5390 | 0.2650 | ~50 (T4, est.) | ✅ Complete (93 ep, best 45); Phase 3 validated |
| YOLOv8s | Merged + Site | TBD | TBD | TBD | TBD | TBD | ⬜ Phase 4 |
| YOLOv8n INT8 | Merged + Site | TBD | TBD | TBD | TBD | **Target: 15+ FPS** | RPi 5 + Hailo-8L | ⬜ Phase 9 |

Final values from `notebooks/results/phase1_metrics.json` (aggregate mAP50; 0.70 gate not met).

---

*Generated: September 9, 2026*

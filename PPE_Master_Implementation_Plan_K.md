# Industrial PPE Detection System
## Central Implementation Plan — Master Reference

> **YOLOv8 · Computer Vision · IoT · Edge AI · Raspberry Pi 5 + Hailo**
>
> *Synthesised from: Deep Research Report · PPE Dataset Analysis · Implementation Plans v1 & v2*
> *September 2026*

---

## Table of Contents

1. [Project Overview & Final Title](#1-project-overview--final-title)
2. [System Architecture](#2-system-architecture)
3. [Hardware Decisions](#3-hardware-decisions)
4. [Final Technology Stack](#4-final-technology-stack)
5. [Dataset Strategy](#5-dataset-strategy)
6. [Training Configuration](#6-training-configuration)
7. [Implementation Phases](#7-implementation-phases-12-phases)
8. [Formal Test Matrix](#8-formal-test-matrix)
9. [Research Evaluation Table](#9-research-evaluation-table)
10. [Recommended Module Structure](#10-recommended-module-structure)
11. [Deferred Features](#11-deferred-features-not-now--after-v1-works)
12. [Master Timeline](#12-master-timeline)
13. [Final Demonstration Scenarios](#13-final-demonstration-scenarios)

---

## 1. Project Overview & Final Title

| Field | Detail |
|---|---|
| **Official Title** | Real-Time Industrial PPE Detection System: YOLOv8 Deep Learning with IoT-Based Safety Monitoring |
| **Domains** | Computer Vision · Deep Learning · Internet of Things · Edge AI |
| **Core Output** | End-to-end worker safety compliance system with attendance, alerting, and IoT enforcement |
| **Timeline** | 12 Phases · ~16–20 weeks total |
| **Training Cost** | $0 (Google Colab free GPU) |
| **Hardware Budget** | ~₹12,000–15,000 (Raspberry Pi 5 + Hailo AI HAT+ + Camera) |

### What This System Does

This is **NOT** simply "train YOLO and deploy on Raspberry Pi." The system is a pipeline of five integrated layers:

- **Computer Vision** — YOLOv8 detects PPE items and persons in real time
- **Worker–PPE Association** — ByteTrack assigns each detected PPE to the correct person
- **Safety Decision Engine** — Compliance policy checks required gear vs. detected gear
- **Attendance & Identity** — RFID + face recognition records entry/exit events
- **IoT + Database** — ESP32/MQTT drives physical alerts; SQLite/AWS stores all events

---

## 2. System Architecture

### End-to-End Pipeline

```
Entry Camera (RTSP/USB)
       │
       ▼
 Frame Capture
       │
       ▼
 YOLOv8 PPE Detector ──── Person + PPE detections
       │
       ▼
 ByteTrack / BoT-SORT ─── Assign Track IDs
       │
       ▼
 PPE→Person Association ─ Who is wearing what?
       │
       ▼
 Compliance Engine ─────── Required PPE vs Detected PPE
       │
    ┌──┴──┐
    │     │
 COMPLIANT  NON-COMPLIANT
    │            │
    ▼            ▼
 Identity     Alert (Buzzer / Light / Notification)
 Check             │
    │          IoT Action (ESP32 / MQTT / Relay)
    ▼
 Attendance Check-In
    │
    ▼
 SQLite + AWS Dashboard
```

### Critical Design Rules *(Non-Negotiable)*

- **Safety decision is LOCAL** — gate logic never depends on cloud/internet availability
- **Never act on a single frame** — require N consecutive frames before triggering a violation
- **Detection ≠ Compliance** — always pass detections through the policy engine
- **Separate debugging concerns** — prove AI first, then app logic, then hardware
- **Attendance is gated on PPE compliance** — a worker without full PPE cannot be checked in

---

## 3. Hardware Decisions

### Primary Edge Platform: Raspberry Pi 5 + Hailo AI HAT+

| Component | Spec / Choice | Reason |
|---|---|---|
| **Raspberry Pi 5** | 8 GB RAM recommended | Handles app logic, DB, MQTT, network, dashboard alongside inference |
| **Hailo AI HAT+** | Hailo-8 (26 TOPS) preferred; Hailo-8L (13 TOPS) minimum | Dedicated NPU for YOLOv8 inference; Pi CPU stays free for app logic |
| **Camera** | USB cam (lab) / RTSP IP cam (production) | Match existing CCTV infrastructure |
| **Storage** | 128 GB+ high-endurance microSD or SSD | Local SQLite DB + snapshot storage |
| **IoT MCU** | ESP32 | Controls relay, buzzer, LEDs, gate mechanism via MQTT |
| **OS** | Raspberry Pi OS 64-bit *(mandatory for Hailo)* | Required by Hailo runtime and Ultralytics RPi AI workflow |
| **Cooling** | Active cooling (official Pi fan or heatsink) | NPU + AI workload generates sustained heat |

### Fallback Path *(Keep Both)*

- **Path A — Fallback:** `YOLOv8n → NCNN → Raspberry Pi 5 CPU only`. Slower (~3–5 FPS) but always works.
- **Path B — Target:** `YOLOv8n → model.export(format="hailo") → HEF → Hailo NPU`. Full performance.

> **Correction:** Ultralytics now ships a native `format="hailo"` exporter that does the ONNX export, node selection, INT8 calibration, and DFC compilation in one call — you no longer hand-write Hailo parser/optimizer CLI commands or hunt for ONNX node names in Netron (see Phase 9 for the corrected steps). This removes most of the risk this section originally warned about. Still keep the NCNN fallback path — it's a different failure mode (Hailo hardware/driver issues, not export-tooling issues) and costs nothing to keep ready.

### Jetson Orin Nano — NOT Primary

NVIDIA repriced Jetson Orin Nano Super to ~$400+ (was $249) in July 2026. It is no longer cost-effective as a primary platform. Use only as a benchmark reference or fallback if Hailo compilation becomes a blocker.

### Budget Estimate

| Item | Cost |
|---|---|
| Raspberry Pi 5 (8 GB) | ~₹7,000–8,000 |
| Hailo AI HAT+ (Hailo-8L 13 TOPS) | ~₹5,000–6,000 *(varies by import)* |
| Camera (USB webcam for lab) | ~₹800–1,500 |
| Storage + PSU + Case + Cooling | ~₹1,500–2,000 |
| ESP32 + Relay + LEDs + Buzzer | ~₹400–600 |
| **Total Estimate** | **~₹14,000–18,000** |

---

## 4. Final Technology Stack

| Layer | Choice | Notes |
|---|---|---|
| **Model** | YOLOv8n (primary) + YOLOv8s (benchmark) | `yolov8n.pt` for edge deployment; `yolov8s.pt` for accuracy comparison |
| **Training Framework** | PyTorch + Ultralytics API | Standard Ultralytics train/val/export workflow |
| **Training Platform** | Google Colab (free T4 GPU) | $0 cost; must set Runtime → GPU or it silently falls back to CPU |
| **Annotation** | Roboflow (primary) / LabelImg (offline) | For custom site image labelling in Phase 4 |
| **Tracking** | ByteTrack (start) → BoT-SORT (if crowded) | Both supported natively in Ultralytics `.track()` API |
| **Computer Vision** | OpenCV | Frame capture, preprocessing, annotation overlay |
| **Edge Export** | `model.export(format="hailo")` (or NCNN for CPU fallback) | Ultralytics owns the whole pipeline internally (`.pt → ONNX → parse → INT8 calibrate → compile`) as a single call — no manual node-name hunting. Requires the Hailo DFC wheel installed alongside `ultralytics`, and **must run on Linux x86_64** (not the Pi, not Windows/macOS) — use a Linux workstation/VM, or Ultralytics Platform's managed export if you don't have one. |
| **IoT Protocol** | MQTT (Mosquitto broker) | Lightweight, device-oriented; Pi publishes events, ESP32 subscribes |
| **Local Database** | SQLite | No server dependency; reliable offline; easy to migrate later |
| **Backend API** | FastAPI (Python) | REST API for dashboard; async; lightweight |
| **Frontend** | React (or simple HTML/JS for MVP) | Live camera feed, compliance stats, attendance logs |
| **Cloud (Optional)** | AWS IoT Core + MQTT | Receives sanitised event JSON; never raw video; offline-first queue |
| **Identity (MVP)** | RFID reader on ESP32 | Simpler than face recognition for v1; add face later |
| **Identity (v2)** | FaceNet / DeepFace embeddings | Add only after core PPE pipeline is validated |

---

## 5. Dataset Strategy

### Selected Datasets

| Dataset | Images | Classes | Use | When |
|---|---|---|---|---|
| Ultralytics Construction-PPE | 1,416 | 11 | Baseline training | Phase 2 — start here, **do not merge yet** |
| Roboflow PPE Detection (Projelerim) | 6,550 | 14 | Extended coverage | Phase 4 — merge after baseline validated |
| Your Own Site Images | 100–300 | Custom | Site fine-tuning | Phase 4 — most impactful single step |

### Unified Class List *(After Merge)*

After merging both datasets, normalize to these 16 classes:

| ID | Class Name | Source | Notes |
|---|---|---|---|
| 0 | `helmet` | Both / Ultralytics | |
| 1 | `no_helmet` | Both / Ultralytics | Critical — flag as compliance violation |
| 2 | `vest` | Both / Ultralytics | ⚠ NO `no_vest` class in either dataset — handle via inference rule |
| 3 | `gloves` | Both / Ultralytics | |
| 4 | `no_gloves` | Both / Ultralytics | Ultralytics provides this; Roboflow does not |
| 5 | `boots` | Both / Ultralytics | |
| 6 | `no_boots` | Both / Ultralytics | Ultralytics provides this; Roboflow does not |
| 7 | `goggles` | Both / Ultralytics | |
| 8 | `no_goggle` | Both / Ultralytics | Ultralytics provides this |
| 9 | `mask` | Both / Ultralytics | New from Roboflow |
| 10 | `safety_suit` | Roboflow | New from Roboflow (coverall/wearpack) |
| 11 | `face_shield` | Roboflow | New from Roboflow |
| 12 | `earmuff` | Roboflow | New from Roboflow |
| 13 | `no_earmuff` | Roboflow | New from Roboflow |
| 14 | `harness` | Roboflow | New from Roboflow |
| 15 | `person` | Both / Ultralytics | |

### The `no_vest` Problem & Fix

Neither dataset has a dedicated `no_vest` class. Two approaches:

- **Approach A — Inference Rule *(Phase 2)*:** Person detected + vest not detected in the assigned bounding region for 5+ consecutive frames = non-compliant. Fast to implement; sufficient for MVP.
- **Approach B — Data Collection *(Phase 4)*:** Collect and label `no_vest` images from your site. Add to training data. Train `ppe-v2`. This is the better long-term fix.

> Start with Approach A. Migrate to Approach B in Phase 4.

### Real-Site Data Collection Checklist

Collect 100–300 site-specific images covering all of the following:

#### Lighting Conditions
- Morning light (strong directional shadows)
- Afternoon light (flat, bright)
- Evening / low light
- Artificial indoor factory lighting
- Mixed light (indoor near doorway)

#### Camera Angles
- Direct frontal (gate camera angle)
- 30° off-centre
- Side-on
- Slightly elevated (CCTV mounting height)
- Entry-gate perspective with person walking toward camera

#### Worker Conditions
- Single worker, full PPE, close range
- Single worker, full PPE, far range
- Multiple workers (2–5) in frame
- Partial occlusion (worker behind barrier/machinery)
- Overlapping workers
- Different body sizes and postures

#### PPE Conditions
- All PPE correctly worn
- Helmet missing *(most important)*
- Vest missing
- Gloves missing
- PPE partially worn or incorrectly worn
- Non-standard or coloured PPE variants

#### Hard Negatives *(Critical for Reducing False Positives)*
- Baseball caps, hoodies, beanies — must **NOT** be classified as helmets
- Sunglasses — must **NOT** be classified as safety goggles
- Non-PPE reflective jackets or bags
- Construction debris or equipment that could look like PPE

---

## 6. Training Configuration

### Three Training Experiments

Run three experiments, not a single blind training run. Each builds on the previous.

#### Experiment 1 — YOLOv8n Baseline *(Speed Benchmark)*

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # pretrained COCO weights — transfer learning

results = model.train(
    data         = "ppe.yaml",
    epochs       = 150,
    imgsz        = 640,
    batch        = 16,
    patience     = 20,       # early stop if no improvement for 20 epochs
    close_mosaic = 15,       # disable mosaic aug last 15 epochs —
                             # improves real-world single-frame accuracy measurably
    workers      = 8,
    seed         = 42,
    project      = "runs/ppe",
    name         = "yolov8n_baseline",
)
```

#### Experiment 2 — YOLOv8s Baseline *(Accuracy Benchmark)*

```python
model = YOLO("yolov8s.pt")

results = model.train(
    data         = "ppe.yaml",
    epochs       = 150,
    imgsz        = 640,
    batch        = 16,
    patience     = 20,
    close_mosaic = 15,
    workers      = 8,
    seed         = 42,
    project      = "runs/ppe",
    name         = "yolov8s_baseline",
)
```

#### Experiment 3 — Site-Fine-Tuned YOLOv8s *(Deployment Candidate)*

Take `best.pt` from Experiment 2. Add site-specific images (Phase 4). Fine-tune:

```python
model = YOLO("runs/ppe/yolov8s_baseline/weights/best.pt")  # resume from best

results = model.train(
    data         = "ppe_sitedata.yaml",  # merged dataset + site images
    epochs       = 50,                   # fine-tune, not full retrain
    imgsz        = 640,
    batch        = 8,
    patience     = 15,
    freeze       = 10,                   # freeze first 10 backbone layers
    lr0          = 0.001,                # lower LR for fine-tuning
    seed         = 42,
    project      = "runs/ppe",
    name         = "yolov8s_site_v2",
)
```

### Why These Choices

- **YOLOv8n for deployment** — keeps latency low on Pi, and this is backed by data, not just convention: Ultralytics' published Hailo-8/8L INT8 quantization benchmarks show YOLOv8n retains ~100% of its PyTorch mAP50 after compilation, vs. ~96% for YOLO11n and ~93% for YOLO26n — the newer models' attention blocks are more INT8-sensitive on Hailo's current compiler (DFC 3.33). **YOLOv8n is the technically correct pick for this specific hardware pairing, not a legacy fallback.**
- **YOLOv8s as benchmark** — shows accuracy headroom; use on Hailo-8 (26 TOPS) if FPS allows
- **`close_mosaic=15`** — mosaic augmentation composites multiple images; turning it off for the final epochs forces the model to predict in realistic single-object framing as it will see at inference
- **`patience=20`** — prevents overfitting; auto-stops training when validation mAP plateaus
- **`seed=42`** — reproducible splits for comparison between experiments
- **Transfer learning from pretrained COCO weights** — not from scratch; dramatically reduces training time and improves small-object detection

### Confidence Thresholds *(Per Class — Tune on Validation Set)*

| Class | Starting Threshold | Reason |
|---|---|---|
| `person` | 0.50 | Standard confidence; rarely ambiguous |
| `helmet` | 0.50 | Critical class; medium threshold balances FP/FN |
| `no_helmet` | 0.50 | Critical violation class |
| `vest` | 0.45 | Large object, slightly lower threshold acceptable |
| `gloves` | 0.40 | Small object; lower threshold to improve recall |
| `no_gloves` | 0.40 | Small object violation |
| `boots` | 0.40 | Partially occluded by ground; lower threshold |
| `no_boots` | 0.40 | Same as boots |
| `goggles` | 0.35 | Smallest PPE; most frequently missed; needs lower threshold |
| `no_goggle` | 0.35 | Same as goggles |

> ⚠ These are starting values. Tune each class threshold using your validation set before deployment.

> **Correction — this table can't be deployed as-is on Hailo.** YOLOv8/YOLO11 detection HEFs bake a single global `conf`/`iou` pair into the on-chip HailoRT NMS **at export time** (Ultralytics defaults: `conf=0.25`, `iou=0.7`) — there's no per-class threshold inside the compiled pipeline. To actually get the per-class behavior above: export with a low global `conf` (e.g. `conf=0.20`, via `model.export(format="hailo", name="hailo8", conf=0.20, ...)`) so low-confidence classes like goggles aren't dropped by the chip, then apply the stricter per-class values from this table as a **host-side filter** on the returned detections, before they reach `associate_ppe()` / the compliance engine.

### Augmentation Strategy

- **Rotation: ±15°** — simulates non-standard camera angles
- **Brightness/contrast jitter** — handles variable factory lighting
- **Horizontal flip** — doubles effective dataset
- **Mosaic augmentation** (default ON; disabled for last 15 epochs via `close_mosaic`)
- **Partial occlusion patches** — simulate workers behind equipment
- Do **NOT** apply extreme augmentation that never occurs in real footage

### Metrics to Track — Beyond mAP

| Category | Metrics |
|---|---|
| **Model** | Precision, Recall, mAP50, mAP50-95, F1, per-class AP *(check goggles/boots specifically)* |
| **Deployment** | FPS, inference latency (ms), CPU %, NPU utilization, RAM, temperature, power draw |
| **Application** | PPE false-positive rate, PPE false-negative rate, violations/hour accuracy |
| **Attendance** | Check-in accuracy, duplicate check-in rate, missed checkout rate, alert latency |

---

## 7. Implementation Phases (12 Phases)

> Follow this exact sequence. Do not skip or reorder phases. Each phase is a **gate** — validate before proceeding.

---

### Phase 0 — Environment Setup *(Days 1–3)*

- Create Google account for Colab access
- Install Python 3.10 or 3.11 locally
- Create free Roboflow account
- Set up Git repository for the project
- `pip install ultralytics`
- `pip install opencv-python`
- `pip install paho-mqtt`
- Set up VS Code with Python extension

> ⚠ When opening Colab: **Runtime → Change runtime type → GPU (T4)**. This is easy to miss and silently falls back to CPU, making training 10–20× slower.

---

### Phase 1 — Dataset Preparation *(Days 4–8)*

#### Step 1.1: Download Ultralytics Construction-PPE

```python
# Auto-downloads when data= is provided in train()
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
# dataset auto-downloads on first training call
```

#### Step 1.2: Inspect the Dataset

- Open 20–30 random images and verify bounding boxes visually
- Check class distribution — goggles/boots typically underrepresented
- If any class has under ~500 instances: oversample those images with different augmentation in the train split
- Remove corrupt, blurred, or mislabelled images

#### Step 1.3: Validate the YAML

```yaml
# ppe.yaml (Ultralytics Construction-PPE)
path: construction-ppe
train: images/train   # 1,132 images
val:   images/val     # 143 images
test:  images/test    # 141 images

names:
  0: helmet
  1: gloves
  2: vest
  3: boots
  4: goggles
  5: none
  6: Person
  7: no_helmet
  8: no_goggle
  9: no_gloves
  10: no_boots
```

#### Step 1.4: Do NOT Merge Datasets Yet

Train on the Ultralytics dataset alone first. Merging before baseline validation is the **#1 source of debugging confusion** in multi-dataset PPE projects.

---

### Phase 2 — YOLOv8 Baseline Training *(Days 9–14)*

- Run Experiment 1 (YOLOv8n) using config from Section 6
- Run Experiment 2 (YOLOv8s) using config from Section 6
- Keep both `best.pt` files safe — every later step depends on them
- Record all metrics in the comparison table (see Section 9)
- **Target: mAP50 ≥ 0.70 before proceeding.** If below 0.60, debug dataset first.

> **Key checkpoint:** check per-class recall, not just aggregate mAP. A model can hit 0.75 overall while missing goggles almost every time.

---

### Phase 3 — Real-World Validation *(Days 15–19)*

- Run inference on the test split — never seen during training
- Run inference on your own phone photos of PPE-related environments
- Run inference on any available CCTV footage
- Test: does it correctly flag a missing helmet?
- Test: does it false-positive on a baseball cap or hood?

#### Build a Hard Negative Set

Create 20–30 photos of: baseball caps, hoodies, beanies, sunglasses, non-safety helmets, reflective bags. If the model fails on these, add them back as training negatives before Phase 4.

> ⚠ A good mAP score does **NOT** guarantee good real-world performance. This step is mandatory.

---

### Phase 4 — Dataset Expansion & Merge *(Days 20–27)*

#### Step 4.1: Merge Roboflow PPE Dataset

Normalize class names across both datasets to the unified 16-class list from Section 5. Remap class IDs using a script:

```python
UNIFIED_CLASSES = [
    "helmet", "no_helmet", "vest", "gloves", "no_gloves",
    "boots", "no_boots", "goggles", "no_goggle", "mask",
    "safety_suit", "face_shield", "earmuff", "no_earmuff",
    "harness", "person"
]

ultralytics_map = {0:0, 7:1, 2:2, 1:3, 9:4, 3:5, 10:6,
                   4:7, 8:8, 6:15}
roboflow_map    = { }  # populate from Roboflow data.yaml

def remap_labels(label_dir, mapping, out_dir):
    for fname in os.listdir(label_dir):
        with open(os.path.join(label_dir, fname)) as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts  = line.split()
            old_id = int(parts[0])
            if old_id in mapping:
                parts[0] = str(mapping[old_id])
                new_lines.append(" ".join(parts) + "\n")
        with open(os.path.join(out_dir, fname), "w") as f:
            f.writelines(new_lines)
```

#### Step 4.2: Remove Duplicates

- Use perceptual hashing or image-similarity comparison before merging
- Check filename overlaps between both datasets
- Same image in both train and validation silently inflates validation score — this is a **serious error**

#### Step 4.3: Add Site-Specific Images

- Collect 100–300 site images per the checklist in Section 5
- Annotate using Roboflow free tier or LabelImg
- Include hard negatives (caps, sunglasses, etc.)
- Include `no_vest` examples (Approach B from Section 5)

#### Step 4.4: Retrain — Experiment 3

Use config from Section 6 Experiment 3. This becomes `ppe-v2.pt` — your deployment candidate.

---

### Phase 5 — Tracking + PPE–Person Association *(Days 28–33)*

#### Step 5.1: Enable ByteTrack

```python
from ultralytics import YOLO

model   = YOLO("best.pt")
results = model.track(
    source  = "video.mp4",
    tracker = "bytetrack.yaml",
    persist = True,
    conf    = 0.4,
    iou     = 0.5,
)
```

#### Step 5.2: ByteTrack Starting Config

```yaml
# bytetrack.yaml
track_high_thresh: 0.5
track_low_thresh:  0.1
new_track_thresh:  0.6
track_buffer:      30    # frames a lost track stays alive
match_thresh:      0.8
```

> These are starting values. Tune based on your validation video. If identity switches are frequent in crowded scenes, switch to BoT-SORT.

#### Step 5.3: PPE–Person Association Logic

For each tracked person bounding box, divide into body regions and check which PPE falls inside:

```python
def associate_ppe(person_bbox, ppe_detections):
    """
    person_bbox = (x1, y1, x2, y2)
    Returns dict of {ppe_class: True/False}
    """
    px1, py1, px2, py2 = person_bbox
    ph = py2 - py1  # person height

    regions = {
        "head":  (px1, py1,            px2, py1 + ph * 0.20),
        "face":  (px1, py1 + ph * 0.05, px2, py1 + ph * 0.30),
        "torso": (px1, py1 + ph * 0.20, px2, py1 + ph * 0.70),
        "hands": (px1, py1 + ph * 0.50, px2, py1 + ph * 0.80),
        "feet":  (px1, py1 + ph * 0.80, px2, py2),
    }

    ppe_map = {
        "helmet":  "head",
        "goggles": "face",
        "mask":    "face",
        "vest":    "torso",
        "gloves":  "hands",
        "boots":   "feet",
        "harness": "torso",
    }

    status = {k: False for k in ppe_map}
    for det in ppe_detections:
        cls, cx, cy = det["class"], det["cx"], det["cy"]
        if cls in ppe_map:
            region = regions[ppe_map[cls]]
            if region[0] <= cx <= region[2] and region[1] <= cy <= region[3]:
                status[cls] = True

    return status
```

---

### Phase 6 — Compliance Engine *(Days 34–38)*

#### Zone-Based PPE Policy Config

```yaml
# zones.yaml
zones:
  main_entry:
    required_ppe: [helmet, vest, gloves, boots]
  welding_zone:
    required_ppe: [helmet, face_shield, gloves]
  chemical_zone:
    required_ppe: [goggles, gloves, mask, safety_suit]
  general_floor:
    required_ppe: [helmet, vest]
```

#### Compliance Evaluation Function

```python
def evaluate_compliance(worker_ppe_status, zone="main_entry"):
    required = load_policy(zone)["required_ppe"]
    missing  = [item for item in required
                if not worker_ppe_status.get(item, False)]

    return {
        "compliant": len(missing) == 0,
        "missing":   missing,
    }
```

> **Bug fix:** the original version had an extra "no_vest rule" block here (`if "vest" in required and not ... : if "vest" not in missing: missing.append("vest")`). It was dead code — `missing` is already built from every required item that isn't detected, so "vest" is already in there whenever that block's condition is true; the `if "vest" not in missing` guard then always blocks the `.append()`. It ran, did nothing, and made the function look like it had special no-vest handling when it didn't. There's nothing special to add: since there's no trained `no_vest` class, "vest missing" is just "vest is required and not in `worker_ppe_status`" — the same path every other required item takes. The actual robustness against single-frame flicker comes entirely from the `WorkerState` temporal confirmation below, which already applies to every class including vest.

#### Temporal Confirmation — Never Act on One Frame

```python
class WorkerState:
    def __init__(self):
        self.violation_frames   = {}
        self.CONFIRM_THRESHOLD  = 5   # frames before triggering
        self.COOLDOWN_SECONDS   = 10  # seconds between repeat alerts

    def update(self, track_id, compliance_result, timestamp):
        if not compliance_result["compliant"]:
            key = (track_id, tuple(sorted(compliance_result["missing"])))
            self.violation_frames[key] = self.violation_frames.get(key, 0) + 1

            if self.violation_frames[key] == self.CONFIRM_THRESHOLD:
                self.trigger_violation(track_id, compliance_result, timestamp)
        else:
            # reset violation counter for this worker
            for key in list(self.violation_frames):
                if key[0] == track_id:
                    del self.violation_frames[key]
```

---

### Phase 7 — Attendance & Identity *(Days 39–45)*

#### MVP: RFID-Based Identity *(Start Here)*

- Worker taps RFID card at gate → ESP32 reads UID → sends to Pi via MQTT
- Pi looks up worker in database by RFID UID
- Camera verifies person present + compliance check passes
- If both conditions met: record attendance check-in

> Do **NOT** start with face recognition. RFID is simpler, faster to implement, and sufficient for the MVP.

#### Entry Line Detection

```yaml
# camera.yaml
entry_line:
  y_position: 400       # pixel row in frame
  direction: "top_to_bottom"  # person moving toward camera
```

Track when the worker centroid crosses the entry line in the correct direction. This gives a **deterministic entry event**, not just "person appeared in frame."

#### Check-In State Machine

```
IDLE → PERSON_DETECTED → TRACK_STABLE → IDENTITY_PENDING
     → PPE_CHECK → COMPLIANT → ATTENDANCE → ENTRY_RECORDED
                ↘ NON_COMPLIANT → DENY/WARN
```

#### Attendance DB Schema

```sql
-- workers table
CREATE TABLE workers (
    worker_id   TEXT PRIMARY KEY,
    name        TEXT,
    department  TEXT,
    rfid_uid    TEXT UNIQUE,
    active      INTEGER DEFAULT 1
);

-- attendance table
CREATE TABLE attendance (
    attendance_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id       TEXT,
    date            TEXT,
    check_in        TEXT,
    check_out       TEXT,
    entry_camera    TEXT,
    duration_mins   INTEGER,
    FOREIGN KEY (worker_id) REFERENCES workers(worker_id)
);

-- violations table
CREATE TABLE violations (
    violation_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id       TEXT,
    timestamp       TEXT,
    camera_id       TEXT,
    zone_id         TEXT,
    missing_ppe     TEXT,       -- JSON array: ["helmet","gloves"]
    confidence      REAL,
    snapshot_path   TEXT
);
```

---

### Phase 8 — IoT Integration *(Days 46–51)*

#### MQTT Topic Hierarchy

```
ppe/
 └── site01/
      └── gate01/
           ├── events       ← violation/attendance JSON events
           ├── attendance   ← check-in/check-out records
           ├── violations   ← PPE violation records
           ├── heartbeat    ← device health every 30s
           └── status       ← camera online/offline
```

#### Event JSON Format

```json
{
  "event_type":  "ppe_violation",
  "device_id":   "gate_01",
  "camera_id":   "cam_entry_01",
  "worker_id":   "EMP1023",
  "timestamp":   "2026-09-08T08:42:15+05:30",
  "zone":        "main_entry",
  "status":      "non_compliant",
  "missing_ppe": ["gloves"],
  "confidence":  0.91
}
```

#### ESP32 → Pi MQTT Architecture

```
Raspberry Pi (Publisher)
       │
       │  MQTT publish: ppe/site01/gate01/violations
       ▼
Mosquitto Broker (on Pi)
       │
       │  MQTT subscribe
       ▼
   ESP32 (Subscriber)
       │
  ┌────┼────────────┐
  ▼    ▼            ▼
Relay  Buzzer  Warning LED
(Gate) (Alert)  (Visual)
```

#### Heartbeat — Device Health *(Every 30s)*

```json
{
  "device_id":   "gate_01",
  "cpu_pct":     34,
  "temperature": 58,
  "fps":         17,
  "camera":      "online",
  "mqtt":        "connected",
  "model":       "ppe-v2"
}
```

---

### Phase 9 — Edge Deployment on Raspberry Pi + Hailo *(Days 52–58)*

#### Step 9.1 & 9.2 (corrected): Export Directly to Hailo HEF

**These used to be two manual steps (export to ONNX, then hand-run the Hailo parser/optimizer/compiler CLI with node names you had to find yourself in Netron). That's no longer necessary** — Ultralytics now owns the entire pipeline behind one export call:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(
    format = "hailo",
    name   = "hailo8",      # match your exact board: "hailo8" (26 TOPS) or "hailo8l" (13 TOPS) —
                             # if omitted this silently defaults to hailo8l and the HEF won't load
                             # on a 26 TOPS board
    imgsz  = 640,
    data   = "ppe.yaml",    # your own dataset for in-domain calibration — do NOT skip this;
                             # the default COCO128 calibration set doesn't represent your site's
                             # lighting/PPE, and Hailo recommends 1,024+ in-domain images
    conf   = 0.20,          # see the correction under Confidence Thresholds above
)
# Output: yolov8n_hailo_model/ containing best.hef + metadata.yaml + nms_config.json
```

Requirements:
- Run this on a **Linux x86_64** machine (a dev laptop/workstation running Ubuntu, or a VM) — not the Pi itself, not Windows/macOS natively. If you don't have one, use Ultralytics Platform's managed Hailo export instead, which needs no local DFC install.
- Install the Hailo DFC wheel for **your** hardware generation first: `pip install ultralytics` then `pip install /path/to/hailo_dataflow_compiler-*.whl` (Hailo-8/8L uses DFC v3.x — download from the Hailo Developer Zone, free registration).
- Copy the **whole output folder** (not just the `.hef`) to the Pi — `metadata.yaml` travels with it so Ultralytics can pick the right post-processing path at inference time.

You no longer need to find ONNX end-node names, write a `.alls` model script, or hand-build an NMS config — the exporter does this automatically and rejects unsupported model/task combinations up front instead of producing a broken HEF.

#### Step 9.3: Raspberry Pi Setup

```bash
# Raspberry Pi OS 64-bit — mandatory for Hailo runtime
sudo apt update && sudo apt upgrade -y
sudo apt install dkms
sudo apt install hailo-all      # installs HailoRT + firmware; do NOT also apt-install
                                 # python3-hailort separately, hailo-all already covers it
sudo reboot

# after reboot, confirm the accelerator is actually detected before writing any app code:
hailortcli fw-control identify

sudo apt install -y python3-picamera2 python3-opencv
pip3 install ultralytics paho-mqtt fastapi uvicorn
```

Run inference through Ultralytics against the exported folder (not the raw `.hef` alone) so it picks the correct YOLOv8/YOLO11/YOLO26 post-processing path from `metadata.yaml`:

```python
from ultralytics import YOLO
model = YOLO("yolov8n_hailo_model")   # the exported folder, not a .hef path
results = model.predict("path/to/frame.jpg")
```

#### Step 9.4: Fallback NCNN Export *(CPU Path)*

```python
model.export(format="ncnn", imgsz=640)
# Run on Pi CPU — slower (~3–5 FPS) but guaranteed to work
```

---

### Phase 10 — Dashboard & Cloud *(Days 59–65)*

#### FastAPI Backend Routes

```
GET  /api/dashboard/summary       # workers present, violations, compliance %
GET  /api/attendance/today        # today's check-ins and check-outs
GET  /api/violations?date=&zone=  # filterable violations list
GET  /api/workers                 # worker roster
POST /api/workers                 # add new worker
GET  /api/health                  # device + model status
WS   /api/live                    # WebSocket for real-time detection feed
```

#### Dashboard Pages

- **Home:** Workers Present / Compliant Entries / PPE Violations / Currently Inside (live counters)
- **Live Feed:** camera stream with bounding boxes, PPE overlay, compliance status per tracked worker
- **Attendance:** check-in/check-out table with filter by date and shift
- **Violations:** timestamped violation log with snapshot, zone, missing PPE, confidence
- **Reports:** CSV/PDF export of daily/weekly compliance summaries
- **Device Health:** camera status, FPS, CPU %, temperature, MQTT connection

#### AWS Cloud Sync *(Offline-First)*

```python
# Offline queue pattern
def publish_event(event_dict):
    try:
        mqtt_client.publish("ppe/site01/gate01/events",
                            json.dumps(event_dict))
        db.mark_synced(event_dict["event_id"])
    except Exception:
        db.queue_for_sync(event_dict)  # store locally

# Background sync job (runs every 60s)
def sync_pending_events():
    pending = db.get_unsynced_events()
    for event in pending:
        try:
            mqtt_client.publish(...)
            db.mark_synced(event["event_id"])
        except Exception:
            pass  # retry next cycle
```

---

### Phase 11 — Full Integration *(Days 66–72)*

- Wire all modules together: `Camera → YOLO → ByteTrack → Association → Compliance → Attendance → MQTT → ESP32 → SQLite → Dashboard`
- Run the full pipeline on **saved test videos** — not live camera yet
- Execute formal test matrix from Section 8
- Fix any inter-module bugs before touching live camera
- Benchmark end-to-end latency: detection → IoT action, detection → dashboard update
- Document model version, dataset version, and metrics for this integration build

---

### Phase 12 — Pilot & Iteration *(Weeks 16–20)*

- Deploy in one real (or simulated) location for 1–2 weeks
- Manually track every false positive and false negative in a log sheet
- Track: missed check-ins, duplicate check-ins, wrong PPE association, IoT failures
- Retrain with collected mistakes → produce `ppe-v3.pt`
- Document final metrics after pilot — these go into your research paper

---

## 8. Formal Test Matrix

| Test | Scenario | Expected Result |
|---|---|---|
| T1 | Worker with all required PPE | COMPLIANT → ENTRY ALLOWED → ATTENDANCE CHECK-IN recorded |
| T2 | Worker — helmet missing | NON-COMPLIANT → ENTRY DENIED → ALERT triggered → No attendance |
| T3 | Worker — gloves missing | NON-COMPLIANT → ENTRY DENIED → violation logged |
| T4 | Worker — vest missing (no_vest rule) | NON-COMPLIANT via inference rule → DENY → alert |
| T5 | Unknown RFID / unregistered worker | IDENTITY FAILED → no attendance → alert supervisor |
| T6 | Registered worker + all PPE worn | Identified + compliant → check-in → gate opens |
| T7 | Baseball cap on non-worker | Should **NOT** trigger helmet detection (hard negative test) |
| T8 | Sunglasses on person | Should **NOT** trigger goggles detection (hard negative test) |
| T9 | Multiple workers (3) in frame | Each tracked and evaluated independently |
| T10 | Partial occlusion of helmet | Temporal logic: 5+ frame confirmation before violation |
| T11 | Same worker standing for 10s at gate | Only ONE check-in recorded (duplicate protection) |
| T12 | Worker exits — tracked on exit camera | CHECKOUT recorded; attendance session closed; duration calculated |
| T13 | Network/Wi-Fi fails mid-session | Local decision still works; events queued in SQLite; syncs when connection returns |
| T14 | MQTT broker disconnects | ESP32 fails gracefully; gate defaults to safe state; reconnect on restore |
| T15 | Low-light scene | Violations still detected — may need lower confidence threshold |

---

## 9. Research Evaluation Table

For publication, fill this table across all experiments:

| Model | Dataset | mAP50 | Precision | Recall | FPS | Device |
|---|---|---|---|---|---|---|
| YOLOv8n | Construction-PPE | … | … | … | ~60 (GPU) | Colab T4 |
| YOLOv8s | Construction-PPE | … | … | … | ~45 (GPU) | Colab T4 |
| YOLOv8s | Merged + Site | … | … | … | … | Colab T4 |
| YOLOv8n | Merged + Site | … | … | … | ~3–5 | RPi 5 CPU |
| YOLOv8n INT8 | Merged + Site | … | … | … | Target: 15+ | RPi 5 + Hailo-8L |

### Key Publication Metric

> **End-to-End Valid Entry Accuracy:** Of all real entry events, what % were correctly handled as — correct worker + correct PPE status + correct access decision + correct attendance record?

This single metric captures the full system, not just the YOLO model.

---

## 10. Recommended Module Structure

```
ppe_monitor/
│
├── config/
│   ├── camera.yaml          # RTSP URL, resolution, FPS
│   ├── zones.yaml           # zone names + required PPE per zone
│   ├── ppe_policy.yaml      # confidence thresholds per class
│   └── mqtt.yaml            # broker host, port, topics
│
├── models/
│   ├── best.pt              # trained weights
│   └── ppe_yolov8n.hef      # compiled Hailo model
│
├── detector/
│   └── ppe_detector.py      # YOLO inference wrapper
│
├── tracker/
│   └── worker_tracker.py    # ByteTrack / BoT-SORT wrapper
│
├── compliance/
│   ├── association.py       # PPE → Person assignment logic
│   ├── engine.py            # compliance evaluation + state machine
│   └── temporal.py          # N-frame confirmation logic
│
├── identity/
│   ├── rfid.py              # RFID reader via ESP32/MQTT
│   └── face.py              # FaceNet embeddings (Phase 12+)
│
├── attendance/
│   └── attendance_manager.py  # check-in/out, entry line, duplicate guard
│
├── iot/
│   ├── mqtt_client.py       # publish/subscribe + offline queue
│   └── gpio_controller.py   # direct GPIO if needed
│
├── database/
│   ├── models.py            # SQLAlchemy or raw SQLite schemas
│   └── repository.py        # CRUD functions
│
├── dashboard/
│   ├── api.py               # FastAPI routes
│   ├── websocket.py         # live feed WebSocket
│   └── frontend/            # React or HTML/JS
│
├── camera/
│   └── stream.py            # RTSP / USB / HEF camera wrapper
│
├── utils/
│   ├── visualization.py     # draw boxes, labels, overlays
│   └── logger.py            # structured event logging
│
└── main.py                  # application entry point
```

---

## 11. Deferred Features *(Not Now — After v1 Works)*

| Feature | Why Deferred |
|---|---|
| **Face Recognition / FaceNet** | Cannot tune identity on top of an unvalidated detector. Add after Phase 11 full integration. |
| **BoT-SORT Tracker** | Start with ByteTrack. Switch only if ID switches are frequent in your specific footage. |
| **Pose Estimation for PPE Association** | Improves PPE-to-body assignment in occlusion; add only if region-based association fails. |
| **Multi-Camera Tracking** | Needs handoff logic and shared track IDs. Prototype one camera first. |
| **Zone Policy UI Editor** | Hard-code `zones.yaml` for now. Add UI editor after MVP works. |
| **HR / Payroll Integration** | Post-pilot feature. Attendance DB exports to CSV for now. |
| **Cloud Video Streaming** | Never stream raw video to cloud. Only sanitised event JSON goes up. |
| **Incremental On-Device Learning** | Advanced feature; add only after stable deployment and sufficient pilot data. |

---

## 12. Master Timeline

| Phase | Focus | Duration | Gate / Output |
|---|---|---|---|
| **0** | Environment Setup | Days 1–3 | Python, Colab, Roboflow, Git all working |
| **1** | Dataset Preparation | Days 4–8 | Clean Ultralytics dataset, YAML validated, class balance checked |
| **2** | YOLOv8 Baseline Training | Days 9–14 | `best.pt` for YOLOv8n + YOLOv8s; mAP50 ≥ 0.70 before proceeding |
| **3** | Real-World Validation | Days 15–19 | Hard-negative test passed; per-class recall acceptable |
| **4** | Dataset Expansion + Fine-Tune | Days 20–27 | `ppe-v2.pt`; merged dataset + site images; Experiment 3 complete |
| **5** | Tracking + PPE Association | Days 28–33 | ByteTrack running; PPE assigned to correct person in multi-worker video |
| **6** | Compliance Engine | Days 34–38 | Zone policy working; temporal confirmation working; no single-frame alerts |
| **7** | Attendance & Identity | Days 39–45 | RFID check-in/out working; duplicate protection verified; SQLite populated |
| **8** | IoT Integration | Days 46–51 | ESP32 buzzer/LED triggers on violation; MQTT event flow verified |
| **9** | Edge Deployment | Days 52–58 | Model running on Pi + Hailo (or CPU fallback); FPS benchmarked |
| **10** | Dashboard + Cloud | Days 59–65 | FastAPI + React dashboard live; AWS sync working with offline queue |
| **11** | Full Integration | Days 66–72 | Complete pipeline verified end-to-end on test videos; test matrix passed |
| **12** | Pilot + Iteration | Weeks 16–20 | Real-location pilot; error log; `ppe-v3.pt`; final metrics for paper |

> **One Rule:** Always prove the AI first. Then prove the application logic. Then prove the hardware. Connect everything last.

---

## 13. Final Demonstration Scenarios

At the final project presentation, the panel should be able to see all three scenarios live:

### Scenario A — Fully Equipped Worker

```
Worker: EMP001  |  PPE: COMPLIANT  |  Entry: ALLOWED
Attendance: CHECK-IN 09:02:11
Dashboard: Workers Present: 1  |  Compliant Entries: 1
```

### Scenario B — Missing Helmet

```
Worker: EMP002  |  PPE: NON-COMPLIANT  |  Missing: HELMET
Entry: DENIED   |  Alert: TRIGGERED (buzzer + LED)
Dashboard: PPE Violations: 1  |  Denied Entries: 1
Attendance: NOT RECORDED
```

### Scenario C — Exit & Checkout

```
Worker: EMP001  |  Checkout: 17:41:22
Duration: 8h 39m  |  Attendance session closed
Dashboard: Currently Inside: 0
```

### What This Demonstrates

- A working, validated YOLOv8 PPE detector *(research contribution)*
- Real-time compliance enforcement — not just passive monitoring
- Integrated attendance tied to PPE status — not just face recognition
- IoT physical enforcement (gate/buzzer/LED) — not just a dashboard
- Edge AI deployment on cost-effective hardware (~₹15,000)
- Offline-first design — safety works without internet

---

*END OF MASTER IMPLEMENTATION PLAN*

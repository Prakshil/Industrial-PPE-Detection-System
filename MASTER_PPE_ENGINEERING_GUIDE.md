# The Industrial PPE Safety Intelligence Platform
## A Senior Engineer's Master Technical Guide: From Dataset Drift to 81.7% mAP50 Edge Intelligence

---

### Author's Foreword: The Engineering Reality of Industrial Safety AI

Building a real-world computer vision system for industrial safety is fundamentally different from training a benchmark model on Academic COCO. In a steel mill, manufacturing plant, chemical refinery, or construction site:
1. **False negatives are life-threatening**: Missing a worker entering a heavy machinery zone without a hard hat or eye protection can lead to severe injury, OSHA stop-work orders, and millions in corporate liability.
2. **False positives destroy trust**: If a camera sounds a physical siren every time a worker bends over or momentary shadow obscures their high-visibility vest, workers will disconnect the siren or safety officers will silence the dashboard.
3. **Compute is resource-constrained**: Cloud inference is impractical for sub-150ms alarm relays due to latency and network dropouts. The model must execute in real time on edge hardware (NVIDIA Jetson, Intel NUC, or on-premises industrial PCs) while maintaining $>80\%$ mAP across all equipment.

This manual documents the entire engineering journey of this project. It is written for senior machine learning engineers, computer vision practitioners, and software architects joining this codebase. It documents every experiment, failure mode, mathematical rationale, data anomaly, architectural pivot, and current operational state.

---

### Master Table of Contents
1. **The Baseline Era & The 11-Class Problem (The Collapse to 0.518 mAP)**
2. **Multi-Dataset Ingestion & The Annotation Geometry Drift Trap (The 18-Class Regression)**
3. **Mitigating Class Imbalance: Rebalancing Heuristics, Targeted Copy-Paste, and RFS**
4. **The Paradigm Shift: Pure Positive PPE Detection & Spatial Anatomical Reasoning**
5. **Comprehensive Historical Results Across All Training Runs**
6. **The Complete 10-Phase Production Edge & Frontend Architecture**
7. **Master Sequential Command Execution Guide (Phase-by-Phase Runbook)**
8. **Edge Device Deployment & Acceleration Manual (Raspberry Pi, ESP32, NVIDIA Jetson, Google Coral & OpenVINO)**

---

```
                                 EVOLUTIONARY TIMELINE OF THE PROJECT
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ PHASE A: Baseline 11-Class Model (0.518 mAP50)                                         │
 │ - Ultralytics construction-ppe dataset (1,132 train images).                           │
 │ - Direct detection of negative classes (no_helmet, no_boots, no_goggle).               │
 │ - Result: Catastrophic failure on negative classes (no_boots @ 0.012 AP).              │
 └───────────────────────────────────┬────────────────────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ PHASE B: Three-Source Dataset Merge (18-Class Schema v2 Regression)                    │
 │ - Ingested Roboflow and WearAware into base set (expanding to 18 classes).             │
 │ - Discovery of Geometry Conflict: Roboflow torso-sized No-Glasses vs base eye strip.   │
 │ - Merge tripled data but degraded AP from 0.522 to ~0.45 due to conflicting targets.   │
 └───────────────────────────────────┬────────────────────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ PHASE C: Balance, Augmentation & Sampling Heuristics                                   │
 │ - Split Rebalancing: integer optimization heuristic ensuring >= 30 boxes in val.      │
 │ - Targeted Copy-Paste: synthesized rare instances of boots/gloves on host backgrounds. │
 │ - Repeat Factor Sampling (RFS): oversampled images with class frequency < 0.05.        │
 │ - Result: Exposure improved, but negative bounding boxes remained inherently flawed.   │
 └───────────────────────────────────┬────────────────────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ PHASE D: Path 1 Paradigm Shift (Clean 6-Class Positive Detection)                      │
 │ - Discarded negative bounding boxes entirely; kept 6 positive physical classes.        │
 │ - Violations deduced via Spatial Anatomical Ratio Partitioning + N-frame state machine.│
 │ - Clamped 106 oversized 45-megapixel images to <= 1920px (prevented OpenCV OOM).      │
 └───────────────────────────────────┬────────────────────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ PHASE E: Dual Configuration Training & Final Convergence                               │
 │ - Run 1 (35 Epochs): Jumped to 0.809 mAP50 (+56% relative gain).                       │
 │ - Run 2 (60 Epochs): Completed full cosine LR decay schedule.                          │
 │ - Final Results: 0.818 mAP50 peak, 0.435 mAP50-95, 81.7% Test mAP50, 86.6% Precision.  │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Chapter 1: The Baseline Era & The 11-Class Problem

#### 1.1 The Original 11-Class Schema
The project originated using the public Ultralytics `construction-ppe` dataset, consisting of **1,132 training images**, 147 validation images, and 144 test images. The schema was configured as:

```python
CLASS_NAMES_V1 = [
    "helmet",    # 0: positive Hard Hat
    "gloves",    # 1: positive Safety Gloves
    "vest",      # 2: positive High-Visibility Vest
    "boots",     # 3: positive Steel-Toe Boots
    "goggles",   # 4: positive Safety Glasses
    "none",      # 5: ambiguous background / missing vest
    "Person",    # 6: human worker bounding box
    "no_helmet", # 7: negative Hard Hat violation
    "no_goggle", # 8: negative Glasses violation
    "no_gloves", # 9: negative Gloves violation
    "no_boots"   # 10: negative Boots violation
]
```

#### 1.2 The Catastrophic Collapse of Negative Classes
When trained on YOLOv8s, the headline validation score showed **0.518–0.522 all-class mAP50**. However, drilling down into individual class AP revealed a fatal divergence:

| Class Type | Class Name | Average Precision (AP@0.5) | Behavior / Analysis |
| :--- | :--- | :---: | :--- |
| **Positive Physical** | `Person` | **0.812** | Distinct humanoid silhouette; easily localized. |
| **Positive Physical** | `helmet` | **0.801** | High-contrast plastic shell on worker's head. |
| **Positive Physical** | `vest` | **0.774** | Fluorescent yellow/orange with reflective stripes. |
| **Negative Absence** | `no_helmet` | 0.485 | Mixed performance; easily confused with bare hair/caps. |
| **Negative Absence** | `no_gloves` | 0.220 | Bare hands tiny at 640px; skin-tone confusion. |
| **Negative Absence** | `no_goggle` | **0.108** | Severe collapse; eye region is only 8×15 pixels. |
| **Negative Absence** | `no_boots` | **0.012** | Total collapse; detector completely failed to learn. |

#### 1.3 The Root Cause: Why Object Detectors Cannot Detect "Absence"
In object detection (YOLO, Faster R-CNN, DETR), anchor assignment and feature pyramids depend on **positive visual invariants** (e.g., rigid edges, distinct color distributions, geometric textures). 

"Absence of boots" does not have an invariant appearance. It can be sneakers, sandals, work shoes, bare socks, or feet occluded by a 2x4 wooden plank, rubble, or shadows on the plant floor. Furthermore, at standard 640×640 input resolution:
- A worker standing 15 meters from the camera occupies roughly $120 \times 40$ pixels.
- The worker's eyes occupy $6 \times 12$ pixels.
- Forcing a Convolutional Neural Network to regress tight bounding coordinates on an invisible feature (`no_goggle`) results in chaotic gradients that destabilize backbone feature weights.

---

### Chapter 2: Multi-Dataset Ingestion & The Annotation Geometry Drift Trap

#### 2.1 The Merge Attempt
To address severe data scarcity in the negative classes (e.g., base validation only had 4 `no_boots` boxes across 2 images), we ingested two large external open-source datasets:
1. **Roboflow Construction PPE** (`PPE_detection`): ~6,400 images.
2. **WearAware v2** (`PPE_detection/wearaware_v2`): ~5,000 images.

We implemented `merge_ppe_datasets.py` and designed **Schema v2** (18 classes), mapping all aliases into a unified format while retaining auxiliary labels (`earmuff`, `mask`, `face_shield`, `harness`) so fully annotated external images would not be discarded.

```
External Class Mapping:
  "Hardhat", "Safety Helmet"           -> 0: helmet
  "Gloves", "Safety Gloves"           -> 1: gloves
  "Safety Vest", "Reflective Jacket"  -> 2: vest
  "Steel Toe Shoes", "Safety Boots"   -> 3: boots
  "Eye Protection", "Safety Glasses"  -> 4: goggles
  "no-vest", "none"                   -> 5: no_vest
  "Worker", "person"                  -> 6: Person
  "No-Safety-Glasses"                 -> 8: no_goggle  <-- THE GEOMETRIC TRAP!
```

#### 2.2 The Geometric Conflict Anomaly
When we merged all three sources into `merged_three_v1` and trained, **the model degraded significantly**—all-class mAP dropped from 0.522 to ~0.45, and `no_goggle` dropped from 0.128 to 0.081.

A forensic geometric analysis of the bounding box coordinates revealed the smoking gun:

```
 BASE ANNOTATION STYLE (no_goggle):
 +-------------------+
 |  [===EYES===]     |  Area = 0.8% of image height
 +-------------------+  Tight bounding box around eye strip only.

 ROBOFLOW ANNOTATION STYLE (No-Safety-Glasses):
 +-------------------+
 |     (HEAD)        |
 |     /    \        |  Area = 11.2x LARGER than base annotation!
 |    /TORSO \       |  Covers the entire head, neck, and upper chest!
 +-------------------+
```

Because Roboflow contributed **4,177 instances** of `No-Safety-Glasses` compared to only **337 instances** in the base dataset, the loss gradients were dominated by Roboflow's expansive definition. However, the evaluation validation set strictly contained base-style tight annotations. The model was correctly predicting Roboflow-style upper-torso boxes, but the validation metric penalized every prediction with an $\text{IoU} < 0.10$ as a pure False Positive. 

**Lesson Learned**: In automated dataset merging, label name alignment (`ALIASES`) is insufficient; **geometric annotation contracts** must be enforced.

---

### Chapter 3: Mitigating Imbalance via Rebalancing, Copy-Paste, and RFS

To fix dataset quality before changing architectures, we implemented three specialized data engineering stages in `PPE_detection/run_full_pipeline.py`:

#### 3.1 Validation Split Rebalancing (`rebalance_val_split.py`)
In the original split, `no_boots` had only 4 boxes in validation, making standard deviation across seeds $\pm 0.08$ (pure noise).
- We formulated an **Integer Constrained Optimization Heuristic** that reallocated images across `train`, `val`, and `test` splits to guarantee:
  $$\forall c \in \text{Classes}: \quad N_{\text{val}}(c) \ge 30, \quad N_{\text{test}}(c) \ge 30$$
- This transformed `merged_three_v2` into `merged_three_v2_balanced`, stabilizing validation metrics.

#### 3.2 Targeted Instance Copy-Paste (`rare_instance_augment.py`)
To solve the physical rarity of `no_boots` and `no_gloves`:
- We extracted segmentation masks and tight crops of verified rare negative instances.
- Using OpenCV alpha blending and spatial bounding constraints, we pasted these rare instances onto diverse, high-resolution host backgrounds (300 synthetic images generated into `augmented_rare/`).

#### 3.3 Repeat Factor Sampling (`repeat_factor_sampling.py`)
Following the LVIS dataset methodology (Gupta et al.), we calculated category exposure frequency:
$$f_c = \frac{N_{\text{images containing class } c}}{N_{\text{total images}}}$$
For classes where $f_c < \text{threshold } (0.05)$, each image containing that class received an oversampling repeat factor:
$$r_i = \max\left(1, \max_{c \in \text{image } i} \sqrt{\frac{0.05}{f_c}}\right)$$
This boosted rare image exposure during the epoch from ~40 occurrences to ~120 occurrences.

**Engineering Verdict**: While these techniques significantly improved exposure, they were fighting a flawed foundational premise: forcing the neural network to identify negative visual concepts.

---

### Chapter 4: The Paradigm Shift — Path 1: Pure Positive Detection & Spatial Reasoning

#### 4.1 The Core Mathematical Insight (Path 1)
Instead of forcing YOLOv8 to learn 12 contradictory concepts, we decoupled the problem into two distinct, high-efficiency stages:
1. **Perception Layer (Computer Vision)**: Detect only positive physical objects that possess invariant visual features:
   $$\mathcal{C}_{\text{positive}} = \{\text{Person}, \text{helmet}, \text{vest}, \text{gloves}, \text{boots}, \text{goggles}\}$$
2. **Cognition Layer (Spatial Anatomical Reasoning)**: Compute spatial intersections between detected gear and the worker's anatomical body subdivisions. If a required item is not present on the corresponding body segment, **infer a violation programmatically**.

```
              ANATOMICAL RATIO PARTITIONING OF WORKER BOUNDING BOX
 0.00 H ┌──────────────────────────────────────┐
        │ [0.00, 0.25] HEAD REGION             │  Target: HELMET
 0.25 H ├──────────────────────────────────────┤
        │ [0.05, 0.30] EYE STRIP               │  Target: GOGGLES
 0.30 H ├──────────────────────────────────────┤
        │                                      │
        │ [0.20, 0.70] TORSO REGION            │  Target: VEST
        │                                      │
 0.70 H ├──────────────────────────────────────┤
        │ [0.50, 0.85] WAIST / HAND REGION     │  Target: GLOVES
 0.85 H ├──────────────────────────────────────┤
        │ [0.75, 1.00] LOWER LEGS / FEET       │  Target: BOOTS
 1.00 H └──────────────────────────────────────┘
```

#### 4.2 Dataset Preparation (`prepare_6class_dataset.py`)
We filtered `merged_three_v2_balanced` down to 6 standardized positive classes into `dataset_6class/`:
- **Train Split**: 6,267 images (5,453 with positive gear, 814 background negative frames).
- **Val Split**: 147 images.
- **Test Split**: 144 images.
- **Total Positive Annotations**: 23,049 labeled objects (`Person`: 7,603; `helmet`: 4,652; `vest`: 4,134; `gloves`: 2,994; `boots`: 2,464; `goggles`: 1,202).

#### 4.3 Resolving the 45-Megapixel OpenCV Worker Crash
During early training batches, the PyTorch DataLoader threw:
`cv2.error: (-4:Insufficient memory) Failed to allocate 1228800 bytes in cv::OutOfMemoryError`
Investigation revealed that 106 images in the training set were massive (up to $8171 \times 5447$ pixels, 45 Megapixels, 134 MB uncompressed each). When multi-process DataLoader workers attempted Mosaic augmentation on these files, the Windows heap fragmented.

**Fix**:
1. Developed an automated clamping pass resizing all images $>1920\text{px}$ down to $\le 1920\text{px}$ using Lanczos downsampling. Normalized YOLO bounding boxes $(x_c, y_c, w, h)$ remained mathematically identical.
2. Set `workers=0` in `train_experiments.py` on Windows to eliminate child-process IPC memory bottlenecks.

---

### Chapter 5: Comprehensive Historical Results Across All Training Runs

Below is the complete, transparent historical progression of model performance across all four major training milestones of this project:

```
  mAP@0.5 Progression Across Project Phases:
  0.90 ┤
  0.80 ┤                                           ┌───────────────┐ (60e: 0.818 Peak)
  0.70 ┤                               ┌───────────┘ 0.809 (35e)
  0.60 ┤
  0.50 ┤ 0.522 (11-Class Base)
  0.40 ┤        └───┐ 0.450 (18-Class Merge Drift)
  0.30 ┤
       └────────────┴──────────────────┴───────────┴───────────────┴────────►
         Phase A          Phase B        Phase D       Phase E (Final)
```

#### Detailed Head-to-Head Comparison Table

| Metric / Class | Run 1: 11-Class Base (`construction-ppe`) | Run 2: 18-Class Merge (`merged_three_v1`) | Run 3: 6-Class 35e (`runs/ppe_6class`) | Run 4: 6-Class 60e (`runs/ppe_6class_60eps`) |
| :--- | :---: | :---: | :---: | :---: |
| **All-Class mAP@0.5** | 0.518 – 0.522 | ~0.450 (Degraded) | **0.809** (+56%) | **0.818 (Peak @ Ep 40)** |
| **All-Class mAP@0.5-95** | 0.252 – 0.256 | ~0.210 (Degraded) | 0.422 (+67%) | **0.435 (+73%)** |
| `Person` AP@0.5 | 0.812 | 0.795 | **0.821** | 0.818 |
| `helmet` AP@0.5 | 0.801 | 0.808 | 0.821 | **0.839** |
| `vest` AP@0.5 | 0.774 | 0.816 | **0.830** | 0.828 |
| `gloves` AP@0.5 | 0.690 | 0.734 (Drift) | **0.810** | 0.789 |
| `boots` AP@0.5 | 0.584 | 0.540 (Drift) | **0.781** | 0.759 |
| `goggles` AP@0.5 | 0.380 | 0.290 (Drift) | 0.788 | **0.812 (+113.7%)** |
| `no_helmet` AP@0.5 | 0.485 | 0.410 (Drift) | *Deduced by Rule Engine* | *Deduced by Rule Engine* |
| `no_gloves` AP@0.5 | 0.220 | 0.180 (Drift) | *Deduced by Rule Engine* | *Deduced by Rule Engine* |
| `no_goggle` AP@0.5 | 0.108 | 0.081 (Collapse) | *Deduced by Rule Engine* | *Deduced by Rule Engine* |
| `no_boots` AP@0.5 | **0.012** | **0.035** (Collapse) | *Deduced by Rule Engine* | *Deduced by Rule Engine* |
| **Test Set Precision** | ~0.610 (Spiky) | ~0.580 (Noisy) | 0.863 (Smooth) | **0.866 (Smooth)** |
| **Test Set Recall** | ~0.520 | ~0.490 | 0.757 | **0.776 (+1.9%)** |
| **Test Set mAP@0.5** | — | — | 0.8156 | **0.8174** |

---

### Chapter 6: The Complete 10-Phase Production Edge & Frontend Architecture

The system is built as a modular industrial edge intelligence platform, partitioned into 10 decoupled phases:

```
  VIDEO SOURCE (Webcam / RTSP / MP4)
         │
         ▼
 ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
 │ Phase 2:      │────►│ Phase 3:      │────►│ Phase 4:      │
 │ Video Ingest  │     │ ByteTrack     │     │ Spatial Assoc │
 │ & YOLOv8s     │     │ Worker IDs    │     │ Body Ratios   │
 └───────────────┘     └───────────────┘     └───────┬───────┘
                                                     │
                                                     ▼
 ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
 │ Phase 8:      │◄────│ Phase 7:      │◄────│ Phase 5:      │
 │ OpenCV HMI    │     │ MQTT Alert    │     │ Zone Policy   │
 │ Visualizer    │     │ Broker        │     │ State Machine │
 └───────────────┘     └───────────────┘     └───────┬───────┘
                                                     │
                                                     ▼
 ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
 │ Phase 10:     │◄────│ Phase 9:      │◄────│ Phase 6:      │
 │ React / Vite  │     │ FastAPI Cloud │     │ SQLite WAL    │
 │ Operator UI   │     │ REST / WSS    │     │ & Evidence    │
 └───────────────┘     └───────────────┘     └───────────────┘
```

#### Detailed Phase Breakdown

* **Phase 1: 6-Class Positive Detector Model**:
  - Model weights: [`runs/ppe_6class_60eps/ultra/ultra/weights/best.pt`](file:///c:/Users/Prakshil/OneDrive/Desktop/ppe_clone/Project%20Sem%207/runs/ppe_6class_60eps/ultra/ultra/weights/best.pt).
  - High precision, low false-positive rate (86.6% precision, 77.6% recall).
* **Phase 2: Video Stream Ingestion Layer ([`PPEDetector`](file:///c:/Users/Prakshil/OneDrive/Desktop/ppe_clone/Project%20Sem%207/ppe_monitor/ppe_monitor/detector/yolo_detector.py#L22))**:
  - Ingests RTSP, webcams, or video files. Automatically partitions raw YOLO detections into `persons` and `ppe_items`. Throttles frame rate to `--max-fps` (default 25 FPS).
* **Phase 3: Multi-Object Worker Tracker ([`PersonTracker`](file:///c:/Users/Prakshil/OneDrive/Desktop/ppe_clone/Project%20Sem%207/ppe_monitor/ppe_monitor/tracker/bytetrack_wrapper.py#L13))**:
  - Wraps `supervision.ByteTrack`. Assigns persistent integer `track_id` to each worker, maintaining identity for up to 30 frames during temporary occlusions.
* **Phase 4: Spatial Anatomical Association Engine ([`association.py`](file:///c:/Users/Prakshil/OneDrive/Desktop/ppe_clone/Project%20Sem%207/ppe_monitor/ppe_monitor/tracker/association.py#L80))**:
  - Slices worker bounding box into vertical anatomical proportions:
    - **Helmet**: `[0.00, 0.25]`
    - **Goggles**: `[0.05, 0.30]`
    - **Vest**: `[0.20, 0.70]`
    - **Gloves**: `[0.50, 0.85]`
    - **Boots**: `[0.75, 1.00]`
  - Matches gear via centroid containment or region $\text{IoU} \ge 0.10$. Rejects gear dropped on the ground.
* **Phase 5: Zone Compliance Engine & Temporal State Machine ([`WorkerState`](file:///c:/Users/Prakshil/OneDrive/Desktop/ppe_clone/Project%20Sem%207/ppe_monitor/ppe_monitor/compliance/worker_state.py#L34), [`ComplianceEngine`](file:///c:/Users/Prakshil/OneDrive/Desktop/ppe_clone/Project%20Sem%207/ppe_monitor/ppe_monitor/compliance/engine.py#L19))**:
  - Configured via [`zones.yaml`](file:///c:/Users/Prakshil/OneDrive/Desktop/ppe_clone/Project%20Sem%207/ppe_monitor/config/zones.yaml).
  - Enforces zone-specific rules (`main_entry`, `active_work_zone`, `heavy_machinery_zone`).
  - Requires $N$ consecutive missing frames ($N=3$ to $5$) before confirming a violation. Suppresses redundant alarms with a 30-second cooldown timer.
* **Phase 6: Offline-First SQLite WAL Persistence ([`EventDatabase`](file:///c:/Users/Prakshil/OneDrive/Desktop/ppe_clone/Project%20Sem%207/ppe_monitor/ppe_monitor/persistence/database.py#L36))**:
  - High-concurrency local SQLite database (`events.db`) operating with Write-Ahead Logging (`PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;`).
  - Enables zero-lock concurrent reads by the FastAPI backend while the edge inference pipeline continuously ingests violations and logs telemetry ticks.
  - Relational Schema:
    - `zones`: Zone definitions, mandatory PPE gear list, $N$-frame confirmation window, cooldown seconds.
    - `cameras`: Video sources, RTSP stream URIs, zone bindings, active camera status.
    - `violation_events`: Full incident ledger containing unique UUID, timestamp, zone, camera, track ID, missing gear array, triage state (`PENDING`, `CONFIRMED`, `DISMISSED`), reviewer notes, operator ID, and evidence crop path.
    - `telemetry_ticks`: High-frequency time-series buffer storing compliance rate, worker counts, active violations, and FPS.
  - Cryptographic Chain of Custody: Automatically crops and stores JPEG violation snapshots in `evidence/` with SHA-256 hashes for OSHA/ISO legal verification.

* **Phase 7: Real-Time MQTT Edge Alerting ([`MQTTClient`](file:///c:/Users/Prakshil/OneDrive/Desktop/ppe_clone/Project%20Sem%207/ppe_monitor/ppe_monitor/messaging/mqtt_client.py#L18))**:
  - Publishes JSON violation payloads to `ppe/<zone_id>/violations` for physical siren relays, PLC controllers, and SCADA integration. Gracefully degrades if the broker is offline.

* **Phase 8: Edge OpenCV HMI Visualizer ([`Visualizer`](file:///c:/Users/Prakshil/OneDrive/Desktop/ppe_clone/Project%20Sem%207/ppe_monitor/ppe_monitor/display/visualizer.py#L45))**:
  - Draws color-coded worker boxes: Green (Compliant), Yellow (Pending), Red (Confirmed Violation).
  - Renders live checklists under workers (`[OK] helmet`, `[X] vest (3/5)`), flashing banners, active worker tally, and real-time inference FPS.

* **Phase 9: Central Platform FastAPI Backend ([`api/main.py`](file:///c:/Users/Prakshil/OneDrive/Desktop/ppe_clone/Project%20Sem%207/ppe_monitor/api/main.py))**:
  - High-performance asynchronous FastAPI service powered by Uvicorn.
  - **RESTful Endpoints**:
    - `GET /api/v1/health`: System health status, database engine state (`SQLite WAL active`), version, and uptime.
    - `GET /api/v1/zones`: Complete registry of plant safety zones and current gear policies.
    - `PUT /api/v1/zones/{zone_id}`: Dynamic runtime policy reconfiguration (update mandatory PPE items, confirmation window, cooldown).
    - `GET /api/v1/cameras`: Camera registry with RTSP links and zone associations.
    - `GET /api/v1/violations`: Filterable incident ledger by zone, triage status (`PENDING`, `CONFIRMED`, `DISMISSED`), and record limit.
    - `POST /api/v1/violations/{violation_id}/action`: Triage workflow actions to transition incidents between `CONFIRMED` and `DISMISSED` with reviewer audit notes.
    - `GET /api/v1/telemetry`: 24-hour compliance rollups, active worker counts, violation counts, and FPS statistics.
    - `POST /api/v1/edge/sync`: Batch ingestion endpoint for edge nodes pushing offline SQLite events to the central ledger.
    - `GET /api/v1/export/audit`: One-click regulatory audit trail export in standard `CSV` or structured `JSON` formatted for OSHA and ISO 45001 auditors.
    - `GET /api/v1/evidence/{filename}`: Secure image retrieval endpoint serving high-resolution forensic incident crops.
    - `GET /api/v1/stream/feed?zone=...`: Low-latency multipart MJPEG streaming feed with real-time bounding box and checklist HUD overlays.
  - **WebSocket Gateway (`/ws/telemetry`)**: Full-duplex WebSocket connection manager broadcasting sub-second worker detections, spatial checklists, and active alarm triggers to all connected dashboard clients.
  - **Unified Production Serving**: Automatically serves the compiled React Single Page Application (`frontend/dist/`) at the root URL `/`, enabling complete zero-dependency single-process deployment.

* **Phase 10: Frontend Industrial Operator Console (React 19 + Vite + Tailwind CSS + Lucide)**:
  - Built strictly adhering to the **industrial white theme** (`#FFFFFF` panels, `#F8FAFC` slate canvas, `#0F172A` high-contrast typography) to prevent eye strain and glare in brightly illuminated plant control rooms, rejecting dark SaaS slop.
  - ANSI Z535 and ISO 7010 safety color standard implementation:
    - **Safety Green** (`#16A34A` / `#DCFCE7`): Fully compliant workers and verified status.
    - **Safety Amber / Caution** (`#D97706` / `#FEF3C7`): Pending confirmation window or temporary occlusion.
    - **Safety Red / Danger** (`#DC2626` / `#FEE2E2`): Confirmed active violations and critical sirens.
  - **Modular Component Breakdown** (`frontend/src/components/`):
    - **`Header.jsx`**: Global plant status indicator, zone switcher dropdown, live synchronized local/UTC clock, siren mute switch, and one-click OSHA compliance CSV/JSON report export.
    - **`CameraFeed.jsx`**: Multi-channel camera matrix switcher (`CAM-01 Loading Bay`, `CAM-02 Assembly Line`, `CAM-03 Chemical Vault`), low-latency live MJPEG canvas, heads-up telemetry HUD, overlay toggles (bounding boxes, confidence values, track labels), and an active violation alarm banner.
    - **`IncidentQueue.jsx`**: High-density triage queue displaying detected violations, worker IDs, missing equipment tags, timestamps, and action buttons (`Confirm Violation`, `Dismiss False Positive`, `Forensic Inspection`).
    - **`EvidenceModal.jsx`**: High-resolution forensic crop inspector, SHA-256 chain-of-custody verification hash, 5-part anatomical checklist with model confidences (`helmet`, `vest`, `boots`, `gloves`, `goggles`), audit notes entry, and operator confirmation.
    - **`AnalyticsOverview.jsx`**: Real-time KPI summary (24h compliance %, active worker count, violations logged today, streaming FPS), horizontal PPE distribution bars, and plant zone risk rankings.
    - **`PolicyConfigModal.jsx`**: Dynamic zone-level safety policy configurator to enforce mandatory gear sets, hysteresis confirmation window ($N$ frames), and cooldown timers.
    - **`App.jsx`**: Master container managing state synchronization, optimistic UI updates, and auto-reconnecting WebSocket connection.
    - **`services/api.js`**: Centralized REST client and WebSocket wrapper handling communication with FastAPI.
  - **Production Build Performance**: Compiled using Vite in 1.40s to a minified 262 KB JavaScript bundle and 19.6 KB CSS bundle with zero dependencies required on edge nodes.

---

### Chapter 7: Master Sequential Command Execution Guide

This section is the definitive reference runbook for executing every phase of the pipeline sequentially from scratch.

#### Stage 1: Ingest & Merge Three Source Datasets
Merges the base Ultralytics dataset with Roboflow and WearAware:

```powershell
.\venv\Scripts\python.exe "PPE_detection\merge_ppe_datasets.py" `
    --base-root "PPE_detection\construction-ppe" `
    --source "roboflow=PPE_detection" `
    --source "wearaware=PPE_detection\wearaware_v2" `
    --out "merged_three_v2" `
    --schema v2 `
    --dedup-sort neutral
```

#### Stage 2: Rebalance Validation and Test Splits
Rebalances splits using integer optimization to ensure $\ge 30$ boxes per class in evaluation:

```powershell
.\venv\Scripts\python.exe "PPE_detection\rebalance_val_split.py" `
    --dataset "merged_three_v2" `
    --out "merged_three_v2_balanced" `
    --min-val-boxes 30
```

#### Stage 3: Extract Clean 6-Class Positive Dataset
Filters the balanced dataset down to the 6 positive classes and clamps oversized images $\le 1920\text{px}$:

```powershell
.\venv\Scripts\python.exe "PPE_detection\prepare_6class_dataset.py" `
    --source "merged_three_v2_balanced" `
    --out "dataset_6class" `
    --overwrite
```

#### Stage 4: Train the YOLOv8s Model

##### Option A: 35-Epoch Calibration Run (~35 mins)
```powershell
.\venv\Scripts\python.exe "PPE_detection\train_experiments.py" `
    --ultra-root "dataset_6class" `
    --runs ultra `
    --classes Person helmet vest gloves boots goggles `
    --epochs 35 `
    --patience 10 `
    --cls-pw 1.0 `
    --batch auto `
    --workers 0 `
    --device 0 `
    --project "runs/ppe_6class"
```

##### Option B: 60-Epoch Full Convergence Run (~58 mins - Current Production Best)
```powershell
.\venv\Scripts\python.exe "PPE_detection\train_experiments.py" `
    --ultra-root "dataset_6class" `
    --runs ultra `
    --classes Person helmet vest gloves boots goggles `
    --epochs 60 `
    --patience 15 `
    --cls-pw 1.0 `
    --batch auto `
    --workers 0 `
    --device 0 `
    --project "runs/ppe_6class_60eps"
```

#### Stage 5: Evaluate Checkpoints on Frozen Test Split
Evaluates model weights on the 144 unseen test images:

```powershell
# Evaluate the 60-epoch production model on the test split:
.\venv\Scripts\python.exe -c "from ultralytics import YOLO; m = YOLO('runs/ppe_6class_60eps/ultra/ultra/weights/best.pt'); res = m.val(data='dataset_6class/data.yaml', split='test'); print('Test mAP50:', res.box.map50, 'Test mAP50-95:', res.box.map)"
```

#### Stage 6: Execute Spatial Association Unit Tests
Verifies anatomical vertical slicing and dropped-gear rejection:

```powershell
$env:PYTHONPATH="ppe_monitor"
.\venv\Scripts\python.exe -m pytest ppe_monitor/tests/test_association.py -v
```

#### Stage 7: Run Live Edge Video Monitoring

##### Running on Live Webcam (Camera Index 0):
```powershell
.\venv\Scripts\python.exe main.py --source 0 --zone active_work_zone --save-evidence
```

##### Running on Recorded MP4 / AVI Test Video File:
```powershell
.\venv\Scripts\python.exe main.py --source "path/to/industrial_test_feed.mp4" --zone main_entry --save-evidence
```

##### Running on IP Camera RTSP Stream:
```powershell
.\venv\Scripts\python.exe main.py --source "rtsp://admin:password@192.168.1.100:554/h264Preview_01_main" --zone heavy_machinery_zone --save-evidence
```

##### Running Headless on Edge Industrial PC (No OpenCV Window):
```powershell
.\venv\Scripts\python.exe main.py --source 0 --no-display --save-evidence --max-fps 25
```

#### Stage 8: Start the Central Safety Gateway & Unified Operator Console
Launches the FastAPI backend serving both the RESTful API, WebSocket telemetry broker, dynamic MJPEG streams, and the compiled industrial React dashboard from a single production process:

```powershell
# Run using the active Python environment:
python -m uvicorn ppe_monitor.api.main:app --host 0.0.0.0 --port 8000
```

- **Industrial Operator Dashboard**: `http://localhost:8000/` (clean white-theme industrial layout)
- **Interactive Swagger REST Documentation**: `http://localhost:8000/docs`
- **Health Check & WAL Mode Verification**: `http://localhost:8000/api/v1/health`
- **Zone Policies & Telemetry Rollup**: `http://localhost:8000/api/v1/zones` & `http://localhost:8000/api/v1/telemetry`
- **Live Video Streaming Feed**: `http://localhost:8000/api/v1/stream/feed?zone=active_work_zone`

#### Stage 9: Frontend Development & Production Compilation
If modifying or extending UI components (`frontend/src/components/`):

##### A. Start Vite Local Development Server with Hot-Module Replacement (HMR):
```powershell
cd frontend
npm run dev
```
Opens the live developer console at `http://localhost:5173/`, proxying REST and WebSocket calls to the FastAPI backend on port 8000.

##### B. Compile Production Distribution Bundle:
```powershell
cd frontend
npm run build
```
Compiles and tree-shakes all React JSX, Tailwind CSS, and Lucide icons into `frontend/dist/` in ~1.40s. Once built, the FastAPI server automatically serves these optimized static assets directly.

#### Stage 10: Run the Production Automated Test Suites
Executes comprehensive regression and functional tests across all platform tiers:

##### A. Database Engine & RESTful API Test Suite:
Validates SQLite WAL concurrency, transactional integrity, incident logging, triage state transitions, zone policy mutations, and all REST endpoints:
```powershell
python ppe_monitor/tests/test_database_and_api.py
```
*Expected Output*: `ALL DATABASE AND FASTAPI API TESTS PASSED!`

##### B. Spatial Anatomical Association Test Suite:
Validates vertical anatomical ratio partitioning, multi-gear association, and rejection of misplaced or dropped equipment:
```powershell
python ppe_monitor/tests/test_association.py
```
*Expected Output*: `ALL SPATIAL ASSOCIATION TESTS PASSED!`

#### Stage 11: Export Regulatory Safety Audit Reports
Generate instant compliance reports formatted for OSHA inspections and ISO 45001 safety audits:

##### A. Export Compliance Audit Ledger as Styled Microsoft Excel Workbook (`.xls`):
Generates a pre-formatted HTML-based workbook with fixed column widths, deep navy headers, alternating row styling, and color-coded status badges. This completely eliminates cell clipping and date `########` truncation in Microsoft Excel:
```powershell
curl -o osha_compliance_audit.xls "http://localhost:8000/api/v1/export/audit?format=excel"
```

##### B. Export Compliance Audit Ledger as CSV (with UTF-8 Byte Order Mark):
Includes a UTF-8 BOM (`\ufeff`) header so that Windows Excel opens the CSV cleanly without character encoding artifacts or broken column boundaries:
```powershell
curl -o osha_compliance_audit.csv "http://localhost:8000/api/v1/export/audit?format=csv"
```

##### C. Export Structured Compliance Audit Ledger as JSON:
```powershell
curl -o osha_compliance_audit.json "http://localhost:8000/api/v1/export/audit?format=json"
```

#### Stage 12: Upload Custom Video for Automated Edge Inspection
Allows operators to upload any industrial video clip (`.mp4`, `.avi`, `.mov`) directly through the UI or REST API to run YOLOv8s detection, continuous ByteTrack, spatial association, and automatic violation logging:

```powershell
curl -X POST "http://localhost:8000/api/v1/video/upload-and-process" `
  -F "file=@path/to/cctv_footage.mp4" `
  -F "zone_id=active_work_zone" `
  -F "max_frames=250"
```
*Result*: Returns JSON metrics (frames processed, unique workers tracked, violations detected, FPS), writes confirmed incidents and SHA-256 evidence crops directly to SQLite `events.db`, and registers the annotated feed under **CAM-04 (Uploaded Video Feed)** for instant playback.

#### Stage 13: Offline-to-Cloud Database Synchronization
When edge nodes operate in offline network isolation, their local SQLite violations are batch-synchronized to the central gateway upon network restoration:

```powershell
python -c "
import requests
from ppe_monitor.persistence.database import EventDatabase

db = EventDatabase('events.db')
unsynced = db.get_unsynced(limit=100)
if unsynced:
    payload = {'edge_node': 'edge_camera_01', 'events': unsynced}
    r = requests.post('http://localhost:8000/api/v1/edge/sync', json=payload)
    print('Sync status:', r.status_code, r.json())
    if r.status_code == 200:
        db.mark_synced([e['id'] for e in unsynced])
        print(f'Successfully synchronized {len(unsynced)} violation records to central ledger.')
else:
    print('Local SQLite database is fully synchronized.')
"
```

---

### Chapter 8: Edge Device Deployment & Acceleration Manual
#### Deploying 81.7% mAP50 YOLOv8s across Raspberry Pi, ESP32, NVIDIA Jetson, Google Coral, and Intel OpenVINO

Edge deployment of real-time industrial safety computer vision requires matching model architecture to memory bandwidth, compute throughput, and thermal/power budgets.

```
                              EDGE DEPLOYMENT HARDWARE SPECTRUM
  Microcontrollers (ESP32)      Single-Board Computers (Pi 5)       Edge Accelerators (Jetson / Coral)
  ┌─────────────────────────┐   ┌───────────────────────────────┐   ┌────────────────────────────────┐
  │ SRAM: 512 KB            │   │ RAM: 4-8 GB                   │   │ RAM: 8-16 GB Unified           │
  │ PSRAM: 8 MB (SPI bus)   │   │ CPU: 4x Cortex-A76            │   │ Compute: 40-100 TOPS (TensorRT)│
  │ Model: Tiny sensor only │   │ Model: INT8 / NCNN / Hailo    │   │ Model: Full YOLOv8s TensorRT   │
  │ FPS: Ingest/Relay only  │   │ FPS: 15-40 FPS                │   │ FPS: 80-120+ FPS               │
  └─────────────────────────┘   └───────────────────────────────┘   └────────────────────────────────┘
```

---

#### 8.1 Raspberry Pi Deployment (Pi 4, Pi 5, & Pi 5 + Hailo-8L AI Kit)

##### A. Raspberry Pi 5 (CPU Inference via NCNN or ONNX Runtime)
- **Bottleneck**: Pure PyTorch Python on ARM Cortex-A76 yields only ~2–4 FPS due to framework overhead and unoptimized matrix kernels.
- **Solution**: Export the production checkpoint (`runs/ppe_6class_60eps/ultra/ultra/weights/best.pt`) to **NCNN** (Tencent's ultra-optimized mobile neural network computing framework) or **ONNX Runtime** utilizing ARM NEON SIMD vector extensions.

**Export Commands**:
```powershell
# 1. Export to NCNN (recommended for pure ARM CPU execution):
yolo export model=runs/ppe_6class_60eps/ultra/ultra/weights/best.pt format=ncnn imgsz=640

# 2. Export to ONNX (with graph simplification):
yolo export model=runs/ppe_6class_60eps/ultra/ultra/weights/best.pt format=onnx imgsz=640 simplify=True
```

**Running on Raspberry Pi 5**:
```python
from ultralytics import YOLO

# Loads the compiled NCNN model directly on ARM:
model = YOLO("runs/ppe_6class_60eps/ultra/ultra/weights/best_ncnn_model")
results = model.predict(source=0, imgsz=640, conf=0.25)
```
- **Performance**:
  - Raspberry Pi 4 (Cortex-A72): ~4–6 FPS (NCNN FP16)
  - Raspberry Pi 5 (Cortex-A76): ~12–16 FPS (NCNN FP16)

##### B. Raspberry Pi 5 + Raspberry Pi AI Kit (Hailo-8L NPU) — *Recommended for Production Pi*
- **Specs**: 13 TOPS neural accelerator connected via PCIe Gen 2/3 M.2 HAT+ (~$70 total add-on).
- **Workflow**:
  1. Export model to ONNX with opset 11:
     ```powershell
     yolo export model=runs/ppe_6class_60eps/ultra/ultra/weights/best.pt format=onnx imgsz=640 opset=11
     ```
  2. Compile to Hailo Executable Format (`.hef`) using the Hailo Dataflow Compiler (DFC):
     ```bash
     hailo parser onnx best.onnx --hw-arch hailo8l
     hailo optimize best.har --calib-set-path dataset_6class/images/val/
     hailo compiler best.har --hw-arch hailo8l
     ```
  3. Execute via `pyhailort` or GStreamer integration in Python.
- **Performance**: **35–45 FPS in real time** at under 2 Watts power draw.

---

#### 8.2 ESP32 & ESP32-CAM (Industrial IoT Distributed Node Topology)

##### The Physical Hardware Reality:
- Standard ESP32 features **520 KB internal SRAM**; ESP32-S3 features 512 KB SRAM and up to 8–16 MB external PSRAM connected over a serial SPI bus (~40–80 MHz).
- The production YOLOv8s model has **11.2 million parameters** (~45 MB in FP32, ~11 MB in INT8).
- **YOLOv8s cannot run directly on an ESP32.** The SPI bus memory bandwidth and dual-core Xtensa CPU (240 MHz without vector tensor units) cannot process 11M matrix multiplications per frame without running out of memory or taking 30–60 seconds per frame.

##### The Industrial Standard: Distributed Sensor/Actuator Architecture
In plant automation, ESP32 microcontrollers are deployed as **low-cost rugged endpoints** for video streaming and physical siren relays, while inference is offloaded to the edge gateway (Raspberry Pi, Jetson, or PC):

```
 ┌──────────────────────┐     RTSP / HTTP JPEG     ┌──────────────────────┐
 │   ESP32-CAM Node     │─────────────────────────►│  Central Edge Gateway│
 │  (Mount at Haz Zone) │     (Sub-100ms Stream)   │ (Pi 5 / Jetson / PC) │
 └──────────┬───────────┘                          └──────────┬───────────┘
            ▲                                                 │
            │              MQTT Violation Alert               │
            └─────────────────────────────────────────────────┘
                         (Triggers Siren / Strobe)
```

1. **Video Ingestion via ESP32-CAM**:
   The ESP32-CAM streams 720p/1080p JPEG over Wi-Fi/Ethernet via RTSP (`rtsp://192.168.1.150:8554/mjpeg/1`).
2. **Central Ingestion Command**:
   ```powershell
   python main.py --source "rtsp://192.168.1.150:8554/mjpeg/1" --zone chemical_bay --save-evidence
   ```
3. **Physical Siren Relay via ESP32 Hardware Subscriber**:
   When the central pipeline confirms a violation, it publishes an MQTT message to `ppe/chemical_bay/violations`. An ESP32 connected to the physical plant siren pulls a relay GPIO pin HIGH:

```cpp
// ESP32 Arduino / ESP-IDF Client:
#include <WiFi.h>
#include <PubSubClient.h>

const char* mqtt_broker = "192.168.1.100"; // Central Gateway IP
const int RELAY_PIN = 4; // GPIO connected to 12V Industrial Siren Relay
WiFiClient espClient;
PubSubClient client(espClient);

void callback(char* topic, byte* payload, unsigned int length) {
  // Confirmed PPE violation received from edge gateway:
  digitalWrite(RELAY_PIN, HIGH); // Siren ON
  delay(3000);                   // Sound alarm for 3 seconds
  digitalWrite(RELAY_PIN, LOW);  // Siren OFF
}

void setup() {
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
  WiFi.begin("PLANT_WIFI", "IndustrialPassword");
  while (WiFi.status() != WL_CONNECTED) delay(500);
  client.setServer(mqtt_broker, 1883);
  client.setCallback(callback);
}

void loop() {
  if (!client.connected()) client.connect("ESP32_Siren_Relay_01");
  client.subscribe("ppe/chemical_bay/violations");
  client.loop();
}
```

*(Note: If a model strictly MUST execute on-chip on an ESP32-S3, one must train a sub-0.5M parameter micro-network such as Edge Impulse FOMO or MobileNetV2-SSD 0.1x via TensorFlow Lite Micro. However, FOMO produces only centroid points rather than bounding boxes, preventing spatial anatomical ratio partitioning).*

---

#### 8.3 NVIDIA Jetson Family (Nano, Orin Nano, Orin NX, AGX Orin)

NVIDIA Jetson is the industry standard for multi-camera edge computer vision.

##### Export to NVIDIA TensorRT Engine:
```powershell
# Export production checkpoint to FP16 TensorRT engine:
yolo export model=runs/ppe_6class_60eps/ultra/ultra/weights/best.pt format=engine half=True device=0
```

##### Run Inference on Jetson:
```powershell
python main.py --weights runs/ppe_6class_60eps/ultra/ultra/weights/best.engine --source 0 --zone active_work_zone
```

##### Jetson Hardware Performance:
| Hardware | Compute | Power | Precision | Latency | Real-World FPS | Multi-Stream Capacity |
|---|---|---|---|---|---|---|
| **Jetson Nano (Legacy)** | 0.5 TFLOPS | 10W | FP16 | ~45ms | 18–22 FPS | 1 Camera |
| **Jetson Orin Nano** | 40 TOPS | 7W–15W | FP16 | ~8ms | **80–110 FPS** | 4–6 Cameras |
| **Jetson Orin NX** | 100 TOPS | 10W–25W | INT8 | ~3.5ms | **180+ FPS** | 8–12 Cameras |
| **Jetson AGX Orin** | 275 TOPS | 15W–60W | INT8 | ~1.8ms | **350+ FPS** | 16–32 Cameras |

---

#### 8.4 Google Coral Edge TPU (USB Accelerator / M.2)

- **Specs**: 4 TOPS custom ASIC accelerator operating at ~2 Watts ($60 USB 3.0 dongle).
- **Quantization Constraint**: Coral requires full 8-bit integer quantization (INT8) across all weights and activations.

**Export Command**:
```powershell
# Export to INT8 TensorFlow Lite model using validation split for calibration:
yolo export model=runs/ppe_6class_60eps/ultra/ultra/weights/best.pt format=edgetpu imgsz=640 data=dataset_6class/data.yaml
```
- **Performance**:
  - Raspberry Pi 4/5 + Coral USB 3.0: **28–35 FPS** in real time.

---

#### 8.5 Intel OpenVINO (Industrial PCs & Intel NUCs)

For standard industrial fanless PCs (x86-64 Intel Core i3/i5/i7/i9 or Celeron/Atom):

**Export Command**:
```powershell
# Export to OpenVINO FP16:
yolo export model=runs/ppe_6class_60eps/ultra/ultra/weights/best.pt format=openvino half=True
```

**Running on Intel NUC / IPC**:
```python
from ultralytics import YOLO

model = YOLO("runs/ppe_6class_60eps/ultra/ultra/weights/best_openvino_model")
results = model.predict(source=0, imgsz=640, device="cpu") # Uses Intel AVX-512 / Iris Xe GPU
```
- **Performance**:
  - Intel Core i5 12th/13th Gen CPU: **35–50 FPS**
  - Intel Iris Xe Integrated GPU: **55–75 FPS**

---

#### 8.6 Edge Platform Comparative Decision Matrix

| Platform | Deployment Role | Model Format | Power Budget | Cost | Expected FPS |
|---|---|---|---|---|---|
| **ESP32-CAM + Relay** | Video Ingest & Siren Actuator | Streamer / MQTT | ~1W | ~$8 | N/A (Offloaded) |
| **Raspberry Pi 4 (CPU)** | Standalone Edge Node | NCNN (FP16) | ~5W | ~$45 | 4–6 FPS |
| **Raspberry Pi 5 (CPU)** | Standalone Edge Node | NCNN (FP16) | ~8W | ~$60 | 12–16 FPS |
| **Pi 5 + AI Kit (Hailo-8L)** | Edge Gateway (1–2 Cams) | Hailo HEF | ~10W | ~$130 | **35–45 FPS** |
| **Pi 4/5 + Coral USB** | Edge Gateway (1 Cam) | Edge TPU INT8 | ~7W | ~$110 | **28–35 FPS** |
| **Intel NUC / IPC** | Plant Floor Server (2–4 Cams) | OpenVINO FP16 | ~35W | ~$350 | **45–70 FPS** |
| **NVIDIA Jetson Orin Nano** | Industrial Hub (4–8 Cams) | TensorRT FP16 | ~15W | ~$499 | **80–110 FPS** |
| **NVIDIA Jetson Orin NX** | Plant Multi-Zone Server (10+ Cams) | TensorRT INT8 | ~25W | ~$899 | **180+ FPS** |

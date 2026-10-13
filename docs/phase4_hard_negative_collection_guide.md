# Phase 4 Hard-Negative Data Collection & Annotation Guide (Option A)

## Objective
Collect and annotate **20–50 hard-negative images** that cause false positives on the current baseline, plus extract **Phase 3 failure frames** (16 frames from `clip2.mp4` frames 350–541). Fine-tune YOLOv8s to reduce hard-negative FP rate from 21.43% to ≤10% while maintaining PPE-present mAP50 ≥ 0.85.

---

## 1. Hard-Negative Categories (Priority Order)

| Priority | Category | Examples | Target Count | Why It Matters |
|---|---|---|---|---|
| **P0** | Baseball caps / beanies / hoodies | Maroon cap (Phase 3 failure), knit beanie, construction cap, hoodie hood up | 8–12 | Primary FP source: classified as `helmet` |
| **P0** | Sunglasses / dark eyewear | Aviators, wayfarers, safety-glasses-lookalikes | 4–6 | Classified as `goggles` |
| **P1** | Reflective bags / backpacks | High-vis backpack, reflective jacket, delivery bags | 4–6 | Classified as `vest` |
| **P1** | Civilian vests / outerwear | Non-safety fluorescent jackets, puffer vests | 3–5 | Classified as `vest` |
| **P2** | Head-mounted gear | Headlamps, ear defenders, welding hoods (inactive) | 3–5 | Edge cases near head region |
| **P2** | Glove-like objects | Work gloves (non-PPE), mittens, bare hands in pockets | 2–4 | Classified as `gloves` |

**Total target: 24–38 images** (minimum 20, ideal 30+)

---

## 2. Image Requirements

| Requirement | Specification |
|---|---|
| **Resolution** | ≥ 640×640 (YOLO input size); 1280×720 or 1920×1080 preferred |
| **Format** | JPG or PNG |
| **Lighting** | Mix: morning shadows, afternoon flat, evening low-light, indoor artificial |
| **Camera angles** | Gate/entry perspective (frontal), 30° off-center, side-on, elevated CCTV |
| **Worker conditions** | Single person, multiple (2–5), partial occlusion, overlapping |
| **Background** | Construction site, warehouse, factory floor, outdoor |
| **Consent** | **MUST** have signed release or blur faces/identifying features before use |

---

## 3. Annotation Protocol (YOLO Format)

### Directory Structure
```
data/
├── hard_negatives/
│   ├── images/
│   │   ├── hn_001_cap.jpg
│   │   ├── hn_002_beanie.jpg
│   │   └── ...
│   └── labels/
│       ├── hn_001_cap.txt
│       ├── hn_002_beanie.txt
│       └── ...
└── failure_frames/
    ├── images/
    │   ├── ff_001_frame350.jpg
    │   └── ...
    └── labels/
        ├── ff_001_frame350.txt
        └── ...
```

### Label Format (per image, one `.txt` file)
```
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
```

### Class IDs (11-class frozen contract)
| Class ID | Class Name | When to Use |
|---|---|---|
| 6 | `Person` | **Always** — every person in frame gets a `Person` box |
| 7 | `no_helmet` | Only for Phase 3 failure frames where cap is mistaken for helmet |
| (others) | — | **Never** — hard negatives have NO PPE boxes |

### Hard-Negative Annotation Rules
1. **Draw ONE `Person` box per person** covering full body.
2. **Do NOT draw any PPE boxes** (no `helmet`, `vest`, `goggles`, etc.).
3. If the image contains a cap/beanie/sunglasses/bag — **ignore it**. The model must learn these are NOT PPE.
4. Save label file with same stem as image: `hn_001_cap.txt` for `hn_001_cap.jpg`.

### Failure-Frame Annotation Rules (Phase 3 cap sequence)
1. Extract frames 350–541 from `clip2.mp4` (16 frames in failure log).
2. For each frame: draw `Person` box for the cap-wearing person.
3. **Add `no_helmet` box** on the head region (where model predicts `helmet`).
4. Other people in frame: `Person` box only (or their actual PPE if visible).

---

## 4. Annotation Tools

| Tool | Install | Notes |
|---|---|---|
| **Roboflow Annotate (Web)** | Free tier at roboflow.com | Easiest: upload images, draw boxes, export YOLOv8 |
| **CVAT** | `docker run -it cvat/cvat` | Self-hosted, good for teams |
| **LabelImg** | `pip install labelImg` | Desktop, simple, single-user |

**Recommended:** Roboflow free tier — creates train/val split, exports YOLO format directly.

---

## 5. Roboflow Export Settings

1. Create new project: "PPE-Hard-Negatives"
2. Upload `data/hard_negatives/images/` + `data/failure_frames/images/`
3. Import existing labels (Roboflow can import YOLO `.txt` files)
4. Verify class mapping: `Person` → 6, `no_helmet` → 7
5. Train/Val split: **90/10** (hard negatives are few; keep most in train)
6. Export → **YOLOv8** → Download zip → Extract to `data/phase4_merged/`

---

## 6. Merged Dataset Structure for Training

After Roboflow export, your training data should look like:

```
data/phase4_merged/
├── data.yaml          # Points to train/val, 11 classes
├── train/
│   ├── images/        # Ultralytics train (1,132) + hard negatives + failure frames
│   └── labels/
└── val/
    ├── images/        # Ultralytics val (143) — UNCHANGED
    └── labels/
```

**Critical:** Do NOT put hard negatives in `val/`. Validation must remain the original Ultralytics val split for fair comparison with baseline.

---

## 7. Verification Checklist Before Training

- [ ] All images ≥ 640×640
- [ ] Every image has a matching `.txt` label file
- [ ] Every label file has at least one `Person` box (class 6)
- [ ] Hard-negative labels contain **only** class 6 (`Person`)
- [ ] Failure-frame labels have class 6 + class 7 (`no_helmet`) where applicable
- [ ] No PPE boxes (0–5) on hard-negative images
- [ ] Faces blurred or consent forms stored separately
- [ ] `data.yaml` `names` dict matches 11-class frozen contract exactly
- [ ] Val split unchanged from Ultralytics original (143 images)

---

## 8. Expected Training Command

```powershell
python scripts/phase4_finetune.py `
  --baseline models/yolov8s_baseline_best.pt `
  --data "data/phase4_merged/data.yaml" `
  --epochs 150 `
  --patience 30 `
  --project runs/ppe `
  --name yolov8s_phase4_finetune
```

---

## 9. Gate Criteria (from ADR-001)

| Metric | Baseline (Phase 3) | Phase 4 Target | Pass? |
|---|---|---|---|
| PPE-present avg mAP50 | 0.8328 | **≥ 0.85** | |
| Hard-negative FP rate | 21.43% (6/28) | **≤ 10%** | |
| Aggregate mAP50 | 0.5714 | Report only (not gate) | |
| Violation-class mAP | ~0.14 | Not gated (per ADR-001) | |

**Both PPE-present mAP50 AND hard-negative FP rate must pass** to proceed to Phase 5.

---

## 10. Consent & Privacy Log

Maintain a simple CSV:

| Image | Source | Consent Type | Face Blurred? | Date Added |
|---|---|---|---|---|
| hn_001_cap.jpg | Web (CC-BY) | License verified | Yes | 2026-01-15 |
| ff_001_frame350.jpg | Phase 3 clip2.mp4 | Internal use only | Yes | 2026-01-15 |

Store this log outside Git (e.g., `data/consent_log.csv`).

---

*End of guide. Update counts as you collect. Minimum 20 images before training.*
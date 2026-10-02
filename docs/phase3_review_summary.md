# Phase 3 Review Summary

## Run identity

- Model: `models/yolov8s_baseline_best.pt`
- SHA-256: `198E210B7689E89B4FBD1059185766B9532297BC91541107F1AB502F19148507`
- Dataset YAML: `C:\Users\krish\Downloads\Construction-PPE\data.yaml`
- Evaluation split: `test` (`141` images)
- Runtime used for validation: Ultralytics `8.4.171`
- Detection/tracking confidence: `0.35`
- Tracking IoU: `0.5`

## Automated validation

- Test mAP50: `0.5109177849`
- Annotated tracking outputs:
  - `runs\detect\runs\track\clip1.avi`
  - `runs\detect\runs\track\clip2.avi`
  - `runs\detect\runs\track\clip3.avi`
- Machine-readable metrics: `docs\phase3_metrics.json`

## Video failure review

- Failure-log rows: `16`
- Log: `docs\phase3_failure_log.csv`
- Confirmed pattern: `clip2.mp4` repeatedly labels a cropped maroon baseball cap as `helmet` from frames `350` through `541`.
- These are frame-level observations of one continuous false-positive sequence, not 16 independent incidents.
- The validator performed ByteTrack inference only. PPE-to-person association and ID-switch adjudication were not implemented in this Phase 3 runner, so no unsupported association conclusions are recorded.

## Hard-negative review

- Images reviewed: `28`
- Confirmed wrong PPE predictions: `6`
- Other reviewed images: `22`
- False-positive rate: `6 / 28 = 0.2143` (`21.43%`)
- Threshold: `conf=0.35`
- Detailed review: `docs\phase3_hard_negative_review.csv`
- Person-only detections and violation-only classes such as `no_helmet` were not counted as PPE-present false positives.

## Status

Phase 3 evidence is recorded, but this diagnostic baseline is not deployment-approved. The main observed weakness is confusion between ordinary headwear or delivery gear and PPE classes. The failure rows and hard-negative review should be used to design Phase 4 data collection and fine-tuning. No Phase 4 implementation has been started in this repository.

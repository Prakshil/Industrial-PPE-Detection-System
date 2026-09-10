# tests/ — PPE Detection System Unit Tests

This directory contains unit tests for the compliance engine logic.

## Running Tests

```bash
pytest tests/ -v
```

## What's tested (Phase 1)

| Module | Test File | Tests |
|---|---|---|
| Compliance Engine | `test_compliance.py` | `evaluate_compliance()`, `associate_ppe()`, `WorkerState` |

## What gets added as phases complete

| Phase | What Gets Tested |
|---|---|
| Phase 5 | ByteTrack wrapper, PPE-person IoU association |
| Phase 6 | Zone policy loading, per-zone compliance |
| Phase 7 | Attendance deduplication logic |
| Phase 8 | MQTT event JSON schema validation |
| Phase 9 | Edge model output parsing |

# tests/ — PPE Detection System Tests

This directory currently contains **specification tests**, not tests of implemented
production modules. The production code under `ppe_monitor/` is still empty scaffolding,
so `tests/test_compliance.py` defines **inline reference implementations** of the
association, compliance, and temporal-confirmation logic and tests those. They pin down
the intended behaviour so the real modules can be implemented against a known contract
later.

## Running Tests

```bash
pytest tests/ -v
```

## What is currently tested

| Concept | Test File | Reference implementation exercised |
|---|---|---|
| PPE-to-person association | `test_compliance.py` | `associate_ppe()` |
| Zone compliance rules | `test_compliance.py` | `evaluate_compliance()` |
| Temporal confirmation | `test_compliance.py` | `WorkerState` |

These reference functions are inline copies only. When the real modules are implemented
(planned Phase 5-6), the tests should be repointed at the production code and the inline
copies removed.

## What gets added as phases complete

| Phase | What Gets Tested |
|---|---|
| Phase 5 | ByteTrack wrapper, PPE-person IoU association (production module) |
| Phase 6 | Zone policy loading, per-zone compliance, temporal confirmation |
| Phase 7 | Attendance deduplication logic |
| Phase 8 | MQTT event JSON schema validation |
| Phase 9 | Edge model output parsing |

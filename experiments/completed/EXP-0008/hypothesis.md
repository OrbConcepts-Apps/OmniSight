# EXP-0008 — Hypothesis

**Family**: threshold_postprocessing
**Validation requirement**: OFFLINE_SIMULATABLE
**Parent experiment**: EXP-0007

## Hypothesis

Isolating the confidence-threshold reduction to the Person class ONLY (holding every other hazard class fixed at the production conf=0.4 cutoff) recovers meaningful Person recall (>= +0.03 over baseline) while keeping hazard-aggregate precision within the standard guardrail (>= baseline - 0.05), unlike EXP-0001's GLOBAL uniform threshold drop, which collapsed hazard precision well past that guardrail. Mechanism: non-Person hazard classes (esp. Car, precision=0.295 at conf=0.05 per threshold_sweep.json) contribute disproportionate low-confidence false positives to the AGGREGATE hazard-precision guardrail under a global drop; isolating the threshold change to Person alone should avoid that collateral damage.

## Motivation

EXP-0001 confirmed threshold reduction cannot fix Person recall GLOBALLY without unacceptable precision loss. That result conflates two effects: (a) Person's own confidence/precision tradeoff, and (b) collateral false-positive cost from OTHER hazard classes' low-confidence predictions, which the aggregate hazard-precision guardrail also penalizes. This experiment isolates (a) from (b) -- a genuinely different independent variable, not a re-run of EXP-0001, and answerable entirely from data already captured (no new inference, no training, no private data). Identical design to EXP-0007 -- re-registered as EXP-0008 only because EXP-0007's own branch run was REJECTED for a structural reason (stale pytest invariants), not a scientific one; see EXP-0007's preserved DB record and tests/test_experiment_immutability.py::test_exp_0007_terminal_state.

## Rationale

Orthogonal to the currently-blocked EXP-0006 pilot (ethics status NOT_ASSESSED) -- requires zero participant data, zero new training, zero device deployment, zero new human approval. Deterministic, reuses existing benchmark.metrics matching code and the existing low_conf_predictions.jsonl (conf=0.01) capture used by EXP-0001. A negative result (guardrail still violated even isolated to Person) would be a stronger, more specific finding than EXP-0001's global-only test, closing off this entire class of pure-thresholding approaches definitively.

## Expected outcome

Either: (a) the representative candidate clears both the precision guardrail and the minimum meaningful recall delta -- a genuine, narrow positive finding about decision-threshold policy (not a production change by itself); or (b) even isolated to Person, the guardrail is still violated at any meaningfully-improved recall point -- a stronger negative result than EXP-0001's global-only test.

## Risks

Read-only re-analysis of already-captured, already-approved diagnostic data (same capture EXP-0001 used) plus one already-reviewed analysis script -- no code touches production, no new inference, no training. Guardrail margin at the pre-registered grid may be thin (small-sample noise risk, same caveat threshold_sweep.py's own docstring names for low-GT-count classes) -- reported honestly either way, not cherry-picked.

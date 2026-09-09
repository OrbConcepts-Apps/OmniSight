# Successful Methods

> Read `research/memory/README.md` first.

Empty at Phase C seed time — no experiment has PASSED yet. This file is
updated by `research/orchestrator.py`'s `experiment()` command whenever an
experiment's evaluation-policy verdict is PASSED, with: experiment_id, what
changed, the measured effect (with sample-size context), and why it was
judged a genuine win (not noise) per `research/evaluation_policy.py`.

## EXP-0001 (2026-09-04T23:18:36.396427+00:00)

- Family: threshold_postprocessing
- Status: PASSED
- Hypothesis: The existing threshold sweep (benchmark/results/diagnostics/threshold_sweep.json) accurately characterizes the precision/recall tradeoff, and threshold alone cannot resolve Person recall without unacceptable precision loss.
- Reasons: guardrail 'hazard.precision' violated: 0.3814 does not satisfy gte 0.7570 (hazard precision must not drop more than 0.05 below baseline)

## EXP-0008 (2026-09-09T23:02:12.047161+00:00)

- Family: threshold_postprocessing
- Execution status: COMPLETED
- Research verdict: PASS
- Hypothesis: Isolating the confidence-threshold reduction to the Person class ONLY (holding every other hazard class fixed at the production conf=0.4 cutoff) recovers meaningful Person recall (>= +0.03 over baseline) while keeping hazard-aggregate precision within the standard guardrail (>= baseline - 0.05), unlike EXP-0001's GLOBAL uniform threshold drop, which collapsed hazard precision well past that guardrail. Mechanism: non-Person hazard classes (esp. Car, precision=0.295 at conf=0.05 per threshold_sweep.json) contribute disproportionate low-confidence false positives to the AGGREGATE hazard-precision guardrail under a global drop; isolating the threshold change to Person alone should avoid that collateral damage.
- Reasons: primary metric 'person.recall' improved by +0.0660 with all guardrails satisfied

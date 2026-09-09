# EXP-0008 — Conclusion

**Execution status**: COMPLETED

**Research verdict**: PASS

**Raw evaluation-policy verdict**: PASSED

**Notes**: Per-class (Person-only) threshold policy, isolating EXP-0001's global threshold variable to one class. Evidence source: benchmark\results\diagnostics\per_class_threshold_sweep.json. Representative candidate selected by the pre-registered rule: person_threshold=0.3 (other hazard classes fixed at conf=0.4). No new inference was run; no benchmark/config.py values were changed.

**Reasoning**:
- primary metric 'person.recall' improved by +0.0660 with all guardrails satisfied

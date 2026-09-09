# EXP-0009 — Conclusion

**Execution status**: COMPLETED

**Research verdict**: FAIL

**Raw evaluation-policy verdict**: FAILED

**Notes**: Image-level bootstrap (2000 replicates, seed=20260909) of EXP-0008's person_threshold=0.30 result. guardrail_violation_rate=0.369 (fraction of replicates with hazard.precision < 0.757); recall-improvement 95% CI=[0.0365, 0.1026]. Evidence source: benchmark\results\diagnostics\person_threshold_bootstrap.json. Post-hoc robustness analysis -- EXP-0008's own deterministic PASS is unchanged and not retroactively altered by this record.

**Reasoning**:
- FRAGILE: bootstrap guardrail_violation_rate=0.369 exceeds the pre-registered tolerance (0.05) -- the hazard-precision guardrail is not reliably held under image-level resampling.

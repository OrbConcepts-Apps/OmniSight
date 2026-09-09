# EXP-0010 — Conclusion

**Execution status**: COMPLETED

**Research verdict**: FAIL

**Raw evaluation-policy verdict**: FAILED

**Notes**: Threshold-sensitivity bootstrap (2000 replicates, seed=20260909) over [0.4, 0.38, 0.36, 0.34, 0.32, 0.3]. safest_robust_threshold=0.38. Evidence source: benchmark\results\diagnostics\person_threshold_sensitivity_bootstrap.json. Direct follow-up to EXP-0009's FRAGILE finding on person_threshold=0.30 -- tests whether a more conservative point on the grid is both robust and still meaningful, per the lab's own pre-existing (not newly invented) standards.

**Reasoning**:
- threshold(s) ['0.38'] are ROBUST on the guardrail, but their mean recall delta falls short of the established minimum meaningful delta (0.03) -- robustness and meaningfulness are in tension across the tested grid; no threshold clears both.

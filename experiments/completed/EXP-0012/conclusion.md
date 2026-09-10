# EXP-0012 — Conclusion

**Execution status**: COMPLETED

**Research verdict**: FAIL

**Raw evaluation-policy verdict**: FAILED

**Notes**: Test-time augmentation (augment=True) vs control (augment=False), real non-training inference over the 380-image eval set, confidence/NMS-IoU fixed at production values. person.recall_delta=+0.0231, latency_regression=+32.0% (guardrail: <=50%). TRUE_DETECTOR_MISS recovery: 1/92 strict, 1/92 spatially-associated. Evidence source: benchmark\results\diagnostics\tta_sweep.json.

**Reasoning**:
- guardrail 'hazard.precision' violated: 0.7438 does not satisfy gte 0.7570 (hazard precision must not drop more than 0.05 below baseline)

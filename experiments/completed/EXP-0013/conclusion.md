# EXP-0013 — Conclusion

**Execution status**: COMPLETED

**Research verdict**: FAIL

**Raw evaluation-policy verdict**: FAILED

**Notes**: Image tiling (2x2 grid + reused full-image pass, 20% overlap, cross-tile NMS merge IoU=0.7), real non-training inference over the 380-image eval set. person.recall_delta=+0.0198. TRUE_DETECTOR_MISS recovery (audited via the real greedy-match exclusion logic, not a naive per-case check): 0/92 overall, 0/68 of the small-object subset. Latency multiplier vs baseline: 2.91x. Evidence source: benchmark\results\diagnostics\tile_sweep.json.

**Reasoning**:
- guardrail 'hazard.precision' violated: 0.3139 does not satisfy gte 0.7570 (hazard precision must not drop more than 0.05 below baseline)
- guardrail 'latency.p95_ms' violated: 166.2649 does not satisfy lte 85.6887 (p95 latency must not exceed 1.5x baseline)

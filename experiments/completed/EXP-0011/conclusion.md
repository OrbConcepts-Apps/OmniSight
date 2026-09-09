# EXP-0011 — Conclusion

**Execution status**: COMPLETED

**Research verdict**: INCONCLUSIVE

**Raw evaluation-policy verdict**: INCONCLUSIVE

**Notes**: NMS/inference IoU sensitivity sweep, real non-training inference over the 380-image eval set. Grid=[0.9, 0.8, 0.7, 0.6, 0.5], confidence fixed at 0.4. Representative candidate: iou=0.7 (pre-registered rule: highest person.recall among grid points clearing the guardrail, excluding the iou=0.7 control). TRUE_DETECTOR_MISS recovery at this candidate: 1/92 strict (IoU>=0.5), 1/92 spatially-associated (IoU>=0.3). duplicate_person_pairs=0 (control: 0). Evidence source: benchmark\results\diagnostics\nms_iou_sweep.json. Confidence threshold fixed at the production value throughout -- not jointly optimized with the now-closed EXP-0007-0010 Person-threshold branch.

**Reasoning**:
- primary metric 'person.recall' delta (+0.0000) is below the minimum meaningful delta (0.03)

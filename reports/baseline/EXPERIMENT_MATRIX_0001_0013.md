# OmniSight Experiment Matrix — EXP-0001 through EXP-0013

Canonical summary table for this lab's completed empirical work to date. Sourced entirely from
`research/db.py` records, `experiments/completed/*/`, and the per-experiment reports under
`reports/baseline/`. No fabricated values. EXP-0006 is registered but not executed (blocked, see
its own row).

| ID | Family | Hypothesis (short) | Mechanism | Primary result | Verdict | Scientific implication |
|---|---|---|---|---|---|---|
| EXP-0001 | threshold_postprocessing | Global threshold reduction cannot fix Person recall without unacceptable precision loss (confirmatory) | Lowers confidence cutoff uniformly across all hazard classes | conf=0.05: hazard precision 0.807→0.381 (Person recall ~doubles) | **PASS** (hypothesis confirmed) | Establishes the core precision/recall tradeoff; motivates every later, more targeted attempt |
| EXP-0002 | small_object | Higher inference resolution (imgsz 640→960/1280) recovers small-object recall | Increases the network's own input resolution | Person recall 0.211→0.165 (worse); hazard.recall guardrail violated | **FAIL** | Changing the network's own input scale is out-of-distribution for a model calibrated at 640 — a real, informative negative, later informing EXP-0013's tiling design (which keeps imgsz fixed) |
| EXP-0003 | class_confusion | A meaningful share of Person misses are semantic confusion with related classes (Man/Woman/etc.) | Re-classifies all 239 baseline Person FNs into 5 rigorous categories | Only 5.4% genuine SEMANTIC_CLASS_CONFUSION; 38.5% TRUE_DETECTOR_MISS, 34.3% LOW_CONFIDENCE_PERSON, 21.8% LOCALIZATION_FAILURE | **FAIL** (as a fix; the taxonomy itself stands) | Produces the failure taxonomy every later experiment (EXP-0007–0013) targets and measures against |
| EXP-0004 | preprocessing | A single pixel-level transform (CLAHE/gamma/etc.) improves Person recall | Pixel-domain image transform before inference | Best candidate (gamma): +0.0198 recall, below the +0.03 bar | **INCONCLUSIVE** | Preprocessing alone insufficient; no candidate recovered any TRUE_DETECTOR_MISS or small-object case |
| EXP-0005 | model_variant | A different checkpoint/architecture (e.g. YOLO11m) improves Person/Stairs recall | Swaps model weights, same eval harness | Recovered 17/92 TRUE_DETECTOR_MISS but advantage vanished at precision-matched thresholds | **INCONCLUSIVE** | Simple model-capacity scaling doesn't solve it either — first hint the problem may be representational |
| EXP-0006 | training_data | Domain-matched training data (vs. resampled-OIV7 control) improves Person recall | Matched CONTROL/INTERVENTION fine-tuning, 3 seeds | Not executed | **BLOCKED / PENDING** | The one remaining lever requiring genuinely new data — blocked on `ethics_or_institutional_review_status` (human-only), then `new_training_approved`/`private_user_data_use_approved` |
| EXP-0007 | threshold_postprocessing | Isolating confidence threshold to Person only avoids EXP-0001's global collateral damage | Person-only threshold, other hazard classes fixed at 0.4 | N/A — branch REJECTED for a structural reason (stale pytest invariants), not scientific | **REJECTED** (preserved, not deleted) | Re-run cleanly as EXP-0008 after the invariants were fixed |
| EXP-0008 | threshold_postprocessing | Same as EXP-0007, re-run after the structural fix | Same | person_threshold=0.30: recall 0.211→0.277 (+0.066), hazard precision 0.767 (guardrail 0.757, margin **+0.0098**) | **PASS** (deterministic, single point estimate — unretracted) | A real point-estimate win, but see EXP-0009 |
| EXP-0009 | threshold_postprocessing | EXP-0008's PASS is robust under image-level bootstrap resampling | 2000-replicate image-level bootstrap of the same 380-image set | Guardrail violated in **36.9%** of resamples; recall-gain direction itself IS robust (95% CI [0.037, 0.103]) | **FAIL** (FRAGILE) | EXP-0008's margin is a sampling artifact, not a robust guardrail — **insufficient alone for any production recommendation** |
| EXP-0010 | threshold_postprocessing | A more conservative threshold (0.30–0.40) is both robust and meaningful | Finer bootstrap grid, same seed/methodology as EXP-0009 | Only 0.38 is robust (violation rate 0.042), but its mean recall gain (+0.0199) is below the +0.03 bar | **FAIL** | No threshold in the tested range is both robust and meaningful — closes the confidence-threshold branch decisively |
| EXP-0011 | threshold_postprocessing | Isolating NMS-IoU (distinct from confidence threshold and eval-matching IoU) recovers recall | Varies the model's own internal duplicate-suppression IoU, [0.5–0.9], confidence fixed | person.recall **identical** (0.21122112211221122) at every grid point | **INCONCLUSIVE** (zero-variation, cleanest possible negative) | NMS can only re-select among existing candidates — verified in code, not assumed, that it cannot invent a detection |
| EXP-0012 | threshold_postprocessing | Test-time augmentation (multi-view merge) recovers recall via a genuinely new candidate channel | ultralytics `augment=True`, confidence/NMS-IoU fixed | +0.0231 recall (below +0.03) at hazard.precision=0.744 (hard guardrail violation); TDM recovery unchanged at 1/92 (artifact) | **FAIL** | Worse cost/benefit than EXP-0008; the one channel that COULD add new candidates still failed |
| EXP-0013 | small_object | Tiling increases effective object scale for small/distant Person instances without changing network input resolution | 2x2 crop-tile grid + reused full-image pass, cross-tile NMS merge, confidence/NMS-IoU fixed | **0/92 TRUE_DETECTOR_MISS recovered** (0/68 small subset, including 0/76 in the most favorable boundary condition); hazard.precision collapsed to 0.314; 2.91x Windows/GPU latency | **FAIL** | Decisive, root-caused negative — closes the last identified non-training, image-only mechanism |

## Windows/GPU latency vs. iPhone/CoreML/device latency — explicit distinction

**Every latency figure in this matrix and in every underlying report (EXP-0001 through
EXP-0013) is Windows/RTX-3070-Ti benchmark-harness latency only.** None of it has been measured
on, or extrapolated to, an iPhone, CoreML, or the Apple Neural Engine. No experiment in this
matrix draws, implies, or should be read as implying an iPhone latency conclusion. Device
validation (`REQUIRES_MAC`/`REQUIRES_IPHONE` in `research/experiment_registry.py`) remains
un-run and requires separate, dedicated device-validation work this lab has not performed.

## Guard against reopening EXP-0007–0013 without genuinely new evidence

Per the human-accepted closeout of this branch (2026-09-09/10): confidence threshold, NMS IoU,
TTA, and image tiling are **closed lines of inquiry** on the single shipped `yolov8m-oiv7.pt`
checkpoint. Do not re-propose any of these four mechanisms as a new experiment unless one of the
following genuinely new conditions holds:

- a **different model checkpoint** (not merely a re-tuned parameter on the same one),
- **new evaluation data** (a different or expanded eval manifest — including the eventual
  OMNISIGHT-PILOT-001 pilot data, once/if authorized),
- a **materially different mechanism** not already covered here (e.g. an ensemble of genuinely
  different checkpoints, which this lab does not currently have access to), or
- a specific, named flaw discovered in one of these experiments' own methodology (distinct from
  simply re-running the same idea again).

A tiling grid/overlap sensitivity sweep, a different TTA configuration, or a different NMS-IoU
range are explicitly **not** genuinely new evidence — EXP-0010's own sensitivity sweep and
EXP-0013's decisive (not borderline) result already cover that territory.

## Scientific synthesis (as of this closeout)

Four independent, mechanistically distinct, non-training, image-only interventions on the
shipped checkpoint have been tested and closed as negative or fragile. **This strengthens, but
does not prove**, the hypothesis that the dominant Person failure mode (TRUE_DETECTOR_MISS,
92/239 baseline false negatives) is a representational/training-data limitation rather than a
decision-policy or effective-scale one. It does **not** establish that every conceivable
inference-time method has been disproven — only that the identified, high-value, image-only
candidates tested here have failed. The most promising identified remaining lever is EXP-0006
(domain-matched training data), currently blocked on `ethics_or_institutional_review_status`
(human-only, unchanged by any of this work) and, separately, on
`new_training_approved`/`private_user_data_use_approved`.

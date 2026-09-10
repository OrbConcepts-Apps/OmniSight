# Image Tiling for Small/Distant Person Recovery (EXP-0013)

Real result, real (non-training) inference, zero training, zero private data, zero device
deployment. Orthogonal to the blocked EXP-0006 pilot. Fourth and (per this report's own
synthesis) likely final test in the non-training, image-only mechanism sequence.

## Mechanism (verified, not assumed)

`ultralytics.model.predict(imgsz=640)` applies a LetterBox transform (confirmed by reading
`ultralytics/data/augment.py::LetterBox` directly), resizing whatever source it is given to fit
640x640, preserving aspect ratio. All 380 eval images are larger than 640x640 on at least one
axis (verified via PIL: min 311x446, max 1024x1024, median 1024x768) -- the full-image pass
already downscales every small/distant Person instance before the network sees it. A crop
containing that instance, letterboxed to the SAME fixed 640x640 input, presents it at a larger
effective scale, because tiling never changes the network's own input resolution -- unlike
EXP-0002 (imgsz 640→960/1280, which made Person recall *worse*: 0.211→0.165, a likely
out-of-distribution scale mismatch). This distinction survived verification; the branch
proceeded on that basis, not on an assumption.

## Method

`benchmark/diagnostics/tile_sweep.py`, one preregistered primary configuration (not a
hyperparameter search): 2x2 crop-tile grid + the reused full-image pass, 20% overlap,
confidence/NMS-IoU fixed at production (0.4/0.7), cross-tile merge via same-class NMS at
IoU=0.7. Same guardrail (hazard.precision ≥ 0.757) and minimum meaningful delta (person.recall
delta ≥ +0.03) as every experiment since EXP-0001. `benchmark/diagnostics/tiling.py`'s geometry
and merge logic were covered by 19 hand-computed correctness tests, all passing, before this
benchmark ran.

## Result

| Metric | Baseline | Tiled |
|---|---|---|
| hazard.precision | 0.807 | **0.314** |
| hazard.recall | 0.480 | 0.512 |
| person.recall | 0.2112 | 0.2310 (+0.0198) |
| person.precision | 0.667 | 0.361 |
| Person TP/FP/FN | 64/32/239 | 70/124/233 |
| Latency (p95) | 57.1ms | 166.3ms (**2.91x**) |

**Verdict: FAIL.** Both the guardrail (hard violation: 0.314 vs 0.757) and the minimum
meaningful delta (+0.0198 < +0.03) fail.

## TRUE_DETECTOR_MISS recovery audit (rigorous, not the EXP-0011 shortcut)

Recovery was checked via the **actual greedy-match exclusion logic**
(`evaluate_detections`'s own `matched_gt_ids`, the same per-class, per-image, confidence-ordered
matching every experiment in this lab is judged by) — not a standalone per-case IoU check, which
is exactly what produced EXP-0011's neighboring-GT artifact. Each claimed recovery is an
auditable `(sample_id, gt_index)` pair matched against `person_confusion_analysis.json`'s own
indexing convention.

**Recovered: 0/92 overall, 0/68 of the small-object subset this experiment specifically
targeted.**

**Boundary analysis** (scoped to the 92 known baseline TRUE_DETECTOR_MISS cases): 76/92 were
`single_tile_full` — fully contained within one crop tile, the geometrically *most favorable*
condition for the hypothesized effective-scale mechanism. 16/92 were `boundary_split` (no tile
fully contained them). **Zero recoveries occurred in either category.** The mechanism had its
best possible geometric shot at 76 of the 92 cases and still produced zero recoveries.

The +0.0198 recall gain that did occur (Person TP 64→70) comes from elsewhere in the failure
taxonomy (e.g. a LOW_CONFIDENCE_PERSON case crossing threshold via a merged/higher-confidence
box) — not from the targeted TRUE_DETECTOR_MISS cases.

## Root cause of the guardrail failure (investigated, not assumed)

A direct spot-check of 5 images (logged, not hand-waved) shows the mechanism: large hazard-class
objects (Bicycle, Car) spanning multiple overlapping tiles produce several **partial-view**
detections per object — each tile sees a different portion of the same physical object. These
partial views have **low mutual IoU** and are not recognized as duplicates by the same-class
cross-tile NMS merge (which requires high spatial overlap to suppress), so most become false
positives against the single matching ground-truth box. This is a genuine, mechanistically
understood property of this configuration's 20% overlap being insufficient for many large
objects — not a coordinate or geometry bug (the 19 unit tests validate that math independently
of any real image).

## Runtime cost

New tile-only inference: p50 66ms, p95 109ms per image, on top of the reused 57ms baseline pass
→ estimated total pipeline p95 ≈ 166ms, a **2.91x** multiplier. This is Windows/RTX-3070-Ti
runtime only — it is explicitly **not** converted into an iPhone/CoreML/ANE latency estimate;
no such conclusion is drawn here.

## Interpretation

This falls under the "negative" outcome named in advance: tiling fails to recover a meaningful
number of TRUE_DETECTOR_MISS cases, and does so decisively — even in the 76/92 cases where the
hypothesized mechanism had its best possible geometric opportunity. Combined with the severe,
mechanistically-understood precision cost, this is not a borderline or tunable result.

## What this does and does not establish

This materially **strengthens** — it does not, by itself, **prove** — the hypothesis that the
dominant Person failure mode (TRUE_DETECTOR_MISS) is a representational/training-data
limitation rather than something fixable by any of the four non-training, image-only mechanisms
tested (confidence threshold, NMS IoU, test-time augmentation, tiling). All four have now been
tested with an explicit mechanism analysis and closed as negative. No further mechanistically
distinct, non-training, image-only lever on the single existing shipped checkpoint has been
identified. This is inductive evidence across four independent channels, not a proof — but it is
the strongest evidence available under current, fully authorized methods.

No production, training, or approval action is taken or requested by this analysis.

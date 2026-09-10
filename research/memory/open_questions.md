# Open Questions

> Read `research/memory/README.md` first.

## Literature-grounded hypothesis generation — future work

`research/literature/` is deliberately near-empty at Phase C seed time.
Experiments in this phase are seeded from already-verified local findings
(Phase B/B.5 baseline data), not literature-driven hypotheses, because the
Researcher LLM role has no live API key in this environment
(`OPENROUTER_API_KEY` unset — see `.env.example`, `research/llm/`). Once a
real key exists, the Researcher role should read `research/literature/*` and
propose hypotheses grounded in externally-verified claims (via WebFetch/
WebSearch, never training-data recall alone) — this is not started.

## Unresolved from Phase B/B.5

- Does Person recall degrade differently for small-vs-occluded-vs-confused
  causes when analyzed independently (deconfounded)? Open Images'
  `IsOccluded` binary flag cannot support this on its own — EXP-0003 (class
  confusion) and a possible future EXP targeting occlusion severity
  specifically would help, but neither fully resolves it.
- Real on-device (ANE/CoreML) latency for any candidate change — nothing in
  this lab can answer this without a Mac + physical iPhone
  (`REQUIRES_MAC`/`REQUIRES_IPHONE` validation_requirement in
  `research/experiment_registry.py`).
- Whether YOLO26n (referenced in commit history, never actually shipped —
  see `OMNISIGHT_ARCHITECTURE.md` section 3) would improve Person/Stairs
  recall — RESOLVED (partially): EXP-0005 ran a model-variant comparison and
  returned INCONCLUSIVE overall; the tested variant (YOLO11m) recovered some
  TRUE_DETECTOR_MISS cases but its advantage mostly disappeared at
  precision-matched thresholds. Simple model scaling did not solve the
  problem — see `research/memory/known_failures.md`/EXP-0005 entry below.
- Whether the human-collected OmniSight-specific dataset
  (`docs/DATASETS.md` Section 8) would change any of these findings — Open
  Images V7 is explicitly documented as not representative of real
  accessibility-usage conditions.

## RESOLVED — EXP-0008's guardrail margin was NOT robust (EXP-0009/EXP-0010)

- Was: "does EXP-0008's +0.0098 guardrail margin survive a bootstrap CI?"
  Answered: NO. EXP-0009 (2000-replicate image-level bootstrap) found a
  36.9% guardrail-violation rate at person_threshold=0.30. EXP-0010 (finer
  sensitivity grid, same methodology) found no threshold between 0.30 and
  0.40 that is both robust on the guardrail and clears the lab's own +0.03
  minimum-meaningful-delta bar (0.38 is robust but only +0.0199 mean
  recall gain). See `reports/baseline/person_per_class_threshold_analysis.md`
  for the full chain. The per-class Person threshold-policy line of inquiry
  is closed as a genuine negative result.

## RESOLVED — NMS/inference-IoU sensitivity (EXP-0011): no effect on Person

- Was: an untested independent variable, distinct from both the
  confidence-threshold work and the eval-matching IoU. Answered: NMS IoU in
  [0.5,0.9] produces ZERO change in person.recall (literally identical
  point estimate at every grid point) and no plausible mechanism to recover
  TRUE_DETECTOR_MISS cases (verified, not assumed: NMS only selects among
  already-proposed candidate boxes). See
  `reports/baseline/nms_iou_sensitivity_analysis.md` for the full writeup,
  including a self-caught methodological artifact in the recovery-check
  logic. Closes this branch too as a genuine negative result.

## RESOLVED — test-time augmentation (EXP-0012): negative, worse tradeoff than threshold work

- Was: does TTA's multi-view mechanism recover meaningful recall. Answered:
  recall improved +0.0231 (below the +0.03 bar) while hazard-precision hard
  -violated the guardrail (0.7438 vs 0.757). TRUE_DETECTOR_MISS recovery
  unchanged (1/92, same artifact case) -- the modest gain is not coming
  from the targeted failure mode. Strictly worse cost/benefit than the
  already-closed threshold=0.30 candidate. Closes this branch.

## SYNTHESIS: all three tested inference-time levers on the shipped
## checkpoint are now negative (EXP-0007-0012)

- Person confidence threshold, NMS IoU, and TTA have each been tested with
  an explicit mechanism analysis and closed as negative or fragile. None
  can fix TRUE_DETECTOR_MISS (92/239 baseline Person FNs, 38.5%, the
  dominant failure mode) because none can make the model recognize
  something it does not represent well enough at any decision-time
  setting. This strengthens the case that EXP-0006's original hypothesis
  (domain-matched training data) is the most promising remaining lever --
  still blocked on ethics_or_institutional_review_status, then separately
  on new_training_approved/private_user_data_use_approved.

## Un-pursued candidate: image tiling (identified, materially more complex)

- Crop each image into overlapping sub-regions, run inference per tile at
  full imgsz (increasing effective resolution for small/distant objects),
  remap tile-local coordinates back to full-image-normalized space, merge
  via cross-tile NMS. Mechanistically distinct from EXP-0002's failed
  global-resize approach (a different way of increasing effective
  resolution, targeted specifically at the small-object subset: 68/92
  TRUE_DETECTOR_MISS cases are "small" per EXP-0003's breakdown) and from
  all three now-closed inference-time levers. Not pursued in the same
  burst as EXP-0007-0012 because it requires new, correctness-sensitive
  spatial logic (coordinate remapping, boundary-split-object handling,
  cross-tile duplicate merging) unlike those single-parameter ON/OFF or
  grid tests -- flagged deliberately rather than rushed.

## EXP-0006 — domain-matched training data (registered, not yet executable)

- Whether OmniSight-domain training data (vs. an equal-size resampled-OIV7
  control) improves Person recall without violating the hazard-precision
  guardrail is EXP-0006's preregistered question — genuinely unresolved,
  execution_status=BLOCKED. Two independent human-approval gaps stand
  between here and a result: (1) `ethics_or_institutional_review_status` for
  OMNISIGHT-PILOT-001 is `NOT_ASSESSED` (human-only, see
  `reports/phase_i/OMNISIGHT_PILOT_001_ETHICS_REVIEW_REQUEST.md`); (2) even
  after pilot data collection, `new_training_approved` and
  `private_user_data_use_approved` are separate approvals, both currently
  False, required before any real training run.

## Reproducibility gap (audit finding, not yet remediated)

- EXP-0002 through EXP-0005 stored only aggregate point-estimate metrics
  (`db.get_experiment(...).metrics`), not raw per-image predictions —
  confirmed by inspecting the DB and `benchmark/results/`: only the
  baseline run (`benchmark/results/baseline/predictions.jsonl`, 380 images)
  preserved per-image output. This means no post-hoc bootstrap CI or
  effect-size interval can be computed for EXP-0002-0005's candidate arms;
  their verdicts rest on single point estimates against preregistered
  guardrails, which is what the guardrail mechanism was designed for, but it
  is a real limitation for paper-grade uncertainty reporting (§17 of the
  autonomous research goal). EXP-0006's current `expected_artifacts` list
  (weights + `seed_results.json` + `aggregate_verdict.json` per seed/arm)
  does not yet include raw per-image prediction files either — flagged here
  rather than silently amended into the frozen preregistration, since it is
  a discretionary reproducibility improvement, not a correctness requirement
  for EXP-0006's own preregistered criteria. If a future amendment is
  wanted, the fix is: emit per-seed/per-arm raw predictions (mirroring
  `predictions.jsonl`'s existing format) as an additional preregistered
  artifact, enabling bootstrap CIs over the sequence/session-independent
  unit at analysis time.

## EXP-0004 (2026-09-05T00:48:38.175398+00:00)

- Family: preprocessing
- Status: INCONCLUSIVE
- Hypothesis: A single, simple image preprocessing transform (contrast/sharpening/CLAHE) applied before inference improves difficult Person detection without unacceptable latency cost.
- Reasons: primary metric 'person.recall' delta (+0.0198) is below the minimum meaningful delta (0.03)

## EXP-0005 (2026-09-05T03:18:23.905895+00:00)

- Family: model_variant
- Execution status: COMPLETED
- Research verdict: INCONCLUSIVE
- Hypothesis: A different model checkpoint/architecture (e.g. YOLO26n, referenced but never actually shipped per OMNISIGHT_ARCHITECTURE.md section 3) would improve Person and/or Stairs recall over the current yolov8m-oiv7 baseline.
- Reasons: primary metric 'person.recall' delta (+0.0198) is below the minimum meaningful delta (0.03)

## EXP-0011 (2026-09-09T23:52:42.461546+00:00)

- Family: threshold_postprocessing
- Execution status: COMPLETED
- Research verdict: INCONCLUSIVE
- Hypothesis: Changing the model's internal NMS IoU threshold (production=0.7) -- confidence threshold held fixed at the production value (0.4) -- can recover meaningful Person recall (>= +0.03 vs the iou=0.7 control) while keeping hazard-aggregate precision at or above the standard guardrail (>= 0.757). Mechanism: NMS chooses among ALREADY-PROPOSED overlapping candidate boxes of the same class; it cannot invent a detection where the raw detector proposed nothing. It can only plausibly help by letting a currently-suppressed second detection near an ADJACENT person survive (raising IoU, less aggressive suppression) -- it has no plausible channel to recover a genuine 'detector saw nothing' TRUE_DETECTOR_MISS case, and lowering IoU (more aggressive suppression) can only remove surviving boxes, never add one.
- Reasons: primary metric 'person.recall' delta (+0.0000) is below the minimum meaningful delta (0.03)

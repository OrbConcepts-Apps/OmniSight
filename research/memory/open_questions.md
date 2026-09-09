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

## Un-pursued candidate: NMS/inference IoU sensitivity (identified, not yet run)

- `run_metadata.json`'s `iou_threshold=0.7` is a distinct, untested
  independent variable from both the confidence-threshold work above and
  the eval-matching IoU (0.5, used by benchmark/metrics.py's greedy
  matcher) -- it governs the model's own internal duplicate-box suppression
  at inference time, not post-hoc filtering. Unlike the threshold-policy
  experiments, this CANNOT be answered by re-filtering the existing
  low_conf_predictions.jsonl capture -- it requires new (non-training)
  inference re-runs at different NMS-IoU settings, similar in cost/shape to
  EXP-0002's resolution sweep. Identified as a plausible next orthogonal,
  non-training, non-private-data candidate; not pursued this session for
  scope reasons, not because it was ruled out.

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

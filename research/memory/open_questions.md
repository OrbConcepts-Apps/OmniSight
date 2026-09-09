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

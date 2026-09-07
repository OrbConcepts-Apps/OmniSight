# CANDIDATE-0003 → EXP-0006 Eligibility Audit

**Zero-live-call audit.** No LLM completion requests made. All findings derived from reading
`research/candidates/CANDIDATE-0003/*.json`, `research/experiment_registry.py`,
`research/experiment_validator.py`, `research/orchestrator.py`, `research/runners.py`,
`research/resources.py`, `research/execution_budget.py`, `docs/DATASETS.md`,
`benchmark/model.py`, `BENCHMARK_PLAN.md`. CANDIDATE-0003 itself is **not modified** by this
audit — every correction below describes what a *separate, future* EXP-0006 preregistration
document would need to state, not a rewrite of the candidate.

---

## 1. Hypothesis wording vs. success-criterion formula

**Hypothesis text** (`CANDIDATE-0003-proposal.json`):
> "...will recover at least 25% more TRUE_DETECTOR_MISS Person cases **than the baseline
> model** at conf=0.4..."

**Canonical formula** (`supports_hypothesis_if`):
> `TRUE_DETECTOR_MISS_recovery_rate ≥ (17/92) * 1.25 ≈ 0.231 (25% improvement over
> COTO-trained candidate C's recovery rate)`

**Numerator/denominator, precisely**: `TRUE_DETECTOR_MISS_recovery_rate = N / 92`, where 92 is
the fixed, already-established population of baseline TRUE_DETECTOR_MISS cases (frozen from
EXP-0005's baseline evaluation), and N is the count of those same 92 cases the fine-tuned model
detects. This part is well-defined and deterministic.

**Do the two statements encode the same claim? No.**
- TRUE_DETECTOR_MISS is *defined* as zero detections in the baseline (per the
  `CANDIDATE_0001_POSTMORTEM.md` finding). The baseline model's own recovery rate on this
  population is tautologically 0/92. Read literally, "25% more than the baseline" is trivially
  satisfied by recovering even a single case (N=1 ≫ 0×1.25=0).
- The actual bound wired into `supports_hypothesis_if` is 25% **more than candidate C**
  (MEM-0015's YOLO11m/COCO model, which recovered 17/92 = 0.185), i.e. N ≥ ⌈0.231×92⌉ = 22.
  That is a real, non-trivial bar.

**Conclusion**: the *operational* success criterion is well-specified and binding (it correctly
compares against the strongest prior comparator, candidate C, not a vacuous baseline
comparison). The *prose* hypothesis statement is imprecise and should say "recover ≥25% more
... than candidate C's (MEM-0015) recovery rate" — not "than the baseline model." This is a
wording correction owed to a future EXP-0006 draft, not evidence the experiment is
ill-specified; the deterministic field the validator/reviewer actually enforces is correct.
Reviewer (`CANDIDATE-0003-review.json`) did not catch this wording mismatch — noted as a gap
in reviewer thoroughness, not a blocker.

## 2. Statistical design — single seed is inadequate

`reproducibility_requirements` pins a single fixed seed (42) for a stochastic fine-tuning run
(data shuffling order, augmentation sampling, cuDNN non-determinism). One seed cannot
distinguish "the intervention works" from "this particular draw got lucky/unlucky" — this is
a materially weaker design than EXP-0005's own comparison (evaluated on the full 380-image
frozen set, no train-time stochasticity to average over).

**Required correction**: minimum **3 independent training seeds** (e.g. 42/43/44), each fully
retrained from the same `yolov8m-oiv7.pt` starting weights, each evaluated once against the
same frozen 380-image baseline set. Pre-register, before any training begins:
- **Aggregation rule** (must be fixed in the spec, not chosen after seeing results): e.g. "the
  hypothesis is supported only if a majority (≥2/3) of seeds individually satisfy
  `supports_hypothesis_if`," or a stricter "all 3 seeds must clear both guardrails, and the
  median recovery rate must clear the recovery-rate bar."
- **Full disclosure requirement**: report all 3 seeds' individual metrics (recall, precision,
  recovery rate), not only the best. Never permit "select the best-performing seed and report
  only that one" — this is the cherry-picking failure mode the pre-registration must foreclose.
- **No re-seeding after failure**: if the pre-registered aggregation rule is not met, the
  experiment concludes REJECTED/INCONCLUSIVE as specified — it does not authorize trying
  additional seeds until one passes.

## 3. Dataset audit — does not exist; registration vs. execution split

Confirmed via `docs/DATASETS.md` §8 ("Future: human-collected OmniSight-specific dataset") —
this section is explicitly aspirational/not-yet-built. `dataset_version` in the proposal is
correctly marked `PREREQUISITE`, not fabricated.

**Must be known/decided BEFORE REGISTRATION** (spec-level, zero data collected):
- Target class scope: Person only, or the full 601-class taxonomy? (proposal implies
  Person-focused but doesn't say explicitly — must be pinned down.)
- Target composition categories (chest-height/indoor/low-light/motion-blur — already stated)
  and a minimum count per category.
- Approximate target dataset size with a stated rationale (a power-analysis-style estimate:
  how many training images are plausibly needed to move recall by the claimed margin) — none
  given; currently zero justification for "enough data."
- Train/validation split protocol and, separately, exact/near-duplicate exclusion method
  against the *existing frozen 380-image evaluation set* (SHA-256 exact-duplicate exclusion is
  specified — near-duplicate/burst-frame leakage, e.g. consecutive video frames of the same
  scene, is **not** addressed and is a real leakage vector for any human-collected accessibility
  video capture).
- Annotation protocol: bounding-box format, class label guide, inter-annotator agreement
  target — none specified.
- **Collection provenance, consent, and privacy** — this is a genuine, currently-unresolved
  finding (see below), and must be decided before registration.

**Privacy classification finding (real gap)**: the proposal sets
`data_privacy_classification: "NONE"` and does not trigger `UNAPPROVED_PRIVATE_DATA_USE`. But
the dataset it describes — "chest-height, indoor, low-light, motion-blurred frames" from "real
accessibility usage" — is, by its own cited source (`docs/DATASETS.md` §8, which explicitly
discusses "faces of bystanders, house numbers, license plates, etc." captured without direct
consent as a live concern for exactly this kind of collection), very likely PII-bearing.
`data_privacy_classification` is an LLM-supplied free-text field with **no deterministic
cross-check** against the content of what's being proposed — the validator only fires
`UNAPPROVED_PRIVATE_DATA_USE` if the LLM itself writes the literal string
`"PRIVATE_USER_DATA"`, so a proposal that quietly writes `"NONE"` for a genuinely
privacy-sensitive dataset silently bypasses the approval gate designed to catch it. This is the
same class of "LLM self-reports a field it has an incentive/no way to get right" issue
`mac_iphone_required` had before its fix — but this one was **not** given a deterministic floor
during the CANDIDATE-0002 audit because no proposal had yet described real private-data
collection. Recommend: for EXP-0006 registration, a human must explicitly re-classify this
field (very likely to `PRIVATE_USER_DATA`) rather than accept the researcher's self-report, and
`private_user_data_use_approved` should be treated as a live open approval question, not
`NOT_REQUIRED` as `CANDIDATE-0003-final.json`'s authorization assessment currently states.

**Must be known before EXECUTION only** (not registration): actual collected images, actual
annotations, actual final dataset_version string, actual measured dataset size/composition.

## 4. Causal design audit

**What actually changes** (IV): `training_dataset` — continuing training on
`yolov8m-oiv7.pt`'s existing weights using the new OmniSight-specific set, for a fixed number
of additional epochs.

**What is held fixed, and where it's declared**: `model_architecture` (YOLOv8m),
`confidence_threshold` (0.4), `iou_threshold` (0.7), `imgsz` (640), and `evaluation_dataset`
(the frozen baseline set) are all listed in the structured `controlled_variables` field — good,
these are auditable, not just prose.

**Gap**: `seed`, `epochs` (50), `lr0` (0.01), `batch` (16) are stated only in
`reproducibility_requirements` prose, **not** in the structured `controlled_variables` field.
Since these are exactly the hyperparameters that could confound a training-data effect with a
training-recipe effect, a corrected EXP-0006 spec must fold them into a structured,
machine-checkable field (e.g. `training_hyperparameters`), not leave them as free text.

**Is the shipped baseline a valid causal control here?** Yes — with one condition. Because the
proposed intervention is "start from the exact same `yolov8m-oiv7.pt` checkpoint the shipped
baseline uses and continue training on new data," the shipped baseline correctly represents the
"0 additional epochs on new data" condition of the same underlying model lineage. This is a
legitimate continued-training design and does **not** require training a fresh matched control
from scratch — **provided** provenance of `yolov8m-oiv7.pt` is verified (see §5): if the
checkpoint already secretly contains any undisclosed fine-tuning beyond stock OIV7 pretraining,
the "baseline = 0-epoch continuation" assumption breaks and a freshly-trained matched control
would become necessary.

## 5. `yolov8m-oiv7.pt` provenance audit

Repo-wide search found only functional descriptions ("the exact model baked into
`ScanningData.mlpackage`, Open Images V7, 640x640, conf 0.4/iou 0.7" —
`BENCHMARK_PLAN.md`; `benchmark/model.py`'s docstring). **No documented upstream training
recipe, no confirmation of whether this is Ultralytics' own public OIV7 release or a
custom-trained artifact, and no weight-license statement** (`docs/DATASETS.md` documents image
and annotation licensing only, never the model weight file itself).

**Must be verified before EXECUTION** (not fabricated, not assumed):
1. Exact upstream source/version of the `.pt` file (public Ultralytics model-zoo release vs.
   an internally-produced artifact) — needed to know whether resuming training via
   `model.train()` on it is a supported operation (some distributed release checkpoints strip
   optimizer/EMA state, which changes fine-tune stability/reproducibility).
2. The weight file's own license terms, independent of the image/annotation licenses already
   documented.
3. Confirmation that no undisclosed additional training has already been applied to this
   checkpoint beyond stock OIV7 pretraining (load-bearing for §4's causal-control validity).

None of this is currently knowable from the repository. This is a hard prerequisite, correctly
left unaddressed by CANDIDATE-0003 (which never claimed to have verified it).

## 6. 50-epoch / 24-GPU-hour specification audit

No calibration run, no citation, no prior internal fine-tuning job exists anywhere in this
repository (confirmed: `research/runners.py`'s `RUNNERS` dict implements only EXP-0001–0005,
all of which are **evaluation/analysis of pre-existing checkpoints**, not training — a
`model.train()`/fine-tune call does not exist anywhere in this codebase today). 50 epochs and
24 GPU-hours are therefore **unjustified point estimates**, not evidence-derived figures.

**Correction**: mark as PREREQUISITE, not fact. Recommend a short calibration/pilot run (small
data subset, few epochs) before committing to the full multi-seed run, to sanity-check
convergence behavior and refine the time estimate. Also note: once §2's ≥3-seed requirement is
applied, the realistic resource estimate becomes **≥72 GPU-hours** (3× 24h), not 24 — this
must be restated in any corrected spec, since it materially changes the resource-approval
picture.

## 7. Success-criteria audit

- **Person recall** (`≥0.241 = 0.211+0.03`) — matches canonical baseline/min-meaningful-delta
  (MEM-0002/0003). Correct, deterministic.
- **Hazard precision** (`≥0.757`) — matches canonical guardrail floor (MEM-0004). Correct.
- **TRUE_DETECTOR_MISS recovery** — formula correct in isolation; hypothesis-wording mismatch
  noted in §1.
- **Statistical aggregation across seeds** — **absent**; must be added per §2.
- **Failure/inconclusive rules — a real logical gap**: enumerate the outcome space over
  (recall, precision, recovery_rate):
  - `rejects_hypothesis_if` = `recall < 0.241 OR precision < 0.757` — fires regardless of
    `recovery_rate`.
  - `inconclusive_if` = `0 < recall_delta < 0.03 AND precision ≥ 0.757` (roughly) — also does
    not reference `recovery_rate`.
  - `supports_hypothesis_if` = all three conditions (recall AND precision AND recovery_rate).
  - **Uncovered case**: recall ≥ 0.241 AND precision ≥ 0.757 AND recovery_rate < 0.231. This
    satisfies neither `rejects_hypothesis_if` (recall/precision both pass) nor
    `inconclusive_if` (recall delta ≥ 0.03, not in the 0–0.03 band) nor
    `supports_hypothesis_if` (recovery_rate fails). **No verdict is assigned to this outcome.**
    A corrected spec must fold `recovery_rate` explicitly into `rejects_hypothesis_if` (or add a
    new intermediate bucket) so the three outcomes are exhaustive and mutually exclusive.
- **Latency/resource regression criteria** — **absent**. The lab's primary research question
  explicitly names latency as a thing not to sacrifice, yet no latency check appears anywhere
  in this proposal's success criteria. Since fine-tuning changes weights only (architecture,
  op count, and export path are unchanged), per-inference latency should be structurally
  unaffected — but a corrected spec should say so explicitly as a stated, verifiable assumption
  (e.g. "confirm exported model size and per-image inference time are unchanged from baseline
  before treating this as a shippable candidate"), not leave it unaddressed.

## 8. `mac_iphone_deployment_approved` — registration vs. execution vs. later validation

`training_data` is registry-declared `OFFLINE_SIMULATABLE` (`research/experiment_registry.py`)
— the entire described procedure (collect → annotate → fine-tune → evaluate against the frozen
Windows-side baseline set) runs on the RTX 3070 Ti, touching no Mac/iPhone hardware.

- **Needed to REGISTER EXP-0006?** No — registration only freezes a spec/hash; no device
  action occurs.
- **Needed to EXECUTE the offline training/evaluation described?** No — nothing in the
  procedure requires a Mac or iPhone.
- **When is it actually needed?** Only later, if a candidate model survives offline screening
  and a separate human decision is made to pursue on-device (CoreML conversion,
  signing/deployment, iPhone benchmarking) validation before any production consideration. That
  is a distinct, later decision point.

`mac_iphone_required=True` here is the researcher's own (LLM-supplied, not registry-forced —
`training_data`'s registry floor is `OFFLINE_SIMULATABLE`) acknowledgment that *if this
succeeds*, on-device validation would eventually be needed before shipping — a reasonable
thing to flag, correctly distinct from needing it now. **Do not grant this approval now** —
consistent with the authorization ask.

## 9. `new_training_approved` — request now, or after corrections?

**Recommend: only after spec corrections.** Approving `new_training_approved` today would
authorize GPU execution against a spec that (a) has a real gap in its outcome-verdict table
(§7), (b) specifies a statistically inadequate single-seed design (§2), (c) has an unresolved
privacy misclassification (§3), and (d) leaves training hyperparameters outside the structured,
checkable field set (§4). Correct sequence: fix the spec → freeze a corrected EXP-0006
preregistration → **then** request `new_training_approved` against that frozen, corrected spec
— never against the current CANDIDATE-0003 content as-is.

## 10. Phase J before EXP-0006?

Concrete infrastructure gaps found (not phase-number ceremony):
- **No training runner exists.** `research/runners.py`'s `RUNNERS` dict covers only
  EXP-0001–0005, and every one of those runners evaluates pre-existing checkpoints — none of
  them calls a training loop. EXP-0006 as proposed would be the **first-ever training job**
  this lab's orchestrator has attempted; the code path to run it does not exist.
- **`execution_budget.py` is wired to nothing.** It is a fail-closed GPU/runtime budget
  framework built during Phase-I-readiness hardening, but no caller anywhere invokes
  `require_execution_budget()` — the very code that would refuse to launch unauthorized/
  over-limit GPU work currently sits inert.
- **`resources.py` only does a point-in-time snapshot check** at launch (RAM/disk/VRAM
  available *right now*) — there is no ongoing supervision of a multi-hour unattended job: no
  wall-clock cap, no crash/kill-switch, no checkpoint-and-resume. Every prior EXP (0001–0005)
  completed in minutes as an offline analysis of already-captured data; nothing in this lab has
  ever run an hours-long unattended process.
- With the §2 correction (≥3 seeds), the realistic job is **≥72 GPU-hours across ≥3 separate
  unattended training runs** — a materially higher execution-risk profile than anything this
  lab has run before, and exactly the kind of job the original architecture's Phase J
  (resource-management/orchestration hardening) was meant to cover.

**Recommendation**: implement (at minimum) a scoped subset of Phase J — an actual training
runner, `execution_budget.py` wired into that runner's launch path, and a wall-clock cap +
crash-resume mechanism for unattended multi-hour jobs — **before** EXP-0006 (or any
training-heavy experiment) is executed. This is independent of, and does not substitute for,
the spec corrections in §1–§7, which should happen regardless and can happen in parallel.

## 11. CANDIDATE-0003 preserved unchanged

No file under `research/candidates/CANDIDATE-0003/` was modified by this audit. This document
is the sole artifact produced. Any corrected design becomes a **new, separate** EXP-0006
preregistration draft (in the style of `research/_exp0004_preregister.py` /
`research/_exp0005_preregister.py`) when a human chooses to produce one — not an edit to
CANDIDATE-0003's frozen proposal, review, or final-report JSON.

---

## Final recommendation

# IMPLEMENT_PHASE_J_BEFORE_EXP0006

Even a fully spec-corrected version of CANDIDATE-0003 cannot be *executed* today: no training
runner exists anywhere in this codebase, the execution-budget fail-closed framework is unwired,
and there is no supervision mechanism for an unattended multi-hour (realistically ≥72
GPU-hour, multi-seed) job. This is a capability gap, not a paperwork gap — registering EXP-0006
without it would freeze a spec the lab currently has no way to run safely.

**This does not mean the science is bad** — CANDIDATE-0003 is the first Phase I candidate to
reach and clear independent review, and its core hypothesis (training-data representation gap)
is genuinely novel versus EXP-0001–0005. The corrections below are refinements to an
already-sound design, not a rejection of its premise.

### Exact corrections / human decisions required, in order

1. Rewrite the hypothesis prose to reference candidate C (MEM-0015), not "the baseline," per §1.
2. Redesign as a ≥3-seed study with a pre-registered aggregation rule and full seed-disclosure
   requirement, per §2.
3. Pin dataset scope (class list, category minimums, target size with rationale), annotation
   protocol, and near-duplicate/burst-frame leakage exclusion (beyond SHA-256 exact-match), per
   §3.
4. **Human decision required**: re-classify `data_privacy_classification` (very likely
   `PRIVATE_USER_DATA` given the described capture content) and correspondingly treat
   `private_user_data_use_approved` as a live, not-`NOT_REQUIRED`, approval question, per §3.
5. Move `seed`/`epochs`/`lr0`/`batch` into a structured, checkable `training_hyperparameters`
   field rather than prose-only `reproducibility_requirements`, per §4.
6. **Prerequisite, before execution**: verify `yolov8m-oiv7.pt`'s upstream source, license, and
   confirm no undisclosed prior fine-tuning, per §5.
7. Treat "50 epochs / 24 GPU-hours" as an unverified estimate; run a small calibration pass
   before committing to the full run, and restate the resource estimate at ≥72 GPU-hours once
   multi-seed is applied, per §6.
8. Close the outcome-verdict gap: fold `recovery_rate` into `rejects_hypothesis_if` (or add an
   explicit intermediate bucket) so all three outcomes are exhaustive, per §7.
9. Add an explicit, stated latency/resource-regression assumption (architecture unchanged ⇒
   latency should be unchanged; confirm rather than leave silent), per §7.
10. **Human decision required**: do NOT grant `mac_iphone_deployment_approved` now — it applies
    only to a later, separate on-device-validation decision, per §8.
11. **Human decision required**: do NOT grant `new_training_approved` until items 1–9 above are
    resolved and frozen into a corrected EXP-0006 preregistration document, per §9.
12. **Infrastructure work required before execution** (not registration): build a training
    runner, wire `execution_budget.py` into its launch path, and add wall-clock-cap +
    crash-resume supervision for unattended multi-hour jobs, per §10.

No EXP-0006 registered. No approvals granted. No data collected. No training run. No live LLM
calls made during this audit.

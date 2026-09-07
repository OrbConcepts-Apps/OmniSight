# EXP-0006 Preregistration Audit

Zero-live-call task. Builds the corrected specification for the experiment that may
become EXP-0006, based on CANDIDATE-0003 and
`reports/phase_i/CANDIDATE_0003_EXP0006_ELIGIBILITY.md`. CANDIDATE-0003 is read-only
input — never modified. The corrected design lives in `research/preregistration.py`
(builder) and `research/preregistrations/EXP-0006-PROPOSED.json` (frozen artifact), a
namespace distinct from both `research/experiment_specs/` (DB-registered specs) and
`research/candidates/` (Phase I autonomous-loop artifacts). **Not registered into
`research/db.py` — EXP-0006 exists only as this file's content.**

## 1. Hypothesis / recovery-metric fix

CANDIDATE-0003's prose ("25% more than baseline") was vacuous — TRUE_DETECTOR_MISS is
*defined* as zero baseline detections, so the baseline's own recovery rate is
tautologically 0/92. The binding number was always the comparison against candidate C
(EXP-0005/MEM-0015, 17/92). Chose **interpretation C** (of the three offered in the
eligibility audit): incremental absolute count beyond the prior best candidate.
`RECOVERY_COUNT_THRESHOLD = 22` (17 + a pre-registered margin of 5). Numerator = count
of the 92 fixed baseline TRUE_DETECTOR_MISS cases the intervention model detects;
denominator = 92 (fixed); reference = candidate C's 17; direction = must exceed by the
margin. No ambiguous "25% more" language anywhere in the corrected text.

## 2. Multi-seed design

3 pre-registered seeds (42, 43, 44), same split/architecture/init/optimizer/epoch
budget across seeds (all embedded in `controlled_variables.training_config`, a
structured, hashable dict). `classify_seed_verdict()` gives every one of the 8
(precision, recall, recovery) combinations an explicit PASS/FAIL/INCONCLUSIVE verdict
— fixing the exact gap CANDIDATE-0003 left open (precision-pass + recall-pass +
recovery-fail was previously unassigned). `classify_aggregate_verdict()` requires a
**majority** (≥2/3) PASS for overall PASS, zero PASS for overall FAIL, and a single
precision-guardrail failure **vetoes** the whole aggregate regardless of the other two
seeds — no favorable seed alone can produce a PASS. Exhaustiveness proven by a
512-combination parametrized test (`test_exhaustive_over_random_sample`).

## 3. Causal control

Two-arm matched design: CONTROL (same starting checkpoint, identical training
procedure, additional training on resampled existing OIV7 data) vs. INTERVENTION
(identical procedure, additional training on the new OmniSight-domain data). This
isolates "additional training occurred" from "OmniSight data specifically helped." The
shipped, never-further-trained baseline (RUN-20260904-002) is retained as a
non-causal descriptive reference only, explicitly labeled as such — never treated as
the causal control, because its own pretraining history is unverified (see §4).

**Is the shipped checkpoint an adequate causal starting point?** Yes — since both
CONTROL and INTERVENTION fork from byte-identical weights, whatever happened before
that fork does not confound the CONTROL-vs-INTERVENTION comparison itself (only the
generalization claim beyond this lab's specific checkpoint). It is **not** adequate as
the control ARM on its own (i.e., 0 additional epochs) against a differently-trained
intervention, because that would conflate "more training happened at all" with
"OmniSight data specifically helped" — hence the matched-control arm.

## 4. Checkpoint provenance

Treated as a blocking prerequisite, not inferred from filename. Known: architecture
(YOLOv8m, confirmed via prior model introspection), class set (OIV7 601 classes,
confirmed via live `model.names` in EXP-0005). Unknown/prerequisite: exact upstream
source, SHA-256, weight-file license (distinct from the image/annotation licenses
`docs/DATASETS.md` already documents), original training dataset/hyperparameters,
whether undisclosed fine-tuning already exists on this specific file, exact framework
version. None fabricated.

## 5. Dataset specification

A manifest **contract**, not a dataset — nothing collected. Format matches
`data/manifests/eval_manifest.jsonl`'s real schema (verified by reading it) plus two
new required fields: `consent_status` and `capture_session_id`. Inclusion/exclusion,
class definition (unchanged single Person class), dual-annotation QA with IoU≥0.7
agreement target, session-disjoint train/val split (also the near-duplicate-leakage
control), SHA-256 exact-dup exclusion (existing) plus a new perceptual-hash
near-duplicate check against the frozen eval manifest, versioning
(`omnisight_v{date}_annotated`), and manifest-hash-once-it-exists are all specified.
No dataset size is fabricated — planning estimate only, explicitly labeled uncalibrated.

## 6. Privacy classification — corrected

CANDIDATE-0003 used `data_privacy_classification=NONE`, which the eligibility audit
found likely wrong given the described capture domain (real people, real
accessibility scenarios). Corrected to **`PRIVATE_USER_DATA`**. Explicit source-type
distinctions given (public/non-sensitive — not used; staged/synthetic — not used here;
consented collection — the required method; incidental-but-consented bystanders —
permitted; non-consenting private imagery — prohibited; covert/unsupported collection
— out of scope). `private_user_data_use_approved` remains False and is now correctly
flagged by the validator as `UNAPPROVED_PRIVATE_DATA_USE` (a real, live pending
approval this proposal did not have before).

## 7. Structured training configuration

All execution-critical hyperparameters (starting checkpoint, epochs/stopping rule,
optimizer, learning rate, scheduler, batch size, image size, augmentation, seeds, data
manifest hash, framework/version, device, checkpoint-selection rule, evaluation
thresholds) live in `controlled_variables.training_config`, a nested dict — not new
dataclass fields (deliberately avoids the Phase-I frozen-hash schema-evolution hazard;
an existing dict field already participates in the whole-proposal content hash).

## 8. Epoch / resource estimate

50 epochs / 24 GPU-hours per seed per arm are labeled `ESTIMATE_UNCALIBRATED`, not
fact. With 2 arms × 3 seeds, the planning ceiling becomes 144 GPU-hours (naive
multiplication, explicitly flagged as a ceiling, not a validated figure). No
calibration run was performed (out of scope for this task, per instruction). Disk/
checkpoint-storage requirements are marked PREREQUISITE rather than a fabricated number.

## 9. Outcome table

See §2 — `classify_seed_verdict`/`classify_aggregate_verdict` together give every
input combination exactly one of PASS/FAIL/INCONCLUSIVE, proven exhaustively by test.

## 10. Latency / practical usability

Not modeled as a numeric criterion in this offline design — `training_config.device`
is explicitly labeled "CUDA (RTX 3070 Ti, Windows) -- offline screening only, never
iPhone/ANE" everywhere it appears, matching EXP-0005's established convention of never
conflating a Windows/CUDA proxy with real iPhone/ANE latency. Device-level latency,
thermal, memory, end-to-end, and TTS-timing criteria are explicitly named as a
**separate, later, downstream validation requirement**, not a prerequisite for this
offline stage.

## 11. Device approval semantics

`mac_iphone_required=True` (this model, if it survives offline screening, will
eventually need device validation before shipping) but `mac_iphone_deployment_approved`
remains **False** and is correctly a `NEEDS_HUMAN_APPROVAL` item, never an `ERROR` —
registering or executing this proposal's OFFLINE stage does not require it, per the
family registry (`training_data` = `OFFLINE_SIMULATABLE`). Downstream device approval
is explicitly represented as a later, separate gate, never a prerequisite for offline
training.

## 12. Training approval

`new_training_approved` remains False. This task's purpose was specification quality,
not training authorization — the two are explicitly kept distinct throughout.

## 13. Phase J execution restriction

Recorded in `implementation_scope`: real execution additionally requires a real
`Runner` implementation, active real-process wall-clock enforcement, process-ownership
verification against a real launched process, `execution_budget` integration,
checkpoint/resume integration, safe termination, and passing integration tests — none
of which exist yet (`reports/phase_j/PHASE_J_SAFETY_AUDIT.md`'s residual-risk section).
No Runner was implemented in this task beyond what already exists (`FakeRunner`,
unchanged).

## 14-15. Registration decision and spec freeze

`validate()` returns **zero ERRORs** (structurally/scientifically valid) and exactly
3 correct `NEEDS_HUMAN_APPROVAL` issues (mac/iphone, new-training, private-data —
verified by test to be exactly this set, no more, no fewer) plus the standing
`NEEDS_HUMAN_REVIEW: SCIENTIFIC_MERIT` item every proposal always carries. `is_valid`
is True; `is_queue_eligible` is correctly False. The spec was frozen
(`ExperimentSpec.freeze("VALIDATED")`), and its hash verified round-trip after saving
and reloading from disk (`research/preregistrations/EXP-0006-PROPOSED.json`).

**Registration into `research/db.py` was deliberately NOT performed.** The
deterministic audit itself finds the spec text ready (zero errors, every unresolved
fact honestly stated as a prerequisite, nothing fabricated) — the authorization for
this task explicitly permits registering a proposal with open prerequisites, as long
as they are represented as explicit blockers and registration does not imply execution
authority (which would be the case here). However, inserting an `EXP-0006` row into
`research/db.py` is a materially more consequential, harder-to-reverse action than
writing a JSON preregistration file, and every prior Phase I/Phase J authorization in
this project has treated "DB contains exactly EXP-0001–EXP-0005, EXP-0006 does not
exist" as a standing, repeatedly-reverified invariant. This task's authorization frames
registration as a conclusion the audit MAY reach, not an instruction to perform it
outright. Consistent with this project's established discipline (never exceed the
exact scope authorized — see `feedback_omnisight_lab_process.md` point 1), the audit's
conclusion is recorded as a recommendation for a human to act on, not self-executed.
**EXP-0006 does not exist in `research/db.py` after this task.**

## Validator limitation, disclosed

`find_rejected_hypothesis_conflicts()` requires an **exact family match** against
existing EXP rows. Since no EXP-0001–0005 row has `family="training_data"`, this
mechanical check cannot fire for this proposal by construction — not because there is
no conceptual overlap with prior findings, but because the check's design only compares
within the same family. The `materially_new_rationale` field voluntarily documents the
relationship to EXP-0004/0005 and CANDIDATE-0001/0002 anyway, disclosed as a
deliberate choice, not a gate-satisfied requirement.

## Recommendation

**READY_TO_REGISTER_EXP0006** — the deterministic audit concludes the spec, as
written, is ready for a human to register (zero structural/scientific errors, every
open prerequisite explicitly represented rather than hidden, no fabricated content).
This is a recommendation for the next explicit human decision, not a self-executed
action: this task did not insert a DB row, per the reasoning in §14-15 above.

# EXP-0006 Dataset Construction & Annotation Protocol

Protocol document only. **No image or video collected. No annotation performed. No
dataset built.** Every schema/validator utility this protocol references
(`research/datasets/manifest_schema.py`, `research/provenance.py`) is exercised only
against synthetic fixture data in tests — never real captures. EXP-0006's frozen
`data_privacy_classification=PRIVATE_USER_DATA` is preserved, never weakened.

## 8. Dataset purpose (narrow scope)

Scoped exactly to EXP-0006's question: improving safety-relevant Person detection,
specifically recovering TRUE_DETECTOR_MISS cases attributable to domain mismatch
between Open Images V7 and real OmniSight usage. **Not** a general-purpose accessibility
dataset. Every capture-scenario category below is justified by an existing, cited
failure-analysis finding (EXP-0003's confusion analysis, EXP-0005's TRUE_DETECTOR_MISS
taxonomy, MEM-0017/MEM-0025's domain-gap findings) — no category was added merely
because it "seems relevant."

## 9. Unit of independence

Because captures may be video-derived, the **frame is not the independent unit**.
Identifiers required on every future manifest record (see `MediaUnitRecord` in
`research/datasets/manifest_schema.py`):

- `session_id` — one physical collection session (one outing/appointment).
- `sequence_id` — one continuous video/burst within a session.
- `frame_id` — one extracted frame within a sequence.
- `environment_domain` — indoor/outdoor + lighting/motion condition label (see §16).
- capture `device`/configuration.
- subject/actor identity — only recorded when consented AND needed (e.g. to track a
  specific staged participant across sessions for balance purposes); never required.

**Split leakage rule**: splits are session-aware, never frame-aware —
`detect_session_split_leakage()` flags any `session_id` appearing in more than one of
{train, val, test} (a session assigned `excluded` may coexist with anything — excluding
frames is never leakage). Near-adjacent frames from one video are therefore
structurally incapable of being split across train/eval, because they always share one
`session_id`.

## 10. Frozen 380-image evaluation set — protection

`research.datasets.manifest_schema.check_exact_overlap_with_frozen_eval()` compares
every candidate SHA-256 against the frozen `data/manifests/eval_manifest.jsonl` set (380
hashes). `flag_near_duplicates()` additionally compares perceptual hashes (once real
ones exist — see §22) against the same reference set. **Default handling for any
detected exact OR near-duplicate match is EXCLUSION, fail-closed, pending human
review** — enforced structurally by `validate_media_unit()`: a record with
`baseline_eval_overlap_status != NONE_DETECTED` MUST have `split=excluded`, or it fails
validation outright (tested).

## 11. Future dataset manifest — schema (no data populated)

`research/datasets/manifest_schema.py::MediaUnitRecord` — every field from the
authorization's list is present: `dataset_version`, `session_id`, `sequence_id`,
`frame_id`, `source_type`, `sha256`, `perceptual_hash`, `capture_timestamp_policy`,
`environment_domain`, `device`, `camera_configuration`, `annotation_status`,
`annotator_ids` (pseudonymous only — enforced by `validate_media_unit()` rejecting any
id containing a secret-shaped substring), `adjudication_status`, `consent_status`,
`privacy_class`, `licensing_status`, `split`, `exclusion_reason`,
`baseline_eval_overlap_status`. No unnecessary PII field exists in the schema at all
(no name, no face-embedding, no raw GPS field — see §27).

## 12. Privacy classification (kept conservative, subcategorized)

EXP-0006's frozen `data_privacy_classification=PRIVATE_USER_DATA` is **not weakened**.
`PRIVACY_CLASSES` subcategorizes for future governance:

- `STAGED_CONSENTED` — explicitly consented participants, staged capture. Preferred.
- `INCIDENTAL_CONSENTED` — incidental appearance, separately consented (e.g. a posted-
  signage pilot with an opt-out process).
- `PUBLIC_LICENSED` — an existing public/licensed source (not used in this design).
- `SYNTHETIC_NONIDENTIFIABLE` — synthetic/staged, no real identifiable person (not used
  in this design, though a future revision could substitute this to avoid private-data
  approval entirely).
- `RESTRICTED_PRIVATE` — requires explicit human approval before any use; the schema's
  own default value, so an unclassified record is fail-closed by construction.

`STAGED_CONSENTED`/`INCIDENTAL_CONSENTED` are the only classes this protocol proposes
using; `RESTRICTED_PRIVATE` material may be used only after a separate, explicit human
approval — never granted by this document.

## 13. Collection design — protocol only, not executed

Staged/consented collection is preferred wherever scientifically adequate. Dimensions to
plan (numbers deliberately NOT fixed here — see §20):

- **Viewpoint diversity**: chest-height and hand-held smartphone angles (the app's real
  usage posture), varied yaw/pitch.
- **Distance bands**: near (<2m), mid (2-6m), far (>6m) — targeting the "small/distant
  person" failure mode named in the authorization.
- **Lighting bands**: daylight, indoor-normal, low-light/dusk, backlit.
- **Occlusion conditions**: none, partial (bag/pole/doorway), heavy (mostly out of frame).
- **Motion conditions**: static, walking-pace motion blur, fast motion blur.
- **Indoor/outdoor balance**: both, matching real assistive-navigation contexts.
- **Clutter/context diversity**: uncluttered, moderately cluttered, visually busy scenes.
- **Positive/negative sampling**: both Person-present and genuinely Person-absent
  frames, at a prevalence that does not artificially invert real deployment prevalence
  (no aggressive rebalancing without separate justification, per the authorization).
- **Hard-negative capture**: scenes with person-shaped distractors (mannequins,
  posters, statues) to stress precision, matching the hazard-precision guardrail's
  concern.
- **Sequence duration**: short bursts favored over long continuous video, to reduce
  the number of near-duplicate frames requiring exclusion review per session.
- **Device metadata**: recorded per session (see `MediaUnitRecord.device`/
  `camera_configuration`).

## 14. Bystanders — conservative default

- Any frame containing a non-consenting, identifiable bystander is **excluded by
  default** (`consent_status=PROHIBITED` forces `split=excluded`, enforced structurally).
- Collection sessions should be **designed** (location/time choice, staged settings) to
  minimize incidental bystander presence in the first place — a policy preference, not
  merely a post-hoc filter.
- **Redaction is NOT treated as a general solution** here: blurring/masking a bystander
  would very likely **invalidate** the very Person-detection ground-truth label the
  dataset needs (a blurred person is no longer a faithful representation of the
  detection target) — so redaction is explicitly NOT proposed as a way to "keep" an
  otherwise-prohibited bystander frame for Person-detection training purposes. It might
  remain viable for unrelated non-Person negative-sample frames, but that specific case
  is flagged as **requiring human/legal/ethics review**, not decided here.
- No collection is authorized regardless of any of the above — protocol design only.

## 15. Minors / sensitive contexts — fail-closed, no exceptions

**No collection is authorized in this task or by this protocol for**: minors (any
frame with an apparent minor is excluded, no age-estimation heuristic is proposed as a
substitute for exclusion), schools, medical environments, private homes, bathrooms/
changing areas, or any other context this project has no basis to claim competence to
govern. This is a hard, non-negotiable exclusion list — future revision requires
explicit separate human/legal/ethics authorization, never an engineering judgment call.

## 16. Annotation ontology (minimum, reproducible)

- **Person bounding boxes** — same format as `data/manifests/eval_manifest.jsonl`
  (normalized `[x, y, w, h]`), `is_occluded`/`is_truncated`/`is_group_of` flags reused
  unchanged from the existing OIV7-derived schema.
- **Visibility/occlusion**: none / partial / heavy (3-way, matching §13's collection
  design categories — not a novel scale).
- **Size/distance proxy**: box area relative to image area — **deterministically
  derived from the bounding box itself**, never a subjective annotator judgment.
- **Blur/motion condition**: static / motion-blurred (binary, checked against a
  deterministic blur-metric threshold where feasible, annotator-confirmed otherwise —
  kept binary specifically to stay reproducible, per the authorization's "no
  unreproducible subjective labels" instruction).
- **Lighting condition**: daylight / indoor-normal / low-light (3-way, matching §13).
- **Clutter/context**: uncluttered / cluttered (binary).
- **Ignored/ambiguous regions**: a dedicated "ignore" region type for genuinely
  ambiguous cases (e.g. a Person that's <10% visible behind an object) — excluded from
  both positive and negative scoring, never forced into a binary label.

No subjective, unreproducible category (e.g. "typical accessibility scenario",
"realistic clutter") is included — every category above is either deterministically
derivable from the box/image or a small, concretely-defined discrete set.

## 17. TRUE_DETECTOR_MISS derivation rule — precise, not an annotation label

**`TRUE_DETECTOR_MISS` is never something an annotator labels.** It is a derived
evaluation category, computed exactly as follows:

```
ground-truth Person bounding boxes (from §16's annotation ontology)
    + frozen baseline model (benchmark/models/yolov8m-oiv7.pt, conf=0.4, iou=0.7) inference
    -> match ground-truth boxes to predictions (IoU-based greedy match,
       reusing benchmark.metrics.greedy_match, same convention as EXP-0001-0005)
    -> a ground-truth Person box with NO matched prediction = TRUE_DETECTOR_MISS
```

Annotators are never asked to label "miss" status directly — doing so would require
them to run the model themselves (out of scope for annotation work) or to guess at the
model's behavior (unreproducible). This distinction is stated explicitly because
CANDIDATE-0003/EXP-0006's own prose could otherwise be misread as implying miss status
is a raw label.

## 18. Annotation QA — operational protocol

- **Independent first/second annotation**: every included frame is annotated by two
  independent annotators with no visibility into each other's work.
- **Disagreement calculation**: per §19 below (corrected).
- **Adjudication**: a third, senior annotator resolves any box pair below the
  agreement threshold, or any category/attribute disagreement; the adjudicated result
  becomes the frame's `annotation_status=ADJUDICATED` ground truth.
- **Category/attribute agreement**: exact-match rate required for occlusion/blur/
  lighting/clutter labels (all discrete, low-cardinality categories per §16) — a
  disagreement on any attribute routes the frame to adjudication regardless of box IoU.
- **Minimum QA sample**: the FIRST batch of any new collection session must be
  double-annotated in full (100%) before the agreement rate is even computed; only
  once a session's measured agreement rate has been established across a full batch
  may spot-check-only QA (a fixed, preregistered percentage, e.g. 20%) be considered
  for subsequent batches from the SAME session/conditions — never adopted before that
  evidence exists.
- **QA threshold failure**: if a batch's measured agreement rate falls below the
  preregistered threshold (§19), the ENTIRE batch is held in `annotation_status=
  FIRST_PASS` (not `ADJUDICATED`), re-annotated or adjudicated at 100%, and the
  annotation guideline is reviewed for ambiguity before the next batch — never silently
  accepted at a lower bar.

## 19. IoU-QA rule — audited, corrected (Amendment Candidate, see §30)

**Original wording** ("IoU ≥0.7 on ≥95% of boxes") is under-specified in three ways:

1. **Which IoU?** — Read in context ("inter-annotator agreement target"), this is
   INTER-ANNOTATOR IoU (first vs. second independent annotator), not adjudicated-vs-
   initial. This much is inferable from context but should be stated explicitly.
2. **Per-image or aggregate?** — Unstated. An aggregate-only rule can hide a batch
   where 100% of disagreement concentrates in a subset of hard images (exactly the
   low-light/motion-blur/occlusion conditions this dataset targets) while still passing
   on paper.
3. **Box correspondence / unmatched boxes** — Unstated. Two annotators' box sets must
   be matched (e.g. via IoU-based greedy assignment) before "IoU >= 0.7" is even
   computable per pair; a box one annotator drew that the other completely missed is a
   DIFFERENT, more serious failure mode (a missed detection, not a boundary-precision
   disagreement) and must not be silently folded into the same statistic.

**Corrected rule** (proposed, not applied to the frozen EXP-0006 spec by this task):

- Match annotator A's and B's boxes via greedy IoU assignment (same algorithm as
  `benchmark.metrics.greedy_match`).
- Report THREE separate statistics per batch: (a) matched-pair IoU ≥ 0.7 rate — target
  ≥95% of matched pairs; (b) unmatched-box rate (boxes with no counterpart within
  IoU ≥ 0.1) — target ≤5% of all boxes across both annotators; (c) per-image minimum —
  no more than 5% of images in the batch may fall below an 80% matched-pair agreement
  rate, catching localized quality collapse an aggregate-only rule would hide.
- All three are computed PER BATCH (a session's worth of double-annotated frames), not
  globally across the whole eventual dataset — so a quality problem is caught close to
  when it happens, not diluted across the full corpus.

## 20. Sample-size planning framework (no pilot run, no fixed target)

Known baseline facts (real, established): 92 TRUE_DETECTOR_MISS Person cases within the
380-image evaluation set; EXP-0005's existing failure-bucket decomposition.

**Deterministic planning method for a future pilot** (not run in this task):

1. Collect a small, explicitly-labeled PILOT batch (size TBD by a human, not by this
   document) under the intended capture protocol.
2. From the pilot, empirically estimate: **Person prevalence** (fraction of frames
   containing ≥1 Person), **failure-bucket prevalence** (fraction of Person-containing
   frames the FROZEN baseline model still misses, per §17's derivation), **sequence
   correlation** (intraclass correlation of failure status within a session/sequence —
   video-derived frames are highly correlated, inflating apparent sample size unless
   corrected for).
3. Compute **effective independent sample size** = (raw frame count) / (design effect
   from sequence correlation), NOT raw frame count — a standard survey-sampling
   correction, necessary here because §9 already establishes frames are not
   independent.
4. **Hard-case coverage**: track, per pilot batch, how many effectively-independent
   hard cases (low-light + motion-blur + partial-occlusion intersection) were
   captured — the final target dataset size should be set to reach a human-decided
   minimum hard-case count, not merely a total-frame count.
5. Only once steps 2-4 produce real numbers should a human fix the final training-data
   target size. This document does not propose one.

## 21. Control/intervention data-matching rule

To ensure the causal difference is domain/data composition, not quantity or compute:

- **Number of training examples**: CONTROL's resampled-OIV7 set size = INTERVENTION's
  OmniSight-domain set size, exactly (drawn AFTER the OmniSight-domain dataset's final
  size is known — this is why final size cannot be fixed before §20's evidence exists).
- **Person annotation count**: matched as closely as the resampling allows (OIV7's own
  existing annotation density is a given; the OmniSight-domain sampling should target
  an equivalent count).
- **Training steps/epochs, optimizer, learning-rate schedule, batch size, augmentation
  policy**: identical between arms — already specified structurally in
  `controlled_variables.training_config` (see `research/preregistration.py`) and
  unaffected by this protocol.
- **Class balance**: both arms are Person-detection-focused; CONTROL's OIV7 resample
  should be filtered to Person-containing images to match INTERVENTION's composition,
  not merely a random OIV7 sample (which would silently reintroduce a domain
  difference — the non-Person-class distribution — that isn't the intended IV either,
  though this dataset itself is scoped to Person, per §8).
- **Checkpoint selection rule**: identical rule applied to both arms (already specified
  in `training_config.checkpoint_selection_rule`).
- **What cannot be fixed until dataset size exists**: the exact CONTROL resample size
  and its exact Person-annotation count — both depend on the (not-yet-determined)
  OmniSight-domain dataset's final size, per §20.

## 22. Duplicate / near-duplicate detection — implemented, tested

`research/datasets/manifest_schema.py`:
- `check_exact_overlap_with_frozen_eval()` — exact SHA-256 match against the frozen
  380-image set.
- `hamming_distance_hex()` + `flag_near_duplicates()` — a deterministic Hamming-distance
  comparison PRIMITIVE for perceptual hashes (no real perceptual hashing library is
  invoked here, since no real images exist to hash in this task — this is the
  comparison logic a future pipeline with real phashes would use).
- **Explicitly NOT overclaimed as perfect semantic-duplicate detection** — perceptual
  hashing catches near-identical recaptures, not semantically-similar-but-distinct
  scenes. `threshold` is a function PARAMETER, never hardcoded, and is itself named as
  a preregistered value requiring future empirical validation (e.g. against a labeled
  set of known-duplicate vs. known-distinct pairs) before being trusted operationally.
- Both functions only ever FLAG candidates for review — never silently exclude or
  include (tested: `test_never_auto_excludes_only_flags`).

## 23. Dataset versioning — immutable

`research.datasets.manifest_schema.dataset_version_hash()` combines four component
hashes (manifest, annotation-guideline version, split assignment, provenance summary)
into one canonical SHA-256. Changing ANY input after freeze produces a NEW hash — this
function has no persistence of its own and never overwrites a prior stored version;
that discipline is a caller contract (documented here, tested for hash-change behavior).
`dataset_version` string format: `omnisight_v{capture_end_date}_annotated`, matching the
already-frozen EXP-0006 spec's own convention.

## 24. Raw vs. derived data — separation contract

- **Raw captured media**: never committed to git (`data/raw/` is already `.gitignore`d
  repo-wide — confirmed, tested).
- **Redacted media** (if the §14 bystander-review process ever approves any): stored
  separately from raw originals, never overwriting the original.
- **Annotation files**: tracked separately from media, reference media by `sha256`/
  `frame_id`, never by absolute path.
- **Derived frames** (extracted from video): distinct from their source sequence,
  linked via `sequence_id`.
- **Manifests**: tracked in git IF and only if they contain no sensitive paths/PII
  (logical identifiers only — enforced by `contains_prohibited_path()`).
- **Train-ready exports**: a further-derived, separately versioned artifact, never
  conflated with the source manifest.

Every derivative retains provenance linkage (via `sha256`/`session_id`/`sequence_id`)
back to its source without needing to store any raw personal path or identifying data.

## 25. Storage contract

- No raw private media is ever committed to git — `data/raw/` is gitignored repo-wide
  (verified: `git check-ignore` on a synthetic path under it returns ignored; verified:
  zero media-extension files currently tracked under `data/raw/`).
- Manifests live in `data/manifests/`, matching the existing tracked
  `eval_manifest.jsonl` convention — JSONL manifests ARE trackable (no PII, logical
  identifiers only); `.sqlite` caches under the same directory remain gitignored
  (pre-existing rule).
- Private/raw data path is configurable outside the repo entirely (a future collection
  tool should accept a `--raw-data-root` pointing anywhere on disk, never assume
  `data/raw/` must physically hold gigabytes of private video under the repo checkout).
- Generated train-ready exports live in a separate directory from source media (see §24).
- Hashes are stable (streaming SHA-256, `research.provenance.hash_file()`, tested
  against files larger than one read-chunk).
- Artifact collision protection: reuses the existing Phase J
  `research.execution_job.artifact_safety.reserve_job_output_dir()` pattern for any
  future dataset-build output directory.
- Deletion/retention decisions remain explicit and human-controlled — no automatic
  deletion logic exists or is proposed anywhere in this protocol.

## 26. Secret / path sanitization

`research.datasets.manifest_schema.contains_prohibited_path()` rejects any field
containing a Windows user-path pattern (`C:\Users\<name>\...`) or POSIX home-directory
pattern (`/home/<name>/`, `/Users/<name>/`) — tested. `validate_media_unit()` also
rejects any `annotator_ids` entry that looks like a secret/API key/token (substring
check against common secret-field names) rather than a pseudonymous identifier — tested.
No field in `MediaUnitRecord` is EXIF/GPS-shaped (see §27).

## 27. EXIF / location metadata policy

**Default: strip GPS/precise-location metadata from every future media export.**
`MediaUnitRecord.capture_timestamp_policy` is itself a POLICY field (`DATE_ONLY` |
`NONE` | `FULL`), not a raw-EXIF passthrough — the schema has no field for raw EXIF or
GPS at all, by design. Preserving full timestamp/location metadata would only be
justified if a specific, EXP-0006-relevant scientific need required it (none currently
identified — indoor/outdoor and lighting-band labels already capture what this
experiment actually needs), and would require a separate, explicit human approval
before being enabled — never a default.

## 28. Provenance registry

`research/provenance.py::ProvenanceRecord`/`ProvenanceRegistry` — schema fields:
`artifact, sha256, source, source_status, license, license_status, acquisition_method,
parent_artifact, conversion_operation, tool_version, verification_status,
evidence_pointer, notes`. Applied to the real existing model artifacts this session —
see `reports/phase_i/EXP0006_CHECKPOINT_PROVENANCE_AUDIT.md` and the machine-readable
`research/provenance_registry_exp0006.json`. Every UNKNOWN value in that registry
remains explicitly UNKNOWN, never inferred or fabricated.

## 29. External questions requiring later, separately-authorized verification

1. Live HTTP confirmation of `benchmark/models/yolov8m-oiv7.pt`'s source URL/release
   asset (never fetched this session).
2. Any upstream errata on that specific Ultralytics release asset.
3. Formal legal review of AGPL-3.0 applicability to OmniSight's actual distribution
   model.
4. (Future, once collection is separately authorized) specific consent-form/IRB-style
   review process appropriate for staged human-subject capture — this protocol
   identifies the NEED (§14/§15) but does not itself constitute that review.

## 30. Proposed EXP-0006 amendment candidates

EXP-0006 (`research/experiment_specs/EXP-0006.json`,
`research/preregistrations/EXP-0006-PROPOSED.json`) was **NOT amended** by this task.
The following are audit findings for a human to later act on via the established
`ExperimentSpec.amend()` mechanism.

### Amendment Candidate 1 — IoU-QA rule precision (§19)

- **Current frozen wording**: "inter-annotator bounding-box IoU agreement target
  >= 0.7 on >= 95% of boxes before a batch is accepted."
- **Proposed corrected wording**: "Match annotator A/B boxes via greedy IoU assignment.
  Per batch: (a) >=95% of matched pairs achieve IoU>=0.7; (b) unmatched-box rate
  <=5% of all boxes; (c) no more than 5% of images fall below an 80% per-image
  matched-pair agreement rate."
- **Reason**: original wording is ambiguous on box-correspondence and per-image vs.
  aggregate scope (§19 above).
- **Scientific impact**: MEDIUM — affects annotation QA reliability, not the causal
  design or success criteria.
- **Required before data collection?** Recommended yes (an annotation vendor/protocol
  needs the precise rule before starting).
- **Required before training?** Not directly, but a training run on data QA'd under
  the ambiguous rule would inherit its uncertainty.
- **Documentation-only?** No — this is a genuine methodological correction, not mere
  wording polish.

### Amendment Candidate 2 — training-config prerequisite fields now resolved (§4 audit)

- **Current frozen wording**: `controlled_variables.training_config` marks
  `lr_scheduler`/`augmentation_policy`/`framework_version` as `PREREQUISITE` and epochs/
  optimizer as unvalidated estimates copied from CANDIDATE-0003.
- **Proposed corrected wording**: real, VERIFIED values now exist for several of these
  from the checkpoint-provenance audit (`train_args.optimizer=SGD`,
  `train_args.lr0=0.01` — already matched what was assumed, now confirmed rather than
  assumed) — but epochs/batch/lr-schedule for the FUTURE fine-tuning run (not the
  ORIGINAL 2023 training run) remain genuinely undetermined pending calibration (Phase J
  restriction, unchanged).
- **Reason**: the checkpoint audit resolved what the ORIGINAL training used, which is
  informative context but is not automatically the correct recipe for continued
  fine-tuning on new data.
- **Scientific impact**: LOW — informative, not blocking.
- **Required before data collection?** No.
- **Required before training?** No — still gated on the Phase J real-Runner-calibration
  restriction, unchanged.
- **Documentation-only?** Yes.

No amendment is required before data collection beyond Candidate 1 (recommended, not
mandatory) and none of these findings change any success criterion, guardrail, seed
plan, or causal design already frozen in EXP-0006.

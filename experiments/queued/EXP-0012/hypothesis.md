# EXP-0012 — Hypothesis

**Family**: threshold_postprocessing
**Validation requirement**: OFFLINE_SIMULATABLE
**Parent experiment**: (none)

## Hypothesis

Test-time augmentation (ultralytics augment=True: multi-view flip/scale inference merged via NMS), confidence and NMS IoU fixed at production values, recovers meaningful Person recall (>= +0.03 vs the augment=False control) while keeping hazard-aggregate precision at or above the standard guardrail (>= 0.757) AND keeping p95 latency within the lab's existing +50% regression guardrail. Mechanism: unlike confidence-threshold or NMS-IoU changes (which can only re-select among a single pass's already-proposed candidates), TTA runs multiple transformed views through the model and can produce a genuinely NEW candidate detection in a view where the original pass proposed nothing -- the one channel tested so far with a real mechanism to potentially recover a TRUE_DETECTOR_MISS case.

## Motivation

Both prior post-hoc, single-pass levers on the shipped checkpoint (EXP-0007-0010 Person confidence threshold, EXP-0011 NMS IoU) are closed as negative results, and both were mechanistically incapable of ever recovering a TRUE_DETECTOR_MISS case (92/239 baseline Person false negatives, the dominant failure category) -- neither can add a detection where the single forward pass proposed none. TTA is mechanistically different: it is real inference (still no training, no new data), but genuinely multi-pass, giving it an actual channel this lab has not yet tested.

## Rationale

Uses only the existing frozen 380-image public eval set and the existing shipped checkpoint. Not a hyperparameter sweep -- a single ON/OFF test, so there is no grid point to cherry-pick from. Real cost is measured directly and checked against the lab's EXISTING latency guardrail (not a new criterion) -- a recall improvement bought with unacceptable real-time latency cost is a genuine, reportable constraint for an assistive-vision system, not something to omit. Orthogonal to and unaffected by the blocked EXP-0006 pilot.

## Expected outcome

Either: (a) no meaningful recall improvement -- closes this line of inquiry as a third consecutive negative result on the shipped checkpoint's inference-time levers; (b) a meaningful recall improvement but at an unacceptable latency cost -- a real, reportable constraint, not a usable production candidate; or (c) a genuine, guardrail-clearing, latency-acceptable improvement -- the strongest candidate this lab has found on the shipped checkpoint, still requiring the same robustness check before being treated as credible.

## Risks

Runs real (non-training) GPU inference, 2-3x the normal per-image cost expected -- no training, no private data, no production/config changes (uses predict_at(), never predict(); benchmark/config.py is never modified). The only risk is scientific: an honest negative or latency-constrained result must be reported as such, not reframed as a win.

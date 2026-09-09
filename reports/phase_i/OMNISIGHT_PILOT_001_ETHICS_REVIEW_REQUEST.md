# OMNISIGHT-PILOT-001 — Ethics / Institutional Review Request Packet

**Purpose of this document**: a self-contained packet the human project operator can
hand directly to a teacher, mentor, school, science-fair authority, institutional
reviewer, or IRB (whichever is applicable to their situation) to obtain the one
determination this project currently requires before any real participant may be
recorded. This document does not itself grant, assume, or imply that determination —
only a qualified human reviewer, external to this codebase, can supply it.

No participant has been recorded. No media exists. No consent has been collected. This
packet is planning material only.

---

## 1. What is being proposed

A small, bounded, staged/consented pilot data collection ("OMNISIGHT-PILOT-001") to
support a computer-vision research project (OmniSight/OmniLab) studying detection of
people by a real-time smartphone assistive-vision system. The pilot's purpose is
**not** to demonstrate a model improvement — it is to estimate practical planning
quantities (how often people appear in typical footage, how often hard conditions like
low light or occlusion occur, how long annotation takes, how much footage must be
excluded for privacy reasons) that are needed to design a later, properly-scoped data
collection, if one is ever approved.

## 2. Exact scope requested

| Parameter | Value |
|---|---|
| Participants | ≤ 6, each an adult who has personally and voluntarily agreed to take part |
| Sessions | ≤ 6 (one per participant) |
| Sequences | ≤ 24 total (≤ 4 per participant session) |
| Duration | ~20 seconds per sequence |
| Setting | Controlled public or private-neutral space chosen by the participant/operator (e.g. an office, a consented outdoor walking route) |
| Subjects filmed | Only consenting participants — no bystanders are an intended subject |
| Hard exclusions | Minors, schools, medical environments, bathrooms/changing areas, private-home interiors, or any other sensitive context are never used, with no exceptions |
| Data leaves this pilot's control | Never — no cloud upload, no external sync, no third-party service of any kind |

This exact scope is cryptographically bound in the codebase
(`pilot_plan_hash = e0488d752b679be2e7d010be2f1ded8f75e3dc1f9d8bdfd85219905b1a257dc3`) —
software enforcement rejects any collection attempt that does not match these numbers
exactly, so the software cannot silently exceed what is described here.

## 3. What is captured

Short smartphone video sequences of a consenting participant in a plain scenario (e.g.
walking, standing, moving through a space), captured on an ordinary consumer phone
camera. No audio content is used for research purposes. No special or covert recording
device is used — participants know they are being recorded and what for.

## 4. What is NOT done with this data

- Not uploaded anywhere outside the collector's own local storage.
- Not used to train any model unless a **separate**, later, explicit approval is
  granted for that specific purpose — collection and training-use are governed by two
  independent software flags in this project, and only the collection flag has been
  granted so far.
- Not deployed to, or tested against, any production OmniSight installation.
- Not published or shared, by default, in any identifiable form — any future
  publication of aggregate research findings would not include raw participant imagery
  unless a participant separately and explicitly opts in.
- Not retained under real names, emails, or other directly identifying information in
  the research dataset itself — only a pseudonymous per-participant identifier is
  recorded there; any real consent/contact record is kept in a separate system, not in
  the tracked research repository.

## 5. Consent commitments (for the future, real consent conversation)

Before any participant is filmed, a human operator must obtain and record, per
participant, confirmation of every item below (see the companion field checklist,
`OMNISIGHT_PILOT_001_FIELD_CHECKLIST.md`, for the exact template used):

1. The participant is eligible under this protocol (no sensitive context, informed
   voluntary participation).
2. Any applicable institutional consent requirement is separately satisfied.
3. The participant understands they will be video/image captured.
4. The participant understands the footage will have people's bounding-box locations
   manually annotated.
5. The participant understands the intended research/model-development use.
6. The participant's publication/image-use preference is recorded (default: private,
   never published).
7. Withdrawal and retention terms have been explained.
8. A pseudonymous participant ID has been assigned.
9. Consent evidence is stored separately from the research dataset.
10. No personally identifying information is copied into the tracked dataset.

No such consent has been collected yet — this is the template that would be used only
after both this ethics determination and a real consent conversation occur.

## 6. Exactly what determination is being requested

The project's own software will not proceed to real participant recording until a
human records one of the following four values against
`ethics_or_institutional_review_status` (the software default is the first, most
conservative value, and no code in this project sets it to any of the other three):

| Value | Meaning |
|---|---|
| `NOT_ASSESSED` (current/default) | No determination has been made yet. Recording remains blocked. |
| `NOT_REQUIRED_BY_INSTITUTION` | A qualified reviewer has determined that, for this specific bounded scope, no formal institutional/ethics review process applies (e.g. an informal school project below the applicable threshold, as determined by someone with the authority to say so). |
| `APPROVED_OR_EXEMPT` | A qualified reviewer/IRB/institution has formally reviewed this exact scope and approved it, or granted a recognized exemption. |
| `BLOCKED_PENDING_REVIEW` | A reviewer has looked at this and determined collection should not proceed as currently scoped (e.g. needs a different consent process, needs a higher level of review, or is declined). |

**This project will not, and structurally cannot, self-select `NOT_REQUIRED_BY_INSTITUTION`
or `APPROVED_OR_EXEMPT`.** Only a human, external to the automated system, recording a
real determination, can move the status out of `NOT_ASSESSED`.

## 7. What happens once a determination is supplied

- If `NOT_REQUIRED_BY_INSTITUTION` or `APPROVED_OR_EXEMPT`: the software's field-clearance
  check (`research.datasets.ethics_review.is_field_clearance_granted`, combined with the
  already-granted software collection gate) will report the pilot as cleared for real
  participant recording, still bounded by the caps in §2 above and still requiring the
  per-participant consent items in §5.
- If `BLOCKED_PENDING_REVIEW`: recording remains blocked, and the operator should
  follow the reviewer's stated reason/next step.
- If left `NOT_ASSESSED`: no change — recording remains blocked, exactly as it is today.

## 8. What is explicitly requested of the reviewer

Please review the scope described in §§1–5 above and indicate which of the four
determinations in §6 applies, given your institution's or your own applicable
standards. No response beyond that determination is required for this software to act
on it.

---

*Generated as planning/handoff material only. Contains no participant data, no
consent records, no names, no contact information, and no fabricated determination.*

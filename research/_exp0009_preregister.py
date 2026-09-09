"""One-shot registration of EXP-0009: a post-hoc image-level bootstrap
robustness analysis of EXP-0008's person_threshold=0.30 PASS.

Run via:

    uv run python -m research._exp0009_preregister

Idempotent: no-ops if EXP-0009 already exists.

EXP-0008 preregistered no uncertainty criterion -- a single deterministic
point-estimate evaluation, like every other experiment in this lab to date.
This experiment does NOT retroactively alter EXP-0008's PASS; it is a
separate, honestly-labeled record answering a different question: does that
PASS survive image-level resampling of the same frozen 380-image eval set.
No new inference, no training, no private data, no device deployment, no
new human approval. Orthogonal to the blocked EXP-0006 pilot.
"""

from __future__ import annotations

from research.config import EXPERIMENTS_DIR
from research.db import Experiment, OmniLabDB
from research.experiment_lifecycle import move_to_status
from research.experiment_schema import write_queued_artifacts

EXPERIMENT_ID = "EXP-0009"


def _experiment() -> Experiment:
    return Experiment(
        experiment_id=EXPERIMENT_ID,
        hypothesis=(
            "EXP-0008's PASS (person_threshold=0.30 recovers person.recall +0.066 while hazard "
            "-aggregate precision clears the 0.757 guardrail by +0.0098) is ROBUST under "
            "image-level bootstrap resampling of the same frozen 380-image eval set: the "
            "guardrail is held in at least 95% of resamples, and the recall-improvement "
            "direction is robust at the 95% level (2.5th percentile of the resampled recall "
            "delta is > 0)."
        ),
        motivation=(
            "EXP-0008's PASS rested on a guardrail margin of only +0.0098 -- thin relative to "
            "sampling noise on a 380-image manifest. Before treating that PASS as evidence of "
            "anything beyond 'clears the guardrail on this one particular sample', its "
            "robustness must be checked. This is a post-hoc robustness analysis, explicitly "
            "labeled as such -- EXP-0008 itself preregistered no uncertainty criterion, and "
            "none is invented for it retroactively; this experiment's OWN criterion (below) "
            "was fixed before any bootstrap replicate was computed."
        ),
        rationale=(
            "Uses only the existing frozen 380-image eval set (no private data, no new "
            "inference, no training, no device deployment, no new human approval). Image-level "
            "(not box-level) resampling respects the real dependence structure -- boxes within "
            "one image are not independent draws. Orthogonal to and unaffected by the blocked "
            "EXP-0006 pilot."
        ),
        independent_variable=(
            "none in the traditional sense -- this experiment resamples the EXISTING evidence "
            "(low_conf_predictions.jsonl, eval_manifest.jsonl) at the image level and recomputes "
            "metrics from scratch per replicate; the model/weights/threshold under test "
            "(person_threshold=0.30 vs baseline 0.40) is unchanged from EXP-0008."
        ),
        controls={
            "model": "yolov8m-oiv7.pt (same weights as the canonical baseline)",
            "manifest": "data/manifests/eval_manifest.jsonl (unchanged, resampled at image level only)",
            "iou_threshold": 0.7,
            "imgsz": 640,
            "person_threshold_tested": 0.30,
            "n_replicates": 2000,
            "seed": 20260909,
        },
        evaluation_method=(
            "benchmark/diagnostics/person_threshold_bootstrap.py draws 2000 image-level "
            "bootstrap replicates (fixed seed=20260909, numpy.random.default_rng) of the 380 "
            "real eval images, recomputing hazard/Person metrics from scratch on each resampled "
            "set (never bootstrapping already-aggregated point estimates); duplicate image "
            "occurrences within one replicate are relabeled with synthetic per-occurrence "
            "sample_ids so they do not compete for the same GT boxes. Reports percentile (2.5/"
            "97.5) CIs and the fraction of replicates violating the 0.757 hazard-precision "
            "guardrail."
        ),
        success_criteria={
            "pre_registered_robustness_criterion": (
                "ROBUST (PASS) if guardrail_violation_rate<=0.05 AND 2.5th-percentile(delta "
                "person.recall)>0; FRAGILE (FAIL) if guardrail_violation_rate>0.05; else "
                "INCONCLUSIVE. Fixed in benchmark/diagnostics/person_threshold_bootstrap.py "
                "before any bootstrap replicate was computed or inspected."
            ),
            "note": (
                "Not the same shape as this lab's other experiments' guardrail-vs-baseline "
                "point-estimate criteria -- this is a distributional robustness check, and is "
                "explicitly reported as such, never as a production readiness claim."
            ),
        },
        risks=(
            "Read-only, deterministic re-analysis of already-captured, already-approved "
            "diagnostic data -- no code touches production, no new inference, no training. The "
            "only 'risk' is scientific: this may show EXP-0008's PASS does not survive "
            "resampling, which must be reported honestly regardless of outcome (no new "
            "significance threshold invented after seeing the result)."
        ),
        expected_outcome=(
            "Either: (a) ROBUST -- EXP-0008's finding survives resampling, and the next cheapest "
            "controlled test (e.g. sensitivity around threshold=0.30, or an independent eval "
            "set) becomes the natural next step before any production consideration; or (b) "
            "FRAGILE/INCONCLUSIVE -- the guardrail margin is a sampling artifact, and EXP-0008's "
            "PASS must be reinterpreted as fragile, not a credible positive finding, without "
            "altering EXP-0008's own preserved deterministic record."
        ),
        parent_experiment_id="EXP-0008",
        experiment_family="threshold_postprocessing",
        model_version="yolov8m-oiv7.pt",
        dataset_version="data/manifests/eval_manifest.jsonl",
        baseline_run_id="RUN-20260904-002",
        validation_requirement="OFFLINE_SIMULATABLE",
        estimated_cost=0.0,
    )


def main() -> None:
    with OmniLabDB() as db:
        existing_ids = {e.experiment_id for e in db.list_experiments()}
        if EXPERIMENT_ID in existing_ids:
            print(f"{EXPERIMENT_ID} already registered -- no-op.")
            return
        exp = _experiment()
        db.create_experiment(exp)
        write_queued_artifacts(exp, EXPERIMENTS_DIR / "queued" / exp.experiment_id)
        move_to_status(exp.experiment_id, "QUEUED")
        print(f"{EXPERIMENT_ID} registered, status=QUEUED.")


if __name__ == "__main__":
    main()

"""One-shot registration of EXP-0010: threshold-sensitivity bootstrap, the
direct follow-up to EXP-0009's FRAGILE finding on person_threshold=0.30.

Run via:

    uv run python -m research._exp0010_preregister

Idempotent: no-ops if EXP-0010 already exists.

Question: is there a MORE conservative person_threshold between 0.30 and
0.40 that is both ROBUST under image-level bootstrap resampling (EXP-0009's
criterion) AND still clears this lab's established minimum-meaningful-
recall-delta bar (>=0.03, used by every experiment since EXP-0001)? Both
thresholds combined here (0.05 guardrail-violation tolerance, 0.03 minimum
meaningful delta) were fixed BEFORE this experiment's grid was computed --
neither is invented or tuned from this experiment's own result.

No new inference, no training, no private data, no device deployment, no
new human approval. Orthogonal to the blocked EXP-0006 pilot.
"""

from __future__ import annotations

from research.config import EXPERIMENTS_DIR
from research.db import Experiment, OmniLabDB
from research.experiment_lifecycle import move_to_status
from research.experiment_schema import write_queued_artifacts

EXPERIMENT_ID = "EXP-0010"


def _experiment() -> Experiment:
    return Experiment(
        experiment_id=EXPERIMENT_ID,
        hypothesis=(
            "At least one person_threshold in the finer grid [0.40,0.38,0.36,0.34,0.32,0.30] is "
            "simultaneously (a) ROBUST under image-level bootstrap resampling -- "
            "guardrail_violation_rate<=0.05, per EXP-0009's criterion -- AND (b) achieves a mean "
            "recall delta vs the production baseline (0.40) of at least the lab's established "
            "minimum meaningful delta (0.03, used by every experiment since EXP-0001)."
        ),
        motivation=(
            "EXP-0009 showed person_threshold=0.30's guardrail margin is FRAGILE (36.9% "
            "bootstrap violation rate). The coarse original grid ([0.40,0.30,0.20,0.10]) cannot "
            "distinguish '0.30 was simply the best of a few bad options' from 'a nearby, more "
            "conservative threshold is actually safe.' This experiment answers that directly, "
            "as the cheapest available non-training controlled test, per this lab's own explicit "
            "next-step guidance after a fragile positive finding: check sensitivity around the "
            "chosen threshold before treating anything as a credible positive result."
        ),
        rationale=(
            "Uses only the existing frozen 380-image eval set (no private data, no new "
            "inference, no training, no device deployment, no new human approval). Reuses "
            "EXP-0009's exact image-level bootstrap methodology and fixed seed, extended to a "
            "finer grid computed in one pass per replicate (statistically cleaner than "
            "re-resampling per threshold). Orthogonal to and unaffected by the blocked EXP-0006 "
            "pilot."
        ),
        independent_variable=(
            "person_confidence_threshold, swept finely between 0.30 and 0.40 "
            "(benchmark/diagnostics/person_threshold_sensitivity_bootstrap.py, grid fixed before "
            "any replicate was computed)"
        ),
        controls={
            "model": "yolov8m-oiv7.pt (same weights as the canonical baseline)",
            "manifest": "data/manifests/eval_manifest.jsonl (unchanged, resampled at image level)",
            "n_replicates": 2000,
            "seed": 20260909,
            "other_hazard_class_threshold": 0.40,
        },
        evaluation_method=(
            "benchmark/diagnostics/person_threshold_sensitivity_bootstrap.py: 2000 image-level "
            "bootstrap replicates (same seed as EXP-0009), evaluating the entire threshold grid "
            "on each resampled image set. Per-threshold classification uses EXP-0009's exact "
            "rule. The experiment-level verdict asks whether any ROBUST threshold ALSO clears "
            "the pre-existing 0.03 minimum-meaningful-delta bar."
        ),
        success_criteria={
            "compound_criterion": (
                "PASS if >=1 threshold is ROBUST (guardrail_violation_rate<=0.05) AND its mean "
                "recall delta vs 0.40 is >=0.03. FAIL otherwise (including the case where a "
                "threshold is robust but its recall gain is below 0.03 -- robustness and "
                "meaningfulness in tension is a real, reportable negative finding, not an "
                "excuse to lower the bar)."
            ),
            "note": "Both component thresholds (0.05 violation tolerance, 0.03 minimum delta) were fixed before this grid was computed, in EXP-0009 and since EXP-0001 respectively.",
        },
        risks=(
            "Read-only, deterministic re-analysis of already-captured data -- no code touches "
            "production, no new inference, no training."
        ),
        expected_outcome=(
            "Either: (a) a genuinely robust AND meaningful threshold exists -- the closest thing "
            "to a credible positive finding this line of inquiry could produce, still requiring "
            "independent validation before any production consideration; or (b) no threshold "
            "clears both bars -- the per-class threshold-policy line of inquiry is closed as a "
            "viable production lever under current evidence, a genuine negative result."
        ),
        parent_experiment_id="EXP-0009",
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

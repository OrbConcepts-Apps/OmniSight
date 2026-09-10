"""EXP-0013: image tiling for small/distant Person recovery.

Real, non-training inference over the existing frozen 380-image eval set.
Never touches benchmark/config.py or benchmark/results/baseline/. See
research/_exp0013_preregister.py for the full mechanism validation and
preregistered primary configuration (PRIMARY_CONFIG there is mirrored here).

Per image:
  1. The FULL-IMAGE pass is REUSED directly from
     benchmark/results/baseline/predictions.jsonl (bit-identical to the
     official baseline by construction -- not re-inferred).
  2. 4 crop tiles (2x2 grid, 20% overlap -- benchmark/diagnostics/tiling.py
     ::generate_tiles) are generated from the image's real pixel dimensions
     (PIL), cropped in memory, and run through NEW (non-training) inference
     via BaselineModel.predict_array_at() at the fixed production
     conf=0.4/iou=0.7/imgsz=640.
  3. Each tile's predictions are remapped to full-image-normalized
     coordinates (tiling.py::remap_bbox_tile_to_full) and combined with the
     reused full-image predictions.
  4. The combined set is deduplicated via cross-"tile" NMS
     (tiling.py::merge_detections_nms, IoU=0.7, same-class-only).

TRUE_DETECTOR_MISS recovery audit (explicitly avoids the EXP-0011
neighboring-GT artifact): uses the ACTUAL greedy-match exclusion logic from
benchmark.metrics.evaluate_detections's returned
matches_at_fixed_iou["Person"].matched_gt_ids -- the same per-class,
per-image, confidence-ordered, GT-once-claimed matching every other
experiment in this lab is judged by -- rather than a standalone per-case
IoU check. global_gi (evaluate_detections' internal Person-only, dataset-
wide GT index) is mapped back to (sample_id, per-sample Person gt_index)
via a reconstructed lookup built in the SAME iteration order
person_confusion_analysis.py used to assign its own gt_index, so every
claimed recovery is directly auditable against a specific
(sample_id, gt_index) record.

Boundary analysis: for each of the 92 known baseline TRUE_DETECTOR_MISS
Person GT boxes, classifies whether it is fully contained within a single
crop tile, split across tile boundaries (partial overlap with >=1 tile, no
tile fully contains it), or not overlapped by any crop tile at all (only
the full-image pass could ever have seen it) -- to interpret whether
recovery/non-recovery correlates with boundary splitting.

Run with: uv run python -m benchmark.diagnostics.tile_sweep
Writes: benchmark/results/diagnostics/tile_sweep.json
(runs real, non-training inference over 4 tiles/image -- several minutes on GPU)
"""

from __future__ import annotations

import json
import time

import numpy as np
from PIL import Image

from benchmark.config import (
    CONF_THRESHOLD,
    EVAL_MANIFEST_PATH,
    HAZARD_CLASS_MAP,
    IOU_THRESHOLD,
    RAW_IMAGE_DIR,
    REPO_ROOT,
)
from benchmark.dataset import assert_eval_only, load_manifest
from benchmark.diagnostics.person_confusion_analysis import OUT_PATH as CONFUSION_ANALYSIS_PATH
from benchmark.diagnostics.tiling import generate_tiles, merge_detections_nms, remap_bbox_tile_to_full
from benchmark.metrics import Detection, GroundTruth, evaluate_detections
from benchmark.model import BaselineModel

HAZARD_CLASSES = tuple(HAZARD_CLASS_MAP.values())

DIAG_DIR = REPO_ROOT / "benchmark" / "results" / "diagnostics"
BASELINE_PRED_PATH = REPO_ROOT / "benchmark" / "results" / "baseline" / "predictions.jsonl"
BASELINE_METRICS_PATH = REPO_ROOT / "benchmark" / "results" / "baseline" / "metrics.json"
OUT_PATH = DIAG_DIR / "tile_sweep.json"

N_COLS = 2
N_ROWS = 2
OVERLAP_FRACTION = 0.20
MERGE_IOU = IOU_THRESHOLD  # 0.7, matches production NMS IoU
GUARDRAIL_FLOOR = 0.757
MIN_MEANINGFUL_DELTA = 0.03


def _load_baseline_predictions() -> dict:
    preds_by_id: dict = {}
    with open(BASELINE_PRED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            preds_by_id[d["sample_id"]] = d["predictions"]
    return preds_by_id


def _load_true_detector_miss_cases() -> list:
    d = json.loads(CONFUSION_ANALYSIS_PATH.read_text(encoding="utf-8"))
    return [r for r in d["records"] if r["primary_category"] == "TRUE_DETECTOR_MISS"]


def _classify_boundary(gt_bbox_norm: tuple, img_w: int, img_h: int, crop_tiles: list) -> str:
    """Classifies a GT box (normalized xywh, full-image coords) relative to
    the 4 crop tiles: 'single_tile_full' if >=1 tile fully contains it,
    'boundary_split' if no tile fully contains it but >=1 partially
    overlaps, 'no_tile_overlap' if no crop tile overlaps it at all (only
    the full-image pass could see it)."""
    gx, gy, gw, gh = gt_bbox_norm
    gx0, gy0 = gx * img_w, gy * img_h
    gx1, gy1 = gx0 + gw * img_w, gy0 + gh * img_h

    any_full, any_partial = False, False
    for t in crop_tiles:
        overlap_x = max(0.0, min(gx1, t.x1) - max(gx0, t.x0))
        overlap_y = max(0.0, min(gy1, t.y1) - max(gy0, t.y0))
        if overlap_x <= 0 or overlap_y <= 0:
            continue
        fully_contained = t.x0 <= gx0 and gx1 <= t.x1 and t.y0 <= gy0 and gy1 <= t.y1
        if fully_contained:
            any_full = True
        else:
            any_partial = True
    if any_full:
        return "single_tile_full"
    if any_partial:
        return "boundary_split"
    return "no_tile_overlap"


def run_sweep() -> dict:
    samples = load_manifest(EVAL_MANIFEST_PATH)
    assert_eval_only(samples)
    baseline_preds = _load_baseline_predictions()
    tdm_cases = _load_true_detector_miss_cases()
    tdm_lookup = {(c["sample_id"], c["gt_index"]): c for c in tdm_cases}

    model = BaselineModel()

    all_merged_dets: list = []
    all_gts: list = []
    global_person_gt_keys: list = []  # position i == the i-th Person GT across the whole dataset, in manifest order
    boundary_by_key: dict = {}  # (sample_id, gt_index) -> boundary classification, Person only
    tile_only_latency_ms: list = []  # sum of the 4 NEW tile passes per image

    for sample in samples:
        image_path = RAW_IMAGE_DIR / sample.filename
        with Image.open(image_path) as im:
            img_w, img_h = im.size
            im_rgb = im.convert("RGB")
            crop_tiles = generate_tiles(img_w, img_h, N_COLS, N_ROWS, OVERLAP_FRACTION, include_full_image=False)

            full_dets = [
                Detection(sample_id=sample.sample_id, class_name=p["class_name"], bbox=tuple(p["bbox"]), confidence=p["confidence"])
                for p in baseline_preds.get(sample.sample_id, [])
                if p["class_name"] in HAZARD_CLASSES
            ]

            tile_dets: list = []
            t_tiles_start = time.perf_counter()
            for tile in crop_tiles:
                crop = im_rgb.crop((tile.x0, tile.y0, tile.x1, tile.y1))
                crop_arr = np.asarray(crop)
                raw_preds = model.predict_array_at(crop_arr, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
                for p in raw_preds:
                    if p.class_name not in HAZARD_CLASSES:
                        continue
                    full_bbox = remap_bbox_tile_to_full(p.bbox, tile, img_w, img_h)
                    tile_dets.append(Detection(sample_id=sample.sample_id, class_name=p.class_name, bbox=full_bbox, confidence=p.confidence))
            tile_only_latency_ms.append((time.perf_counter() - t_tiles_start) * 1000.0)

        merged = merge_detections_nms(full_dets + tile_dets, iou_threshold=MERGE_IOU)
        all_merged_dets.extend(merged)

        person_local_idx = 0
        for lbl in sample.labels:
            if lbl.class_name in HAZARD_CLASSES:
                all_gts.append(GroundTruth(sample_id=sample.sample_id, class_name=lbl.class_name, bbox=lbl.bbox))
            if lbl.class_name == "Person":
                global_person_gt_keys.append((sample.sample_id, person_local_idx))
                boundary_by_key[(sample.sample_id, person_local_idx)] = _classify_boundary(lbl.bbox, img_w, img_h, crop_tiles)
                person_local_idx += 1

    overall, per_class, matches = evaluate_detections(all_merged_dets, all_gts, list(HAZARD_CLASSES), map_ious=(0.5,))
    person = per_class["Person"]

    # Recovery audit: uses the REAL greedy-match exclusion logic, not a naive per-case IoU check.
    person_match = matches["Person"]
    recovered_keys = set()
    for sample_id, global_gi in person_match.matched_gt_ids:
        key = global_person_gt_keys[global_gi]
        if key in tdm_lookup:
            recovered_keys.add(key)

    recovered_small = [k for k in recovered_keys if tdm_lookup[k]["gt_is_small"]]
    recovered_non_small = [k for k in recovered_keys if not tdm_lookup[k]["gt_is_small"]]

    boundary_summary = {"single_tile_full": 0, "boundary_split": 0, "no_tile_overlap": 0}
    boundary_by_recovery = {"recovered": {"single_tile_full": 0, "boundary_split": 0, "no_tile_overlap": 0},
                             "not_recovered": {"single_tile_full": 0, "boundary_split": 0, "no_tile_overlap": 0}}
    for key, cls in boundary_by_key.items():
        if key not in tdm_lookup:
            continue  # boundary analysis is scoped to the 92 known TDM cases
        boundary_summary[cls] += 1
        bucket = "recovered" if key in recovered_keys else "not_recovered"
        boundary_by_recovery[bucket][cls] += 1

    baseline_metrics_json = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
    baseline_p95 = baseline_metrics_json["latency_ms"]["p95"]
    tile_only_latency_ms.sort()
    tile_p50 = tile_only_latency_ms[len(tile_only_latency_ms) // 2]
    tile_p95 = tile_only_latency_ms[int(len(tile_only_latency_ms) * 0.95)]
    total_p95_estimate = baseline_p95 + tile_p95  # baseline full-image pass (reused/known) + new tile passes

    baseline_hazard_precision = 0.8070175438596491  # official baseline, verified elsewhere; reused, not recomputed
    baseline_person_recall = 0.21122112211221122

    delta_recall = person.recall - baseline_person_recall
    guardrail_satisfied = overall.precision >= GUARDRAIL_FLOOR
    meaningful = delta_recall >= MIN_MEANINGFUL_DELTA

    return {
        "procedure": {
            "n_cols": N_COLS, "n_rows": N_ROWS, "overlap_fraction": OVERLAP_FRACTION,
            "merge_iou": MERGE_IOU, "confidence_threshold": CONF_THRESHOLD, "per_tile_nms_iou": IOU_THRESHOLD,
            "full_image_pass_source": "reused from benchmark/results/baseline/predictions.jsonl (not re-inferred)",
            "guardrail_floor": GUARDRAIL_FLOOR, "min_meaningful_delta": MIN_MEANINGFUL_DELTA,
        },
        "hazard": {"precision": overall.precision, "recall": overall.recall, "tp": overall.tp, "fp": overall.fp, "fn": overall.fn},
        "person": {
            "recall": person.recall, "precision": person.precision,
            "tp": person.tp, "fp": person.fp, "fn": person.fn, "num_gt": person.num_gt,
        },
        "summary": {
            "person_recall_delta_vs_baseline": delta_recall,
            "guardrail_satisfied": guardrail_satisfied,
            "meaningful_delta": meaningful,
            "point_estimate_pass": guardrail_satisfied and meaningful,
        },
        "true_detector_miss_recovery": {
            "denominator": len(tdm_cases),
            "recovered_count": len(recovered_keys),
            "recovered_small_subset_count": len(recovered_small),
            "recovered_small_subset_denominator": sum(1 for c in tdm_cases if c["gt_is_small"]),
            "recovered_non_small_count": len(recovered_non_small),
            "recovered_keys_audit": sorted([f"{sid}#{gi}" for sid, gi in recovered_keys]),
        },
        "boundary_analysis": {
            "all_92_tdm_cases": boundary_summary,
            "by_recovery_outcome": boundary_by_recovery,
        },
        "latency_ms": {
            "baseline_full_image_p95_reused": baseline_p95,
            "new_tile_passes_only_p50": tile_p50,
            "new_tile_passes_only_p95": tile_p95,
            "estimated_total_pipeline_p95": total_p95_estimate,
            "multiplier_vs_baseline": total_p95_estimate / baseline_p95,
        },
    }


def main() -> None:
    if not BASELINE_PRED_PATH.exists():
        raise FileNotFoundError(f"{BASELINE_PRED_PATH} not found.")
    results = run_sweep()
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(json.dumps({"summary": results["summary"], "true_detector_miss_recovery": results["true_detector_miss_recovery"],
                       "latency_ms": results["latency_ms"]}, indent=2))


if __name__ == "__main__":
    main()

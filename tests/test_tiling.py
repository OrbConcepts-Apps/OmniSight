"""Correctness fixtures for benchmark/diagnostics/tiling.py, run BEFORE any
EXP-0013 benchmark execution, per this task's explicit requirement.
Synthetic geometry only -- every expected value here is hand-computed, not
derived from the module under test."""

from __future__ import annotations

import pytest

from benchmark.diagnostics.tiling import Tile, generate_tiles, merge_detections_nms, remap_bbox_tile_to_full
from benchmark.metrics import Detection


class TestGenerateTiles:
    def test_deterministic_2x1_grid_with_clean_overlap(self):
        # img_w=600, n_cols=2, overlap=0.5 -> tile_w = 600/(2-0.5)=400, stride=200.
        # Tile0: x=[0,400). Tile1: x=[200,600). Overlap region width=200=0.5*400. Exact integers.
        tiles = generate_tiles(img_w=600, img_h=600, n_cols=2, n_rows=1, overlap_fraction=0.5, include_full_image=False)
        assert len(tiles) == 2
        assert (tiles[0].x0, tiles[0].y0, tiles[0].x1, tiles[0].y1) == (0, 0, 400, 600)
        assert (tiles[1].x0, tiles[1].y0, tiles[1].x1, tiles[1].y1) == (200, 0, 600, 600)
        assert tiles[0].is_full_image is False and tiles[1].is_full_image is False

    def test_full_image_included_as_tile_zero(self):
        tiles = generate_tiles(img_w=600, img_h=600, n_cols=2, n_rows=1, overlap_fraction=0.5, include_full_image=True)
        assert tiles[0].tile_id == 0
        assert tiles[0].is_full_image is True
        assert (tiles[0].x0, tiles[0].y0, tiles[0].x1, tiles[0].y1) == (0, 0, 600, 600)
        assert len(tiles) == 3  # full image + 2 tiles

    def test_row_major_deterministic_order_2x2(self):
        tiles = generate_tiles(img_w=1000, img_h=800, n_cols=2, n_rows=2, overlap_fraction=0.2, include_full_image=True)
        assert [t.tile_id for t in tiles] == [0, 1, 2, 3, 4]
        assert tiles[0].is_full_image
        # row 0: tiles 1,2 (left-to-right); row 1: tiles 3,4.
        assert tiles[1].y0 == tiles[2].y0 == 0
        assert tiles[1].x0 < tiles[2].x0
        assert tiles[3].y0 == tiles[4].y0 > 0
        assert tiles[3].x0 < tiles[4].x0

    def test_non_square_image(self):
        tiles = generate_tiles(img_w=1024, img_h=768, n_cols=2, n_rows=2, overlap_fraction=0.2, include_full_image=False)
        for t in tiles:
            assert 0 <= t.x0 < t.x1 <= 1024
            assert 0 <= t.y0 < t.y1 <= 768

    def test_edge_tiles_never_exceed_image_bounds(self):
        for w, h in [(311, 446), (1024, 1024), (997, 613)]:  # includes real dataset extremes + an odd size
            tiles = generate_tiles(img_w=w, img_h=h, n_cols=2, n_rows=2, overlap_fraction=0.2, include_full_image=True)
            for t in tiles:
                assert t.x0 >= 0 and t.y0 >= 0
                assert t.x1 <= w and t.y1 <= h
                assert t.x1 > t.x0 and t.y1 > t.y0

    def test_deterministic_repeated_calls_identical(self):
        a = generate_tiles(997, 613, 2, 2, 0.2, True)
        b = generate_tiles(997, 613, 2, 2, 0.2, True)
        assert a == b

    def test_invalid_dimensions_rejected(self):
        with pytest.raises(ValueError):
            generate_tiles(0, 600, 2, 2, 0.2)
        with pytest.raises(ValueError):
            generate_tiles(600, 600, 2, 2, 1.0)  # overlap must be < 1
        with pytest.raises(ValueError):
            generate_tiles(600, 600, 0, 2, 0.2)  # n_cols must be >= 1


class TestRemapBboxTileToFull:
    def test_box_entirely_inside_one_tile(self):
        # Tile at x=[200,600) (width=400), y=[0,600) (width=600) within a 600x600 image.
        tile = Tile(tile_id=1, x0=200, y0=0, x1=600, y1=600)
        # Tile-local normalized box: center at (0.5,0.5), size (0.1,0.1).
        full = remap_bbox_tile_to_full((0.5, 0.5, 0.1, 0.1), tile, img_w=600, img_h=600)
        x, y, w, h = full
        assert x == pytest.approx(400 / 600)  # 200 + 0.5*400 = 400
        assert y == pytest.approx(300 / 600)  # 0 + 0.5*600 = 300
        assert w == pytest.approx(40 / 600)   # 0.1*400
        assert h == pytest.approx(60 / 600)   # 0.1*600

    def test_box_crossing_tile_boundary_is_clipped_to_image(self):
        # Same tile as above (x1=600=img_w, right at the image edge). A box near
        # the tile's own right edge (px+pw>1, a plausible detector output right
        # at a crop boundary) must clip to the ORIGINAL IMAGE bound (600), not
        # silently extend past it.
        tile = Tile(tile_id=1, x0=200, y0=0, x1=600, y1=600)
        full = remap_bbox_tile_to_full((0.95, 0.5, 0.2, 0.1), tile, img_w=600, img_h=600)
        x, y, w, h = full
        # x_full = 200 + 0.95*400 = 580; x1_full = 200 + (0.95+0.2)*400 = 660 -> clipped to 600.
        assert x == pytest.approx(580 / 600)
        assert w == pytest.approx((600 - 580) / 600)  # clipped width = 20/600

    def test_box_at_image_edge_zero_origin_tile(self):
        tile = Tile(tile_id=1, x0=0, y0=0, x1=400, y1=600)
        full = remap_bbox_tile_to_full((0.0, 0.0, 0.05, 0.05), tile, img_w=600, img_h=600)
        x, y, w, h = full
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)
        assert w == pytest.approx(0.05 * 400 / 600)
        assert h == pytest.approx(0.05 * 600 / 600)

    def test_full_image_tile_is_identity_remap(self):
        tile = Tile(tile_id=0, x0=0, y0=0, x1=1000, y1=800, is_full_image=True)
        full = remap_bbox_tile_to_full((0.25, 0.25, 0.1, 0.1), tile, img_w=1000, img_h=800)
        assert full == pytest.approx((0.25, 0.25, 0.1, 0.1))

    def test_non_square_image_remap(self):
        tile = Tile(tile_id=1, x0=0, y0=0, x1=512, y1=384)
        full = remap_bbox_tile_to_full((0.5, 0.5, 0.1, 0.2), tile, img_w=1024, img_h=768)
        x, y, w, h = full
        assert x == pytest.approx((0.5 * 512) / 1024)
        assert y == pytest.approx((0.5 * 384) / 768)
        assert w == pytest.approx((0.1 * 512) / 1024)
        assert h == pytest.approx((0.2 * 384) / 768)


class TestMergeDetectionsNms:
    def test_no_detections_returns_empty(self):
        assert merge_detections_nms([], iou_threshold=0.5) == []

    def test_box_in_overlapping_tiles_deduplicated(self):
        # Same real-world Person, detected independently by two overlapping
        # tiles, remapped to nearly-identical full-image coordinates -- must
        # collapse to one detection (the higher-confidence one kept).
        d1 = Detection(sample_id="s", class_name="Person", bbox=(0.40, 0.40, 0.10, 0.20), confidence=0.55)
        d2 = Detection(sample_id="s", class_name="Person", bbox=(0.405, 0.402, 0.098, 0.198), confidence=0.72)
        merged = merge_detections_nms([d1, d2], iou_threshold=0.5)
        assert len(merged) == 1
        assert merged[0].confidence == 0.72

    def test_same_class_non_overlapping_both_kept(self):
        d1 = Detection(sample_id="s", class_name="Person", bbox=(0.1, 0.1, 0.1, 0.1), confidence=0.6)
        d2 = Detection(sample_id="s", class_name="Person", bbox=(0.8, 0.8, 0.1, 0.1), confidence=0.5)
        merged = merge_detections_nms([d1, d2], iou_threshold=0.5)
        assert len(merged) == 2

    def test_different_class_overlapping_boxes_both_kept(self):
        # A Person and a Clothing box at the SAME location must NOT suppress
        # each other -- NMS is strictly per-class.
        d1 = Detection(sample_id="s", class_name="Person", bbox=(0.4, 0.4, 0.1, 0.2), confidence=0.6)
        d2 = Detection(sample_id="s", class_name="Clothing", bbox=(0.4, 0.4, 0.1, 0.2), confidence=0.5)
        merged = merge_detections_nms([d1, d2], iou_threshold=0.5)
        assert len(merged) == 2
        assert {d.class_name for d in merged} == {"Person", "Clothing"}

    def test_deterministic_output_order_repeated_calls(self):
        dets = [
            Detection(sample_id="s", class_name="Person", bbox=(0.1, 0.1, 0.1, 0.1), confidence=0.6),
            Detection(sample_id="s", class_name="Car", bbox=(0.5, 0.5, 0.2, 0.2), confidence=0.9),
            Detection(sample_id="s", class_name="Person", bbox=(0.8, 0.8, 0.1, 0.1), confidence=0.5),
        ]
        a = merge_detections_nms(list(dets), iou_threshold=0.5)
        b = merge_detections_nms(list(reversed(dets)), iou_threshold=0.5)  # input order shuffled
        assert a == b  # output order must not depend on input order

    def test_three_way_duplicate_keeps_only_highest_confidence(self):
        d1 = Detection(sample_id="s", class_name="Person", bbox=(0.40, 0.40, 0.10, 0.20), confidence=0.30)
        d2 = Detection(sample_id="s", class_name="Person", bbox=(0.405, 0.402, 0.098, 0.198), confidence=0.72)
        d3 = Detection(sample_id="s", class_name="Person", bbox=(0.398, 0.399, 0.101, 0.199), confidence=0.55)
        merged = merge_detections_nms([d1, d2, d3], iou_threshold=0.5)
        assert len(merged) == 1
        assert merged[0].confidence == 0.72

    def test_invalid_iou_threshold_rejected(self):
        with pytest.raises(ValueError):
            merge_detections_nms([], iou_threshold=0.0)
        with pytest.raises(ValueError):
            merge_detections_nms([], iou_threshold=1.5)

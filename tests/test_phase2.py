"""End-to-end unit tests for Phase 2 data pipelines, edge-case guards, augmenters, and Golden Val QC."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.augment_and_crop import generate_copypaste_dataset, generate_crop_zoom_dataset, generate_weather_dataset
from src.box_to_mask_sam2 import convert_bbox_dataset
from src.build_master import build_master_dataset
from src.eval_golden import _fallback_score_split, analyze_predictions_against_gt
from src.map_and_merge_public import convert_public_source
from src.prepare_vn_frames import RawDetection, extract_frames_from_sources, pseudo_label_folder
from src.selfcheck_bridge import materialize_selfcheck
from src.validate_dataset import load_competition_config, run_qc


def _save_patterned_image(path: Path, size: tuple[int, int] = (640, 480), seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(30, 220, (size[1], size[0], 3), dtype=np.uint8)
    # Draw crisp rectangles so dhash & Laplacian sharpness are distinct and high
    cv2.rectangle(arr, (40 + seed * 7, 40), (200 + seed * 7, 200), (255, 255, 255), -1)
    cv2.line(arr, (0, 0), (size[0] - 1, size[1] - 1), (10, 20, 240), 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


class Phase2PipelineTests(unittest.TestCase):
    def test_coco_conversion_and_multi_rider_merge(self) -> None:
        cfg = load_competition_config("configs/competition.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir = root / "coco_imgs"
            _save_patterned_image(img_dir / "street_01.jpg", (640, 480), seed=3)
            coco = {
                "images": [{"id": 1, "file_name": "street_01.jpg", "width": 640, "height": 480}],
                "categories": [
                    {"id": 1, "name": "person"},
                    {"id": 2, "name": "motorcycle"},
                    {"id": 3, "name": "car"},
                ],
                "annotations": [
                    # Motorcycle + 2 riders sitting on it -> merged into ONE class 5 (Motorcyclist)
                    {"id": 10, "image_id": 1, "category_id": 2, "iscrowd": 0, "segmentation": [[180, 240, 320, 240, 320, 360, 180, 360]]},
                    {"id": 11, "image_id": 1, "category_id": 1, "iscrowd": 0, "segmentation": [[200, 150, 260, 150, 260, 320, 200, 320]]},
                    {"id": 12, "image_id": 1, "category_id": 1, "iscrowd": 0, "segmentation": [[250, 160, 310, 160, 310, 320, 250, 320]]},
                    # Unrelated pedestrian far away -> stays Pedestrian (8)
                    {"id": 13, "image_id": 1, "category_id": 1, "iscrowd": 0, "segmentation": [[480, 180, 540, 180, 540, 380, 480, 380]]},
                    # Car -> stays Car (0)
                    {"id": 14, "image_id": 1, "category_id": 3, "iscrowd": 0, "segmentation": [[20, 220, 140, 220, 140, 340, 20, 340]]},
                ],
            }
            ann_path = root / "coco.json"
            ann_path.write_text(json.dumps(coco), encoding="utf-8")

            out_dir = root / "staged_coco"
            res = convert_public_source("coco", "coco", img_dir, ann_path, out_dir, registry_path=None)
            self.assertEqual(res["kept_images"], 1)
            self.assertEqual(res["auto_merged_riders"], 1)
            self.assertEqual(res["class_counts"].get("Motorcyclist"), 1)
            self.assertEqual(res["class_counts"].get("Pedestrian"), 1)
            self.assertEqual(res["class_counts"].get("Car"), 1)
            self.assertNotIn("Motorcycle", res["class_counts"])

            qc = run_qc(out_dir, cfg)
            self.assertTrue(qc["ok"])
            self.assertEqual(qc["edge_case_flag_counts"], {})

    def test_idd_and_bdd100k_filters(self) -> None:
        cfg = load_competition_config("configs/competition.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # IDD test: autorickshaw -> Other vehicle (3), rider + bicycle -> Cyclist (7)
            # Also verify duplicate stems across different sequence folders (201/ vs 202/) do not collide!
            for seq, cls_label in (("201", "autorickshaw"), ("202", "car")):
                idd_img = root / "idd_imgs" / "train" / seq / "0001_leftImg8bit.jpg"
                _save_patterned_image(idd_img, (800, 600), seed=5 if seq == "201" else 6)
                idd_lbl = root / "idd_lbls" / "train" / seq / "0001_gtFine_polygons.json"
                idd_lbl.parent.mkdir(parents=True)
                objs = (
                    [
                        {"label": "autorickshaw", "polygon": [[80, 300], [220, 300], [220, 480], [80, 480]]},
                        {"label": "bicycle", "polygon": [[400, 340], [520, 340], [520, 500], [400, 500]]},
                        {"label": "rider", "polygon": [[420, 230], [500, 230], [500, 440], [420, 440]]},
                    ]
                    if seq == "201"
                    else [{"label": "car", "polygon": [[100, 300], [280, 300], [280, 460], [100, 460]]}]
                )
                idd_lbl.write_text(json.dumps({"imgWidth": 800, "imgHeight": 600, "objects": objs}), encoding="utf-8")
            idd_out = root / "staged_idd"
            r_idd = convert_public_source("idd", "idd", root / "idd_imgs", root / "idd_lbls", idd_out, registry_path=None)
            self.assertEqual(r_idd["kept_images"], 2)
            self.assertEqual(r_idd["class_counts"].get("Other vehicle"), 1)
            self.assertEqual(r_idd["class_counts"].get("Cyclist"), 1)
            self.assertEqual(r_idd["class_counts"].get("Car"), 1)
            self.assertTrue(run_qc(idd_out, cfg)["ok"])

            # BDD100K test with night filter
            bdd_imgs = root / "bdd_imgs"
            _save_patterned_image(bdd_imgs / "day.jpg", (640, 480), seed=8)
            _save_patterned_image(bdd_imgs / "night.jpg", (640, 480), seed=9)
            bdd_json = root / "bdd.json"
            bdd_json.write_text(json.dumps([
                {
                    "name": "day.jpg",
                    "attributes": {"weather": "clear", "timeofday": "daytime"},
                    "labels": [{"category": "car", "poly2d": [{"vertices": [[50, 200], [180, 200], [180, 300], [50, 300]]}]}],
                },
                {
                    "name": "night.jpg",
                    "attributes": {"weather": "rainy", "timeofday": "night"},
                    "labels": [{"category": "traffic light", "poly2d": [{"vertices": [[300, 80], [340, 80], [340, 160], [300, 160]]}]}],
                },
            ]), encoding="utf-8")
            bdd_out = root / "staged_bdd"
            r_bdd = convert_public_source(
                "bdd100k", "bdd100k", bdd_imgs, bdd_json, bdd_out, time_filter={"night"}, registry_path=None
            )
            self.assertEqual(r_bdd["kept_images"], 1)
            self.assertEqual(r_bdd["class_counts"].get("Traffic light"), 1)

    def test_sam2_box_to_mask_pole_bleed_guard(self) -> None:
        cfg = load_competition_config("configs/competition.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_p = root / "imgs" / "sign_01.jpg"
            lbl_p = root / "lbls" / "sign_01.txt"
            _save_patterned_image(img_p, (640, 640), seed=11)
            lbl_p.parent.mkdir(parents=True)
            # Sign bbox at x=[0.4, 0.6], y=[0.2, 0.4]
            lbl_p.write_text("42 0.50 0.30 0.20 0.20\n", encoding="utf-8")

            def mock_sam2_with_pole_bleed(img_bgr: np.ndarray, boxes: list[list[float]]) -> list[np.ndarray]:
                h, w = img_bgr.shape[:2]
                m = np.zeros((h, w), dtype=np.uint8)
                x1, y1, x2, y2 = [int(round(v)) for v in boxes[0]]
                # Intentionally bleed far downward (simulating a tall pole below the sign)
                m[y1:y2, x1:x2] = 1
                m[y2:min(h, y2 + 220), (x1 + x2) // 2 - 8 : (x1 + x2) // 2 + 8] = 1
                return [m]

            out_dir = root / "staged_vnts"
            res = convert_bbox_dataset(
                root / "imgs",
                root / "lbls",
                out_dir,
                predictor=mock_sam2_with_pole_bleed,
                default_class=9,
                registry_path=None,
            )
            self.assertEqual(res["instances_written"], 1)
            self.assertEqual(res["bleed_clipped"], 1)
            qc = run_qc(out_dir, cfg)
            self.assertTrue(qc["ok"])
            # Must NOT trigger SIGN_POLE_SUSPECT because clip_mask_to_box cut off the pole bleed
            self.assertNotIn("SIGN_POLE_SUSPECT", qc["edge_case_flag_counts"])

    def test_vn_frames_prep_augmenters_master_and_eval(self) -> None:
        cfg = load_competition_config("configs/competition.yaml")
        sc = materialize_selfcheck("train_and_export.ipynb", "work/_selfcheck")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 1. Golden Val with 1 image (1 Sign + 1 Car)
            gval = root / "golden_val"
            _save_patterned_image(gval / "images" / "val" / "gv_01.jpg", (960, 720), seed=21)
            (gval / "labels" / "val").mkdir(parents=True)
            (gval / "labels" / "val" / "gv_01.txt").write_text(
                "0 0.10 0.50 0.30 0.50 0.30 0.75 0.10 0.75\n"
                "9 0.70 0.15 0.78 0.15 0.78 0.25 0.70 0.25\n",
                encoding="utf-8",
            )

            # 2. Raw VN images: 1 leak (copy of gv_01), 1 blurry, 2 unique sharp images
            raw_vn = root / "raw_vn"
            _save_patterned_image(raw_vn / "leak_gv.jpg", (960, 720), seed=21)
            blur_img = np.full((720, 960, 3), 128, dtype=np.uint8)
            Image.fromarray(blur_img).save(raw_vn / "blur.jpg")
            _save_patterned_image(raw_vn / "vn_01.jpg", (960, 720), seed=31)
            _save_patterned_image(raw_vn / "vn_02.jpg", (960, 720), seed=45)

            staged_vn = root / "staged_vn"
            ext = extract_frames_from_sources(raw_vn, staged_vn / "images" / "train", exclude_dir=gval)
            self.assertEqual(ext["frames_kept"], 2)
            self.assertGreaterEqual(ext["skipped_duplicates"], 1)
            self.assertGreaterEqual(ext["skipped_blur"], 1)

            def mock_teacher(_img: np.ndarray) -> list[RawDetection]:
                return [
                    RawDetection("motorcycle", 0.90, np.array([[0.30, 0.55], [0.46, 0.55], [0.46, 0.78], [0.30, 0.78]])),
                    RawDetection("person", 0.85, np.array([[0.33, 0.40], [0.43, 0.40], [0.43, 0.72], [0.33, 0.72]])),
                    RawDetection("traffic light", 0.35, np.array([[0.80, 0.12], [0.85, 0.12], [0.85, 0.22], [0.80, 0.22]])),
                    RawDetection("truck", 0.88, np.array([[0.05, 0.50], [0.22, 0.50], [0.22, 0.75], [0.05, 0.75]])),
                ]

            pl = pseudo_label_folder(staged_vn / "images" / "train", staged_vn / "labels" / "train", mock_teacher, registry_path=None)
            self.assertEqual(pl["images_labeled"], 2)
            self.assertEqual(pl["auto_merged_riders"], 2)

            # 3. Augmenters: crop-zoom, weather, copy-paste
            bank = root / "rare_bank"
            _save_patterned_image(bank / "images" / "train" / "rare_01.jpg", (960, 720), seed=61)
            (bank / "labels" / "train").mkdir(parents=True)
            (bank / "labels" / "train" / "rare_01.txt").write_text(
                "3 0.20 0.55 0.36 0.55 0.36 0.78 0.20 0.78\n"
                "9 0.75 0.18 0.775 0.18 0.775 0.215 0.75 0.215\n",
                encoding="utf-8",
            )
            cz = generate_crop_zoom_dataset(bank, root / "aug_cz", crop_size=640, seed=39)
            self.assertGreaterEqual(cz["crops_written"], 1)
            we = generate_weather_dataset(staged_vn, root / "aug_weather", effects=("night", "rain"), max_images=2, seed=39)
            self.assertEqual(we["images_written"], 2)
            cp = generate_copypaste_dataset(bank, staged_vn, root / "aug_cp", rare_classes=(3, 9), n_output=2, seed=39)
            self.assertGreaterEqual(cp["images_written"], 1)

            # 4. Build Master Dataset + Tier-1 QC
            master_res = build_master_dataset(
                train_sources=[staged_vn, bank, root / "aug_cz", root / "aug_cp"],
                golden_val_dir=gval,
                master_dir=root / "master",
            )
            self.assertTrue(master_res["qc_ok"])
            self.assertEqual(master_res["val_images"], 1)
            self.assertGreaterEqual(master_res["train_images"], 4)

            # 5. Evaluate fallback scorer & Confident Learning QC
            img_p = gval / "images" / "val" / "gv_01.jpg"
            w, h = 960, 720
            m_car = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m_car, [np.array([[96, 360], [288, 360], [288, 540], [96, 540]])], 1)
            m_extra = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m_extra, [np.array([[480, 360], [580, 360], [580, 540], [480, 540]])], 1)
            pred_json = {
                "images": [{
                    "file": "gv_01.jpg",
                    "height": h,
                    "width": w,
                    "instances": [
                        {"c": 1, "s": 0.92, "rle": sc.rle.counts_to_string(sc.rle.mask_to_counts(m_car))},   # wrong class: Bus(1) vs GT Car(0)
                        {"c": 4, "s": 0.88, "rle": sc.rle.counts_to_string(sc.rle.mask_to_counts(m_extra))}, # missing label suspect
                    ],
                }]
            }
            score_res = _fallback_score_split(pred_json, gval / "labels" / "val", [img_p], cfg["classes"], rle_mod=sc.rle)
            self.assertIn("Car", score_res["per_class_ap"])
            diag = analyze_predictions_against_gt(pred_json, gval / "labels" / "val", [img_p], cfg["classes"], rle_mod=sc.rle)
            self.assertEqual(len(diag["confident_qc"]["wrong_class_suspects"]), 1)
            self.assertEqual(len(diag["confident_qc"]["missing_label_suspects"]), 1)


if __name__ == "__main__":
    unittest.main()

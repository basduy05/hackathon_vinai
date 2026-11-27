"""End-to-end unit tests for Phase 2 data pipelines, edge-case guards, augmenters, and Golden Val QC."""
from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
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

            # Verify merge_polygons_union handles slightly separated polygons (bridging branch)
            from src.seg_utils import merge_polygons_union
            p_a = np.array([[0.20, 0.50], [0.30, 0.50], [0.30, 0.60], [0.20, 0.60]])
            p_b = np.array([[0.21, 0.35], [0.29, 0.35], [0.29, 0.498], [0.21, 0.498]])
            bridged = merge_polygons_union([p_a, p_b])
            self.assertIsNotNone(bridged)

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

    def test_v2_polygon_repair_and_rider_bbox_mount_merge(self) -> None:
        from src.map_and_merge_public import apply_geometric_mask_merger, load_class_mapping
        from src.seg_utils import SegInstance, write_yolo_seg_label

        cfg = load_competition_config("configs/competition.yaml")
        map_cfg = load_class_mapping("configs/class_mapping.yaml")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _save_patterned_image(root / "images" / "train" / "v2_01.jpg", (640, 480), seed=77)
            # Bowtie self-intersecting car polygon -> must be auto-repaired inside SegInstance
            bowtie = np.array([[0.10, 0.50], [0.30, 0.70], [0.30, 0.50], [0.10, 0.70]])
            car_inst = SegInstance(cls=0, pts=bowtie, source_label="car")

            # Separated rider + motorcycle (0 polygon intersection, but bbox_iou >= 0.08 & centered above)
            moto = SegInstance(
                cls=4,
                pts=np.array([[0.45, 0.62], [0.65, 0.62], [0.65, 0.85], [0.45, 0.85]]),
                source_label="motorcycle",
            )
            rider = SegInstance(
                cls=8,
                pts=np.array([[0.50, 0.40], [0.60, 0.40], [0.60, 0.61], [0.50, 0.61]]),
                source_label="person",
            )
            # Extend rider bounding box slightly into moto seat via legs without polygon intersection
            rider_with_legs = SegInstance(
                cls=8,
                pts=np.array([[0.50, 0.40], [0.60, 0.40], [0.60, 0.61], [0.49, 0.72], [0.48, 0.72], [0.50, 0.60]]),
                source_label="person",
            )
            merged, ambig, mstats = apply_geometric_mask_merger(
                [car_inst, moto, rider_with_legs],
                map_cfg["geometric_merge_rules"],
                {"person", "rider"},
                "v2_01.jpg",
            )
            self.assertEqual(mstats["auto_merged"], 1)
            self.assertEqual(len(ambig), 0)
            self.assertEqual(sorted(it.cls for it in merged), [0, 5])

            write_yolo_seg_label(root / "labels" / "train" / "v2_01.txt", [(it.cls, it.pts) for it in merged])
            qc = run_qc(root, cfg)
            self.assertTrue(qc["ok"])
            self.assertEqual(qc["poly_issue_count"], 0)
            self.assertEqual(qc["edge_case_flag_counts"], {})

    def test_v2_kaggle_adapter_and_benchmark_report(self) -> None:
        from src.benchmark_report import _generate_charts, compute_benchmark_from_predictions, render_markdown_report
        from src.kaggle_adapter import auto_convert_detected_dataset, detect_dataset_format

        cfg = load_competition_config("configs/competition.yaml")
        sc = materialize_selfcheck("train_and_export.ipynb", "work/_selfcheck")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Mock Kaggle YOLO BBox dataset (like Vehicle-YOLO with threewheel, van, motorbike, person)
            kdir = root / "vehicle_dataset_for_yolo"
            _save_patterned_image(kdir / "train" / "images" / "veh_01.jpg", (640, 480), seed=91)
            (kdir / "train" / "labels").mkdir(parents=True)
            (kdir / "train" / "labels" / "veh_01.txt").write_text(
                "0 0.25 0.60 0.25 0.30\n1 0.70 0.55 0.20 0.35\n", encoding="utf-8"
            )
            (kdir / "data.yaml").write_text("names: ['threewheel', 'van']\n", encoding="utf-8")

            detected = detect_dataset_format(kdir)
            self.assertIsNotNone(detected)
            self.assertEqual(detected["format"], "yolo_bbox")

            def box_mask_pred(img_bgr: np.ndarray, boxes: list[list[float]]) -> list[np.ndarray]:
                h, w = img_bgr.shape[:2]
                out = []
                for x1, y1, x2, y2 in boxes:
                    m = np.zeros((h, w), dtype=np.uint8)
                    m[max(0, int(y1)) : min(h, int(y2)), max(0, int(x1)) : min(w, int(x2))] = 1
                    out.append(m)
                return out

            out_k = root / "staged_kaggle"
            conv = auto_convert_detected_dataset(detected, out_k, sam_predictor=box_mask_pred, registry_path=None)
            self.assertEqual(conv["instances_written"], 2)
            qc = run_qc(out_k, cfg)
            self.assertTrue(qc["ok"])
            self.assertEqual(qc["per_class"]["Other vehicle"]["instances"], 1)
            self.assertEqual(qc["per_class"]["Car"]["instances"], 1)

            # Test benchmark report rendering + chart generation
            val_dir = root / "val_eval"
            _save_patterned_image(val_dir / "images" / "v1.jpg", (640, 480), seed=92)
            (val_dir / "labels").mkdir(parents=True)
            (val_dir / "labels" / "v1.txt").write_text("3 0.15 0.45 0.35 0.45 0.35 0.75 0.15 0.75\n", encoding="utf-8")
            m_mask = np.zeros((480, 640), dtype=np.uint8)
            cv2.rectangle(m_mask, (96, 216), (224, 360), 1, -1)
            pred_payload = {
                "images": [{
                    "file": "v1.jpg",
                    "height": 480,
                    "width": 640,
                    "instances": [{"c": 3, "s": 0.91, "rle": sc.rle.counts_to_string(sc.rle.mask_to_counts(m_mask))}],
                }]
            }
            det_val = compute_benchmark_from_predictions(
                pred_payload, val_dir / "labels", [val_dir / "images" / "v1.jpg"], cfg["classes"], sc.rle
            )
            summary = {
                "run_name": "unit_test_v2",
                "team_id": 39,
                "dataset_dir": str(val_dir),
                "model_path": "unit.onnx",
                "classes": cfg["classes"],
                "qc_report": qc,
                "onnx_validation": {"ok": True, "nc": 11, "message_vi": "OK"},
                "zip_summary": {"zip_bytes": 10 * 1024 * 1024},
                "latency_ms_p50": 50.0,
                "decode_ms_p50": 10.0,
                "detscore_val": det_val,
                "ultralytics_val": None,
            }
            charts = _generate_charts(summary, root / "charts", baseline=summary)
            md = render_markdown_report(summary, baseline=summary, charts=charts)
            self.assertIn("unit_test_v2", md)
            self.assertIn("Other vehicle", md)
            self.assertTrue(len(charts) >= 1)

    def test_v2_local_pack_and_train_history_report(self) -> None:
        from unittest.mock import patch

        from src.benchmark_report import build_quick_local_pack, render_markdown_report, unpack_and_run_local_pack
        from src.validate_dataset import load_competition_config

        cfg = load_competition_config("configs/competition.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds_dir = root / "master"
            for idx in range(4):
                _save_patterned_image(ds_dir / "images" / "val" / f"v_{idx:02d}.jpg", (320, 240), seed=200 + idx)
                (ds_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
                cls_id = 3 if idx == 0 else (7 if idx == 1 else 0)
                (ds_dir / "labels" / "val" / f"v_{idx:02d}.txt").write_text(
                    f"{cls_id} 0.2 0.2 0.6 0.2 0.6 0.6 0.2 0.6\n", encoding="utf-8"
                )

            sub_zip = root / "submission.zip"
            sub_zip.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
            best_pt = root / "runs" / "exp" / "weights" / "best.pt"
            best_pt.parent.mkdir(parents=True)
            best_pt.write_bytes(b"dummy_pt")
            run_dir = root / "runs" / "exp"
            (run_dir / "results.csv").write_text(
                "epoch,metrics/mAP50-95(M)\n1,0.412\n2,0.456\n", encoding="utf-8"
            )
            (run_dir / "results.png").write_bytes(b"png")
            extra_md = root / "benchmark_report.md"
            extra_md.write_text("# Old", encoding="utf-8")

            pack_zip = build_quick_local_pack(
                out_zip=root / "quick_local_pack.zip",
                submission_or_onnx=sub_zip,
                dataset_dir=ds_dir,
                weights_pt=best_pt,
                extra_files=[extra_md],
                max_val_images=2,
                max_val_bytes=1,  # Force stratified selection
                seed=39,
            )
            self.assertTrue(pack_zip.is_file())
            with zipfile.ZipFile(pack_zip) as zf:
                names = set(zf.namelist())
                self.assertIn("submission.zip", names)
                self.assertIn("best.pt", names)
                self.assertIn("val_dataset/data.yaml", names)
                self.assertIn("train_artifacts/results.csv", names)
                self.assertIn("train_artifacts/results.png", names)
                self.assertIn("benchmark_report.md", names)
                # Both rarest classes (Other vehicle=3, Cyclist=7) should be prioritized in 2-image stratified sample
                self.assertIn("val_dataset/images/val/v_00.jpg", names)
                self.assertIn("val_dataset/images/val/v_01.jpg", names)

            with patch("src.benchmark_report.run_benchmark", return_value={"ok": True}) as mock_rb:
                unpack_and_run_local_pack(pack_zip, root / "local_out", run_name="local_v2")
                self.assertTrue(mock_rb.called)
                kw = mock_rb.call_args.kwargs
                self.assertEqual(kw["run_name"], "local_v2")
                self.assertTrue((kw["dataset_dir"] / "data.yaml").is_file())
                self.assertIn(str((root / "local_out" / "_unpacked" / "val_dataset").resolve()), (kw["dataset_dir"] / "data.yaml").read_text(encoding="utf-8"))
                self.assertEqual(kw["train_run_dir"], root / "local_out" / "_unpacked" / "train_artifacts")

            summary = {
                "run_name": "hist_test",
                "team_id": 39,
                "dataset_dir": str(ds_dir),
                "model_path": str(sub_zip),
                "classes": cfg["classes"],
                "qc_report": {
                    "splits": {"val": {"images": 4, "instances": 4, "empty_images": 0}},
                    "fatal_count": 0,
                    "poly_issue_count": 0,
                    "duplicates": {"leak_exact": [], "leak_near": []},
                    "ambiguity_queue_size": 0,
                    "edge_case_flag_counts": {},
                },
                "onnx_validation": {
                    "ok": True, "nc": 11, "message_vi": "OK",
                    "details": {"n_nodes": 521, "conv_nodes": 120, "total_params": 2700000, "opset": 17, "bytes": 10800000},
                },
                "zip_summary": {"zip_bytes": 10000000},
                "latency_ms_p50": 42.0,
                "decode_ms_p50": 6.5,
                "output_bytes": 240000,
                "train_history": {"epochs_ran": 2, "best_epoch": 2, "best_mask_map50_95": 0.456},
                "ultralytics_val": {
                    "box": {"map50": 0.65, "map50_95": 0.45, "mp": 0.70, "mr": 0.60},
                    "seg": {"map50": 0.61, "map75": 0.44, "map50_95": 0.43, "mp": 0.68, "mr": 0.58, "mean_f1_at_max": 0.62, "optimal_conf_at_max_f1": 0.35, "per_class": {}},
                },
                "detscore_val": {
                    "evaluator": "detscore", "n_images": 4,
                    "detscore": {"score": 44.5, "map50": 0.62, "n_gt_per_class": {}, "per_class_ap": {}, "per_class_ap50": {}},
                    "at_conf_025_iou_050": {
                        "precision": 0.71, "recall": 0.61, "f1": 0.656,
                        "tp_mask_iou_stats": {"mean": 0.81, "median": 0.83},
                        "tp_box_iou_stats": {"mean": 0.85, "median": 0.87},
                        "tp_conf_stats": {"mean": 0.74}, "fp_conf_stats": {"mean": 0.32},
                        "per_class_prf": {},
                    },
                    "best_f1_sweep": {"conf": 0.35, "f1": 0.67},
                    "conf_sweep": [{"conf": 0.25, "tp": 10, "fp": 4, "fn": 6, "precision": 0.71, "recall": 0.61, "f1": 0.656}],
                },
            }
            md = render_markdown_report(summary, baseline=None, charts=["results.png"])
            self.assertIn("0.4560", md)
            self.assertIn("521", md)
            self.assertIn("Mask `mAP75`", md)
            self.assertIn("![results.png](results.png)", md)

    def test_v3_kaggle_adapter_classes_txt_failfast_and_zip_unwrapper(self) -> None:
        from src.kaggle_adapter import (
            auto_convert_detected_dataset,
            convert_yolo_dataset,
            detect_dataset_format,
            detect_val_split,
            discover_all_kaggle_inputs,
            unzip_archives_in_dir,
        )

        cfg = load_competition_config("configs/competition.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 1. Nested Kaggle mount (/kaggle/input/datasets/owner/slug) + classes.txt (no data.yaml)
            ds_root = root / "kaggle_in" / "datasets" / "owner" / "vehicle_yolo"
            _save_patterned_image(ds_root / "train" / "images" / "v1.jpg", (640, 480), seed=301)
            (ds_root / "train" / "labels").mkdir(parents=True)
            # raw classes: 0=car, 1=threewheel, 2=bus, 3=truck, 4=motorbike, 5=van
            (ds_root / "train" / "labels" / "v1.txt").write_text(
                "1 0.25 0.60 0.20 0.30\n5 0.70 0.55 0.20 0.35\n3 0.50 0.50 0.15 0.20\n", encoding="utf-8"
            )
            _save_patterned_image(ds_root / "valid" / "images" / "vv1.jpg", (640, 480), seed=302)
            (ds_root / "valid" / "labels").mkdir(parents=True)
            (ds_root / "valid" / "labels" / "vv1.txt").write_text("1 0.3 0.5 0.2 0.2\n", encoding="utf-8")
            (ds_root / "classes.txt").write_text("car\nthreewheel\nbus\ntruck\nmotorbike\nvan\n", encoding="utf-8")

            found = discover_all_kaggle_inputs(root / "kaggle_in")
            self.assertEqual(len(found), 1)
            self.assertNotEqual(found[0]["name"], "datasets")
            val_info = detect_val_split(found[0])
            self.assertIsNotNone(val_info)

            def box_mask_pred(img_bgr: np.ndarray, boxes: list[list[float]]) -> list[np.ndarray]:
                h, w = img_bgr.shape[:2]
                return [
                    cv2.rectangle(np.zeros((h, w), dtype=np.uint8), (max(0, int(b[0])), max(0, int(b[1]))), (min(w - 1, int(b[2])), min(h - 1, int(b[3]))), 1, -1)
                    for b in boxes
                ]

            out_ds = root / "out_cls"
            res = auto_convert_detected_dataset(found[0], out_ds, sam_predictor=box_mask_pred, registry_path=None)
            self.assertEqual(res["instances_written"], 3)
            qc = run_qc(out_ds, cfg)
            # threewheel -> Other vehicle(3), van -> Car(0), truck -> Truck(2), Motorcyclist must be 0!
            self.assertEqual(qc["per_class"]["Other vehicle"]["instances"], 1)
            self.assertEqual(qc["per_class"]["Car"]["instances"], 1)
            self.assertEqual(qc["per_class"]["Truck"]["instances"], 1)
            self.assertEqual(qc["per_class"]["Motorcyclist"]["instances"], 0)

            # 2. Missing both data.yaml and classes.txt must raise ValueError (never silent 0..10 map)
            bad_dir = root / "no_meta"
            _save_patterned_image(bad_dir / "images" / "x.jpg", (320, 240), seed=303)
            (bad_dir / "labels").mkdir(parents=True)
            (bad_dir / "labels" / "x.txt").write_text("3 0.5 0.5 0.3 0.3\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                convert_yolo_dataset(
                    bad_dir / "images", bad_dir / "labels", root / "bad_out",
                    fmt="yolo_bbox", predictor=box_mask_pred, source_name="no_meta", registry_path=None,
                )

            # 3. Zip extraction + VN traffic sign code prefixes (P.102, W.201a, DP.135) -> Sign(9)
            z_src = root / "vnts_repo"
            z_src.mkdir()
            (z_src / "illustration.webp").write_bytes(b"fake")
            z_path = z_src / "traffic-sign-detection-vietnam.zip"
            with zipfile.ZipFile(z_path, "w") as zf:
                zf.writestr("dataset/data.yaml", "names: ['DP.135', 'P.102', 'W.201a']\n")
                _save_patterned_image(root / "tmp_im.jpg", (320, 240), seed=304)
                zf.write(root / "tmp_im.jpg", "dataset/train/images/s1.jpg")
                zf.writestr("dataset/train/labels/s1.txt", "1 0.5 0.4 0.15 0.2\n")
            unzip_archives_in_dir(z_src, remove_zip=True)
            self.assertFalse(z_path.exists())
            info_vnts = detect_dataset_format(z_src)
            self.assertIsNotNone(info_vnts)
            out_vnts = root / "out_vnts"
            auto_convert_detected_dataset(info_vnts, out_vnts, sam_predictor=box_mask_pred, registry_path=None)
            qc_v = run_qc(out_vnts, cfg)
            self.assertEqual(qc_v["per_class"]["Sign"]["instances"], 1)

    def test_v3_vn_hood_filter_and_pretrain_sanity_gate(self) -> None:
        from src.eval_golden import check_semantic_consistency_against_coco, render_per_class_preview_grid

        cfg = load_competition_config("configs/competition.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 1. Ego-hood Car filter & spurious Pedestrian inside Motorcyclist suppression
            raw_in = root / "vn_in"
            _save_patterned_image(raw_in / "frame_01.jpg", (640, 360), seed=310)
            out_vn = root / "vn_out"
            hood_poly = np.array([[0.15, 0.72], [0.85, 0.72], [0.90, 0.99], [0.10, 0.99]])
            real_car = np.array([[0.20, 0.40], [0.40, 0.40], [0.40, 0.60], [0.20, 0.60]])
            moto_poly = np.array([[0.65, 0.45], [0.80, 0.45], [0.80, 0.75], [0.65, 0.75]])
            rider_poly = np.array([[0.67, 0.35], [0.78, 0.35], [0.78, 0.62], [0.67, 0.62]])
            extra_ped_inside = np.array([[0.69, 0.37], [0.76, 0.37], [0.76, 0.56], [0.69, 0.56]])

            def teacher_fn(_bgr: np.ndarray) -> list[RawDetection]:
                return [
                    RawDetection("car", 0.55, hood_poly),
                    RawDetection("car", 0.91, real_car),
                    RawDetection("motorcycle", 0.85, moto_poly),
                    RawDetection("person", 0.88, rider_poly),
                    RawDetection("person", 0.42, extra_ped_inside),
                ]

            pl_res = pseudo_label_folder(
                raw_in, out_vn / "labels" / "train", predictor=teacher_fn, source_mapping_name="coco", registry_path=None
            )
            self.assertEqual(pl_res["hood_cars_dropped"], 1)
            # Both person boxes align over motorcycle -> merged/suppressed into single Motorcyclist, leaving 1 Car + 1 Motorcyclist
            lbl_lines = [l for l in (out_vn / "labels" / "train" / "frame_01.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
            cls_ids = [int(l.split()[0]) for l in lbl_lines]
            self.assertEqual(sorted(cls_ids), [0, 5])

            # 2. Pre-train Semantic Sanity Gate catches van->Motorcyclist & truck->Other vehicle
            bad_samples = [
                {
                    "image": f"bad_{i}.jpg",
                    "gt": [(5, (0.1, 0.2, 0.8, 0.8)), (3, (0.1, 0.2, 0.8, 0.8))],
                    "coco_pred": [("car", 0.9, (0.1, 0.2, 0.8, 0.8)), ("truck", 0.88, (0.1, 0.2, 0.8, 0.8))],
                }
                for i in range(4)
            ]
            bad_verdict = check_semantic_consistency_against_coco(bad_samples, cfg["classes"])
            self.assertFalse(bad_verdict["ok"])
            self.assertTrue(len(bad_verdict["alerts"]) >= 1)

            good_samples = [
                {
                    "image": f"good_{i}.jpg",
                    "gt": [(0, (0.1, 0.2, 0.4, 0.5)), (5, (0.5, 0.3, 0.7, 0.7))],
                    "coco_pred": [("car", 0.92, (0.1, 0.2, 0.4, 0.5)), ("motorcycle", 0.85, (0.5, 0.3, 0.7, 0.7))],
                }
                for i in range(4)
            ]
            good_verdict = check_semantic_consistency_against_coco(good_samples, cfg["classes"])
            self.assertTrue(good_verdict["ok"])

            # 3. Per-class grid & area_buckets in run_qc
            _save_patterned_image(out_vn / "images" / "train" / "frame_01.jpg", (640, 360), seed=310)
            qc = run_qc(out_vn, cfg)
            self.assertIn("area_buckets", qc["per_class"]["Car"])
            self.assertIn("by_split_ambiguity", qc)
            grid_path = render_per_class_preview_grid(out_vn, root / "grid.jpg", split="train", samples_per_class=2)
            self.assertTrue(grid_path.is_file() and grid_path.stat().st_size > 1000)


if __name__ == "__main__":
    unittest.main()

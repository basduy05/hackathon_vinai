"""Regression proofs for V3.1 holdout integrity, source gates, and augmentation banks."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.augment_and_crop import generate_copypaste_dataset
from src.data_gates import build_golden_holdout, exclude_golden_from_staged_train
from src.eval_golden import check_semantic_consistency_against_coco, run_pretrain_semantic_sanity_check
from src.prepare_vn_frames import RawDetection, pseudo_label_folder
from src.validate_dataset import load_competition_config


def stage(root, split, stem, classes, seed=1, tiny=False):
    image = root / "images" / split / f"{stem}.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (640, 640, 3), dtype=np.uint8)).save(image)
    label = root / "labels" / split / f"{stem}.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    poly = "0.10 0.10 0.101 0.10 0.101 0.101 0.10 0.101" if tiny else "0.10 0.10 0.25 0.10 0.25 0.25 0.10 0.25"
    label.write_text("".join(f"{c} {poly}\n" for c in classes), encoding="utf-8")
    return image


class V31Tests(unittest.TestCase):
    def test_holdout_deduplicates_and_excludes_leaks_before_augmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orig = stage(root / "val_source", "val", "original", range(11))
            dup = stage(root / "train_source", "train", "renamed", range(11), seed=2)
            shutil.copy2(orig, dup)
            unrelated = stage(root / "train_source", "train", "other", (0,), seed=3)
            rep = build_golden_holdout([root / "val_source"], [root / "train_source"], root / "golden",
                                       min_per_class=1, target_images=1)
            self.assertTrue(rep["ok"])
            self.assertEqual(rep["images"], 1)
            self.assertEqual(rep["manifest"][0]["original_split"], "val")
            removed = exclude_golden_from_staged_train([root / "train_source"], root / "golden")
            self.assertEqual(sum(removed.values()), 1)
            self.assertTrue(orig.exists() and unrelated.exists())
            self.assertFalse(dup.exists())
            self.assertFalse((root / "train_source/labels/train/renamed.txt").exists())
            with self.assertRaises(ValueError):
                build_golden_holdout([], [], root / "golden")

    def test_holdout_does_not_count_tiny_or_duplicate_objects_toward_quota(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = stage(root / "v", "val", "a", range(11))
            duplicate = stage(root / "v", "val", "b", range(11), seed=2)
            shutil.copy2(original, duplicate)
            stage(root / "v", "val", "tiny", range(11), seed=3, tiny=True)
            rep = build_golden_holdout([root / "v"], [], root / "out", min_per_class=2, target_images=1)
            self.assertFalse(rep["ok"])
            self.assertEqual(rep["images"], 1)
            self.assertEqual(rep["skipped_duplicates"], 1)
            self.assertEqual(set(rep["missing_instances"].values()), {1})

    def test_holdout_recomputes_deficits_and_uses_train_only_when_needed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage(root / "v", "val", "common", range(10))
            stage(root / "t", "train", "rare", (10,), seed=2)
            rep = build_golden_holdout([root / "v"], [root / "t"], root / "out", min_per_class=1, target_images=1)
            self.assertTrue(rep["ok"])
            self.assertEqual(rep["images"], 2)  # Image target must not override missing-class quotas.
            self.assertEqual(rep["manifest"][-1]["original_split"], "train")

    def test_source_gate_never_passes_without_evidence_and_marks_sign_limit(self):
        cfg = load_competition_config("configs/competition.yaml")
        self.assertFalse(check_semantic_consistency_against_coco([], cfg["classes"])["ok"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage(root / "vehicles", "train", "a", (0,))
            rep = run_pretrain_semantic_sanity_check(root / "vehicles", coco_predictor=lambda _: [])
            self.assertEqual(rep["status"], "insufficient_evidence")
            stage(root / "signs", "train", "b", (9,), seed=2)
            rep = run_pretrain_semantic_sanity_check(root / "signs", coco_predictor=lambda _: [])
            self.assertEqual(rep["status"], "not_applicable")
            self.assertIsNone(rep["ok"])

    def test_multisource_bank_contains_signs_and_rare_vehicles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage(root / "idd", "train", "a", (3, 6, 7))
            stage(root / "vnts", "train", "b", (9,), seed=2)
            stage(root / "bg", "train", "c", (), seed=3)
            rep = generate_copypaste_dataset([root / "idd", root / "vnts"], root / "bg", root / "out", n_output=3)
            self.assertEqual(set(rep["bank_instances_per_class"]), {"3", "6", "7", "9"})
            self.assertGreater(rep["pasted_instances"], 0)

    def test_bdd_val_dispatch_preserves_split_and_daytime_frames(self):
        from src.kaggle_adapter import auto_convert_detected_dataset, detect_val_split
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage(root, "train", "unused", ())
            stage(root, "val", "day", ())
            annotations = [{"name": "day.png", "attributes": {"timeofday": "daytime", "weather": "clear"},
                            "labels": [{"category": "car", "box2d": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}}]}]
            tr = root / "labels_train.json"
            vl = root / "labels_val.json"
            tr.write_text(json.dumps(annotations))
            vl.write_text(json.dumps(annotations))
            info = {"format": "bdd100k_json", "name": "bdd100k", "images_dir": str(root / "images/train"), "labels_path": str(tr)}
            val_info = detect_val_split(info)
            self.assertIsNotNone(val_info)
            def masks(bgr, boxes):
                result = []
                for box in boxes:
                    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
                    x1, y1, x2, y2 = map(int, box)
                    mask[y1:y2, x1:x2] = 1
                    result.append(mask)
                return result
            res = auto_convert_detected_dataset(val_info, root / "out", sam_predictor=masks, split="val", registry_path=None)
            self.assertEqual(res["instances_written"], 1)
            self.assertTrue((root / "out/images/val/day.png").exists())
            self.assertFalse((root / "out/images/train").exists())

    def test_pseudo_aux_sign_and_post_merge_area_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            im = stage(root / "raw", "train", "a", ())
            small = np.array([[.1, .1], [.108, .1], [.108, .108], [.1, .108]])  # 26px²: prefilter keeps, postfilter drops
            sign = np.array([[.5, .1], [.6, .1], [.6, .2], [.5, .2]])
            res = pseudo_label_folder(
                im.parent, root / "labels", lambda _: [RawDetection("car", .9, small)],
                aux_sign_predictor=lambda _: [RawDetection("stop sign", .9, sign), RawDetection("car", .9, sign)],
            )
            lines = (root / "labels/a.txt").read_text().splitlines()
            self.assertEqual(res["instances_written"], 1)
            self.assertEqual([int(x.split()[0]) for x in lines], [9])

    def test_other_vehicle_coco_confusion_is_soft_warning_while_real_corruption_blocks(self) -> None:
        cfg = load_competition_config("configs/competition.yaml")
        # Autorickshaw/trailer (class 3) predicted by COCO as car/truck must pass with soft review notes
        idd_like = [
            {
                "image": f"idd_{i}.jpg",
                "gt": [(3, (0.1, 0.2, 0.4, 0.5)), (0, (0.5, 0.2, 0.8, 0.6))],
                "coco_pred": [("truck" if i % 2 else "car", 0.72, (0.1, 0.2, 0.4, 0.5)), ("car", 0.90, (0.5, 0.2, 0.8, 0.6))],
            }
            for i in range(4)
        ]
        rep = check_semantic_consistency_against_coco(idd_like, cfg["classes"])
        self.assertTrue(rep["ok"])
        self.assertEqual(rep["status"], "pass")
        self.assertTrue(len(rep["soft_warnings"]) >= 1)
        self.assertEqual(sum(rep["soft_by_pair"].values()), 4)

        # Pedestrian labeled as Other vehicle must still hard-fail
        bad_person = [
            {
                "image": f"ped_{i}.jpg",
                "gt": [(3, (0.1, 0.2, 0.3, 0.8))],
                "coco_pred": [("person", 0.90, (0.1, 0.2, 0.3, 0.8))],
            }
            for i in range(4)
        ]
        bad_rep = check_semantic_consistency_against_coco(bad_person, cfg["classes"])
        self.assertFalse(bad_rep["ok"])
        self.assertEqual(bad_rep["status"], "fail")

    def test_discover_kaggle_inputs_skips_legacy_hackathon_output(self) -> None:
        from src.kaggle_adapter import discover_all_kaggle_inputs
        with tempfile.TemporaryDirectory() as tmp:
            k_in = Path(tmp) / "input"
            stage(k_in / "hackathon" / "master_v2", "train", "v2_bad", (5,))
            (k_in / "hackathon" / "master_v2" / "data.yaml").write_text("names: [Car]\nnc: 1\n", encoding="utf-8")
            stage(k_in / "datasets" / "owner" / "clean-ds", "train", "good", (0,))
            (k_in / "datasets" / "owner" / "clean-ds" / "data.yaml").write_text("names: [Car]\nnc: 1\n", encoding="utf-8")

            found = discover_all_kaggle_inputs(k_in, skip_names={"vehicle-dataset-for-yolo", "hackathon", "master_v2"})
            names = [d["name"] for d in found]
            self.assertEqual(names, ["clean-ds"])

    def test_sam2_prefilters_tiny_boxes_before_predictor(self) -> None:
        from src.box_to_mask_sam2 import convert_bbox_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            im1 = stage(root / "src", "train", "mixed", ())
            im2 = stage(root / "src", "train", "tiny_only", (), seed=2)
            # mixed: 1 valid 160x160px box + 1 tiny 2x2px (<40px2); tiny_only: 1 tiny 2x2px
            (root / "src/labels/train/mixed.txt").write_text(
                "0 0.5 0.5 0.25 0.25\n0 0.1 0.1 0.003 0.003\n", encoding="utf-8"
            )
            (root / "src/labels/train/tiny_only.txt").write_text(
                "0 0.1 0.1 0.003 0.003\n", encoding="utf-8"
            )
            calls: list[int] = []

            def recording_predictor(bgr: np.ndarray, boxes: list[list[float]]) -> list[np.ndarray]:
                calls.append(len(boxes))
                masks = []
                for x1, y1, x2, y2 in boxes:
                    m = np.zeros(bgr.shape[:2], dtype=np.uint8)
                    m[int(y1):int(y2), int(x1):int(x2)] = 1
                    masks.append(m)
                return masks

            res = convert_bbox_dataset(
                im1.parent, root / "src/labels/train", root / "out",
                predictor=recording_predictor, registry_path=None, source_name="prefilter_test",
            )
            # Only the 1 large box reaches SAM2; tiny_only skips SAM2 completely
            self.assertEqual(calls, [1])
            self.assertEqual(res["instances_written"], 1)

    def test_coco_and_yolo_bounded_selection_prioritize_rare_classes(self) -> None:
        from src.kaggle_adapter import _fast_list_images, _select_prioritized_yolo_images
        from src.map_and_merge_public import _parse_coco_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 1. Bounded YOLO selection stops cleanly at scan_budget and retains rare classes
            imgs_dir = root / "yolo/images/train"
            lbls_dir = root / "yolo/labels/train"
            imgs_dir.mkdir(parents=True)
            lbls_dir.mkdir(parents=True)
            for i in range(40):
                (imgs_dir / f"im_{i:03d}.png").write_bytes(b"")
                cls_id = 9 if i in (5, 19, 31) else 0
                (lbls_dir / f"im_{i:03d}.txt").write_text(f"{cls_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            listed = _fast_list_images(imgs_dir)
            self.assertEqual(len(listed), 40)
            picked = _select_prioritized_yolo_images(
                listed, imgs_dir, lbls_dir, {0: 0, 9: 9}, None, max_images=6, prioritize_classes=(9,), seed=39
            )
            self.assertEqual(len(picked), 6)
            picked_names = {p.name for p in picked}
            self.assertTrue({"im_005.png", "im_019.png", "im_031.png"}.issubset(picked_names))

            # 2. COCO metadata pre-selection skips decoding unselected records when max_images is set
            coco_imgs = root / "coco_imgs"
            stage(root / "tmp_coco", "train", "c0", ())
            coco_imgs.mkdir(parents=True)
            for name in ("car1.png", "car2.png", "bike.png"):
                Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(coco_imgs / name)
            ann_file = root / "instances.json"
            ann_file.write_text(json.dumps({
                "categories": [{"id": 3, "name": "car"}, {"id": 2, "name": "bicycle"}],
                "images": [
                    {"id": 1, "file_name": "car1.png", "width": 100, "height": 100},
                    {"id": 2, "file_name": "car2.png", "width": 100, "height": 100},
                    {"id": 3, "file_name": "bike.png", "width": 100, "height": 100},
                ],
                "annotations": [
                    {"id": 10, "image_id": 1, "category_id": 3, "iscrowd": 0, "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]]},
                    {"id": 11, "image_id": 2, "category_id": 3, "iscrowd": 0, "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]]},
                    {"id": 12, "image_id": 3, "category_id": 2, "iscrowd": 0, "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]]},
                ],
            }), encoding="utf-8")
            recs = _parse_coco_dataset(ann_file, coco_imgs, {"car": 0, "bicycle": 6}, max_images=1, prioritize_classes=(6,))
            self.assertEqual(recs[0]["image_path"].name, "bike.png")


if __name__ == "__main__":
    unittest.main()

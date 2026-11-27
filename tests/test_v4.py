"""Regression proofs for V4 root-cause fixes: Mapillary v2.0, COCO EOF JSON, partial-label completion, Crop-Zoom, and build_master."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.augment_and_crop import generate_crop_zoom_dataset, generate_weather_dataset
from src.build_master import build_master_dataset
from src.kaggle_adapter import auto_convert_detected_dataset, detect_dataset_format, detect_val_split
from src.prepare_vn_frames import RawDetection, complete_partial_labels_with_teacher
from src.seg_utils import read_yolo_seg_file


def _save_img(path: Path, size: tuple[int, int] = (640, 640), seed: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


class V4Tests(unittest.TestCase):
    def test_detect_coco_json_with_categories_at_eof(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "coco_ds" / "coco2017"
            _save_img(root / "train2017" / "000001.jpg", (320, 240), seed=5)
            ann_dir = root / "annotations"
            ann_dir.mkdir(parents=True, exist_ok=True)
            # Push "categories" far past 1200 bytes (as in real COCO instances_train2017.json)
            payload = {
                "info": {"description": "X" * 12000},
                "images": [{"id": 1, "file_name": "000001.jpg", "width": 320, "height": 240}],
                "annotations": [],
                "categories": [{"id": 1, "name": "person"}],
            }
            (ann_dir / "instances_train2017.json").write_text(json.dumps(payload), encoding="utf-8")
            info = detect_dataset_format(Path(tmp) / "coco_ds")
            self.assertIsNotNone(info)
            self.assertEqual(info["format"], "coco_json")

    def test_mapillary_v2_detection_and_polygon_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mroot = Path(tmp) / "mapillary_vistas" / "Mapillary Vistas"
            mroot.mkdir(parents=True, exist_ok=True)
            (mroot / "config_v2.0.json").write_text(json.dumps({"labels": []}), encoding="utf-8")
            _save_img(mroot / "training" / "images" / "m_01.jpg", (1600, 1200), seed=10)
            _save_img(mroot / "validation" / "images" / "mv_01.jpg", (1600, 1200), seed=11)
            poly_dir = mroot / "training" / "v2.0" / "polygons"
            val_poly_dir = mroot / "validation" / "v2.0" / "polygons"
            poly_dir.mkdir(parents=True, exist_ok=True)
            val_poly_dir.mkdir(parents=True, exist_ok=True)

            objs = [
                {"label": "human--rider--motorcyclist", "polygon": [[100, 300], [300, 300], [300, 700], [100, 700]]},
                {"label": "human--rider--bicyclist", "polygon": [[400, 300], [550, 300], [550, 700], [400, 700]]},
                {"label": "object--vehicle--other-vehicle", "polygon": [[650, 400], [950, 400], [950, 800], [650, 800]]},
                {"label": "human--person--individual", "polygon": [[1050, 400], [1150, 400], [1150, 800], [1050, 800]]},
                {"label": "object--traffic-sign--front", "polygon": [[1200, 100], [1350, 100], [1350, 260], [1200, 260]]},
                {"label": "object--traffic-light--general-single", "polygon": [[1400, 80], [1500, 80], [1500, 250], [1400, 250]]},
            ]
            (poly_dir / "m_01.json").write_text(json.dumps({"width": 1600, "height": 1200, "objects": objs}), encoding="utf-8")
            (val_poly_dir / "mv_01.json").write_text(json.dumps({"width": 1600, "height": 1200, "objects": objs}), encoding="utf-8")

            info = detect_dataset_format(Path(tmp) / "mapillary_vistas")
            self.assertIsNotNone(info)
            self.assertEqual(info["format"], "mapillary_vistas")
            val_info = detect_val_split(info)
            self.assertIsNotNone(val_info)

            out_dir = Path(tmp) / "staged_map"
            res = auto_convert_detected_dataset(info, out_dir, sam_predictor=None, split="train", registry_path=None)
            self.assertEqual(res["kept_images"], 1)
            self.assertEqual(
                set(res["class_counts"].keys()),
                {"Motorcyclist", "Cyclist", "Other vehicle", "Pedestrian", "Sign", "Traffic light"},
            )
            # Verify max_image_side=1280 downscale preserved polygons
            im_out = next((out_dir / "images" / "train").glob("*"))
            with Image.open(im_out) as im:
                self.assertEqual(max(im.size), 1280)

    def test_partial_label_completion_preserves_gt_and_adds_missing_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "vnts_sam2"
            _save_img(ds / "images" / "train" / "vn_01.jpg", (640, 640), seed=21)
            lbl = ds / "labels" / "train" / "vn_01.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            # Human-verified Sign(9) at top-right
            lbl.write_text("9 0.70 0.10 0.85 0.10 0.85 0.28 0.70 0.28\n", encoding="utf-8")

            def mock_coco_teacher(_bgr):
                return [
                    # Overlapping sign -> must be filtered out because 9 is preserved
                    RawDetection("stop sign", 0.95, np.array([[0.71, 0.11], [0.84, 0.11], [0.84, 0.27], [0.71, 0.27]])),
                    # Non-overlapping Car(0)
                    RawDetection("car", 0.85, np.array([[0.10, 0.50], [0.35, 0.50], [0.35, 0.75], [0.10, 0.75]])),
                    # Person + Motorcycle -> merged to Motorcyclist(5)
                    RawDetection("person", 0.82, np.array([[0.48, 0.38], [0.58, 0.38], [0.58, 0.68], [0.48, 0.68]])),
                    RawDetection("motorcycle", 0.80, np.array([[0.47, 0.52], [0.59, 0.52], [0.59, 0.76], [0.47, 0.76]])),
                ]

            rep = complete_partial_labels_with_teacher(
                dataset_dir=ds,
                predictor=mock_coco_teacher,
                preserved_classes={9},
                fill_classes={0, 1, 2, 4, 5, 6, 7, 8, 10},
                split="train",
                source_mapping_name="coco",
            )
            self.assertEqual(rep["gt_kept"], 1)
            self.assertEqual(rep["teacher_added_total"], 2)
            insts = read_yolo_seg_file(lbl)
            self.assertEqual(sorted(it.cls for it in insts), [0, 5, 9])

    def test_crop_zoom_works_on_640px_vn_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "lvnduy_640"
            _save_img(src / "images" / "train" / "s1.jpg", (640, 640), seed=31)
            lbl = src / "labels" / "train" / "s1.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            # Small sign ~20x20px (400 px^2 at 640 scale)
            lbl.write_text("9 0.48 0.48 0.511 0.48 0.511 0.511 0.48 0.511\n", encoding="utf-8")
            cz = generate_crop_zoom_dataset(src, Path(tmp) / "cz_out", crop_size=640, seed=39)
            self.assertEqual(cz["crops_written"], 1)
            out_im = next((Path(tmp) / "cz_out" / "images" / "train").glob("*"))
            with Image.open(out_im) as im:
                self.assertEqual(im.size, (640, 640))

    def test_build_master_preserves_aug_despite_similar_dhash_but_blocks_val_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Golden Val image
            val_src = root / "golden_val"
            _save_img(val_src / "images" / "val" / "gv_01.jpg", (640, 640), seed=99)
            (val_src / "labels" / "val").mkdir(parents=True, exist_ok=True)
            (val_src / "labels" / "val" / "gv_01.txt").write_text("0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n", encoding="utf-8")

            # Train source image
            tr_src = root / "mapillary"
            _save_img(tr_src / "images" / "train" / "t_01.jpg", (640, 640), seed=41)
            (tr_src / "labels" / "train").mkdir(parents=True, exist_ok=True)
            (tr_src / "labels" / "train" / "t_01.txt").write_text("0 0.2 0.2 0.5 0.2 0.5 0.5 0.2 0.5\n", encoding="utf-8")

            # Weather augmentation on train image (dhash very close to t_01)
            aug_we = root / "aug_weather"
            generate_weather_dataset(tr_src, aug_we, effects=("glare",), max_images=1, seed=39)

            # Synthetic aug source that tries to leak gv_01
            aug_leak = root / "aug_copypaste"
            (aug_leak / "images" / "train").mkdir(parents=True, exist_ok=True)
            (aug_leak / "labels" / "train").mkdir(parents=True, exist_ok=True)
            Image.open(val_src / "images" / "val" / "gv_01.jpg").save(aug_leak / "images" / "train" / "leak.jpg")
            (aug_leak / "labels" / "train" / "leak.txt").write_text("0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n", encoding="utf-8")

            master = root / "master"
            res = build_master_dataset([tr_src, aug_we, aug_leak], val_src, master)
            self.assertEqual(res["train_images"], 2)  # t_01 + aug_weather preserved!
            self.assertEqual(res["skipped_duplicates_or_leak"], 1)  # gv_01 leak blocked!
            self.assertTrue(res["qc_ok"])

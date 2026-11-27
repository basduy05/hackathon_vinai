"""Offline proofs for V5; no pretrained model download or GPU training."""
import copy
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.prepare_vn_frames import RawDetection
from src.score_focus_v5 import (
    CorrectedSAM2Predictor, cap_generated_images, choose_official_candidate,
    complete_labels_with_consensus, holdout_fingerprint, make_repeat_factor_dataset,
    quarantine_uncertain,
)
from src.seg_utils import read_yolo_seg_file, write_data_yaml
from src.validate_dataset import load_competition_config
from tools.build_kaggle_v5_score_focus import SOURCE, TRAIN, build_notebook, unpack_bundle, verify


def save_image(root, name, ids, split="train", seed=1):
    path = root / "images" / split / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (160, 160, 3), dtype=np.uint8)).save(path)
    label = root / "labels" / split / f"{name}.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("".join(f"{c} .1 .1 .3 .1 .3 .3 .1 .3\n" for c in ids))
    return path


def candidate(name, score, fingerprint="fixed"):
    return {"name": name, "holdout_sha256": fingerprint, "n_images": 40,
            "score_summary": {"evaluator": "official_detscore", "score": score,
                              "n_gt_per_class": {str(c): 30 for c in range(11)}, "classes_without_gt": []}}


class ScoreFocusTests(unittest.TestCase):
    def test_sam_preserves_bgr_and_restores_original_mask_dimensions(self):
        predictor = CorrectedSAM2Predictor.__new__(CorrectedSAM2Predictor)
        predictor.device = "cpu"
        image = np.zeros((8, 12, 3), dtype=np.uint8)
        image[..., 0] = 210
        tensor = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.ones((1, 4, 6))))
        def model(received, **kwargs):
            np.testing.assert_array_equal(received, image)
            return [SimpleNamespace(masks=SimpleNamespace(data=tensor))]
        predictor.model = model
        masks = predictor(image, [[1, 1, 4, 4], [4, 4, 5, 5]])
        self.assertEqual([m.shape for m in masks], [(8, 12), (8, 12)])
        self.assertTrue(masks[0].all())
        self.assertFalse(masks[1].any())
        self.assertEqual(predictor(image, []), [])

    def test_consensus_keeps_gt_and_rejects_unstable_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_image(root / "partial", "frame", (9,))
            original_gt = read_yolo_seg_file(root / "partial/labels/train/frame.txt")[0].pts.copy()
            car = np.array([[.5,.5],[.8,.5],[.8,.8],[.5,.8]])
            bus = np.array([[.1,.5],[.3,.5],[.3,.8],[.1,.8]])
            calls = 0
            def teacher(_):
                nonlocal calls
                calls += 1
                if calls % 2:
                    return [RawDetection("car", .9, car), RawDetection("bus", .9, bus)]
                flipped = car.copy()
                flipped[:, 0] = 1 - flipped[:, 0]
                return [RawDetection("car", .9, flipped)]
            rep = complete_labels_with_consensus(root / "partial", teacher, {9}, {0,1})
            self.assertEqual(rep["teacher_added_by_class"], {"Car": 1})
            self.assertEqual(rep["records"][0]["unresolved_confident_objects"], 1)
            gt = read_yolo_seg_file(root / "partial/labels/train/frame.txt")
            self.assertEqual([it.cls for it in gt], [9,0])
            np.testing.assert_array_equal(gt[0].pts, original_gt)
            result = quarantine_uncertain(root / "partial", [rep], root / "quarantine")
            self.assertEqual(result["quarantined_images"], 1)
            self.assertTrue((root / "quarantine/partial/labels/train/frame.txt").exists())

    def test_consensus_does_not_add_pedestrian_inside_existing_rider(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_image(root, "frame", ())
            (root / "labels/train/frame.txt").write_text("5 .2 .2 .8 .2 .8 .9 .2 .9\n")
            ped = np.array([[.3,.3],[.4,.3],[.4,.5],[.3,.5]])
            def teacher(_):
                return [RawDetection("person", .95, ped)]
            rep = complete_labels_with_consensus(root, teacher, {5}, {8})
            self.assertEqual(rep["teacher_added_total"], 0)
            self.assertEqual(len(read_yolo_seg_file(root / "labels/train/frame.txt")), 1)

    def test_repeat_factor_keeps_all_classes_and_never_resamples_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            classes = load_competition_config("configs/competition.yaml")["classes"]
            write_data_yaml(root, classes, has_val=True)
            for index in range(30):
                save_image(root, f"mapillary_{index}", (0, 7) if index == 0 else (0,), seed=index)
            save_image(root, "aug_pseudo", (7,), seed=99)
            save_image(root, "val_frame", (7,), split="val")
            yaml_path = root / "weighted.yaml"
            report = make_repeat_factor_dataset(root, yaml_path, max_extra_fraction=.35)
            entries = yaml_path.with_suffix(".train.txt").read_text().splitlines()
            self.assertEqual(len(set(entries)), 31)
            self.assertGreater(len(entries), 31)
            self.assertLessEqual(len(entries), 31 + int(31 * .35))
            self.assertEqual(sum("aug_pseudo" in e for e in entries), 1)
            self.assertFalse(any("val_frame" in e for e in entries))
            self.assertEqual(report["evaluable_instances"][0], 30)
            with self.assertRaises(ValueError):
                make_repeat_factor_dataset(root, root / "clean.yaml", clean_only=True)

    def test_official_selection_keeps_main_when_refinement_degrades(self):
        entries = [candidate("main", 35), candidate("refine", 34)]
        self.assertEqual(choose_official_candidate(entries)["name"], "main")
        invalid = copy.deepcopy(entries)
        invalid[1]["score_summary"]["evaluator"] = "local_cv2_fallback"
        with self.assertRaises(ValueError):
            choose_official_candidate(invalid)
        entries[1]["holdout_sha256"] = "other"
        with self.assertRaises(ValueError):
            choose_official_candidate(entries)

    def test_fingerprint_changes_when_validation_labels_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_image(root, "v", (0,), split="val")
            before = holdout_fingerprint(root)
            (root / "labels/val/v.txt").write_text("1 .1 .1 .3 .1 .3 .3 .1 .3\n")
            self.assertNotEqual(before, holdout_fingerprint(root))

    def test_crop_cap_keeps_image_label_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(4):
                save_image(root, str(i), (7,), seed=i)
            self.assertEqual(cap_generated_images(root, 2), 2)
            self.assertEqual({p.stem for p in (root / "images/train").glob("*")},
                             {p.stem for p in (root / "labels/train").glob("*")})

    def test_notebook_is_independent_and_preserves_all_v4_payload_files(self):
        before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        old = json.loads(SOURCE.read_text(encoding="utf-8"))
        new = build_notebook()
        self.assertEqual(verify(new), 19)
        old_bundle = unpack_bundle(old)[2]
        new_bundle = unpack_bundle(new)[2]
        self.assertTrue(all(new_bundle[k] == v for k,v in old_bundle.items()))
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), before)
        self.assertNotIn('CLEAN_TRAIN_SOURCES.append(vnts_out)', ''.join(new["cells"][4]["source"]))
        self.assertIn('CorrectedSAM2Predictor', ''.join(new["cells"][4]["source"]))
        self.assertIn('master_v5', ''.join(new["cells"][1]["source"]))

    def test_train_cell_routes_exports_to_best_official_candidate(self):
        """Execute orchestration with fake train/export; no GPU or real model is used."""
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            master, outputs = root / "master", root / "outputs"
            outputs.mkdir()
            class FakeYOLO:
                def __init__(self, *_):
                    self.trainer = None
                def train(self, **kwargs):
                    best = root / kwargs["name"] / "weights/best.pt"
                    best.parent.mkdir(parents=True, exist_ok=True)
                    best.write_bytes(b"test checkpoint")
                    self.trainer = SimpleNamespace(best=best)
            def export(weights, out, **kwargs):
                out.mkdir(parents=True, exist_ok=True)
                for file in ("model.onnx", "submission.zip"):
                    (out / file).write_bytes(b"validated fake")
                return {"onnx": str(out / "model.onnx"), "zip": str(out / "submission.zip")}
            def evaluate(model, *args, **kwargs):
                entry = candidate("unused", 20 if model.parent.name == "main" else 19)
                return {"n_images": entry["n_images"], "score_summary": entry["score_summary"]}
            scope = dict(MASTER_DIR=master, OUTPUTS_DIR=outputs, REPO_DIR=root, IS_KAGGLE=False,
                         RUN_TRAIN_IN_THIS_NOTEBOOK=True, RUN_CLEAN_REFINEMENT=True, TRAIN_EPOCHS=1,
                         REFINE_EPOCHS=1, TRAIN_BATCH=1, TRAIN_PATIENCE=1, DEVICE="cpu", TEAM_ID=39,
                         RUN_NAME="mock", torch=torch, shutil=__import__("shutil"), json=json, Path=Path)
            with patch("ultralytics.YOLO", FakeYOLO), patch("src.export_verify_onnx.export_and_verify", export), \
                 patch("src.eval_golden.run_eval_or_qc", evaluate), \
                 patch("src.score_focus_v5.make_repeat_factor_dataset", return_value={}), \
                 patch("src.score_focus_v5.holdout_fingerprint", return_value="fixed"):
                exec(compile(TRAIN, "v5_train_cell", "exec"), scope)
            self.assertEqual(json.loads((outputs / "checkpoint_selection.json").read_text())["selected"], "main")
            self.assertTrue((outputs / "submission.zip").is_file())
            self.assertEqual(scope["SELECTED_TRAIN_DIR"].name, "mock_main")


if __name__ == "__main__":
    unittest.main()

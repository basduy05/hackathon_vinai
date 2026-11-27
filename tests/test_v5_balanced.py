"""Focused checks for the balanced notebook's time budget and fallback submission."""
import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.balanced_budget import StopAtDeadline, phase_hours
from tools.build_kaggle_v5_balanced import build_notebook, verify


class BalancedTests(unittest.TestCase):
    def test_budget_includes_preparation_and_reserves_export(self):
        self.assertEqual(phase_hours(100, 7.5, .75, 3., now=100+3600), 3.)
        self.assertAlmostEqual(phase_hours(100, 7.5, .75, 3., now=100+6*3600), .75)
        self.assertEqual(phase_hours(100, 7.5, .75, 3., now=100+8*3600), 0.)

    def test_callback_requests_normal_stop_only_at_deadline(self):
        now = [10.]
        callback = StopAtDeadline(.01, clock=lambda: now[0])
        trainer = SimpleNamespace(stop=False)
        callback(trainer)
        self.assertFalse(trainer.stop)
        now[0] = 46.
        callback(trainer)
        self.assertTrue(trainer.stop and callback.triggered)

    def test_recipe_preserves_original_files_and_quality_guards(self):
        originals = [Path("kaggle_team34_pipeline.ipynb"), Path("kaggle_team34_v5_score_focus.ipynb")]
        before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in originals]
        notebook = build_notebook()
        verify(notebook)
        scope = {}
        exec("".join(notebook["cells"][1]["source"]), scope)
        self.assertEqual((scope["TRAIN_EPOCHS"], scope["REFINE_EPOCHS"], scope["TRAIN_BATCH"]), (65, 8, 16))
        self.assertFalse(any("coco-2017" in slug for slug in scope["AUTO_DOWNLOAD_KAGGLE_SLUGS"]))
        self.assertTrue(scope["QUARANTINE_TEACHER_DISAGREEMENT"])
        self.assertEqual(before, [hashlib.sha256(p.read_bytes()).hexdigest() for p in originals])

    def test_main_export_survives_skipped_refinement_and_is_not_exported_twice(self):
        import torch
        notebook = build_notebook()
        train_cell = "".join(notebook["cells"][11]["source"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = root / "outputs"
            outputs.mkdir()
            exports, trained = [], []
            class FakeModel:
                def __init__(self, *_):
                    self.trainer = None
                def add_callback(self, event, callback):
                    assert event == "on_train_batch_end"
                def train(self, **kwargs):
                    trained.append(kwargs)
                    checkpoint = root / kwargs["name"] / "weights/best.pt"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint.write_bytes(b"checkpoint")
                    self.trainer = SimpleNamespace(best=checkpoint)
            def exporter(weights, out, **kwargs):
                exports.append(out)
                out.mkdir(parents=True, exist_ok=True)
                for filename in ("model.onnx", "submission.zip"):
                    (out / filename).write_bytes(b"model artifact")
                return {"onnx": str(out / "model.onnx"), "zip": str(out / "submission.zip")}
            def evaluator(*args, **kwargs):
                return {"n_images": 320, "score_summary": {"evaluator":"official_detscore", "score":30,
                    "n_gt_per_class":{str(c):25 for c in range(11)}, "classes_without_gt":[]}}
            scope = dict(MASTER_DIR=root/"master", OUTPUTS_DIR=outputs, REPO_DIR=root, IS_KAGGLE=False,
                RUN_TRAIN_IN_THIS_NOTEBOOK=True, RUN_CLEAN_REFINEMENT=True, TRAIN_EPOCHS=65, REFINE_EPOCHS=8,
                TRAIN_BATCH=16, TRAIN_PATIENCE=15, DEVICE="cpu", TEAM_ID=39, RUN_NAME="balanced",
                RUN_STARTED=time.monotonic(), TOTAL_BUDGET_HOURS=7.5, EXPORT_RESERVE_HOURS=.75,
                MAIN_CAP_HOURS=3, REFINE_CAP_HOURS=.35, torch=torch, shutil=__import__("shutil"), json=json, Path=Path)
            with patch("ultralytics.YOLO", FakeModel), patch("src.export_verify_onnx.export_and_verify", exporter), \
                 patch("src.eval_golden.run_eval_or_qc", evaluator), \
                 patch("src.score_focus_v5.make_repeat_factor_dataset", return_value={}), \
                 patch("src.score_focus_v5.holdout_fingerprint", return_value="same_holdout"), \
                 patch("src.balanced_budget.phase_hours", side_effect=[1., 0.]):
                exec(compile(train_cell, "balanced_train_cell", "exec"), scope)
            self.assertEqual(len(trained), 1)
            self.assertEqual(trained[0]["mask_ratio"], 4)
            self.assertEqual(len(exports), 1)
            self.assertTrue((outputs / "recovery_main_submission.zip").is_file())
            self.assertTrue((outputs / "submission.zip").is_file())


if __name__ == "__main__":
    unittest.main()

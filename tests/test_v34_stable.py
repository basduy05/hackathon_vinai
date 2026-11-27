"""Failure injection and cell-order smoke tests. GPU/network/model exports are mocked."""
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import time
import shutil
import types
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import yaml

from src.stable_v34 import cached_stage, retry_download, train_resilient, unpack_readonly_input
from src.validate_dataset import load_competition_config
from tools.build_kaggle_v34_stable import TRAIN, build_notebook, verify
from tools.build_kaggle_v5_score_focus import unpack_bundle


class StableTests(unittest.TestCase):
    def test_retry_transient_failure_but_not_403(self):
        attempts = []
        def intermittent():
            attempts.append(1)
            if len(attempts) < 3:
                raise TimeoutError("connection timeout")
            return "done"
        self.assertEqual(retry_download(intermittent, sleep=lambda _:None), "done")
        self.assertEqual(len(attempts), 3)
        calls = []
        def denied():
            calls.append(1)
            exc = RuntimeError("Forbidden")
            exc.response = types.SimpleNamespace(status_code=403)
            raise exc
        with self.assertRaises(RuntimeError):
            retry_download(denied, sleep=lambda _:None)
        self.assertEqual(len(calls), 1)

    def test_readonly_archive_is_not_modified_and_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input"
            source.mkdir()
            archive = source / "dataset.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("train/labels/a.txt", "0 .5 .5 .2 .2")
            original = archive.read_bytes()
            out = unpack_readonly_input(source, root / "scratch", min_free_bytes=0)
            self.assertTrue(next(out.rglob("a.txt")).is_file())
            self.assertEqual(original, archive.read_bytes())
            self.assertEqual(list(source.iterdir()), [archive])
            with zipfile.ZipFile(source / "bad.zip", "w") as z:
                z.writestr("../escape.txt", "bad")
            with self.assertRaises(ValueError):
                unpack_readonly_input(source, root / "scratch2", min_free_bytes=0)
            self.assertFalse((root / "escape.txt").exists())

    def test_archive_disk_shortage_fails_before_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "input").mkdir()
            with zipfile.ZipFile(root / "input/a.zip", "w") as z:
                z.writestr("large", "x"*100)
            with patch("src.stable_v34.shutil.disk_usage", return_value=types.SimpleNamespace(free=10)):
                with self.assertRaises(OSError):
                    unpack_readonly_input(root / "input", root / "scratch", min_free_bytes=0)
            self.assertFalse(any((root / "scratch").rglob("large")))

    def test_cache_rejects_incomplete_or_mutated_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "staged"
            calls = []
            def builder(out):
                calls.append(1)
                (out / "images/train").mkdir(parents=True)
                (out / "labels/train").mkdir(parents=True)
                (out / "images/train/a.png").write_bytes(b"image")
                (out / "labels/train/a.txt").write_text("original")
                return {"kept_images":1}
            cached_stage(root, {"recipe":34}, builder)
            cached_stage(root, {"recipe":34}, builder)
            self.assertEqual(len(calls), 1)
            (root / "labels/train/a.txt").write_text("teacher modification")
            cached_stage(root, {"recipe":34}, builder)
            self.assertEqual(len(calls), 2)
            (root / ".stage_complete.json").unlink()
            cached_stage(root, {"recipe":34}, builder)
            self.assertEqual(len(calls), 3)

    def test_cuda_oom_halves_batch_and_other_errors_propagate(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "best.pt"
            batches = []
            class Model:
                def __init__(self, *_):
                    pass
                def train(self, **kwargs):
                    batches.append(kwargs["batch"])
                    if kwargs["batch"] > 4:
                        raise RuntimeError("CUDA out of memory")
                    checkpoint.write_bytes(b"weights")
                    self.trainer = types.SimpleNamespace(best=checkpoint)
            best, attempts = train_resilient(Model, "weights.pt", {"batch":16, "name":"run"})
            self.assertEqual(batches, [16,8,4])
            self.assertEqual(best, checkpoint)
            self.assertEqual(attempts[-1]["status"], "complete")
            class Bad(Model):
                def train(self, **kwargs):
                    raise RuntimeError("labels corrupted")
            with self.assertRaisesRegex(RuntimeError, "labels corrupted"):
                train_resilient(Bad, "weights.pt", {"batch":16, "name":"run"})

    def test_minimum_batch_oom_terminates(self):
        class Model:
            def __init__(self, *_):
                pass
            def train(self, **kwargs):
                raise RuntimeError("CUDA out of memory")
        with self.assertRaisesRegex(RuntimeError, "minimum batch"):
            train_resilient(Model, "weights.pt", {"batch":2, "name":"run"})

    def test_embedded_modules_import_in_fresh_python_process(self):
        bundle = unpack_bundle(build_notebook())[2]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            modules = []
            for name, text in bundle.items():
                file = root / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(text, encoding="utf-8")
                if name.startswith("src/") and name.endswith(".py") and not name.endswith("__init__.py"):
                    modules.append(name[:-3].replace("/", "."))
            command = "import importlib; " + "; ".join(f"importlib.import_module({name!r})" for name in modules)
            process = subprocess.run([sys.executable, "-X", "utf8", "-c", command], cwd=root,
                capture_output=True, text=True, timeout=60, env=dict(os.environ, PYTHONPATH=str(root)))
            self.assertEqual(process.returncode, 0, process.stderr)

    def test_failed_teacher_excludes_partial_source_without_removing_human_source(self):
        cell = "".join(build_notebook()["cells"][8]["source"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full, partial = root / "mapillary_full", root / "lvnduy_partial"
            output = root / "outputs"
            output.mkdir()
            scope = dict(STAGED_SOURCES=[full,partial], OUTPUTS_DIR=output, RUN_PARTIAL_LABEL_COMPLETION=True,
                RUN_STAGE1_MAPILLARY_TEACHER=True, TEACHER_WEIGHTS="teacher.pt", DEVICE="cpu", Path=Path,
                json=json, shutil=shutil)
            with patch("src.prepare_vn_frames.UltralyticsTeacherPredictor", side_effect=RuntimeError("mock teacher download failed")):
                exec(compile(cell,"v34_teacher_fallback","exec"),scope)
            self.assertEqual(scope["STAGED_SOURCES"], [full])
            self.assertTrue((output / "teacher_completion_failure.json").is_file())

    def test_refinement_failure_preserves_main_submission_and_official_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs"
            output.mkdir()
            best = root / "runs/main/weights/best.pt"
            best.parent.mkdir(parents=True)
            best.write_bytes(b"fake weights")
            def train(factory, weights, kwargs, callback):
                if "refine" in kwargs["name"]:
                    raise RuntimeError("CUDA memory still insufficient at minimum batch")
                return best, [{"batch":16,"status":"complete"}]
            def export(weights, folder, **kwargs):
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "model.onnx").write_bytes(b"fake validated")
                (folder / "submission.zip").write_bytes(b"fake validated zip")
                return {"onnx":str(folder / "model.onnx")}
            def evaluate(*args, **kwargs):
                return {"n_images":320, "score_summary":{"evaluator":"official_detscore", "score":20,
                    "n_gt_per_class":{str(c):25 for c in range(11)}, "classes_without_gt":[]}}
            scope = dict(Path=Path, time=time, shutil=shutil, RUN_ROOT=root, MASTER_DIR=root / "master",
                OUTPUTS_DIR=output, REPO_DIR=root, RUN_TRAIN_IN_THIS_NOTEBOOK=True, RUN_CLEAN_REFINEMENT=True,
                RUN_STARTED=time.monotonic(), TOTAL_BUDGET_HOURS=7.5, EXPORT_RESERVE_HOURS=.75,
                MAIN_CAP_HOURS=3., REFINE_CAP_HOURS=.35, TEAM_ID=39, DEVICE="cpu", TRAIN_BATCH=16,
                TRAIN_EPOCHS=65, TRAIN_PATIENCE=15, REFINE_EPOCHS=8, RUN_NAME="stable")
            with patch("src.stable_v34.train_resilient", train), \
                 patch("src.score_focus_v5.make_repeat_factor_dataset", return_value={}), \
                 patch("src.score_focus_v5.holdout_fingerprint", return_value="fixed"), \
                 patch("src.export_verify_onnx.export_and_verify", export), \
                 patch("src.eval_golden.run_eval_or_qc", evaluate):
                exec(compile(TRAIN,"stable_train_fallback","exec"),scope)
            self.assertTrue((output / "submission.zip").is_file())
            self.assertTrue((output / "recovery_main_submission.zip").is_file())
            self.assertTrue((output / "optional_refine_failure.json").is_file())
            self.assertEqual(json.loads((output / "checkpoint_selection.json").read_text())["selected"], "main")

    def test_notebook_fresh_data_to_final_submission_cell_order(self):
        """Execute all cells on tiny data; mock external network, GPU, train/export/scoring only."""
        project = Path.cwd()
        classes = load_competition_config("configs/competition.yaml")["classes"]
        notebook = build_notebook()
        verify(notebook)
        pinned = {"ultralytics":"8.4.214", "onnx":"1.23.2", "onnxslim":"0.1.99", "onnxruntime":"1.32.0"}
        real_version = importlib.metadata.version
        def version(package):
            return pinned.get(package) or real_version(package)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "native_dataset"
            fixture.mkdir()
            (fixture / "data.yaml").write_text(yaml.safe_dump({"names":classes}))
            for split, count in (("train", 60),("val",30)):
                for index in range(count):
                    image = fixture / split / "images" / f"{split}_{index}.png"
                    label = fixture / split / "labels" / f"{split}_{index}.txt"
                    image.parent.mkdir(parents=True, exist_ok=True)
                    label.parent.mkdir(parents=True, exist_ok=True)
                    seed = index + (100 if split == "val" else 0)
                    Image.fromarray(np.random.default_rng(seed).integers(0,255,(160,160,3),dtype=np.uint8)).save(image)
                    lines = []
                    for cid in range(11):
                        x, y = .04+(cid%4)*.23, .04+(cid//4)*.30
                        lines.append(f"{cid} {x} {y} {x+.15} {y} {x+.15} {y+.15} {x} {y+.15}\n")
                    label.write_text("".join(lines))
            kaggle = types.ModuleType("kagglehub")
            pycoco = types.ModuleType("pycocotools")
            pycoco.__path__ = []
            pycoco_mask = types.ModuleType("pycocotools.mask")
            ipython = types.ModuleType("IPython")
            ipython.__path__ = []
            display_module = types.ModuleType("IPython.display")
            display_module.display = lambda *args, **kwargs: None
            display_module.Image = lambda *args, **kwargs: None
            display_module.FileLink = lambda *args, **kwargs: None
            def download(slug):
                if "mapillary" not in slug:
                    raise ValueError("mock optional source unavailable")
                return str(fixture)
            kaggle.dataset_download = download
            class Model:
                def __init__(self, *_):
                    pass
                def add_callback(self, *args):
                    pass
                def train(self, **kwargs):
                    checkpoint = Path(kwargs["project"]) / kwargs["name"] / "weights/best.pt"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint.write_bytes(b"fake checkpoint")
                    self.trainer = types.SimpleNamespace(best=checkpoint)
            def exporter(weights, out, **kwargs):
                out.mkdir(parents=True, exist_ok=True)
                (out / "model.onnx").write_bytes(b"fake ONNX, validator mocked")
                with zipfile.ZipFile(out / "submission.zip", "w") as z:
                    z.write(out / "model.onnx", "model.onnx")
                return {"onnx":str(out / "model.onnx"), "zip":str(out / "submission.zip")}
            def evaluate(model, images, labels, *args, **kwargs):
                from src.seg_utils import read_yolo_seg_file
                from collections import Counter
                counts = Counter(it.cls for f in labels.glob("*.txt") for it in read_yolo_seg_file(f))
                return {"n_images":len(list(images.glob("*.png"))), "score_summary":{
                    "evaluator":"official_detscore", "score":20 if model.parent.name == "main" else 19,
                    "n_gt_per_class":{classes[c]:counts[c] for c in range(11)}, "classes_without_gt":[]}}
            scope = {}
            previous_tempdir = tempfile.tempdir
            previous_tmpdir = os.environ.get("TMPDIR")
            try:
                os.chdir(root)
                with patch.dict("sys.modules", {"kagglehub":kaggle, "pycocotools":pycoco, "pycocotools.mask":pycoco_mask,
                                               "IPython":ipython, "IPython.display":display_module}), \
                     patch("importlib.metadata.version", side_effect=version), \
                     patch("ultralytics.YOLO", Model), \
                     patch("src.export_verify_onnx.export_and_verify", exporter), \
                     patch("src.eval_golden.run_eval_or_qc", evaluate):
                    for index, cell in enumerate(notebook["cells"]):
                        if cell["cell_type"] != "code":
                            continue
                        if index == 2:
                            scope.update(INSTALL_DEPS=False, REQUIRE_GPU=False, RUN_KAGGLE_INPUT_PUBLIC=False,
                                         AUTO_GOLDEN_VAL_TARGET=40, MIN_RARE_TRAIN_INSTANCES=10,
                                         RUN_SANITY_GATE=False, RUN_BENCHMARK_AND_QC=False,
                                         RUN_CROP_ZOOM=False, RUN_COPY_PASTE=False, RUN_WEATHER_AUG=False)
                        exec(compile("".join(cell["source"]), f"v34_smoke_cell_{index}", "exec"), scope)
                out = scope["OUTPUTS_DIR"]
                self.assertTrue((out / "submission.zip").is_file())
                self.assertTrue((out / "recovery_main_submission.zip").is_file())
                self.assertEqual(json.loads((out / "checkpoint_selection.json").read_text())["selected"], "main")
                self.assertEqual(len(json.loads((out / "source_failures.json").read_text())), 2)
                self.assertFalse((out / "optional_report_failure.json").exists())
            finally:
                os.chdir(project)
                tempfile.tempdir = previous_tempdir
                if previous_tmpdir is None:
                    os.environ.pop("TMPDIR", None)
                else:
                    os.environ["TMPDIR"] = previous_tmpdir


if __name__ == "__main__":
    unittest.main()

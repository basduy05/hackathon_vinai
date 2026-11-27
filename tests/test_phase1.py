"""Phase 1 verification tests for selfcheck_bridge, validate_dataset, and export_verify_onnx."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.export_verify_onnx import build_export_kwargs, write_reproducible_zip
from src.selfcheck_bridge import materialize_selfcheck
from src.validate_dataset import load_competition_config, run_qc


class Phase1Tests(unittest.TestCase):
    def test_selfcheck_bridge_loads(self) -> None:
        sc = materialize_selfcheck("train_and_export.ipynb", "work/_selfcheck")
        self.assertEqual(sc.config.nc, 11)
        self.assertEqual(sc.config.imgsz, 640)
        self.assertEqual(sc.config.min_gt_area_px, 64.0)

    def test_sample_data_qc_passes(self) -> None:
        cfg = load_competition_config("configs/competition.yaml")
        report = run_qc(Path("sample_data"), cfg)
        self.assertTrue(report["ok"])
        self.assertEqual(report["fatal_count"], 0)
        self.assertEqual(report["poly_issue_count"], 0)
        self.assertEqual(report["per_class"]["Motorcyclist"]["tiny_lt_64px"], 1)
        self.assertEqual(report["ambiguity_queue_size"], 0)

    def test_edge_case_detectors_and_fatal_errors(self) -> None:
        cfg = load_competition_config("configs/competition.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images" / "train").mkdir(parents=True)
            (root / "labels" / "train").mkdir(parents=True)
            img = root / "images" / "train" / "test_01.jpg"
            Image.new("RGB", (640, 640), (120, 120, 120)).save(img)

            lbl = root / "labels" / "train" / "test_01.txt"
            lbl.write_text(
                "\n".join([
                    # RIDER_LEAK: Pedestrian(8) overlapping Motorcycle(4)
                    "4 0.30 0.50 0.45 0.50 0.45 0.70 0.30 0.70",
                    "8 0.33 0.35 0.42 0.35 0.42 0.65 0.33 0.65",
                    # SIGN_POLE_SUSPECT: tall narrow interior sign
                    "9 0.60 0.20 0.64 0.20 0.64 0.40 0.60 0.40",
                    # TRAILER_MERGED: wide elongated truck
                    "2 0.10 0.75 0.70 0.75 0.70 0.95 0.10 0.95",
                ]),
                encoding="utf-8",
            )
            report = run_qc(root, cfg)
            self.assertTrue(report["ok"])
            flags = report["edge_case_flag_counts"]
            self.assertEqual(flags.get("RIDER_LEAK"), 1)
            self.assertEqual(flags.get("SIGN_POLE_SUSPECT"), 1)
            self.assertEqual(flags.get("TRAILER_MERGED"), 1)

            # Fatal error check: 5-column bbox line
            lbl.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            bad_report = run_qc(root, cfg)
            self.assertFalse(bad_report["ok"])
            self.assertEqual(bad_report["fatal_count"], 1)

    def test_export_contract_and_zip_guard(self) -> None:
        sc = materialize_selfcheck("train_and_export.ipynb", "work/_selfcheck")
        kwargs = build_export_kwargs(sc.constants["EXPORT_CONTRACT"])
        self.assertEqual(kwargs, {"format": "onnx", "imgsz": 640, "opset": 17, "simplify": True, "dynamic": False, "batch": 1})
        with self.assertRaises(ValueError):
            build_export_kwargs({**sc.constants["EXPORT_CONTRACT"], "nms": False})
        with self.assertRaises(ValueError):
            build_export_kwargs({**sc.constants["EXPORT_CONTRACT"], "half": True})

        with tempfile.TemporaryDirectory() as tmp:
            dummy_onnx = Path(tmp) / "model.onnx"
            dummy_onnx.write_bytes(b"dummy-onnx-content")
            zip_path = write_reproducible_zip(dummy_onnx, Path(tmp) / "submission.zip")
            summary = sc.package.inspect_zip(str(zip_path), sc.config)
            self.assertEqual(summary.entry_name, "model.onnx")


if __name__ == "__main__":
    unittest.main()

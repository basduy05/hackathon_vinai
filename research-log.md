# Research Log — AIInAction RoboTaxi Data Centric

## 2026-11-27 — Phase 1 Initialization
- **Contract audit ([train_and_export.ipynb](train_and_export.ipynb))**:
  - Pinned toolchain: `ultralytics==8.4.214`, `onnx==1.23.2`, `onnxslim==0.1.99`, `onnxruntime==1.32.0`, `pycocotools`.
  - Required ONNX export: Layout A (`nms` unset, `half=False`, `opset=17`, `simplify=True`, `dynamic=False`, `batch=1`). Outputs `[1, 47, 8400]` and `[1, 32, 160, 160]`.
  - Scoring rule: `100 × mask mAP@[.5:.95]`, `MIN_GT_AREA_PX = 64.0` at 640 scale.
- **Sample data audit (`sample_data/`)**:
  - 10 synthetic illustrations (`960×540`), 105 instances across all 11 classes. Not for submission training.
- **Action**: Building Phase 1 foundation (`configs/`, `src/selfcheck_bridge.py`, `src/validate_dataset.py`, `src/export_verify_onnx.py`).

## 2026-11-27 10:36 — Phase 2 Pipeline Kickoff (Đội 34 – Nguyễn Bá Duy, `TEAM_ID = 39`)
- Synchronized `team_id: 39` into `configs/competition.yaml` and `research-state.yaml`.
- Upgraded `annotation_guideline_and_qc.md` to v2 (synced with `100 × mask mAP@[.5:.95]`, `MIN_GT_AREA_PX = 64 px² @ 640`, `max_det = 100`, VN edge-case decision tree, and 3-tier QC).
- Implementing Phase 2 modular data engineering & evaluation pipeline (`seg_utils.py`, `map_and_merge_public.py`, `box_to_mask_sam2.py`, `prepare_vn_frames.py`, `augment_and_crop.py`, `eval_golden.py`).
- **Phase 2 Pipeline Verification**: Implemented `seg_utils.py`, `map_and_merge_public.py`, `box_to_mask_sam2.py`, `prepare_vn_frames.py`, `augment_and_crop.py`, `eval_golden.py`, `build_master.py`, and `tests/test_phase2.py`. All 8 unit tests (`test_phase1` + `test_phase2`) and `sample_data` Tier-1 QC passed.
- **Environment check note**: Local Python 3.14.4 has `ultralytics==8.4.215`, `onnxruntime==1.29.0`, and lacks `pycocotools`; use `train_and_export.ipynb` pinned environment on Colab/Kaggle for final ONNX export (`8.4.214`) and official `detscore` evaluation.

## 2026-11-27 11:45 — V2 Pipeline Upgrades (Post du_lieu_v1.zip QC Audit)
- Diagnosed `du_lieu_v1.zip` QC report: `497 RIDER_LEAK` (727 Motorcycle vs 80 Motorcyclist), `10` self-intersecting polygons, `0` Other vehicle, `46` Sign.
- Upgraded `seg_utils.py` (`SegInstance` auto-repair + `nearest_points` bridge), `map_and_merge_public.py` + `class_mapping.yaml` (added `bbox_iou + above_mounted` merger rule + `threewheel/van/motor/bike` aliases).
- Added `src/kaggle_adapter.py` (6 Kaggle formats + HuggingFace Parquet loader) and `src/benchmark_report.py` (automated BTC-style Markdown + Matplotlib comparison reports), and rebuilt `kaggle_team34_pipeline.ipynb` V2.

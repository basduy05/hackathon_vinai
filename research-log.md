# Research Log — AIInAction RoboTaxi Data Centric

## 2026-11-27 — Phase 1 Initialization
- **Contract audit ([train_and_export.ipynb](train_and_export.ipynb))**:
  - Pinned toolchain: `ultralytics==8.4.214`, `onnx==1.23.2`, `onnxslim==0.1.99`, `onnxruntime==1.32.0`, `pycocotools`.
  - Required ONNX export: Layout A (`nms` unset, `half=False`, `opset=17`, `simplify=True`, `dynamic=False`, `batch=1`). Outputs `[1, 47, 8400]` and `[1, 32, 160, 160]`.
  - Scoring rule: `100 × mask mAP@[.5:.95]`, `MIN_GT_AREA_PX = 64.0` at 640 scale.
- **Sample data audit (`sample_data/`)**:
  - 10 synthetic illustrations (`960×540`), 105 instances across all 11 classes. Not for submission training.
- **Action**: Building Phase 1 foundation (`configs/`, `src/selfcheck_bridge.py`, `src/validate_dataset.py`, `src/export_verify_onnx.py`).

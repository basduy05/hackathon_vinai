# Findings — Data-Centric Insights & Edge Cases

## Key Technical Findings
1. **ONNX Export Layout A Constraint**: `yolo26n-seg` must be exported without `nms=False` (end-to-end head is rejected) and without `half=True` (FP16 is rejected). `detscore_selfcheck` verifies exact Conv-layer signature SHA-256.
2. **Small Object Cutoff (`64 px²` at 640)**: Ground truth and unmatched predictions smaller than `64 px²` at 640-scale (`8×8 px`) are ignored by the grader. Labeling ultra-tiny faraway objects wastes capacity; Crop-Zoom in training helps small signs/lights stay above this threshold while training clean features.
3. **Sample Data Observation**: `sample_data/labels/sample_07.txt` contains 1 tiny Motorcyclist (`45.3 px²` at 640-scale) — useful test case to confirm `validate_dataset.py` catches `< 64 px²` warnings.

## Experiment Observations
*(Will be updated after Baseline v0 and each Data-Centric iteration)*

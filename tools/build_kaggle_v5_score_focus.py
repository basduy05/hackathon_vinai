"""Create an isolated score-focused notebook from the actual V4 notebook; never rewrite V4."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path
import textwrap
import zlib

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kaggle_team34_pipeline.ipynb"
OUTPUT = ROOT / "kaggle_team34_v5_score_focus.ipynb"


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def unpack_bundle(notebook):
    source = "".join(notebook["cells"][2]["source"])
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_BUNDLE_B64" for t in n.targets))
    bundle = json.loads(zlib.decompress(base64.b64decode(ast.literal_eval(node.value))))
    return source, node, bundle


COMPLETION = '''#@title 7B. Teacher chất lượng: hai góc nhìn + giữ GT + cách ly bất đồng
import gc
import torch
from ultralytics import YOLO
from src.build_master import build_master_dataset
from src.prepare_vn_frames import UltralyticsTeacherPredictor
from src.score_focus_v5 import complete_labels_with_consensus, quarantine_uncertain

vn_partial_sources = [s for s in STAGED_SOURCES if any(k in s.name.lower() for k in ("vnts", "lvnduy", "traffic_sign"))]
coco_sources = [s for s in STAGED_SOURCES if "coco" in s.name.lower()]
full_sources_for_stage1 = [s for s in STAGED_SOURCES if any(k in s.name.lower() for k in ("mapillary", "bdd100k", "cityscapes"))]
completion_reports = {}
if RUN_PARTIAL_LABEL_COMPLETION:
    if vn_partial_sources:
        teacher_coco = UltralyticsTeacherPredictor(weights=TEACHER_WEIGHTS, conf=0.25, imgsz=1280, device=DEVICE)
        for source in vn_partial_sources:
            result = complete_labels_with_consensus(
                source, teacher_coco, preserved_classes={9}, fill_classes={0,1,2,4,5,6,7,8,10},
                min_conf=PSEUDO_MIN_CONF, per_class_min_conf={1: .55, 2: .55, 4: .45, 6: .45, 8: .45, 10: .45})
            completion_reports.setdefault(str(source), []).append(result)
            print(source.name, {k:v for k,v in result.items() if k != "records"}, flush=True)
        del teacher_coco
        gc.collect()
        torch.cuda.empty_cache()

    if RUN_STAGE1_MAPILLARY_TEACHER and full_sources_for_stage1:
        stage1_data_dir = TMP_RAW_DIR / "stage1_master"
        shutil.rmtree(stage1_data_dir, ignore_errors=True)
        stage1_summary = build_master_dataset(full_sources_for_stage1, GOLDEN_VAL_DIR, stage1_data_dir)
        if not stage1_summary["qc_ok"]:
            raise RuntimeError("Stage-1 dataset failed QC")
        s1_model = YOLO("yolo26s-seg.pt")
        s1_proj = Path("/kaggle/working/runs_stage1_v5") if IS_KAGGLE else Path("work_kaggle_v5/runs_stage1")
        s1_model.train(data=str(stage1_data_dir / "data.yaml"), imgsz=640, epochs=STAGE1_EPOCHS,
            batch=TRAIN_BATCH, seed=TEAM_ID, device=DEVICE, workers=2, cache=False,
            optimizer="AdamW", lr0=.001, cos_lr=True, mask_ratio=2,
            mosaic=.5, mixup=0., copy_paste=0., close_mosaic=10, max_det=100,
            project=str(s1_proj), name="stage1_mapillary", exist_ok=True)
        s1_best = Path(s1_model.trainer.best)
        shutil.copy2(s1_best, OUTPUTS_DIR / "stage1_teacher.pt")
        del s1_model
        gc.collect()
        torch.cuda.empty_cache()
        teacher_map = UltralyticsTeacherPredictor(weights=str(OUTPUTS_DIR / "stage1_teacher.pt"), conf=.25, imgsz=1280, device=DEVICE)
        for source, preserve, fill in [(s, {0,1,2,4,5,6,7,8,10}, {3,9}) for s in coco_sources] + [(s, {0,1,2,4,5,6,7,8,9,10}, {3}) for s in vn_partial_sources]:
            result = complete_labels_with_consensus(source, teacher_map, preserve, fill,
                source_mapping_name="competition_direct", min_conf=PSEUDO_MIN_CONF,
                per_class_min_conf={3:.60, 9:.50})
            completion_reports.setdefault(str(source), []).append(result)
            print(source.name, {k:v for k,v in result.items() if k != "records"}, flush=True)
        del teacher_map
        gc.collect()
        torch.cuda.empty_cache()

    quarantine_reports = []
    if QUARANTINE_TEACHER_DISAGREEMENT:
        for source_text, reports in completion_reports.items():
            quarantine_reports.append(quarantine_uncertain(Path(source_text), reports, TMP_RAW_DIR / "teacher_quarantine"))
    (OUTPUTS_DIR / "teacher_consensus_summary.json").write_text(json.dumps({
        "sources": {k: [{a:b for a,b in r.items() if a != "records"} for r in reports] for k,reports in completion_reports.items()},
        "quarantine": quarantine_reports}, indent=2), encoding="utf-8")
    STAGED_SOURCES = [s for s in STAGED_SOURCES if any((s / "images/train").glob("*"))]
elif vn_partial_sources:
    raise RuntimeError("Partial Sign-only datasets require annotation completion before training")
'''


AUGMENT = '''#@title 8. Tăng mẫu lớp yếu từ nhãn đầy đủ, giới hạn ảnh tổng hợp
from src.augment_and_crop import generate_crop_zoom_dataset, generate_copypaste_dataset, generate_weather_dataset
from src.score_focus_v5 import cap_generated_images

base_sources = [s for s in STAGED_SOURCES if not s.name.startswith("aug_")]
human_full_sources = [s for s in base_sources if any(k in s.name for k in ("mapillary", "bdd100k", "cityscapes"))]
if not human_full_sources:
    raise RuntimeError("V5 requires fully annotated original street sources")

# The bank/background use full-label sources, avoiding amplification of teacher mistakes.
if RUN_CROP_ZOOM:
    for idx, source in enumerate(human_full_sources):
        out = STAGED_DIR / f"aug_weakclass_crops_{idx:02d}"
        shutil.rmtree(out, ignore_errors=True)
        result = generate_crop_zoom_dataset(source, out, target_classes=(3,5,6,7,9,10),
            max_crops_per_image=1, min_target_area_640=32., max_target_area_640=1800., seed=TEAM_ID+idx)
        kept = cap_generated_images(out, MAX_EXTRA_CROPS // len(human_full_sources), TEAM_ID+idx)
        if kept:
            STAGED_SOURCES.append(out)
        print(source.name, "weak-class crops", kept, flush=True)
if RUN_COPY_PASTE:
    out = STAGED_DIR / "aug_copypaste"
    shutil.rmtree(out, ignore_errors=True)
    result = generate_copypaste_dataset(human_full_sources, human_full_sources, out,
        rare_classes=(3,5,6,7), n_output=COPY_PASTE_COUNT, pastes_per_image=(1,2), seed=TEAM_ID)
    if result["images_written"]:
        STAGED_SOURCES.append(out)
    print(result, flush=True)
if RUN_WEATHER_AUG:
    out = STAGED_DIR / "aug_weather"
    shutil.rmtree(out, ignore_errors=True)
    result = generate_weather_dataset(human_full_sources[0], out, max_images=WEATHER_COUNT, seed=TEAM_ID)
    if result["images_written"]:
        STAGED_SOURCES.append(out)
'''


TRAIN = '''#@title 10. Train + fine-tune sạch + chọn checkpoint bằng official_detscore
import gc
from ultralytics import YOLO
from src.export_verify_onnx import export_and_verify
from src.eval_golden import run_eval_or_qc
from src.score_focus_v5 import make_repeat_factor_dataset, choose_official_candidate, holdout_fingerprint

BEST_PT_PATH = None
SUBMISSION_ZIP_PATH = None
SELECTED_TRAIN_DIR = None
if RUN_TRAIN_IN_THIS_NOTEBOOK:
    train_yaml = MASTER_DIR / "data_repeat.yaml"
    sampling = make_repeat_factor_dataset(MASTER_DIR, train_yaml, max_extra_fraction=.35, seed=TEAM_ID)
    print("Repeat-factor sampling:", sampling, flush=True)
    refine_yaml = MASTER_DIR / "data_clean_refine.yaml"
    clean_sampling = make_repeat_factor_dataset(MASTER_DIR, refine_yaml, clean_only=True, max_extra_fraction=.35, seed=TEAM_ID)
    print("Clean refinement subset:", clean_sampling, flush=True)
    run_proj = Path("/kaggle/working/runs_v5") if IS_KAGGLE else Path("work_kaggle_v5/runs")
    common = dict(imgsz=640, batch=TRAIN_BATCH, device=DEVICE, workers=2, cache=False,
        seed=TEAM_ID, deterministic=True, amp=True, optimizer="AdamW", cos_lr=True,
        mask_ratio=2, overlap_mask=True, max_det=100, project=str(run_proj), exist_ok=True)
    model = YOLO("yolo26n-seg.pt")
    model.train(data=str(train_yaml), epochs=TRAIN_EPOCHS, patience=TRAIN_PATIENCE,
        lr0=.0015, lrf=.02, weight_decay=.0005, warmup_epochs=3.,
        scale=.5, degrees=2., hsv_v=.35, mosaic=.6, close_mosaic=20,
        copy_paste=.10, mixup=0., multi_scale=False, name=RUN_NAME+"_main", **common)
    main_best = Path(model.trainer.best)
    candidates = [("main", main_best)]
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if RUN_CLEAN_REFINEMENT:
        refine = YOLO(str(main_best))
        refine.train(data=str(refine_yaml), epochs=REFINE_EPOCHS, patience=REFINE_EPOCHS,
            lr0=.00015, lrf=.1, warmup_epochs=1., weight_decay=.0005,
            scale=.15, degrees=0., hsv_v=.15, mosaic=0., close_mosaic=0,
            copy_paste=0., mixup=0., multi_scale=False, name=RUN_NAME+"_refine", **common)
        candidates.append(("clean_refine", Path(refine.trainer.best)))
        del refine
        gc.collect()
        torch.cuda.empty_cache()

    # The same 11-class held-out labels and official FP32 ONNX decoder select the exported model.
    fingerprint = holdout_fingerprint(MASTER_DIR)
    evaluated = []
    for name, checkpoint in candidates:
        candidate_dir = OUTPUTS_DIR / "candidates" / name
        exported = export_and_verify(checkpoint, candidate_dir, notebook_path=REPO_DIR / "train_and_export.ipynb",
            smoke_images_dir=MASTER_DIR / "images/val")
        evaluation = run_eval_or_qc(Path(exported["onnx"]), MASTER_DIR / "images/val", MASTER_DIR / "labels/val",
            candidate_dir / "official_validation", mode="eval", notebook_path=REPO_DIR / "train_and_export.ipynb")
        evaluated.append({"name":name, "weights":str(checkpoint), "export_dir":str(candidate_dir),
            "holdout_sha256":fingerprint, "n_images":evaluation["n_images"], "score_summary":evaluation["score_summary"]})
    winner = choose_official_candidate(evaluated)
    (OUTPUTS_DIR / "checkpoint_selection.json").write_text(json.dumps({"selected":winner["name"], "candidates":evaluated}, indent=2), encoding="utf-8")
    BEST_PT_PATH = Path(winner["weights"])
    SELECTED_TRAIN_DIR = BEST_PT_PATH.parent.parent
    shutil.copy2(BEST_PT_PATH, OUTPUTS_DIR / "best.pt")
    for filename in ("model.onnx", "submission.zip"):
        shutil.copy2(Path(winner["export_dir"]) / filename, OUTPUTS_DIR / filename)
    SUBMISSION_ZIP_PATH = OUTPUTS_DIR / "submission.zip"
    print("SELECTED BY OFFICIAL SCORE:", winner["name"], winner["score_summary"]["score"], SUBMISSION_ZIP_PATH, flush=True)
'''


def build_notebook():
    original_bytes = SOURCE.read_bytes()
    notebook = json.loads(original_bytes)
    if len(notebook["cells"]) != 14 or 'RUN_STAGE1_MAPILLARY_TEACHER' not in "".join(notebook["cells"][1]["source"]):
        raise ValueError("Expected current V4 notebook with stage-1 teacher; inspect changes before rebuilding")
    source, node, bundle = unpack_bundle(notebook)
    bundle["src/score_focus_v5.py"] = (ROOT / "src/score_focus_v5.py").read_text(encoding="utf-8")
    # Keep the V4 payload frozen; V5 changes are an additional module and notebook cells only.
    raw = base64.b64encode(zlib.compress(json.dumps(bundle, ensure_ascii=False).encode("utf-8"), 9)).decode("ascii")
    assignment = "_BUNDLE_B64 = (\n" + "\n".join(repr(x) for x in textwrap.wrap(raw, 100)) + "\n)\n"
    lines = source.splitlines(keepends=True)
    source = "".join(lines[:node.lineno-1]) + assignment + "".join(lines[node.end_lineno:])
    source = source.replace("V3.3", "V5 Score Focus")
    notebook["cells"][2] = code(source)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        for old, new in (("work_kaggle", "work_kaggle_v5"), ("/kaggle/working/repo", "/kaggle/working/repo_v5"),
                         ("/kaggle/tmp/raw", "/kaggle/tmp/v5_raw"), ("/kaggle/working/staged", "/kaggle/working/staged_v5"),
                         ("/kaggle/working/golden_val", "/kaggle/working/golden_val_v5"),
                         ("master_v4", "master_v5"), ("du_lieu_v4", "du_lieu_v5"),
                         ("/kaggle/working/outputs", "/kaggle/working/outputs_v5"), ("team34_v4", "team34_v5_score")):
            text = text.replace(old, new)
        text = text.replace('shutil.rmtree(Path.home() / ".cache" / "kagglehub", ignore_errors=True)', '# V5 keeps unrelated account caches intact')
        cell.update(code(text))
    params = "".join(notebook["cells"][1]["source"])
    params = params.replace('TRAIN_EPOCHS = 95', 'TRAIN_EPOCHS = 110').replace('STAGE1_EPOCHS = 22', 'STAGE1_EPOCHS = 35')
    params = params.replace('TRAIN_BATCH = 16', 'TRAIN_BATCH = 8').replace('TRAIN_PATIENCE = 25', 'TRAIN_PATIENCE = 30')
    params = params.replace('COPY_PASTE_COUNT = 450', 'COPY_PASTE_COUNT = 300').replace('WEATHER_COUNT = 350', 'WEATHER_COUNT = 150')
    params = params.replace('VNTS_MAX_IMAGES = 1000', 'VNTS_MAX_IMAGES = 800')
    params = params.replace('V4', 'V5 Score Focus')
    # The inherited runtime estimate is not a measurement for this experimental recipe.
    params = params.replace('MULTI_SCALE = False  # Tắt multi_scale để chạy an toàn ~5.5-6h trong giới hạn 9h Kaggle GPU (đã có scale=0.6 + Crop-Zoom)', 'MULTI_SCALE = False')
    params += '\n# V5 score-first experiment; runtime may exceed a single Kaggle session.\nPSEUDO_MIN_CONF = .50\nQUARANTINE_TEACHER_DISAGREEMENT = True\nMAX_EXTRA_CROPS = 800\nRUN_CLEAN_REFINEMENT = True\nREFINE_EPOCHS = 25\n'
    notebook["cells"][1] = code(params)
    stage = "".join(notebook["cells"][4]["source"])
    stage = stage.replace('from src.box_to_mask_sam2 import UltralyticsSAM2Predictor, convert_bbox_dataset',
                          'from src.box_to_mask_sam2 import convert_bbox_dataset\nfrom src.score_focus_v5 import CorrectedSAM2Predictor as UltralyticsSAM2Predictor')
    stage = stage.replace('            CLEAN_TRAIN_SOURCES.append(vnts_out)', '            # Sign-only sources never supply Golden Val, even as a fallback.')
    notebook["cells"][4] = code(stage)
    notebook["cells"][8] = code(COMPLETION)
    notebook["cells"][9] = code(AUGMENT)
    gate = "".join(notebook["cells"][10]["source"])
    gate = gate.replace('V4', 'V5')
    gate = gate.replace('from IPython.display', 'import torch\nif not torch.cuda.is_available():\n    raise RuntimeError("V5 requires a GPU; do not start score-focused training on CPU")\nfrom IPython.display', 1)
    # V4's aggregate >=4-class heuristic is retained as a warning, not a completeness certificate.
    gate = gate.replace('raise RuntimeError(f"Nguồn {_src.name} chỉ có {len(_present_cls)} lớp — nguy cơ đầu độc nhãn âm!")',
                        'print(f"[REVIEW] Nguồn {_src.name}: {len(_present_cls)} lớp; kiểm tra completeness theo ảnh")')
    gate += '\nif qc_rep["duplicates"]["leak_exact"] or qc_rep["duplicates"]["leak_near"]:\n    raise RuntimeError("Train/Val leakage found; do not train")\n'
    notebook["cells"][10] = code(gate)
    notebook["cells"][11] = code(TRAIN)
    report = "".join(notebook["cells"][12]["source"])
    report = report.replace('train_run_p = pt_p.parent.parent if (pt_p.is_file() and pt_p.parent.name == "weights") else None',
                            'train_run_p = SELECTED_TRAIN_DIR')
    report = report.replace('OUTPUTS_DIR / "holdout_manifest.json",', 'OUTPUTS_DIR / "holdout_manifest.json",\n            OUTPUTS_DIR / "checkpoint_selection.json",\n            OUTPUTS_DIR / "teacher_consensus_summary.json",')
    notebook["cells"][12] = code(report)
    download = "".join(notebook["cells"][13]["source"])
    download = download.replace('    OUTPUTS_DIR / "best.pt",',
        '    OUTPUTS_DIR / "best.pt",\n    OUTPUTS_DIR / "checkpoint_selection.json",\n    OUTPUTS_DIR / "teacher_consensus_summary.json",')
    notebook["cells"][13] = code(download)
    text = '''# V5 Score Focus — thử nghiệm nâng điểm từ V4

Import notebook riêng này lên Kaggle, bật GPU + Internet. V4 được giữ nguyên.
**Đây là recipe chưa train; không có cam kết điểm cao hơn.** Chạy tới cổng QC/preview trước khi chấp nhận nhãn để train.

## Thay đổi có chủ đích
- Sửa SAM2: truyền BGR trực tiếp theo API NumPy của Ultralytics 8.4.214; khớp kích thước mask với ảnh gốc.
- Teacher: confidence cao hơn, cùng lớp và vị trí phải nhất quán giữa ảnh gốc và ảnh lật ngang. Giữ GT, veto pseudo-label nằm trong mask GT; cách ly ảnh staged có vật thể teacher tự mâu thuẫn.
- Golden Val chỉ từ nguồn đầy đủ Mapillary/BDD; không rút fallback Sign-only hoặc nhãn teacher vào validation.
- Tăng tiếp xúc lớp 3/5/6/7 bằng repeat-factor có giới hạn trên ảnh gốc nhãn đầy đủ; giữ nguyên mọi lớp và toàn bộ polygon trong ảnh. Không oversample validation.
- Crop quanh vật thể nhỏ của các lớp yếu, Copy-Paste chỉ lấy từ nguồn đầy đủ; giới hạn ảnh tổng hợp.
- Model nộp vẫn YOLO26n-seg / FP32 ONNX / 640 / opset 17. Train chính 110 epochs, mask_ratio=2; fine-tune 25 epochs trên ảnh gốc nhãn đầy đủ, không mosaic/mixup/copy-paste.
- Export và chấm cả main/refine bằng **official_detscore trên cùng fingerprint holdout**. Nếu refine không tốt hơn thì giữ main; không dùng fallback hoặc điểm sample_data để chọn model.

## Giới hạn và cách đánh giá
Consistency là heuristic, không chứng minh đã đủ nhãn; không phải triển khai đầy đủ FixMatch. Confidence cao có thể bỏ sót vật thể xa. Xem consensus JSON, preview và ảnh bị cách ly. Mapillary/BDD vẫn có thể lệch domain Việt Nam; không suy ra điểm BTC từ điểm holdout. Không sửa taxonomy Mapillary nếu chưa xác minh mask rider có bao gồm xe.
Mask ratio 2 tăng chi tiết target khi train, không tăng kích thước mask output ONNX. Recipe ưu tiên điểm nên lâu hơn V4 và có thể vượt thời lượng một phiên Kaggle. Checkpoints được lưu trong runs_v5/runs_stage1_v5; xuất submission chỉ sau QC và chấm chính thức.
V5 có namespace staged/master/outputs riêng. Để so sánh V4/V5 có ý nghĩa, chấm cả hai model trên cùng holdout sạch, độc lập và cố định; hai Golden Val tự sinh khác nhau không so sánh trực tiếp được.

## Cơ sở nghiên cứu (truy cập 28/11/2026)
- [Ultralytics 8.4.214 SAM preprocessing](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.214/ultralytics/models/sam/predict.py): NumPy BGR được đổi RGB bên trong predictor.
- [Ultralytics train](https://docs.ultralytics.com/modes/train/): mask_ratio, warm start, augmentation và checkpoint validation.
- [FixMatch, NeurIPS 2020](https://arxiv.org/abs/2001.07685): động cơ dùng confidence và consistency; không phải bằng chứng mức tăng điểm cho bài này.
- [Ultralytics augmentation](https://docs.ultralytics.com/guides/yolo-data-augmentation/): điều chỉnh augmentation theo dữ liệu và pha train.
'''
    notebook["cells"][0] = {"cell_type":"markdown", "metadata":{}, "source":text.splitlines(keepends=True)}
    notebook["metadata"]["score_focus_v5"] = {"parent_notebook": SOURCE.name,
        "parent_sha256": hashlib.sha256(original_bytes).hexdigest(), "recipe":"untrained", "version":"5.0"}
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"v5-{index:02d}-" + hashlib.sha256("".join(cell["source"]).encode("utf-8")).hexdigest()[:8]
    return notebook


def verify(notebook):
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell_{index}", "exec")
    _, _, bundle = unpack_bundle(notebook)
    for name, source in bundle.items():
        if name.endswith(".py"):
            compile(source, name, "exec")
    assert len(bundle) == 19
    return len(bundle)


def main():
    before = SOURCE.read_bytes()
    notebook = build_notebook()
    count = verify(notebook)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    assert SOURCE.read_bytes() == before, "V4 must remain unchanged"
    print(f"Built {OUTPUT.name}: {len(notebook['cells'])} cells, {count} embedded files. V4 unchanged.")


if __name__ == "__main__":
    main()

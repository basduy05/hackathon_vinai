"""Build the consolidated V3.4 Stable notebook; prior notebooks are never rewritten."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import textwrap
import zlib

from tools.build_kaggle_v5_balanced import build_notebook as balanced_notebook
from tools.build_kaggle_v5_score_focus import code, unpack_bundle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "kaggle_team34_v34_stable.ipynb"


PARAMS = '''#@title 1. V3.4 Stable: cấu hình duy nhất và đường dẫn tuyệt đối
import os
import sys
import time
from pathlib import Path
TEAM_NAME, TEAM_ID = "Đội 34 – Nguyễn Bá Duy", 39
IS_KAGGLE = Path("/kaggle/working").exists()
RUN_STARTED = globals().get("V34_RUN_STARTED", time.monotonic())
V34_RUN_STARTED = RUN_STARTED
RUN_ROOT = (Path("/kaggle/working/team34_v34") if IS_KAGGLE else Path("work_kaggle_v34")).resolve()
REPO_DIR, TMP_RAW_DIR = RUN_ROOT / "repo", RUN_ROOT / "scratch"
STAGED_DIR, GOLDEN_VAL_DIR = RUN_ROOT / "staged", RUN_ROOT / "golden_val"
MASTER_DIR, OUTPUTS_DIR = RUN_ROOT / "master", RUN_ROOT / "outputs"
OUT_ZIP_PATH = OUTPUTS_DIR / "prepared_v34.zip"
for directory in (REPO_DIR, TMP_RAW_DIR, STAGED_DIR, OUTPUTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
os.environ["KAGGLEHUB_CACHE"] = str(TMP_RAW_DIR / "kh_cache")
INSTALL_DEPS, REQUIRE_GPU = True, True
RUN_KAGGLE_INPUT_PUBLIC = True
AUTO_DOWNLOAD_KAGGLE_SLUGS = [
    "kaggleprollc/mapillary-vistas-image-data-collection",
    "a7madmostafa/bdd100k-yolo",
    "lvnduy/street-traffic-signs-in-vietnam-yolov8",
]
SKIP_STOCK_INPUT_SLUGS = ["vehicle-dataset-for-yolo", "obstacle-detection-dataset", "hackathon", "master_v", "team34_v34"]
MAPILLARY_MAX_IMAGES, BDD_MAX_IMAGES = 2800, 600
KAGGLE_MAX_IMAGES_PER_SOURCE, VAL_MAX_IMAGES_PER_SOURCE = 500, 240
SAM2_MODEL = "sam2_b.pt"
RUN_PARTIAL_LABEL_COMPLETION, RUN_STAGE1_MAPILLARY_TEACHER = True, True
STAGE1_EPOCHS, TEACHER_WEIGHTS = 15, "yolo11x-seg.pt"
PSEUDO_MIN_CONF, QUARANTINE_TEACHER_DISAGREEMENT = .50, True
RUN_CROP_ZOOM, RUN_COPY_PASTE, RUN_WEATHER_AUG = True, True, True
MAX_EXTRA_CROPS, COPY_PASTE_COUNT, WEATHER_COUNT = 400, 200, 75
AUTO_GOLDEN_VAL_TARGET, GOLDEN_MIN_PER_CLASS = 320, 25
MIN_RARE_TRAIN_INSTANCES, SANITY_IMAGES_PER_SOURCE = 200, 48
RUN_SANITY_GATE, FAIL_ON_SANITY_ALERT = True, True
RUN_TRAIN_IN_THIS_NOTEBOOK, RUN_CLEAN_REFINEMENT = True, True
TRAIN_EPOCHS, TRAIN_BATCH, TRAIN_PATIENCE, REFINE_EPOCHS = 65, 16, 15, 8
TOTAL_BUDGET_HOURS, EXPORT_RESERVE_HOURS = 7.5, .75
TEACHER_CAP_HOURS, MAIN_CAP_HOURS, REFINE_CAP_HOURS = .65, 3., .35
RUN_BENCHMARK_AND_QC, RUN_LOCAL_TRAIN_QC = True, False
QC_TRAIN_SAMPLE_IMAGES, LOCAL_PACK_MAX_VAL_IMAGES = 500, 60
RUN_NAME, BASELINE_SUMMARY_JSON = "team34_v34_stable", ""
print("V3.4 Stable | outputs:", OUTPUTS_DIR, flush=True)
'''


SETUP = '''
# Install before importing binary/ML packages; fail here rather than hours later at export.
import importlib.metadata
import subprocess
import tempfile
import platform
if sys.version_info < (3,11):
    raise RuntimeError("The competition-pinned onnxruntime requires Python >=3.11; choose a current Kaggle image")
PINNED = {"ultralytics":"8.4.214", "onnx":"1.23.2", "onnxslim":"0.1.99", "onnxruntime":"1.32.0"}
if INSTALL_DEPS:
    mismatches = []
    for package, version in PINNED.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            if package in sys.modules:
                raise RuntimeError(f"{package} already imported at {installed}; restart the kernel before installing {version}")
            mismatches.append(f"{package}=={version}")
    specs = mismatches + ["pycocotools", "shapely", "huggingface_hub", "datasets", "kagglehub", "pyyaml", "matplotlib", "ipython"]
    if specs:
        pip = subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--retries", "3", "--timeout", "60", *specs], capture_output=True, text=True, timeout=900)
        (OUTPUTS_DIR / "pip_install.log").write_text(pip.stdout + pip.stderr, encoding="utf-8")
        if pip.returncode:
            raise RuntimeError("Dependency installation failed; inspect outputs/pip_install.log and enable Internet")
    importlib.invalidate_caches()
versions = {package: importlib.metadata.version(package) for package in PINNED}
if versions != PINNED:
    raise RuntimeError(f"Pinned package mismatch: {versions}; restart kernel and run setup first")
for package in ("cv2", "numpy", "shapely", "pycocotools.mask", "onnx", "onnxruntime", "ultralytics"):
    importlib.import_module(package)
import torch
if REQUIRE_GPU and not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU accelerator before downloading/preparing data")
DEVICE = "0" if torch.cuda.is_available() else "cpu"
if DEVICE == "0":
    torch.zeros(1, device="cuda")  # Verify CUDA actually initializes, not merely enumerates a device.
    torch.cuda.empty_cache()
TEMP_FILES_DIR = TMP_RAW_DIR / "tempfiles"
TEMP_FILES_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(TEMP_FILES_DIR)
tempfile.tempdir = str(TEMP_FILES_DIR)
repo_str = str(REPO_DIR.resolve())
if repo_str not in sys.path:
    sys.path.insert(0, repo_str)
os.chdir(REPO_DIR)
from src.stable_v34 import atomic_json, release_gpu
atomic_json(OUTPUTS_DIR / "environment.json", {"python":platform.python_version(), "versions":versions,
    "torch":torch.__version__, "device":DEVICE,
    "gpu":torch.cuda.get_device_name(0) if DEVICE == "0" else None})
print("V3.4 preflight passed:", versions, DEVICE, flush=True)
'''


INITIALIZE = '''#@title 4. Khởi tạo staging; chỉ tải SAM2 khi thật sự cần bbox -> mask
from src.stable_v34 import LazySAM
STAGED_SOURCES, CLEAN_VAL_SOURCES, CLEAN_TRAIN_SOURCES = [], [], []
SAM_PREDICTOR = LazySAM(SAM2_MODEL, DEVICE)
SOURCE_FAILURES = []
'''


SOURCES = '''#@title 5. Tải có retry, hỗ trợ input chỉ đọc và cache conversion có kiểm chứng
import kagglehub
from src.kaggle_adapter import detect_dataset_format, detect_val_split, auto_convert_detected_dataset
from src.stable_v34 import retry_download, unpack_readonly_input, cached_stage, atomic_json

for index, slug in enumerate(AUTO_DOWNLOAD_KAGGLE_SLUGS):
    short = slug.split("/")[-1]
    try:
        mounted = next((info for info in DISCOVERED_KAGGLE if short.replace("-", "_") in str(info).lower().replace("-", "_")), None)
        info = mounted
        if info is None:
            source_root = Path(retry_download(lambda: kagglehub.dataset_download(slug)))
            info = detect_dataset_format(source_root)
            if info is None:
                extracted = unpack_readonly_input(source_root, TMP_RAW_DIR / "extracted" / short)
                info = detect_dataset_format(extracted)
        if info is None:
            raise ValueError("Dataset format not detected; attach an extracted, annotated dataset")
        info = dict(info, name=slug.replace("/", "__"), url="https://www.kaggle.com/datasets/"+slug)
        safe = info["name"].replace("-", "_")
        full = any(marker in safe for marker in ("mapillary", "bdd100k", "cityscapes"))
        cap = MAPILLARY_MAX_IMAGES if "mapillary" in safe else BDD_MAX_IMAGES if "bdd100k" in safe else KAGGLE_MAX_IMAGES_PER_SOURCE
        out = STAGED_DIR / f"kaggle_{index:02d}_{safe}"
        signature = {"recipe":"3.4", "source":slug, "descriptor":info, "cap":cap, "seed":TEAM_ID}
        result = cached_stage(out, signature, lambda destination: auto_convert_detected_dataset(
            info, destination, sam_predictor=SAM_PREDICTOR, max_images=cap, seed=TEAM_ID, split="train",
            registry_path=REPO_DIR / "datasets_registry.md"))
        STAGED_SOURCES.append(out)
        if full:
            CLEAN_TRAIN_SOURCES.append(out)
            val_info = detect_val_split(info)
            if val_info:
                val_out = STAGED_DIR / f"_val_{index:02d}_{safe}"
                val_cap = VAL_MAX_IMAGES_PER_SOURCE if "mapillary" in safe else 90
                val_signature = dict(signature, descriptor=val_info, cap=val_cap, split="val")
                cached_stage(val_out, val_signature, lambda destination: auto_convert_detected_dataset(
                    val_info, destination, sam_predictor=SAM_PREDICTOR, max_images=val_cap,
                    seed=TEAM_ID, split="val", registry_path=None))
                CLEAN_VAL_SOURCES.append(val_out)
        print("[SOURCE OK]", slug, result, flush=True)
    except Exception as exc:
        failure = {"source":slug, "error_type":type(exc).__name__, "message":str(exc)}
        SOURCE_FAILURES.append(failure)
        print("[SOURCE FAILED]", failure, flush=True)
    finally:
        atomic_json(OUTPUTS_DIR / "source_failures.json", SOURCE_FAILURES)
        # Never delete mounted /kaggle/input resources or unrelated user caches.
        owned_cache = TMP_RAW_DIR / "kh_cache"
        if owned_cache.is_dir():
            shutil.rmtree(owned_cache)
SAM_PREDICTOR.release()
if not CLEAN_TRAIN_SOURCES:
    raise RuntimeError("No fully annotated street source available. See source_failures.json; attach Mapillary/BDD and rerun data cells")
'''


TRAIN = '''#@title 10. Train phục hồi OOM, xuất dự phòng, chấm chính thức rồi chọn checkpoint
from ultralytics import YOLO
from src.stable_v34 import train_resilient, release_gpu, atomic_json
from src.balanced_budget import phase_hours, StopAtDeadline
from src.score_focus_v5 import make_repeat_factor_dataset, holdout_fingerprint, choose_official_candidate
from src.export_verify_onnx import export_and_verify
from src.eval_golden import run_eval_or_qc

BEST_PT_PATH = SUBMISSION_ZIP_PATH = SELECTED_TRAIN_DIR = None
if RUN_TRAIN_IN_THIS_NOTEBOOK:
    train_yaml = MASTER_DIR / "data_repeat.yaml"
    sampling = make_repeat_factor_dataset(MASTER_DIR, train_yaml, max_extra_fraction=.20, seed=TEAM_ID)
    print("[SAMPLING]", sampling, flush=True)
    run_proj = RUN_ROOT / "runs"
    common = dict(imgsz=640, batch=TRAIN_BATCH, device=DEVICE, workers=2, cache=False, seed=TEAM_ID,
        deterministic=True, amp=True, optimizer="AdamW", cos_lr=True, mask_ratio=4, overlap_mask=True,
        max_det=100, save_period=5, project=str(run_proj), exist_ok=True)
    main_hours = phase_hours(RUN_STARTED, TOTAL_BUDGET_HOURS, EXPORT_RESERVE_HOURS, MAIN_CAP_HOURS)
    if main_hours < .25:
        raise RuntimeError("Not enough time for meaningful training; keep master data and start a new GPU session")
    phase_end = time.monotonic() + main_hours*3600
    make_deadline = lambda: StopAtDeadline(max(.001, (phase_end-time.monotonic())/3600))
    kwargs = dict(common, data=str(train_yaml), epochs=TRAIN_EPOCHS, patience=TRAIN_PATIENCE,
        lr0=.0015, lrf=.02, weight_decay=.0005, warmup_epochs=3., scale=.5, degrees=2., hsv_v=.35,
        mosaic=.6, close_mosaic=12, copy_paste=.10, mixup=0., multi_scale=False, name=RUN_NAME+"_main")
    main_best, attempts = train_resilient(YOLO, "yolo26n-seg.pt", kwargs, make_deadline)
    atomic_json(OUTPUTS_DIR / "train_attempts.json", {"main":attempts})
    # Publish a validated recovery artifact before any optional work.
    main_dir = OUTPUTS_DIR / "candidates/main"
    exported = {"main": export_and_verify(main_best, main_dir, notebook_path=REPO_DIR / "train_and_export.ipynb",
        smoke_images_dir=MASTER_DIR / "images/val")}
    shutil.copy2(main_dir / "submission.zip", OUTPUTS_DIR / "recovery_main_submission.zip")
    shutil.copy2(main_dir / "submission.zip", OUTPUTS_DIR / "submission.zip")
    candidates = [("main", main_best)]
    refine_hours = phase_hours(RUN_STARTED, TOTAL_BUDGET_HOURS, EXPORT_RESERVE_HOURS, REFINE_CAP_HOURS)
    if RUN_CLEAN_REFINEMENT and refine_hours >= .10:
        try:
            refine_yaml = MASTER_DIR / "data_clean_refine.yaml"
            make_repeat_factor_dataset(MASTER_DIR, refine_yaml, clean_only=True, max_extra_fraction=.20, seed=TEAM_ID)
            phase_end = time.monotonic() + refine_hours*3600
            refine_kwargs = dict(common, mask_ratio=2, batch=min(TRAIN_BATCH,8), data=str(refine_yaml),
                epochs=REFINE_EPOCHS, patience=REFINE_EPOCHS, lr0=.00015, lrf=.1, warmup_epochs=1.,
                scale=.15, degrees=0., hsv_v=.15, mosaic=0., close_mosaic=0, copy_paste=0., mixup=0.,
                multi_scale=False, name=RUN_NAME+"_refine")
            refine_best, refine_attempts = train_resilient(YOLO, main_best, refine_kwargs, make_deadline)
            atomic_json(OUTPUTS_DIR / "refine_attempts.json", refine_attempts)
            candidates.append(("clean_refine", refine_best))
        except Exception as exc:
            atomic_json(OUTPUTS_DIR / "optional_refine_failure.json", {"error_type":type(exc).__name__, "message":str(exc), "fallback":"validated main checkpoint"})
            print("[REFINE SKIPPED]", exc, flush=True)
        finally:
            release_gpu()
    fingerprint = holdout_fingerprint(MASTER_DIR)
    evaluated = []
    for name, weights in candidates:
        candidate_dir = OUTPUTS_DIR / "candidates" / name
        try:
            if name not in exported:
                exported[name] = export_and_verify(weights, candidate_dir, notebook_path=REPO_DIR / "train_and_export.ipynb",
                    smoke_images_dir=MASTER_DIR / "images/val")
            evaluation = run_eval_or_qc(Path(exported[name]["onnx"]), MASTER_DIR / "images/val", MASTER_DIR / "labels/val",
                candidate_dir / "official_validation", mode="eval", notebook_path=REPO_DIR / "train_and_export.ipynb")
            candidate = {"name":name, "weights":str(weights), "export_dir":str(candidate_dir),
                "holdout_sha256":fingerprint, "n_images":evaluation["n_images"], "score_summary":evaluation["score_summary"]}
            choose_official_candidate([candidate])  # A fallback metric must never choose the submitted model.
            evaluated.append(candidate)
        except Exception as exc:
            atomic_json(candidate_dir / "evaluation_failure.json", {"error_type":type(exc).__name__, "message":str(exc)})
            if name == "main":
                raise RuntimeError("Official evaluation of main failed; recovery_main_submission.zip is validated but not scored") from exc
            print("[OPTIONAL CANDIDATE SKIPPED]", name, exc, flush=True)
    winner = choose_official_candidate(evaluated)
    atomic_json(OUTPUTS_DIR / "checkpoint_selection.json", {"selected":winner["name"], "candidates":evaluated})
    BEST_PT_PATH = Path(winner["weights"])
    SELECTED_TRAIN_DIR = BEST_PT_PATH.parent.parent
    shutil.copy2(BEST_PT_PATH, OUTPUTS_DIR / "best.pt")
    for filename in ("model.onnx", "submission.zip"):
        shutil.copy2(Path(winner["export_dir"]) / filename, OUTPUTS_DIR / filename)
    SUBMISSION_ZIP_PATH = OUTPUTS_DIR / "submission.zip"
    print("[VALIDATED AND SCORED]", winner["name"], winner["score_summary"]["score"], SUBMISSION_ZIP_PATH, flush=True)
'''


def build_notebook():
    notebook = balanced_notebook()
    source, node, bundle = unpack_bundle(notebook)
    bundle["src/stable_v34.py"] = (ROOT / "src/stable_v34.py").read_text(encoding="utf-8")
    raw = base64.b64encode(zlib.compress(json.dumps(bundle, ensure_ascii=False).encode("utf-8"), 9)).decode("ascii")
    lines = source.splitlines(keepends=True)
    assignment = "_BUNDLE_B64 = (\n" + "\n".join(repr(x) for x in textwrap.wrap(raw, 100)) + "\n)\n"
    source = "".join(lines[:node.lineno-1]) + assignment + "".join(lines[node.end_lineno:])
    # Keep only extraction: dependency installation and path initialization are consolidated below.
    source = source[:source.index('if INSTALL_DEPS:')]
    notebook["cells"][2] = code(source)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            text = "".join(cell["source"]).replace("team34_v5_balanced", "team34_v34_stable")
            cell.update(code(text))
    notebook["cells"][1] = code(PARAMS)
    notebook["cells"][2] = code("".join(notebook["cells"][2]["source"]).replace("V5 Score Focus", "V3.4 Stable") + SETUP)
    notebook["cells"][4] = code(INITIALIZE)
    notebook["cells"][5] = code(SOURCES)
    notebook["cells"][6] = code('#@title 6. Giải phóng SAM2 trước teacher/train\nSAM_PREDICTOR.release()\nprint("SAM2 VRAM released", flush=True)\n')
    completion = "".join(notebook["cells"][8]["source"])
    completion = completion.replace('if RUN_STAGE1_MAPILLARY_TEACHER and full_sources_for_stage1:',
        'if RUN_STAGE1_MAPILLARY_TEACHER and full_sources_for_stage1 and (vn_partial_sources or coco_sources):')
    completion = completion.replace('from src.build_master import build_master_dataset',
        'from src.build_master import build_master_dataset\nfrom src.stable_v34 import train_resilient, release_gpu, atomic_json')
    start = completion.index('        s1_model = YOLO(')
    end = completion.index('        shutil.copy2(s1_best,', start)
    completion = completion[:start] + '''        s1_proj = RUN_ROOT / "runs_teacher"
        teacher_hours = phase_hours(RUN_STARTED, TOTAL_BUDGET_HOURS, EXPORT_RESERVE_HOURS + 1.5, TEACHER_CAP_HOURS)
        if teacher_hours < .10:
            raise RuntimeError("Insufficient time for optional teacher training")
        teacher_end = time.monotonic() + teacher_hours*3600
        teacher_kwargs = dict(data=str(stage1_data_dir / "data.yaml"), imgsz=640, epochs=STAGE1_EPOCHS,
            batch=min(8,TRAIN_BATCH), seed=TEAM_ID, device=DEVICE, workers=2, cache=False,
            optimizer="AdamW", lr0=.001, cos_lr=True, mask_ratio=4, save_period=5,
            mosaic=.5, mixup=0., copy_paste=0., close_mosaic=5, max_det=100,
            project=str(s1_proj), name="stage1_teacher", exist_ok=True)
        s1_best, teacher_attempts = train_resilient(YOLO, "yolo26s-seg.pt", teacher_kwargs,
            lambda: StopAtDeadline(max(.001,(teacher_end-time.monotonic())/3600)))
        atomic_json(OUTPUTS_DIR / "teacher_train_attempts.json", teacher_attempts)
''' + completion[end:]
    completion = completion.replace('        del s1_model\n', '')
    # Teacher completion is optional, but retaining an incompletely labeled VN source is not.
    completion = '''#@title 7B. Bổ sung nhãn VN; loại nguồn partial nếu teacher không hoàn tất
try:
''' + textwrap.indent(completion, '    ') + '''
except Exception as exc:
    from src.stable_v34 import atomic_json, release_gpu
    atomic_json(OUTPUTS_DIR / "teacher_completion_failure.json", {"error_type":type(exc).__name__, "message":str(exc), "action":"exclude every partial-label source from training"})
    STAGED_SOURCES = [s for s in STAGED_SOURCES if not any(k in s.name.lower() for k in ("vnts", "lvnduy", "traffic_sign", "coco"))]
    print("[OPTIONAL PARTIAL SOURCES EXCLUDED]", exc, flush=True)
finally:
    for _name in ("teacher_coco", "teacher_map", "s1_model"):
        globals().pop(_name, None)
    from src.stable_v34 import release_gpu
    release_gpu()
'''
    notebook["cells"][8] = code(completion)
    gate = "".join(notebook["cells"][10]["source"])
    gate = gate.replace('if not torch.cuda.is_available():', 'if REQUIRE_GPU and not torch.cuda.is_available():').replace('V5', 'V3.4')
    gate += '''
from src.stable_v34 import atomic_json, dataset_fingerprint
atomic_json(OUTPUTS_DIR / "dataset_manifest.json", {"recipe":"3.4-stable", "seed":TEAM_ID,
    "fingerprint":dataset_fingerprint(MASTER_DIR), "master_summary":master_summary,
    "source_failures":SOURCE_FAILURES})
'''
    notebook["cells"][10] = code(gate)
    notebook["cells"][11] = code(TRAIN)
    # Plot/report generation is optional and must not invalidate an already scored submission.
    reporting = "".join(notebook["cells"][12]["source"])
    reporting = reporting.replace(reporting.splitlines()[1], 'QUICK_PACK_PATH = OUTPUTS_DIR / "quick_local_pack.zip"', 1)
    notebook["cells"][12] = code('#@title 11. Báo cáo phụ, không thay đổi submission đã kiểm định\ntry:\n' + textwrap.indent(reporting, '    ') + '''
except Exception as exc:
    from src.stable_v34 import atomic_json
    atomic_json(OUTPUTS_DIR / "optional_report_failure.json", {"error_type":type(exc).__name__, "message":str(exc)})
    print("[REPORT OPTIONAL]", exc, "Validated submission remains in", OUTPUTS_DIR, flush=True)
''')
    downloads = '''#@title 12. Tải submission và chẩn đoán
from IPython.display import FileLink, display
os.chdir(Path("/kaggle/working") if IS_KAGGLE else RUN_ROOT)
for filename in ("submission.zip", "recovery_main_submission.zip", "checkpoint_selection.json", "best.pt",
                 "environment.json", "source_failures.json", "train_attempts.json", "dataset_manifest.json",
                 "holdout_manifest.json", "benchmark_report.md", "optional_refine_failure.json",
                 "teacher_completion_failure.json", "prepared_v34.zip"):
    path = OUTPUTS_DIR / filename
    if path.is_file():
        display(FileLink(os.path.relpath(path, Path.cwd())))
print("V3.4 output directory:", OUTPUTS_DIR)
'''
    notebook["cells"][13] = code(downloads)
    intro = '''# Team 34 — V3.4 Stable, bản hợp nhất tối ưu và kiểm tra lỗi

**File độc lập: import notebook này lên Kaggle, bật GPU và Internet, chạy từ Cell 1.**
Tên phiên bản V3.4 theo yêu cầu; giữ các sửa chất lượng của nhánh V5 Balanced, không ghi đè notebook cũ.

## Cấu hình
YOLO26n-seg 11 lớp, train chính 65 epochs/batch16/mask_ratio4; teacher 15 epochs; fine-tune tối đa 8 epochs/batch8/mask_ratio2. Giữ 640/FP32/opset17 theo hợp đồng BTC. Ước lượng thời gian chưa được đo trên Kaggle; không bảo đảm điểm cao hơn khi chưa chạy đối chiếu.

## Xử lý các lỗi đã xác định
- Kiểm tra đúng phiên bản các gói và CUDA ngay từ đầu, trước khi xử lý dữ liệu. Log pip và environment được giữ trong outputs.
- Kagglehub có thể trả thư mục `/kaggle/input` chỉ đọc: tuyệt đối không giải nén hoặc xóa bên trong đó. ZIP được giải nén sang scratch riêng, kiểm tra đường dẫn và dung lượng trước.
- Chỉ tải SAM2 khi gặp nguồn bbox; truyền BGR đúng API; giải phóng VRAM trước teacher/train.
- Retry có giới hạn cho lỗi mạng tạm thời; không lặp lại 401/403/404 hay che lỗi dữ liệu. Nguồn tải lỗi được ghi source_failures.json. Thiếu nguồn đầy đủ/GT tối thiểu phải dừng để tránh train sai.
- Cache conversion chỉ dùng khi fingerprint và cấu hình khớp. Cache chưa hoàn tất không được xem là thành công. Rerun sau khi label bị thay đổi sẽ chuyển đổi lại từ nguồn, không tái sử dụng label giả như GT.
- CUDA OOM: giảm batch có giới hạn, giải phóng model/tensor trước lần thử tiếp; lỗi khác được báo riêng. Checkpoint mỗi 5 epochs; không vô hiệu hóa validator để cố xuất file.
- Nếu teacher lỗi, loại nguồn partial-label khỏi train; không giữ ảnh Sign-only làm nền âm. Nếu fine-tune/biểu đồ lỗi, giữ submission chính đã kiểm định. Official scoring của main vẫn là bước bắt buộc.
- Xuất `recovery_main_submission.zip` đã qua validator trước công việc phụ. `checkpoint_selection.json` chứng minh checkpoint nào thắng trên cùng holdout.

## Rerun và phục hồi
Đường dẫn tuyệt đối nằm dưới `/kaggle/working/team34_v34`. Đồng hồ không đặt lại khi rerun Cell 1 trong cùng kernel; mở kernel mới để bắt đầu một phiên mới. Nếu một nguồn mạng lỗi, xem source_failures.json, attach nguồn được giải nén sẵn và rerun các cell dữ liệu theo thứ tự. Không chạy train bỏ qua cổng dữ liệu.
Giữ master/prepared_v34.zip và thư mục runs để phục hồi một phiên bị ngắt. Cache chỉ tồn tại nếu ổ working còn được giữ; notebook không tự khôi phục dữ liệu của phiên Kaggle đã bị hủy.

## Giới hạn của kiểm chứng
Test local và chạy mô phỏng có thể chứng minh đường dẫn/chọn model/fallback, nhưng không chứng minh truy cập Kaggle API, CUDA thực, đủ bộ nhớ trên mọi GPU hoặc chất lượng AP. Không có cam kết “không bao giờ lỗi”. Nếu thiếu quyền dữ liệu, không đủ đĩa hoặc lớp GT, notebook dừng sớm với log thay vì train hoặc nộp một model không hợp lệ.
Ngân sách 7,5 giờ là cơ chế dừng train, không phải giới hạn cứng cho tải/giải nén/validation. Giữ 45 phút export; mạng hoặc dữ liệu chậm vẫn có thể vượt thời lượng phiên.

## Nguồn kỹ thuật
- [Kagglehub: Kaggle integration/read-only shared resources](https://github.com/Kaggle/kagglehub/blob/main/README.md)
- [Ultralytics 8.4.214 SAM preprocessing](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.214/ultralytics/models/sam/predict.py)
- [Ultralytics training](https://docs.ultralytics.com/modes/train/)
- [FixMatch: confidence/consistency motivation](https://arxiv.org/abs/2001.07685) — không phải triển khai đầy đủ FixMatch.
'''
    notebook["cells"][0] = {"cell_type":"markdown", "metadata":{}, "source":intro.splitlines(keepends=True)}
    notebook["metadata"]["stable_v34"] = {"version":"3.4-stable", "gpu_training_verified":False}
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"v34-{index:02d}-" + hashlib.sha256("".join(cell["source"]).encode()).hexdigest()[:8]
    return notebook


def verify(notebook):
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"v34_cell_{i}", "exec")
    bundle = unpack_bundle(notebook)[2]
    for name, text in bundle.items():
        if name.endswith(".py"):
            compile(text, name, "exec")
    assert len(bundle) == 21


if __name__ == "__main__":
    notebook = build_notebook()
    verify(notebook)
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Created {OUTPUT.name}: 14 cells, 21 embedded files; earlier notebooks unchanged.")

"""Build self-contained V3.3 Kaggle notebook ``kaggle_team34_pipeline.ipynb`` for Team 34 (TEAM_ID=39)."""
from __future__ import annotations

import ast
import base64
import json
import py_compile
import tempfile
import textwrap
import zlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_NB = REPO_ROOT / "kaggle_team34_pipeline.ipynb"

EMBED_REL_PATHS = [
    "configs/competition.yaml",
    "configs/class_mapping.yaml",
    "datasets_registry.md",
    "train_and_export.ipynb",
    "src/__init__.py",
    "src/selfcheck_bridge.py",
    "src/validate_dataset.py",
    "src/export_verify_onnx.py",
    "src/seg_utils.py",
    "src/map_and_merge_public.py",
    "src/box_to_mask_sam2.py",
    "src/prepare_vn_frames.py",
    "src/augment_and_crop.py",
    "src/eval_golden.py",
    "src/build_master.py",
    "src/data_gates.py",
    "src/kaggle_adapter.py",
    "src/benchmark_report.py",
]


def _md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.rstrip("\n").splitlines()],
    }


def _code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.rstrip("\n").splitlines()],
    }


def build_notebook() -> dict:
    bundle: dict[str, str] = {}
    for rel in EMBED_REL_PATHS:
        p = REPO_ROOT / rel
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file to embed: {p}")
        bundle[rel] = p.read_text(encoding="utf-8")

    raw_bytes = json.dumps(bundle, ensure_ascii=False).encode("utf-8")
    b64_str = base64.b64encode(zlib.compress(raw_bytes, level=9)).decode("ascii")
    b64_wrapped = "\n".join(repr(chunk) for chunk in textwrap.wrap(b64_str, 100))

    cells = [
        _md_cell(
            """# Đội 34 – Nguyễn Bá Duy (`TEAM_ID = 39`): Pipeline Data-Centric V3.3 Trên Kaggle

**Chỉ cần tải duy nhất file này lên Kaggle (`File → Import Notebook`), bật `GPU T4` + `Internet = ON`, rồi bấm `Save Version -> Save & Run All`!**

### Những lỗi V2 đã được sửa triệt để trong bản V3.3
1. **Phát hiện lỗi đảo lớp (`van -> Motorcyclist`, `truck -> Other vehicle`)**: `kaggle_adapter` nay đọc cả `classes.txt`, `obj.names`, 58 mã biển báo VN (`P.*`, `W.*`, `R.*`) và **cấm fallback đồng nhất `{0:0..10:10}`**, kết hợp loại bỏ bộ ảnh stock studio `vehicle-dataset-for-yolo`.
2. **Thay 100% slug lỗi 403 bằng 5 bộ dữ liệu đường phố sạch đã kiểm chứng**:
   - `lvnduy/street-traffic-signs-in-vietnam-yolov8` (Biển báo VN thực tế)
   - `abtinzandi/obstacle-detection-dataset` (Đường phố đô thị)
   - `a7madmostafa/bdd100k-yolo` (Dashcam BDD100K ngày/đêm/mưa)
   - `redzapdos123/indian-driving-dataset-detections-yolov11` (`autorickshaw/trailer` → `Other vehicle`, `rider+2-wheeler` → `5/7`)
   - `awsaf49/coco-2017-dataset` (Mask COCO polygon chuẩn xác, ưu tiên lọc lớp hiếm `Bicycle`, `Cyclist`, `Motorcycle`, `Traffic light`)
3. **Giải nén `.zip` chuẩn trên HuggingFace (`star092304/Traffic-sign-detection-VietNam`)**: Không còn lấy nhầm 2 ảnh minh họa README.
4. **Streaming Tải → Chuyển đổi → Xóa cache thô ngay**: Giảm dung lượng giữ đồng thời; vẫn cần đủ chỗ cho một bộ dữ liệu được giải nén.
5. **Golden Val có quota + Sanity Gate theo nguồn trước khi train**: Golden Val ưu tiên lấy từ tập `val` gốc và nguồn nhãn người được chỉ định (mask SAM2 vẫn cần kiểm tra) + chạy đối chiếu Zero-Shot COCO & xuất lưới kiểm tra 11 lớp `preview_per_class.jpg` ngay ở Cell 9!"""
        ),
        _code_cell(
            """#@title 1. Tham số Đội 34, Nguồn Sạch V3 & Tùy Chọn Train { display-mode: "form" }
TEAM_NAME = "Đội 34 – Nguyễn Bá Duy"
TEAM_ID = 39  # Seed cố định của đội

import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

IS_KAGGLE = Path("/kaggle/working").exists()
REPO_DIR = Path("/kaggle/working/repo") if IS_KAGGLE else Path("work_kaggle/repo")
TMP_RAW_DIR = Path("/kaggle/tmp/raw") if IS_KAGGLE else Path("work_kaggle/tmp_raw")
STAGED_DIR = Path("/kaggle/working/staged") if IS_KAGGLE else Path("work_kaggle/staged")
GOLDEN_VAL_DIR = Path("/kaggle/working/golden_val") if IS_KAGGLE else Path("work_kaggle/golden_val")
MASTER_DIR = Path("/kaggle/working/master_v3") if IS_KAGGLE else Path("work_kaggle/master_v3")
OUT_ZIP_PATH = Path("/kaggle/working/du_lieu_v3.zip") if IS_KAGGLE else Path("work_kaggle/du_lieu_v3.zip")
OUTPUTS_DIR = Path("/kaggle/working/outputs") if IS_KAGGLE else Path("work_kaggle/outputs")

# Đặt cache kagglehub sang /kaggle/tmp để không tốn dung lượng /kaggle/working
os.environ["KAGGLEHUB_CACHE"] = str(TMP_RAW_DIR / "kh_cache")

INSTALL_DEPS = True

# ---- 1A. Bộ 5 Kaggle Public Datasets ĐÃ KIỂM CHỨNG (tải tuần tự -> convert -> xóa cache thô) ----
AUTO_DOWNLOAD_KAGGLE_SLUGS = [
    "lvnduy/street-traffic-signs-in-vietnam-yolov8",          # ~837 MB: 58 mã biển báo VN thật -> SAM2 -> Sign(9)
    "abtinzandi/obstacle-detection-dataset",                  # ~1.5 GB: đường phố đô thị
    "a7madmostafa/bdd100k-yolo",                              # ~5.7 GB: dashcam BDD100K (night/rain/urban)
    "redzapdos123/indian-driving-dataset-detections-yolov11", # ~21.9 GB: giao thông châu Á, autorickshaw -> Other vehicle(3)
    "awsaf49/coco-2017-dataset",                              # ~25.2 GB: polygon mask chuẩn (ưu tiên lớp hiếm)
]
RUN_KAGGLE_INPUT_PUBLIC = True
SKIP_STOCK_INPUT_SLUGS = [
    "vehicle-dataset-for-yolo",  # Bỏ qua bộ ảnh stock studio gây sai tỉ lệ nếu lỡ gắn sẵn
    "hackathon",                 # Chặn output cũ của chính notebook này (tránh nạp lại master_v2 bẩn)
    "master_v1",
    "master_v2",
    "bdd100k-with-yolo-setup",
    "derekxue",
]
KAGGLE_MAX_IMAGES_PER_SOURCE = 1000
COCO_MAX_IMAGES = 1500
VAL_MAX_IMAGES_PER_SOURCE = 90  # Lấy tối đa từ val gốc mỗi nguồn sạch cho Golden Val

# ---- 1B. Biển báo Việt Nam & Dashcam Việt Nam (HuggingFace) ----
RUN_VNTS_HF = True
VNTS_HF_REPO = "star092304/Traffic-sign-detection-VietNam"
VNTS_MAX_IMAGES = 1000
SAM2_MODEL = "sam2_b.pt"

# Mặc định TẮT Zalo dashcam OpenHay vì video gốc bị làm mờ Gaussian toàn bộ xe/người; bật True nếu đã có video sạch
RUN_VN_DASHCAM = False
VN_DASHCAM_HF_REPO = "OpenHay/zalo-ai-2025-public-test-data-v2"
VN_DASHCAM_LOCAL_INPUT = ""
DASHCAM_MAX_FRAMES = 250
TEACHER_WEIGHTS = "yolo11x-seg.pt"

# ---- 1C. Augment, Golden Val & Sanity Gate ----
RUN_CROP_ZOOM = True
RUN_COPY_PASTE = True
COPY_PASTE_COUNT = 350
RUN_WEATHER_AUG = True
WEATHER_COUNT = 350
AUTO_GOLDEN_VAL_TARGET = 320
GOLDEN_MIN_PER_CLASS = 25
SANITY_IMAGES_PER_SOURCE = 48
MIN_RARE_TRAIN_INSTANCES = 150
RUN_SANITY_GATE = True
FAIL_ON_SANITY_ALERT = True

# ---- 1D. Huấn luyện, Đóng Gói Test Laptop & Tải File ----
RUN_TRAIN_IN_THIS_NOTEBOOK = True
TRAIN_EPOCHS = 65
TRAIN_BATCH = 16
TRAIN_PATIENCE = 20
RUN_BENCHMARK_AND_QC = True
QC_TRAIN_SAMPLE_IMAGES = 500
RUN_NAME = "team34_v33"
BASELINE_SUMMARY_JSON = ""
LOCAL_PACK_MAX_VAL_IMAGES = 60
INLINE_MAX_MB = 12.0
SPLIT_PACK_FOR_DOWNLOAD = False
SPLIT_ZIP_FOR_DOWNLOAD = False

print(f"{TEAM_NAME} | TEAM_ID={TEAM_ID} | IS_KAGGLE={IS_KAGGLE} | V3.3 Clean Pipeline", flush=True)"""
        ),
        _code_cell(
            f"""#@title 2. Cài đặt gói chuẩn & giải nén mã nguồn nhúng vào REPO_DIR
import base64
import importlib.metadata
import json
import shutil
import subprocess
import sys
import zlib

_BUNDLE_B64 = (
{b64_wrapped}
)
EMBEDDED_FILES = json.loads(zlib.decompress(base64.b64decode(_BUNDLE_B64)).decode("utf-8"))

for rel_path, content in EMBEDDED_FILES.items():
    target = REPO_DIR / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

print(f"Đã giải nén {{len(EMBEDDED_FILES)}} tệp mã nguồn V3.3 vào: {{REPO_DIR.resolve()}}", flush=True)

if INSTALL_DEPS:
    specs = [
        "ultralytics==8.4.214",
        "onnx==1.23.2",
        "onnxslim==0.1.99",
        "onnxruntime==1.32.0",
        "pycocotools",
        "shapely",
        "huggingface_hub",
        "datasets",
        "kagglehub",
        "pyyaml",
        "matplotlib",
    ]
    print("Đang cài đặt thư viện chuẩn cuộc thi...")
    proc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", *specs], capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-1000:], proc.stderr[-1000:])
        raise RuntimeError(f"pip install failed with code {{proc.returncode}}")

for pkg in ("ultralytics", "onnx", "onnxslim", "onnxruntime", "pycocotools", "shapely", "datasets", "kagglehub"):
    try:
        print(f"  - {{pkg}}: {{importlib.metadata.version(pkg)}}")
    except importlib.metadata.PackageNotFoundError:
        print(f"  - {{pkg}}: NOT INSTALLED")

import torch
DEVICE = "0" if torch.cuda.is_available() else "cpu"
print("GPU:", torch.cuda.get_device_name(0) if DEVICE == "0" else "CPU (CẢNH BÁO: Rất chậm!)")

repo_str = str(REPO_DIR.resolve())
if repo_str not in sys.path:
    sys.path.insert(0, repo_str)
os.chdir(repo_str)
TMP_RAW_DIR.mkdir(parents=True, exist_ok=True)
STAGED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)"""
        ),
        _code_cell(
            """#@title 3. Dò /kaggle/input (chống bọc thư mục datasets/) & Chuẩn bị hàng đợi Kagglehub
from src.kaggle_adapter import detect_dataset_format, discover_all_kaggle_inputs

DISCOVERED_KAGGLE = []
KAGGLEHUB_QUEUE = []

if RUN_KAGGLE_INPUT_PUBLIC:
    skip_slugs = {s.lower() for s in SKIP_STOCK_INPUT_SLUGS}
    mounted = discover_all_kaggle_inputs(Path("/kaggle/input"), skip_names=skip_slugs)
    for info in mounted:
        nm_low = f"{info.get('name', '')} {info.get('images_dir', '')} {info.get('labels_path', '')}".lower()
        if any(sk in nm_low for sk in skip_slugs):
            print(f"[SKIP] Bỏ qua nguồn cũ/stock không dùng cho V3: {info.get('name')}")
            continue
        DISCOVERED_KAGGLE.append(info)

    mounted_text = " ".join(f"{d.get('name', '')} {d.get('images_dir', '')}".lower() for d in DISCOVERED_KAGGLE)
    for slug in AUTO_DOWNLOAD_KAGGLE_SLUGS:
        slug_short = slug.split("/")[-1].lower()
        if slug_short in mounted_text:
            print(f"[SKIP] Đã gắn sẵn trong /kaggle/input: {slug}")
        else:
            KAGGLEHUB_QUEUE.append(slug)

print(f"Đã gắn sẵn: {len(DISCOVERED_KAGGLE)} | Sẽ tải-chuyển đổi-xóa cache tuần tự ở Cell 5: {len(KAGGLEHUB_QUEUE)}")
for s in KAGGLEHUB_QUEUE:
    print("  - https://www.kaggle.com/datasets/" + s)"""
        ),
        _code_cell(
            """#@title 4. Luồng 1: Biển báo Việt Nam HF (tự giải nén .zip -> SAM 2 tách mask sạch)
import random
from huggingface_hub import snapshot_download
from src.box_to_mask_sam2 import UltralyticsSAM2Predictor, convert_bbox_dataset
from src.kaggle_adapter import export_hf_parquet_to_yolo_bbox, unzip_archives_in_dir
from src.seg_utils import transfer_image
from src.validate_dataset import IMG_EXTS

STAGED_SOURCES: list[Path] = []
CLEAN_VAL_SOURCES: list[Path] = []
CLEAN_TRAIN_SOURCES: list[Path] = []
SAM_PREDICTOR = UltralyticsSAM2Predictor(model_name=SAM2_MODEL, device=DEVICE)


def _stage_yolo_bbox_folder(img_paths: list[Path], out_raw: Path, max_n: int) -> int:
    rng = random.Random(TEAM_ID)
    chosen = sorted(rng.sample(img_paths, max_n)) if len(img_paths) > max_n else img_paths
    (out_raw / "images").mkdir(parents=True, exist_ok=True)
    (out_raw / "labels").mkdir(parents=True, exist_ok=True)
    kept = 0
    for idx, im in enumerate(chosen):
        parts = list(im.parts)
        lbl = None
        if "images" in parts:
            pos = len(parts) - 1 - parts[::-1].index("images")
            lbl_parts = parts[:]
            lbl_parts[pos] = "labels"
            cand = Path(*lbl_parts).with_suffix(".txt")
            if cand.is_file():
                lbl = cand
        if lbl is None:
            cand2 = im.with_suffix(".txt")
            if cand2.is_file():
                lbl = cand2
        if lbl is None or not lbl.read_text(encoding="utf-8", errors="replace").strip():
            continue  # Tuyệt đối không nạp ảnh README hoặc ảnh không nhãn!
        ustem = f"{idx:05d}_{im.stem}"
        transfer_image(im, out_raw / "images" / f"{ustem}{im.suffix.lower()}", mode="copy")
        shutil.copy2(lbl, out_raw / "labels" / f"{ustem}.txt")
        kept += 1
    return kept


if RUN_VNTS_HF:
    vnts_raw = TMP_RAW_DIR / "vnts_hf"
    vnts_sub = TMP_RAW_DIR / "vnts_subset"
    vnts_val_sub = TMP_RAW_DIR / "vnts_val_subset"
    for d in (vnts_raw, vnts_sub, vnts_val_sub):
        shutil.rmtree(d, ignore_errors=True)

    print(f"Đang tải {VNTS_HF_REPO}...")
    snapshot_download(repo_id=VNTS_HF_REPO, repo_type="dataset", local_dir=str(vnts_raw))
    extracted = unzip_archives_in_dir(vnts_raw, remove_zip=True)
    if extracted:
        print(f"Đã giải nén {len(extracted)} file zip trong {VNTS_HF_REPO}.")

    all_imgs = sorted(
        p for p in vnts_raw.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
        and not any(bad in p.name.lower() for bad in ("illustration", "class_distribution", "readme"))
    )
    train_imgs = [p for p in all_imgs if "train" in p.as_posix().lower()]
    val_imgs = [p for p in all_imgs if any(k in p.as_posix().lower() for k in ("/val/", "/valid/"))]
    if not train_imgs and all_imgs:
        train_imgs = [p for p in all_imgs if p not in set(val_imgs)]

    if train_imgs:
        n_tr = _stage_yolo_bbox_folder(train_imgs, vnts_sub, VNTS_MAX_IMAGES)
        n_vl = _stage_yolo_bbox_folder(val_imgs, vnts_val_sub, VAL_MAX_IMAGES_PER_SOURCE) if val_imgs else 0
    elif list(vnts_raw.rglob("*.parquet")):
        print("Repo HF dùng Parquet -> giải xuất qua datasets.load_dataset...")
        shutil.rmtree(vnts_raw, ignore_errors=True)
        export_hf_parquet_to_yolo_bbox(VNTS_HF_REPO, vnts_sub, max_images=VNTS_MAX_IMAGES, seed=TEAM_ID, default_class=9)
        n_tr = len(list((vnts_sub / "images").glob("*")))
        n_vl = 0
    else:
        n_tr = n_vl = 0

    vnts_out = STAGED_DIR / "vnts_sam2"
    shutil.rmtree(vnts_out, ignore_errors=True)
    if n_tr > 0:
        print(f"Chạy SAM 2 trên {n_tr} ảnh biển báo Việt Nam (train)...")
        vnts_summary = convert_bbox_dataset(
            images_dir=vnts_sub / "images",
            labels_dir=vnts_sub / "labels",
            out_dir=vnts_out,
            predictor=SAM_PREDICTOR,
            split="train",
            default_class=9,
            registry_path=REPO_DIR / "datasets_registry.md",
            source_name="star092304_vn_signs",
            dataset_url=f"https://huggingface.co/datasets/{VNTS_HF_REPO}",
            license_name="CC BY 4.0",
        )
        print("Kết quả VNTS -> SAM2 (Train):", json.dumps(vnts_summary, indent=2, ensure_ascii=False))
        if vnts_summary["instances_written"] > 0:
            STAGED_SOURCES.append(vnts_out)
            CLEAN_TRAIN_SOURCES.append(vnts_out)

    if n_vl > 0:
        vnts_val_out = STAGED_DIR / "_val_vnts_sam2"
        shutil.rmtree(vnts_val_out, ignore_errors=True)
        vnts_val_sum = convert_bbox_dataset(
            images_dir=vnts_val_sub / "images",
            labels_dir=vnts_val_sub / "labels",
            out_dir=vnts_val_out,
            predictor=SAM_PREDICTOR,
            split="val",
            default_class=9,
            registry_path=None,
            source_name="star092304_vn_signs_val",
        )
        if vnts_val_sum["instances_written"] > 0:
            CLEAN_VAL_SOURCES.append(vnts_val_out)

    for d in (vnts_raw, vnts_sub, vnts_val_sub):
        shutil.rmtree(d, ignore_errors=True)"""
        ),
        _code_cell(
            """#@title 5. Luồng 2: Tải-Chuyển Đổi-Xóa Cache Tuần Tự Từng Kaggle Dataset (Tiết Kiệm Đĩa + Đủ 11 Lớp)
import kagglehub
from src.kaggle_adapter import auto_convert_detected_dataset, detect_val_split, unzip_archives_in_dir


def _run_one_kaggle_source(idx_tag: str, info: dict) -> None:
    safe_name = info["name"].replace("-", "_").replace(" ", "_").lower()
    out_d = STAGED_DIR / f"kaggle_{idx_tag}_{safe_name}"
    shutil.rmtree(out_d, ignore_errors=True)
    cap = COCO_MAX_IMAGES if "coco" in safe_name else KAGGLE_MAX_IMAGES_PER_SOURCE
    print(f"Đang xử lý [{info['format']}] {info['name']} (cap={cap})...", flush=True)
    res = auto_convert_detected_dataset(
        info=info,
        out_dir=out_d,
        sam_predictor=SAM_PREDICTOR,
        exclude_dir=GOLDEN_VAL_DIR if GOLDEN_VAL_DIR.exists() else None,
        max_images=cap,
        seed=TEAM_ID,
        split="train",
        registry_path=REPO_DIR / "datasets_registry.md",
    )
    print("  -> Train:", json.dumps(res, indent=2, ensure_ascii=False))
    if (out_d / "images" / "train").exists() and any((out_d / "images" / "train").glob("*")):
        STAGED_SOURCES.append(out_d)

    trusted = any(slug.split("/")[-1].replace("-", "_") in safe_name for slug in AUTO_DOWNLOAD_KAGGLE_SLUGS)
    if trusted and out_d in STAGED_SOURCES:
        CLEAN_TRAIN_SOURCES.append(out_d)
    val_info = detect_val_split(info) if trusted else None
    if val_info is not None:
        val_out = STAGED_DIR / f"_val_{idx_tag}_{safe_name}"
        shutil.rmtree(val_out, ignore_errors=True)
        v_res = auto_convert_detected_dataset(
            info=val_info,
            out_dir=val_out,
            sam_predictor=SAM_PREDICTOR,
            max_images=VAL_MAX_IMAGES_PER_SOURCE,
            seed=TEAM_ID,
            split="val",
            registry_path=None,
        )
        print("  -> Val gốc sạch:", json.dumps(v_res, indent=2, ensure_ascii=False))
        if (val_out / "images" / "val").exists() and any((val_out / "images" / "val").glob("*")):
            CLEAN_VAL_SOURCES.append(val_out)


for idx, info in enumerate(DISCOVERED_KAGGLE):
    try:
        _run_one_kaggle_source(f"mnt{idx:02d}", info)
    except Exception as exc:
        print(f"  [WARN] Bỏ qua {info.get('name')}: {exc}")

for idx, slug in enumerate(KAGGLEHUB_QUEUE):
    print(f"\\n[KAGGLEHUB {idx + 1}/{len(KAGGLEHUB_QUEUE)}] Đang tải https://www.kaggle.com/datasets/{slug} ...")
    try:
        dl_dir = Path(kagglehub.dataset_download(slug))
        unzip_archives_in_dir(dl_dir, remove_zip=True)
        info = detect_dataset_format(dl_dir)
        if info is None:
            print(f"  [WARN] Không nhận diện được cấu trúc dữ liệu tại {dl_dir}")
        else:
            info["name"] = slug.replace("/", "__")
            info["url"] = f"https://www.kaggle.com/datasets/{slug}"
            _run_one_kaggle_source(f"dl{idx:02d}", info)
    except Exception as exc:
        print(f"  [WARN] Lỗi tải/xử lý {slug}: {exc}")
    finally:
        # Xóa ngay cache thô sau mỗi bộ để giải phóng hàng chục GB cho bộ tiếp theo
        shutil.rmtree(TMP_RAW_DIR / "kh_cache", ignore_errors=True)
        shutil.rmtree(Path.home() / ".cache" / "kagglehub", ignore_errors=True)"""
        ),
        _code_cell(
            """#@title 6. Luồng 3 (Tùy chọn): Dashcam Việt Nam + Lọc Nắp Capo & Chồng Lấn Rider
import zipfile
from src.prepare_vn_frames import UltralyticsTeacherPredictor, extract_frames_from_sources, pseudo_label_folder

if RUN_VN_DASHCAM:
    vn_src = Path(VN_DASHCAM_LOCAL_INPUT) if VN_DASHCAM_LOCAL_INPUT else (TMP_RAW_DIR / "vn_dashcam")
    if not VN_DASHCAM_LOCAL_INPUT:
        shutil.rmtree(vn_src, ignore_errors=True)
        print(f"Đang tải dashcam VN {VN_DASHCAM_HF_REPO}...")
        snapshot_download(repo_id=VN_DASHCAM_HF_REPO, repo_type="dataset", local_dir=str(vn_src))
        for zp in sorted(vn_src.rglob("*.zip")):
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(vn_src / zp.stem)

    vn_staged = STAGED_DIR / "vn_dashcam_pseudo"
    shutil.rmtree(vn_staged, ignore_errors=True)
    ext_res = extract_frames_from_sources(
        input_path=vn_src,
        out_img_dir=vn_staged / "images" / "train",
        exclude_dir=GOLDEN_VAL_DIR if GOLDEN_VAL_DIR.exists() else None,
        stride_sec=1.5,
        min_sharpness=60.0,
        max_frames=DASHCAM_MAX_FRAMES,
    )
    print("Trích frame VN:", json.dumps({k: v for k, v in ext_res.items() if k != "image_paths"}, indent=2))
    if ext_res["frames_kept"] > 0:
        teacher = UltralyticsTeacherPredictor(weights=TEACHER_WEIGHTS, conf=0.25, imgsz=1280, device=DEVICE)
        pl_res = pseudo_label_folder(
            images_dir=vn_staged / "images" / "train",
            labels_out_dir=vn_staged / "labels" / "train",
            predictor=teacher,
            source_mapping_name="coco",
            filter_ego_hood=True,
            filter_rider_ped_overlap=True,
            registry_path=REPO_DIR / "datasets_registry.md",
            dataset_name="vn_dashcam_openhay",
            dataset_url=f"https://huggingface.co/datasets/{VN_DASHCAM_HF_REPO}",
            license_name="Open-source / Research",
        )
        print("Pseudo-label VN (đã lọc nắp capo + Merger V2):", json.dumps(pl_res, indent=2, ensure_ascii=False))
        STAGED_SOURCES.append(vn_staged)
    if not VN_DASHCAM_LOCAL_INPUT:
        shutil.rmtree(vn_src, ignore_errors=True)
else:
    print("Bỏ qua vn_dashcam_openhay (RUN_VN_DASHCAM = False) để giữ ảnh đường phố sắc nét chuẩn nhãn người.")"""
        ),
        _code_cell(
            """#@title 7. Khóa Golden Val: quota theo mẫu được chấm + manifest nguồn + chống rò trước augment
from src.data_gates import build_golden_holdout, exclude_golden_from_staged_train

shutil.rmtree(GOLDEN_VAL_DIR, ignore_errors=True)
holdout = build_golden_holdout(
    CLEAN_VAL_SOURCES, CLEAN_TRAIN_SOURCES, GOLDEN_VAL_DIR,
    min_per_class=GOLDEN_MIN_PER_CLASS, target_images=AUTO_GOLDEN_VAL_TARGET, seed=TEAM_ID,
)
print(json.dumps({k: v for k, v in holdout.items() if k != "manifest"}, indent=2, ensure_ascii=False))
shutil.copy2(GOLDEN_VAL_DIR / "holdout_manifest.json", OUTPUTS_DIR / "holdout_manifest.json")
if not holdout["ok"]:
    raise RuntimeError(f"Golden Val thiếu mẫu được chấm; bổ sung nguồn nhãn người trước train: {holdout['missing_instances']}")
# Chỉ xóa bản staged dùng tạm, không sửa dữ liệu upstream. Loại cả bản trùng ở nguồn khác trước khi augment.
print("Removed staged train/val overlap:", exclude_golden_from_staged_train(STAGED_SOURCES, GOLDEN_VAL_DIR))"""
        ),
        _code_cell(
            """#@title 8. Tăng cường Data-Centric (Crop-Zoom Biển Báo + Copy-Paste Lớp Hiếm + Weather Trên Nguồn Sạch)
from src.augment_and_crop import generate_copypaste_dataset, generate_crop_zoom_dataset, generate_weather_dataset

base_sources = [s for s in STAGED_SOURCES if not s.name.startswith("aug_")]
sign_sources = [s for s in base_sources if any(k in s.name for k in ("vnts", "traffic_sign", "lvnduy"))]

if RUN_CROP_ZOOM and sign_sources:
    for idx_cz, s_dir in enumerate(sign_sources):
        cz_out = STAGED_DIR / f"aug_cropzoom_{idx_cz:02d}"
        shutil.rmtree(cz_out, ignore_errors=True)
        cz_res = generate_crop_zoom_dataset(s_dir, cz_out, split="train", crop_size=640, max_crops_per_image=2, seed=TEAM_ID + idx_cz)
        print(f"Crop-Zoom biển báo ({s_dir.name}):", cz_res)
        if cz_res["crops_written"] > 0:
            STAGED_SOURCES.append(cz_out)

if RUN_COPY_PASTE and base_sources:
    bank_sources = [s for s in base_sources if any(k in s.name for k in ("indian", "idd", "coco", "vnts", "lvnduy", "traffic_sign"))]
    bg_src = next((s for s in base_sources if any(k in s.name for k in ("bdd100k", "indian", "idd", "obstacle"))), None)
    if not bank_sources or bg_src is None:
        raise RuntimeError("Copy-Paste cần ngân hàng lớp hiếm và nền đường phố đúng nguồn")
    cp_out = STAGED_DIR / "aug_copypaste"
    shutil.rmtree(cp_out, ignore_errors=True)
    cp_res = generate_copypaste_dataset(bank_sources, bg_src, cp_out, split="train", rare_classes=(3, 6, 7, 9, 10), n_output=COPY_PASTE_COUNT, seed=TEAM_ID)
    print(f"Copy-Paste lớp hiếm (banks={[s.name for s in bank_sources]}, bg={bg_src.name}):", cp_res)
    if cp_res["images_written"] > 0:
        STAGED_SOURCES.append(cp_out)

if RUN_WEATHER_AUG and base_sources:
    w_src = next((s for s in base_sources if "bdd100k" in s.name or "indian" in s.name or "coco" in s.name), base_sources[0])
    we_out = STAGED_DIR / "aug_weather"
    shutil.rmtree(we_out, ignore_errors=True)
    we_res = generate_weather_dataset(w_src, we_out, split="train", max_images=WEATHER_COUNT, seed=TEAM_ID)
    print(f"Weather Augmentation (src={w_src.name}):", we_res)
    if we_res["images_written"] > 0:
        STAGED_SOURCES.append(we_out)"""
        ),
        _code_cell(
            """#@title 9. Gộp Master V3, QC Tầng 1 & Sanity Gate Chống Sai Nhãn Trước Khi Train
from IPython.display import Image as IPyImage, display
from src.build_master import build_master_dataset
from src.eval_golden import render_per_class_preview_grid, run_pretrain_semantic_sanity_check
from src.validate_dataset import load_competition_config, print_qc_summary, run_qc

shutil.rmtree(MASTER_DIR, ignore_errors=True)
master_summary = build_master_dataset(
    train_sources=STAGED_SOURCES,
    golden_val_dir=GOLDEN_VAL_DIR,
    master_dir=MASTER_DIR,
    zip_out=OUT_ZIP_PATH,
)
print("Kết quả Build Master V3:", json.dumps(master_summary, indent=2, ensure_ascii=False))

cfg = load_competition_config(REPO_DIR / "configs/competition.yaml")
qc_rep = run_qc(MASTER_DIR, cfg, out_dir=OUTPUTS_DIR / "qc_master")
print_qc_summary(qc_rep, cfg)

preview_grid_p = OUTPUTS_DIR / "preview_per_class.jpg"
render_per_class_preview_grid(MASTER_DIR, preview_grid_p, split="train", samples_per_class=4, seed=TEAM_ID)
if preview_grid_p.is_file():
    if IS_KAGGLE:
        shutil.copy2(preview_grid_p, Path("/kaggle/working/preview_per_class.jpg"))
    display(IPyImage(filename=str(preview_grid_p)))

# Coverage and Tier-1 QC are mandatory, independent of semantic gate flags.
if not qc_rep["ok"]:
    raise RuntimeError("Tier-1 QC failed; inspect qc_master reports before training")
rare_shortfalls = {
    cfg["classes"][c]: qc_rep["per_class"][cfg["classes"][c]]["by_split"].get("train", 0)
    for c in (6, 7)
    if qc_rep["per_class"][cfg["classes"][c]]["by_split"].get("train", 0) < MIN_RARE_TRAIN_INSTANCES
}
if rare_shortfalls:
    raise RuntimeError(f"Train thiếu Bicycle/Cyclist (minimum={MIN_RARE_TRAIN_INSTANCES}): {rare_shortfalls}")

if RUN_SANITY_GATE:
    source_sanity = {}
    for source in STAGED_SOURCES:
        if source.name.startswith("aug_"):
            continue
        sanity = run_pretrain_semantic_sanity_check(
            source, split="train", sample_images=SANITY_IMAGES_PER_SOURCE, seed=TEAM_ID, weights="yolo26n-seg.pt")
        source_sanity[source.name] = sanity
        print(source.name, json.dumps(sanity, indent=2, ensure_ascii=False))
    (OUTPUTS_DIR / "source_semantic_sanity.json").write_text(json.dumps(source_sanity, indent=2), encoding="utf-8")
    blocked = {name: result for name, result in source_sanity.items() if result["ok"] is False}
    if blocked and FAIL_ON_SANITY_ALERT:
        raise RuntimeError(f"Sanity Gate failed or lacked evidence for sources: {list(blocked)}")

shutil.copy2(REPO_DIR / "datasets_registry.md", OUTPUTS_DIR / "datasets_registry.md")
print(f"\\n[XONG DATA V3] {OUT_ZIP_PATH} ({OUT_ZIP_PATH.stat().st_size / 2**20:.1f} MiB)")"""
        ),
        _code_cell(
            """#@title 10. Huấn luyện YOLO26n-seg & Xuất submission.zip chuẩn BTC
from ultralytics import YOLO
from src.export_verify_onnx import export_and_verify

BEST_PT_PATH = None
SUBMISSION_ZIP_PATH = None

if RUN_TRAIN_IN_THIS_NOTEBOOK:
    run_proj = Path("/kaggle/working/runs") if IS_KAGGLE else Path("work_kaggle/runs")
    model = YOLO("yolo26n-seg.pt")
    model.train(
        data=str(MASTER_DIR / "data.yaml"),
        imgsz=640,
        epochs=TRAIN_EPOCHS,
        batch=TRAIN_BATCH,
        patience=TRAIN_PATIENCE,
        seed=TEAM_ID,
        deterministic=True,
        device=DEVICE,
        workers=2,
        cache=False,
        optimizer="AdamW",
        lr0=0.002,
        cos_lr=True,
        scale=0.5,
        mosaic=1.0,
        close_mosaic=10,
        copy_paste=0.10,
        project=str(run_proj),
        name=RUN_NAME,
        exist_ok=True,
    )
    BEST_PT_PATH = Path(model.trainer.best)
    shutil.copy2(BEST_PT_PATH, OUTPUTS_DIR / "best.pt")

    exp_res = export_and_verify(
        weights=BEST_PT_PATH,
        out_dir=OUTPUTS_DIR,
        notebook_path=REPO_DIR / "train_and_export.ipynb",
        smoke_images_dir=MASTER_DIR / "images" / "val",
    )
    SUBMISSION_ZIP_PATH = Path(exp_res["zip"])
    print(f"\\n[SAN SANG NOP BAI] {SUBMISSION_ZIP_PATH} ({SUBMISSION_ZIP_PATH.stat().st_size / 2**20:.2f} MiB)")"""
        ),
        _code_cell(
            """#@title 11. Xuất Báo cáo So sánh, Confident QC & Gói quick_local_pack.zip
QUICK_PACK_PATH = Path("/kaggle/working/quick_local_pack.zip") if IS_KAGGLE else Path("work_kaggle/quick_local_pack.zip")
sub_p = Path(SUBMISSION_ZIP_PATH) if SUBMISSION_ZIP_PATH else (OUTPUTS_DIR / "submission.zip")
pt_p = Path(BEST_PT_PATH) if BEST_PT_PATH else (OUTPUTS_DIR / "best.pt")
train_run_p = pt_p.parent.parent if (pt_p.is_file() and pt_p.parent.name == "weights") else None

if RUN_BENCHMARK_AND_QC and sub_p.is_file() and (MASTER_DIR / "images").is_dir():
    from src.benchmark_report import run_benchmark
    from src.eval_golden import run_eval_or_qc
    from src.selfcheck_bridge import materialize_selfcheck

    bench_dir = OUTPUTS_DIR / "benchmark" / RUN_NAME
    rep = run_benchmark(
        submission_or_onnx=sub_p,
        dataset_dir=MASTER_DIR,
        out_dir=bench_dir,
        run_name=RUN_NAME,
        weights_pt=pt_p if pt_p.is_file() else None,
        baseline_json=Path(BASELINE_SUMMARY_JSON) if BASELINE_SUMMARY_JSON else None,
        train_run_dir=train_run_p,
        notebook_path=REPO_DIR / "train_and_export.ipynb",
    )
    shutil.copy2(bench_dir / "benchmark_report.md", OUTPUTS_DIR / "benchmark_report.md")
    shutil.copy2(bench_dir / "benchmark_summary.json", OUTPUTS_DIR / "benchmark_summary.json")
    for img_art in list(bench_dir.glob("*.png")) + list(bench_dir.glob("*.jpg")):
        shutil.copy2(img_art, OUTPUTS_DIR / img_art.name)

    sc = materialize_selfcheck(REPO_DIR / "train_and_export.ipynb", Path("/kaggle/working/_selfcheck_eval") if IS_KAGGLE else Path("work_kaggle/_selfcheck_eval"))
    zip_s = sc.package.inspect_zip(str(sub_p), sc.config)
    onnx_p = sc.package.extract_model(str(sub_p), OUTPUTS_DIR / "_tmp_extract", sc.config, zip_s)
    qc_rep = run_eval_or_qc(
        model_path=onnx_p,
        images_dir=MASTER_DIR / "images" / "train",
        labels_dir=MASTER_DIR / "labels" / "train",
        out_dir=bench_dir / "qc_confident_train",
        mode="qc-train",
        notebook_path=REPO_DIR / "train_and_export.ipynb",
        max_images=QC_TRAIN_SAMPLE_IMAGES,
        seed=TEAM_ID,
    )
    shutil.copy2(bench_dir / "qc_confident_train" / "confident_qc_report.json", OUTPUTS_DIR / "confident_qc_report.json")
    print("Confident QC suspects:", {k: len(v) for k, v in qc_rep["confident_qc"].items()})

if sub_p.is_file() and (MASTER_DIR / "images").is_dir():
    from src.benchmark_report import build_quick_local_pack

    extra_docs = [
        p for p in [
            OUTPUTS_DIR / "benchmark_report.md",
            OUTPUTS_DIR / "benchmark_summary.json",
            OUTPUTS_DIR / "datasets_registry.md",
            OUTPUTS_DIR / "confident_qc_report.json",
            OUTPUTS_DIR / "preview_per_class.jpg",
            OUTPUTS_DIR / "holdout_manifest.json",
            OUTPUTS_DIR / "source_semantic_sanity.json",
            Path(BASELINE_SUMMARY_JSON) if BASELINE_SUMMARY_JSON else None,
        ]
        if p and Path(p).is_file()
    ]
    build_quick_local_pack(
        out_zip=QUICK_PACK_PATH,
        submission_or_onnx=sub_p,
        dataset_dir=MASTER_DIR,
        weights_pt=pt_p if pt_p.is_file() else None,
        train_run_dir=train_run_p,
        extra_files=extra_docs,
        competition_path=REPO_DIR / "configs/competition.yaml",
        max_val_images=LOCAL_PACK_MAX_VAL_IMAGES,
        seed=TEAM_ID,
    )
    if IS_KAGGLE:
        shutil.copy2(sub_p, Path("/kaggle/working/submission.zip"))
        if (OUTPUTS_DIR / "benchmark_report.md").is_file():
            shutil.copy2(OUTPUTS_DIR / "benchmark_report.md", Path("/kaggle/working/benchmark_report.md"))
    print(f"[XONG PACK TEST LAPTOP] {QUICK_PACK_PATH} ({QUICK_PACK_PATH.stat().st_size / 2**20:.2f} MiB)")"""
        ),
        _code_cell(
            """#@title 12. Tải File Tự Động (Nút bấm tải thẳng + quick_local_pack.zip + CLI nhanh)
from IPython.display import FileLink, HTML, display

if IS_KAGGLE:
    os.chdir("/kaggle/working")

def split_file_for_download(src_path: Path, out_sub: str, chunk_mb: int) -> list[Path]:
    chunk_bytes = chunk_mb * 1024 * 1024
    parts: list[Path] = []
    if not src_path.is_file() or src_path.stat().st_size <= chunk_bytes:
        return [src_path] if src_path.is_file() else []
    part_dir = OUTPUTS_DIR / out_sub
    shutil.rmtree(part_dir, ignore_errors=True)
    part_dir.mkdir(parents=True, exist_ok=True)
    with open(src_path, "rb") as fh:
        idx = 0
        while True:
            buf = fh.read(chunk_bytes)
            if not buf:
                break
            p = part_dir / f"{src_path.name}.part_{idx:03d}"
            p.write_bytes(buf)
            parts.append(p)
            idx += 1
    return parts

_INLINE_BUDGET_BYTES = [24 * 1024 * 1024]

def make_direct_download_button(path: Path, max_inline_mb: float = INLINE_MAX_MB) -> None:
    size_b = path.stat().st_size
    size_mb = size_b / (1024 * 1024)
    rel_p = os.path.relpath(path.resolve(), Path.cwd().resolve()).replace(os.sep, "/")
    if size_mb <= max_inline_mb and _INLINE_BUDGET_BYTES[0] >= size_b:
        _INLINE_BUDGET_BYTES[0] -= size_b
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        html = (
            f'<div style="margin:6px 0;padding:8px 12px;border:1px solid #d0d7de;border-radius:6px;background:#f6f8fa;">'
            f'<b>{path.name}</b> ({size_mb:.2f} MiB) &nbsp; '
            f'<a download="{path.name}" href="data:application/octet-stream;base64,{b64}" '
            f'style="display:inline-block;padding:4px 12px;background:#1f6feb;color:#fff;border-radius:4px;text-decoration:none;font-weight:600;">'
            f'Tải trực tiếp</a></div>'
        )
        display(HTML(html))
    else:
        print(f"  [{size_mb:6.2f} MiB] {path.name}:")
        display(FileLink(rel_p, result_html_prefix="👉 Bấm để tải: "))

print("=== 1. FILE NỘP BÀI & BÁO CÁO NGAY (Nhấn nút xanh để tải thẳng) ===")
priority_targets = [
    OUTPUTS_DIR / "submission.zip",
    OUTPUTS_DIR / "benchmark_report.md",
    OUTPUTS_DIR / "preview_per_class.jpg",
    OUTPUTS_DIR / "benchmark_summary.json",
    OUTPUTS_DIR / "holdout_manifest.json",
    OUTPUTS_DIR / "source_semantic_sanity.json",
    OUTPUTS_DIR / "best.pt",
    OUTPUTS_DIR / "datasets_registry.md",
    OUTPUTS_DIR / "confident_qc_report.json",
    OUTPUTS_DIR / "per_class_ap_chart.png",
    OUTPUTS_DIR / "conf_sweep_chart.png",
    OUTPUTS_DIR / "results.png",
    OUTPUTS_DIR / "MaskF1_curve.png",
    OUTPUTS_DIR / "MaskPR_curve.png",
]
for f in priority_targets:
    if f.is_file():
        make_direct_download_button(f)

print("\\n=== 2. GÓI TEST MODEL TRÊN LAPTOP ===")
quick_pack = Path("/kaggle/working/quick_local_pack.zip") if IS_KAGGLE else Path("work_kaggle/quick_local_pack.zip")
if quick_pack.is_file():
    make_direct_download_button(quick_pack)
    if SPLIT_PACK_FOR_DOWNLOAD:
        for part_p in split_file_for_download(quick_pack, "pack_parts", chunk_mb=10):
            make_direct_download_button(part_p, max_inline_mb=10.5)

if SPLIT_ZIP_FOR_DOWNLOAD and OUT_ZIP_PATH.is_file():
    print("\\n=== 3. TẢI TOÀN BỘ DATASET (Cắt mảnh 95 MiB) ===")
    for part_p in split_file_for_download(OUT_ZIP_PATH, "zip_parts", chunk_mb=95):
        make_direct_download_button(part_p, max_inline_mb=0.0)

print("\\n=== LỆNH CHẠY THỬ TRÊN LAPTOP ===")
print(f"  python -m src.benchmark_report --unpack-pack quick_local_pack.zip --out-dir work/local_eval --run-name {RUN_NAME}")"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def verify_notebook(nb: dict) -> None:
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        ast.parse(src, filename=f"cell_{idx}")

    cell2_src = "".join(nb["cells"][2]["source"])
    ns: dict = {}
    exec(compile(cell2_src.split("\nfor rel_path, content")[0], "cell_2_bundle", "exec"), ns)
    data = ns["EMBEDDED_FILES"]
    assert set(data.keys()) == set(EMBED_REL_PATHS)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for rel, text in data.items():
            target = tmp_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            if target.suffix == ".py":
                py_compile.compile(str(target), doraise=True)


def main() -> None:
    nb = build_notebook()
    verify_notebook(nb)
    OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Successfully generated & verified V3.3 {OUT_NB} ({OUT_NB.stat().st_size / 1024:.1f} KiB, {len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()

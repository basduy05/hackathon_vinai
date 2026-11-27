# Kế Hoạch Khắc Phục Toàn Diện V3: Giải Mã Lỗi 5.03 Điểm Public LB & Tái Cấu Trúc Pipeline Data-Centric

## 1. Bối Cảnh & Tóm Tắt Chẩn Đoán (Vì sao chỉ được `5.03` trên hệ thống BTC trong khi tự chấm Val là `38.50`?)

Bằng chứng kiểm chứng trực tiếp từ `work/kaggle_v2_out/outputs/` và chấm chéo trên `sample_data` xác nhận **không có lỗi format ONNX hay lệch layout**: cùng mô hình `submission.zip` đó khi chấm trên 10 ảnh sạch `sample_data` đạt **`5.48`** (`AP50 = 0.0811`) — khớp gần như nguyên vẹn với điểm Public LB **`5.03`** (`mAP@50 = 9.37`) của BTC. Lý do điểm Val nội bộ cao giả tạo (`38.50`) nhưng điểm test thật rơi về `5.03` nằm ở **4 lỗi nghiêm trọng đồng thời trong chuỗi dữ liệu V2**:

1. **Lỗi #1 (Chí mạng — 63.5% tập train bị đảo lộn ID lớp!)**:
   - Bộ `nadinpethiyagoda/vehicle-dataset-for-yolo` (`1,953` ảnh / `3,072` ảnh train) dùng file `vehicle dataset/classes.txt` (`car, threewheel, bus, truck, motorbike, van`), **không có `data.yaml`**.
   - Trong [kaggle_adapter.py](file:///c:/Users/basduy05/Downloads/VINAI/hackathon/src/kaggle_adapter.py#L241-L245), khi không thấy `data.yaml`, code tự gán **bảng ánh xạ đồng nhất `{0:0, 1:1, ..., 10:10}`** (thể hiện rõ trong [datasets_registry.md](file:///c:/Users/basduy05/Downloads/VINAI/hackathon/work/kaggle_v2_out/outputs/datasets_registry.md#L14)). Hệ quả trực tiếp nhìn thấy trong ảnh `val_batch0_labels.jpg`:
     - `1: threewheel` bị dạy thành **`1: Bus`** → `Bus` trên BTC còn **`3.50`**.
     - `2: bus` bị dạy thành **`2: Truck`** → `Truck` trên BTC còn **`6.83`**.
     - `3: truck` bị dạy thành **`3: Other vehicle`** → tự chấm Val `Other vehicle = 90.49` giả tạo, ra BTC còn **`0.40`**.
     - `5: van` bị dạy thành **`5: Motorcyclist`** → hàng loạt xe 16 chỗ / minivan được gán nhãn hồng `Motorcyclist`, ra BTC còn **`4.78`**.
   - Cộng thêm `aug_copypaste` (`350` ảnh) tiếp tục cắt chính các hình `Truck` (bị gán nhầm là `Other vehicle`) và `threewheel` (nhầm là `Bus`) dán lên cảnh đường phố.

2. **Lỗi #2 (Chí mạng — Toàn bộ nguồn dữ liệu đường phố chuẩn COCO / BDD100K / IDD / Biển báo VN đều nạp 0 ảnh!)**:
   - Ba slug Kaggle trong V2 (`theainerd/indian-driving-dataset`, `solarmainframe/bdd100k`, `giddyyupp/micro-coco2017`) đều trả về **`403 Forbidden`** qua Kaggle API → **0 ảnh** được nạp.
   - Bộ HuggingFace `star092304/Traffic-sign-detection-VietNam` chứa toàn bộ dữ liệu bên trong file nén `traffic-sign-detection-vietnam.zip`, nhưng Cell 4 chỉ thấy 2 ảnh minh họa README (`class_distribution.png`, `illustration.webp`), **không giải nén `.zip`**, tạo ra đúng **2 ảnh nhãn rỗng `{}`** và đẩy cả 2 vào Golden Val → **0 ảnh biển báo thật trong Train**, `Sign` trên BTC còn **`1.81`**, `Crop-Zoom` tạo **0 ảnh**.
   - Do không có COCO/IDD/BDD100K, hai lớp **`Bicycle`** (`88` train inst) và **`Cyclist`** (`41` train inst) cực kỳ thiếu mẫu → trên BTC còn **`0.01`** và **`2.74`**.

3. **Lỗi #3 (Nhiễu domain & False Negative từ `vn_dashcam_openhay`)**:
   - Kiểm tra trực quan 4 khung hình `vn_dashcam` trong `quick_local_pack.zip` cho thấy bộ `OpenHay/zalo-ai-2025-public-test-data-v2` bị **làm mờ bảo mật toàn bộ thân xe và người (`privacy gaussian blur`)**, đồng thời nhãn giả từ `yolo11x-seg` (COCO) mắc 3 lỗi:
     1. Gán nhầm **nắp capo / bảng táp-lô xe gắn camera** ở cạnh dưới ảnh thành **`Car`**.
     2. COCO chỉ có `stop sign`, nên ~90% biển báo VN (cấm, tốc độ, chỉ dẫn xanh) bị bỏ trống làm nền âm → dạy model **bỏ qua biển báo**.
     3. Các `Pedestrian` / xe máy ở xa (`< 40 px²` hoặc mất mask xe máy dưới chân) bị gán sai thành `Pedestrian` hoặc bị bỏ sót làm nền âm → `Pedestrian` chỉ đạt `0.0953` trên Val và **`0.92`** trên BTC.

4. **Lỗi #4 (Golden Val bị nhiễm bẩn → số Val ảo `38.50`)**:
   - Cell 7 rút `Golden Val` từ các nguồn đã tạo (bao gồm ảnh xe tải gắn nhãn `Other vehicle`, xe van gắn nhãn `Motorcyclist`, `2` ảnh README rỗng, và `41` frame làm mờ `vn_dashcam`). Golden Val vì thế tự đánh giá sai lệch hoàn toàn so với bộ test sạch của BTC.

![Đối chiếu nhãn GT vs Dự đoán V2 trên Zalo Dashcam — dễ thấy nắp capo góc dưới bị gán nhãn Car và biển báo VN không được gán nhãn ở GT](file:///C:/Users/basduy05/.gemini/antigravity/brain/69f122bd-9237-4239-b2b8-883b0f875221/scratch/inspect_val_imgs/vis_vn_dashcam_pseudo_vn_dashcam_175ba98d_1000_clip_002_0008_0018_Y_frame_0.jpg)

![Đối chiếu nhãn GT vs Dự đoán V2 — Pedestrian dự đoán chồng lên Motorcyclist và các xe bị làm mờ Gaussian trong OpenHay](file:///C:/Users/basduy05/.gemini/antigravity/brain/69f122bd-9237-4239-b2b8-883b0f875221/scratch/inspect_val_imgs/vis_vn_dashcam_pseudo_vn_dashcam_175ba98d_1000_clip_002_0008_0018_Y_frame_4.jpg)

---

## User Review Required

> [!IMPORTANT]
> **Quyết định thiết kế cốt lõi cần bạn duyệt trước khi sửa code (V3)**:
> 1. **Loại bỏ 1,953 ảnh studio/close-up từ `nadinpethiyagoda/vehicle-dataset-for-yolo` khỏi tập Train/Val trực tiếp** (chỉ giữ tùy chọn trích lọc riêng lớp `threewheel` ở tỉ lệ nhỏ hoặc ghép `Copy-Paste` đúng tỉ lệ đường phố), đồng thời tắt mặc định nguồn làm mờ nặng `vn_dashcam_openhay` (hoặc giới hạn tối đa `250` ảnh đã lọc nắp capo + ghép nhãn biển báo) để tránh kéo tụt nét biên ảnh.
> 2. **Thay toàn bộ 3 Kaggle slug lỗi 403 bằng 4–5 slug đã kiểm chứng tải `200 OK` qua Kaggle API**, kết hợp giải nén đúng `.zip` trên HuggingFace.
> 3. **Dùng chính kiến trúc 11 lớp sạch cho Golden Val** (`>= 20–30` GT/lớp chỉ lấy từ nguồn nhãn người chuẩn xác như COCO 2017 + BDD100K + IDD + VNTS, tuyệt đối không lấy từ pseudo-label hoặc stock car).
> 4. **Quản lý dung lượng đĩa Kaggle (`/kaggle/tmp/kh_cache`)**: Tải tuần tự từng bộ Kaggle nặng (như `coco-2017-dataset` ~25 GB, `idd` ~22 GB) → trích lọc ~1,000–1,500 ảnh chuẩn ra `/kaggle/working/staged` → **xóa ngay thư mục thô** trước khi tải bộ kế tiếp để không bị tràn ổ cứng.

---

## Proposed Changes

### Component 1: Sửa Triệt Để `kaggle_adapter.py` & `class_mapping.yaml` (Chống Ánh Xạ Nhầm Lớp)

#### [MODIFY] [kaggle_adapter.py](file:///c:/Users/basduy05/Downloads/VINAI/hackathon/src/kaggle_adapter.py)
- **Đọc được `classes.txt` / `classes_en.txt` / `obj.names` và Roboflow `data.yaml`**:
  - Trong `detect_dataset_format`, mở rộng nhánh (4) tìm thêm `classes.txt`, `obj.names`, `notes.json` (Roboflow) và ghi nhận vào descriptor (`classes_txt` / `data_yaml`).
  - Trong `discover_all_kaggle_inputs`, xử lý cấu trúc mount mới của Kaggle (`/kaggle/input/datasets/<owner>/<slug>/...` hoặc `/kaggle/input/<slug>/...`): nếu gặp thư mục bọc ngoài như `datasets/` hay `competitions/`, đi sâu 1–2 cấp vào đúng thư mục dataset thực thay vì trả về tên chung `"datasets"`.
- **Cấm fallback đồng nhất ngầm `{0:0, ..., 10:10}`**:
  - Trong `convert_yolo_dataset`, nếu `raw_names` rỗng VÀ không có `override_map` hoặc `default_class`, **ném `ValueError` / bỏ qua nguồn** thay vì ngầm ánh xạ `0..10 -> 0..10`.
  - Hỗ trợ `prefix_default_class` hoặc bảng alias cho các bộ nhiều lớp biển báo chi tiết (như 58 mã biển báo VN `P.*`, `R.*`, `W.*`, `DP.*`, `S.*` của `lvnduy/street-traffic-signs-in-vietnam-yolov8` hoặc `maitam/vietnamese-traffic-signs` → tự map hết về `9: Sign`).
- **Hỗ trợ trích xuất chọn lọc các lớp hiếm trong `convert_yolo_dataset` & `convert_public_source`**:
  - Thêm tham số `include_raw_classes` / `prioritize_classes` (ưu tiên giữ lại tất cả ảnh chứa lớp hiếm `3 Other vehicle`, `6 Bicycle`, `7 Cyclist`, `9 Sign`, `10 Traffic light` trước khi lấy mẫu `Car`) để không bị `Car` chiếm 80% ngân sách ảnh của COCO/BDD/IDD.
  - Tự động xử lý file `.zip` khi tải repo HuggingFace/Kaggle trong `export_hf_parquet_to_yolo_bbox` và bộ giải nén chung.

#### [MODIFY] [class_mapping.yaml](file:///c:/Users/basduy05/Downloads/VINAI/hackathon/configs/class_mapping.yaml)
- Bổ sung alias chuẩn xác từ các `data.yaml` đã kiểm chứng thực tế:
  - **`idd_yolo` / `kaggle_generic_aliases`**: `autorickshaw -> 3`, `caravan -> 3`, `trailer -> 3`, `vehicle fallback -> 3`, `motor -> 4`, `bike -> 6`, `rider -> 8` (`needs_rider_merge: true`), `traffic sign -> 9`, `traffic light -> 10`.
  - Bổ sung nhận diện mã biển báo VN (`dp.*`, `p.*`, `r.*`, `s.*`, `w.*`) hoặc ánh xạ `default_class=9` cho dataset chuyên biển báo.

---

### Component 2: Làm Sạch Luồng `prepare_vn_frames.py` (Lọc Nắp Capo & Chống Rò Nhãn)

#### [MODIFY] [prepare_vn_frames.py](file:///c:/Users/basduy05/Downloads/VINAI/hackathon/src/prepare_vn_frames.py)
- **Lọc tự động nắp capo / bảng táp-lô xe quay camera**: Thêm filter hình học trong `pseudo_label_folder`: bỏ mọi instance `Car (0)` có vùng chạm sát mép dưới khung hình (`y2 >= 0.94`, `x2 - x1 >= 0.32`, và `y_center >= 0.72` hoặc chiếm phần lớn chiều rộng đáy ảnh).
- **Hạ `min_area_640` trước bước ghép rider (`Geometric Mask Merger`) xuống `20.0 px²`** (sau khi ghép mới lọc `< 36.0 px²`) và cho phép truyền thêm một `aux_sign_predictor` (mô hình hoặc hộp gợi ý biển báo) để không biến biển báo VN thật thành nền âm.
- **Lọc nhiễu chồng chéo `Pedestrian` trên `Motorcyclist`**: Sau `apply_geometric_mask_merger`, kiểm tra nếu một mask `Pedestrian (8)` có tỷ lệ chồng lấn phần lớn (`>= 0.55` diện tích người) nằm bên trong một bbox/mask `Motorcyclist (5)` thì gộp hoặc loại trùng thay vì để song song cả `Motorcyclist` lẫn `Pedestrian`.

---

### Component 3: Sanity Gate Phát Hiện Sai Nhãn & Đánh Giá Sạch (`validate_dataset.py` + `benchmark_report.py`)

#### [MODIFY] [validate_dataset.py](file:///c:/Users/basduy05/Downloads/VINAI/hackathon/src/validate_dataset.py)
- Bổ sung báo cáo phân bố **diện tích chuẩn 640 (`<64`, `64–256`, `256–1024`, `>1024 px²`)** theo từng lớp trong Tier-1 QC để phát hiện ngay tình trạng tập train chỉ toàn xe close-up khổng lồ (`>50%` khung hình) trong khi test toàn vật thể nhỏ/trung bình.
- Sửa bug nhỏ ở `benchmark_report.py` dòng 350–353 khi in `ambiguity_queue_size` giống nhau cho cả `train` và `val` (tách đếm riêng theo `split`).

#### [NEW Helper in `eval_golden.py` or `validate_dataset.py`]: Pre-Train Zero-Shot Semantic Sanity Check
- Chạy nhanh `yolo26n-seg.pt` (hoặc `yolo11n-seg.pt` COCO pretrained) trên ~40 ảnh mẫu ngẫu nhiên của từng nguồn dữ liệu mới nạp; báo **CẢNH BÁO ĐỎ / CHẶN TRAIN** nếu:
  - Bất kỳ lớp `Motorcyclist (5)` hay `Other vehicle (3)` nào trùng IoU ≥ 0.6 với dự đoán COCO `car` / `truck` quá `25%` số mẫu (chặn đứng vĩnh viễn lỗi `van -> Motorcyclist`, `truck -> Other vehicle` trước khi train!).
  - Sinh sẵn ảnh lưới `preview_per_class.jpg` (1 hàng/lớp cho đủ 11 lớp) ở Cell 9 để người dùng kiểm chứng trực quan trước khi bước vào Cell 10.

---

### Component 4: Tái Thiết Kế Pipeline Kaggle V3 (`build_kaggle_nb.py` → `kaggle_team34_pipeline.ipynb`)

Các thay đổi cụ thể trong notebook V3:
1. **Cell 1 & Cell 3 — Bộ nguồn Verified Public Datasets mới (tất cả đã kiểm tra tồn tại qua Kaggle API & HF API)**:
   - **`awsaf49/coco-2017-dataset`** (COCO Instance Seg thật: `Car, Bus, Truck, Motorcycle, Bicycle, Pedestrian, Traffic light, Stop Sign` + Geometric Merger → `Motorcyclist, Cyclist`, lọc ưu tiên `Bicycle/Motorcycle/Person/Traffic light`).
   - **`a7madmostafa/bdd100k-yolo`** (BDD100K dashcam ngày/đêm/mưa: `person, rider, car, bus, truck, bike, motor, traffic light, traffic sign` → SAM 2 + Geometric Merger).
   - **`redzapdos123/indian-driving-dataset-detections-yolov11`** (IDD đường phố châu Á đông đúc: có **`autorickshaw, caravan, trailer, vehicle fallback` → `Other vehicle (3)`** + `rider, motorcycle, bicycle, person, sign, light` → SAM 2 + Merger).
   - **`lvnduy/street-traffic-signs-in-vietnam-yolov8`** (`837 MB`, Public Domain: 58 loại biển báo giao thông VN thực tế → SAM 2 → `Sign (9)`).
   - **`abtinzandi/obstacle-detection-dataset`** (`1.5 GB`, MIT: bổ sung xe, xe đạp, người đi bộ, biển báo góc đường).
   - **`star092304/Traffic-sign-detection-VietNam`** (HF): sửa Cell 4 giải nén bắt buộc `*.zip` (`traffic-sign-detection-vietnam.zip`), bỏ qua ảnh hướng dẫn ở root, kiểm tra `n_instances > 0` mới nạp.
   - **Tải-Xử Lý-Xóa tuần tự (Disk-Safe Streaming)**: Cài `KAGGLEHUB_CACHE` tại `/kaggle/tmp/kh_cache`; xử lý xong từng dataset là xóa ngay thư mục thô trong `/kaggle/tmp/kh_cache` để tổng đĩa tạm không vượt trần Kaggle.
   - *(Mặc định đặt `RUN_VN_DASHCAM = False` hoặc `DASHCAM_MAX_FRAMES = 250` do ảnh OpenHay bị Gaussian blur toàn thân vật thể; ưu tiên dùng dữ liệu sắc nét + lọc nắp capo).*
2. **Cell 7 — Khóa Golden Val Sạch Tuyệt Đối**:
   - Chỉ lấy `Golden Val` phân tầng từ **các nguồn người gắn nhãn chuẩn** (ưu tiên `COCO val2017` traffic + `BDD100K val` + `IDD val` + `VNTS val`), bảo đảm **mỗi lớp trong `0..10` có ít nhất `20–30` instances sạch** và **0 ảnh từ `vehicle-dataset` hay pseudo-label**.
3. **Cell 8 — Augment Sạch Trên Đường Phố**:
   - `Crop-Zoom` trên `vnts_sam2` & `lvnduy` (đảm bảo có ít nhất vài trăm patch biển báo vừa thang 640).
   - `Copy-Paste` lấy ngân hàng từ **`IDD autorickshaw/trailer (3)` + `COCO/IDD Bicycle (6)` + `Cyclist (7)` + `VNTS Sign (9)`**, dán lên nền đường phố sắc nét của `BDD100K / IDD`.
   - `Weather Augmentation` (đêm/mưa/lóa nắng) áp trên tối đa 300–400 ảnh đường phố sạch.
4. **Cell 9 — Sanity Check & Per-Class Preview Gate**:
   - Chạy Pre-Train Zero-Shot Semantic Sanity Check trước khi cho train; báo chi tiết diện tích `<64px` từng lớp.
5. **Cell 10–12 — Train Ổn Định + Báo Cáo Nhanh**:
   - Huấn luyện `yolo26n-seg.pt`, `imgsz=640`, `epochs=65`, `batch=16`, `patience=20` (`optimizer="auto"` hoặc `AdamW` `lr0=0.002`, `close_mosaic=10`, `copy_paste=0.1`) trên ~`4,000–5,500` ảnh sạch.
   - Giới hạn `run_eval_or_qc` trên tập train ở `max_images=600` (thay vì toàn bộ 4,000+ ảnh bằng CPU mất 10 phút) để hoàn thành nhanh hơn mà vẫn lọc được nhãn nghi sai.

---

## Verification Plan

### Automated Tests (Chạy trên Windows Workspace sau khi bạn duyệt plan và sửa code)
1. **Unit tests mở rộng (`tests/test_phase2.py`)**:
   - Test `classes.txt` parsing cho layout kiểu `nadinpethiyagoda/vehicle-dataset-for-yolo` + khẳng định **fail-fast** khi dataset thiếu cả `data.yaml` lẫn `classes.txt` (không còn silent `{0:0}` identity corruption).
   - Test giải nén `.zip` mô phỏng `star092304/Traffic-sign-detection-VietNam` (bỏ qua `illustration.webp` không nhãn ở root, lấy đúng ảnh+nhãn trong `dataset/train`).
   - Test ánh xạ Roboflow 58 lớp biển báo VN (`P.102`, `W.201a`, ...) → `9: Sign` và IDD YOLO (`autorickshaw`, `rider`, `bike`, `motor`) → đúng 11 lớp + Geometric Merger.
   - Test bộ lọc nắp capo `Car` ở mép dưới và loại trùng `Pedestrian` nằm trong `Motorcyclist` trong `pseudo_label_folder`.
   - Test Pre-Train Semantic Sanity Check bắt lỗi đảo lớp giả lập (`van` bị gán nhãn `Motorcyclist`).
   - Chạy `python -m unittest discover -s tests -v`.
2. **Notebook Compilation & AST Verify**:
   - Chạy `build_kaggle_nb.py` để tái tạo `kaggle_team34_pipeline.ipynb` V3, kiểm tra `ast.parse` và `py_compile` toàn bộ file nhúng.

### Manual Verification (Trên Kaggle & Kiểm Thử Chéo Sau Khi Train V3)
1. Sau khi chạy Cell 9 trên Kaggle, kiểm tra log xác nhận:
   - Không còn nguồn nào `403`; đủ nguồn `COCO 2017`, `BDD100K`, `IDD`, `VNTS / LVNDUY` được `Converted`.
   - Sanity Gate báo `PASS`, và ảnh `preview_per_class.jpg` hiển thị **đúng tuyệt đối** `Motorcyclist = xe máy có người`, `Other vehicle = xe ba gác/kéo/đặc chủng`, `Bus = xe buýt`, `Truck = xe tải`, `Sign = biển báo VN`, `Bicycle/Cyclist >= 150` train instances.
2. Sau khi train xong Cell 10–12:
   - Dùng `kaggle kernels output` tải `outputs/submission.zip` và `quick_local_pack.zip`.
   - Chạy `validate_onnx` + chấm thử trên `sample_data` (mốc kỳ vọng: điểm trên `sample_data` vọt từ `5.48` lên đáng kể, đồng thời `Golden Val` sạch đạt ổn định ở mức hợp lý thay vì ảo).
   - Nộp `submission.zip` mới lên BTC và kiểm tra bảng AP 11 lớp xem `Truck`, `Bus`, `Other vehicle`, `Motorcyclist`, `Pedestrian`, `Bicycle`, `Cyclist`, `Sign` có tăng đồng đều không.

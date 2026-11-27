# Báo Cáo Tham Chiếu Thông Số Ban Đầu, Độ Chính Xác, Confidence, IoU, F1 & Hiệu Suất (`yolo26n-seg`)

Tài liệu này tổng hợp toàn bộ thông số tham chiếu về **độ chính xác (mAP, AP50, AP75)**, **Confidence**, **IoU**, **Precision / Recall / F1**, **chất lượng dữ liệu** và **hiệu suất hệ thống** của mô hình **`yolo26n-seg`**, trích xuất từ 3 nguồn đã kiểm chứng:
1. **Tài nguyên chính thức của Ban Tổ Chức**: `train_and_export.ipynb` (`v3.0.0+7880385f`), `work/_selfcheck/detscore_selfcheck/` (`detscore v0.2.0`), `sample_data/`.
2. **Dữ liệu & cấu hình hiện có trong workspace**: `configs/competition.yaml`, `src/eval_golden.py`, `work/qc/sample_data/qc_report.json`, `work/qc/master/qc_report.json`, và đo kiểm trực tiếp cấu trúc `nc = 11` trên `ultralytics==8.4.215`.
3. **Benchmark chuẩn Ultralytics** cho `YOLO26n-seg` trên COCO val2017 (`imgsz = 640`).

> [!IMPORTANT]
> **Xác nhận minh bạch về số liệu baseline của Ban Tổ Chức**:
> - Tất cả các cell trong `train_and_export.ipynb` đều trống output (`execution_count = None`), và `reserved_fingerprints.json` có `entries: []`.
> - **BTC KHÔNG cung cấp sẵn** file trọng số baseline hay bảng điểm đo sẵn (`mAP`, `Precision`, `Recall`, `F1`) trên tập test giao thông Việt Nam; `sample_data` chỉ là 10 ảnh vẽ tổng hợp minh họa định dạng (README_VI.md ghi rõ không dùng để train hoặc chấm).
> - Thay vào đó, BTC **quy định cứng toàn bộ ngưỡng Confidence, IoU, Mask Threshold, công thức mAP, ngưỡng QC dữ liệu, kiến trúc tham chiếu và thông lượng T4** — được liệt kê chi tiết bên dưới để làm khung tham chiếu chuẩn cho mọi vòng thực nghiệm của đội.

---

## 1. Bảng Tổng Hợp Các Ngưỡng Độ Chính Xác, Confidence, IoU, Precision/Recall & F1 Theo Quy Định BTC

### 1.1. Các thông số vận hành cố định trong Bộ Chấm Chính Thức (`detscore_selfcheck`)

| Chỉ số / Tham số | Giá trị BTC quy định | Nguồn trong gói BTC | Vai trò & Ý nghĩa thực chiến |
| :--- | :--- | :--- | :--- |
| **Metric xếp hạng chính** | **`100 × mask mAP@[.5:.95]`** | `score.py`, `competition_config.yaml` | Điểm duy nhất dùng xếp hạng Public & Private LB. |
| **Các ngưỡng IoU tính mAP** | **`0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95`** (10 mức) | `competition_config.yaml` (`scoring.iou_thresholds`) | Mask phải khớp sát biên mới ăn được các mức IoU cao (`0.75–0.95`). |
| **Chỉ số phụ báo cáo** | **`AP50`** (`IoU ≥ 0.50`), **`AP75`** (`IoU ≥ 0.75`), **`per_class_ap`**, **`per_class_ap50`** | `score.py`, `train_and_export.ipynb` | Dùng để chẩn đoán lớp nào bị thiếu dữ liệu vs lớp nào lệch biên mask. |
| **Confidence khi chấm (`postprocess.conf`)** | **`0.001`** (`multi_label: true`, tối đa `30,000` ứng viên trước NMS) | `competition_config.yaml`, `decode.py` | Hạ sát `0.001` để tích phân toàn bộ đường cong PR (101 điểm recall `0.00:0.01:1.00`). |
| **IoU NMS (`postprocess.iou`)** | **`0.7`** (class-aware greedy NMS trên bbox) | `competition_config.yaml`, `decode.py` | Chỉ loại bbox trùng cùng lớp có `box IoU > 0.7`. Khác lớp không triệt tiêu nhau. |
| **Giới hạn số đối tượng (`max_det`)** | **`100`** instances / ảnh | `competition_config.yaml`, `decode.py` | Sắp xếp theo `conf` giảm dần, cắt tại `100`. Cảnh đông >100 xe/người sẽ mất vật thể `conf` thấp. |
| **Ngưỡng nhị phân Mask (`mask_threshold`)** | **`0.5`** (tương đương `logit > 0.0`) | `competition_config.yaml`, `decode.py` | Chuyển xác suất mask prototype thành mặt nạ nhị phân trước khi tính `pycocotools` Mask IoU. |
| **Ngưỡng cắt diện tích nhỏ (`min_gt_area_px`)** | **`64.0 px²`** tại thang `640` (`ignore_small_predictions: true`) | `competition_config.yaml`, `score.py` | GT `< 64 px²` (~`8×8 px`) bị bỏ qua (`ignore`); unmatched pred `< 64 px²` cũng không tính FP. |

### 1.2. Confidence, IoU & Công thức Precision / Recall / F1 tại các tầng Phân tích Lỗi & QC

Bộ chấm `detscore_selfcheck` chấm bằng tích phân **mAP@[.5:.95]** nên không xuất một giá trị đơn lẻ `F1` ở ngưỡng cố định; tuy nhiên **Confidence, IoU, Precision, Recall, và F1** được xác định rõ tại 3 điểm kiểm tra sau:

| Công cụ / Giai đoạn | Ngưỡng Confidence | Ngưỡng IoU | Cách tính / Ý nghĩa Precision, Recall & F1 |
| :--- | :--- | :--- | :--- |
| **1. `analyze_errors` trong `train_and_export.ipynb` (Cell 4 & 6)** | **`CONF_SHOW = 0.25`** | **`iou_match = 0.50`** *(Mask IoU)* | Đếm trực tiếp số `matched` (cùng lớp, `IoU ≥ 0.50`), `wrong_class` (khác lớp, `IoU ≥ 0.50`), `missed` (FN), và `false_pos` (FP) trên từng ảnh Val. |
| **2. `eval_golden.py` (QC Tầng 2 — Workspace)** | **`conf_thr = 0.25`** (Confusion)<br>**`high_conf_thr = 0.70`** (Flag lỗi) | **`0.45`** (Match chung)<br>**`0.55`** (Wrong class)<br>**`0.20`** (Missing GT)<br>**`bbox ≥ 0.70 & mask < 0.55`** (Bad boundary) | Tạo **Confusion Matrix `(12 × 12)`** (11 lớp + `BG_OR_MISSED`). Từ ma trận này tính ngay được per-class và macro **Precision, Recall, F1** tại điểm vận hành `conf = 0.25`. |
| **3. `model.val()` chuẩn của Ultralytics (Bước 3 train)** | Quét từ `0.0` → `1.0` | `0.50` (cho Precision/Recall/F1 curve) & `0.50:0.95` (mAP) | Tự động xuất `F1_curve.png`, `P_curve.png`, `R_curve.png`, `PR_curve.png` và tìm **ngưỡng `conf` tối ưu cho `max F1`** (lưu ý: đây là conf hiển thị/ứng dụng, còn chấm thi BTC luôn cố định `conf = 0.001`). |
| **4. QC Hình học Nhãn (`validate_dataset.py` & `competition.yaml`)** | *(Chỉ dùng Ground Truth)* | **`0.85`** (`duplicate_mask_iou`)<br>**`0.10`** (`rider_leak_iou`) | Bắt lỗi gán trùng polygon cùng lớp (`IoU > 0.85`) và lỗi quên gộp `Pedestrian + Motorcycle/Bicycle` (`IoU > 0.10`). |

**Công thức quy đổi chuẩn từ Confusion Matrix (tại `conf = 0.25`, `IoU_mask = 0.50`) để so sánh giữa các vòng:**
$$\text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{Wrong\_into}_c}, \qquad \text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c + \text{Wrong\_out}_c}, \qquad \text{F1}_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

---

## 2. Thông Số Hiệu Năng & Kiến Trúc Mô Hình Tham Chiếu (`yolo26n-seg`, `imgsz = 640`)

### 2.1. Benchmark công bố chính thức của Ultralytics `YOLO26n-seg` (COCO val2017, 80 lớp)

| Biến thể | `mAP^box (50-95)` | `mAP^mask (50-95)` | Params (M) | FLOPs (B) | CPU ONNX (ms) | T4 TRT 10 (ms) | Ghi chú |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`YOLO26n-seg`** | **`39.6`** *(34.4 Non-E2E)* | **`33.9`** *(~33.9 điểm thang 100)* | **`2.7`** | **`9.3`** | **`53.3 ± 0.5`** | **`2.1 ± 0.0`** | **Duy nhất được phép nộp.** *(Ultralytics chỉ công bố `mAP 50-95`, không tách riêng `AP50`/`AP75`/`F1` trong bảng chuẩn).* |
| `YOLO26s-seg` | `47.3` | `40.0` | `10.4` | `34.5` | `118.4 ± 0.9` | `3.3 ± 0.0` | Không được phép nộp |
| `YOLO26m-seg` | `52.5` | `44.1` | `23.6` | `121.7` | `328.2 ± 2.4` | `6.7 ± 0.1` | Không được phép nộp |
| `YOLO26l-seg` | `54.4` | `45.5` | `28.0` | `140.1` | `387.0 ± 3.7` | `8.0 ± 0.1` | Không được phép nộp |
| `YOLO26x-seg` | `56.5` | `47.0` | `62.8` | `314.0` | `787.0 ± 6.8` | `16.4 ± 0.1` | Không được phép nộp |

### 2.2. Thông số Kiến trúc & Export Thực tế (`nc = 11` theo hợp đồng BTC)

| Chỉ số kỹ thuật | Khi Train (`nc = 11`, Unfused Dual-Head) | Export ONNX Layout A (`nc = 11`, Fused) | Giới hạn cho phép (`validator`) | Nguồn kiểm chứng |
| :--- | :---: | :---: | :---: | :--- |
| **Số tầng PyTorch** | `309` layers *(có `one2one` + `Proto26.semseg`)* | `136` layers *(chỉ nhánh `one2many`)* | — | Đo trực tiếp trên `ultralytics` |
| **Tổng tham số PyTorch** | `3,057,605` (~3.06M) | `2,691,029` (~2.69M) | ≤ `10.6M` phần tử | Đo trực tiếp & `competition_config.yaml` |
| **Conv/ConvTranspose nodes & params** | — | **`118` nodes / `2,681,904` params** | Khớp chính xác signature | `reference_signature.json` |
| **ONNX Graph nodes / initializers** | — | **`416` nodes / `153` initializers** | ≤ `600` / ≤ `400` | `EXPORT_CONTRACT` |
| **Input / Output Tensor** | — | `images [1,3,640,640]` → `output0 [1,47,8400]`, `output1 [1,32,160,160]` | FP32 (`half=False`, no NMS) | `EXPORT_CONTRACT`, `validate.py` |
| **Kích thước `model.onnx` / `.zip`** | — | **~`10.8 – 11.0 MB`** | ≤ `42 MiB` zip / ≤ `45 MiB` raw | `errors.py`, `competition_config.yaml` |

> [!WARNING]
> **Ràng buộc kiến trúc ảnh hưởng trực tiếp tới IoU & Precision/Recall**:
> 1. **Prototype `160 × 160`**: Mỗi ô prototype bằng `4 × 4 px` ở thang `640`. Đối tượng đúng ngưỡng `64 px²` (`8 × 8 px`) chỉ nằm trên `2 × 2 = 4` ô prototype — rất dễ mất điểm ở `AP75` và các mức IoU `0.75–0.95` nếu nhãn không bám sát biên.
> 2. **Trần `max_det = 100` & `conf = 0.001` & `multi_label = true`**: Một anchor có thể bắn ra nhiều lớp nếu dữ liệu train bị lẫn lộn nhãn (`4 vs 5`, `0 vs 1`). Khi ảnh đông vượt 100 predictions, những dự đoán có `conf` thấp hơn sẽ bị cắt bỏ, làm tụt **Recall** và **mAP**.

---

## 3. Thông Số Chất Lượng & Tham Chiếu Dữ Liệu Của Ban Tổ Chức

### 3.1. Ngưỡng kiểm định chất lượng dữ liệu (`train_and_export.ipynb` & `competition.yaml`)

| Chỉ tiêu dữ liệu | Ngưỡng BTC / QC quy định | Ý nghĩa cảnh báo |
| :--- | :--- | :--- |
| **`MIN_TRAIN_INSTANCES`** | `100` *(rất ít `< 25`)* | Lớp dưới 100 instance trong train bị cảnh báo thiếu ví dụ |
| **`min_gt_per_class_per_split`** | `10` | Lớp dưới 10 instance trong Val khiến AP rất nhiễu |
| **`MAX_IMBALANCE_RATIO`** | `30.0 : 1` | Cảnh báo khi `max_class_count / min_class_count > 30` |
| **`DUP_HAMMING_MAX`** | `2` *(trên `dHash 64-bit`)* | Phát hiện ảnh trùng / gần giống và rò rỉ train–val |
| **`COORD_TOLERANCE`** | `1e-3` *(notebook)* / `1e-4` *(QC script)* | Tọa độ ngoài `[0, 1]` vượt tolerance sẽ báo lỗi nặng |
| **`MIN_GT_AREA_PX`** | `64.0 px²` ở thang `640` | Vật thể nhỏ hơn ngưỡng này bị bỏ qua khi chấm điểm |

### 3.2. Kết quả đo đạc thực tế trên `sample_data` (bộ mẫu BTC) và `data/master` hiện tại

| Bộ dữ liệu | Số ảnh | Ảnh rỗng | Tổng instances | Lỗi định dạng (`fatal` / `poly`) | Trùng lặp / Rò rỉ (`dHash ≤ 2`) | Cờ Edge Case QC | Vật thể `< 64 px² @640` | Tỉ lệ mất cân bằng (`max/min`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`sample_data`** *(10 ảnh tổng hợp `960×540` của BTC)* | `10` | `0` | `105` *(đủ 11 lớp)* | `0` / `0` | `0` | `0` cờ | `1 / 105` (`0.95%`, `45.3 px²` ở `sample_07`) | **`5.5 : 1`** (`22` Motorcyclist vs `4` Other vehicle) |
| **`data/master` hiện tại** *(từ `work/qc/master/qc_report.json`)* | `7` (`6` train / `1` val) | `0` | `23` *(thiếu `5/11` lớp)* | `0` / `0` | `0` | `0` cờ | `0` | Chưa đủ 11 lớp *(chỉ mới là mẫu test pipeline)* |

**Phân bố chi tiết từng lớp trong `sample_data` (`960 × 540`):**
- `0 Car`: `18` (`8` ảnh) | `area640` `436.2 – 6602.0` (`med 1700.3`) | `12–33` đỉnh
- `1 Bus`: `7` (`5` ảnh) | `area640` `1440.0 – 23504.7` (`med 15300.2`) | `4–32` đỉnh
- `2 Truck`: `8` (`6` ảnh) | `area640` `1270.0 – 20127.6` (`med 8420.9`) | `4–45` đỉnh
- `3 Other vehicle`: `4` (`4` ảnh) | `area640` `592.9 – 30016.5` (`med 4241.0`) | `9–88` đỉnh
- `4 Motorcycle`: `13` (`8` ảnh) | `area640` `1678.7 – 3576.2` (`med 2869.3`) | `26–47` đỉnh
- `5 Motorcyclist`: `22` (`9` ảnh) | `area640` **`45.3`**` – 3851.3` (`med 1060.2`) | `11–64` đỉnh *(1 mẫu `< 64 px²`)*
- `6 Bicycle`: `6` (`4` ảnh) | `area640` `2042.4 – 2346.9` (`med 2202.1`) | `47–53` đỉnh
- `7 Cyclist`: `5` (`3` ảnh) | `area640` `313.3 – 2101.1` (`med 1829.6`) | `31–56` đỉnh
- `8 Pedestrian`: `7` (`5` ảnh) | `area640` `106.4 – 2015.8` (`med 1770.6`) | `6–52` đỉnh
- `9 Sign`: `8` (`7` ảnh) | `area640` `450.2 – 1653.3` (`med 953.1`) | `4–16` đỉnh
- `10 Traffic light`: `7` (`5` ảnh) | `area640` `1047.1 – 1152.0` (`med 1123.5`) | `4` đỉnh

---

## 4. Thông Số Hiệu Suất Huấn Luyện & Suy Luận Trên Hạ Tầng BTC

| Nhóm hiệu suất | Thông số tham chiếu | Ý nghĩa vận hành |
| :--- | :--- | :--- |
| **Suy luận CPU ONNX (`onnxruntime`)** | Chuẩn Ultralytics: **`53.3 ± 0.5 ms/ảnh`** (`~18.8 FPS`, 1 luồng) | Container BTC cấp **`2` CPU**, chạy **song song 2 ảnh** (`ThreadPoolExecutor(workers=2)`, mỗi worker 1 luồng), `2048 MB` RAM, `300 s` timeout, `256 MiB` JSON RLE. Đủ suy luận ~`6,000 – 9,000` ảnh trong hạn mức `300 s`. |
| **Suy luận GPU T4 (TensorRT 10)** | Chuẩn Ultralytics: **`2.1 ± 0.0 ms/ảnh`** | Dùng để tham chiếu tốc độ phần cứng; khi chấm thi BTC chỉ chạy CPU ONNX. |
| **Huấn luyện Colab T4 (`imgsz=640`, `batch=16`)** | **`22.0 ảnh/giây`** (`T4_TRAIN_IMAGES_PER_SECOND["n"]`) | Công thức BTC: $T_{\text{phút}} = \frac{\text{EPOCHS} \times (N_{\text{train}}/22.0 + N_{\text{val}}/66.0)}{60}$. Ví dụ `1,000` ảnh × `60 epochs` (80/20) ≈ **`39 phút`**; `3,000` ảnh × `60 epochs` ≈ **`118 phút`**. |

---

## 5. Bộ Chỉ Số Bắt Buộc Ghi Nhận Khi Chạy `Baseline v0` Để So Sánh

Để lấp khoảng trống việc BTC không cung cấp điểm số baseline sẵn, ngay khi hoàn tất **Baseline v0** (trên tập dữ liệu thực đầu tiên với cấu hình mặc định `epochs=60, batch=16, imgsz=640, seed=39`), đội cần xuất và ghi lại đủ **8 nhóm chỉ số tham chiếu** sau trên cùng **Golden Validation Set**:
1. **Điểm chính thức & mAP**: `Score (100 × mask mAP@[.5:.95])`, `mAP50`, `mAP75`, và `per_class_ap` + `per_class_ap50` cho cả 11 lớp.
2. **Precision / Recall / F1 (tại `conf = 0.25, IoU = 0.50` & `max F1` từ `model.val()`)**: `Precision`, `Recall`, `F1-score` tổng và từng lớp, kèm `optimal_conf_for_f1`.
3. **Phân tích lỗi (`analyze_errors` & `eval_golden.py`)**: `missed (FN)`, `false_pos (FP)`, `wrong_class`, `bad_mask_boundary`, `confusion_matrix (12×12)`.
4. **Hiệu suất & Tài nguyên**: `model.onnx` size (`MiB`), `submission.zip` size (`MiB`), `latency_ms_p50` + `decode_ms_p50` (CPU), thời gian train/epoch, và `Public LB Score`.

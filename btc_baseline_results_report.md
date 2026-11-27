# Báo Cáo Kết Quả Chạy Thực Nghiệm Mặc Định Của Ban Tổ Chức (`BTC Baseline Run`)

Báo cáo này tổng hợp toàn bộ kết quả đo đạc thực tế khi chạy quy trình chuẩn từ [train_and_export.ipynb](file:///c:/Users/basduy05/Downloads/VINAI/hackathon/train_and_export.ipynb) trên [sample_data](file:///c:/Users/basduy05/Downloads/VINAI/hackathon/sample_data/README_VI.md) trong thư mục scratch tách biệt (giữ nguyên 100% mã nguồn workspace chính).

> [!NOTE]
> **Mục đích & Phạm vi tham chiếu**: Vì `sample_data` chỉ gồm **10 ảnh tổng hợp minh họa định dạng** (`960 × 540`, `105` instances), các con số dưới đây là **mốc tham chiếu kỹ thuật mặc định** (hành vi hội tụ của `yolo26n-seg`, ngưỡng Confidence/IoU/F1, kích thước ONNX, tốc độ `detscore`, và các cặp lớp dễ nhầm lẫn). Khi bạn có tập dữ liệu giao thông Việt Nam thực tế, hãy chạy lại đúng bộ tham số mặc định này để lập mốc `Baseline v0` thật và so sánh trực tiếp với các vòng cải tiến dữ liệu.

---

## 1. Cấu Hình Chạy Mặc Định & Dữ Liệu Đầu Vào

| Tham số / Hạng mục | Giá trị thực thi | Ghi chú |
| :--- | :--- | :--- |
| **Seed / Chia tập** | `TEAM_ID = 39`, `SEED = 39`, `VAL_RATIO = 0.2` | **Train**: `8` ảnh (`sample_01..07, 10`, `83` instances, đủ `11/11` lớp)<br>**Val**: `2` ảnh (`sample_08, 09`, `22` instances, `10/11` lớp — thiếu `Cyclist`) |
| **QC Dữ liệu (Tier-1)** | `ok = true`, `0` lỗi format/poly/dup/leak | `1` instance `< 64 px² @640` (`Motorcyclist` `45.3 px²` ở `sample_07.txt`, nằm trong Train) |
| **Mô hình & Huấn luyện** | `yolo26n-seg.pt`, `imgsz = 640`, `epochs = 60`, `batch = 16`, `patience = 20`, `deterministic = True` | Chạy đủ `60/60` epochs trên CPU (`Intel Core i7-8850H 2.60GHz`, `workers = 0`) |
| **Thời gian train thực tế** | **`480.25 s` (`8.0 phút`)** | ~`7.5 s/epoch` train + `0.5 s/epoch` val trên CPU |
| **Best checkpoint (`best.pt`)** | `6,595,933` bytes (`6.29 MiB`) | Epoch đạt `mAP50-95(M)` cao nhất khi train là **Epoch 45** (`0.5732`); Epoch 60 đạt `0.5616` |

---

## 2. Kết Quả Tổng Hợp Hiệu Năng: mAP, Precision, Recall, F1, Confidence & IoU

### 2.1. Bảng so sánh tổng thể giữa 2 bộ đánh giá (`detscore_selfcheck` BTC vs `Ultralytics model.val()`)

| Chỉ số | `detscore_selfcheck` trên **Val (2 ảnh)** *(Chuẩn BTC — `model.onnx`)* | `Ultralytics val()` trên **Val (2 ảnh)** *(`best.pt`)* | `detscore_selfcheck` trên **Toàn bộ 10 ảnh `sample_data`** *(Tham khảo thêm `Cyclist`)* |
| :--- | :---: | :---: | :---: |
| **Score chính thức (`100 × mask mAP@[.5:.95]`)** | **`59.59`** | **`56.16`** | **`67.91`** |
| **Mask `mAP50`** | **`0.7948`** | **`0.7605`** | **`0.8570`** |
| **Mask `mAP75`** | — | **`0.6367`** | — |
| **Box `mAP50` / `mAP75` / `mAP50-95`** | — | `0.7959` / `0.7605` / `0.6897` | — |
| **Precision (`P`)** | **`0.7200`** *(`@conf=0.25, IoU=0.50`)*<br>**`0.8095`** *(`@max F1, conf=0.45`)* | **`0.7050`** *(macro)* | **`0.7982`** *(`@conf=0.25, IoU=0.50`)* |
| **Recall (`R`)** | **`0.8182`** *(`@conf=0.25, IoU=0.50`)*<br>**`0.7727`** *(`@max F1, conf=0.45`)* | **`0.7250`** *(macro)* | **`0.8750`** *(`@conf=0.25, IoU=0.50`)* |
| **F1-Score** | **`0.7660`** *(`@conf=0.25`)*<br>**`0.7907`** *(micro-sweep `@conf=0.45`)* | **`0.5835`** *(macro mean `m.f1`)*<br>**`0.58` tại `conf=0.396`** *(`MaskF1_curve`)* | **`0.8349`** *(tối ưu `@conf=0.25`)* |
| **Mask IoU của các TP (`mean / median`)** | **`0.8268 / 0.8670`** *(`min 0.5058 – max 0.9553`)* | — | **`0.8695 / 0.8864`** *(`min 0.5058 – max 0.9746`)* |
| **Box IoU của các TP (`mean / median`)** | **`0.8913 / 0.9378`** *(`min 0.4455 – max 0.9734`)* | — | **`0.9279 / 0.9478`** *(`min 0.4455 – max 1.0000`)* |
| **Confidence TP vs FP (`mean / median`)** | **TP: `0.7665 / 0.8041`**<br>**FP: `0.5206 / 0.4627`** | — | **TP: `0.7475 / 0.8027`**<br>**FP: `0.4926 / 0.4493`** |

*(Ghi chú: Độ lệch nhỏ giữa `detscore` `59.59` và `Ultralytics val` `56.16` trên tập Val 2 ảnh là bình thường vì `detscore` tính COCO RLE trên độ phân giải gốc `960×540` và 101 điểm recall bằng `decode.py`, còn Ultralytics val tính mask ở độ phân giải thu gọn mặc định `retina_masks=False`. Điểm khác nhau giữa `0.7907` micro-F1 và `0.58` macro-F1 đến từ việc macro-F1 bị 2 lớp `Other vehicle` và `Sign` có `F1=0` ở Val kéo trung bình xuống).*

---

### 2.2. Bảng chi tiết từng lớp trên Val (`2` ảnh, `22` instances) & Toàn bộ `sample_data` (`10` ảnh)

| ID | Lớp | GT Val | `detscore` Val `AP50-95` | `detscore` Val `AP50` | Val `P / R / F1` *(`@conf=0.25`)* | Ultralytics Val Mask `P / R / F1 / AP50-95` | All-10 `AP50-95` *(tham khảo)* | All-10 `F1@0.25` |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `0` | **Car** | `5` | `0.8525` | `1.0000` | `0.833 / 1.000 / 0.909` | `0.841 / 1.000 / 0.914 / 0.748` | `0.8642` | `0.9000` |
| `1` | **Bus** | `1` | `1.0000` | `1.0000` | `0.500 / 1.000 / 0.667` | `0.438 / 1.000 / 0.609 / 0.896` | `0.8918` | `0.9333` |
| `2` | **Truck** | `2` | **`0.1873`** | `0.3939` | `0.250 / 0.500 / 0.333` | `0.215 / 0.500 / 0.301 / 0.130` | `0.7834` | `0.7368` |
| `3` | **Other vehicle** | `1` | **`0.0000`** | `0.0000` | `0.000 / 0.000 / 0.000` | `1.000 / 0.000 / 0.000 / 0.000` | **`0.4545`** | `0.6667` |
| `4` | **Motorcycle** | `4` | `0.6639` | `1.0000` | `1.000 / 1.000 / 1.000` | `0.887 / 1.000 / 0.940 / 0.665` | `0.7219` | `1.0000` |
| `5` | **Motorcyclist** | `4` | `0.5723` | `0.9505` | `0.800 / 1.000 / 0.889` | `0.588 / 0.750 / 0.659 / 0.604` | `0.6858` | `0.7925` |
| `6` | **Bicycle** | `1` | `0.7000` | `1.0000` | `1.000 / 1.000 / 1.000` | `0.710 / 1.000 / 0.830 / 0.697` | `0.6267` | `1.0000` |
| `7` | **Cyclist** | `0` | *(n/a)* | *(n/a)* | *(n/a)* | *(n/a)* | **`0.3340`** | **`0.0000`** *(5/5 nhầm sang Motorcyclist)* |
| `8` | **Pedestrian** | `1` | `0.8000` | `1.0000` | `1.000 / 1.000 / 1.000` | `0.914 / 1.000 / 0.955 / 0.796` | `0.6004` | `0.8000` |
| `9` | **Sign** | `2` | **`0.4832`** | `0.6040` | `0.000 / 0.000 / 0.000` | `1.000 / 0.000 / 0.000 / 0.584` | `0.6551` | `0.6667` |
| `10` | **Traffic light** | `1` | `0.7000` | `1.0000` | `0.500 / 1.000 / 0.667` | `0.456 / 1.000 / 0.627 / 0.498` | `0.8520` | `0.9333` |

---

### 2.3. Đường Quét Confidence (`conf_sweep`) trên Val (`IoU_mask = 0.50`)

| `conf` | `TP` | `FP` | `FN` | `Precision` | `Recall` | `F1` | Ghi chú |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| `0.05` | `19` | `35` | `3` | `0.3519` | `0.8636` | `0.5000` | Nhiều dự đoán nhiễu nền |
| `0.15` | `18` | `12` | `4` | `0.6000` | `0.8182` | `0.6923` | |
| **`0.25`** | `18` | `7` | `4` | `0.7200` | `0.8182` | `0.7660` | Ngưỡng hiển thị mặc định BTC (`CONF_SHOW`) |
| `0.35` | `18` | `7` | `4` | `0.7200` | `0.8182` | `0.7660` | Ổn định |
| **`0.45`** | `17` | `4` | `5` | `0.8095` | `0.7727` | **`0.7907`** | **Đạt đỉnh micro-F1 trên Val** |
| `0.60` | `15` | `2` | `7` | `0.8824` | `0.6818` | `0.7692` | |
| `0.70` | `12` | `0` | `10` | **`1.0000`** | `0.5455` | `0.7059` | Không còn FP nào từ ngưỡng `0.70` |

---

## 3. Phân Tích Lỗi & Biểu Đồ Trực Quan

### 3.1. Các lỗi edge-case phát hiện được ngay cả trên tập mẫu nhỏ
1. **Nhầm `Cyclist` (`7`) → `Motorcyclist` (`5`)**: Trên toàn bộ 10 ảnh `sample_data`, cả **5/5** instance `Cyclist` (ở ngưỡng `conf >= 0.25`) bị dự đoán nhầm thành `Motorcyclist`. Do cấu trúc hai bánh + người ngồi rất giống nhau và chỉ có `5` mẫu train trong `sample_data`, mô hình dễ gộp nhầm sang lớp đa số `Motorcyclist` (`18` mẫu train).
2. **Bỏ sót `Other vehicle` (`3`) và `Sign` (`9`)**: Ở Val (`conf >= 0.25`), `Other vehicle` bị sót `1/1` (`sample_09.jpg`), `Sign` bị sót `2/2` (`sample_08.jpg`, `sample_09.jpg`, conf nằm dưới `0.25`).
3. **Nhầm lẫn nhóm xe lớn (`Bus` vs `Truck`) & báo nhầm nền (`Traffic light` / `Car`)**: Trên `sample_09.jpg` có `1` `Truck` bị nhận thành `Bus`, và nền tòa nhà tạo FP `Traffic light 0.5`.

### 3.2. Biểu đồ huấn luyện, đường cong F1/PR và đối chiếu ảnh Val

![Quá trình huấn luyện 60 epochs (Loss & Metrics)](C:/Users/basduy05/.gemini/antigravity/brain/fc976a1b-7491-47dc-bfe4-c6be0efe52ba/scratch/btc_baseline/work/runs/train/results.png)

![Đường cong F1 theo Confidence của Mask trên Val](C:/Users/basduy05/.gemini/antigravity/brain/fc976a1b-7491-47dc-bfe4-c6be0efe52ba/scratch/btc_baseline/work/runs/val_eval/MaskF1_curve.png)

![Đường cong Precision-Recall của Mask trên Val](C:/Users/basduy05/.gemini/antigravity/brain/fc976a1b-7491-47dc-bfe4-c6be0efe52ba/scratch/btc_baseline/work/runs/val_eval/MaskPR_curve.png)

![Nhãn gốc tập Val (sample_08 & sample_09)](C:/Users/basduy05/.gemini/antigravity/brain/fc976a1b-7491-47dc-bfe4-c6be0efe52ba/scratch/btc_baseline/work/runs/val_eval/val_batch0_labels.jpg)

![Dự đoán của mô hình mặc định trên Val](C:/Users/basduy05/.gemini/antigravity/brain/fc976a1b-7491-47dc-bfe4-c6be0efe52ba/scratch/btc_baseline/work/runs/val_eval/val_batch0_pred.jpg)

---

## 4. Kiểm Định ONNX & Hiệu Suất Suy Luận Thực Tế

| Hạng mục kiểm định | Kết quả đo được | Ngưỡng hợp đồng BTC | Trạng thái |
| :--- | :--- | :--- | :---: |
| **`validate_onnx` (`detscore_selfcheck`)** | `"Tệp ONNX hợp lệ."`, `nc=11`, `names_metadata_present=True` | `yolo26n-seg`, Layout A FP32 | **PASS** |
| **Thông số đồ thị ONNX** | `416` nodes, `118` Conv/ConvTranspose, `2,696,944` Conv params, `2,748,255` total params, `opset 17` | `≤ 600` nodes, `≤ 10.6M` params | **PASS** |
| **Dung lượng `model.onnx` / `submission.zip`** | **`10.62 MiB`** (`11,136,148 B`) / **`9.45 MiB`** (`9,913,987 B`), nén `1.123:1` | `≤ 45 MiB` / `≤ 42 MiB`, ratio `≤ 20:1` | **PASS** |
| **Tốc độ Ultralytics CPU (`val_eval`)** | `preprocess 0.95 ms` + `inference 56.74 ms` + `postprocess 16.51 ms` (**`74.2 ms/ảnh`**) | — | Đo trên i7-8850H |
| **Tốc độ `detscore` ONNX CPU (`2` luồng, 10 ảnh)** | Model `latency_ms_p50 = 148.89 ms` + `decode_ms_p50 = 95.68 ms` (**`244.6 ms/ảnh`** tại `conf=0.001, max_det=100`) | Container `300 s`, `max_output_mb = 256` | `output_bytes = 273 KB` / 10 ảnh |

*(Lưu ý kỹ thuật: `detscore` decode lâu hơn Ultralytics val ở đây vì `conf=0.001` giữ đủ `100` ứng viên trên mỗi ảnh (`1,000` instances/10 ảnh) và giải mã RLE ở độ phân giải gốc `960×540` trên CPU laptop i7-8850H).*

---

## 5. Bảng Tham Chiếu Nhanh Để So Sánh Giữa Vòng Mặc Định vs Vòng Cải Tiến Dữ Liệu

Khi bạn chuyển sang tập dữ liệu giao thông Việt Nam thật và chạy các vòng Data-Centric, lưu bảng cột sau để đối chiếu trực tiếp:

| Trục so sánh | Giá trị mốc chạy thử BTC (`sample_data`) | Mục tiêu khi có dữ liệu thật (`Baseline v0` → `Iter N`) |
| :--- | :---: | :--- |
| **Official Score (`100 × mAP@[.5:.95]`)** | `59.59` *(Val 2 ảnh tổng hợp)* | Tăng đều qua từng vòng trên Golden Val cố định (`≥ 10` mẫu/lớp) |
| **Mask `mAP50` / `mAP75`** | `0.7948` *(detscore)* / `0.6367` *(Ultralytics)* | Theo dõi chênh lệch `mAP50 - mAP75` để biết cần siết biên mask hay thêm ảnh |
| **Precision / Recall / F1 (`@conf=0.25`)** | `0.7200` / `0.8182` / `0.7660` (`max F1 = 0.7907 @0.45`) | Cân bằng FP (`0.46 conf`) vs FN ở lớp hiếm (`Other vehicle`, `Cyclist`, `Sign`) |
| **Mask IoU / Box IoU của TP (`mean`)** | `0.8268` / `0.8913` | Đẩy `Mask IoU mean > 0.85` bằng SAM 2 & sửa polygon sát biên |
| **Confidence phân tách (`TP mean` vs `FP mean`)** | `0.7665` vs `0.5206` | Tăng khoảng cách tin cậy giữa TP và FP nhờ Hard Negative Mining |
| **Trọng lượng file & Đạt chuẩn BTC** | `9.45 MiB` zip / `validate_onnx = PASS` | Luôn giữ `zip ≤ 42 MiB` và `validate_onnx = ok` |

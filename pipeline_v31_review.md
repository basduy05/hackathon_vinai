# Pipeline V3.1 — nghiên cứu, thay đổi và bằng chứng

Ngày: 27/11/2026. Phạm vi: cải thiện độ tin cậy dữ liệu và đánh giá của V3; chưa train hoặc nộp bài mới.

## Cơ sở lựa chọn

- [Ultralytics: preprocessing annotated data](https://docs.ultralytics.com/guides/preprocessing-annotated-data/) khuyến nghị tách validation trước augmentation để tránh rò rỉ. V3.1 loại cả ảnh trùng nội dung và gần trùng với Golden Val khỏi mọi bản staged train trước Crop-Zoom/Copy-Paste/Weather. Chỉ loại ảnh gốc ở bước gộp master là quá muộn: ảnh augment có thể mang nội dung validation nhưng hash đã khác.
- [scikit-learn: common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) giải thích vì sao leakage tạo kết quả đánh giá lạc quan. Vì vậy quota Golden Val được tính theo vật thể đạt ngưỡng diện tích của bộ chấm, ảnh trùng không làm tăng quota, và thiếu lớp phải dừng train.
- [Simple Copy-Paste, CVPR 2021](https://arxiv.org/abs/2012.07177) nghiên cứu augmentation cho instance segmentation. Điều này hỗ trợ thử Copy-Paste, nhưng không chứng minh mức tăng điểm cho bộ dữ liệu này. V3.1 kết hợp ngân hàng nhiều nguồn và chọn lớp trước khi chọn instance để ngân hàng biển báo lớn không lấn hết Bicycle/Cyclist.
- [Ultralytics: augmentation guide](https://docs.ultralytics.com/guides/yolo-data-augmentation/) mô tả điều kiện sử dụng Copy-Paste với polygon. V3.1 giữ cấu hình train hiện có để có thể so sánh ảnh hưởng của thay đổi dữ liệu.

## Đã sửa

1. `src/data_gates.py`: chọn holdout ưu tiên validation gốc, bổ sung từ nguồn train được chỉ định khi cần; cập nhật lại độ thiếu lớp sau mỗi ảnh. Đếm polygon đạt `min_gt_area_px` (hiện 64 px² ở thang 640), loại ảnh trùng/gần trùng và xuất manifest. Notebook yêu cầu 25 instances/lớp; thiếu bất kỳ lớp nào thì dừng trước augmentation/train. Con số 25 là ngưỡng kỹ thuật theo kế hoạch, không bảo đảm ước lượng AP có độ tin cậy thống kê cao.
2. Golden Val chỉ nhận nguồn từ danh sách Kaggle cấu hình hoặc nguồn HF biển báo đã chỉ định. Dataset mount không rõ nguồn không tự động được coi là nhãn người đáng tin cho holdout.
3. Loại overlap với Golden Val khỏi các **bản staged tạm** trước khi augment; giữ nguyên dữ liệu upstream. Manifest ghi lại ảnh được chọn từ nguồn và split nào.
4. Semantic Gate chạy tối đa 48 ảnh **mỗi nguồn train không phải augment**, IoU 0.60; không báo PASS khi có dưới 3 cặp đối chiếu. Nguồn chỉ chứa Pedestrian/Sign/Traffic light được ghi `not_applicable` vì bộ luật hiện tại chỉ đối chiếu mâu thuẫn lớp xe 0–7; phải kiểm tra trực quan. Đây không phải chứng nhận nhãn đúng.
5. Copy-Paste dùng nhiều ngân hàng IDD/COCO/VNTS cùng lúc, chọn lớp đồng đều trước khi chọn instance. Không tự lấy một nền bất kỳ khi thiếu nguồn đường phố. Thống kê ngân hàng được trả về trong kết quả.
6. Pseudo-label lọc 20 px² trước merger, 36 px² sau merger, ngưỡng chiều rộng nắp capo 0.32; thêm callback `aux_sign_predictor`. Callback chỉ nhận nhãn mà mapping hiện tại nhận diện là Sign; chưa có checkpoint biển báo phụ được chọn hoặc đánh giá. Dashcam vẫn tắt mặc định.
7. BDD JSON adapter giữ đúng `split=val` thay vì ghi vào train, phát hiện cặp file train/val, và không còn loại toàn bộ frame ban ngày trong nhánh JSON.
8. Notebook chặn khi Tier-1 QC không đạt hoặc train có dưới 150 nhãn Bicycle/Cyclist mỗi lớp. Số này tính trên tập sau augmentation, không phải 150 mẫu gốc độc lập. Weather đặt 350 ảnh; QC train giữ giới hạn 500.
9. Đưa generator vào `tools/build_kaggle_nb.py`, sử dụng đường dẫn tương đối theo repository và nhúng thêm `data_gates.py`. Giữ nguyên generator cũ ở thư mục Antigravity. Cache cleanup chỉ nhắm cache tạm do notebook cấu hình, không xóa cache Kagglehub toàn tài khoản.

## Kiểm chứng đã chạy

```powershell
python -X utf8 tools/build_kaggle_nb.py
python -X utf8 -m unittest discover -s tests -v
```

- Sinh notebook V3.1: 13 cells; AST của tất cả code cells hợp lệ; biên dịch Python trong 18 tệp payload thành công.
- 20/20 tests PASS, gồm 7 regression tests mới: quota/leakage, bỏ mẫu nhỏ/trùng, bổ sung lớp thiếu, thiếu bằng chứng semantic, ngân hàng nhiều nguồn, aux sign/postfilter và BDD validation ban ngày.
- `git diff --check` trên các tệp tracked liên quan không báo lỗi whitespace.

## Những gì chưa thể kết luận

- Chưa tải lại các Kaggle/HF datasets, chưa xác minh quyền truy cập, license hoặc số mẫu thực tế trong phiên này. Download/convert vẫn có thể lỗi; các gate sẽ dừng khi dữ liệu còn lại không đủ yêu cầu.
- Golden Val gồm polygon upstream và mask SAM2 từ bbox người gán. Không được gọi toàn bộ là mask người gán hoặc “sạch tuyệt đối”. Vẫn phải xem preview, ảnh gốc và mask, đặc biệt lớp 3/5/7/9.
- Dhash là heuristic theo ảnh; chưa có metadata sequence/video để bảo đảm không rò giữa các frame khác nhau của cùng clip. Manifest hỗ trợ truy vết nhưng chưa giải quyết hoàn toàn vấn đề này.
- COCO có thể nhận nhầm autorickshaw thành car/truck; semantic alert là tín hiệu cần kiểm tra. Nguồn biển báo không được COCO kiểm tra đầy đủ. Không được suy ra PASS nghĩa là không có sai nhãn hoặc thiếu nhãn.
- Nguồn chuyên biển báo có thể thiếu nhãn người/xe; cần kiểm tra cách dùng để tránh dạy chúng thành nền âm. V3.1 chưa bổ sung missing-label-aware loss hay gán nhãn bổ sung cho toàn bộ nguồn này.
- Chưa có bằng chứng cải thiện AP, điểm sample_data hay Public LB. Chưa chạy GPU training hoặc ONNX benchmark V3.1.

## Bước đo chất lượng tiếp theo trên Kaggle

1. Import `kaggle_team34_pipeline.ipynb`, bật Internet/GPU; chạy tới Cell 9. Nếu quota thiếu, bổ sung ảnh nhãn chuẩn đúng lớp hoặc tăng cap của nguồn phù hợp thay vì hạ gate để chạy tiếp.
2. Xem `holdout_manifest.json`, `source_semantic_sanity.json`, báo cáo QC và `preview_per_class.jpg`; xác nhận mask/rider và nguồn gốc ảnh validation. Nếu gate báo thiếu bằng chứng, kiểm tra nguồn và tăng mẫu đối chiếu phù hợp.
3. Sau khi chất lượng dữ liệu được xác nhận, chạy train/export; chấm Golden Val theo từng lớp, ONNX contract và sample_data theo cùng quy trình V2. Không điều chỉnh dữ liệu/train liên tục để tối ưu riêng 10 ảnh sample_data.
4. Giữ cố định holdout đã kiểm tra cho các lần so sánh sau. Thử ablation có/không Copy-Paste và nguồn biển báo chuyên biệt; chỉ giữ thay đổi khi AP lớp yếu tăng mà AP lớp khác không suy giảm đáng kể.

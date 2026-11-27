# Dữ liệu mẫu (ảnh minh họa tổng hợp)

Bộ này gồm 10 ảnh **vẽ bằng mã**, không phải ảnh chụp thật, chỉ để minh họa định dạng nhãn YOLO-seg và 11 lớp của cuộc thi.
Không dùng để huấn luyện mô hình nộp bài và không phải bộ kiểm thử.

```
images/  ảnh .jpg
labels/  một tệp .txt cho mỗi ảnh, cùng tên
data.yaml
```

Mỗi dòng trong tệp nhãn là một instance (một đối tượng), tọa độ đa giác chuẩn hóa về [0, 1] theo kích thước ảnh:

```
<id lớp> x1 y1 x2 y2 x3 y3 ...
```

Thứ tự id lớp (theo competition_config.yaml):

- 0 = Car (Ô tô con)
- 1 = Bus (Xe buýt / xe khách)
- 2 = Truck (Xe tải)
- 3 = Other vehicle (Phương tiện khác)
- 4 = Motorcycle (Xe máy (không người))
- 5 = Motorcyclist (Xe máy (có người))
- 6 = Bicycle (Xe đạp (không người))
- 7 = Cyclist (Xe đạp (có người))
- 8 = Pedestrian (Người đi bộ)
- 9 = Sign (Biển báo)
- 10 = Traffic light (Đèn giao thông)

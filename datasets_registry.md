# Public Datasets Registry (Mandatory for Final Presentation)

> Mọi bộ dữ liệu công khai (open-source) sử dụng trong quá trình huấn luyện phải được ghi đầy đủ tại đây: **tên, đường dẫn (URL), giấy phép (license), số ảnh đã dùng, và cách ánh xạ về 11 lớp**.

| # | Tên bộ dữ liệu | URL | Giấy phép (License) | Dạng nhãn gốc | Số ảnh đã dùng (Train / Val) | Cách ánh xạ về 11 lớp | Trạng thái |
| :-: | :--- | :--- | :--- | :--- | :-: | :--- | :--- |
| 1 | *(Chờ nạp trong Phase 2 — VD: Mapillary Vistas)* | https://www.mapillary.com/dataset/vistas | CC BY-NC-SA 4.0 | Instance Seg | 0 | Xem `configs/class_mapping.yaml` | Planned |
| 2 | *(Chờ nạp trong Phase 2 — VD: IDD - India Driving Dataset)* | http://idd.insaan.iiit.ac.in/ | Academic / Research | Instance Seg | 0 | `auto-rickshaw -> 3`, `rider+bike -> 5/7` | Planned |
| 3 | *(Chờ nạp trong Phase 2 — VD: BDD100K Seg)* | https://github.com/bdd100k/bdd100k | BDD100K License | Instance Seg | 0 | `rider+bike -> 5/7`, night/rain subset | Planned |
| 4 | *(Chờ nạp trong Phase 2 — VD: VNTS / Zalo Traffic Sign)* | https://huggingface.co/datasets/thanhhiepvos/vietnam_traffic_sign | Open-source / Academic | BBox -> SAM2 | 0 | `All VN signs -> 9 (Sign)` | Planned |
| 5 | *(Chờ nạp trong Phase 2 — VD: Roboflow VN Traffic)* | https://roboflow.com | CC BY 4.0 | BBox/Poly -> SAM2 | 0 | Xem `configs/class_mapping.yaml` | Planned |
| 6 | *(Chờ nạp trong Phase 2 — VD: COCO 2017 Traffic Subset)* | https://cocodataset.org | CC BY 4.0 | Instance Seg | 0 | `person+bike -> 5/7`, street filter | Planned |

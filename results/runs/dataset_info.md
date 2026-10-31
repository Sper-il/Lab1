# FashionMNIST — 10 lớp & Thông số Dataset

> **Nguồn:** `src/data/dataset.py` (CLASS_NAMES), `src/data/eda.py` (get_dataset_stats),
> `data/splits/split_info.txt` (số lượng thực tế sau split).
> **Sinh ra:** từ run EDA + split pipeline thực tế, seed = 42.

---

## 1. Bảng 10 lớp

| ID | Tên lớp | Tiếng Việt | Train | Val | Test |
|----|----------|-------------|-------|-----|------|
| 0 | T-shirt/top | Áo thun | 5.000 | 1.000 | 1.000 |
| 1 | Trouser | Quần | 5.000 | 1.000 | 1.000 |
| 2 | Pullover | Áo len | 5.000 | 1.000 | 1.000 |
| 3 | Dress | Váy | 5.000 | 1.000 | 1.000 |
| 4 | Coat | Áo khoác | 5.000 | 1.000 | 1.000 |
| 5 | Sandal | Dép | 5.000 | 1.000 | 1.000 |
| 6 | Shirt | Áo sơ mi | 5.000 | 1.000 | 1.000 |
| 7 | Sneaker | Giày thể thao | 5.000 | 1.000 | 1.000 |
| 8 | Bag | Túi xách | 5.000 | 1.000 | 1.000 |
| 9 | Ankle boot | Ủng | 5.000 | 1.000 | 1.000 |
| | **Tổng** | | **50.000** | **10.000** | **10.000** |

> Ảnh mẫu từng lớp: `results/runs/per_class_samples.png`
> Biểu đồ cân bằng: `results/runs/class_distribution.png`

---

## 2. Thông số dataset

| Thông số | Giá trị |
|----------|---------|
| Tổng số mẫu gốc | 70.000 (60.000 train + 10.000 test) |
| Kích thước ảnh | 28 × 28 pixel, 1 kênh (xám) |
| Shape tensor | `(1, 28, 28)` |
| Dtype | `float32` |
| Số lớp | 10 |
| Tổng tham số ảnh | 784 pixel |
| Giá trị pixel | [0.0, 1.0] |
| **Mean** | **0.2860** |
| **Std** | **0.3530** |
| Mẫu thiếu (NaN) | 0 |
| Ảnh hỏng | 0 |
| Mẫu hợp lệ | 60.000 / 60.000 (100%) |
| Cân bằng lớp | Hoàn hảo — 6.000/lớp trong tập gốc |

---

## 3. Chia tập (sau split, seed = 42)

| Tập | Số mẫu | Tỷ lệ | Số batch (bs=128) | Vai trò |
|-----|---------|--------|--------------------|---------|
| **Train** | 50.000 | 5/6 từ 60k train gốc | 391 | Model học trọng số |
| **Validation** | 10.000 | 1/6 từ 60k train gốc | 79 | Đánh giá trong lúc train, quyết định early stopping |
| **Test** | 10.000 | — (tập test gốc) | 79 | Giữ kín, mở 1 lần ở bước đánh giá cuối |
| **Tổng** | 70.000 | | 549 | |

- **Stratified split**: bật → mỗi lớp giữ đúng 5.000 train + 1.000 val + 1.000 test
- **Tỷ lệ 5/6 – 1/6**: chọn tỷ lệ này vì chỉ nó cho ra số lượng tròn mỗi lớp
  (5.000 / 1.000 từ 6.000 ảnh gốc)
- **Random seed**: 42 → reproducible, chạy lại cho kết quả giống hệt
- Metadata lưu tại: `data/splits/split_info.txt`

---

## 4. Chuẩn hóa

Công thức áp dụng (mean/std tính từ EDA):

```
normalized = (pixel - 0.2860) / 0.3530
```

Ví dụ: pixel 0.5 → (0.5 − 0.286) / 0.353 = **0.606**

---

## 5. Kết quả đánh giá cuối cùng

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **92.10%** |
| Precision (macro) | 92.11% |
| Recall (macro) | 92.10% |
| F1-Score (macro) | 92.09% |
| Best epoch | 11 / 15 |
| Training time | ~75 s/epoch (CPU) |

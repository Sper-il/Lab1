# FashionMNIST Classification — Quy Trình Thực Hiện Dự Án

**Tác giả:** Sper-il  
**Ngày:** 2026-10-15  
**Dataset:** [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)  
**Framework:** PyTorch 2.14+  
**GitHub:** https://github.com/Sper-il/Lab1

---

## Mục lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Giai đoạn 1: Chuẩn bị dữ liệu](#2-giai-đoạn-1-chuẩn-bị-dữ-liệu)
3. [Giai đoạn 2: Phát triển mô hình](#3-giai-đoạn-2-phát-triển-mô-hình)
4. [Giai đoạn 3: Huấn luyện và đánh giá](#4-giai-đoạn-3-huấn-luyện-và-đánh-giá)
5. [Tổng kết kết quả](#5-tổng-kết-kết-quả)
6. [Cấu trúc dự án](#6-cấu-trúc-dự-án)

---

## 1. Tổng quan dự án

### 1.1 Bài toán

Xây dựng mạng nơ-ron để phân loại ảnh xám 28×28 của các mặt hàng thời trang thành 10 danh mục:

| Mã | Nhãn |
|----|------|
| 0 | Áo thun (T-shirt/top) |
| 1 | Quần dài (Trouser) |
| 2 | Áo len (Pullover) |
| 3 | Váy (Dress) |
| 4 | Áo khoác (Coat) |
| 5 | Dép (Sandal) |
| 6 | Áo sơ mi (Shirt) |
| 7 | Giày thể thao (Sneaker) |
| 8 | Túi xách (Bag) |
| 9 | Ủng (Ankle boot) |

### 1.2 Thông số Dataset

| Thông số | Giá trị |
|----------|---------|
| Số mẫu huấn luyện | 60,000 |
| Số mẫu kiểm tra | 10,000 |
| Kích thước ảnh | 28×28 (1 kênh) |
| Số lớp | 10 (cân bằng) |
| Mục tiêu độ chính xác | ≥ 90% |

### 1.3 Công nghệ sử dụng

| Thành phần | Công nghệ |
|------------|-----------|
| Framework | PyTorch 2.14 |
| Xử lý dữ liệu | NumPy, Pandas |
| Trực quan hóa | Matplotlib, Seaborn |
| Đánh giá | scikit-learn |
| Cấu hình | YAML |

---

## 2. Giai đoạn 1: Chuẩn bị dữ liệu

### 2.1 Bước 1 — Phân tích khám phá dữ liệu (EDA)

**Mục tiêu:** Hiểu cấu trúc và đặc điểm của dataset.

**Công việc:**
- Tải dataset FashionMNIST qua `torchvision.datasets.FashionMNIST`
- Tính toán thống kê: mean, std, min, max
- Trực quan hóa mẫu ảnh từ mỗi lớp
- Phân tích phân bố cường độ pixel
- Kiểm tra cân bằng lớp

**File kết quả:**
```
results/runs/
├── class_distribution.png    # Biểu đồ cột số lượng từng lớp
├── sample_images.png         # Lưới ảnh mẫu
├── pixel_histogram.png       # Histogram cường độ pixel
└── per_class_samples.png     # Một mẫu từ mỗi lớp
```

**Kết quả chính:**
- 60,000 mẫu huấn luyện, cân bằng hoàn hảo (6,000/lớp)
- Mean: 0.286, Std: 0.353
- Khoảng pixel: [0, 1]
- Không có dữ liệu thiếu hoặc hỏng

### 2.2 Bước 2 — Làm sạch dữ liệu

**Mục tiêu:** Kiểm tra chất lượng dữ liệu và xử lý các mẫu không hợp lệ.

**Công việc:**
- Kiểm tra giá trị NaN/Inf
- Xác thực kích thước ảnh
- Kiểm tra phạm vi nhãn
- Đảm bảo không có ảnh hỏng

**Kết quả:** 60,000/60,000 mẫu hợp lệ (100%)

### 2.3 Bước 3 — Tiền xử lý

**Mục tiêu:** Chuẩn hóa dữ liệu để tăng tốc độ hội tụ khi huấn luyện.

**Công việc:**
- Chuẩn hóa sử dụng mean và std đã tính
- Áp dụng transforms: `transforms.Normalize(mean=0.286, std=0.353)`
- Tắt data augmentation (Giai đoạn 1)

**Cấu hình:**
```python
Normalize(mean=0.2863, std=0.3540)
```

### 2.4 Bước 4 — Chia dữ liệu

**Mục tiêu:** Chia dữ liệu thành tập huấn luyện, tập validation và tập kiểm tra.

**Công việc:**
- Tỷ lệ chia: Huấn luyện 90% / Validation 10% / Kiểm tra 100% (tách riêng)
- Sử dụng phương pháp chia stratified để giữ nguyên phân bố lớp
- Đặt random seed = 42 để đảm bảo reproducibility
- Lưu thông tin chia dữ liệu

**Kết quả:**
| Tập | Số mẫu | Số batch (bs=64) |
|------|--------|------------------|
| Huấn luyện | 54,000 | 844 |
| Validation | 6,000 | 94 |
| Kiểm tra | 10,000 | 157 |

**Triển khai:** `src/data/split.py` → `DataSplitManager`

---

## 3. Giai đoạn 2: Phát triển mô hình

### 3.1 Bước 1 — Mô hình Baseline

**Mục tiêu:** Thiết lập baseline đơn giản để so sánh.

**Mô hình:** BaselineMLP
```python
BaselineMLP:
  Linear(784 → 256)
  ReLU
  Linear(256 → 128)
  ReLU
  Linear(128 → 10)
```

### 3.2 Bước 2 — Khám phá kiến trúc

**Mục tiêu:** So sánh nhiều kiến trúc để tìm mô hình tốt nhất.

**Các mô hình đã triển khai:**

| Mô hình | Kiến trúc | Tham số |
|---------|-----------|---------|
| DeepMLP | 784→512→256→128→10 + Dropout | ~500K |
| SimpleCNN | 2 tầng Conv + 2 tầng FC | ~100K |
| ResNetMini | Residual blocks + skip connections | ~200K |

**Triển khai:** `src/models/model_factory.py` → `ModelFactory`

### 3.3 Bước 3 — Tinh chỉnh siêu tham số

**Mục tiêu:** Tìm siêu tham số tối ưu cho quá trình huấn luyện.

**Chiến lược tìm kiếm:**
- Grid search + Random search
- 8 trials × 3 epochs mỗi trial

**Các siêu tham số đã thử:**
| Tham số | Giá trị |
|---------|---------|
| Learning rate | 0.001, 0.0001 |
| Optimizer | Adam, SGD |
| Batch size | 64, 128 |
| Dropout | 0.3, 0.5 |

**Cấu hình tốt nhất tìm được:**
```yaml
model_name: simple_cnn
learning_rate: 0.001
optimizer: adam
batch_size: 128
dropout: 0.5
```

**Triển khai:** `src/training/hyperparameter_tuning.py`

### 3.4 Bước 4 — Chọn mô hình

**Mục tiêu:** Chọn mô hình tốt nhất dựa trên độ chính xác validation.

**Tiêu chí chọn:**
- Độ chính xác validation cao nhất
- Kích thước mô hình hợp lý
- Thời gian huấn luyện phù hợp

**Người chiến thắng:** SimpleCNN (val_acc ~90%)

---

## 4. Giai đoạn 3: Huấn luyện và đánh giá

### 4.1 Bước 1 — Pipeline huấn luyện

**Mục tiêu:** Huấn luyện mô hình với giám sát và lưu checkpoint phù hợp.

**Vòng lặp huấn luyện:**
```
for epoch in range(epochs):
    # Giai đoạn huấn luyện
    model.train()
    for batch in train_loader:
        data, target = batch
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # Giai đoạn validation
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            data, target = batch
            output = model(data)
            # Tính toán metrics
```

**Callbacks:**
- **EarlyStopping:** Theo dõi val_loss, patience=5, min_delta=0.0001
- **Checkpoint:** Lưu trọng số mô hình tốt nhất và cuối cùng

**Cấu hình (Lần chạy cuối):**
| Tham số | Giá trị |
|---------|---------|
| Epochs | 15 |
| Batch size | 128 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | CrossEntropy |
| Dropout | 0.5 |
| Device | CPU |

**Triển khai:** `src/training/trainer.py`, `src/training/callbacks.py`

### 4.2 Bước 2 — Trực quan hóa

**Mục tiêu:** Trực quan hóa tiến trình huấn luyện.

**Biểu đồ đã tạo:**
- Đồ thị loss (huấn luyện vs validation)
- Đồ thị accuracy (huấn luyện vs validation)

**Kết quả:** `results/runs/loss_curve.png`, `results/runs/accuracy_curve.png`

### 4.3 Bước 3 — Đánh giá

**Mục tiêu:** Đánh giá mô hình đã huấn luyện trên tập kiểm tra.

**Các chỉ số đã tính:**
| Chỉ số | Công thức | Giá trị |
|--------|-----------|---------|
| Accuracy | Correct / Total | 92.10% |
| Precision (macro) | Trung bình precision mỗi lớp | 92.11% |
| Recall (macro) | Trung bình recall mỗi lớp | 92.10% |
| F1-Score (macro) | 2 × P × R / (P + R) | 92.09% |

**Triển khai:** `src/evaluation/evaluator.py`

### 4.4 Bước 4 — Phân tích lỗi

**Mục tiêu:** Hiểu các trường hợp phân loại sai của mô hình.

**Phân tích:**
- Trực quan hóa ma trận nhầm lẫn
- Các cặp lớp hay bị nhầm lẫn nhất
- Độ chính xác theo từng lớp

**Các cặp hay bị nhầm lẫn nhất:**
| Lớp thực | → Dự đoán | Số lần |
|-----------|----------|---------|
| Shirt | T-shirt/top | 95 |
| T-shirt/top | Shirt | 90 |
| Shirt | Pullover | 64 |
| Coat | Pullover | 63 |
| Coat | Shirt | 55 |

**Nhận xét:** Shirt và T-shirt/top hay bị nhầm lẫn do hình dạng tương tự.

**Triển khai:** `src/evaluation/error_analysis.py`, `src/evaluation/confusion_matrix.py`

### 4.5 Bước 5 — Trực quan hóa dự đoán

**Mục tiêu:** Trực quan hóa dự đoán của mô hình so với nhãn thực.

**Kết quả:** `results/runs/predictions.png`
- Lưới các ảnh mẫu
- Mã màu: xanh lá = đúng, đỏ = sai
- Hiển thị cả nhãn thực và nhãn dự đoán

**Triển khai:** `src/evaluation/visualizer.py`

### 4.6 Bước 6 — Lưu và tải mô hình

**Mục tiêu:** Lưu và tải các mô hình đã huấn luyện.

**Các artifact đã lưu:**
| File | Mô tả |
|------|-------|
| `trained_model.pt` | Trọng số mô hình tốt nhất |
| `final_model.pt` | Mô hình cuối cùng kèm metrics |
| `training_history.json` | Metrics theo từng epoch |

**Triển khai:** `src/evaluation/model_io.py`

---

## 5. Tổng kết kết quả

### 5.1 Hiệu suất cuối cùng

| Chỉ số | Huấn luyện | Validation | Kiểm tra |
|--------|------------|------------|----------|
| Accuracy | 94.72% | 92.80% | **92.10%** |
| Best Epoch | — | 11/15 | — |

### 5.2 Tiến trình huấn luyện

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|------------|----------|----------|---------|
| 1 | 0.585 | 0.351 | 79.03% | 87.08% |
| 5 | 0.265 | 0.237 | 90.41% | 91.42% |
| 10 | 0.190 | 0.222 | 92.91% | 92.63% |
| 15 | 0.141 | 0.230 | 94.72% | 92.80% |

### 5.3 Các thành tựu chính

✅ **Mục tiêu độ chính xác (90%) đã vượt qua: 92.10%**  
✅ Tất cả 27 unit tests đều pass  
✅ Pipeline hoàn chỉnh: Dữ liệu → Huấn luyện → Đánh giá → Báo cáo  
✅ Mã nguồn mở: GitHub repository

---

## 6. Cấu trúc dự án

```
e:\lab1/
├── src/
│   ├── data/                    # Giai đoạn 1: Chuẩn bị dữ liệu
│   │   ├── dataset.py          # FashionMNIST loader
│   │   ├── eda.py             # Phân tích khám phá
│   │   ├── preprocessing.py    # Làm sạch & transforms
│   │   └── split.py           # Chia train/val/test
│   ├── models/                 # Giai đoạn 2: Kiến trúc mô hình
│   │   ├── architectures/      # MLP, CNN, ResNet
│   │   └── model_factory.py    # Model registry
│   ├── training/               # Giai đoạn 3: Huấn luyện
│   │   ├── trainer.py         # Vòng lặp huấn luyện
│   │   ├── callbacks.py       # Early stopping, checkpoint
│   │   ├── losses.py          # Loss functions
│   │   ├── optimizers.py      # Optimizer factory
│   │   └── visualizer.py      # Biểu đồ huấn luyện
│   └── evaluation/             # Giai đoạn 3: Đánh giá
│       ├── evaluator.py       # Tính toán metrics
│       ├── error_analysis.py   # Phân tích confusion
│       ├── confusion_matrix.py # Trực quan hóa CM
│       ├── visualizer.py      # Hiển thị dự đoán
│       └── model_io.py        # Lưu/tải mô hình
├── tests/training/             # Unit tests (27 tests)
├── models/checkpoints/        # Mô hình đã lưu
├── results/runs/              # Biểu đồ và kết quả
├── docs/reports/              # Báo cáo các giai đoạn
├── configs/                   # Cấu hình YAML
├── main.py                    # Entry point huấn luyện
└── README.md                  # Tài liệu dự án
```

---

## Cách tái tạo dự án

### 1. Clone repository
```bash
git clone https://github.com/Sper-il/Lab1.git
cd Lab1
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Chạy toàn bộ pipeline
```bash
# Huấn luyện
python main.py

# Đánh giá
$env:PYTHONPATH = "."
python src/evaluation/run_evalution.py
```

### 4. Chạy tests
```bash
pytest tests/training -v
```

---

## Kết luận

Dự án này minh họa một workflow machine learning hoàn chỉnh:

1. **Chuẩn bị dữ liệu** → Làm sạch, tiền xử lý, chia dữ liệu
2. **Phát triển mô hình** → Thiết kế, so sánh, chọn mô hình
3. **Tinh chỉnh siêu tham số** → Tối ưu các tham số huấn luyện
4. **Huấn luyện** → Huấn luyện với giám sát và checkpoint
5. **Đánh giá** → Tính metrics, phân tích lỗi

**Kết quả cuối cùng:** Độ chính xác 92.10% trên tập kiểm tra FashionMNIST, vượt mục tiêu 90%.

---

*Tài liệu được tạo: 2026-10-15*

# Practice 1 — PyTorch FashionMNIST Classification

> **Phân loại ảnh grayscale 28×28 từ FashionMNIST vào 10 lớp**, end-to-end PyTorch pipeline
> (load → transform → CNN → train → evaluate → save/load → visualize). Mục tiêu chính là **học**,
> không benchmark.

## 1. Project Overview

Practice 1 là bài tập nền tảng PyTorch cho người mới: xây dựng pipeline phân loại ảnh thời
trang từ dataset **FashionMNIST** (10 lớp: T-shirt, Trouser, Pullover, Dress, Coat, Sandal,
Shirt, Sneaker, Bag, Ankle boot). Mỗi ảnh là grayscale 28×28, pipeline từ raw `.png` đến
model `.pt` đều nằm trong project.

Project dành cho **sinh viên đang học PyTorch end-to-end**. Mục tiêu là hiểu rõ từng bước
(forward, loss, backward, optimize), không tối ưu accuracy tuyệt đối.

**Tại sao quan trọng:**
- FashionMNIST khó hơn MNIST nên phân biệt được model yếu / mạnh.
- 70.000 ảnh đủ lớn để CNN baseline đạt ≥88% accuracy.
- Cover đủ 7 mục trong "Deliver" của đề: code, report, loss graphs, image displays,
  experiment hyperparameters, save/load, official PyTorch docs.

## 2. Problem Statement

**Bài toán:** Cho ảnh grayscale 28×28 từ FashionMNIST, dự đoán 1 trong 10 lớp thời trang.

**Định lượng:**
- **Input:** tensor `[1, 28, 28]`, giá trị pixel [0, 255] (uint8) → [0, 1] sau ToTensor
  → chuẩn hóa về `mean=0.2860, std=0.3530`.
- **Output:** class index (0–9) + vector xác suất 10 chiều.
- **Tiêu chí:** test accuracy ≥ 88%, training loss giảm đều, training xong trong < 10 phút
  trên CPU.

**Nỗi đau giải quyết:** MNIST quá dễ (baseline 99%), không phân biệt được model. FashionMNIST
có cấu trúc tương tự (grayscale, 28×28, 10 lớp) nhưng khó hơn nên buộc người học phải hiểu
CNN mới đạt baseline.

## 3. Main Features

| Độ ưu tiên | Tính năng | Lý do |
|------------|-----------|-------|
| **Must** | Load FashionMNIST + split train/val/test | Bắt buộc theo đề. |
| **Must** | Apply transform (ToTensor + Normalize) | Bắt buộc theo đề. |
| **Must** | Build neural network (CNN 2 conv + 2 FC) | Bắt buộc theo đề. |
| **Must** | Training loop (forward, loss, backward, optimize) | Bắt buộc theo đề. |
| **Must** | Evaluate model accuracy trên test set | Bắt buộc theo đề. |
| **Must** | Save & load model roundtrip | Bắt buộc theo đề. |
| **Should** | Visualize loss curve (PNG) | Giúp debug training, đề yêu cầu. |
| **Should** | Display predicted vs actual images (PNG) | Đề yêu cầu, trực quan. |
| **Could** | Experiment hyperparameters (lr × optimizer) | Đề yêu cầu "Tasks", Phase 2. |
| **Could** | Confusion matrix | Phân tích lỗi, không bắt buộc. |

## 4. System Architecture

```
[User] → [Sinh data: split.py] → [data/raw/*.png]
                ↓
        [DataLoader: dataset.py]
                ↓
        [FashionCNN: fashion_cnn.py]
                ↓
        [Trainer: train.py (M3)]
                ↓
        [Evaluator: evaluate.py (M4)]
                ↓
        [Plot: plot.py (M5)]
                ↓
        [results/runs/<ts>/: metrics.json + model.pt + *.png]
```

Chi tiết hơn xem [`docs/architecture.md`](docs/architecture.md) và [`docs/data-flow.md`](docs/data-flow.md).

## 5. Folder Structure

```
lab1/
├── configs/baseline.yaml          # Hyperparameter (data, model, train)
├── data/raw/                      # 70.000 ảnh .png (gitignore, M1 sinh)
│   ├── train/<class>/*.png        # 50.000
│   ├── val/<class>/*.png          # 10.000
│   └── test/<class>/*.png         # 10.000
├── data/README.md                 # Hướng dẫn sinh data
├── src/
│   ├── data/                      # M1 ✅
│   │   ├── split.py               # split_fashionmnist_to_disk()
│   │   └── dataset.py             # build_dataloaders(), summarize()
│   ├── models/                    # M2 ⏳ (sẽ chứa fashion_cnn.py)
│   ├── training/                  # M3 ⏳
│   ├── evaluation/                # M4 ⏳
│   └── visualization/             # M5 ⏳
├── tests/                         # pytest, 1 file / module
│   └── test_dataset.py            # 11/11 pass
├── docs/
│   ├── architecture.md            # Kiến trúc tổng quan
│   ├── data-flow.md               # Luồng dữ liệu chi tiết
│   ├── api.md                     # API nội bộ từng hàm
│   ├── AGENTS.md                      # Working rules cho AI agent
│   └── AI_Usage.md                    # 7 mục theo template
│
├── results/runs/<ts>/             # Output mỗi lần chạy (M3+)
├── progress_status.md             # Single source of truth
├── CHANGELOG.md                   # Model/dataset version
├── README.md                      # File này (10 mục chuẩn)
├── requirements.txt
└── .gitignore
```

## 6. Installation / Run / Test

```bash
# 1. Clone (hoặc copy folder)
cd E:\lab1

# 2. Tạo virtualenv
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Sinh dữ liệu trên disk (1 lần, ~2-3 phút)
python -m src.data.split --root data/raw --val-split 10000 --seed 42

# 5. Kiểm tra 3 DataLoader
python -c "from src.data.dataset import build_dataloaders, summarize; loaders = build_dataloaders('configs/baseline.yaml'); [summarize(l, n) for n, l in loaders.items()]"

# 6. Test
pytest -v                          # tất cả test
pytest tests/test_dataset.py -v    # chỉ test data (11 test)

# 7. Train (sau khi M2-M5 sẵn sàng)
# python -m src.train --config configs/baseline.yaml
```

## 7. Example Input → Output

**Input** (1 sample từ `data/raw/test/8/Bag_000123.png`):

```
Ảnh grayscale 28×28, pixel uint8 [0, 255], thuộc class 8 (Bag).
```

**Sau transform** (bước 2 trong pipeline):

```python
tensor([[[-0.81, -0.78, ..., 0.32],
         ...
         [ 0.12,  0.45, ...,-0.34]]])   # shape [1, 28, 28], float32
# pixel range sau Normalize(0.2860, 0.3530): khoảng [-0.81, 2.02]
```

**Sau DataLoader** (gom 64 ảnh thành 1 batch):

```python
batch_images.shape == [64, 1, 28, 28]   # float32
batch_labels.shape == [64]              # int64, mỗi giá trị trong [0, 9]
```

**Sau FashionCNN** (M2 — chưa code, ví dụ minh họa):

```python
logits.shape == [64, 10]                # float32, chưa qua softmax
# logits cho 1 ảnh Bag: [0.5, -1.2, 0.8, -0.3, 0.1, -2.0, 0.6, -1.5, 2.3, -0.7]
# argmax → class 8 (Bag) ✅
```

**Sau evaluate** (M4 — chưa code):

```json
{
  "test_accuracy": 0.91,
  "per_class_accuracy": {
    "T-shirt": 0.87, "Trouser": 0.98, "Pullover": 0.89, ...
  }
}
```

## 8. Results

> Chưa có kết quả training. Sẽ cập nhật sau M3 (training) + M4 (evaluation) + M5 (visualization).

**Lần chạy mới nhất (M1 — data only):**

| Metric | Target | Đạt được | Run |
|--------|--------|----------|-----|
| Số ảnh train | 50.000 | ✅ 50.000 | 2026-10-11 |
| Số ảnh val | 10.000 | ✅ 10.000 | 2026-10-11 |
| Số ảnh test | 10.000 | ✅ 10.000 | 2026-10-11 |
| Batch shape | `[64, 1, 28, 28]` | ✅ `[64, 1, 28, 28]` | 2026-10-11 |
| Test pass | 11/11 | ✅ 11/11 | 2026-10-11 |
| Test accuracy | ≥ 88% | ⏳ (M4) | — |
| Training time | < 10 phút trên CPU | ⏳ (M3) | — |

## 9. Limitations

- **Chỉ grayscale 28×28** — FashionMNIST gốc. Không hỗ trợ ảnh RGB, không hỗ trợ ảnh
  kích thước khác.
- **CPU training chậm** — Baseline 10 epoch trên CPU mất ~5-8 phút. Không có GPU acceleration
  tự động cho mini-batch lớn hơn.
- **Không có augmentation** ở M1. M6 sẽ thử thêm nếu cần.
- **Không có stratified split** — 10 lớp đều (6k train, 1k val, 1k test) nên random split
  cho kết quả tương đương.
- **Không có learning rate scheduler** ở baseline. Có thể thêm `ReduceLROnPlateau` ở M6.
- **Không có early stopping** — fix 10 epoch, save best model theo val acc.
- **Không có class-weighted loss** — 10 lớp đều nên không cần.
- **Không có model registry** — mỗi lần train ghi đè lên `results/runs/<ts>/`, dùng
  timestamp để phân biệt.

## 10. Future Improvements

| Cải tiến | Lý do | Module |
|----------|-------|--------|
| Thêm data augmentation (RandomHorizontalFlip, RandomCrop) | Tăng accuracy ~1-2% | M6 |
| Thử ResNet-18 thay CNN đơn giản | So sánh baseline vs transfer learning | M6+ |
| Export model sang ONNX | Inference nhanh hơn, chạy edge device | Sau M5 |
| Thêm TensorBoard / W&B tracking | Trực quan hóa training real-time | M6 |
| Thêm learning rate scheduler (CosineAnnealing) | Tăng accuracy ~0.5% | M6 |
| Thêm early stopping (patience=3) | Tránh overfit, tiết kiệm thời gian | M6 |
| Thêm confusion matrix + per-class F1 | Phân tích lỗi chi tiết | M4+ |
| Thêm API RESTful (FastAPI) | Serve model qua HTTP | Sau M5 |
| Thêm Docker + docker-compose | Reproducible môi trường | Sau M5 |
| Thêm CI (GitHub Actions) chạy pytest | Đảm bảo code luôn pass | Sau M5 |

---

**Liên kết hữu ích:**

- [`progress_status.md`](progress_status.md) — nhật ký theo vòng lặp 7 bước, decisions log, risks.
- [`docs/architecture.md`](docs/architecture.md) — kiến trúc 5 module, design conventions.
- [`docs/data-flow.md`](docs/data-flow.md) — luồng dữ liệu end-to-end.
- [`docs/api.md`](docs/api.md) — API nội bộ từng hàm.
- [`AI_Usage.md`](AI_Usage.md) — ghi nhận việc dùng AI.
- [`CHANGELOG.md`](CHANGELOG.md) — model/dataset version.
- [`AGENTS.md`](AGENTS.md) — working rules cho AI agent.
- [`data/README.md`](data/README.md) — hướng dẫn sinh dữ liệu.

**Tài liệu PyTorch tham khảo:**

- [Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Build Model](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [Autograd](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [Optimization](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Save / Load Model](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)

**License:** Dự án phục vụ mục đích học tập (Practice 1).

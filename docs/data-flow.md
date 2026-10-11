# Data Flow

Sơ đồ luồng dữ liệu chi tiết qua toàn bộ pipeline, từ file dataset thô
đến kết quả cuối cùng (metrics + ảnh).

## Quy trình 2 bước cho data (M1)

M1 tách thành 2 bước rõ ràng: **split (1 lần)** rồi **transform (mỗi lần load)**.
Lý do: tránh data leakage, augmentation khác nhau giữa train vs val/test,
reproducible trên disk.

### Bước 1 — Split (chạy 1 lần, sinh `.png` ra disk)

```
torchvision.datasets.FashionMNIST
  (60.000 train + 10.000 test, ảnh 28x28 grayscale, label 0–9)
        │
        │  download=True, cache tại data/raw/_torchvision_cache/
        ▼
split_fashionmnist_to_disk(root="data/raw", val_split=10000, seed=42)
        │
        │  random.shuffle(indices) với seed=42
        ▼
┌──────────────────────────┬──────────────────────────┬──────────────────────────┐
│ data/raw/train/          │ data/raw/val/            │ data/raw/test/           │
│ 50.000 ảnh .png          │ 10.000 ảnh .png          │ 10.000 ảnh .png          │
│ tên: <class_name>_<idx>  │ tên: <class_name>_<idx>  │ tên: <class_name>_<idx>  │
└──────────────────────────┴──────────────────────────┴──────────────────────────┘
```

Mỗi file có dạng `data/raw/{split}/<class_id>/<class_name>_<original_idx>.png`,
ví dụ: `data/raw/train/0/T-shirt_top_000001.png`.

### Bước 2 — Transform (mỗi lần load, on-the-fly)

```
data/raw/train/  ─┐
data/raw/val/    ─┼── ImageFolder ── transforms.Compose ── DataLoader
data/raw/test/   ─┘
```

**Transform pipeline:**

| Tập | Augmentation | ToTensor | Normalize |
|-----|--------------|----------|-----------|
| train | (chưa có ở M1, M6 sẽ thử) | ✅ | mean=0.2860, std=0.3530 |
| val | — | ✅ | mean=0.2860, std=0.3530 |
| test | — | ✅ | mean=0.2860, std=0.3530 |

**Mean/std giải thích:** FashionMNIST có pixel gốc uint8 [0, 255]. Sau khi chia cho
255 (`ToTensor`) → `[0, 1]`. Sau `Normalize(mean=0.2860, std=0.3530)` → khoảng `[-0.81, 2.02]`
cho phần lớn pixel (mean và std thực nghiệm của toàn bộ training set).

**DataLoader config:**

| Tập | batch_size | shuffle | num_workers | pin_memory |
|-----|------------|---------|-------------|------------|
| train | 64 | True | 2 | True nếu có CUDA |
| val | 64 | False | 2 | True nếu có CUDA |
| test | 64 | False | 2 | True nếu có CUDA |

→ `train` có 782 batch (50.000 / 64), `val` 157 batch, `test` 157 batch.

## Sơ đồ end-to-end (qua M1 → M5)

```
BƯỚC 1 — SPLIT (M1, 1 lần)
torchvision FashionMNIST (28×28 grayscale)
        │
        ▼
split_fashionmnist_to_disk(seed=42)
        │
        ▼
data/raw/train/<class>/<idx>.png   (50.000 ảnh)
data/raw/val/<class>/<idx>.png     (10.000 ảnh)
data/raw/test/<class>/<idx>.png    (10.000 ảnh)

BƯỚC 2 — TRANSFORM + DATALOADER (M1, mỗi lần chạy)
ImageFolder + transforms.Compose
   train: Grayscale → ToTensor → Normalize(0.2860, 0.3530)
   val:   Grayscale → ToTensor → Normalize(0.2860, 0.3530)
   test:  Grayscale → ToTensor → Normalize(0.2860, 0.3530)
        │
        ▼
DataLoader (batch_size=64, shuffle train, num_workers=2)
        │
        ▼
batch shape: [64, 1, 28, 28] float32, labels [64] int64

BƯỚC 3 — FORWARD (M2 — chưa code)
FashionCNN (2 conv 3x3 + 2 FC + Dropout)
        │
        ▼
logits shape: [64, 10] float32

BƯỚC 4 — LOSS + OPTIMIZER (M3 — chưa code)
CrossEntropyLoss(logits, labels) → scalar loss
        │
        ▼
Adam.step() (lr=0.001, betas=(0.9, 0.999)) → cập nhật weights

BƯỚC 5 — VALIDATE mỗi epoch (M3)
validate() trên val_loader (không cập nhật weights)
        │
        ▼
lưu (train_loss, train_acc, val_loss, val_acc) vào history dict

BƯỚC 6 — TEST cuối (M4 — chưa code)
test_accuracy() trên test_loader
        │
        ▼
float (% accuracy trên 10.000 ảnh test)

BƯỚC 7 — VISUALIZE (M5 — chưa code)
plot_loss_curve(history)  → results/runs/<ts>/loss_curve.png
show_predictions(model, n=10)  → results/runs/<ts>/predictions.png
        │
        ▼
results/runs/<timestamp>/
   ├── metrics.json      # history + test_accuracy
   ├── model.pt          # best model state_dict
   ├── loss_curve.png    # biểu đồ loss
   └── predictions.png   # 10 ảnh + nhãn dự đoán vs thật
```

## Ví dụ cụ thể 1 ảnh

```
Input:   data/raw/train/8/Bag_000123.png     (ảnh túi xách, label gốc = 8)
            │
            ▼ ImageFolder đọc file
PIL:     Image 28×28, mode='L', pixel uint8 [0, 255]
            │
            ▼ transforms.Compose (train transform)
Tensor:  shape [1, 28, 28], dtype float32
         giá trị sau Normalize(0.2860, 0.3530): khoảng [-0.81, 2.02]
            │
            ▼ DataLoader gom 64 ảnh thành 1 batch
Batch:   shape [64, 1, 28, 28], labels [64] int64
            │
            ▼ FashionCNN.forward()
Logits:  shape [64, 10], vd [2.3, -0.5, 0.1, ..., 1.2]
            │
            ▼ CrossEntropyLoss
Loss:    scalar, vd 1.85
            │
            ▼ argmax(logits, dim=1) → class dự đoán
Pred:    vd 8 (đúng)
```

## Tại sao tách bước 1 và bước 2?

| Vấn đề | Cách cũ (1 bước) | Cách M1 (2 bước) |
|--------|-------------------|-------------------|
| Data leakage | Tính mean/std toàn bộ → split → vẫn có thể leak | Split ảnh gốc ra disk trước → mean/std là cố định từ literature |
| Augmentation train khác val | Phải dùng Subset với transform chung | Mỗi tập ImageFolder có transform riêng |
| Reproducible | Phải cùng seed, cùng torch version | 70k file cố định, hash có thể kiểm |
| Debug | Phải in tensor | Mở folder bằng File Explorer xem ảnh |
| Tốn RAM | Toàn bộ tensor trong RAM | Ảnh đọc on-the-fly, RAM chỉ chứa 1 batch |
| Tốn disk | Ít | 70k file PNG ~140 MB (chấp nhận được) |

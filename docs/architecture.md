# Architecture

Practice 1 — FashionMNIST Classification. Tài liệu này mô tả kiến trúc phần mềm
ở mức **module** (5 module theo skill `ai-fusion-workflow`).

## Mục tiêu thiết kế

- Tách rời 5 phần việc: data, model, training, evaluation, visualization.
- Mỗi phần **tự test được** và **chạy được độc lập**.
- Config truyền từ YAML, code không hard-code hyperparameter.
- Tái sử dụng được cho mọi biến thể baseline / experiment (M5+).

## Sơ đồ tổng quan

```
┌─────────────────────────────────────────────────────────────────────┐
│                       configs/baseline.yaml                          │
│   (data.root, batch_size, num_workers, mean, std, val_split, seed)  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────┐   ┌──────────────────┐
│  src/data/       │   │  src/data/       │
│   split.py       │──▶│   dataset.py     │──▶ {train, val, test} DataLoaders
│ (chạy 1 lần)    │   │  (mỗi lần run)  │
└──────────────────┘   └──────────────────┘
         │                       │
         ▼                       ▼
  70.000 file .png         ┌──────────────────┐
  data/raw/{train,         │  src/models/     │
   val,test}/<class>/      │   fashion_cnn.py │──▶ nn.Module
   <name>_<idx>.png        └──────────────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │  src/training/   │──▶ history dict + best model
                           │   train.py       │    (loss/acc mỗi epoch)
                           └──────────────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │  src/evaluation/ │──▶ test_accuracy: float
                           │   evaluate.py    │
                           └──────────────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │  src/visualiz./  │──▶ results/runs/<ts>/
                           │   plot.py        │    metrics.json + ảnh PNG
                           └──────────────────┘
```

## 5 module chính

| # | Module | Đường dẫn | Trạng thái | Mục đích |
|---|--------|-----------|------------|----------|
| M0 | Scaffold | `src/`, `tests/`, `configs/`, `requirements.txt` | ✅ | Khung thư mục, 6 `__init__.py`, config YAML, requirements. |
| M1 | Data | `src/data/{split,dataset}.py` | ✅ | Tải FashionMNIST, split ra `.png` trên disk, build 3 DataLoader. |
| M2 | Model | `src/models/fashion_cnn.py` | ⏳ | `FashionCNN` 2 conv + 2 FC, đếm params, forward shape `[B,10]`. |
| M3 | Training | `src/training/train.py` | ⏳ | Loop epoch, validate, lưu best model + history. |
| M4 | Evaluation | `src/evaluation/evaluate.py` | ⏳ | `test_accuracy` trên tập test. |
| M5 | Visualization | `src/visualization/plot.py` | ⏳ | Vẽ loss curve, show_predictions. |

## Quy ước thiết kế

### 1. Mỗi hàm trả về dict hoặc dataclass — không in lung tung
Mọi hàm public trong `src/` trả về cấu trúc dữ liệu rõ ràng. Logging dùng `logger`
của module, không dùng `print()` trừ khi cần kết quả tóm tắt (vd `summarize()`).

### 2. Config từ YAML, code không hard-code
- Mọi giá trị thay đổi được (batch_size, mean, std, lr, optimizer, …) đều đọc từ
  `configs/baseline.yaml`.
- Code chỉ fallback về default khi key không tồn tại (vd `data.mean`).

### 3. Reproducibility
- `seed=42` cho cả `random.split` (M1) và `torch.manual_seed` (M3+).
- Split trên disk (M1) đảm bảo 3 tập giống nhau qua các lần chạy.

### 4. Tách split và transform (chống data leakage)
- M1 ghi 70.000 file `.png` ra `data/raw/{train,val,test}/<class>/` 1 lần duy nhất.
- Transform (`ToTensor + Normalize`) áp dụng riêng từng tập khi load.
- Ưu điểm: val/test có thể dùng augmentation khác train; mỗi tập có thể có
  stats riêng; reproducible trên disk.

### 5. PyTorch convention
- `nn.Module` subclassing cho model.
- Subclass `Subset` / `ImageFolder` cho dataset.
- `DataLoader` với `pin_memory` auto-detect CUDA.
- Model chạy được cả CPU lẫn GPU; device chọn qua `torch.device`.

## Cấu trúc thư mục hiện tại

```
E:\lab1\
├── configs/
│   └── baseline.yaml          # Mọi hyperparameter
├── data/
│   └── raw/
│       ├── train/<class>/*.png   # 50.000 ảnh (sinh bởi M1)
│       ├── val/<class>/*.png     # 10.000 ảnh
│       └── test/<class>/*.png    # 10.000 ảnh
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── split.py          # split_fashionmnist_to_disk()
│   │   └── dataset.py        # build_dataloaders(), summarize()
│   ├── models/
│   │   └── __init__.py       # (M2 sẽ thêm fashion_cnn.py)
│   ├── training/
│   │   └── __init__.py       # (M3)
│   ├── evaluation/
│   │   └── __init__.py       # (M4)
│   └── visualization/
│       └── __init__.py       # (M5)
├── tests/
│   └── test_dataset.py       # 11 test cho M1
├── docs/
│   ├── architecture.md       # File này
│   ├── data-flow.md          # Luồng dữ liệu chi tiết
│   └── api.md                # API nội bộ
├── progress_status.md        # Nhật ký module theo vòng lặp
├── requirements.txt
├── .gitignore
└── README.md
```

## Vòng lặp phát triển (theo skill ai-fusion-workflow)

Mỗi module M2 → M5 đều đi qua 7 bước:

1. **Describe** — mô tả trong `progress_status.md`
2. **Generate** — viết code
3. **Review** — self-check logic
4. **Run** — chạy thực tế, in kết quả
5. **Test** — pytest
6. **Refactor** — sửa warning / code smell
7. **Commit** — Conventional Commit, cập nhật `progress_status.md`

## Quyết định kiến trúc (liên kết Decisions Log trong progress_status.md)

- **5 module tách rời** thay vì 1 file lớn — dễ test, dễ đổi từng phần.
- **Split trước, transform sau** — chống data leakage.
- **Ảnh `.png` trên disk** — reproducible, debug được, không tốn RAM.
- **`ImageFolder` + `transforms.Compose`** — chuẩn torchvision, ít code tự viết.
- **Cấu hình qua YAML** — thay đổi không cần sửa code.
- **Run experiment grid (lr × optimizer) ở M6** — Phase 2.

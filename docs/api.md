# API nội bộ

Tài liệu các hàm public trong `src/`. Mỗi hàm có **input / output / side effect** rõ ràng.

## Quy ước

- Hàm trả về `dict` hoặc `dataclass` — không in lung tung bằng `print()` (trừ helper
  tóm tắt như `summarize()`).
- Logging qua `logger = logging.getLogger(__name__)`.
- Mọi hàm có docstring mô tả Args / Returns / Raises.
- Type hint đầy đủ.

---

## M1 — Data module

### `src/data/split.py`

#### `split_fashionmnist_to_disk(root, val_split=10000, seed=42) -> Path`

Tải FashionMNIST, chia train/val/test, lưu tất cả ra file `.png` trên disk.

| | |
|---|---|
| **Input** | `root: str \| Path` — thư mục gốc (sẽ chứa 3 subfolder `train/val/test`)<br>`val_split: int = 10000` — số mẫu validation lấy từ train gốc<br>`seed: int = 42` — seed cho random split, reproducible |
| **Output** | `Path` — đường dẫn root sau khi ghi xong |
| **Side effect** | Tạo 70.000 file `.png` vào `<root>/{train,val,test}/<class_id>/`<br>Cache của torchvision tại `<root>/_torchvision_cache/` |
| **Idempotent** | Có — nếu đủ 50k/10k/10k ảnh thì skip, không ghi lại |
| **Raises** | `ValueError` nếu `val_split < 0` hoặc `val_split > 60000` |

**CLI:**
```bash
python -m src.data.split --root data/raw --val-split 10000 --seed 42
```

**Cấu trúc output:**
```
data/raw/
├── train/<class_id>/<class_name>_<idx>.png   # 50.000 file
├── val/<class_id>/<class_name>_<idx>.png     # 10.000 file
├── test/<class_id>/<class_name>_<idx>.png    # 10.000 file
└── _torchvision_cache/                       # cache nội bộ của torchvision
```

---

### `src/data/dataset.py`

#### `build_dataloaders(config_path="configs/baseline.yaml") -> Dict[str, DataLoader]`

Đọc config YAML, build 3 DataLoader từ cấu trúc thư mục `.png` đã sinh bởi
`split_fashionmnist_to_disk`.

| | |
|---|---|
| **Input** | `config_path: str \| Path = "configs/baseline.yaml"` |
| **Output** | `dict` với key `"train"`, `"val"`, `"test"` → `DataLoader` |
| **Side effect** | Không (chỉ đọc file, không ghi) |
| **Raises** | `FileNotFoundError` nếu config không tồn tại<br>`FileNotFoundError` nếu thư mục `.png` chưa được sinh (chưa chạy split) |

**Transform pipeline:**

| Tập | Augmentation | ToTensor | Normalize |
|-----|--------------|----------|-----------|
| train | (chưa ở M1) | ✅ | mean=0.2860, std=0.3530 |
| val | — | ✅ | mean=0.2860, std=0.3530 |
| test | — | ✅ | mean=0.2860, std=0.3530 |

**Batch shape:** `[64, 1, 28, 28]` float32, labels `[64]` int64.

**Config keys đọc từ `configs/baseline.yaml`:**
```yaml
data:
  root: data/raw              # thư mục chứa train/val/test
  batch_size: 64
  num_workers: 2
  val_split: 10000
  mean: 0.2860
  std: 0.3530
  seed: 42
```

#### `summarize(loader, name) -> None`

In thông tin 1 loader (số mẫu, số batch, classes, shape, dtype, pixel range).
Dùng để verify nhanh sau khi build.

**Output (ví dụ):**
```
[train]
  dataset size: 50000
  num batches:  782
  classes:      ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  batch shape:  images=(64, 1, 28, 28), labels=(64,)
  dtypes:       images=torch.float32, labels=torch.int64
  pixel range:  min=-0.8102, max=2.0227
  label range:  min=0, max=9
```

---

## M2 — Model module (chưa code)

Dự kiến API khi M2 triển khai (Phase 1+2 đã mô tả, chưa Generate).

### `src/models/fashion_cnn.py`

#### `class FashionCNN(nn.Module)`

CNN 2 lớp conv + 2 FC.

| | |
|---|---|
| **Input forward** | `x: Tensor` shape `[B, 1, 28, 28]` float32 |
| **Output forward** | `logits: Tensor` shape `[B, 10]` float32 |
| **Params** | ~225.000 (cố định theo config) |
| **Layer** | `Conv2d(1→32, 3) → ReLU → MaxPool2d(2) → Conv2d(32→64, 3) → ReLU → MaxPool2d(2) → Flatten → Linear(64×5×5→128) → ReLU → Dropout(0.5) → Linear(128→10)` |

#### `count_parameters(model) -> int`

Đếm tổng params và params trainable.

---

## M3 — Training module (chưa code)

### `src/training/train.py`

#### `train_one_epoch(model, loader, criterion, optimizer, device) -> (avg_loss, avg_acc)`

Train 1 epoch, trả về loss trung bình và accuracy trung bình.

#### `validate(model, loader, criterion, device) -> (avg_loss, avg_acc)`

Validate (không cập nhật weights).

#### `fit(model, train_loader, val_loader, config, device) -> (history, best_model)`

Loop toàn bộ epoch, lưu best model theo val_acc, trả về history dict.

---

## M4 — Evaluation module (chưa code)

### `src/evaluation/evaluate.py`

#### `test_accuracy(model, loader, device) -> float`

Accuracy (%) trên tập test, dùng để đánh giá cuối cùng.

---

## M5 — Visualization module (chưa code)

### `src/visualization/plot.py`

#### `plot_loss_curve(history, path) -> None`

Vẽ train loss + val loss theo epoch, lưu PNG.

#### `show_predictions(model, loader, n=10, path) -> None`

Lưu ảnh 10 mẫu test kèm nhãn dự đoán vs nhãn thật (màu xanh = đúng, đỏ = sai).

---

## Tests

Mỗi module có file test riêng ở `tests/`.

| File | Số test | Trạng thái |
|------|---------|------------|
| `tests/test_dataset.py` | 11 | ✅ 11/11 pass |
| `tests/test_model.py` | (M2) | ⏳ |
| `tests/test_training.py` | (M3) | ⏳ |
| `tests/test_evaluation.py` | (M4) | ⏳ |
| `tests/test_visualization.py` | (M5) | ⏳ |

**Chạy tất cả test:**
```bash
pytest -v
```

**Chạy 1 file:**
```bash
pytest tests/test_dataset.py -v
```

---

## Tham chiếu

- `progress_status.md` — mục **Interfaces / APIs** có bảng tóm tắt song song.
- `docs/architecture.md` — sơ đồ kiến trúc tổng quan.
- `docs/data-flow.md` — luồng dữ liệu chi tiết.

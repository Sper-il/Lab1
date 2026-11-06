# `data/` — hướng dẫn dữ liệu

> Thư mục này **được gitignore** (xem `.gitignore`). Mỗi máy tự tải FashionMNIST
> gốc về qua `src.data.split`; chỉ `data/splits/` (metadata text) được commit.

## Cấu trúc

```
data/
├── processed/                        # gitignore — FashionMNIST gốc + cache
│   ├── FashionMNIST/raw/             # file idx3/idx1 gốc của torchvision
│   ├── train/<class_id>/*.png        # 50.000 ảnh (artifact Phase 1, không còn được đọc)
│   ├── val/<class_id>/*.png          # 10.000 ảnh (artifact Phase 1, không còn được đọc)
│   └── test/<class_id>/*.png         # 10.000 ảnh (artifact Phase 1, không còn được đọc)
├── splits/                           # metadata thuần text (committed)
│   └── split_info.txt                # thông tin split: sizes, seed
└── README.md                         # file này
```

> **Quy ước**:
> - `data/processed/` = nơi torchvision tải FashionMNIST gốc về (dùng làm `root`)
> - `data/splits/` = metadata (file text nhỏ, có thể commit)

## 10 lớp (theo thứ tự index)

> Bảng đầy đủ kèm số lượng train/val/test từng lớp: `results/runs/dataset_info.md`

| ID | Tên lớp |
|----|---------|
| 0  | T-shirt/top |
| 1  | Trouser |
| 2  | Pullover |
| 3  | Dress |
| 4  | Coat |
| 5  | Sandal |
| 6  | Shirt |
| 7  | Sneaker |
| 8  | Bag |
| 9  | Ankle boot |

Mỗi lớp có **5.000 ảnh train + 1.000 ảnh val + 1.000 ảnh test** (chia đều từ
6.000 train + 1.000 test gốc của FashionMNIST).

## Cách chuẩn bị dữ liệu

### Bước 1 — Tải gốc & chia tập (1 lần, ~2-3 phút)

```bash
python -m src.data.split --root data/processed --val-split 10000 --seed 42
```

Lệnh này sẽ:

1. Tải FashionMNIST gốc (~30 MB) về `data/processed/FashionMNIST/raw/` (lần đầu).
2. Chia 60.000 ảnh train gốc thành **50.000 train + 10.000 val** (stratified, seed=42).
3. Test set giữ nguyên **10.000 ảnh** của FashionMNIST, không chia lại.
4. Ghi metadata ra `data/splits/split_info.txt` và in số batch của từng tập.

> **Lưu ý:** `src.data.split` làm việc trên **tensor gốc của torchvision**, không
> xuất ra file `.png`. Các thư mục `data/processed/{train,val,test}/` trong repo
> là artifact của pipeline Phase 1 cũ và **không còn được đọc** bởi code hiện tại —
> `run_training.py` và `run_experiments.py` đều gọi `split_pipeline` trực tiếp.

### Bước 2 — Xem lại kết quả split

```bash
# Metadata split (train/val/test sizes + seed)
cat data/splits/split_info.txt
# → Train Size: 50000 · Val Size: 10000 · Test Size: 10000

# Số batch thực tế
python -m src.data.split --root data/processed --val-split 10000 --seed 42
# → Train batches: 782 · Val batches: 157 · Test batches: 157  (bs=64)
```

## Verify

```bash
# Test pass
pytest tests -q
# → 27 passed
```

## Tham số

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--root` | `data/processed` | Thư mục chứa FashionMNIST gốc |
| `--val-split` | `10000` | Số ảnh validation lấy từ train gốc (10.000 ảnh = 1/6) |
| `--seed` | `42` | Random seed cho split (reproducible) |
| `--batch-size` | `64` | Batch size cho DataLoader |

## Vì sao tỷ lệ 5/6 – 1/6?

Từ 6.000 ảnh gốc mỗi lớp:
- `1/6` → **1.000** ảnh val → train còn **5.000** ảnh (số tròn, không lệch lớp nào)
- `0.1` → 600 ảnh val → train 5.400 (vẫn tròn, nhưng không khớp với
  `split_info.txt` và các báo cáo đã commit)

Toàn bộ project dùng chung `DEFAULT_VAL_RATIO = 1/6` trong `src/data/split.py`.

## Vì sao không commit `data/`?

- Tổng dung lượng ~140 MB (70k × 2 KB/ảnh PNG) — quá lớn cho git diff.
- Mỗi máy phải tải từ `torchvision` về (~30 MB) — vẫn nhỏ hơn nhiều so với
  commit cả folder.
- Nếu cần share, dùng `git lfs` (chưa setup ở project này).

## Tham khảo

- `src/data/split.py` — code split
- `src/data/dataset.py` — code load
- `docs/PHASE1_REPORT.md` — báo cáo Phase 1
- `docs/architecture.md` — system architecture

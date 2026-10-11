# `data/` — hướng dẫn dữ liệu

> Thư mục này **được gitignore** (xem `.gitignore`). Mỗi máy phải tự sinh lại
> 70.000 ảnh `.png` từ FashionMNIST gốc qua `src.data.split`.

## Cấu trúc

```
data/
├── raw/                              # gitignore
│   ├── train/<class_id>/*.png        # 50.000 ảnh
│   ├── val/<class_id>/*.png          # 10.000 ảnh
│   ├── test/<class_id>/*.png         # 10.000 ảnh
│   └── _torchvision_cache/            # cache nội bộ của torchvision (gitignore)
└── README.md                         # file này
```

## 10 lớp (theo thứ tự index)

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

Mỗi lớp có **6.000 ảnh train + 1.000 ảnh val + 1.000 ảnh test** (chia đều từ
6.000 train + 1.000 test gốc của FashionMNIST).

## Cách sinh dữ liệu

### Bước 1 — Chạy script split (1 lần, ~2-3 phút)

```bash
python -m src.data.split --root data/raw --val-split 10000 --seed 42
```

Script này sẽ:

1. Tải FashionMNIST gốc (~30 MB) về `data/raw/_torchvision_cache/` (lần đầu).
2. Chia 60.000 ảnh train gốc thành **50.000 train + 10.000 val** (random, seed=42).
3. Ghi tất cả 70.000 ảnh ra `.png` (28×28, grayscale) với tên
   `<class_name>_<original_idx>.png` trong folder tương ứng.
4. In bảng số lượng ảnh / lớp / split.

### Bước 2 — Idempotency

Nếu `data/raw/train/<class>/*.png` đã đủ 5.000 file, script sẽ **skip** toàn bộ
quá trình. Nếu muốn ép sinh lại:

```bash
# Xóa rồi chạy lại
Remove-Item -Recurse -Force data\raw    # PowerShell
# rm -rf data/raw                       # bash
python -m src.data.split --root data/raw --val-split 10000 --seed 42
```

## Verify sau khi sinh

```bash
# Tổng số file
Get-ChildItem -Recurse data\raw\*.png | Measure-Object    # PowerShell
find data/raw -name "*.png" | wc -l                       # bash
# → 70000

# Test pass
pytest tests/test_dataset.py -v
# → 11 passed
```

## Tham số

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--root` | `data/raw` | Thư mục output |
| `--val-split` | `10000` | Số mẫu validation lấy từ train gốc |
| `--seed` | `42` | Random seed cho split (reproducible) |

## Tại sao `.png` thay vì tensor trong RAM?

- **Reproducible trên disk** — hash file kiểm tra được, debug từng ảnh qua File Explorer.
- **Chống data leakage** — split ảnh gốc trước khi áp transform, mean/std tính từ
  literature không leak vào val/test.
- **Augmentation linh hoạt** — mỗi tập có thể có transform riêng (vd train có random
  flip, val/test giữ nguyên) mà không phải lưu nhiều bản tensor.
- **RAM thấp** — chỉ 1 batch tensor ở trong RAM lúc train, ảnh đọc on-the-fly.

## Tại sao không commit `data/`?

- Tổng dung lượng ~140 MB (70k × 2 KB/ảnh PNG) — quá lớn cho git diff.
- Mỗi máy phải tải từ `torchvision` về (~30 MB) — vẫn nhỏ hơn nhiều so với
  commit cả folder.
- Nếu cần share, dùng `git lfs` (chưa setup ở project này).

## Tham khảo

- `src/data/split.py` — code split
- `src/data/dataset.py` — code load
- `tests/test_dataset.py` — 11 test
- `docs/PHASE1_REPORT.md` — báo cáo Phase 1
- `docs/architecture.md` — system architecture

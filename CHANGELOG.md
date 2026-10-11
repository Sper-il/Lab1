# CHANGELOG — Practice 1 (FashionMNIST)

> Lịch sử thay đổi theo version. Áp dụng cho **model + dataset + pipeline**.
> Format theo [Keep a Changelog](https://keepachangelog.com/vi/1.1.0/).
> Ngày quy ước: YYYY-MM-DD.

## [Unreleased]

### Planned
- M2: `FashionCNN` v1 — kiến trúc 2 conv + 2 FC + Dropout, ~225K params.
- M3: training script với `train_one_epoch`, `validate`, `fit`, save best model.
- M4: `test_accuracy` + classification report.
- M5: `plot_loss_curve`, `show_predictions` → PNG trong `results/runs/<ts>/`.
- M6: experiment grid lr × optimizer (Phase 2).

---

## [0.1.0] — 2026-10-11 — Data pipeline MVP

### Added
- **Dataset v1.0** — 70.000 ảnh `.png` (50.000 train / 10.000 val / 10.000 test)
  từ `torchvision.datasets.FashionMNIST` gốc, split reproducible với seed=42.
  Path: `data/raw/{train,val,test}/<class_id>/<class_name>_<idx>.png`.
- **`src/data/split.py`** — `split_fashionmnist_to_disk(root, val_split, seed)`.
  Idempotent: nếu ảnh đã tồn tại đủ thì skip, không ghi lại.
- **`src/data/dataset.py`** — `build_dataloaders(config)` + `summarize(loader, name)`.
  Transform: `ToTensor + Normalize(0.2860, 0.3530)`. Auto `pin_memory` khi có CUDA.
- **`tests/test_dataset.py`** — 11 test pass (số file, class folder, shape batch,
  dtype, pixel range, số batch, idempotency, …).
- **`docs/architecture.md`, `docs/data-flow.md`, `docs/api.md`** — tài liệu nội bộ.
- **`AGENTS.md`** — working rules cho AI agent (naming, DoD, Conventional Commit).

### Configuration
- `configs/baseline.yaml` — `batch_size=64, num_workers=2, val_split=10000, seed=42,
  mean=0.2860, std=0.3530`.

### Known limitations
- Chỉ grayscale 28×28 (FashionMNIST gốc), không hỗ trợ ảnh RGB.
- Split 50k/10k/10k cố định; không có stratified split (10 class đều nên không cần).
- Không có augmentation ở M1; sẽ thử ở M6 nếu accuracy chưa đạt.

### Reproducibility
```bash
pip install -r requirements.txt
python -m src.data.split --root data/raw --val-split 10000 --seed 42
pytest -v
```

### Commits in this version
- `599d20f` — M0 scaffold
- `ff2e11a` — M1 data module
- `ca59d91` — mark M1 done
- `e32241d` — điền docs
- `eed745e` — README + AI_Usage
- `d271a1c` — AGENTS.md + __init__.py

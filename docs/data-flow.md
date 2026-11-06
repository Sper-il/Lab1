# Luồng dữ liệu (Data Flow)

> Chi tiết quy ước shape/định dạng xem `docs/data.md`. Kiến trúc tổng thể xem `docs/architecture.md`.

## 1. Sơ đồ tổng thể

```
FashionMNIST gốc (torchvision, ~30 MB)
  → data/processed/FashionMNIST/raw/           # tải 1 lần
  → split_pipeline (stratified, seed=42)       # src/data/split.py
  → train 50.000 / val 10.000 / test 10.000    # tensor trong RAM, batch (B,1,28,28)
  → DataLoader (bs=128, num_workers=0)         # train 391 / val 79 / test 79 batch
  → Model (logits (B,10))
  → CrossEntropyLoss / softmax → metrics
  → checkpoint + history + plots               # models/checkpoints/, results/runs/
```

## 2. Từng chặng (kèm shape)

| Chặng | Module | Input → Output | Tần suất |
|-------|--------|----------------|----------|
| Tải gốc | `torchvision.datasets.FashionMNIST` | URL → `data/processed/FashionMNIST/raw/*.ubyte` | 1 lần/máy |
| Split | `split_pipeline` (`src/data/split.py`) | 60k tensor `(1,28,28)` → Subset 50k/10k + test 10k | Mỗi lần train |
| Batch | `DataLoader` | Subset → `(images (B,1,28,28), labels (B,))` | Mỗi step |
| Forward | `ModelFactory.create("simple_cnn")` | `(B,1,28,28)` → logits `(B,10)` | Mỗi step |
| Loss | `CrossEntropyLoss` | `(logits (B,10), labels (B,))` → scalar | Mỗi step |
| Optimize | Adam lr=0,001 (`src/training/optimizers.py`) | grads → update 421.642 params | Mỗi train step |
| Validate | `Trainer.validate` (`src/training/trainer.py`) | val_loader → `val_loss`, `val_acc` (weighted theo mẫu) | Mỗi epoch |
| Checkpoint | `callbacks.py` + `model_io.py` | best (`val_loss` min) → `trained_model.pt`; last → `last_model.pt` | Mỗi epoch cải thiện + cuối run |
| Visualize | `training/visualizer.py`, `evaluation/*` | `training_history.json` → `loss_curve.png`, `accuracy_curve.png`, `confusion_matrix.png`, `predictions.png` | Cuối run |
| Evaluate | `src/evaluation/run_evaluation.py` | `trained_model.pt` + test_loader → accuracy 92,10% + per-class + top-5 confused | 1 lần cuối |

## 3. Realtime vs batch

- Project chạy **batch/offline**: train 15 epochs (~75 s/epoch CPU), grid search 16 trials × 10 epochs (~4,5 giờ CPU), evaluation 1 pass trên 10.000 test.
- Không có serving realtime; demo tương đương là `python src/evaluation/run_evaluation.py` (load checkpoint → predict → vẽ plots).

## 4. Dung lượng và giới hạn

- RAM: batch 128 × 784 float32 ≈ 0,4 MB/batch — nhẹ, chạy được CPU lab.
- `data/` ~140 MB nếu giữ 70k PNG cũ (đã gitignore); checkpoint `trained_model.pt` ~1,7 MB.
- `num_workers=0` (Windows tương thích), `seed=42` cho split + train reproducible.
- Rủi ro từng gặp: `DataLoader` shuffle val bị lệch class do bug split (đã fix); BatchNorm flaky khi smoke test batch=4 (đã đổi sang 64 samples × 5 steps).

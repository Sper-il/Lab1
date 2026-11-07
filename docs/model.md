# Báo cáo tổng quan Model

> Báo cáo chính về kiến trúc mô hình — tổng hợp toàn bộ 4 kiến trúc, cơ chế `ModelFactory`, pilot so sánh và grid search, không phải báo cáo cá nhân.
> Báo cáo cá nhân của Người 1 xem tại `docs/reports/nguoi1_models_report.md`.
> Nguồn code: `src/models/`, `src/training/hyperparameter_tuning.py`, `src/training/hyperopt_scheduler.py`, `run_experiments.py`, `configs/hyperopt_config.yaml`.
> Kết quả: `results/runs/architecture_selection.md`, `results/runs/hyperopt.{json,md,_best.json}`.

## 1. Mục tiêu

- Dựng 4 kiến trúc phủ từ đơn giản đến residual để so sánh công bằng trên cùng split 50k/10k/10k (`seed=42`): `baseline_mlp`, `deep_mlp`, `simple_cnn`, `resnet_mini`.
- Mọi kiến trúc nhận `(B,1,28,28)` → trả logits `(B,10)`, không `softmax` ở cuối để dùng trực tiếp với `CrossEntropyLoss`.
- Tìm cấu hình tốt nhất bằng grid search tất định 16 trials × 10 epochs, bàn giao cho training full (`docs/training.md`).

## 2. Danh mục kiến trúc (`src/models/`)

| Tên đăng ký | Class / file | Params | Cấu trúc thật trong code |
|-------------|--------------|--------|---------------------------|
| `baseline_mlp` | `BaselineMLP` (`src/models/baseline_model.py`) | 203.530 | `Flatten` → `Linear(784→256)` → `ReLU` → `Dropout(0,2)` → `Linear(256→10)`. Mức sàn sanity-check. |
| `deep_mlp` | `DeepMLP` (`src/models/architectures/mlp.py`) | 118.650 | 3 hidden FC `80 → 256 → 128`, mỗi tầng `Linear → BatchNorm1d → ReLU → Dropout(0,3)` + `Linear(128→10)`. Tự `view` khi nhận `(B,1,28,28)`. |
| `simple_cnn` | `SimpleCNN` (`src/models/architectures/simple_cnn.py`) | 421.642 | 2 conv blocks (`Conv(1→32) → ReLU → MaxPool` → `Conv(32→64) → ReLU → MaxPool`, kernel 3, padding 1) → `Flatten(64·7·7=3136)` → `Linear(3136→128)` → `ReLU` → `Dropout(0,5)` → `Linear(128→10)`. **Cố ý không dùng BatchNorm.** |
| `resnet_mini` | `ResNetMini` + `BasicBlock` (`src/models/architectures/advanced_cnn.py`) | 174.970 | Stem `Conv(1→16)` + `BN`; 3 residual stages (2 blocks/stage, widths 16/32/64, strides 1/2/2, shortcut `1×1` khi lệch shape) → `AdaptiveAvgPool2d(1,1)` → `FC(64→10)`. |

### 3. Kiến trúc code

```
configs/hyperopt_config.yaml (strategy grid, seed 42, 4 model × 2 lr × 2 optimizer)
  → run_experiments.py (--smoke / --epochs / --batch-size / --skip-architecture-selection)
      ├── split_pipeline (train 50k / val 10k / test 10k, stratified, seed=42)
      ├── ModelFactory.create(name) + get_default_kwargs (dropout mặc định)
      ├── compare_architectures (pilot, Adam lr=1e-3 cố định) → architecture_selection.md
      └── HyperparameterTuner.grid_search + HyperoptScheduler → hyperopt.{json,md,_best.json}
```

- `ModelFactory` (`src/models/model_factory.py`): registry duy nhất Người 2/Người 3 dùng để tạo model (`create("simple_cnn")`); `list_models()` giữ đúng thứ tự search-space; dropout mặc định `baseline_mlp=0,2`, `deep_mlp=0,3`, `simple_cnn=0,5`, `resnet_mini={}` (không dropout).
- `compare_architectures` (`src/models/model_selection.py`): train mỗi kiến trúc cùng số epoch, cùng Adam lr=1e-3, cùng split → xếp hạng theo `val_acc`, xuất bảng markdown.
- `HyperparameterTuner` + `HyperoptScheduler` (`src/training/hyperparameter_tuning.py`, `hyperopt_scheduler.py`): duyệt tích Cartesian exhaustively, ghi `hyperopt.json` (16 rows), `hyperopt.md` (bảng), `hyperopt_best.json` (trial 9).

## 4. Kết quả pilot `compare_architectures` (2 epochs, lr=1e-3)

Nguồn: `results/runs/architecture_selection.md`.

| Model | Params | Train acc | Val acc | Train loss | Val loss | s/epoch |
|-------|--------|-----------|---------|------------|----------|---------|
| `simple_cnn` | 421.642 | 87,05% | 89,48% | 0,3858 | 0,2917 | 96,30 |
| `resnet_mini` | 174.970 | 89,11% | 88,63% | 0,3008 | 0,3098 | 258,94 |
| `baseline_mlp` | 203.530 | 86,20% | 87,05% | 0,3770 | 0,3507 | 30,92 |
| `deep_mlp` | 118.650 | 83,56% | 86,10% | 0,4601 | 0,3774 | 33,36 |

- Thứ tự ổn định: CNN vượt MLP ngay từ pilot; `simple_cnn` nhanh gấp ~2,7× `resnet_mini` (96 vs 259 s/epoch ở pilot).
- Smoke test (`python run_experiments.py --smoke`): forward `(64,1,28,28) → (64,10)` + loss giảm sau 5 steps Adam lr=1e-3 cho cả 4 kiến trúc.

## 5. Grid search full 16 trials × 10 epochs (160 epochs CPU ~4,5 giờ)

Search space (`configs/hyperopt_config.yaml`): 4 model × lr `{0,001; 0,0005}` × optimizer `{adam, sgd}`, `epochs=10`, `batch_size=128`, `weight_decay=0,0`, `seed=42`.

| Trial | Model | LR | Optimizer | Train acc | Val acc |
|------:|-------|----|-----------|-----------|---------|
| **9** | **simple_cnn** | **0,001** | **adam** | **92,79%** | **92,33%** |
| 15 | resnet_mini | 0,0005 | adam | 95,60% | 92,20% |
| 13 | resnet_mini | 0,001 | adam | 95,54% | 91,78% |
| 11 | simple_cnn | 0,0005 | adam | 92,11% | 91,23% |
| 14 | resnet_mini | 0,001 | sgd | 92,10% | 90,12% |

Bảng đầy đủ 16 trials xem tại `results/runs/hyperopt.md`.

- **Best trial 9:** `simple_cnn` + Adam + lr=0,001 + batch 128 + dropout 0,5 → train 92,79% / val **92,33%** (~75 s/epoch ở grid).
- **Runner-up trial 15:** `resnet_mini` + Adam + lr=0,0005 → val 92,20% nhưng train 95,60% (gap 3,4% so với 0,46% của best) → overfit rõ hơn và chậm gấp 2,9× (~219 vs ~75 s/epoch) nên bị loại.
- **Adam hơn SGD 2–6%** mọi kiến trúc ở cùng lr; SGD + lr=0,0005 học quá chậm (84–89% sau 10 epochs).

## 6. Model được chọn và bàn giao

```python
{
    "model_name": "simple_cnn",
    "learning_rate": 0.001,
    "optimizer": "adam",
    "batch_size": 128,
    "weight_decay": 0.0,
    "dropout": 0.5,
    "epochs": 10,   # mỗi trial grid; training full dùng 15 epochs (xem docs/training.md)
}
```

- Tạo model: `from src.models.model_factory import ModelFactory; model = ModelFactory.create("simple_cnn")`.
- Training full 15 epochs với đúng config này cho best epoch 11 (`val_loss=0,215183`, `val_acc=92,85%`) và test **92,10%** — chi tiết xem `docs/training.md`, `docs/evaluation.md`.
- Checkpoint verify 3 epochs cũ (`models/checkpoints/best_model.pt`, val 90,12%) chỉ là integration test, không phải weights bàn giao cuối; weights chính thức là `models/checkpoints/trained_model.pt` → `final_model.pt`.

## 7. Artifact bàn giao

| File | Nội dung |
|------|----------|
| `results/runs/architecture_selection.md` | Bảng pilot 4 kiến trúc (2 epochs) |
| `results/runs/hyperopt.json` | 16 trial rows (máy đọc) |
| `results/runs/hyperopt.md` | Bảng markdown 16 trials (người đọc) |
| `results/runs/hyperopt_best.json` | Trial 9 (`simple_cnn` + Adam + lr=0,001 → val 92,33%) |

## 8. Cách chạy lại

```bash
# Smoke test 4 kiến trúc (~5 giây, CPU)
python run_experiments.py --smoke

# Grid search nhanh (ví dụ 3 epochs/trial, ~3 phút)
python run_experiments.py --epochs 3 --batch-size 128

# Grid full 16 trials × 10 epochs (~4,5 giờ CPU)
python run_experiments.py --batch-size 128
```

## 9. Quyết định kỹ thuật chính

| Quyết định | Lựa chọn khác | Lý do |
|------------|---------------|-------|
| Grid search exhaustively thay vì random | Random search | Tất định, phủ kín, chạy lại ra đúng kết quả, không bỏ sót tổ hợp |
| Chọn `simple_cnn` thay vì `resnet_mini` | ResNetMini val 92,20% ≈ best | Best hơn 0,13%, ít overfit hơn (gap 0,46% vs 3,4%), nhanh gấp 2,9× nên hợp CPU lab |
| `SimpleCNN` không BatchNorm | Thêm BatchNorm | Giữ đơn giản theo đề; BatchNorm chỉ dùng cho `DeepMLP`/`ResNetMini` |
| `ModelFactory` làm điểm tiếp xúc duy nhất | Import class trực tiếp | Người 2/Người 3 không cần biết kiến trúc cụ thể, chỉ cần tên string |
| Dropout mặc định theo kiến trúc (0,2 / 0,3 / 0,5) | Một dropout chung | Giữ so sánh công bằng, mỗi họ model có mức regularize phù hợp |

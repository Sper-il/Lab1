# Network & Hyperparameter Report (Người 1)

> **Deliverable:** Build Network + Hyperparameter Experiment
> **Người phụ trách:** Trương Hoài Tú (Người 1)
> **Ngày chạy:** 2026-10-12
> **Dataset:** FashionMNIST (60k train / 10k test, 28×28 grayscale, 10 classes)
> **Device:** CPU (CUDA không khả dụng)
>
> ---

> ## ✅ Trạng thái hoàn thành (theo `PHASE2_PLAN.md` checklist)
>
> | Hạng mục | Trạng thái | Ghi chú |
> |-----------|:-----------:|---------|
> | `src/models/baseline_model.py` | ✅ Xong | `BaselineMLP` |
> | `src/models/architectures/mlp.py` | ✅ Xong | `DeepMLP` |
> | `src/models/architectures/simple_cnn.py` | ✅ Xong | `SimpleCNN` |
> | `src/models/architectures/advanced_cnn.py` | ✅ Xong | `ResNetMini` + `BasicBlock` |
> | `src/models/architectures/__init__.py` | ✅ Xong | Export 3 architectures |
> | `src/models/model_factory.py` | ✅ Xong | 4 models registered |
> | `src/models/model_selection.py` | ✅ Xong | `compare_architectures` |
> | `src/models/__init__.py` | ✅ Xong | Export `BaselineMLP`, `ModelFactory`, … |
> | `src/training/hyperparameter_tuning.py` | ✅ Xong | `HyperparameterTuner.grid/random_search` |
> | `src/training/hyperopt_scheduler.py` | ✅ Xong | Writer → `hyperopt.{json,md,_best.json}` |
> | `configs/hyperopt_config.yaml` | ✅ Xong | Grid + random search space |
> | `run_experiments.py` | ✅ Xong | Entry point (`--smoke`, `--strategy grid/random`, `--n-trials`, `--epochs`) |
> | Forward pass test 4 architectures | ✅ Xong | `(B,1,28,28) → (B,10)` |
> | Grid search (8 trials × 3 epoch) | ✅ Xong | best val_acc = **90.53%** |
> | Real training 3 epochs best config | ✅ Xong | val_acc = **90.12%** (saved `best_model.pt`) |
> | Best model + best params identified | ✅ Xong | `simple_cnn` + adam + lr=0.001 |
> | `docs/reports/nguoi1_models_report.md` | ✅ Xong | Báo cáo này |
>
> **Tổng kết: 17/17 hạng mục — HOÀN THÀNH 100%** (theo đề + PHASE2_PLAN)

---

## 1. Models Implemented

Bốn kiến trúc được cài đặt và đăng ký qua `ModelFactory` (xem `src/models/model_factory.py`):

| Tên đăng ký | Class | #params | Mô tả |
|--------------|-------|---------|-------|
| `baseline_mlp` | `BaselineMLP` | 203,530 | 1 hidden layer 256 → ReLU → Dropout → 10. Sanity-check baseline. |
| `deep_mlp`     | `DeepMLP`     | 118,650 | 3 hidden FC layers `[512, 256, 128]` + `BatchNorm1d` + Dropout 0.3. |
| `simple_cnn`   | `SimpleCNN`   | 421,642 | 2 conv blocks (`Conv→ReLU→Pool` ×2, channels 32→64) + 2 FC (`64·7·7 → 128 → 10`) + Dropout 0.5. |
| `resnet_mini`  | `ResNetMini`  | 174,970 | Stem `1→16`, 3 residual stages (2 blocks/stage, widths 16/32/64, strides 1/2/2), `AdaptiveAvgPool2d(1,1)`, FC 64→10. |

> Tất cả 4 architecture đều qua **forward test** trong `run_experiments.py --smoke`: forward `(B,1,28,28) → (B,10)`, loss giảm sau 5 steps Adam lr=1e-3 trên batch 64 samples (fix smoke test từ 4 samples gây flaky BatchNorm).

### Đặc điểm thiết kế chung

* Tất cả các model chấp nhận cả input `(B, 1, 28, 28)` lẫn đã-được-flatten `(B, 784)` (riêng `DeepMLP` tự `view`).
* Không có `softmax` ở cuối — để dùng với `nn.CrossEntropyLoss` (logits + softmax numerically stable).
* `BatchNorm` chỉ dùng cho MLP sâu + ResNet (CNN đơn giản cố ý giữ nguyên per đề).
* Residual block trong `ResNetMini.BasicBlock` dùng `Conv→BN→ReLU` chuẩn ResNet và `1×1 projection` khi shape thay đổi (stride hoặc channels).

---

## 2. Hyperparameter Search Results

### 2.1 Search Space (`configs/hyperopt_config.yaml`)

Tìm kiếm kết hợp 4 architecture với các hyperparameter:

```yaml
strategy: random          # mặc định; grid cũng implement
n_trials: 12
epochs_per_trial: 3

grid_search:
  model_name:    [simple_cnn, deep_mlp]
  learning_rate: [0.001, 0.0005]
  optimizer:     [adam, sgd]
  epochs: 3

random_search:
  model_name:    [baseline_mlp, deep_mlp, simple_cnn, resnet_mini]
  learning_rate: [0.0001, 0.001, 0.01]
  batch_size:    [32, 64, 128]
  weight_decay:  [0.0, 0.0001, 0.001]
  optimizer:     [adam, sgd]
  dropout:       [0.2, 0.3, 0.5]
  epochs: 3
```

> Pilot chính thức dùng `grid_search` (8 trials) cho reproducibility & coverage đầy đủ các combination; `random_search` đã implement đầy đủ (`HyperparameterTuner.random_search` hỗ trợ `(low, high)`, `(low, high, 'log')`, `(low, high, 'int')`).

### 2.2 Kết quả GRID SEARCH thật — 8 trials × 3 epoch (CPU)

> ⚙️ **Lệnh chạy:** `python run_experiments.py --strategy grid --batch-size 128`
> ⏱️ **Thời gian:** ~19 phút (CPU) — hoàn thành 12:32 ngày 2026-10-12
> 📁 **Artifacts:** `results/runs/hyperopt.json` (2,470 bytes), `hyperopt.md` (1,062 bytes), `hyperopt_best.json` (215 bytes)

Bảng đầy đủ 8 trials (sort theo `val_acc` giảm dần):

| Trial | Model        | LR     | Optimizer  | Train Acc | Val Acc   | Params   | s/epoch |
|-------|--------------|--------|------------|-----------|-----------|----------|---------|
| **1** ⭐ | **simple_cnn**  | **0.001** | **adam**   | **88.81%** | **90.53%** | 421,642 | 63.14 |
| 3     | simple_cnn    | 0.0005 | adam       | 87.22%    | 88.97%    | 421,642  | 62.04  |
| 5     | deep_mlp      | 0.001  | adam       | 84.91%    | 87.13%    | 118,650  | 31.26  |
| 7     | deep_mlp      | 0.0005 | adam       | 84.76%    | 87.07%    | 118,650  | 27.24  |
| 6     | deep_mlp      | 0.001  | sgd        | 80.49%    | 83.98%    | 118,650  | 25.40  |
| 8     | deep_mlp      | 0.0005 | sgd        | 77.54%    | 82.25%    | 118,650  | 26.23  |
| 2     | simple_cnn    | 0.001  | sgd        | 77.50%    | 80.83%    | 421,642  | 65.29  |
| 4     | simple_cnn    | 0.0005 | sgd        | 74.31%    | 78.13%    | 421,642  | 62.51  |

### 2.3 Nhận xét kết quả grid search

* **Adam luôn tốt hơn SGD** ở cùng lr (~7–10% gap). Đúng kỳ vọng trên dataset nhỏ + ít epoch.
* **simple_cnn + adam lr=1e-3** là tổ hợp tốt nhất (**90.53%**) — CNN tận dụng được cấu trúc không gian ảnh.
* **deep_mlp + adam** đạt ~87% — không tệ cho fully-connected, nhưng không beat CNN.
* **SGD ở lr=0.0005** dưới học rất chậm (78–82% chỉ sau 3 epoch).
* **Tốc độ:** `deep_mlp` nhanh hơn `simple_cnn` ~2× (118k vs 421k params), nhưng accuracy thấp hơn 3–4%.
* **Baseline & ResNet:** không nằm trong grid (giữ cho random search); pilot `compare_architectures` riêng đã verify:
  - `baseline_mlp` (2 epoch, 1e-3): val_acc = **86.75%**
  - `deep_mlp` (2 epoch, 1e-3): val_acc = **86.25%**
  - `simple_cnn` (2 epoch, 1e-3): val_acc = **88.58%**
  - `resnet_mini` (1 epoch, 1e-3): val_acc = 86.78% (epoch đầu)

---

## 3. Selected Model

### 3.1 Best Hyperparameters (gửi Người 2)

```python
{
    "model_name":    "simple_cnn",
    "learning_rate": 0.001,
    "optimizer":     "adam",
    "batch_size":    128,
    "weight_decay":  0.0,
    "dropout":       0.5,      # default của SimpleCNN
    "epochs":        3,        # pilot; Người 2 sẽ train full ~10–15 epoch
}
```

Tạo model qua:
```python
from src.models.model_factory import ModelFactory
model = ModelFactory.create("simple_cnn")  # dropout=0.5 default
```

### 3.2 Expected Accuracy

| Metric | Pilot (3 epoch) | Expected (10–15 epoch) |
|--------|-----------------|------------------------|
| Train accuracy | 88.79% | 92–94% |
| Val accuracy   | **90.53%** | **91–93%** |
| Val loss       | 0.27 | ~0.20 |

### 3.3 Trained Model Best Checkpoint (verify integration)

> 🎯 **Đây là kết quả training thật** với best config — không phải smoke test.

```
=== Training 3 epochs thật trên 54k train / 6k val ===
epoch 1: train_loss=0.5761  train_acc=79.14%  val_loss=0.3444  val_acc=86.98%
epoch 2: train_loss=0.3712  train_acc=86.77%  val_loss=0.2999  val_acc=89.43%
epoch 3: train_loss=0.3173  train_acc=88.79%  val_loss=0.2689  val_acc=90.12% ⭐
```

**Checkpoint saved:** `models/checkpoints/best_model.pt` (1,690,359 bytes)
- `model_state_dict`: 421,642 params
- `model_name`: `simple_cnn`
- `val_acc`: 90.12%
- `hyperparams`: `{model_name, learning_rate=0.001, optimizer=adam, batch_size=128, epochs=3}`

Đây là integration test đảm bảo `ModelFactory.create("simple_cnn")` → train được bằng trainer chuẩn (forward → loss → backward → optimize) mà không cần sửa gì thêm. Người 2 có thể load checkpoint này để:
- Resume training (10–15 epoch) để đạt val_acc ~91–93%
- Hoặc fine-tune với config khác

---

## 4. Files Created / Modified

### Files mới (Người 1)

| Path | Mục đích |
|------|----------|
| `src/__init__.py` | Khai báo package `src`. |
| `src/models/baseline_model.py` | `BaselineMLP`. |
| `src/models/architectures/__init__.py` | Export `DeepMLP`, `SimpleCNN`, `ResNetMini`. |
| `src/models/architectures/mlp.py` | `DeepMLP` (3 hidden FC + BatchNorm + Dropout). |
| `src/models/architectures/simple_cnn.py` | `SimpleCNN` (2 conv blocks + 2 FC). |
| `src/models/architectures/advanced_cnn.py` | `BasicBlock`, `ResNetMini` (3 residual stages). |
| `src/models/model_factory.py` | `ModelFactory` (registry + kwargs defaults). |
| `src/models/model_selection.py` | `compare_architectures` cho quick architecture ranking. |
| `src/training/hyperparameter_tuning.py` | `HyperparameterTuner`, `TrialResult`, grid/random search. |
| `src/training/hyperopt_scheduler.py` | `HyperoptScheduler`, `SchedulerConfig`, writer (`results/runs/hyperopt.{json,md}` + `_best.json`). |
| `configs/hyperopt_config.yaml` | Search-space định nghĩa (grid + random). |
| `run_experiments.py` | Entry point end-to-end của Người 1 (có `--smoke`, `--strategy`, `--n-trials`, `--epochs`). |

### Files sửa (bug fix)

| Path | Sửa gì | Tại sao |
|------|--------|---------|
| `src/data/split.py` | (a) `from dataset import …` → `from src.data.dataset import …`. <br>(b) `np.random.shuffle(stratified_indices)` ở cuối `_stratified_split`. | (a) Tránh `ModuleNotFoundError` khi gọi qua `sys.path`. <br>(b) Trước fix, val_loader chỉ chứa toàn class 8 ⇒ val_acc = 0%. Sau fix, phân bố cân bằng ≈600/lớp. |
| `run_experiments.py` | Smoke test: 4 samples → 64 samples × 5 steps | BatchNorm statistics quá nhiễu với batch=4, gây flaky fail. |

### Files cập nhật (export)

| Path | Sửa gì |
|------|--------|
| `src/models/__init__.py` | Export `BaselineMLP`, `ModelFactory`, `compare_architectures`. |
| `src/training/__init__.py` | Export `HyperparameterTuner`, `HyperoptScheduler`, `TrialResult`. |

### Artifacts sinh ra khi chạy thật

```
results/runs/
├── hyperopt.json          # 2,470 bytes — 8 trial raw rows (real)
├── hyperopt.md            # 1,062 bytes — markdown table (real)
├── hyperopt_best.json     # 215 bytes — best config
├── class_distribution.png # EDA (Phase 1)
├── per_class_samples.png  # EDA (Phase 1)
├── pixel_histogram.png    # EDA (Phase 1)
└── sample_images.png      # EDA (Phase 1)

models/checkpoints/
└── best_model.pt          # 1,690,359 bytes — simple_cnn @ 3 epoch, val_acc=90.12%
```

---

## 5. Test Coverage (đã chạy thật)

| Test | Kết quả |
|------|:-------:|
| Smoke test 4 architectures (forward shape + loss giảm) | ✅ pass (64 samples × 5 steps) |
| Forward `(B, 1, 28, 28) → (B, 10)` cho mọi model | ✅ pass |
| Grid search với 8 trials × 3 epoch | ✅ pass (~19 phút CPU) |
| Random search sampling (log/uniform/int/choice) | ✅ pass (unit) |
| Real training 1 epoch trên FashionMNIST thật | ✅ pass (val_acc = 86.98% cho simple_cnn @ lr=1e-3) |
| Real training 3 epochs trên FashionMNIST thật | ✅ pass (val_acc = **90.12%**) |
| Save checkpoint `best_model.pt` | ✅ pass (1,690,359 bytes) |
| Persist results → `results/runs/hyperopt.{json,md,_best.json}` | ✅ pass |

### Reproduce

```bash
# Forward + loss-decrease test
python run_experiments.py --smoke

# Full pilot (8 trials × 3 epoch, ~19 phút CPU)
python run_experiments.py --strategy grid --batch-size 128

# Random search (12 trials × 3 epoch, ~30 phút CPU)
python run_experiments.py --strategy random --n-trials 12 --epochs 3 --batch-size 128
```

---

## 6. Handoff cho Người 2 / Người 3

### Sang Người 2 (Training loop + Visualize loss)

* **Architecture**: `model = ModelFactory.create("simple_cnn")` (mặc định `dropout=0.5`, `num_classes=10`).
* **Hyperparameters**: `lr=1e-3`, `optimizer=Adam`, `batch_size=128`, `weight_decay=0.0`, `epochs=10–15` cho full training.
* **Loss**: `nn.CrossEntropyLoss()`.
* **Loader**: dùng `from src.data.split import split_pipeline` → `train_loader, val_loader, test_loader, split_info = split_pipeline(batch_size=128)` — đã verify ở pilot này.
* **Checkpoint có sẵn**: `models/checkpoints/best_model.pt` (val_acc=90.12% sau 3 epoch) — có thể load để resume hoặc fine-tune.
* **File gợi ý**: `src/training/trainer.py` (forward → loss → backward → optimize) + `src/training/visualizer.py` cho loss curve.

### Sang Người 3 (Evaluation + Save/Load + Image Display)

* Checkpoint từ full training sẽ ở `models/checkpoints/best_model.pt` hoặc `models/checkpoints/final_model.pt` — load lại bằng:
  ```python
  from src.models.model_factory import ModelFactory
  model = ModelFactory.create("simple_cnn")
  ck = torch.load("models/checkpoints/best_model.pt", map_location="cpu")
  model.load_state_dict(ck["model_state_dict"])
  ```
* Dự kiến `test_accuracy ≥ 91%` sau khi Người 2 train full 10–15 epoch.

---

**Người 1 — Network & Hyperparameter Experiment — HOÀN THÀNH 100% ✅**
*(Trương Hoài Tú, 2026-10-12 — kết quả thật từ grid search 8 trials + real training 3 epochs)*

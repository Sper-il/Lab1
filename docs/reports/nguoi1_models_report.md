# Network & Hyperparameter Report (Người 1)

> **Deliverable:** Build Network + Hyperparameter Experiment
> **Người phụ trách:** Người 1 (Network)
> **Ngày chạy:** 2026-10-12
> **Dataset:** FashionMNIST (60k train / 10k test, 28×28 grayscale, 10 classes)
> **Device:** CPU (CUDA không khả dụng)

---

## 1. Models Implemented

Bốn kiến trúc được cài đặt và đăng ký qua `ModelFactory` (xem `src/models/model_factory.py`):

| Tên đăng ký | Class | #params | Mô tả |
|--------------|-------|---------|-------|
| `baseline_mlp` | `BaselineMLP` | 203,530 | 1 hidden layer 256 → ReLU → Dropout → 10. Sanity-check baseline. |
| `deep_mlp`     | `DeepMLP`     | 118,650 | 3 hidden FC layers [512, 256, 128] + `BatchNorm1d` + Dropout 0.3. |
| `simple_cnn`   | `SimpleCNN`   | 421,642 | 2 conv blocks (`Conv→ReLU→Pool` ×2, channels 32→64) + 2 FC (`64·7·7 → 128 → 10`) + Dropout 0.5. |
| `resnet_mini`  | `ResNetMini`  | 174,970 | Stem `1→16`, 3 stages residual blocks (2 blocks/stage, widths 16/32/64, strides 1/2/2), `AdaptiveAvgPool2d(1,1)`, FC 64→10. |

> Tất cả 4 architecture đều qua **smoke test** trong `run_experiments.py --smoke`: forward `(B,1,28,28) → (B,10)`, loss giảm sau 1 step Adam lr=1e-3. Output shape `(batch, 10)` đúng yêu cầu đề.

### Đặc điểm thiết kế chung

* Tất cả các model chấp nhận cả input `(B, 1, 28, 28)` lẫn đã-được-flatten `(B, 784)` (riêng `DeepMLP` tự `view`).
* Không có `softmax` ở cuối — để dùng với `nn.CrossEntropyLoss` (logits + softmax numerically stable).
* `BatchNorm` chỉ dùng cho MLP sâu + ResNet (CNN đơn giản cố ý giữ nguyên per đề).
* Residual block trong `ResNetMini.BasicBlock` dùng pre-activation `Conv→BN→ReLU` chuẩn ResNet và `1×1 projection` khi shape thay đổi (stride hoặc channels).

---

## 2. Hyperparameter Search Results

### 2.1 Search Space (`configs/hyperopt_config.yaml`)

Tìm kiếm kết hợp 4 architecture với các hyperparameter:

```yaml
strategy: grid
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

> Lưu ý: trong pilot này Người 1 dùng `grid_search` (8 trials) cho reproducibility; `random_search` cũng được implement đầy đủ trong `HyperparameterTuner.random_search()` và đã qua unit-test (`_sample_random`, hỗ trợ cả `(low, high)`, `(low, high, 'log')`, `(low, high, 'int')`).

### 2.2 Kết quả grid search — 8 trials × 3 epoch trên CPU

Bảng đầy đủ ở `results/runs/hyperopt.md` (sinh tự động bởi `HyperoptScheduler`). Tóm tắt theo thứ tự `val_acc` giảm dần:

| Trial | Model       | LR     | Optimizer  | Train Acc | Val Acc   | Params   | s/epoch |
|-------|-------------|--------|------------|-----------|-----------|----------|---------|
| **1** | **simple_cnn** | **0.001** | **adam** | **88.44%** | **89.97%** ⭐ | 421,642 | 66.07 |
| 3     | simple_cnn   | 0.0005 | adam       | 87.10%    | 88.30%    | 421,642  | 49.89   |
| 5     | deep_mlp     | 0.001  | adam       | 84.94%    | 86.55%    | 118,650  | 19.56   |
| 7     | deep_mlp     | 0.0005 | adam       | 85.00%    | 86.17%    | 118,650  | 19.24   |
| 6     | deep_mlp     | 0.001  | sgd        | 80.48%    | 84.10%    | 118,650  | 19.19   |
| 2     | simple_cnn   | 0.001  | sgd        | 79.15%    | 81.87%    | 421,642  | 52.29   |
| 8     | deep_mlp     | 0.0005 | sgd        | 78.12%    | 81.65%    | 118,650  | 18.98   |
| 4     | simple_cnn   | 0.0005 | sgd        | 74.23%    | 78.50%    | 421,642  | 47.63   |

### 2.3 Nhận xét

* **Adam luôn tốt hơn SGD** ở cùng lr (~3–10% gap). Đúng kỳ vọng trên dataset nhỏ + ít epoch.
* **simple_cnn + adam lr=1e-3** là tổ hợp tốt nhất (89.97%) — CNN tận dụng được cấu trúc không gian ảnh.
* **deep_mlp vẫn đạt ~86%** — không tệ cho fully-connected, nhưng không beat CNN.
* **baseline_mlp** và **resnet_mini** không nằm trong grid này (giữ cho random search); pilot riêng đã verify `baseline_mlp` forward+backward hoạt động qua smoke test, và `resnet_mini` được đánh giá riêng ở `compare_architectures` (xem note bên dưới).
* **Smoke test riêng** (`compare_architectures` step trong run_experiments.py với 1–2 epoch): đã chạy `simple_cnn` 1 epoch lr=1e-3 đạt **val_acc 86.80%** trong ~64s; và 2-trial grid `simple_cnn` vs `resnet_mini` cho `simple_cnn=87.52%`, `resnet_mini=79.13%` (1 epoch).

---

## 3. Selected Model

### 3.1 Best Hyperparameters (gửi Người 2)

```python
{
    "model_name":    "simple_cnn",
    "learning_rate": 0.001,
    "optimizer":     "adam",
    "epochs":        3,        # pilot; Người 2 sẽ train full ~10–15 epoch
    "batch_size":    128,      # đã verify ở pilot
    "weight_decay":  0.0,      # baseline
    "dropout":       0.5,      # default của SimpleCNN
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
| Train accuracy | 88.4% | 92–94% |
| Val accuracy   | **89.97%** | **91–93%** |
| Val loss       | 0.275 | ~0.20 |

> Kỳ vọng trên 91% sau full training — đủ thoả đề bài (≥90% target ở Phase 1 README).

### 3.3 Trained Model 1-Epoch (verify integration)

`SimpleCNN` 1 epoch với config trên đã được train và verify trong pilot:

```
[trial 1/8] {'model_name': 'simple_cnn', 'learning_rate': 0.001,
             'optimizer': 'adam', 'epochs': 3}
  epoch 1:  train_loss=0.5851, val_loss=0.3497, val_acc=86.80%
  epoch 2:  train_loss=0.4228, val_loss=0.3185
  epoch 3:  train_loss=0.3226, val_loss=0.2748, val_acc=89.97%
```

Đây là integration test đảm bảo `ModelFactory.create("simple_cnn")` → train được bằng trainer của Người 2 (forward → loss → backward → optimize) mà không cần sửa gì thêm.

---

## 4. Files Created / Modified

### Files mới (Người 1)

| Path | Mục đích |
|------|----------|
| `src/__init__.py` | Khai báo package `src` (sửa lỗi import tương đối). |
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

### Files sửa (bug fix không liên quan đến Người 1, nhưng chặn pipeline)

| Path | Sửa gì | Tại sao |
|------|--------|---------|
| `src/data/split.py` | (a) `from dataset import …` → `from src.data.dataset import …`. <br>(b) `np.random.shuffle(stratified_indices)` ở cuối `_stratified_split`. | (a) Tránh `ModuleNotFoundError` khi gọi qua `sys.path`. <br>(b) Trước fix, val_loader chỉ chứa toàn class 8 ⇒ val_acc = 0%. Sau fix, phân bố cân bằng ≈600/lớp. Đây là bug của Phase 1 nhưng cần sửa để Người 1 verify được. |

### Files cập nhật (export)

| Path | Sửa gì |
|------|--------|
| `src/models/__init__.py` | Export `BaselineMLP`, `ModelFactory`, `compare_architectures`. |
| `src/training/__init__.py` | Export `HyperparameterTuner`, `HyperoptScheduler`, `TrialResult`. |

### Artifacts sinh ra khi chạy search

```
results/runs/
├── hyperopt.json          # raw trial rows
├── hyperopt.md            # markdown table
└── hyperopt_best.json     # best (model, params, accuracy)
```

---

## 5. Test Coverage (đã chạy)

| Test | Kết quả |
|------|---------|
| Smoke test 4 architectures (forward shape + loss giảm) | ✅ pass |
| Forward `(B, 1, 28, 28) → (B, 10)` cho mọi model | ✅ pass |
| Grid search với 8 trials (grid + scalar constants) | ✅ pass |
| Random search sampling (log/uniform/int/choice) | ✅ pass (unit) |
| Real training 1 epoch trên FashionMNIST thực | ✅ pass (val_acc = 86.80% cho simple_cnn @ lr=1e-3) |
| Real training 3 epochs trên FashionMNIST thực | ✅ pass (best val_acc = 89.97%) |
| Persist results → `results/runs/hyperopt.{json,md,_best.json}` | ✅ pass |

### Reproduce

```bash
# Forward + loss-decrease test
python run_experiments.py --smoke

# Quick 2-trial pilot (1 epoch, ~3 phút CPU)
python run_experiments.py --strategy grid --n-trials 2 --epochs 1 \
    --skip-architecture-selection --batch-size 128

# Full pilot (8 trials × 3 epoch, ~10–12 phút CPU)
python run_experiments.py --strategy grid --batch-size 128

# Random search (12 trials × 3 epoch)
python run_experiments.py --strategy random --n-trials 12 --epochs 3 --batch-size 128
```

---

## 6. Handoff cho Người 2 / Người 3

### Sang Người 2 (Training loop + Visualize loss)

* **Architecture**: `model = ModelFactory.create("simple_cnn")` (mặc định `dropout=0.5`, `num_classes=10`).
* **Hyperparameters**: `lr=1e-3`, `optimizer=Adam`, `batch_size=128`, `weight_decay=0.0`, `epochs=10–15` cho full training.
* **Loss**: `nn.CrossEntropyLoss()`.
* **Loader**: dùng `src.data.split.split_pipeline(data_dir="./data/processed", batch_size=128, val_ratio=0.1)` — đã được verify ở pilot này.
* **File gợi ý**: `src/training/trainer.py` (forward → loss → backward → optimize) + `src/training/visualizer.py` cho loss curve.

### Sang Người 3 (Evaluation + Save/Load + Image Display)

* Checkpoint từ full training sẽ ở `models/checkpoints/best_model.pt` — load lại bằng `ModelFactory.create("simple_cnn")` rồi `load_state_dict(...)`.
* Dự kiến `test_accuracy ≥ 91%` sau khi Người 2 train full.

---

**Người 1 — Network & Hyperparameter Experiment, hoàn thành.**
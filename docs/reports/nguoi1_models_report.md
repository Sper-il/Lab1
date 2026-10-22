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
> | `src/training/hyperparameter_tuning.py` | ✅ Xong | `HyperparameterTuner.grid_search` |
> | `src/training/hyperopt_scheduler.py` | ✅ Xong | Writer → `hyperopt.{json,md,_best.json}` |
> | `configs/hyperopt_config.yaml` | ✅ Xong | Grid search space (16 trials) |
> | `run_experiments.py` | ✅ Xong | Entry point (`--smoke`, `--strategy grid`, `--epochs`) |
> | Forward pass test 4 architectures | ✅ Xong | `(B,1,28,28) → (B,10)` |
> | Grid search (16 trials × 10 epoch, 4 architectures) | ✅ Xong | best val_acc = **92.33%** |
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

Tìm kiếm bằng **grid search** (tich Cartesian, tat dinh) tren 4 architecture:

`yaml
strategy: grid
epochs_per_trial: 10
seed: 42

grid_search:
  model_name:    [baseline_mlp, deep_mlp, simple_cnn, resnet_mini]
  learning_rate: [0.001, 0.0005]
  optimizer:     [adam, sgd]
  epochs: 10
`

> **4 model x 2 learning rate x 2 optimizer = 16 trials x 10 epoch = 160 epoch.**
> Grid **phu kin toan bo** khong gian tim kiem va **tat dinh** (chay lai cho dung ket qua cu) — khong co to hop nao bi bo sot nhu cac phuong phap lay mau ngau nhien.

### 2.2 Ket qua GRID SEARCH that — 16 trials x 10 epoch (CPU)

> Lech chay: `python run_experiments.py --batch-size 128`
> Artifacts: `results/runs/hyperopt.json`, `hyperopt.md`, `hyperopt_best.json`

Bang day du 16 trials (sort theo `val_acc` giam dan):

| Trial | Model        | LR     | Optimizer  | Train Acc | Val Acc   | Params   | s/epoch |
|-------|--------------|--------|------------|-----------|-----------|----------|---------|
| **9**  | **simple_cnn**  | **0.001** | **adam**   | **92.79%** | **92.33%** | 421,642 | 75.20 |
| 15    | resnet_mini  | 0.0005 | adam       | 95.60%    | 92.20%    | 174,970  | 218.63 |
| 13    | resnet_mini  | 0.001  | adam       | 95.54%    | 91.78%    | 174,970  | 209.03 |
| 11    | simple_cnn  | 0.0005 | adam       | 92.11%    | 91.23%    | 421,642  | 71.28 |
| 14    | resnet_mini  | 0.001  | sgd        | 92.10%    | 90.12%    | 174,970  | 201.81 |
| 3     | baseline_mlp| 0.0005 | adam       | 90.36%    | 89.88%    | 203,530  | 29.28  |
| 1     | baseline_mlp| 0.001  | adam       | 90.63%    | 89.75%    | 203,530  | 29.15  |
| 5     | deep_mlp    | 0.001  | adam       | 87.95%    | 89.33%    | 118,650  | 27.78  |
| 16    | resnet_mini  | 0.0005 | sgd        | 89.91%    | 88.78%    | 174,970  | 218.03 |
| 7     | deep_mlp    | 0.0005 | adam       | 87.82%    | 88.72%    | 118,650  | 27.68  |
| 6     | deep_mlp    | 0.001  | sgd        | 84.74%    | 87.10%    | 118,650  | 26.79  |
| 10    | simple_cnn  | 0.001  | sgd        | 85.23%    | 87.05%    | 421,642  | 67.44  |
| 8     | deep_mlp    | 0.0005 | sgd        | 82.96%    | 86.23%    | 118,650  | 26.29  |
| 2     | baseline_mlp| 0.001  | sgd        | 85.74%    | 86.08%    | 203,530  | 27.88  |
| 12    | simple_cnn  | 0.0005 | sgd        | 83.56%    | 85.57%    | 421,642  | 68.26  |
| 4     | baseline_mlp| 0.0005 | sgd        | 83.72%    | 84.37%    | 203,530  | 26.20  |

### 2.3 Nhận xét kết quả grid search

* **Adam luôn tốt hơn SGD** ở cùng lr — gap **2–6%** ở cả 4 kiến trúc. Đúng kỳ vọng trên dataset nhỏ + ít epoch.
* **simple_cnn + adam lr=1e-3** là tổ hợp tốt nhất (**92.33%**) — CNN tận dụng được cấu trúc không gian ảnh.
* **resnet_mini + adam lr=5e-4** đứng thứ nhì (**92.20%**) — gần bằng simple_cnn nhưng **train_acc = 95.60%**, cho thấy overfit rõ hơn (gap 3,4% so với 0,46% của simple_cnn).
* **baseline_mlp / deep_mlp** đạt ~89–90% — không tệ cho fully-connected, nhưng không beat CNN.
* **SGD ở lr=0.0005** dưới học rất chậm (84–89% sau 10 epoch).
* **Tốc độ:** `baseline_mlp`/`deep_mlp` nhanh nhất (~27–29 s/epoch), `simple_cnn` ~75 s/epoch, `resnet_mini` ~218 s/epoch (chậm gấp **2,9×** simple_cnn) — không đáng đổi vì chỉ nhỉnh hơn 0,13% val_acc.
* **Pilot `compare_architectures` riêng** (2 epoch, 1e-3) cũng xác nhận thứ tự kiến trúc:
  - `simple_cnn` (2 epoch, 1e-3): val_acc = **89.48%** (421,642 params, 96,30 s/epoch)
  - `resnet_mini` (2 epoch, 1e-3): val_acc = **88.63%** (174,970 params, 258,94 s/epoch)
  - `baseline_mlp` (2 epoch, 1e-3): val_acc = **87.05%** (203,530 params, 30,92 s/epoch)
  - `deep_mlp` (2 epoch, 1e-3): val_acc = **86.10%** (118,650 params, 33,36 s/epoch)

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

| Metric | Pilot (3 epoch) | Grid thật (10 epoch) |
|--------|-----------------|---------------------|
| Train accuracy | 88.79% | **92.79%** |
| Val accuracy   | **90.12%** | **92.33%** |
| Val loss       | 0.27 | **0.21** |

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
| `src/training/hyperparameter_tuning.py` | `HyperparameterTuner`, `TrialResult`, grid search. |
| `src/training/hyperopt_scheduler.py` | `HyperoptScheduler`, `SchedulerConfig`, writer (`results/runs/hyperopt.{json,md}` + `_best.json`). |
| `configs/hyperopt_config.yaml` | Search-space định nghĩa (grid search). |
| `run_experiments.py` | Entry point end-to-end của Người 1 (có `--smoke`, `--strategy`, `--epochs`). |

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
├── hyperopt.json          # 4,985 bytes — 16 trial raw rows (real)
├── hyperopt.md            # 1,730 bytes — markdown table (real)
├── hyperopt_best.json     # 216 bytes — best config
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
| Grid search với 16 trials × 10 epoch × 4 architectures | ✅ pass (160 epoch CPU) |
| Grid search exhaustive (không bỏ sót tổ hợp) | ✅ pass (unit) |
| Real training 1 epoch trên FashionMNIST thật | ✅ pass (val_acc = 86.98% cho simple_cnn @ lr=1e-3) |
| Real training 3 epochs trên FashionMNIST thật | ✅ pass (val_acc = **90.12%**) |
| Save checkpoint `best_model.pt` | ✅ pass (1,690,359 bytes) |
| Persist results → `results/runs/hyperopt.{json,md,_best.json}` | ✅ pass |

### Reproduce

```bash
# Forward + loss-decrease test
python run_experiments.py --smoke

# Full grid search (16 trials × 10 epoch, 160 epoch)
python run_experiments.py --batch-size 128
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
*(Trương Hoài Tú, 2026-10-12 — kết quả thật từ grid search 16 trials × 10 epoch + real training)*

# Progress Status - FashionMNIST PyTorch Project

**Last Updated:** 2026-10-15 16:35
**Project Status:** Phase 2 COMPLETE — Người 1 ✅ DONE; Người 2 ✅ DONE; Người 3 ✅ DONE
**Current Milestone:** Full Pipeline Integration — Test Accuracy: 92.10%
**Last Run:** Official full training 15 epochs — best epoch 11, val_loss=0.215334, val_acc=92.88%; last epoch 15, val_acc=93.03%; Evaluation test_acc=92.10%

---

## Project Context

| Field | Value |
|-------|-------|
| **Project Name** | FashionMNIST Classification |
| **Domain** | Computer Vision / Image Classification |
| **Framework** | PyTorch 2.14.1+cpu |
| **Python Version** | 3.8+ |
| **Start Date** | 2026-10-11 |
| **Target Accuracy** | 90%+ |
| **Dataset Size** | 60,000 train / 10,000 test |
| **Device** | CPU (CUDA not available) |

---

## What / Why / Where-When-Which / How

### What?
- **Problem:** Multi-class image classification on FashionMNIST dataset
- **Data:** 28x28 grayscale images, 10 fashion categories
- **Output:** Class label prediction with confidence score

### Why?
- **Motivation:** Learn PyTorch fundamentals with a real dataset
- **Justification:** FashionMNIST is more challenging than MNIST, better for learning
- **Benefits:** Transferable skills to other CV tasks

### Where / When / Which?
- **Where:** Local machine (Windows, PowerShell)
- **When:** October 2026
- **Which Tools:** PyTorch 2.14, Matplotlib, Seaborn, scikit-learn, NumPy, Pandas

### How?
- **Approach:** Modular pipeline: EDA → Cleaning → Preprocessing → Split → (Model)
- **Data Flow:** Raw → Preprocessed → Split → Train/Val/Test
- **Measurement:** Accuracy, Precision, Recall, F1-Score (Phase 2+)

---

## Phase 1 Results (2026-10-11)

### ✅ Step 1: EDA (Exploratory Data Analysis)
- **Files:** `src/data/dataset.py`, `src/data/eda.py`
- **Outputs:**
  - `results/runs/class_distribution.png`
  - `results/runs/sample_images.png`
  - `results/runs/pixel_histogram.png`
  - `results/runs/per_class_samples.png`
- **Key Findings:**
  - 60,000 training samples, perfectly balanced (6,000 per class)
  - 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
  - Image shape: (1, 28, 28), dtype: float32
  - Pixel value range: [0, 1]
  - Mean: 0.286, Std: 0.353

### ✅ Step 2: Data Cleaning
- **File:** `src/data/preprocessing.py` → `DataCleaner`
- **Result:** 60,000/60,000 samples valid (100%), 0 invalid
- **Notes:** FashionMNIST is clean by default; no NaN/Inf values, all labels valid

### ✅ Step 3: Preprocessing
- **File:** `src/data/preprocessing.py` → `DataPreprocessor`
- **Configuration:**
  - Normalization: enabled
  - Mean: 0.2860, Std: 0.3530
  - Augmentation: disabled (Phase 1)
- **Computed Statistics:**
  - Mean: 0.2863
  - Std: 0.3540
  - Min: 0.0000, Max: 1.0000
- **Image range after normalize:** [-0.8102, 2.0227]

### ✅ Step 4: Data Split
- **File:** `src/data/split.py` → `DataSplitter`, `DataSplitManager`
- **Configuration:**
  - Train ratio: `5/6` (~83.3%)
  - Val ratio: `1/6` (~16.7%)
  - Random seed: 42 (reproducibility)
  - Stratified split: enabled (preserves class distribution)
- **Result:**
  - Train: 50,000 samples (5,000/class)
  - Val: 10,000 samples (1,000/class)
  - Test: 10,000 samples (1,000/class, dataset's own official split)
- **Saved Info:** `data/splits/split_info.txt`

---

## Architecture Snapshot

### System Architecture (Phase 1 + Phase 2 Người 1)
```
┌──────────────────────────────────────────────────────────────┐
│                     Data Preparation (Phase 1) ✅            │
│  ┌────────┐  ┌────────┐  ┌─────────────┐  ┌────────────┐  │
│  │  EDA   │→ │Cleaning│→ │Preprocessing│→ │  Splitting │   │
│  │  ✅    │  │   ✅   │  │     ✅      │  │     ✅     │   │
│  └────────┘  └────────┘  └─────────────┘  └────────────┘  │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│       Model Development (Phase 2 - Người 1) ✅ DONE         │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │
│  │  Baseline  │→ │  Selection │→ │ Hyperparameter Tune│    │
│  │    ✅      │  │    ✅      │  │        ✅          │    │
│  └────────────┘  └────────────┘  └────────────────────┘    │
│  BaselineMLP ✅  DeepMLP ✅  SimpleCNN ✅  ResNetMini ✅   │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  Training & Evaluation (Phase 2 - Người 2/3 + Phase 3) ✅   │
│  ┌────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐   │
│  │ Train  │→ │Evaluate  │→ │  Errors  │→ │ Save/Load   │  │
│  │   ✅   │  │    ✅    │  │    ✅    │  │     ✅      │  │
│  └────────┘  └──────────┘  └──────────┘  └─────────────┘   │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│               Reporting & Experiments (Phase 4) ✅           │
│  ┌─────────────────┐  ┌──────────────────────────────┐      │
│  │ Experiment Log  │→ │       Final Report            │      │
│  └─────────────────┘  └──────────────────────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

### Modules

| Module | File | Status | Ghi chú |
|--------|------|--------|---------|
| Dataset Loader | `src/data/dataset.py` | ✅ Complete | Phase 1 |
| EDA | `src/data/eda.py` | ✅ Complete | Phase 1 |
| Cleaning | `src/data/preprocessing.py` | ✅ Complete | Phase 1 |
| Preprocessing | `src/data/preprocessing.py` | ✅ Complete | Phase 1 |
| Data Split | `src/data/split.py` | ✅ Complete | Phase 1 (bug fix included) |
| Baseline MLP | `src/models/baseline_model.py` | ✅ Complete | **Người 1** |
| Deep MLP | `src/models/architectures/mlp.py` | ✅ Complete | **Người 1** |
| Simple CNN | `src/models/architectures/simple_cnn.py` | ✅ Complete | **Người 1** |
| ResNet Mini | `src/models/architectures/advanced_cnn.py` | ✅ Complete | **Người 1** |
| ModelFactory | `src/models/model_factory.py` | ✅ Complete | **Người 1** |
| Model Selection | `src/models/model_selection.py` | ✅ Complete | **Người 1** |
| HyperparameterTuner | `src/training/hyperparameter_tuning.py` | ✅ Complete | **Người 1** |
| HyperoptScheduler | `src/training/hyperopt_scheduler.py` | ✅ Complete | **Người 1** |
| `run_experiments.py` | `run_experiments.py` | ✅ Complete | **Người 1** entry point |
| Trainer | `src/training/trainer.py` | ✅ Complete | **Người 2** |
| Callbacks | `src/training/callbacks.py` | ✅ Complete | **Người 2** |
| Visualizer (loss) | `src/training/visualizer.py` | ✅ Complete | **Người 2** |
| Evaluator | `src/evaluation/evaluator.py` | ✅ Complete | **Người 3** |
| Error Analysis | `src/evaluation/error_analysis.py` | ✅ Complete | **Người 3** |
| Model I/O | `src/evaluation/model_io.py` | ✅ Complete | **Người 3** |
| Confusion Matrix | `src/evaluation/confusion_matrix.py` | ✅ Complete | **Người 3** |
| Prediction Visualizer | `src/evaluation/visualizer.py` | ✅ Complete | **Người 3** |

---

## Active Module: Người 3 — Evaluation + Save/Load + Display

- **Describe** — Đánh giá model đã train trên test set: accuracy, precision, recall, F1, confusion matrix, error analysis, predicted vs actual.
- **Goal / purpose** — Hoàn thành Phase 2 và tạo báo cáo Người 3.
- **Pipeline** — Load checkpoint → ModelEvaluator → ConfusionMatrixPlotter → ErrorAnalyzer → PredictionVisualizer → Final model save.
- **How / experiment plan (sub-tasks)**
  - [x] Load best checkpoint từ Người 2 (`trained_model.pt`).
  - [x] Compute test metrics: accuracy, precision, recall, F1.
  - [x] Analyze common confusions (Shirt ↔ T-shirt/top là nhiều nhất).
  - [x] Generate predictions plot và confusion matrix.
  - [x] Save final model với metrics.
- **Status** — Describe ✅ / Implement ✅ / Run ✅ / Commit ✅
- **Last run** — `python src/evaluation/run_evaluation.py`, test_acc=92.10%, precision=92.11%, recall=92.10%, f1=92.09%.

## Module Log

| # | Module | Ngày | Status | Commit | Ghi chú |
|---|--------|------|--------|--------|---------|
| 0 | Scaffold và Phase 1 data preparation | 2026-10-11 | ✅ | — | Lịch sử chi tiết được giữ trong mục Historical Module Log. |
| 1 | Người 1 — Models + Hyperparameter | 2026-10-12 | ✅ | — | Best config: `simple_cnn`, Adam, `lr=0.001`. |
| 2 | Người 2 — Training + Visualization | 2026-10-14 | ✅ | — | 27 tests passed; full run 15 epochs; artifacts đã bàn giao. |
| 3 | Người 3 — Evaluation + Save/Load + Display | 2026-10-15 | ✅ | 232adce | test_acc=92.10%, confusion matrix, predictions plot |

## Historical Module Log

| Date | Module | Action | Result |
|------|--------|--------|--------|
| 2026-10-11 22:30 | Documentation | Created README.md, AI_Usage.md, progress_status.md, audit_code.md (audit_code.md removed later) | ✅ |
| 2026-10-11 22:30 | Structure | Scaffolded folder structure | ✅ |
| 2026-10-11 22:30 | EDA | Created eda.py, ran visualization | ✅ |
| 2026-10-11 22:31 | Cleaning | Created DataCleaner, validated 60000/60000 samples | ✅ |
| 2026-10-11 22:31 | Preprocessing | Created DataPreprocessor, computed mean=0.286, std=0.354 | ✅ |
| 2026-10-11 22:31 | Data Split | Created DataSplitManager, split into 50000/10000/10000 | ✅ |
| 2026-10-11 22:32 | Pipeline Test | Ran end-to-end Phase 1 pipeline | ✅ |
| 2026-10-11 22:35 | Re-run Phase 1 | Re-ran pipeline for final verification | ✅ |
| 2026-10-11 22:36 | Re-run EDA | Re-generated all EDA plots | ✅ |
| 2026-10-12 07:01 | Data Layout | Moved 70k split PNGs from data/raw/ → data/processed/ (proper input vs output separation); updated split.py default + data/README.md | ✅ |
| 2026-10-12 08:xx | Người 1 | BaselineMLP, DeepMLP, SimpleCNN, ResNetMini implemented via ModelFactory | ✅ |
| 2026-10-12 09:xx | Người 1 | HyperparameterTuner + HyperoptScheduler with grid search | ✅ |
| 2026-10-12 09:xx | Người 1 | Bug fix: split.py import path + stratified shuffle | ✅ |
| 2026-10-12 09:57 | Người 1 | Full grid search 8 trials × 3 epochs → best val_acc=89.97% (simple_cnn + adam + lr=0.001) | ✅ |
| 2026-10-12 10:xx | Người 1 | End-to-end pipeline verified + `docs/TEAM_GUIDE.md` + `nguoi1_models_report.md` | ✅ |
| 2026-10-12 11:10 | Người 1 | **Phase 2 Người 1 — HOÀN THÀNH 100% (16/16 tasks)** | ✅ |
| 2026-10-12 12:32 | Người 1 | **Re-run grid search 8 trials × 3 epoch thật** → best val_acc=90.53% (simple_cnn + adam + lr=0.001) | ✅ |
| 2026-10-12 13:15 | Người 1 | **Real training 3 epoch thật với best config** → val_acc=90.12% → saved `models/checkpoints/best_model.pt` (1.69MB) | ✅ |
| 2026-10-21 21:5x | Người 1 | **Bỏ random search** khỏi project; grid search mở rộng lên **16 trials × 10 epoch × 4 kiến trúc** (tổng 160 epoch) — tất định, phủ kín | ✅ |
| 2026-10-22 02:04 | Người 1 | **Chạy thật grid search 16 trials × 10 epoch** (~4,5 giờ CPU) → best val_acc=**92.33%** (simple_cnn + adam + lr=0.001, trial 9); runner #2 resnet_mini + adam + lr=0.0005 (92.20%). Artifacts `results/runs/hyperopt.{json,md,_best.json}` đã cập nhật; docs đã đồng bộ | ✅ |
| 2026-10-22 | Data Split | **Đổi split từ 54k/6k → 50k/10k/10k** (`val_ratio=1/6`). Đồng thời fix bug phân tầng: `_stratified_split` gộp rồi xáo trộn toàn cục khiến `create_split_from_existing` cắt đại trà ⇒ lệch class (val 947–1.040/lớp). Sau fix: đúng **5.000 / 1.000 / 1.000** mỗi lớp. `split_pipeline`, `run_training.py`, `run_experiments.py` dùng chung `DEFAULT_VAL_RATIO`. 27/27 tests pass | ✅ |
| 2026-10-22 | Documentation | Đồng bộ toàn bộ docs theo split 50k/10k/10k và trạng thái Phase 2–3 hoàn tất | ✅ |

---

## Decisions Log

| Ngày | Quyết định | Lựa chọn khác | Lý do |
|------|------------|---------------|------|
| 2026-10-11 | Dùng cấu trúc module, stratified split, seed 42 và normalize FashionMNIST | Cấu trúc phẳng, split ngẫu nhiên, seed khác | Dễ bảo trì, giữ phân phối lớp và tái lập kết quả. |
| 2026-10-11 | Chạy CPU khi CUDA không khả dụng | Ép dùng CUDA | Môi trường hiện tại không có CUDA; CPU vẫn đáp ứng FashionMNIST. |
| 2026-10-14 | Dùng cấu hình bàn giao của Người 1: `simple_cnn`, Adam, `lr=0.001`, batch `128`, dropout `0.5` | Đổi architecture hoặc hyperparameter | Giữ nhất quán với kết quả lựa chọn model của Người 1. |
| 2026-10-14 | Entry point cấp cao là `main.py`, gọi orchestration `run_training.py` | Gọi trực tiếp `run_training` từ bên ngoài | Chạy toàn bộ pipeline bằng một lệnh và giữ orchestration tách module. |
| 2026-10-14 | Chọn best theo `val_loss`, `patience=5`, `min_delta=0.0001`; lưu cả best và last checkpoint | Chỉ theo dõi accuracy hoặc chỉ lưu một checkpoint | `val_loss` ổn định hơn để phát hiện ngừng cải thiện và hỗ trợ khôi phục model tốt nhất. |

## Historical Decisions Log

```
[2026-10-11 22:30] Decision: Use modular folder structure | Reasoning: Separation of concerns, easier to test and maintain
[2026-10-11 22:30] Decision: Use stratified split | Reasoning: Preserve class distribution in train/val/test
[2026-10-11 22:30] Decision: Random seed = 42 | Reasoning: Reproducibility across runs
[2026-10-11 22:30] Decision: Train/Val/Test = 5/6 : 1/6 : separate test set | Reasoning: 50k/10k/10k gives clean per-class counts (5,000/1,000/1,000 out of 6,000)
[2026-10-11 22:30] Decision: Normalize to mean=0.286, std=0.353 | Reasoning: Standard FashionMNIST values, faster convergence
[2026-10-11 22:30] Decision: Batch size = 64 | Reasoning: Standard choice, good balance of speed and accuracy
[2026-10-11 22:30] Decision: CPU only (CUDA unavailable) | Reasoning: PyTorch CPU mode, acceptable for FashionMNIST
```

---

## Risks & Mitigations

| Rủi ro | Khả năng | Tác động | Giảm thiểu | Phát hiện |
|--------|----------|----------|------------|-----------|
| Overfitting | Medium | High | Dropout, early stopping, chọn best theo `val_loss` | So sánh train/validation curves |
| Slow training on CPU | High | Medium | Giới hạn epoch hợp lý, dùng model nhỏ | Theo dõi thời gian từng epoch |
| Poor generalization | Medium | High | Validation set, đánh giá test và per-class ở Người 3 | Test accuracy và confusion matrix |

## Open Questions

- [x] Người 3 đã hoàn tất evaluation, save/load và predicted-vs-actual.
- [ ] Full pipeline integration test và báo cáo tổng hợp đã hoàn tất chưa?
- [x] Người dùng đã nghiệm thu kết quả Người 2.
- [x] Đã được phép xóa spec/plan triển khai sau khi chuyển nội dung cần thiết vào báo cáo.
- [x] Người 3 evaluation đã hoàn thành với test_acc=92.10%.

## Definition of Done (áp dụng cho module Người 2)

- [x] Đúng yêu cầu training loop và visualize loss trong kế hoạch.
- [x] Code chạy end-to-end bằng `python main.py`.
- [x] Unit/integration tests pass: `27 passed`.
- [x] `last_model.pt` được đánh dấu local-only và không đưa vào artifact tracked.
- [x] Code đã được self-review và review cuối cho path CPU/default.
- [x] Có thể giải thích pipeline, cấu hình và semantics best/last checkpoint.
- [x] `progress_status.md` và báo cáo Người 2 đã cập nhật.
- [x] Commit Conventional Commit cho thay đổi tài liệu này.

---

## Milestones

- [x] ~~Project setup & documentation~~ (2026-10-11 22:30)
- [x] ~~EDA & data exploration~~ (2026-10-11 22:31)
- [x] ~~Data preprocessing pipeline~~ (2026-10-11 22:31)
- [x] ~~Train/val/test split~~ (2026-10-11 22:32)
- [x] ~~Phase 1 end-to-end test~~ (2026-10-11 22:32)

### Phase 2 - The plan (3 người, align với đề)

- [x] **Người 1: Trương Hoài Tú — Build Network + Hyperparameter Experiment ✅ DONE**
  - [x] Baseline MLP model (`baseline_model.py`)
  - [x] Multiple architectures — DeepMLP, SimpleCNN, ResNetMini (`architectures/*.py`)
  - [x] ModelFactory registry (4 models)
  - [x] compare_architectures / model_selection
  - [x] Hyperparameter tuning — Grid search 16 trials × 10 epoch (4 architectures)
  - [x] Best model + best params identified (simple_cnn + adam + lr=0.001 → 92.33% grid 16×10)
  - [x] `run_experiments.py` entry point (`--smoke`, `--strategy`, `--epochs`)
  - [x] Bug fix: split.py import path + stratified shuffle
  - [x] `docs/TEAM_GUIDE.md` — hướng dẫn cho cả team
  - [x] `docs/reports/nguoi1_models_report.md` — báo cáo chi tiết
- [x] **Người 2: Training Loop + Visualize Loss** ✅ DONE
  - [x] Training loop (forward, loss, backward, optimize) → `src/training/trainer.py`
  - [x] Callbacks → `src/training/callbacks.py` (early stopping, checkpoint)
  - [x] Loss/optimizer factories → `src/training/losses.py`, `src/training/optimizers.py`
  - [x] Loss curve visualizer → `src/training/visualizer.py`
  - [x] Orchestration và exports → `run_training.py`, `src/training/__init__.py`
  - [x] `main.py` entry point cho Người 2
  - [x] Training tests, full run và artifacts bàn giao
- [x] **Người 3: Evaluate + Save/Load + Display** ✅ DONE
  - [x] Model evaluation (test_acc=92.10%) → `src/evaluation/evaluator.py`
  - [x] Save/Load model → `src/evaluation/model_io.py`
  - [x] Confusion matrix → `src/evaluation/confusion_matrix.py`
  - [x] Error analysis → `src/evaluation/error_analysis.py`
  - [x] Predicted vs actual images → `src/evaluation/visualizer.py`
  - [x] `src/evaluation/run_evaluation.py` entry point
- [ ] **Final report & visualization** ⏳ PENDING

---

## Results (Phase 1 - Data Preparation)

### Dataset Statistics
- **Total samples:** 60,000 training + 10,000 test
- **Image shape:** (1, 28, 28)
- **Mean:** 0.2860
- **Std:** 0.3530
- **Min:** 0.0000
- **Max:** 1.0000
- **Class balance:** Perfectly balanced (6,000 per class)

### Data Quality
- **Valid samples:** 60,000/60,000 (100%)
- **Invalid samples:** 0
- **Missing values:** None
- **Corrupt images:** None

### Data Splits
| Split | Samples | Per class |
|-------|---------|-----------|
| Train | 50,000 | 5,000 |
| Val   | 10,000 | 1,000 |
| Test  | 10,000 | 1,000 |

---

## Next Steps (Phase 2 - 3 người, align với đề)

### ✅ Người 2: Training Loop + Visualize Loss — HOÀN THÀNH

Đã triển khai:

1. **Training core** — `src/training/trainer.py`, `src/training/losses.py`, `src/training/optimizers.py`:
   - `train_one_epoch(model, loader, optimizer, device)` → trả train_loss, train_acc
   - `evaluate(model, loader, device)` → trả val_acc
   - Full training: loop nhiều epochs, record history
   - Dùng `ModelFactory.create("simple_cnn")` + config từ Người 1

2. **Callbacks** — `src/training/callbacks.py`:
   - `EarlyStopping`: monitor val_loss, patience=5
   - `Checkpoint`: save best model
   - `Logger`: record metrics per epoch

3. **Visualization** — `src/training/visualizer.py`:
   - Train vs val loss per epoch
   - Train vs val accuracy per epoch

4. **Orchestration và exports** — `run_training.py`, `src/training/__init__.py`, `main.py`:
   ```powershell
   python main.py
   ```

### 🔴 Người 3: Evaluate + Save/Load + Display — SẴN SÀNG NHẬN BÀN GIAO

1. **`src/evaluation/evaluator.py`** — đánh giá model:
   - Load best checkpoint
   - Compute test accuracy (kỳ vọng ≥ 90%)
   - Confusion matrix, precision, recall, F1

2. **`src/evaluation/model_io.py`** — save/load:
   - Save trained model (state_dict)
   - Load và resume

3. **`src/evaluation/error_analysis.py`** — phân tích lỗi:
   - Misclassified samples
   - Per-class accuracy

4. **`docs/reports/nguoi2_training_report.md`** — báo cáo Người 2

5. **`docs/reports/nguoi3_evaluation_report.md`** — báo cáo Người 3

### 📋 Best config từ Người 1 (dùng cho Người 2)

```python
{
    "model_name":    "simple_cnn",
    "learning_rate": 0.001,
    "optimizer":     "adam",
    "epochs":        10,       # Người 2 quyết định (recommend 10-15)
    "batch_size":    128,
    "weight_decay":  0.0,
    "dropout":       0.5,
}
```

Xem chi tiết tại `docs/PHASE2_PLAN.md` và `docs/reports/nguoi1_models_report.md`.

---

## Logs & Artifacts

### Phase 1 artifacts
- **Pipeline run log:** `logs/phase1_run.log`
- **EDA visualization log:** `logs/eda_visualization.log`
- **EDA plots:** `results/runs/*.png`
- **Split info:** `data/splits/split_info.txt`

### Phase 2 Người 1 artifacts
- **Hyperopt results:** `results/runs/hyperopt.json`, `hyperopt.md`, `hyperopt_best.json`
- **Architecture comparison:** `results/runs/architecture_selection.md`
- **Người 1 report:** `docs/reports/nguoi1_models_report.md`
- **Team guide:** `docs/TEAM_GUIDE.md`
- **Config:** `configs/hyperopt_config.yaml`
- **Entry point:** `run_experiments.py`

### Phase 2 Người 2 artifacts (complete)
- Best checkpoint: `models/checkpoints/trained_model.pt`
- Last checkpoint: `models/checkpoints/last_model.pt` (local-only)
- Full history: `results/runs/training_history.json`
- Loss curves: `results/runs/loss_curve.png`, `results/runs/accuracy_curve.png`
- Report: `docs/reports/nguoi2_training_report.md`

### Phase 2 Người 3 artifacts (pending)
- Evaluation results: `results/runs/evaluation.json`
- Confusion matrix: `results/runs/confusion_matrix.png`
- Error analysis: `results/runs/error_samples.png`
- Report: `docs/reports/nguoi3_evaluation_report.md`


---

## Người 2 completion addendum — 2026-10-14

### Implementation and verification

- [x] Training loop: forward → CrossEntropyLoss → backward → optimizer step.
- [x] Validation with eval mode and no-grad; weighted loss/accuracy.
- [x] Early stopping: monitor `val_loss`, mode `min`, patience `5`, min_delta `0.0001`.
- [x] Best/last checkpoints and best-weight restoration.
- [x] Loss and accuracy plots, JSON history and metadata.
- [x] YAML loader, CLI overrides, seed/device handling and public exports.
- [x] `python -m pytest tests/training -v` → **27 passed**.

### Effective configuration

```yaml
model: simple_cnn
dropout: 0.5
epochs: 15
batch_size: 128
learning_rate: 0.001
optimizer: adam
loss: cross_entropy
weight_decay: 0.0
device: auto  # resolved to CPU because CUDA is unavailable
seed: 42
num_workers: 0
augment: false
early_stopping: val_loss / min / patience=5 / min_delta=0.0001
```

### Official full run

Command: `python main.py`

- Completed 15/15 epochs.
- Best checkpoint: epoch 11, `val_loss=0.215334`, `val_acc=92.88%`.
- Last checkpoint: epoch 15, `val_loss=0.219415`, `val_acc=93.03%`.
- Final training accuracy: `94.93%`.

### Handoff artifacts

- `models/checkpoints/trained_model.pt`
- `models/checkpoints/last_model.pt` (local-only)
- `results/runs/training_history.json`
- `results/runs/loss_curve.png`
- `results/runs/accuracy_curve.png`
- `docs/reports/nguoi2_training_report.md`

### Decisions log

- [2026-10-14] Người 2 full-train dùng cấu hình đã bàn giao: simple_cnn, Adam, lr=0.001, batch=128, dropout=0.5.
- [2026-10-14] Dùng `main.py` làm entry point; `run_training.py` là orchestration module được gọi từ main.
- [2026-10-14] Chọn best theo `val_loss`, lưu cả best và last checkpoint, khôi phục best weights.
- [2026-10-14] Người 2 implementation and full run complete; Người 2 đã được nghiệm thu, Người 3 evaluation và final project acceptance remain pending.

# Progress Status - FashionMNIST PyTorch Project

**Last Updated:** 2026-10-12 13:15
**Project Status:** Phase 2 IN PROGRESS — Người 1 ✅ DONE (real results)
**Current Milestone:** Người 1: Network + Hyperparameter Experiment ✅
**Last Run:** Grid search 8 trials + Real training 3 epochs at 13:15 ✅ (val_acc=90.12% on real training, 90.53% in grid search)

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
  - Train ratio: 0.9 (90%)
  - Val ratio: 0.1 (10%)
  - Random seed: 42 (reproducibility)
  - Stratified split: enabled (preserves class distribution)
- **Result:**
  - Train: 54,000 samples (844 batches @ batch_size=64)
  - Val: 6,000 samples (94 batches)
  - Test: 10,000 samples (157 batches)
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
│  Training & Evaluation (Phase 2 - Người 2/3 + Phase 3) ⏳  │
│  ┌────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐   │
│  │ Train  │→ │Evaluate  │→ │  Errors  │→ │ Save/Load   │  │
│  │  ⏳    │  │   ⏳     │  │   ⏳     │  │    ⏳       │  │
│  └────────┘  └──────────┘  └──────────┘  └─────────────┘   │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│               Reporting & Experiments (Phase 4) ⏳            │
│  ┌─────────────────┐  ┌──────────────────────────────┐      │
│  │ Experiment Track│→ │       Final Report            │      │
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
| Trainer | `src/training/trainer.py` | ⏳ Empty | **Người 2** |
| Callbacks | `src/training/callbacks.py` | ⏳ Empty | **Người 2** |
| Visualizer (loss) | `src/training/visualizer.py` | ⏳ Empty | **Người 2** |
| Evaluator | `src/evaluation/evaluator.py` | ⏳ Empty | **Người 3** |
| Error Analysis | `src/evaluation/error_analysis.py` | ⏳ Empty | **Người 3** |
| Model I/O | `src/evaluation/model_io.py` | ⏳ Empty | **Người 3** |
| Experiment Tracker | `src/evaluation/experiment_tracker.py` | ⏳ Empty | **Người 3** |

---

## Module Log

| Date | Module | Action | Result |
|------|--------|--------|--------|
| 2026-10-11 22:30 | Documentation | Created README.md, AI_Usage.md, progress_status.md, audit_code.md (audit_code.md removed later) | ✅ |
| 2026-10-11 22:30 | Structure | Scaffolded folder structure | ✅ |
| 2026-10-11 22:30 | EDA | Created eda.py, ran visualization | ✅ |
| 2026-10-11 22:31 | Cleaning | Created DataCleaner, validated 60000/60000 samples | ✅ |
| 2026-10-11 22:31 | Preprocessing | Created DataPreprocessor, computed mean=0.286, std=0.354 | ✅ |
| 2026-10-11 22:31 | Data Split | Created DataSplitManager, split into 54000/6000/10000 | ✅ |
| 2026-10-11 22:32 | Pipeline Test | Ran end-to-end Phase 1 pipeline | ✅ |
| 2026-10-11 22:35 | Re-run Phase 1 | Re-ran pipeline for final verification | ✅ |
| 2026-10-11 22:36 | Re-run EDA | Re-generated all EDA plots | ✅ |
| 2026-10-12 07:01 | Data Layout | Moved 70k split PNGs from data/raw/ → data/processed/ (proper input vs output separation); updated split.py default + data/README.md | ✅ |
| 2026-10-12 08:xx | Người 1 | BaselineMLP, DeepMLP, SimpleCNN, ResNetMini implemented via ModelFactory | ✅ |
| 2026-10-12 09:xx | Người 1 | HyperparameterTuner + HyperoptScheduler with grid/random search | ✅ |
| 2026-10-12 09:xx | Người 1 | Bug fix: split.py import path + stratified shuffle | ✅ |
| 2026-10-12 09:57 | Người 1 | Full grid search 8 trials × 3 epochs → best val_acc=89.97% (simple_cnn + adam + lr=0.001) | ✅ |
| 2026-10-12 10:xx | Người 1 | End-to-end pipeline verified + `docs/TEAM_GUIDE.md` + `nguoi1_models_report.md` | ✅ |
| 2026-10-12 11:10 | Người 1 | **Phase 2 Người 1 — HOÀN THÀNH 100% (16/16 tasks)** | ✅ |
| 2026-10-12 12:32 | Người 1 | **Re-run grid search 8 trials × 3 epoch thật** → best val_acc=90.53% (simple_cnn + adam + lr=0.001) | ✅ |
| 2026-10-12 13:15 | Người 1 | **Real training 3 epoch thật với best config** → val_acc=90.12% → saved `models/checkpoints/best_model.pt` (1.69MB) | ✅ |

---

## Decisions Log

```
[2026-10-11 22:30] Decision: Use modular folder structure | Reasoning: Separation of concerns, easier to test and maintain
[2026-10-11 22:30] Decision: Use stratified split | Reasoning: Preserve class distribution in train/val/test
[2026-10-11 22:30] Decision: Random seed = 42 | Reasoning: Reproducibility across runs
[2026-10-11 22:30] Decision: Train/Val/Test = 0.9/0.1/1.0 | Reasoning: Standard split, FashionMNIST has separate test set
[2026-10-11 22:30] Decision: Normalize to mean=0.286, std=0.353 | Reasoning: Standard FashionMNIST values, faster convergence
[2026-10-11 22:30] Decision: Batch size = 64 | Reasoning: Standard choice, good balance of speed and accuracy
[2026-10-11 22:30] Decision: CPU only (CUDA unavailable) | Reasoning: PyTorch CPU mode, acceptable for FashionMNIST
```

---

## Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Overfitting | Medium | High | Add dropout, early stopping (Phase 2+) |
| Slow training on CPU | High | Medium | Use smaller model, limit epochs |
| Poor generalization | Medium | High | Use validation set, data augmentation |

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
  - [x] Hyperparameter tuning — Grid search 8 trials + Random search
  - [x] Best model + best params identified (simple_cnn + adam + lr=0.001 → 89.97%)
  - [x] `run_experiments.py` entry point (`--smoke`, `--strategy`, `--n-trials`, `--epochs`)
  - [x] Bug fix: split.py import path + stratified shuffle
  - [x] `docs/TEAM_GUIDE.md` — hướng dẫn cho cả team
  - [x] `docs/reports/nguoi1_models_report.md` — báo cáo chi tiết
- [ ] **Người 2: Training Loop + Visualize Loss** ⏳ CHƯA LÀM
  - [ ] Training loop (forward, loss, backward, optimize) → `src/training/trainer.py`
  - [ ] Callbacks → `src/training/callbacks.py` (early stopping, checkpoint)
  - [ ] Loss curve visualizer → `src/training/visualizer.py`
  - [ ] `main.py` entry point cho Người 2
- [ ] **Người 3: Evaluate + Save/Load + Display** ⏳ CHƯA LÀM
  - [ ] Model evaluation (accuracy ≥ 90%) → `src/evaluation/evaluator.py`
  - [ ] Save/Load model → `src/evaluation/model_io.py`
  - [ ] Display predicted vs actual images → `src/training/visualizer.py` hoặc evaluator
- [ ] Final report & visualization

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
| Split | Samples | Batches (bs=64) |
|-------|---------|------------------|
| Train | 54,000 | 844 |
| Val   | 6,000  | 94  |
| Test  | 10,000 | 157 |

---

## Next Steps (Phase 2 - 3 người, align với đề)

### 🔴 Người 2: Training Loop + Visualize Loss — BẮT ĐẦU NGAY

Cần implement:

1. **`src/training/trainer.py`** — training loop chính:
   - `train_one_epoch(model, loader, optimizer, device)` → trả train_loss, train_acc
   - `evaluate(model, loader, device)` → trả val_acc
   - Full training: loop nhiều epochs, record history
   - Dùng `ModelFactory.create("simple_cnn")` + config từ Người 1

2. **`src/training/callbacks.py`** — callbacks:
   - `EarlyStopping`: monitor val_loss, patience=5
   - `Checkpoint`: save best model
   - `Logger`: record metrics per epoch

3. **`src/training/visualizer.py`** — vẽ loss/acc curve:
   - Train vs val loss per epoch
   - Train vs val accuracy per epoch

4. **`main.py`** — entry point cho training:
   ```powershell
   python main.py --model simple_cnn --epochs 10 --lr 0.001 --optimizer adam
   ```

### 🔴 Người 3: Evaluate + Save/Load + Display — CHỜ NGƯỜI 2 XONG

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

### Phase 2 Người 2 artifacts (pending)
- Trained checkpoint: `models/checkpoints/best_model.pt`
- Training log: `logs/training.log`
- Loss curves: `results/runs/loss_curve.png`, `results/runs/accuracy_curve.png`
- Report: `docs/reports/nguoi2_training_report.md`

### Phase 2 Người 3 artifacts (pending)
- Evaluation results: `results/runs/evaluation.json`
- Confusion matrix: `results/runs/confusion_matrix.png`
- Error analysis: `results/runs/error_samples.png`
- Report: `docs/reports/nguoi3_evaluation_report.md`

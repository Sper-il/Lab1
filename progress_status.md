# Progress Status - FashionMNIST PyTorch Project

**Last Updated:** 2026-10-11 22:36  
**Project Status:** Phase 1 COMPLETE (EDA → Data Split)  
**Current Milestone:** Data Preparation Pipeline ✅  
**Last Run:** Final verification at 22:35-22:36 ✅

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

### System Architecture (Phase 1)
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
│                  Model Development (Phase 2) ⏳              │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │
│  │  Baseline  │→ │  Selection │→ │ Hyperparameter Tune│    │
│  └────────────┘  └────────────┘  └────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                  Training & Evaluation (Phase 3) ⏳          │
│  ┌────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐    │
│  │ Train  │→ │Evaluate  │→ │  Errors  │→ │ Save/Load   │   │
│  └────────┘  └──────────┘  └──────────┘  └─────────────┘    │
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

| Module | File | Status |
|--------|------|--------|
| Dataset Loader | `src/data/dataset.py` | ✅ Complete |
| EDA | `src/data/eda.py` | ✅ Complete |
| Cleaning | `src/data/preprocessing.py` | ✅ Complete |
| Preprocessing | `src/data/preprocessing.py` | ✅ Complete |
| Data Split | `src/data/split.py` | ✅ Complete |
| Baseline Model | `src/models/baseline_model.py` | ⏳ Empty |
| Model Selection | `src/models/model_selection.py` | ⏳ Empty |
| Trainer | `src/training/trainer.py` | ⏳ Empty |
| Hyperparameter Tuning | `src/training/hyperparameter_tuning.py` | ⏳ Empty |
| Evaluator | `src/evaluation/evaluator.py` | ⏳ Empty |
| Error Analysis | `src/evaluation/error_analysis.py` | ⏳ Empty |
| Model I/O | `src/evaluation/model_io.py` | ⏳ Empty |
| Experiment Tracker | `src/evaluation/experiment_tracker.py` | ⏳ Empty |

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

- [ ] **Người 1:** Build Network + Hyperparameter Experiment
  - [ ] Baseline MLP model
  - [ ] Multiple architectures (MLP, SimpleCNN, ResNet)
  - [ ] Hyperparameter tuning
  - [ ] Best model + best params identified
- [ ] **Người 2:** Training Loop + Visualize Loss
  - [ ] Training loop (forward, loss, backward, optimize)
  - [ ] Callbacks, losses, optimizers
  - [ ] Visualize loss curve
- [ ] **Người 3:** Evaluate + Save/Load + Display
  - [ ] Model evaluation (accuracy ≥ 90%)
  - [ ] Save/Load model
  - [ ] Display predicted vs actual images
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

Theo đề bài, Phase 2 cần:

1. **Người 1:** Build neural network + Experiment với hyperparameters
   - File: `src/models/baseline_model.py`, `architectures/*.py`, `training/hyperparameter_tuning.py`

2. **Người 2:** Training loop (forward, loss, backward, optimize) + Visualize loss
   - File: `src/training/trainer.py`, `visualizer.py`

3. **Người 3:** Evaluate accuracy + Save/Load model + Display predictions
   - File: `src/evaluation/evaluator.py`, `model_io.py`, `visualizer.py`

Xem chi tiết tại `docs/PHASE2_PLAN.md`.

---

## Logs & Artifacts

- **Pipeline run log:** `logs/phase1_run.log`
- **EDA visualization log:** `logs/eda_visualization.log`
- **EDA plots:** `results/runs/*.png`
- **Split info:** `data/splits/split_info.txt`

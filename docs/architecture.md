# System Architecture

## Overview

FashionMNIST classification system using PyTorch.

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                     Data Preparation                          │
│  ┌────────┐  ┌────────┐  ┌─────────────┐  ┌────────────┐  │
│  │  EDA   │→ │Cleaning│→ │Preprocessing│→ │  Splitting │   │
│  └────────┘  └────────┘  └─────────────┘  └────────────┘  │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                  Model Development ✅ DONE                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │
│  │  Baseline  │→ │  Selection │→ │ Hyperparameter Tune│    │
│  │    ✅      │  │    ✅      │  │        ✅          │    │
│  └────────────┘  └────────────┘  └────────────────────┘    │
│  Best: simple_cnn + Adam lr=0,001 → val 92,33% (16 trials) │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                  Training & Evaluation ✅ DONE                │
│  ┌────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐    │
│  │ Train  │→ │Evaluate  │→ │  Errors  │→ │ Save/Load   │   │
│  │   ✅   │  │    ✅    │  │    ✅    │  │     ✅      │   │
│  └────────┘  └──────────┘  └──────────┘  └─────────────┘    │
│  Full 15 epochs CPU → test 92,10% (best epoch 11)          │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│               Reporting & Experiments ✅ DONE                 │
│  ┌─────────────────┐  ┌──────────────────────────────┐      │
│  │ Experiment Track│→ │       Final Report            │      │
│  │       ✅        │  │              ✅               │      │
│  └─────────────────┘  └──────────────────────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

## Implemented Modules (Phase 1 + Phase 2 hoàn thành)

### Data Module (`src/data/`)

1. **dataset.py** - FashionMNIST loading and DataLoader
   - TransformBuilder for train/test transforms
   - DataLoader factory function
   - Class name mapping

2. **eda.py** - Exploratory Data Analysis
   - Dataset statistics
   - Class distribution plots
   - Sample visualizations
   - Pixel intensity analysis

3. **preprocessing.py** - Data Cleaning & Preprocessing
   - Image/label validation
   - Dataset statistics computation
   - Normalization utilities
   - Batch analysis

4. **split.py** - Train/Validation/Test Split
   - Stratified splitting
   - Subset creation
   - DataSplitManager

## Phase 2: Model Development, Training & Evaluation ✅ DONE (3 người, align với đề)
Xem chi tiết tại `docs/PHASE2_PLAN.md`, `docs/PHASE2_FINAL_REPORT.md` và 3 báo cáo `docs/reports/nguoi{1,2,3}_*.md`.

**Người 1 - Build Network + Hyperparameter Experiment:**
- `src/models/baseline_model.py` - BaselineMLP
- `src/models/architectures/mlp.py` - DeepMLP
- `src/models/architectures/simple_cnn.py` - SimpleCNN
- `src/models/architectures/advanced_cnn.py` - ResNetMini
- `src/models/model_factory.py` - Factory pattern
- `src/models/model_selection.py` - So sánh models
- `src/training/hyperparameter_tuning.py` - Hyperparameter search
- `configs/hyperopt_config.yaml` - Search space

**Người 2 - Training Loop + Visualize Loss:**
- `src/training/trainer.py` - Training loop (forward, loss, backward, optimize)
- `src/training/callbacks.py` - Early stopping, checkpoint
- `src/training/losses.py` - Custom losses
- `src/training/optimizers.py` - Optimizer setup
- `src/training/visualizer.py` - Visualize loss curve
- `main.py` - Main training script

**Người 3 - Evaluate + Save/Load + Display:**
- `src/evaluation/evaluator.py` - Model evaluation (test 92,10%)
- `src/evaluation/metrics.py` - Accuracy, Precision, Recall, F1
- `src/evaluation/model_io.py` - Save/Load model (`final_model.pt`)
- `src/evaluation/visualizer.py` - Display predicted vs actual (`predictions.png`)
- `src/evaluation/confusion_matrix.py` - Confusion matrix (`confusion_matrix.png`)
- `src/evaluation/error_analysis.py` - Error analysis (top-5 confused pairs)

## Kết quả cuối (Phase 2 ✅ DONE)

- Best config (grid 16 trials × 10 epochs): `simple_cnn` + Adam + lr=0,001 → val 92,33% (`results/runs/hyperopt.md`).
- Full training 15 epochs CPU (`python main.py`): best epoch 11 (val_loss 0,215183, val_acc 92,85%), đỉnh val_acc 92,88% epoch 13.
- Test (`python src/evaluation/run_evaluation.py`): accuracy **92,10%**, precision/recall/F1 macro ~92,1%.
- Artifacts: `models/checkpoints/trained_model.pt` (best), `models/checkpoints/final_model.pt` (kèm metrics), `results/runs/training_history.json`, `loss_curve.png`, `accuracy_curve.png`, `confusion_matrix.png`, `predictions.png`.
- Báo cáo: `docs/reports/nguoi1_models_report.md`, `nguoi2_training_report.md`, `nguoi3_evaluation_report.md`, `docs/PHASE2_FINAL_REPORT.md`.

## Tech Stack

- **Framework:** PyTorch
- **Visualization:** Matplotlib, Seaborn
- **Metrics:** scikit-learn
- **Data:** NumPy, Pandas

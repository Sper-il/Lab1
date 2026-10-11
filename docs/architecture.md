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
│                  Model Development (Upcoming)                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │
│  │  Baseline  │→ │  Selection │→ │ Hyperparameter Tune│    │
│  └────────────┘  └────────────┘  └────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                  Training & Evaluation (Upcoming)              │
│  ┌────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐    │
│  │ Train  │→ │Evaluate  │→ │  Errors  │→ │ Save/Load   │   │
│  └────────┘  └──────────┘  └──────────┘  └─────────────┘    │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│               Reporting & Experiments (Upcoming)              │
│  ┌─────────────────┐  ┌──────────────────────────────┐      │
│  │ Experiment Track│→ │       Final Report            │      │
│  └─────────────────┘  └──────────────────────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

## Current Implementation (Phase 1)

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

## Upcoming Phases

### Phase 2: Model Development, Training & Evaluation (3 người, align với đề)
Xem chi tiết tại `docs/PHASE2_PLAN.md`.

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
- `src/evaluation/evaluator.py` - Model evaluation
- `src/evaluation/metrics.py` - Accuracy, Precision, Recall, F1
- `src/evaluation/model_io.py` - Save/Load model
- `src/evaluation/visualizer.py` - Display predicted vs actual
- `src/evaluation/confusion_matrix.py` - Confusion matrix
- `src/evaluation/error_analysis.py` - Error analysis

## Tech Stack

- **Framework:** PyTorch
- **Visualization:** Matplotlib, Seaborn
- **Metrics:** scikit-learn
- **Data:** NumPy, Pandas

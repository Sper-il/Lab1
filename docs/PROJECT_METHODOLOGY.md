# FashionMNIST Classification — Project Methodology

**Author:** Sper-il  
**Date:** 2026-10-15  
**Dataset:** [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)  
**Framework:** PyTorch 2.14+  
**GitHub:** https://github.com/Sper-il/Lab1

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Phase 1: Data Preparation](#2-phase-1-data-preparation)
3. [Phase 2: Model Development](#3-phase-2-model-development)
4. [Phase 3: Training & Evaluation](#4-phase-3-training--evaluation)
5. [Results Summary](#5-results-summary)
6. [Project Structure](#6-project-structure)

---

## 1. Project Overview

### 1.1 Problem Statement

Build a neural network to classify 28×28 grayscale images of fashion items into 10 categories:

| Class ID | Label |
|----------|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

### 1.2 Dataset Statistics

| Metric | Value |
|--------|-------|
| Training samples | 60,000 |
| Test samples | 10,000 |
| Image shape | 28×28 (1 channel) |
| Classes | 10 (balanced) |
| Target accuracy | ≥ 90% |

### 1.3 Technical Stack

| Component | Technology |
|-----------|------------|
| Framework | PyTorch 2.14 |
| Data processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Metrics | scikit-learn |
| Config | YAML |

---

## 2. Phase 1: Data Preparation

### 2.1 Step 1 — Exploratory Data Analysis (EDA)

**Objective:** Understand the dataset structure and characteristics.

**Tasks:**
- Load FashionMNIST dataset via `torchvision.datasets.FashionMNIST`
- Compute statistics: mean, std, min, max
- Visualize sample images from each class
- Analyze pixel intensity distribution
- Verify class balance

**Output Files:**
```
results/runs/
├── class_distribution.png    # Bar chart of class counts
├── sample_images.png          # Grid of sample images
├── pixel_histogram.png        # Pixel intensity histogram
└── per_class_samples.png      # One sample per class
```

**Key Findings:**
- 60,000 training samples, perfectly balanced (6,000 per class)
- Mean: 0.286, Std: 0.353
- Pixel range: [0, 1]
- No missing or corrupt data

### 2.2 Step 2 — Data Cleaning

**Objective:** Validate data quality and handle invalid samples.

**Tasks:**
- Check for NaN/Inf values
- Validate image shapes
- Verify label ranges
- Ensure no corrupt images

**Result:** 60,000/60,000 valid samples (100%)

### 2.3 Step 3 — Preprocessing

**Objective:** Normalize data for better training convergence.

**Tasks:**
- Normalize using computed mean and std
- Apply transforms: `transforms.Normalize(mean=0.286, std=0.353)`
- Disable augmentation (Phase 1)

**Configuration:**
```python
Normalize(mean=0.2863, std=0.3540)
```

### 2.4 Step 4 — Data Splitting

**Objective:** Divide data into train/validation/test sets.

**Tasks:**
- Split ratio: Train 90% / Validation 10% / Test 100% (separate)
- Use stratified sampling to preserve class distribution
- Set random seed = 42 for reproducibility
- Save split information

**Result:**
| Split | Samples | Batches (bs=64) |
|-------|---------|------------------|
| Train | 54,000 | 844 |
| Validation | 6,000 | 94 |
| Test | 10,000 | 157 |

**Implementation:** `src/data/split.py` → `DataSplitManager`

---

## 3. Phase 2: Model Development

### 3.1 Step 1 — Baseline Model

**Objective:** Establish a simple baseline to compare against.

**Model:** BaselineMLP
```python
BaselineMLP:
  Linear(784 → 256)
  ReLU
  Linear(256 → 128)
  ReLU
  Linear(128 → 10)
```

### 3.2 Step 2 — Architecture Exploration

**Objective:** Compare multiple architectures to find the best performer.

**Models Implemented:**

| Model | Architecture | Parameters |
|-------|--------------|------------|
| DeepMLP | 784→512→256→128→10 + Dropout | ~500K |
| SimpleCNN | 2 Conv layers + 2 FC layers | ~100K |
| ResNetMini | Residual blocks + skip connections | ~200K |

**Implementation:** `src/models/model_factory.py` → `ModelFactory`

### 3.3 Step 3 — Hyperparameter Tuning

**Objective:** Find optimal hyperparameters for training.

**Search Strategy:**
- Grid search + Random search
- 8 trials × 3 epochs per trial

**Hyperparameters Tested:**
| Parameter | Values |
|-----------|--------|
| Learning rate | 0.001, 0.0001 |
| Optimizer | Adam, SGD |
| Batch size | 64, 128 |
| Dropout | 0.3, 0.5 |

**Best Configuration Found:**
```yaml
model_name: simple_cnn
learning_rate: 0.001
optimizer: adam
batch_size: 128
dropout: 0.5
```

**Implementation:** `src/training/hyperparameter_tuning.py`

### 3.4 Step 4 — Model Selection

**Objective:** Select the best model based on validation accuracy.

**Selection Criteria:**
- Highest validation accuracy
- Reasonable model size
- Training time considerations

**Winner:** SimpleCNN (val_acc ~90%)

---

## 4. Phase 3: Training & Evaluation

### 4.1 Step 1 — Training Pipeline

**Objective:** Train the model with proper monitoring and checkpointing.

**Training Loop:**
```
for epoch in range(epochs):
    # Training phase
    model.train()
    for batch in train_loader:
        data, target = batch
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            data, target = batch
            output = model(data)
            # Compute metrics
```

**Callbacks:**
- **EarlyStopping:** Monitor val_loss, patience=5, min_delta=0.0001
- **Checkpoint:** Save best and last model weights

**Configuration (Final Run):**
| Parameter | Value |
|-----------|-------|
| Epochs | 15 |
| Batch size | 128 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | CrossEntropy |
| Dropout | 0.5 |
| Device | CPU |

**Implementation:** `src/training/trainer.py`, `src/training/callbacks.py`

### 4.2 Step 2 — Visualization

**Objective:** Visualize training progress.

**Generated Plots:**
- Loss curve (train vs val)
- Accuracy curve (train vs val)

**Output:** `results/runs/loss_curve.png`, `results/runs/accuracy_curve.png`

### 4.3 Step 3 — Evaluation

**Objective:** Evaluate trained model on test set.

**Metrics Computed:**
| Metric | Formula | Value |
|--------|---------|-------|
| Accuracy | Correct / Total | 92.10% |
| Precision (macro) | Avg precision per class | 92.11% |
| Recall (macro) | Avg recall per class | 92.10% |
| F1-Score (macro) | 2 × P × R / (P + R) | 92.09% |

**Implementation:** `src/evaluation/evaluator.py`

### 4.4 Step 4 — Error Analysis

**Objective:** Understand model misclassifications.

**Analysis:**
- Confusion matrix visualization
- Top confused class pairs
- Per-class accuracy breakdown

**Top Confused Pairs:**
| True Class | → Predicted | Count |
|------------|-------------|-------|
| Shirt | T-shirt/top | 95 |
| T-shirt/top | Shirt | 90 |
| Shirt | Pullover | 64 |
| Coat | Pullover | 63 |
| Coat | Shirt | 55 |

**Insight:** Shirt and T-shirt/top are frequently confused due to similar shapes.

**Implementation:** `src/evaluation/error_analysis.py`, `src/evaluation/confusion_matrix.py`

### 4.5 Step 5 — Visualization of Predictions

**Objective:** Visualize model predictions vs actual labels.

**Output:** `results/runs/predictions.png`
- Grid of sample images
- Color-coded: green = correct, red = wrong
- Shows both true and predicted labels

**Implementation:** `src/evaluation/visualizer.py`

### 4.6 Step 6 — Model Persistence

**Objective:** Save and load trained models.

**Saved Artifacts:**
| File | Description |
|------|-------------|
| `trained_model.pt` | Best model weights |
| `final_model.pt` | Final model with metrics |
| `training_history.json` | Epoch-by-epoch metrics |

**Implementation:** `src/evaluation/model_io.py`

---

## 5. Results Summary

### 5.1 Final Performance

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| Accuracy | 94.72% | 92.80% | **92.10%** |
| Best Epoch | — | 11/15 | — |

### 5.2 Training Progress

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|------------|----------|----------|---------|
| 1 | 0.585 | 0.351 | 79.03% | 87.08% |
| 5 | 0.265 | 0.237 | 90.41% | 91.42% |
| 10 | 0.190 | 0.222 | 92.91% | 92.63% |
| 15 | 0.141 | 0.230 | 94.72% | 92.80% |

### 5.3 Key Achievements

✅ **Target accuracy (90%) exceeded: 92.10%**  
✅ All 27 unit tests passed  
✅ Full pipeline: Data → Train → Evaluate → Report  
✅ Open source: GitHub repository

---

## 6. Project Structure

```
e:\lab1/
├── src/
│   ├── data/                    # Phase 1: Data preparation
│   │   ├── dataset.py          # FashionMNIST loader
│   │   ├── eda.py             # Exploratory analysis
│   │   ├── preprocessing.py    # Cleaning & transforms
│   │   └── split.py           # Train/val/test split
│   ├── models/                 # Phase 2: Model architectures
│   │   ├── architectures/      # MLP, CNN, ResNet
│   │   └── model_factory.py    # Model registry
│   ├── training/               # Phase 3: Training
│   │   ├── trainer.py         # Training loop
│   │   ├── callbacks.py       # Early stopping, checkpoint
│   │   ├── losses.py          # Loss functions
│   │   ├── optimizers.py      # Optimizer factory
│   │   └── visualizer.py      # Training plots
│   └── evaluation/             # Phase 3: Evaluation
│       ├── evaluator.py       # Metrics computation
│       ├── error_analysis.py   # Confusion analysis
│       ├── confusion_matrix.py # CM visualization
│       ├── visualizer.py      # Prediction display
│       └── model_io.py        # Save/load models
├── tests/training/             # Unit tests (27 tests)
├── models/checkpoints/        # Saved models
├── results/runs/              # Plots and outputs
├── docs/reports/              # Phase reports
├── configs/                   # YAML configs
├── main.py                    # Training entry point
└── README.md                  # Project documentation
```

---

## How to Reproduce

### 1. Clone the repository
```bash
git clone https://github.com/Sper-il/Lab1.git
cd Lab1
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run full pipeline
```bash
# Training
python main.py

# Evaluation
$env:PYTHONPATH = "."
python src/evaluation/run_evalution.py
```

### 4. Run tests
```bash
pytest tests/training -v
```

---

## Conclusion

This project demonstrates a complete machine learning workflow:

1. **Data Preparation** → Clean, preprocess, split data
2. **Model Development** → Design, compare, select models
3. **Hyperparameter Tuning** → Optimize training parameters
4. **Training** → Train with monitoring and checkpointing
5. **Evaluation** → Compute metrics, analyze errors

**Final Result:** 92.10% test accuracy on FashionMNIST, exceeding the 90% target.

---

*Document generated: 2026-10-15*

# FashionMNIST Classification with PyTorch

## 1. Project Overview

**Project Name:** FashionMNIST PyTorch Classification  
**Project Type:** Image Classification (Deep Learning)  
**Core Functionality:** Implement a neural network to classify 10 categories of fashion items from the FashionMNIST dataset  
**Target Users:** Students, Researchers, ML Engineers learning PyTorch  
**Status:** Phase 1–3 Complete — Test Accuracy **92.10%** (target 90%)

## 2. Problem Statement

Fashion-MNIST is a dataset of Zalando's article images, consisting of:
- **60,000** training samples
- **10,000** test samples
- **28x28** grayscale images
- **10 classes:** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

**Task:** Build a PyTorch neural network to accurately classify these fashion items.

## 3. Learning Objectives

- Master PyTorch Tensors operations
- Understand Datasets/Loaders with transforms
- Build and train neural networks
- Implement Autograd for gradient computation
- Use Optimization algorithms (SGD, Adam)
- Save and load trained models

## 4. Project Structure

```
e:\lab1/
├── data/
│   ├── raw/                # Original FashionMNIST dataset (gitignored)
│   │   ├── FashionMNIST/   # Original archive
│   │   └── _torchvision_cache/  # Torchvision download cache
│   ├── processed/          # Split images as PNG (gitignored, 70k files)
│   │   ├── train/<class_id>/*.png   # 50.000 train images
│   │   ├── val/<class_id>/*.png     # 10.000 val images
│   │   └── test/<class_id>/*.png    # 10.000 test images
│   ├── splits/             # Train/val/test split info
│   │   └── split_info.txt
│   └── README.md           # Data folder guide
├── src/
│   ├── data/               # Phase 1: Data preparation modules
│   │   ├── __init__.py
│   │   ├── dataset.py      # FashionMNIST loader
│   │   ├── eda.py          # Exploratory Data Analysis
│   │   ├── preprocessing.py # Cleaning & preprocessing
│   │   └── split.py        # Train/val/test split (50k/10k/10k, stratified)
│   ├── models/             # Phase 2: Neural network architectures
│   │   ├── __init__.py
│   │   ├── baseline_model.py    # BaselineMLP
│   │   ├── model_factory.py     # Model registry (4 models)
│   │   ├── model_selection.py   # compare_architectures
│   │   └── architectures/       # DeepMLP, SimpleCNN, ResNetMini
│   │       ├── __init__.py
│   │       ├── mlp.py
│   │       ├── simple_cnn.py
│   │       └── advanced_cnn.py
│   ├── training/           # Phase 2/3: Training loops + tuning
│   │   ├── __init__.py
│   │   ├── trainer.py              # Training loop (Người 2)
│   │   ├── callbacks.py            # EarlyStopping + checkpoints (Người 2)
│   │   ├── losses.py               # Loss factory (Người 2)
│   │   ├── optimizers.py           # Optimizer factory (Người 2)
│   │   ├── visualizer.py           # Loss/accuracy curves (Người 2)
│   │   ├── hyperparameter_tuning.py # Grid search (Người 1)
│   │   └── hyperopt_scheduler.py   # Search orchestration (Người 1)
│   └── evaluation/         # Phase 3: Evaluation (Người 3)
│       ├── __init__.py
│       ├── evaluator.py         # Accuracy, precision, recall, F1
│       ├── confusion_matrix.py  # Confusion matrix heatmap
│       ├── error_analysis.py    # Top confused pairs
│       ├── model_io.py          # Save/load checkpoints
│       ├── visualizer.py        # Predicted vs actual images
│       └── run_evaluation.py    # Entry point
├── models/
│   └── checkpoints/        # Saved weights: trained_model.pt, final_model.pt, ...
├── notebooks/              # Jupyter notebooks (empty)
├── tests/
│   └── training/           # 27 unit tests (Người 2)
├── docs/                   # 📁 All project documentation
│   ├── README.md           # Docs index
│   ├── AI_Usage.md         # AI assistance report
│   ├── audit_code.md       # Code audit report
│   ├── CHANGELOG.md        # Version history
│   ├── architecture.md     # System architecture
│   ├── EXPERIMENTS.md      # Experiment log
│   ├── PHASE1_REPORT.md    # Phase 1 summary report
│   ├── PHASE2_PLAN.md      # Phase 2 plan (3 people, aligned với đề)
│   ├── PROJECT_METHODOLOGY.md # Quy trình + giải thích chi tiết
│   ├── TEAM_GUIDE.md       # Hướng dẫn chạy cho cả team
│   ├── PRESENTATION_CHEATSHEET.md # Kịch bản thuyết trình
│   ├── QUICK_START.md      # Quick start guide
│   └── reports/            # Phase 2 reports
│       ├── REPORT_TEMPLATE.md
│       ├── nguoi1_models_report.md
│       ├── nguoi2_training_report.md
│       └── nguoi3_evaluation_report.md
├── configs/
│   ├── model_config.yaml   # Training config (Người 2)
│   └── hyperopt_config.yaml # Grid search space (Người 1)
├── results/
│   └── runs/               # Plots, experiment outputs, dataset_info.md
├── logs/                   # Training logs
├── main.py                    # Entry point → run_training.run_training()
├── run_training.py            # Training orchestration (Người 2)
├── run_experiments.py          # Grid search entry point (Người 1)
├── requirements.txt           # Python dependencies
├── progress_status.md         # (root) Project status
├── README.md                  # (root) This file
├── đề.txt                     # Đề bài
└── .gitignore
```

## 5. Deliverables

| Item | Description | Status |
|------|-------------|--------|
| Data Pipeline | EDA, cleaning, preprocessing, splitting | ✅ Complete |
| Python Code | Complete implementation with modular structure | ✅ Complete |
| Report | Brief analysis of model performance | ✅ 3 reports + methodology |
| Loss Graphs | Visualization of training progress | ✅ `results/runs/loss_curve.png` |
| Image Displays | Predicted vs actual image samples | ✅ `results/runs/predictions.png` |
| Model Save/Load | Checkpoint write/read + final model export | ✅ Complete |

## 6. Tech Stack

- **Framework:** PyTorch 2.14.1+cpu
- **Python:** 3.8+
- **Visualization:** Matplotlib, Seaborn
- **Data Processing:** NumPy, Pandas
- **Metrics:** scikit-learn
- **Logging:** Python logging + loguru
- **Jupyter:** Jupyter Notebook/Lab
- **Optional:** Weights & Biases, TensorBoard

## 7. Workflow

### Phase 1: Data Preparation ✅ COMPLETE
1. **EDA** → Explore dataset statistics and visualize samples
2. **Cleaning** → Handle missing/invalid data
3. **Preprocessing** → Apply transforms (normalize, augment)
4. **Data Split** → Train/validation/test split

### Phase 2: Model Development ✅ COMPLETE
5. **Baseline Model** → Simple neural network baseline
6. **Model Selection** → Compare architectures
7. **Hyperparameter Tuning** → Optimize learning rate, batch size, etc.

### Phase 3: Training & Evaluation ✅ COMPLETE
8. **Training** → Full model training with logging
9. **Evaluation** → Metrics computation and analysis
10. **Error Analysis** → Investigate misclassifications

### Phase 4: Reporting ✅ MOSTLY COMPLETE
11. **Model Saving/Loading** → Done (`trained_model.pt`, `final_model.pt`)
12. **Reports** → Done (3 per-person reports + `PROJECT_METHODOLOGY.md`)

## 8. Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Run Phase 1 (data, one-off)
```bash
# Test the data pipeline end-to-end (load → clean → preprocess → split)
python -m src.data.preprocessing

# Regenerate the 70k split PNGs (idempotent)
python -m src.data.split --root data/processed --val-split 10000 --seed 42

# Regenerate the 4 EDA plots into results/runs/
python -c "from src.data.eda import EDAAnalyzer; EDAAnalyzer('data/processed').generate_full_report('results/runs')"
```

### Run Training (Người 2)
```bash
# Full pipeline with configs/model_config.yaml (15 epochs, CPU ~25 min)
python main.py

# Override individual values
python main.py --epochs 10 --batch-size 128 --lr 0.001
```

### Run Grid Search (Người 1)
```bash
# Quick smoke test of all 4 architectures (~5s)
python run_experiments.py --smoke

# Full grid: 4 models x 2 lr x 2 optimizers = 16 trials x 10 epochs (~4.5 h CPU)
python run_experiments.py --batch-size 128
```

### Run Evaluation (Người 3)
```bash
$env:PYTHONPATH = "."; python src/evaluation/run_evaluation.py
```

### Run Tests
```bash
$env:PYTHONPATH = "."; python -m pytest tests -q   # 27 passed
```

## 9. Data Summary (Phase 1)

| Metric | Value |
|--------|-------|
| Total training samples | 60,000 |
| Total test samples | 10,000 |
| Train/Val/Test split | 50,000 / 10,000 / 10,000 |
| Per-class split | 5,000 train / 1,000 val / 1,000 test |
| Image shape | (1, 28, 28) |
| Mean / Std | 0.2860 / 0.3530 |
| Class balance | Perfect (6,000 each in the raw training set) |
| Data quality | 100% valid (60,000/60,000) |
| Split strategy | Stratified, `seed=42`, `val_ratio=1/6` |

### EDA Visualizations
- `results/runs/class_distribution.png` - Class distribution
- `results/runs/sample_images.png` - 25 sample images
- `results/runs/pixel_histogram.png` - Pixel intensity distribution
- `results/runs/per_class_samples.png` - One sample per class

## 10. Model Results (Phase 2 + 3)

### Best configuration (from grid search, 16 trials × 10 epochs)

| Parameter | Value |
|-----------|-------|
| Architecture | `simple_cnn` (421,642 params) |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Dropout | 0.5 |
| Best grid val accuracy | 92.33% |

### Final training run (15 epochs, CPU)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **92.10%** |
| Precision (macro) | 92.11% |
| Recall (macro) | 92.10% |
| F1-Score (macro) | 92.09% |
| Validation Accuracy | 92.88% (best, epoch 11), 93.03% (last, epoch 15) |
| Best Epoch | 11/15 (by `val_loss`) |
| Training Time | ~75 s/epoch (CPU) |

### Top Confused Classes
| True Class | → Predicted | Count |
|------------|-------------|-------|
| Shirt | T-shirt/top | 95 |
| T-shirt/top | Shirt | 90 |
| Shirt | Pullover | 64 |
| Coat | Pullover | 63 |
| Coat | Shirt | 55 |

> These numbers come from the last full run. See `docs/reports/` for the
> per-person reports and `results/runs/` for the generated artifacts.

### Artifacts
- **Training curves:** `results/runs/loss_curve.png`, `results/runs/accuracy_curve.png`
- **Grid search:** `results/runs/hyperopt.md`, `hyperopt.json`, `hyperopt_best.json`
- **Architecture comparison:** `results/runs/architecture_selection.md`
- **Confusion matrix:** `results/runs/confusion_matrix.png`
- **Predictions plot:** `results/runs/predictions.png`
- **Dataset info:** `results/runs/dataset_info.md`
- **Final model:** `models/checkpoints/final_model.pt`

## 11. Documentation

### 📄 Root files
| File | Description |
|------|-------------|
| [README.md](README.md) | Project overview (this file) |
| [progress_status.md](progress_status.md) | Project progress tracking |
| [đề.txt](đề.txt) | Đề bài / Exercise requirements |
| [requirements.txt](requirements.txt) | Python dependencies |

### 📁 data/
| File | Description |
|------|-------------|
| [data/README.md](data/README.md) | Data folder guide (cách sinh lại data từ FashionMNIST) |

### 📁 docs/
| File | Description |
|------|-------------|
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/AI_Usage.md](docs/AI_Usage.md) | AI assistance report |
| [docs/architecture.md](docs/architecture.md) | System architecture |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Experiment log |
| [docs/PHASE1_REPORT.md](docs/PHASE1_REPORT.md) | Phase 1 summary |
| [docs/PHASE2_PLAN.md](docs/PHASE2_PLAN.md) | Phase 2 plan (3 people, aligned với đề) |
| [docs/PROJECT_METHODOLOGY.md](docs/PROJECT_METHODOLOGY.md) | Quy trình thực hiện + giải thích kỹ thuật |
| [docs/TEAM_GUIDE.md](docs/TEAM_GUIDE.md) | Hướng dẫn chạy cho cả team |
| [docs/PRESENTATION_CHEATSHEET.md](docs/PRESENTATION_CHEATSHEET.md) | Kịch bản thuyết trình |
| [docs/QUICK_START.md](docs/QUICK_START.md) | Quick start guide |
| [docs/audit_code.md](docs/audit_code.md) | Code audit report |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | Version history |

### 📁 docs/reports/ (Phase 2 reports)
| Report | Person | Module (theo đề) |
|--------|--------|------------------|
| [docs/reports/REPORT_TEMPLATE.md](docs/reports/REPORT_TEMPLATE.md) | - | Template cho reports |
| [docs/reports/nguoi1_models_report.md](docs/reports/nguoi1_models_report.md) | Người 1 | Build Network + Hyperparameter Experiment |
| [docs/reports/nguoi2_training_report.md](docs/reports/nguoi2_training_report.md) | Người 2 | Training Loop + Visualize Loss |
| [docs/reports/nguoi3_evaluation_report.md](docs/reports/nguoi3_evaluation_report.md) | Người 3 | Evaluate + Save/Load + Display Predictions |

## 12. References

- [PyTorch Official Documentation](https://pytorch.org/docs/)
- [FashionMNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

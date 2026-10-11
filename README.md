# FashionMNIST Classification with PyTorch

## 1. Project Overview

**Project Name:** FashionMNIST PyTorch Classification  
**Project Type:** Image Classification (Deep Learning)  
**Core Functionality:** Implement a neural network to classify 10 categories of fashion items from the FashionMNIST dataset  
**Target Users:** Students, Researchers, ML Engineers learning PyTorch  
**Status:** Phase 1 Complete (Data Preparation)

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
│   ├── raw/                # Original FashionMNIST dataset
│   ├── processed/          # Transformed/augmented data
│   └── splits/             # Train/val/test split info
│       └── split_info.txt
├── src/
│   ├── data/               # Phase 1: Data preparation modules
│   │   ├── __init__.py
│   │   ├── dataset.py      # FashionMNIST loader
│   │   ├── eda.py          # Exploratory Data Analysis
│   │   ├── preprocessing.py # Cleaning & preprocessing
│   │   └── split.py        # Train/val/test splitting
│   ├── models/             # Phase 2: Neural network architectures
│   │   ├── __init__.py
│   │   ├── baseline_model.py    (placeholder)
│   │   └── model_selection.py   (placeholder)
│   ├── training/           # Phase 2: Training loops
│   │   ├── __init__.py
│   │   ├── trainer.py              (placeholder)
│   │   └── hyperparameter_tuning.py (placeholder)
│   └── evaluation/         # Phase 2: Evaluation metrics
│       ├── __init__.py
│       ├── evaluator.py         (placeholder)
│       ├── error_analysis.py    (placeholder)
│       ├── model_io.py          (placeholder)
│       └── experiment_tracker.py (placeholder)
├── models/
│   └── checkpoints/        # Saved model weights
├── notebooks/              # Jupyter notebooks
├── tests/                  # Unit tests
├── docs/                   # 📁 All project documentation
│   ├── README.md           # This file
│   ├── AI_Usage.md         # AI assistance report
│   ├── progress_status.md  # Project progress tracking
│   ├── audit_code.md       # Code audit report
│   ├── CHANGELOG.md        # Version history
│   ├── ARCHITECTURE.md     # System architecture
│   ├── EXPERIMENTS.md      # Experiment log
│   ├── PHASE1_REPORT.md    # Phase 1 summary report
│   └── QUICK_START.md      # Quick start guide
├── configs/
│   └── model_config.yaml   # Model configuration
├── results/
│   └── runs/               # EDA plots & experiment outputs
│       ├── class_distribution.png
│       ├── sample_images.png
│       ├── pixel_histogram.png
│       └── per_class_samples.png
├── logs/                   # Training logs
│   ├── phase1_run.log
│   ├── eda_visualization.log
│   ├── final_run.log
│   └── final_eda.log
├── run_phase1_pipeline.py     # Run Phase 1 pipeline
├── run_eda_visualization.py   # Run EDA visualization
├── main.py                    # Main training script (Phase 2+)
├── requirements.txt           # Python dependencies
└── .gitignore
```

## 5. Deliverables

| Item | Description | Status |
|------|-------------|--------|
| Python Code | Complete implementation with modular structure | ⏳ In Progress |
| Report | Brief analysis of model performance | ⏳ Phase 2+ |
| Loss Graphs | Visualization of training progress | ⏳ Phase 2+ |
| Image Displays | Predicted vs actual image samples | ⏳ Phase 2+ |
| Data Pipeline | EDA, cleaning, preprocessing, splitting | ✅ Complete |

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

### Phase 2: Model Development ⏳ PENDING
5. **Baseline Model** → Simple neural network baseline
6. **Model Selection** → Compare architectures
7. **Hyperparameter Tuning** → Optimize learning rate, batch size, etc.

### Phase 3: Training & Evaluation ⏳ PENDING
8. **Training** → Full model training with logging
9. **Evaluation** → Metrics computation and analysis
10. **Error Analysis** → Investigate misclassifications

### Phase 4: Deployment & Reporting ⏳ PENDING
11. **Model Saving/Loading** → Export and deploy models
12. **Experiment Management** → Track experiments and generate reports

## 8. Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Run Phase 1 Pipeline (Current)
```bash
# Run end-to-end Phase 1 pipeline
python run_phase1_pipeline.py

# Generate EDA visualizations
python run_eda_visualization.py
```

### Run Training (Phase 2 - Coming Soon)
```bash
python main.py --epochs 10 --batch_size 64 --lr 0.001
```

### Run Evaluation (Phase 2 - Coming Soon)
```bash
python src/evaluation/evaluate_model.py --model models/checkpoints/best_model.pt
```

## 9. Current Results (Phase 1)

| Metric | Value |
|--------|-------|
| Total training samples | 60,000 |
| Total test samples | 10,000 |
| Train/Val/Test split | 54,000 / 6,000 / 10,000 |
| Image shape | (1, 28, 28) |
| Mean / Std | 0.2860 / 0.3530 |
| Class balance | Perfect (6,000 each) |
| Data quality | 100% valid (60,000/60,000) |
| Batch size | 64 |

### EDA Visualizations
- `results/runs/class_distribution.png` - Class distribution
- `results/runs/sample_images.png` - 25 sample images
- `results/runs/pixel_histogram.png` - Pixel intensity distribution
- `results/runs/per_class_samples.png` - One sample per class

## 10. Expected Results (Phase 2+)

- **Baseline Accuracy:** ~85%
- **Optimized Accuracy:** ~90%+
- **Training Time:** 5-15 minutes per epoch (CPU)

## 11. Documentation

| Document | Description |
|----------|-------------|
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/AI_Usage.md](docs/AI_Usage.md) | AI assistance report |
| [docs/progress_status.md](docs/progress_status.md) | Project progress tracking |
| [docs/audit_code.md](docs/audit_code.md) | Code audit report |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | Version history |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Experiment log |
| [docs/PHASE1_REPORT.md](docs/PHASE1_REPORT.md) | Phase 1 summary |
| [docs/QUICK_START.md](docs/QUICK_START.md) | Quick start guide |
| [docs/PHASE2_PLAN.md](docs/PHASE2_PLAN.md) | Phase 2 work plan (3 people, aligned với đề) |

### Phase 2 Reports (docs/reports/)

| Report | Person | Module (theo đề) |
|--------|--------|------------------|
| [docs/reports/nguoi1_models_report.md](docs/reports/nguoi1_models_report.md) | Người 1 | Build Network + Hyperparameter Experiment |
| [docs/reports/nguoi2_training_report.md](docs/reports/nguoi2_training_report.md) | Người 2 | Training Loop + Visualize Loss |
| [docs/reports/nguoi3_evaluation_report.md](docs/reports/nguoi3_evaluation_report.md) | Người 3 | Evaluate + Save/Load + Display Predictions |

## 12. References

- [PyTorch Official Documentation](https://pytorch.org/docs/)
- [FashionMNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

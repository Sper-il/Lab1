# Quick Start Guide - FashionMNIST Project

## Overview

This project implements a FashionMNIST classifier using PyTorch. The project follows a 12-step ML workflow.

## Current Status: Phase 1 Complete

Only the data preparation steps have been implemented:
- ✅ EDA
- ✅ Data Cleaning
- ✅ Preprocessing
- ✅ Data Split

Other modules are placeholders (empty files) for future phases.

## Project Structure

```
e:\lab1/
├── README.md                    # Project documentation
├── AI_Usage.md                  # AI assistance report
├── progress_status.md           # Project progress tracking
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── main.py                      # Main training script (Phase 2+ placeholder)
│
├── data/                        # Dataset storage
│   ├── raw/                     # Original FashionMNIST (downloaded, gitignored)
│   │   ├── FashionMNIST/        # Original archive
│   │   └── _torchvision_cache/  # Torchvision download cache
│   ├── processed/               # Split PNGs (gitignored, 70k files)
│   │   ├── train/<class_id>/*.png  # 50.000 train
│   │   ├── val/<class_id>/*.png    # 10.000 val
│   │   └── test/<class_id>/*.png   # 10.000 test
│   └── splits/                  # Split info (committed)
│       └── split_info.txt
│
├── src/                         # Source code
│   ├── data/                    # Phase 1: Data preparation
│   │   ├── dataset.py           # Dataset loader
│   │   ├── eda.py               # EDA module
│   │   ├── preprocessing.py     # Cleaning + preprocessing
│   │   └── split.py             # Train/val/test split
│   ├── models/                  # Phase 2: Models (placeholder)
│   ├── training/                # Phase 2: Training (placeholder)
│   └── evaluation/              # Phase 2: Evaluation (placeholder)
│
├── models/                      # Trained model checkpoints
│   └── checkpoints/
│
├── configs/                     # Configuration files
│   └── model_config.yaml
│
├── docs/                        # Additional documentation
│   ├── README.md
│   ├── CHANGELOG.md
│   ├── ARCHITECTURE.md
│   ├── EXPERIMENTS.md
│   ├── audit_code.md
│   ├── PHASE1_REPORT.md         # Phase 1 summary report
│   ├── PHASE2_PLAN.md           # Phase 2 plan (3 people)
│   ├── QUICK_START.md           # This file
│   └── reports/                 # Phase 2 reports
│
├── results/                     # Experiment results
│   └── runs/                    # EDA plots + experiment outputs
│       ├── class_distribution.png
│       ├── sample_images.png
│       ├── pixel_histogram.png
│       └── per_class_samples.png
│
├── notebooks/                   # Jupyter notebooks (for later)
├── tests/                       # Unit tests (for later)
└── logs/                        # Log files
    ├── phase1_run.log
    └── eda_visualization.log
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Split Data (1 lần, ~2-3 phút)

```bash
python -m src.data.split --root data/processed --val-split 10000 --seed 42
```

Script này sẽ:
- Tải FashionMNIST gốc về `data/raw/_torchvision_cache/` (lần đầu)
- Chia 60.000 ảnh train gốc thành 50.000 train + 10.000 val (random, seed=42)
- Ghi 70.000 ảnh `.png` (28×28 grayscale) vào `data/processed/{train,val,test}/<class_id>/`
- Idempotent — skip nếu đã tồn tại

### 3. Run EDA Visualization (1 lần, ~20s)

```bash
python -c "from src.data.eda import EDAAnalyzer; EDAAnalyzer('data/processed').generate_full_report('results/runs')"
```

Generates 4 plots in `results/runs/`:
- Class distribution
- Sample images (25 random)
- Pixel intensity histogram
- Per-class samples (1 mỗi class)

## Phase 1 Results Summary

| Metric | Value |
|--------|-------|
| Total samples | 70,000 (60k train + 10k test) |
| Train/Val/Test split | 54,000 / 6,000 / 10,000 |
| Image shape | (1, 28, 28) |
| Mean | 0.2860 |
| Std | 0.3530 |
| Class balance | Perfect (6,000 each) |
| Data quality | 100% valid |

## Next Steps (Phase 2 - 3 người)

Phase 1 is complete. The next phase will implement (theo đề bài):

### 👤 Người 1: Build Network + Hyperparameter Experiment
- Baseline MLP model
- Multiple architectures (MLP, SimpleCNN, ResNet)
- Hyperparameter tuning
- Files: `src/models/baseline_model.py`, `architectures/*.py`, `training/hyperparameter_tuning.py`

### 👤 Người 2: Training Loop + Visualize Loss
- Training loop (forward, loss, backward, optimize)
- Callbacks, losses, optimizers
- Visualize loss curve
- Files: `src/training/trainer.py`, `visualizer.py`

### 👤 Người 3: Evaluate + Save/Load + Display
- Model evaluation (accuracy ≥ 90%)
- Save/Load model
- Display predicted vs actual images
- Files: `src/evaluation/evaluator.py`, `model_io.py`, `visualizer.py`

See chi tiết tại `docs/PHASE2_PLAN.md`.

## Notes

- CPU only (CUDA not available on this machine)
- Random seed = 42 for reproducibility
- All logs are saved to `logs/` directory
- All outputs are saved to `results/runs/`

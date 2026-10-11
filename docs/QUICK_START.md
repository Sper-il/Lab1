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
├── run_phase1_pipeline.py       # Run Phase 1 pipeline
├── run_eda_visualization.py     # Generate EDA plots
├── main.py                      # Main training script (Phase 2+)
│
├── data/                        # Dataset storage
│   ├── raw/                     # Original FashionMNIST (downloaded)
│   ├── processed/               # Processed data
│   └── splits/                  # Split info
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
│   └── PHASE1_REPORT.md         # Phase 1 summary report
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

### 2. Run Phase 1 Pipeline

```bash
python run_phase1_pipeline.py
```

This will:
- Load FashionMNIST dataset
- Validate data quality
- Compute preprocessing statistics
- Split into train/val/test

### 3. Run EDA Visualization

```bash
python run_eda_visualization.py
```

This generates plots in `results/runs/`:
- Class distribution
- Sample images
- Pixel histogram
- Per-class samples

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

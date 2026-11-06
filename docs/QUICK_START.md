# Quick Start Guide - FashionMNIST Project

## Overview

This project implements a FashionMNIST classifier using PyTorch. The project follows a 12-step ML workflow.

## Current Status: Phase 2 COMPLETE — Test Accuracy 92,10% (vượt mục tiêu 90%)

Đã hoàn thành toàn bộ pipeline:
- ✅ EDA, Data Cleaning, Preprocessing, Data Split (stratified 50k/10k/10k, seed=42)
- ✅ 4 kiến trúc + ModelFactory + grid search 16 trials (best `simple_cnn` + Adam + lr=0,001 → val 92,33%)
- ✅ Full training 15 epochs CPU (`python main.py` → best epoch 11, test 92,10%)
- ✅ Evaluation + Save/Load + predicted vs actual (`python src/evaluation/run_evaluation.py`)
- ✅ Báo cáo chính: `docs/training.md`, `docs/evaluation.md`, `docs/demo.md`, `docs/PHASE2_FINAL_REPORT.md`

## Project Structure

```
e:\lab1/
├── README.md                    # Project documentation
├── AI_Usage.md                  # AI assistance report
├── progress_status.md           # Project progress tracking
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── main.py                      # Entry point training (YAML + CLI override → run_training.py)
│
├── data/                        # Dataset storage
│   ├── processed/FashionMNIST/  # FashionMNIST gốc của torchvision (tải 1 lần, gitignored)
│   ├── processed/train|val|test/<class_id>/*.png  # artifact Phase 1 cũ (70k PNG, không còn được đọc)
│   └── splits/                  # Split info (committed)
│       └── split_info.txt       # Train 50000 · Val 10000 · Test 10000, seed 42
│
├── src/                         # Source code
│   ├── data/                    # Phase 1: Data preparation
│   │   ├── dataset.py           # Dataset loader
│   │   ├── eda.py               # EDA module
│   │   ├── preprocessing.py     # Cleaning + preprocessing
│   │   └── split.py             # Train/val/test split (stratified 5/6–1/6, seed 42)
│   ├── models/                  # Người 1: 4 kiến trúc + ModelFactory + model_selection
│   ├── training/                # Người 1 (tuning) + Người 2 (Trainer, callbacks, visualizer)
│   └── evaluation/              # Người 3: evaluator, model_io, confusion, error, visualizer
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
│   ├── architecture.md
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
- Tải FashionMNIST gốc về `data/processed/FashionMNIST/raw/` (lần đầu, ~30 MB)
- Chia 60.000 ảnh train gốc thành 50.000 train + 10.000 val (stratified, seed=42)
- Test giữ nguyên 10.000 ảnh gốc, không chia lại
- Ghi metadata vào `data/splits/split_info.txt` + in số batch từng tập

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
| Train/Val/Test split | 50,000 / 10,000 / 10,000 |
| Image shape | (1, 28, 28) |
| Mean | 0.2860 |
| Std | 0.3530 |
| Class balance | Perfect (6,000 each) |
| Data quality | 100% valid |

## Kết quả Phase 2 (đã hoàn thành — test 92,10%)

Phase 1 xong, Phase 2 cũng đã xong toàn bộ (theo đề bài):

### 👤 Người 1: Build Network + Hyperparameter Experiment ✅ DONE
- 4 kiến trúc (`baseline_mlp`, `deep_mlp`, `simple_cnn`, `resnet_mini`) + ModelFactory
- Grid search 16 trials × 10 epochs → best `simple_cnn` + Adam + lr=0,001 → val 92,33%
- Files: `src/models/baseline_model.py`, `architectures/*.py`, `training/hyperparameter_tuning.py`

### 👤 Người 2: Training Loop + Visualize Loss ✅ DONE
- Training loop (forward → loss → backward → optimize) + early stopping + checkpoint
- Full 15 epochs CPU → best epoch 11 (val_loss 0,215183), test 92,10%
- Files: `src/training/trainer.py`, `callbacks.py`, `losses.py`, `optimizers.py`, `visualizer.py`

### 👤 Người 3: Evaluate + Save/Load + Display ✅ DONE
- Test accuracy 92,10%, precision/recall/F1 ~92,1%, confusion matrix, predicted vs actual
- Save `final_model.pt` kèm metrics
- Files: `src/evaluation/evaluator.py`, `model_io.py`, `visualizer.py`, `confusion_matrix.py`

Chi tiết xem `docs/PHASE2_PLAN.md`, `docs/PHASE2_FINAL_REPORT.md`, `docs/training.md`, `docs/evaluation.md`.

## Notes

- CPU only (CUDA not available on this machine)
- Random seed = 42 for reproducibility
- All logs are saved to `logs/` directory
- All outputs are saved to `results/runs/`

# Phase 1 Report - Data Preparation

**Date:** 2026-10-11  
**Status:** ✅ COMPLETE  
**Duration:** ~2 minutes

---

## Executive Summary

Phase 1 (Data Preparation) successfully completed all 4 required steps:
1. ✅ **EDA** - Exploratory Data Analysis
2. ✅ **Cleaning** - Data quality validation
3. ✅ **Preprocessing** - Normalization setup
4. ✅ **Data Split** - Train/Val/Test split

The pipeline runs end-to-end without errors and is ready for Phase 2 (Model Development).

---

## Detailed Results

### Step 1: EDA (Exploratory Data Analysis)

**Objective:** Understand the dataset structure and characteristics.

**Findings:**
- **Dataset:** FashionMNIST (60,000 training + 10,000 test samples)
- **Image shape:** (1, 28, 28) - grayscale
- **Classes:** 10 (perfectly balanced, 6,000 samples each)
- **Pixel value range:** [0, 1] after ToTensor
- **Mean:** 0.2860
- **Std:** 0.3530

**Generated Visualizations:**
- `results/runs/class_distribution.png` - Bar chart of class distribution
- `results/runs/sample_images.png` - Grid of 25 random samples
- `results/runs/pixel_histogram.png` - Pixel intensity distribution
- `results/runs/per_class_samples.png` - One sample per class

**Conclusion:** Dataset is clean, balanced, and ready for preprocessing.

---

### Step 2: Data Cleaning

**Objective:** Validate data quality and remove invalid samples.

**Validation Criteria:**
- No NaN/Inf in image tensors
- No empty tensors
- All labels in valid range [0, 9]
- All image values finite

**Results:**
- **Valid samples:** 60,000 / 60,000 (100%)
- **Invalid samples:** 0
- **Action required:** None (FashionMNIST is pre-cleaned)

**Conclusion:** No cleaning required. Dataset is production-ready.

---

### Step 3: Preprocessing

**Objective:** Apply normalization to prepare data for model training.

**Configuration:**
```yaml
normalize: True
mean: 0.2860
std: 0.3530
augment: False  # Will be enabled in Phase 2 if needed
```

**Computed Statistics (from 10,000 samples):**
- Mean: 0.2863
- Std: 0.3540
- Min: 0.0000
- Max: 1.0000

**Effect of Normalization:**
- Image range after normalize: [-0.8102, 2.0227]
- This is the expected range for FashionMNIST after standard normalization

**Conclusion:** Preprocessing pipeline working correctly.

---

### Step 4: Data Split

**Objective:** Split data into train/validation/test sets.

**Configuration:**
```yaml
train_ratio: 0.9
val_ratio: 0.1
test_ratio: 0.0  # FashionMNIST provides separate test set
random_seed: 42
stratified: True  # Preserve class distribution
```

**Results:**

| Split | Samples | Batches (bs=64) | Percentage |
|-------|---------|-----------------|------------|
| Train | 54,000 | 844 | 90% (of training data) |
| Val   | 6,000  | 94  | 10% (of training data) |
| Test  | 10,000 | 157 | 100% (separate test set) |

**Sample Batch Verification:**
- Shape: (64, 1, 28, 28) - correct
- Labels shape: (64,) - correct
- Image range: [-0.8102, 2.0227] - normalized correctly

**Conclusion:** Data split pipeline working correctly. Reproducible with seed=42.

---

## Pipeline Execution

### How to Run

> **Note:** Phase 1 đã chạy thành công và sinh ra 70.000 ảnh PNG trong
> `data/processed/`. Các lần chạy lại dùng module trực tiếp (không còn
> `run_phase1_pipeline.py` — đã được cleanup trong commit `70691e2`).

```bash
# Từ project root, kích hoạt venv rồi:

# Test toàn bộ data pipeline (load → cleaning → preprocessing → split)
python -m src.data.preprocessing

# Sinh lại 70.000 ảnh PNG (idempotent — skip nếu đã tồn tại)
python -m src.data.split --root data/processed --val-split 10000 --seed 42

# Sinh lại EDA plots (4 file PNG trong results/runs/)
python -c "from src.data.eda import EDAAnalyzer; EDAAnalyzer('data/processed').generate_full_report('results/runs')"
```

### Output Locations

| Output | Path |
|--------|------|
| 70k split PNGs | `data/processed/{train,val,test}/<class_id>/*.png` |
| EDA plots | `results/runs/*.png` |
| Split info | `data/splits/split_info.txt` |
| Run logs | `logs/phase1_run.log` |
| EDA log | `logs/eda_visualization.log` |
| Raw archive | `data/raw/FashionMNIST/` (chỉ chứa FashionMNIST gốc, KHÔNG chứa ảnh đã xử lý) |

---

## Files Created

### Source Code (4 files)
- `src/data/dataset.py` - Dataset loader + DataLoader factory
- `src/data/eda.py` - EDA analysis + visualizations
- `src/data/preprocessing.py` - Data cleaning + preprocessing
- `src/data/split.py` - Train/val/test split

### Documentation
- `README.md` - Project overview (root)
- `progress_status.md` - Progress tracking (root)
- `data/README.md` - Data folder guide
- `docs/README.md` - Documentation index
- `docs/AI_Usage.md` - AI assistance report
- `docs/architecture.md` - System architecture
- `docs/EXPERIMENTS.md` - Experiment log
- `docs/PHASE1_REPORT.md` - Phase 1 summary (this file)
- `docs/PHASE2_PLAN.md` - Phase 2 plan (3 people)
- `docs/QUICK_START.md` - Quick start guide
- `docs/audit_code.md` - Code audit report
- `docs/CHANGELOG.md` - Version history

### Empty Placeholders (7 files for Phase 2+)
- `src/models/baseline_model.py`
- `src/models/model_selection.py`
- `src/training/trainer.py`
- `src/training/hyperparameter_tuning.py`
- `src/evaluation/evaluator.py`
- `src/evaluation/error_analysis.py`
- `src/evaluation/model_io.py`

### Configuration
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore patterns
- `configs/model_config.yaml` - Model config (placeholder)

---

> **Lịch sử cleanup (commit `70691e2` - 2026-10-11):** Các runner scripts
> tạm thời (`run_phase1_pipeline.py`, `run_eda_visualization.py`) đã bị xóa
> vì pipeline chạy ổn định, không cần wrapper. Để chạy lại Phase 1, dùng
> module trực tiếp (xem **How to Run** ở trên).

---

## Next Phase: Model Development (Phase 2)

Ready to implement:
1. Baseline MLP model
2. Model selection framework
3. Training loop
4. Hyperparameter tuning

**Prerequisites met:**
- ✅ Data loaded and accessible
- ✅ Preprocessing pipeline ready
- ✅ Train/Val/Test splits available
- ✅ Normalization statistics computed
- ✅ Logging infrastructure in place

---

## Approval

- [x] All 4 steps completed
- [x] Pipeline runs end-to-end without errors
- [x] Outputs generated and saved
- [x] Documentation updated
- [x] Ready for Phase 2

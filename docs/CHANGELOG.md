# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2026-10-11

### Added
- Initial project structure
- Documentation: README.md, AI_Usage.md, progress_status.md
- Data preparation pipeline (EDA, cleaning, preprocessing, split)
- Module placeholders for upcoming phases

### Phase 1: Data Preparation
- `src/data/dataset.py` - FashionMNIST dataset loader
- `src/data/eda.py` - Exploratory data analysis
- `src/data/preprocessing.py` - Data cleaning and preprocessing
- `src/data/split.py` - Train/val/test splitting

## [Phase 2] - 2026-10-12 → 2026-10-15

### Người 1 — Build Network + Hyperparameter Experiment (2026-10-12 → 2026-10-22)
- `src/models/baseline_model.py` (`BaselineMLP`, 203.530 params)
- `src/models/architectures/mlp.py` (`DeepMLP`, 118.650 params)
- `src/models/architectures/simple_cnn.py` (`SimpleCNN`, 421.642 params — best)
- `src/models/architectures/advanced_cnn.py` (`ResNetMini` + `BasicBlock`, 174.970 params)
- `src/models/model_factory.py`, `src/models/model_selection.py` (`compare_architectures`)
- `src/training/hyperparameter_tuning.py`, `src/training/hyperopt_scheduler.py`
- `configs/hyperopt_config.yaml`, `run_experiments.py` (`--smoke`, `--strategy grid`, `--epochs`)
- Grid search 16 trials × 10 epochs → best val 92,33% (`simple_cnn` + Adam + lr=0,001); artifacts `results/runs/hyperopt.{json,md,_best.json}`, `architecture_selection.md`
- Bug fix `src/data/split.py`: import path + `np.random.shuffle(stratified_indices)` (fix val lệch class)
- Report: `docs/reports/nguoi1_models_report.md`

### Người 2 — Training Loop + Visualize Loss (2026-10-14)
- `src/training/trainer.py` (forward → loss → backward → optimize, weighted metrics)
- `src/training/callbacks.py` (EarlyStopping `val_loss/min/patience=5/min_delta=0,0001`, Checkpoint, Logger)
- `src/training/losses.py`, `src/training/optimizers.py`, `src/training/visualizer.py`
- `run_training.py`, `main.py` (entry point + YAML + CLI override + seed/device)
- Full run 15 epochs CPU: best epoch 11 (val_loss 0,215183, val_acc 92,85%); artifacts `models/checkpoints/trained_model.pt`, `last_model.pt` (local-only), `results/runs/training_history.json`, `loss_curve.png`, `accuracy_curve.png`
- Tests: 27/27 passed (`tests/training/`)
- Report: `docs/reports/nguoi2_training_report.md`

### Người 3 — Evaluate + Save/Load + Display (2026-10-15)
- `src/evaluation/evaluator.py` (test 92,10%, precision/recall/F1 ~92,1% macro)
- `src/evaluation/model_io.py` (save/load/pretrained, `final_model.pt`)
- `src/evaluation/confusion_matrix.py`, `error_analysis.py`, `visualizer.py` (PredictionVisualizer)
- `src/evaluation/run_evaluation.py`; artifacts `confusion_matrix.png`, `predictions.png`, `results/runs/test_metrics.md`
- Report: `docs/reports/nguoi3_evaluation_report.md`

### Docs tổng hợp (2026-10-15 → 2026-11-06)
- `docs/description.md` (Define 6 mục), `docs/data.md` (quy ước dữ liệu), `docs/data-flow.md` (luồng dữ liệu), `docs/api.md` (API nội bộ)
- `docs/PHASE2_FINAL_REPORT.md` (báo cáo tổng hợp: best model, metrics, lessons, recommendations)
- `docs/EXPERIMENTS.md` (log pilot + grid + full train + evaluation)
- `docs/AI_Usage.md` (viết lại theo chuẩn 7 mục), `docs/architecture.md` (cập nhật Completed), `docs/README.md` (index mới)

## [Future Releases]

### Phase 3 (đề xuất)
- Data augmentation nhẹ cho nhóm Shirt/T-shirt/top
- `weight_decay` / label smoothing cho ResNetMini
- Export ONNX + đo latency/inference-time (eval pipeline)
- `results/runs/report.html` offline 1 file theo `modules/visualization.md`
- Demo 1 lệnh `python -m src.demo` + `results/demo/<ts>/index.html` theo `modules/demo.md`

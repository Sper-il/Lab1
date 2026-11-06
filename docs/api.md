# API nội bộ giữa các module

> Quy ước shape/dtype chung xem `docs/data.md`. Luồng tổng thể xem `docs/data-flow.md`.

## 1. `src.data` — Data pipeline

### `split_pipeline(batch_size=128, ...) -> (train_loader, val_loader, test_loader, split_info)`
- Nguồn: `src/data/split.py` (dùng chung bởi `run_training.py`, `run_experiments.py`).
- Output: `DataLoader` cho ảnh `(B, 1, 28, 28)` float32 + nhãn `(B,)` int64; `split_info` dict chứa sizes, seed, batch counts.
- Tham số chính: `--root data/processed`, `--val-split 10000`, `--seed 42`, `--batch-size 64|128`.

### `TransformBuilder` / `build_dataloader` (`src/data/dataset.py`)
- Input: config normalize `mean=0,2860, std=0,3530`.
- Output: `DataLoader` + `CLASS_NAMES` (10 tên lớp theo đúng thứ tự index).

### `EDAAnalyzer('data/processed').generate_full_report('results/runs')` (`src/data/eda.py`)
- Output: 4 PNG (`class_distribution`, `sample_images`, `pixel_histogram`, `per_class_samples`) + stats (mean/std/min/max).

### `DataCleaner` / `DataPreprocessor` (`src/data/preprocessing.py`)
- `DataCleaner`: validate 60.000/60.000 (100%), 0 NaN/Inf.
- `DataPreprocessor`: normalize + thống kê; image range sau normalize `[-0,8102, 2,0227]`.

## 2. `src.models` — Kiến trúc (Người 1)

### `ModelFactory.create(name, **kwargs) -> nn.Module`
- Nguồn: `src/models/model_factory.py`. Registry: `baseline_mlp`, `deep_mlp`, `simple_cnn`, `resnet_mini`.
- Input chấp nhận `(B, 1, 28, 28)` (riêng `DeepMLP` tự flatten); output logits `(B, 10)`.
- Best: `ModelFactory.create("simple_cnn")` — 421.642 params, dropout mặc định 0,5.

### `compare_architectures(...)` (`src/models/model_selection.py`)
- Quick ranking 2 epochs: `simple_cnn` 89,48% > `resnet_mini` 88,63% > `baseline_mlp` 87,05% > `deep_mlp` 86,10%.

### `HyperparameterTuner.grid_search(param_grid)` + `HyperoptScheduler` (`src/training/hyperparameter_tuning.py`, `hyperopt_scheduler.py`)
- Input: search space `configs/hyperopt_config.yaml` (4 model × 2 lr × 2 optimizer = 16 trials × 10 epochs).
- Output: `results/runs/hyperopt.json` (16 rows), `hyperopt.md` (bảng), `hyperopt_best.json` (trial 9: `simple_cnn` + Adam + lr=0,001 → val 92,33%).
- Entry: `python run_experiments.py --smoke` (forward test) / `--batch-size 128` (full grid).

## 3. `src.training` — Training (Người 2)

### `Trainer(model, train_loader, val_loader, criterion, optimizer, device, callbacks)`
- `train_epoch() -> (train_loss, train_acc)` — forward → loss → `zero_grad`/backward → `step`, weighted theo số mẫu.
- `validate() -> (val_loss, val_acc)` — `eval()` + `no_grad()`, weighted.
- `train(num_epochs) -> history {train_loss, val_loss, train_acc, val_acc}`.
- Cấu hình hiệu lực (`configs/model_config.yaml`): `simple_cnn`, Adam, lr=0,001, bs=128, dropout 0,5, epochs=15, device auto→CPU, seed 42, early stopping `val_loss/min/patience=5/min_delta=0,0001`.

### `EarlyStopping` / `Checkpoint` / `Logger` (`src/training/callbacks.py`)
- Best theo `val_loss` → `models/checkpoints/trained_model.pt` (epoch 11, val_loss 0,215183, val_acc 92,85%); last → `last_model.pt` (epoch 15, local-only).

### `TrainingVisualizer` (`src/training/visualizer.py`)
- `plot_loss_curve` / `plot_accuracy_curve` / `plot_combined` → `results/runs/loss_curve.png`, `accuracy_curve.png`.

### Orchestration
- `python main.py [--epochs 10 --batch-size 128 --lr 0.001]` → đọc YAML + CLI override → `run_training(config)` trong `run_training.py` → lắp data/model/loss/optimizer/callbacks/Trainer/visualizer → persist checkpoint + `training_history.json` + plots.
- Tests: `python -m pytest tests -q` → **27 passed**.

## 4. `src.evaluation` — Evaluation (Người 3)

### `ModelEvaluator(model, test_loader, device)` (`src/evaluation/evaluator.py`)
- `predict() -> (preds, labels, probs)` trên 10.000 test.
- `evaluate() -> (metrics, preds, labels, probs)` — accuracy **92,10%**, precision 92,11%, recall 92,10%, F1 92,09% (macro).

### `ModelIO` (`src/evaluation/model_io.py`)
- `save_model(model, path, model_name, optimizer, history, metrics)` → checkpoint dict `{model_state_dict, optimizer_state_dict, epoch, metrics, history, model_name}`.
- `load_model(model, path, device, optimizer, strict)` → `(model, checkpoint)`; verify save→load→predict cho cùng kết quả.
- `save_pretrained(model, path)` → weights-only cho deployment.
- Final: `models/checkpoints/final_model.pt` (kèm metrics).

### `ConfusionMatrixPlotter` / `ErrorAnalyzer` / `PredictionVisualizer`
- `confusion_matrix.py` → normalized heatmap `results/runs/confusion_matrix.png`; top-5 confused: Shirt→T-shirt/top 95, T-shirt/top→Shirt 90, Shirt→Pullover 64, Coat→Pullover 63, Coat→Shirt 55.
- `error_analysis.py` → `analyze_common_confusions()` top-K cặp nhầm.
- `visualizer.py` → grid predicted vs actual, xanh=đúng đỏ=sai, `results/runs/predictions.png`.
- Entry: `$env:PYTHONPATH="."; python src/evaluation/run_evaluation.py`.

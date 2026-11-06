# Báo cáo tổng hợp Phase 2 — FashionMNIST PyTorch Classification

> **Ngày:** 2026-10-15 · **Trạng thái:** HOÀN THÀNH 100% · **Kết quả:** Test Accuracy **92,10%** (vượt mục tiêu 90%)
> **Nguồn số liệu:** `results/runs/hyperopt.{json,md,_best.json}`, `architecture_selection.md`, `training_history.json`, `test_metrics.md`, `dataset_info.md` và 3 báo cáo `docs/reports/nguoi{1,2,3}_*.md`.

## 1. Executive summary

- **Mục tiêu đề:** build network, training loop (forward/loss/backward/optimize), evaluate accuracy, save/load model, thử network/hyperparameters, visualize loss, display predicted vs actual, bàn giao code + brief report + loss graphs + image displays.
- **Kết quả:** hoàn thành toàn bộ. Best config từ grid search 16 trials × 10 epochs là `simple_cnn` + Adam + lr=0,001 (val 92,33%); full training 15 epochs trên CPU cho **test 92,10%**, precision/recall/F1 macro ~92,1%.
- **Bàn giao:** code modular `src/{data,models,training,evaluation}`, checkpoint `trained_model.pt` + `final_model.pt`, curves `loss_curve.png`/`accuracy_curve.png`, `confusion_matrix.png`, `predictions.png`, 3 báo cáo cá nhân + báo cáo này.

## 2. Best model architecture

- **Kiến trúc:** `SimpleCNN` (`src/models/architectures/simple_cnn.py`) — 2 conv blocks (Conv→ReLU→Pool ×2, channels 32→64) + 2 FC (`64·7·7 → 128 → 10`) + Dropout 0,5. Tổng **421.642 params**.
- **Vì sao chọn:** đứng đầu cả pilot `compare_architectures` (val 89,48% sau 2 epochs) lẫn grid search full (val 92,33%); train/val gap chỉ 0,46% (ít overfit hơn ResNetMini gap 3,4%); nhanh gấp 2,9× ResNetMini (~75 vs ~218 s/epoch CPU). Chi tiết xem `results/runs/architecture_selection.md`, `results/runs/hyperopt.md`.
- **Hyperparameters hiệu lực:** Adam, lr=0,001, batch 128, weight_decay 0,0, dropout 0,5, epochs 15, seed 42, device auto→CPU, early stopping `val_loss/min/patience=5/min_delta=0,0001`, chọn best theo `val_loss`.

## 3. Final performance metrics

### 3.1. Full training 15 epochs (`python main.py`)

| Epoch | Train loss | Val loss | Train acc | Val acc |
|---:|---:|---:|---:|---:|
| 1 | 0,585270 | 0,351479 | 79,03% | 87,08% |
| 5 | 0,265313 | 0,237384 | 90,41% | 91,42% |
| 10 | 0,190164 | 0,221764 | 92,91% | 92,63% |
| 11 | 0,178463 | **0,215183** | 93,47% | 92,85% |
| 13 | 0,159146 | 0,216467 | 94,03% | **92,88%** (đỉnh val_acc) |
| 15 | 0,141153 | 0,229658 | **94,72%** | 92,80% |

Best theo `val_loss` là epoch 11 (val_loss 0,215183, val_acc 92,85%); early stopping không kích hoạt (sau best chỉ 4 epoch không cải thiện, chưa đủ patience 5). Train/val gap cuối ~2% → overfitting nhẹ, đã kiểm soát bằng best-weight restoration.

### 3.2. Test set (`python src/evaluation/run_evaluation.py`)

| Chỉ số | Giá trị |
|--------|---------|
| Accuracy | **92,10%** |
| Precision (macro) | 92,11% |
| Recall (macro) | 92,10% |
| F1-Score (macro) | 92,09% |

Top-5 cặp nhầm: Shirt→T-shirt/top 95, T-shirt/top→Shirt 90, Shirt→Pullover 64, Coat→Pullover 63, Coat→Shirt 55. Nhóm Shirt/T-shirt/top/Pullover/Coat có hình dáng tương tự là nguồn lỗi chính — xem `results/runs/confusion_matrix.png`, `predictions.png`.

## 4. Mapping yêu cầu đề → bằng chứng

| Yêu cầu đề | Người | Bằng chứng |
|------------|-------|------------|
| Load FashionMNIST + transforms | Phase 1 | `src/data/dataset.py`, `split.py`, `data/splits/split_info.txt` |
| Build neural network | Người 1 | `src/models/{baseline_model,architectures/*,model_factory,model_selection}.py` |
| Training loop (forward, loss, backward, optimize) | Người 2 | `src/training/trainer.py`, 27/27 tests pass |
| Evaluate accuracy | Người 3 | `src/evaluation/evaluator.py`, test 92,10% |
| Save/load model | Người 3 | `src/evaluation/model_io.py`, `trained_model.pt` → `final_model.pt` |
| Experiment network/hyperparameters | Người 1 | Grid 16 trials, `hyperopt.{json,md,_best.json}` |
| Visualize loss | Người 2 | `loss_curve.png`, `accuracy_curve.png`, `training_history.json` |
| Display predicted vs actual | Người 3 | `predictions.png` (xanh=đúng, đỏ=sai) |
| Python code + brief report + loss graphs + image displays | Cả team | Repo này + 3 báo cáo cá nhân + báo cáo tổng hợp này |

## 5. Lessons learned

1. **Adam vượt SGD 2–6%** ở cùng lr trên dataset nhỏ + ít epoch — đúng kỳ vọng lý thuyết, được xác nhận trên cả 4 kiến trúc.
2. **CNN thắng MLP** nhờ tận dụng cấu trúc không gian (~92,3% vs ~89–90%), nhưng ResNet sâu hơn không đáng chi phí CPU (chậm 2,9×, hơn 0,13%).
3. **Chọn best theo `val_loss`** thay vì `val_acc` cho weights ổn định hơn khi train/val bắt đầu phân kỳ.
4. **Stratified split + seed cố định** là bắt buộc — bug thiếu shuffle từng làm val lệch 947–1.040/lớp.
5. **BatchNorm cần batch đủ lớn** — smoke test batch=4 flaky, đổi sang 64 samples × 5 steps hết flaky.

## 6. Recommendations

- Thêm augmentation nhẹ (shift/rotation) để giảm nhầm nhóm Shirt/T-shirt/top.
- Thử `weight_decay` nhỏ và label smoothing cho ResNetMini (train_acc 95,6% cho thấy overfit).
- Export ONNX + đo latency/inference-time để hoàn thiện lớp eval pipeline theo skill.
- Render `results/runs/report.html` offline 1 file (metrics + loss + mẫu tốt/xấu + latency) theo `modules/visualization.md`.
- Viết demo 1 lệnh `python -m src.demo` + `results/demo/<ts>/index.html` theo `modules/demo.md`.

## 7. Hồ sơ artifact

- Config: `configs/model_config.yaml`, `configs/hyperopt_config.yaml`
- Checkpoint: `models/checkpoints/trained_model.pt` (best epoch 11), `last_model.pt` (local-only), `final_model.pt` (kèm metrics)
- History/metrics: `results/runs/training_history.json`, `hyperopt.json`, `hyperopt_best.json`, `test_metrics.md`
- Plots: `loss_curve.png`, `accuracy_curve.png`, `confusion_matrix.png`, `predictions.png`, 4 EDA PNG
- Báo cáo: `docs/reports/nguoi1_models_report.md`, `nguoi2_training_report.md`, `nguoi3_evaluation_report.md`, `docs/PHASE1_REPORT.md`, `docs/PROJECT_METHODOLOGY.md`
- Tái tạo: `python main.py` (train) · `python run_experiments.py --smoke` / `--batch-size 128` (grid) · `$env:PYTHONPATH="."; python src/evaluation/run_evaluation.py` (eval) · `python -m pytest tests -q` (27 passed)

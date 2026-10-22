# Báo cáo Người 3 — Evaluation + Save/Load + Display

> **Trạng thái:** Hoàn thành triển khai và đánh giá; chờ nghiệm thu cuối.
> **Ngày cập nhật:** 2026-10-15
> **Nguồn:** Checkpoint từ Người 2 (`trained_model.pt`), `docs/PHASE2_PLAN.md`, báo cáo Người 2.

## 1. Phạm vi và phân loại yêu cầu

| Nhãn | Phạm vi |
|---|---|
| ĐỀ BÀI | Đánh giá accuracy, precision, recall, F1; hiển thị predicted vs actual; save/load model. |
| BỔ SUNG | Confusion matrix, error analysis, visualization, final model save kèm metrics. |

## 2. Thành phần đã hoàn thành

### 2.1. Model Evaluation

- [x] Load best checkpoint từ Người 2 (`trained_model.pt`).
- [x] Compute metrics: accuracy, precision (macro), recall (macro), F1 (macro).
- [x] Evaluator class (`src/evaluation/evaluator.py`) với predict() và evaluate().

### 2.2. Confusion Matrix

- [x] `src/evaluation/confusion_matrix.py` với normalized confusion matrix.
- [x] Heatmap visualization với class labels.
- [x] Save to PNG.

### 2.3. Error Analysis

- [x] `src/evaluation/error_analysis.py` với analyze_common_confusions().
- [x] Top K confused class pairs.

### 2.4. Predicted vs Actual Display

- [x] `src/evaluation/visualizer.py` với PredictionVisualizer.
- [x] Grid display với color coding (green=correct, red=wrong).
- [x] Save to PNG.

### 2.5. Model I/O

- [x] `src/evaluation/model_io.py` với ModelIO.load_model() và save_model().
- [x] Save final model kèm metrics.

### 2.6. Entry Point

- [x] `src/evaluation/run_evaluation.py` orchestration script.

## 3. Kết quả đánh giá

### 3.1. Test Set Metrics

| Metric | Value |
|---|---|
| **Test Accuracy** | **92.10%** |
| Precision (macro) | 92.11% |
| Recall (macro) | 92.10% |
| F1-Score (macro) | 92.09% |

### 3.2. Top 5 Confused Class Pairs

| # | True Class | → Predicted | Count |
|---|---|---|---|
| 1 | Shirt | T-shirt/top | 95 |
| 2 | T-shirt/top | Shirt | 90 |
| 3 | Shirt | Pullover | 64 |
| 4 | Coat | Pullover | 63 |
| 5 | Coat | Shirt | 55 |

**Nhận xét:** Shirt và T-shirt/top dễ bị nhầm lẫn nhất (185 lần), do hình dạng tương tự. Coat và Pullover cũng hay bị nhầm.

## 4. Artifact bàn giao

| File | Mô tả |
|---|---|
| `models/checkpoints/final_model.pt` | Final model với metrics |
| `results/runs/confusion_matrix.png` | Confusion matrix heatmap |
| `results/runs/predictions.png` | Predicted vs actual images |

## 5. Danh sách file đã tạo/sửa

### Tạo mới

- `src/evaluation/model_io.py` (implement đầy đủ)
- `src/evaluation/visualizer.py` (implement PredictionVisualizer)
- `src/evaluation/run_evaluation.py`

### Sửa

- `src/evaluation/error_analysis.py` (fix encoding)
- `src/evaluation/evaluator.py`
- `src/evaluation/confusion_matrix.py` (rename từ "confusion matrix.py")

## 6. Trạng thái bàn giao

- [x] Code evaluation hoàn thành.
- [x] Full evaluation run thành công.
- [x] Artifacts đã sinh và kiểm tra được.
- [x] Báo cáo Người 3 đã tạo.
- [x] `progress_status.md` đã cập nhật.
- [x] `README.md` đã cập nhật.

## 7. Cách chạy

```powershell
cd e:\lab1
$env:PYTHONPATH = "."
python src/evaluation/run_evaluation.py
```

## 8. Tóm tắt

Phase 2 đã hoàn thành 100%:
- ✅ Người 1: Model selection + hyperparameter tuning
- ✅ Người 2: Training loop + visualization
- ✅ Người 3: Evaluation + confusion matrix + error analysis + display

**Kết quả cuối cùng: Test Accuracy 92.10%** (vượt mục tiêu 90%).

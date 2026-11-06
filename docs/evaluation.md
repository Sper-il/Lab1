# Báo cáo tổng quan Evaluation

> Báo cáo chính của pipeline đánh giá — tổng hợp toàn bộ bước eval trên test set giữ kín, không phải báo cáo cá nhân.
> Báo cáo cá nhân của Người 3 xem tại `docs/reports/nguoi3_evaluation_report.md`.
> Nguồn code: `src/evaluation/`, entry `src/evaluation/run_evaluation.py`.
> Checkpoint đầu vào: `models/checkpoints/trained_model.pt` (best epoch 11 từ training).
> Số liệu: `results/runs/test_metrics.{json,md}`, `confusion_matrix.png`, `predictions.png`.

## 1. Mục tiêu

- Đánh giá **một lần duy nhất** trên test set 10.000 ảnh giữ kín (không tuning trên test).
- Báo cáo 2 lớp theo chuẩn skill: model level (accuracy, precision, recall, F1 + confusion matrix) và error analysis (top cặp nhầm + gợi ý cải thiện).
- Verify vòng save → load → predict cho cùng kết quả, bàn giao `final_model.pt` kèm metrics.

## 2. Quy trình (`python src/evaluation/run_evaluation.py`)

```
trained_model.pt (best epoch 11)
  → DataSplitManager (test_loader, batch 64)
  → ModelFactory.create("simple_cnn") + ModelIO.load_model
  → ModelEvaluator.predict (no_grad, softmax → probs)
  → ModelEvaluator.evaluate (accuracy/precision/recall/F1 macro)
  → ErrorAnalyzer.analyze_common_confusions (top-5)
  → save_metrics_report → results/runs/test_metrics.{json,md}
  → PredictionVisualizer (20 ảnh, xanh=đúng đỏ=sai) → predictions.png
  → ConfusionMatrixPlotter (normalized) → confusion_matrix.png
  → ModelIO.save_model → models/checkpoints/final_model.pt (kèm metrics)
```

Lệnh chạy:

```powershell
cd e:\lab1
$env:PYTHONPATH = "."
python src/evaluation/run_evaluation.py
```

Nếu thiếu checkpoint, script báo lỗi rõ và dừng (yêu cầu chạy training của Người 2 trước), không crash khó hiểu.

## 3. Module và API

### 3.1. `ModelEvaluator` (`src/evaluation/evaluator.py`)

- `predict() -> (preds, labels, probs)`: `model.eval()` + `torch.no_grad()`, softmax theo dim=1 lấy confidence, `argmax` lấy nhãn; chuyển về NumPy để tính metric bằng scikit-learn.
- `evaluate() -> (metrics, preds, labels, probs)`: accuracy, precision/recall/F1 **macro** (trung bình không trọng số trên 10 lớp — công bằng với lớp khó), in ra console + trả về dict.
- `CLASS_NAMES` cố định đúng thứ tự index 0–9 (T-shirt/top … Ankle boot).

### 3.2. `ModelIO` (`src/evaluation/model_io.py`)

- `load_model(model, path, device) -> (model, checkpoint)`: chấp nhận cả checkpoint dict (`model_state_dict` / `state_dict` / raw state), chuyển về device, `model.eval()`; thiếu file → `FileNotFoundError` với đường dẫn rõ ràng.
- `save_model(model, path, model_name, metrics, **metadata)`: tự tạo thư mục cha, lưu `model_state_dict` + `model_name` + `metrics` + metadata bổ sung.
- Verify chuẩn: load `trained_model.pt` → predict → save `final_model.pt` → load lại cho cùng predictions.

### 3.3. `ConfusionMatrixPlotter` (`src/evaluation/confusion_matrix.py`)

- `compute_matrix` (raw counts) + `plot(normalize=True)`: chuẩn hóa theo hàng (recall từng lớp), heatmap Seaborn `Blues` 10×10 kèm tên lớp, figsize 10×8, dpi 100.
- Output: `results/runs/confusion_matrix.png`.

### 3.4. `ErrorAnalyzer` (`src/evaluation/error_analysis.py`)

- `get_misclassified_samples`: lọc index predict sai.
- `analyze_common_confusions(true, pred, top_k=5)`: đếm cặp (True → Pred), sắp xếp giảm dần, in bảng console, trả về list đã sắp xếp để ghi báo cáo.

### 3.5. `PredictionVisualizer` (`src/evaluation/visualizer.py`)

- `plot_predictions(images, true, pred, num_samples=20, save_path)`: grid 5 cột, squeeze `(1,28,28) → (28,28)`, cmap gray, tiêu đề `True/Pred` màu xanh (đúng) / đỏ (sai); tự tạo thư mục, tương thích headless (backend Agg không gọi `show()` chặn luồng).
- `plot_confidence_distribution(probs, true)`: histogram confidence đúng (xanh) vs sai (đỏ) — phát hiện overconfidence.
- Output: `results/runs/predictions.png`.

### 3.6. `save_metrics_report` (`src/evaluation/metrics_report.py`)

- Ghi `results/runs/test_metrics.json` (metrics + top confusions, máy đọc) và `test_metrics.md` (bảng markdown, người đọc) từ cùng một nguồn — không lệch số.

## 4. Kết quả trên test set (10.000 ảnh)

| Chỉ số | Giá trị | Mục tiêu | Đạt? |
|--------|---------|----------|------|
| Accuracy | **92,10%** | ≥ 90% | ✅ |
| Precision (macro) | 92,11% | — | ✅ |
| Recall (macro) | 92,10% | — | ✅ |
| F1-Score (macro) | 92,09% | — | ✅ |

### Top-5 cặp lớp bị nhầm

| # | Lớp thực | Bị nhầm thành | Số lần |
|---|-----------|--------------|--------|
| 1 | Shirt | T-shirt/top | 95 |
| 2 | T-shirt/top | Shirt | 90 |
| 3 | Shirt | Pullover | 64 |
| 4 | Coat | Pullover | 63 |
| 5 | Coat | Shirt | 55 |

**Nhận xét:** 185/367 lỗi top-5 nằm ở cặp Shirt ↔ T-shirt/top (hình dáng gần như giống hệt ở 28×28); nhóm Coat/Pullover/Shirt cũng tương tự. Đây là giới hạn của độ phân giải thấp + model CNN nhỏ, không phải bug pipeline. Hướng cải thiện: augmentation nhẹ (shift/rotation), label smoothing, hoặc model lớn hơn nếu có GPU.

## 5. Artifact bàn giao

| File | Nội dung |
|------|----------|
| `models/checkpoints/final_model.pt` | Weights + `model_name=simple_cnn` + metrics test (bàn giao cuối) |
| `results/runs/test_metrics.json` | Metrics + top confusions (máy đọc) |
| `results/runs/test_metrics.md` | Bảng markdown (người đọc) |
| `results/runs/confusion_matrix.png` | Heatmap normalized 10×10 |
| `results/runs/predictions.png` | 20 ảnh predicted vs actual (xanh=đúng, đỏ=sai) |

## 6. Đối chiếu với training

- Best validation (epoch 11): val_acc 92,85% → test 92,10%: gap 0,75% là mức generalization bình thường, không có dấu hiệu leakage (test giữ kín đến bước này mới mở).
- Train_acc cuối 94,72% vs test 92,10%: overfit nhẹ đã dự báo từ training, nằm trong ngưỡng chấp nhận được của model CNN nhỏ trên CPU.

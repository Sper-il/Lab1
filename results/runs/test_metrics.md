# Test Set Evaluation — SimpleCNN

Nguồn: `python src/evaluation/run_evaluation.py` (checkpoint `models/checkpoints/trained_model.pt`)

## 1. Metrics trên tập test (10.000 ảnh)

| Chỉ số | Giá trị |
|--------|---------|
| **Accuracy** | **92.10%** |
| Precision (macro) | 92.11% |
| Recall (macro) | 92.10% |
| F1-Score (macro) | 92.09% |

## 2. Top 5 cặp lớp bị nhầm nhiều nhất

| # | Lớp thực | Bị nhầm thành | Số lần |
|---|-----------|--------------|---------|
| 1 | Shirt | T-shirt/top | 95 |
| 2 | T-shirt/top | Shirt | 90 |
| 3 | Shirt | Pullover | 64 |
| 4 | Coat | Pullover | 63 |
| 5 | Coat | Shirt | 55 |

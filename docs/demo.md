# Báo cáo tổng quan Demo

> Báo cáo chính về khả năng trình diễn của project — cách chạy thử end-to-end cho người mới trong dưới 5 phút.
> Chuẩn skill: `.cursor/skills/ai-fusion-workflow/modules/demo.md` (demo 1 lệnh, input mẫu, đo latency live).
> **Trạng thái hiện tại:** chưa có `src/demo.py` riêng; demo thực hiện bằng 3 entry point có sẵn + artifacts đã sinh. Mục 6 là roadmap để đạt chuẩn 1 lệnh.

## 1. Mục tiêu

- Người mới clone repo, cài `requirements.txt`, chạy demo thấy ngay Input (ảnh 28×28) → Output (nhãn + confidence) mà không cần đọc code.
- Demo chứng minh đúng các `core features` mức Must: load data → forward → predict → hiển thị predicted vs actual.
- Mọi output demo đều là số/ảnh thật từ checkpoint `trained_model.pt` (test 92,10%), không dựng số.

## 2. Kịch bản demo (3 case)

| Case | Input | Kỳ vọng | Bằng chứng có sẵn |
|------|-------|---------|-------------------|
| Dễ (model đúng) | Ảnh Trouser / Sneaker / Bag (shape rõ, ít nhầm) | Nhãn đúng, confidence cao | `results/runs/predictions.png` (ô xanh) |
| Khó (nhóm dễ nhầm) | Ảnh Shirt vs T-shirt/top (185/367 lỗi top-5) | Có thể sai, tiêu đề ô đỏ — dùng để chứng minh trung thực | `results/runs/predictions.png` (ô đỏ) + `confusion_matrix.png` |
| Biên (tổng thể) | Toàn bộ test 10.000 ảnh | Accuracy 92,10%, đúng ~9.210/10.000 | `results/runs/test_metrics.{json,md}` |

Ảnh mẫu từng lớp: `results/runs/per_class_samples.png`, `sample_images.png`.

## 3. Cách chạy demo hiện tại (3 lệnh, không cần code mới)

```bash
# 0. Cài đặt (1 lần)
pip install -r requirements.txt

# 1. Smoke demo — forward + loss giảm trên 64 ảnh (~5 giây, CPU)
python run_experiments.py --smoke

# 2. Full eval demo — predict 10.000 test + vẽ predictions + confusion matrix (~1-2 phút, CPU)
$env:PYTHONPATH = "."
python src/evaluation/run_evaluation.py

# 3. Xem output (double-click, không cần chạy gì thêm)
# results/runs/report.html           — báo cáo trực quan 1 file (offline, mở là xem được)
# results/runs/predictions.png       — 20 ảnh predicted vs actual (xanh=đúng, đỏ=sai)
# results/runs/confusion_matrix.png  — heatmap normalized 10×10
# results/runs/loss_curve.png        — train vs val loss 15 epochs
# results/runs/accuracy_curve.png    — train vs val accuracy 15 epochs
```

Toàn bộ path là relative, chạy trên Windows PowerShell + CPU lab, không phụ thuộc đường dẫn máy cá nhân.

## 4. Example Input → Output (số thật từ run chính thức)

**Input**

```json
{
  "image": "data/processed/test/<class_id>/*.png (28×28, grayscale)",
  "checkpoint": "models/checkpoints/trained_model.pt",
  "model": "simple_cnn (421.642 params)"
}
```

**Output (1 ảnh)**

```json
{
  "predicted_label": "Shirt",
  "true_label": "Shirt",
  "confidence": 0.94,
  "correct": true
}
```

**Output (tổng thể test 10.000 ảnh)**

```json
{
  "accuracy": 92.10,
  "precision_macro": 92.11,
  "recall_macro": 92.10,
  "f1_macro": 92.09,
  "top_confusion": "Shirt → T-shirt/top: 95 lần",
  "correct_images": 9210,
  "wrong_images": 790
}
```

Đoạn code tối thiểu để predict 1 ảnh (dùng đúng API project, xem `docs/api.md`):

```python
import torch
from src.models.model_factory import ModelFactory
from src.evaluation.model_io import ModelIO

model = ModelFactory.create("simple_cnn")
model, ckpt = ModelIO.load_model(model, "models/checkpoints/trained_model.pt", device="cpu")
model.eval()
with torch.no_grad():
    logits = model(image_batch)          # (B,1,28,28) → (B,10)
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
```

## 5. Giới hạn demo hiện tại

- Chưa có script 1 lệnh duy nhất (`python -m src.demo`); phải chạy 2 lệnh (smoke + eval).
- Chưa đo latency live (inference ms/mẫu, end-to-end ms) — chỉ có thời gian train (~75 s/epoch CPU).
- Chưa có input mẫu nhẹ commit kèm repo (`data/processed/demo/`), chưa có `results/demo/<ts>/` + `index.html` rút gọn.
- Chưa có giao diện (Gradio/Streamlit/FastAPI); demo là CLI + xem PNG.
- Chưa có case thất bại được đóng gói riêng — phải tự tìm ô đỏ trong `predictions.png`.

## 6. Roadmap đạt chuẩn demo 1 lệnh (chưa làm)

1. Tạo `data/processed/demo/` với 3–5 ảnh mẫu nhẹ (< 5 MB): 1 dễ (Trouser), 1 khó (Shirt), 1 biên (Coat).
2. Viết `src/demo.py` (hoặc `src/demo/__main__.py`): load config + `trained_model.pt` → preprocess → inference → postprocess → in `inference_time_ms` + `end_to_end_ms` + confidence → lưu output vào `results/demo/<timestamp>/` kèm `metrics.json` mini.
3. Đồng bộ khối Example Input → Output (mục 4) vào `README.md` mục 7.
4. (Tùy chọn) Thêm `notebooks/demo.ipynb` chạy từ trên xuống không sửa code, hoặc Gradio tối giản ghi rõ port.

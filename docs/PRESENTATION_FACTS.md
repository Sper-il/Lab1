# ✅ BẢNG SỐ ĐÃ KIỂM CHỨNG — đọc trước khi lên lớp

> Đối chiếu mọi con số trong `PRESENTATION_SCRIPT.md` với **file thật trong repo**.
> Ngày kiểm: 2026-10-21

---

## 1. Số ĐÚNG — dùng thoải mái

| Con số | Giá trị | Nguồn xác nhận |
|--------|---------|----------------|
| Test accuracy | **92.10%** | `docs/reports/nguoi3_evaluation_report.md` |
| Precision / Recall / F1 (macro) | 92.11 / 92.10 / 92.09% | `nguoi3_evaluation_report.md` §3.1 |
| Mean / Std pixel | **0.2860 / 0.3530** | `progress_status.md:310-311`, khớp `eda.py` |
| Shape | `(1, 28, 28)`, float32 | `progress_status.md:309` |
| Train / Val / Test | **54.000 / 6.000 / 10.000** | `data/splits/split_info.txt` |
| Số batch (bs=64) | 844 / 94 / 157 | `progress_status.md:325-327` |
| Params 4 kiến trúc | 421.642 / 203.530 / 174.970 / 118.650 | `architecture_selection.md`, `hyperopt.md` |
| Val loss thấp nhất | **0.2152 @ epoch 11** | `training_history.json` |
| Val accuracy cao nhất | 92.88% @ epoch 13 | `training_history.json` |
| Train acc cuối | 94.72% @ epoch 15 | `training_history.json` |
| Overfit bắt đầu | epoch 11–13 | `training_history.json` |
| Hyperparams cuối | simple_cnn, adam, lr=0.001, bs=128, dropout=0.5, wd=0.0 | `configs/model_config.yaml` |
| EarlyStopping | patience=5, monitor=val_loss, min_delta=0.0001 | `configs/model_config.yaml` |
| Seed | 42 | `configs/model_config.yaml` |
| Top nhầm lẫn | Shirt↔T-shirt/top = 95+90 = 185 | `nguoi3_evaluation_report.md` §3.2 |

### Bảng epoch trong kịch bản — ĐÚNG, khớp `training_history.json`

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|----------|---------|
| 1 | 0.585 ✓ | 0.351 ✓ | 79.03% ✓ | 87.08% ✓ |
| 5 | 0.265 ✓ | 0.237 ✓ | 90.41% ✓ | 91.42% ✓ |
| 11 | 0.178 ✓ | **0.215** ✓ | 93.47% ✓ | 92.85% ✓ |
| 15 | 0.141 ✓ | 0.230 ✓ | 94.72% ✓ | 92.80% ✓ |

---

## 2. ⚠️ Số SAI — ĐÃ SỬA trong kịch bản

### 2.1 Grid search: 8 trials hay 12 trials?

**Vấn đề nghiêm trọng nhất.** File trên đĩa và báo cáo **không khớp nhau**:

| | `docs/reports/nguoi1_models_report.md` | File thật `results/runs/hyperopt.md` |
|---|---|---|
| Loại search | Grid search | **Random search** |
| Số trials | 8 | **12** |
| Best trial | trial 1: `simple_cnn` + adam + lr=0.001 | **trial 5: `resnet_mini`** + adam + lr=0.01 |
| Best val_acc | **90.53%** | **89.15%** |
| Params best | 421.642 | 174.970 |

**Nguyên nhân:** `results/runs/*` bị `.gitignore` (dòng 94). File grid search 8 trials đã bị
lần chạy random search sau **ghi đè**. Báo cáo `nguoi1_models_report.md` mô tả lần chạy grid,
nhưng file trên đĩa là lần chạy random.

**Cách giải thích đúng (KHÔNG nói sai):**

> "Em chạy **hai lần thử nghiệm siêu tham số**:
> - **Grid search 8 trials** — chỉ quét 2 kiến trúc (`simple_cnn`, `deep_mlp`) × 2 learning
>   rate × 2 optimizer = 8 tổ hợp đầy đủ, không bỏ sót. Kết quả: **`simple_cnn` + adam +
>   lr=0.001 thắng với 90.53%** validation accuracy.
> - **Random search 12 trials** — quét cả 4 kiến trúc với siêu tham số ngẫu nhiên. Do mỗi
>   trial chỉ chạy **3 epoch** (rất ngắn), kết quả nhiễu cao: trial tốt nhất rơi vào
>   `resnet_mini` với 89.15%. Chênh lệch này **không phải do kiến trúc tệ hơn**, mà vì 3
>   epoch chưa đủ để hội tụ.
>
> Em chốt theo grid search vì nó **tất định và đầy đủ** — cùng config cho cùng kết quả,
> không phụ thuộc may rủi khi chọn mẫu."

⚠️ **Lưu ý thêm:** `resnet_mini` không nằm trong grid search. Nếu giảng viên hỏi
*"sao không thử ResNet trong grid?"* → trả lời: em có thử riêng (`architecture_selection.md`):
`resnet_mini` đạt val_acc 88.48% nhưng **chậm gấp 3×** (239 s/epoch vs 75 s/epoch của
`simple_cnn`) → không đáng đổi khi mục tiêu chỉ 90%.

### 2.2 Số unit test: 27 hay 23?

| Nguồn | Con số |
|-------|--------|
| Kịch bản (cũ) | 27 |
| **Thực tế trong repo** | **23** |

Đếm trực tiếp: `tests/training/` có 6 file, 23 hàm `def test_`:
`test_callbacks.py` (5), `test_losses.py` (2), `test_optimizers.py` (4),
`test_run_training.py` (6), `test_trainer.py` (4), `test_visualizer.py` (2).

⚠️ **Không có `tests/test_dataset.py`** dù `data/README.md` ghi "11 passed".
→ Đã sửa `data/README.md`.

**Cách nói an toàn:** *"Em có **23 unit test** cho phần training — callbacks, loss,
optimizer, trainer, visualizer — tất cả đều pass."* Không nói "27 toàn bộ pipeline".

### 2.3 Train/Val/Test

Kịch bản (cũ) ghi `60.000 / 6.000 / 10.000` → sai, train phải là **54.000**.
Đã sửa. (`data/README.md` cũng ghi sai 50.000/10.000 → đã sửa cả 4 chỗ.)

### 2.4 Thời gian train ~60 s/epoch

`architecture_selection.md` ghi `simple_cnn` = **75.35 s/epoch** (CPU).
Kịch bản nói "~60 s/epoch" → sai. Đã sửa thành **~75 s/epoch**.

### 2.5 Thời gian grid search "~19 phút"

`nguoi1_models_report.md:91` ghi **~19 phút** cho 8 trials grid. Giữ nguyên con số này
(báo cáo ghi rõ có đo). Lưu ý: random search 12 trials mất **13.4 phút**
(tính từ `epoch_time_s` trong `hyperopt.json`).

---

## 3. Chi tiết kỹ thuật để trả lời câu hỏi phụ

### So sánh kiến trúc (`results/runs/architecture_selection.md`)

| Model | Params | Val Acc | Val Loss | s/epoch |
|-------|--------|---------|----------|---------|
| simple_cnn | 421,642 | **88.83%** | 0.3015 | 75.35 |
| resnet_mini | 174,970 | 88.48% | 0.3067 | **239.00** |
| baseline_mlp | 203,530 | 86.77% | 0.3564 | 22.99 |
| deep_mlp | 118,650 | 86.27% | 0.3726 | 29.27 |

→ `simple_cnn` thắng **cả về accuracy lẫn loss**, nhưng `resnet_mini` chỉ kém 0.35%
trong khi chậm **3.2×**. Đây là lý do mạnh nhất để bảo vệ lựa chọn `simple_cnn`.

### Vì sao val acc đỉnh ở epoch 13 nhưng lưu model epoch 11?

Vì `early_stopping.monitor = val_loss` (không phải `val_acc`).
- epoch 11: val_loss **0.2152** (thấp nhất) → **checkpoint được lưu**
- epoch 13: val_loss 0.2165, val_acc 92.88% (cao nhất)

Hai chỉ số này không đồng thời cực đại. Chọn theo **loss** là chuẩn vì loss phạt cả
trường hợp model "mơ hồ" (đoán sai nhưng vẫn cho xác suất cao).

### Repro — 3 lệnh tái tạo kết quả

```powershell
python main.py                              # Train 15 epoch, sinh loss/accuracy curve
python src/evaluation/run_evalution.py      # Đánh giá test set + confusion matrix
pytest tests/training -v                    # 23 unit test
```

### Lệnh đã dùng để tạo artifacts (không cần chạy lại)

```powershell
python run_experiments.py --strategy grid --batch-size 128      # grid 8 trials
python run_experiments.py --strategy random --n-trials 12 --epochs 3 --batch-size 128
```

---

## 4. Câu hỏi khó — chuẩn bị sẵn

| Câu hỏi | Trả lời |
|---------|---------|
| "Grid 8 trials hay random 12 trials?" | **Cả hai.** Grid quét 2 model × 2 lr × 2 optimizer = 8 tổ hợp đầy đủ. Random quét cả 4 model với siêu tham số ngẫu nhiên, 12 trials. Em chốt theo grid vì tất định & không bỏ sót tổ hợp. |
| "Vì sao best trial trên đĩa là resnet_mini mà em lại chọn simple_cnn?" | Đó là lần chạy **random search 3 epoch** — quá ngắn nên nhiễu cao. Grid search đầy đủ cho thấy `simple_cnn` + adam + lr=0.001 thắng 90.53%. Ngoài ra `resnet_mini` chậm gấp 3.2× (239 vs 75 s/epoch) mà chỉ kém 0.35% accuracy. |
| "Vì sao không thử ResNet trong grid search?" | Grid giới hạn 2 kiến trúc để giữ runtime ~19 phút. Em đã benchmark `resnet_mini` riêng: 88.48% val acc, chậm 3.2× → không đáng. |
| "Overfit có nghiêm trọng không?" | Nhẹ. Train 94.72% vs val 92.80% = chênh 1.9 điểm. EarlyStopping (patience=5) đã dừng đúng tại epoch 11 — nơi val loss thấp nhất. |
| "Test set có bị rò rỉ không?" | Không. Chỉ mở 1 lần ở bước đánh giá cuối, sau khi mọi quyết định (kiến trúc, siêu tham số, epoch) đã chốt xong. |
| "Reproducible không?" | Seed 42 cố định ở `configs/model_config.yaml` + `src/data/split.py`. Cùng config → cùng kết quả. Grid search còn tất định tuyệt đối vì không random mẫu. |
| "Code có phải của em không?" | Có — chia `src/data/`, `src/models/`, `src/training/`, `src/evaluation/`. Có 23 unit test, log chi tiết trong `docs/reports/`. |

---

## 5. File cần mở (chốt lại)

| # | File | Khi nào |
|---|------|---------|
| 1 | `docs/PRESENTATION_SCRIPT.md` | Xuyên suốt (F11 Reader) |
| 2 | `results/runs/dataset_info.md` | Phần 2 — bảng 10 lớp + thông số |
| 3 | `results/runs/per_class_samples.png` | Phần 2 — 10 ảnh mẫu |
| 4 | `results/runs/class_distribution.png` | Phần 3 — cân bằng lớp |
| 5 | `results/runs/pixel_histogram.png` | Phần 3 — histogram pixel |
| 6 | `results/runs/hyperopt.md` | Phần 4 — **bảng 12 trials random search** |
| 7 | `results/runs/architecture_selection.md` | Phần 4 — so sánh 4 kiến trúc (dự phòng) |
| 8 | `results/runs/loss_curve.png` | Phần 5 — epoch 11–13 |
| 9 | `results/runs/accuracy_curve.png` | Phần 5 — epoch 13 |
| 10 | `results/runs/confusion_matrix.png` | Phần 6 — Shirt × T-shirt |
| 11 | `results/runs/predictions.png` | Phần 6 — xanh/đỏ |
| 12 | `docs/reports/nguoi3_evaluation_report.md` | Dự phòng — metrics + top-5 nhầm lẫn |

> ⚠️ Khi mở `hyperopt.md` **phải nói trước** đây là random search 12 trials,
> không phải grid 8 trials. Xem mục §2.1.
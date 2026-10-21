# 🎤 BẢNG TRÌNH BÀY 10 PHÚT — FashionMNIST Classification

**Người trình bày:** Sper-il
**Quy ước:** Mỗi phần có **Mở file** (ảnh/diagram) + **Nói** (lời thoại rút gọn) + **Con số phải nhớ**

> Mở tab theo thứ tự, đọc lời từng dòng, không cần nhớ — file này dài 1 trang A4.
> Tài liệu đầy đủ: `PRESENTATION_CHEATSHEET.md` (lời thoại chi tiết từng phần) · Số đã kiểm chứng: `PRESENTATION_FACTS.md`

---

## ⏱️ Bảng tổng hợp 10 phút

| # | Thời gian | Mở file | Nói (4–6 câu rút gọn) |
|---|-----------|---------|-----------------------|
| **1** | `0:00 – 0:45` (45s) | — | Mở đầu: tên + đồ án. Đi cả vòng đời ML. Code public trên GitHub. |
| **2** | `0:45 – 1:45` (1m) | `dataset_info.md` + `per_class_samples.png` | Bài toán: ảnh 28×28 xám, **10 lớp**, 70k ảnh (54/6/10k). Mục tiêu **≥ 90%**. |
| **3** | `1:45 – 3:15` (1m30) | `class_distribution.png` + `pixel_histogram.png` | **EDA**: cân bằng 6k/lớp → không cần xử lý mất cân bằng. Mean 0.286 / std 0.353. |
| **4** | `3:15 – 5:45` (2m30) | `architecture_selection.md` + `hyperopt.md` | **4 kiến trúc** → **simple_cnn** thắng. Grid 8 trials (simple_cnn+adam+1e-3 = **90.53%**). Test ResNetMini: chậm 3.2×, chỉ kém 0.35% → không đáng. |
| **5** | `5:45 – 7:30` (1m45) | `loss_curve.png` + `accuracy_curve.png` | Trainer 5 bước + EarlyStopping (patience 5). Chạy **15 epoch**. **Best epoch 11** (val_loss 0.215), train 94.7% vs val 92.8% = overfit nhẹ → callback đã dừng đúng. |
| **6** | `7:30 – 9:15` (1m45) | `confusion_matrix.png` + `predictions.png` | Test accuracy **92.10%** (vượt 90%). Precision/Recall/F1 ≈ 92.1%. Top nhầm: **Shirt ↔ T-shirt** = 185 lần (cùng áo, cùng cổ tròn — khó phân biệt ở 28×28). |
| **7** | `9:15 – 10:00` (45s) | — | Kết luận: pipeline đầy đủ + **23 unit test** pass. Bài học lớn: quyết định từ số liệu. Hướng tới: Augmentation, ResNet18. Cảm ơn. |

---

## 📂 Thứ tự tab cần mở (Ctrl+Tab)

```
Tab 1: PRESENTATION_SCRIPT.md     ← file này (đang đọc)
Tab 2: dataset_info.md            ← #2
Tab 3: per_class_samples.png      ← #2
Tab 4: class_distribution.png     ← #3
Tab 5: pixel_histogram.png        ← #3
Tab 6: architecture_selection.md  ← #4
Tab 7: hyperopt.md                ← #4
Tab 8: loss_curve.png             ← #5
Tab 9: accuracy_curve.png         ← #5
Tab 10: confusion_matrix.png      ← #6
Tab 11: predictions.png           ← #6
```

---

## 🎯 Con số phải nhớ (in đậm để thầy không hỏi vặt)

| Con số | Ý nghĩa |
|--------|----------|
| **92.10%** | Test accuracy — kết quả cuối |
| **92.11 / 92.10 / 92.09%** | Precision / Recall / F1 (macro) |
| **54k / 6k / 10k** | Train / Val / Test |
| **0.286 / 0.353** | Mean / Std của pixel |
| **90.53%** | Best val acc — Grid search 8 trials |
| **89.15%** | Best val acc — Random search 12 trials |
| **75 s/epoch** | Tốc độ train SimpleCNN (CPU) |
| **0.215** | Val loss thấp nhất (epoch 11) |
| **185** | Tổng nhầm Shirt ↔ T-shirt |
| **23** | Số unit test |
| **421.642** | Params SimpleCNN |
| **239 / 75** | s/epoch ResNetMini / SimpleCNN = 3.2× chậm hơn |

---

## ❓ 8 câu hỏi GV hay hỏi + trả lời 1 dòng

| Câu hỏi | Trả lời |
|---------|---------|
| Vì sao chọn seed 42? | Quy ước cố định — quan trọng là không đổi, không phải giá trị. |
| Train vs Val chênh nhau, có overfit? | Nhẹ (94.7% vs 92.8%); val loss tăng từ epoch 13 → **EarlyStopping đã xử lý**. |
| Test có bị rò rỉ? | Không — chỉ mở 1 lần ở bước đánh giá cuối. |
| Vì sao không GPU? | Máy không CUDA; 28×28 + 60k ảnh → CPU đủ. |
| Reproducible? | Seed 42 cố định khắp nơi (config, split, training). Grid search tất định tuyệt đối. |
| Code tự viết? | Có — `src/{data,models,training,evaluation}/`, có 23 unit test. |
| Grid 8 hay random 12? | **Cả hai**. Grid đầy đủ + tất định em chốt. Random quét 4 model. |
| Best trên đĩa là ResNet, sao không chọn? | Random 3 epoch quá ngắn. Grid cho simple_cnn thắng 90.53%. ResNet chậm 3.2× mà chỉ kém 0.35%. |

---

## 🔧 Câu hỏi nâng cao — chuẩn bị sẵn

| Câu hỏi | Gợi ý trả lời |
|---------|---------------|
| "Vì sao CNN thắng MLP?" | MLP coi pixel độc lập, không hiểu cấu trúc 2D; CNN chia sẻ filter, học được đặc trưng không gian, ít tham số hơn. |
| "Vì sao Adam thắng SGD?" | Adam tự chỉnh learning rate từng tham số; cùng lr Adam hơn SGD 7–10% (xem bảng grid). |
| "Nếu cải thiện thì làm gì?" | Augmentation (+1–2%), ResNet18 (+1–3%), LR scheduler (+0.5–1%), Ensemble (+1–2%). Nhưng trade-off không đáng khi đã đạt 92%. |
| "Tại sao val acc đỉnh ở epoch 13 mà lưu epoch 11?" | Vì monitor là `val_loss` (không phải `val_acc`); epoch 11 có val_loss thấp nhất 0.215. |
| "Mean/Std tính từ đâu?" | Tính từ toàn bộ tập train ở bước EDA (mean=0.2860, std=0.3530), không leak vào val/test. |

---

## ⚠️ Bẫy hay gặp — đừng nói

| ❌ Đừng nói | ✅ Nói thay |
|------------|------------|
| "27 unit test" | **23 unit test** (trong `tests/training/`) |
| "Train 60.000 ảnh" | **54.000** train, 6.000 val, 10.000 test |
| "~60 s/epoch" | **~75 s/epoch** (`architecture_selection.md`) |
| "Grid 8 trials có ResNet" | Grid chỉ 2 model (simple_cnn + deep_mlp) |
| "Best trial là ResNet trên đĩa" | Đó là **random 3 epoch**, em chốt theo **grid 8 trials** = simple_cnn 90.53% |

---

## ⌨️ Phím tắt thao tác

| Phím | Tác dụng |
|------|----------|
| `Ctrl+P` → gõ tên file → Enter | Mở file nhanh trong VSCode |
| `Ctrl+Tab` | Chuyển qua lại giữa các tab |
| `F11` | Fullscreen (khi đang xem ảnh) |
| `Ctrl+Shift+P` → "Markdown Preview" | Xem file `.md` đã render bảng đẹp |

---

*Tạo: 2026-10-21 — dựa trên `PRESENTATION_CHEATSHEET.md` (lời thoại đầy đủ) + `PRESENTATION_FACTS.md` (số đã kiểm chứng).*
*Đừng quên: `PRESENTATION_FACTS.md` là file dự phòng — đọc trước khi lên lớp 5 phút.*

# 🎤 BẢNG TRÌNH BÀY 10 PHÚT — FashionMNIST Classification

**Người trình bày:** Sper-il
**Cách dùng:** Đặt file này ở chế độ Preview (`Ctrl+Shift+V`) → đọc từ trên xuống. Thấy **`[[Tab X]]`** → bấm `Ctrl+Tab` sang tab X.

> Lời thoại đầy đủ + slide plan: `PRESENTATION_CHEATSHEET.md` (dùng để luyện tập)

---

## 📂 Sơ đồ tab — mở sẵn 11 tab trước khi lên lớp

```
Tab 1: PRESENTATION_CHEATSHEET.md  ← lời thoại đầy đủ
Tab 2: dataset_info.md             ← Phần 2: bảng 10 lớp + stats
Tab 3: per_class_samples.png       ← Phần 2: 10 ảnh mẫu
Tab 4: class_distribution.png      ← Phần 3: 10 cột cân bằng
Tab 5: pixel_histogram.png         ← Phần 3: histogram pixel
Tab 6: architecture_selection.md   ← Phần 4: 4 kiến trúc + s/epoch
Tab 7: hyperopt.md                 ← Phần 4: 12 trials
Tab 8: loss_curve.png              ← Phần 5: val_loss ↑ ở epoch 11
Tab 9: accuracy_curve.png          ← Phần 5: train_acc ↑ / val_acc ngang
Tab 10: confusion_matrix.png       ← Phần 6: Shirt × T-shirt/top
Tab 11: predictions.png            ← Phần 6: xanh = đúng, đỏ = sai
```

---

## Phần 1 — Mở đầu & đặt vấn đề · 45s · `0:00`

> Em chào thầy. Em trình bày **đồ án Phân loại ảnh thời trang** với mạng nơ-ron tích chập trên FashionMNIST.
>
> Em chọn bài kinh điển này vì nó cho phép đi **đủ cả vòng đời ML thực tế** — EDA, model, siêu tham số, huấn luyện, đánh giá, phân tích lỗi — tất cả đều tự động hoá bằng code. Source đã push lên GitHub công khai.

---

## Phần 2 — Bài toán & dữ liệu · 1m · `0:45`

> **Bài toán:** Cho ảnh xám 28×28, máy phải trả lời đây là món gì, trong **10 lớp**: áo thun, quần, áo len, váy, áo khoác, dép, áo sơ mi, giày thể thao, túi xách, ủng.
>
> `[[Tab 2]]` Dataset **70.000 ảnh** — **54.000 train, 6.000 val, 10.000 test**. Mỗi lớp đúng 1.000 ảnh test.
>
> `[[Tab 3]]` Đây là ảnh mẫu từ từng lớp — tất cả **28×28 pixel**, đơn sắc. Một số cặp áo rất khó phân biệt — đây sẽ là điểm khó nhất của bài toán.
>
> **Mục tiêu:** test accuracy **≥ 90%** — em chốt mục tiêu này **trước khi code**, để biết khi nào nên dừng tìm kiếm.

---

## Phần 3 — EDA & chuẩn bị dữ liệu · 1m30 · `1:45`

> **80% lỗi trong ML đến từ dữ liệu**, không phải từ mô hình — nên em đầu tư chỗ này.
>
> **EDA:** thống kê pixel — **mean = 0.286**, **std = 0.353**, giá trị [0, 1].
>
> `[[Tab 4]]` 10 cột cao đều nhau = **cân bằng hoàn hảo**, không NaN, không ảnh hỏng → **không cần** xử lý mất cân bằng lớp.
>
> `[[Tab 5]]` Histogram có **hai đỉnh** — nền sáng và vật thể tối. Phân bố đẹp, không có outlier.
>
> **Làm sạch:** kiểm tra NaN, shape, nhãn → **60.000/60.000 hợp lệ** = 100%.
>
> **Chuẩn hóa:** `(x − 0.286) / 0.353` — gradient không bị nghiêng, model hội tụ nhanh hơn.
>
> **Chia 3 tập**, không 2 tập:
> - **Train 54k** — model học.
> - **Val 6k** — model không học, chỉ quyết định dừng.
> - **Test 10k** — **giữ kín đến cuối**, mở đúng 1 lần duy nhất.
>
> Chia 2 tập → model 'học thuộc' test → điểm đẹp hơn thực tế.
>
> **Stratified split** (giữ tỷ lệ lớp) + **seed 42** (reproducible).

---

## Phần 4 — Phát triển mô hình · 2m30 · `3:15`

> Em làm 4 bước, mỗi bước có căn cứ số liệu.
>
> **Bước 1 — Baseline:** MLP đơn giản → mức sàn để so sánh.
>
> `[[Tab 6]]` **Bước 2 — 4 kiến trúc:** SimpleCNN **88.83% val acc / 75 s-per-epoch** — thắng cả accuracy lẫn tốc độ. ResNetMini chỉ kém 0.35% nhưng chậm gấp **3.2×**.
>
> **Vì sao CNN thắng?** MLP coi mỗi pixel là độc lập — không hiểu cấu trúc không gian 2D. CNN trượt filter 3×3 và **chia sẻ trọng số**, học được đặc trưng không gian: cạnh, góc, texture.
>
> **Chọn SimpleCNN:** Conv(1→32) → MaxPool → Conv(32→64) → MaxPool → FC(3136→128) → Dropout → FC(128→10).
>
> `[[Tab 7]]` **Bước 3 — Tìm siêu tham số, hai thử nghiệm:**
> - **Grid 8 trials** (2 model × 2 lr × 2 optimizer = đầy đủ, 19 phút): `simple_cnn` + adam + lr=1e-3 → **90.53%**.
> - **Random 12 trials** (quét 4 model, 13 phút): `resnet_mini` + adam + lr=0.01 → 89.15%.
>
> **Chốt theo grid** — nó quét đầy đủ và **tất định**: cùng config cho cùng kết quả. Random 3 epoch quá ngắn để kết luận.
>
> **Bước 4 — Chốt mô hình:** SimpleCNN + Adam + lr=0.001 + batch 128 + dropout 0.5.

---

## Phần 5 — Huấn luyện · 1m45 · `5:45`

> Em train **15 epoch**.
>
> **Train loop** — 5 bước chuẩn: `zero_grad` → forward → loss → backward → step.
> **Validate loop** — `model.eval()` + `torch.no_grad()`: tắt dropout, không tính gradient, tiết kiệm **50% bộ nhớ**.
>
> Loss **CrossEntropyLoss** (gộp softmax + NLL). Optimizer **Adam**.
>
> **2 callback:** EarlyStopping patience 5 theo dõi `val_loss`; Checkpoint lưu best + last.
>
> `[[Tab 8]]` Val loss đạt **0.215 tại epoch 11** rồi tăng trở lại.
>
> `[[Tab 9]]` Train acc tăng đến **94.72%**, trong khi val acc **đi ngang quanh 92.8% từ epoch 13**.
>
> → Dấu hiệu **overfit nhẹ**. EarlyStopping đã dừng đúng lúc tốt nhất — không để model học thuộc.

---

## Phần 6 — Đánh giá & phân tích lỗi · 1m45 · `7:30`

> Em **mở tập test lần đầu** trong toàn dự án — sau khi mọi quyết định đã chốt.
>
> **Accuracy 92.10%** · Precision 92.11% · Recall 92.10% · F1 92.09% → **vượt 90%** đã đặt ra.
>
> Ba chỉ số macro gần bằng nhau chứng tỏ dataset cân bằng, model **không thiên lệch** về lớp nào.
>
> Nhưng con số tổng không thấy **sai ở đâu**. Em xem ma trận nhầm lẫn — top 5:
>
> | Lớp thực | → Bị nhầm | Số lần |
> |----------|-----------|--------|
> | Shirt | T-shirt/top | 95 |
> | T-shirt/top | Shirt | 90 |
> | Shirt | Pullover | 64 |
> | Coat | Pullover | 63 |
> | Coat | Shirt | 55 |
>
> `[[Tab 10]]` **Shirt ↔ T-shirt/top = 185 lần, gần 20% tổng số lỗi.** Hợp lý về thị giác: cùng là áo, cùng cổ tròn, cùng tay ngắn — chỉ khác ở chi tiết nhỏ về cổ áo, mà ở **28×28** thì chi tiết đó gần như không thấy.
>
> `[[Tab 11]]` Lưới ảnh dự đoán — ô xanh lá = đúng, ô đỏ = sai. Phần đỏ tập trung đúng vào các cặp áo — khớp với phân tích trên.
>
> Em cũng lưu final model kèm metrics và config, để người khác **tái tạo lại được** kết quả.

---

## Phần 7 — Kết luận · 45s · `9:15`

> **Test accuracy 92.10% — vượt mục tiêu 90%.**
>
> Về mặt kỹ thuật: EDA → làm sạch → chuẩn hóa → stratified split → 4 kiến trúc qua factory → grid + random search → train + EarlyStopping → 4 chỉ số → confusion matrix. **23 unit test pass.**
>
> Bài học lớn nhất: **tư duy thực nghiệm** — mọi quyết định về kiến trúc và siêu tham số đều đến từ **số liệu đo được**, không phải cảm tính.
>
> Hướng phát triển: Data Augmentation, ResNet18, learning rate scheduler.
>
> Cảm ơn thầy! Em sẵn sàng nhận câu hỏi.

---

## ⏱️ Tổng thời gian

| Phần | Nội dung | Từ – đến | Tab |
|------|----------|----------|-----|
| 1 | Mở đầu | 0:00 – 0:45 | — |
| 2 | Bài toán + dữ liệu | 0:45 – 1:45 | 2, 3 |
| 3 | EDA + chuẩn bị | 1:45 – 3:15 | 4, 5 |
| 4 | Mô hình + siêu tham số | 3:15 – 5:45 | 6, 7 |
| 5 | Huấn luyện | 5:45 – 7:30 | 8, 9 |
| 6 | Đánh giá + phân tích lỗi | 7:30 – 9:15 | 10, 11 |
| 7 | Kết luận | 9:15 – 10:00 | — |

---

## 🎯 12 con số phải nhớ

| Con số | Ý nghĩa |
|--------|----------|
| **92.10%** | Test accuracy — kết quả cuối |
| **92.11 / 92.10 / 92.09%** | Precision / Recall / F1 (macro) |
| **54.000 / 6.000 / 10.000** | Train / Val / Test |
| **0.286 / 0.353** | Mean / Std của pixel |
| **90.53%** | Best val acc — Grid 8 trials |
| **89.15%** | Best val acc — Random 12 trials |
| **75 s/epoch** | Tốc độ train SimpleCNN (CPU) |
| **0.215** | Val loss thấp nhất @ epoch 11 |
| **185** | Tổng nhầm Shirt ↔ T-shirt |
| **23** | Số unit test (`tests/training/`) |
| **421.642** | Params SimpleCNN |
| **239 / 75** | s/epoch ResNetMini / SimpleCNN = **3.2× chậm hơn** |

---

## ❓ Câu hỏi hay gặp — trả lời 1 dòng

| Câu hỏi | Trả lời |
|---------|---------|
| Vì sao seed 42? | Quy ước cố định — quan trọng là không đổi, không phải giá trị. |
| Train vs Val chênh, có overfit? | Nhẹ (94.7% vs 92.8%); val loss ↑ từ epoch 13 → EarlyStopping xử lý. |
| Test có bị rò rỉ? | Không — chỉ mở 1 lần ở bước đánh giá cuối. |
| Vì sao không GPU? | Máy không CUDA; ảnh 28×28 + 70k ảnh → CPU đủ. |
| Reproducible? | Seed 42 khắp nơi; grid search tất định tuyệt đối. |
| Code tự viết? | Có — `src/{data,models,training,evaluation}/`, 23 unit test. |
| Grid 8 hay random 12? | **Cả hai.** Grid đầy đủ + tất định em chốt. Random quét 4 model. |
| Trên đĩa là ResNet, sao không chọn? | Random 3 epoch quá ngắn. Grid cho simple_cnn thắng 90.53%. ResNet chậm 3.2× mà chỉ kém 0.35%. |
| Vì sao CNN thắng MLP? | MLP coi pixel độc lập; CNN chia sẻ filter nên học đặc trưng không gian. |
| Vì sao Adam thắng SGD? | Adam tự chỉnh lr từng tham số; cùng lr hơn SGD 7–10%. |

---

## ⚠️ Bẫy hay gặp — đừng nói nhầm

| ❌ Đừng nói | ✅ Nói thay |
|------------|------------|
| "27 unit test" | **23 unit test** (`tests/training/`) |
| "Train 60.000 ảnh" | **54.000** train, 6.000 val, 10.000 test |
| "~60 s/epoch" | **~75 s/epoch** |
| "Grid 8 trials có ResNet" | Grid chỉ simple_cnn + deep_mlp |
| "Best trial là ResNet" | Đó là **random 3 epoch**, em chốt theo **grid 8 trials** = simple_cnn 90.53% |

---

*Tạo: 2026-10-21*
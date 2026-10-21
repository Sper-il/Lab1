# 🎤 BẢNG TRÌNH BÀY 10 PHÚT — FashionMNIST Classification

**Người trình bày:** Sper-il
**Cách dùng:** Đặt tab `PRESENTATION_SCRIPT.md` ở chế độ Preview (Ctrl+Shift+V) → Ctrl+Tab khi thấy `[Tab X]` trong bài.

> `[[Tab X]]` = bấm Ctrl+Tab để chuyển sang tab X trong mục "Thứ tự tab" phía dưới.
> Lời thoại đầy đủ hơn: `PRESENTATION_CHEATSHEET.md` · Số kiểm chứng: `PRESENTATION_FACTS.md`

---

## 📂 Thứ tự tab — mở sẵn trước khi lên lớp (Ctrl+Tab để chuyển)

```
Tab 2: dataset_info.md            ← Phần 2: bảng 10 lớp + stats
Tab 3: per_class_samples.png      ← Phần 2: 10 ảnh mẫu
Tab 4: class_distribution.png     ← Phần 3: 10 cột cân bằng
Tab 5: pixel_histogram.png        ← Phần 3: histogram pixel
Tab 6: architecture_selection.md  ← Phần 4: so sánh 4 kiến trúc
Tab 7: hyperopt.md                ← Phần 4: 12 trials
Tab 8: loss_curve.png             ← Phần 5: val_loss tăng lại ở epoch 11
Tab 9: accuracy_curve.png         ← Phần 5: train_acc vs val_acc
Tab 10: confusion_matrix.png      ← Phần 6: Shirt × T-shirt/top
Tab 11: predictions.png           ← Phần 6: xanh = đúng, đỏ = sai
```

---

## Phần 1 — Mở đầu & đặt vấn đề  (45s)

> Em chào thầy. Em trình bày **đồ án Phân loại ảnh thời trang** với mạng nơ-ron tích chập trên FashionMNIST.
>
> Em chọn bài kinh điển này vì nó cho phép đi **đủ cả vòng đời ML thực tế** — EDA, model, siêu tham số, huấn luyện, đánh giá, phân tích lỗi — tất cả đều tự động hoá bằng code. Source đã push lên GitHub công khai, thầy có thể kiểm tra bất kỳ lúc nào.

---

## Phần 2 — Bài toán & dữ liệu  (1m)

> **Bài toán:** Cho ảnh xám 28×28, máy phải trả lời đây là món gì, trong **10 lớp**: áo thun, quần, áo len, váy, áo khoác, dép, áo sơ mi, giày thể thao, túi xách, ủng.
>
> `[[Tab 2]]` Dataset gồm **70.000 ảnh** — chia thành **54.000 train, 6.000 val, 10.000 test**. Mỗi lớp đúng 1.000 ảnh test.
>
> `[[Tab 3]]` Đây là một ảnh mẫu từ từng lớp — mời thầy quan sát: tất cả các món đều **28×28 pixel**, đơn sắc, rất khó phân biệt một số cặp áo.
>
> **Mục tiêu:** đạt test accuracy **≥ 90%**. Em chốt mục tiêu này **trước** khi code, để biết khi nào nên dừng tìm kiếm thay vì mò mãi.

---

## Phần 3 — EDA & chuẩn bị dữ liệu  (1m30)

> **80% lỗi trong ML đến từ dữ liệu**, không phải từ mô hình — nên em đầu tư chỗ này.
>
> **Bước 1 — EDA:** em tính thống kê pixel: **mean = 0.286**, **std = 0.353**, giá trị trong [0, 1].
>
> `[[Tab 4]]` 10 cột cao đều nhau = dataset **cân bằng hoàn hảo** — không cần xử lý mất cân bằng, tiết kiệm cả tuần làm việc.
>
> `[[Tab 5]]` Histogram cường độ pixel cũng phân bố đẹp, có hai đỉnh — tương ứng nền sáng và vật thể tối. Không có outlier.
>
> **Bước 2 — Làm sạch:** kiểm tra NaN, Inf, shape, nhãn. Kết quả **60.000/60.000 hợp lệ** = 100%.
>
> **Bước 3 — Chuẩn hóa:** `(x − 0.286) / 0.353` — gradient không bị nghiêng, model hội tụ nhanh hơn.
>
> **Bước 4 — Chia 3 tập**, không 2 tập:
> - **Train** 54k — model học.
> - **Val** 6k — model không học, chỉ quyết định dừng.
> - **Test** 10k — **giữ kín đến cuối**, mở đúng 1 lần.
>
> Chia 2 tập thì model 'học thuộc' test → điểm đẹp hơn thực tế.
>
> Hai kỹ thuật: **Stratified split** (giữ tỷ lệ lớp), **seed 42** (reproducible — cùng config cho cùng kết quả).

---

## Phần 4 — Phát triển mô hình  (2m30)

> Em làm 4 bước, mỗi bước có căn cứ số liệu.
>
> **Bước 1 — Baseline:** dựng MLP đơn giản để có mức sàn so sánh.
>
> `[[Tab 6]]` **Bước 2 — So sánh 4 kiến trúc:** SimpleCNN đạt **88.83% val acc, 75 s/epoch** — thắng cả về accuracy lẫn tốc độ. ResNetMini chỉ kém 0.35% nhưng chậm gấp 3.2×.
>
> Bước 2 chốt: chọn **SimpleCNN**: Conv(1→32) → MaxPool → Conv(32→64) → MaxPool → FC(3136→128) → Dropout → FC(128→10).
>
> `[[Tab 7]]` **Bước 3 — Tìm siêu tham số.** Em chạy **hai thử nghiệm**:
> - **Grid 8 trials** (quét *đầy đủ* 2 model × 2 lr × 2 optimizer, 19 phút): `simple_cnn` + adam + lr=1e-3 thắng **90.53%**.
> - **Random 12 trials** (quét cả 4 model, 13 phút): `resnet_mini` + adam + lr=0.01 đạt 89.15%.
>
> Em **chốt theo grid** vì nó quét đầy đủ, **tất định** — cùng config cho cùng kết quả. Random 3 epoch quá ngắn để kết luận `resnet` tệ hơn.
>
> **Bước 4 — Chốt mô hình:** **SimpleCNN + Adam + lr=0.001 + batch 128 + dropout 0.5**.

---

## Phần 5 — Huấn luyện  (1m45)

> Em train **15 epoch** theo `configs/model_config.yaml`.
>
> Mỗi epoch có 2 vòng:
> - **Train loop** — 5 bước chuẩn: zero_grad → forward → loss → backward → step.
> - **Validate loop** — `model.eval()` + `torch.no_grad()`: tắt dropout, tiết kiệm 50% bộ nhớ.
>
> Loss: **CrossEntropyLoss** (gộp softmax + NLL). Optimizer: **Adam**.
>
> **2 callback:**
> - **EarlyStopping** patience 5, theo dõi `val_loss`.
> - **Checkpoint** lưu best model (theo val_loss) + last model (resume nếu crash).
>
> `[[Tab 8]]` Kết quả: train loss giảm đều, val loss đạt **0.215 tại epoch 11** rồi tăng trở lại.
>
> `[[Tab 9]]` Train acc cứ tăng đến **94.72%**, val acc **đi ngang quanh 92.8%** từ epoch 13.
>
> → Dấu hiệu **overfit nhẹ**. EarlyStopping đã dừng đúng lúc tốt nhất — không để model học thuộc.

---

## Phần 6 — Đánh giá & phân tích lỗi  (1m45)

> Em **mở test lần đầu** trong toàn dự án — sau khi mọi quyết định đã chốt.
>
> 4 chỉ số: **Accuracy 92.10%** · Precision 92.11% · Recall 92.10% · F1 92.09%. **Vượt 90%** đã đặt ra.
>
> Ba chỉ số macro gần bằng nhau chứng tỏ dataset cân bằng, model **không thiên lệch** lớp nào.
>
> `[[Tab 10]]` Nhưng con số tổng không thấy **sai ở đâu**. Em xem ma trận nhầm lẫn — top 5:
>
> | Lớp thực | → Bị nhầm | Số lần |
> |----------|-----------|--------|
> | Shirt    | T-shirt/top | 95 |
> | T-shirt/top | Shirt   | 90 |
> | Shirt    | Pullover  | 64 |
> | Coat     | Pullover  | 63 |
> | Coat     | Shirt     | 55 |
>
> **Hai lớp Shirt ↔ T-shirt = 185 lần, chiếm gần 20% tổng lỗi.** Hợp lý: cả hai là áo, cùng cổ tròn, cùng tay ngắn — ở **28×28** chi tiết cổ áo gần như không thấy.
>
> `[[Tab 11]]` Đây là lưới dự đoán — ô xanh lá là đúng, ô đỏ là sai. Mời thầy thấy: phần lớn đúng, sai tập trung vào các cặp áo.
>
> Em cũng lưu final model kèm metrics và config để **tái tạo được** kết quả.

---

## Phần 7 — Kết luận  (45s)

> **Test accuracy 92.10% — vượt mục tiêu 90%.**
>
> Về mặt kỹ thuật: EDA → làm sạch → chuẩn hóa → stratified split → 4 kiến trúc qua factory → grid + random search → train + EarlyStopping → 4 chỉ số → confusion matrix. **23 unit test pass.**
>
> Bài học lớn: **tư duy thực nghiệm** — mọi quyết định (kiến trúc, siêu tham số, epoch) đều từ **số liệu đo được**, không phải cảm tính.
>
> Hướng phát triển: Augmentation, ResNet18, LR scheduler — đã ghi kế hoạch chi tiết trong `PROJECT_METHODOLOGY.md`.
>
> Cảm ơn thầy! Em sẵn sàng nhận câu hỏi.

---

## ⏱️ Tổng thời gian

| Phần | Nội dung | Từ – đến |
|------|----------|----------|
| 1 | Mở đầu | 0:00 – 0:45 |
| 2 | Bài toán + dữ liệu | 0:45 – 1:45 |
| 3 | EDA + chuẩn bị | 1:45 – 3:15 |
| 4 | Mô hình + siêu tham số | 3:15 – 5:45 |
| 5 | Huấn luyện | 5:45 – 7:30 |
| 6 | Đánh giá + lỗi | 7:30 – 9:15 |
| 7 | Kết luận | 9:15 – 10:00 |

---

## 🎯 12 con số phải nhớ (GV hỏi vặt nhiều)

| Con số | Ý nghĩa |
|--------|----------|
| **92.10%** | Test accuracy — kết quả cuối |
| **92.11 / 92.10 / 92.09%** | Precision / Recall / F1 macro |
| **54k / 6k / 10k** | Train / Val / Test |
| **0.286 / 0.353** | Mean / Std của pixel |
| **90.53%** | Best val acc — Grid 8 trials |
| **89.15%** | Best val acc — Random 12 trials |
| **75 s/epoch** | Tốc độ train SimpleCNN (CPU) |
| **0.215** | Val loss thấp nhất @ epoch 11 |
| **185** | Tổng nhầm Shirt ↔ T-shirt |
| **23** | Unit test (trong `tests/training/`) |
| **421.642** | Params SimpleCNN |
| **239 / 75** | s/epoch ResNetMini / SimpleCNN = **3.2× chậm hơn** |

---

## ❓ Câu hỏi hay gặp — trả lời 1 dòng

| Câu hỏi | Trả lời |
|---------|---------|
| Vì sao seed 42? | Quy ước cố định — quan trọng là không đổi, không phải giá trị. |
| Train vs Val chênh nhau, có overfit? | Nhẹ (94.7% vs 92.8%); val loss tăng từ epoch 13 → EarlyStopping đã xử lý. |
| Test có bị rò rỉ? | Không — chỉ mở 1 lần ở bước đánh giá cuối. |
| Vì sao không GPU? | Máy không CUDA; 28×28 + 70k ảnh → CPU đủ. |
| Reproducible? | Seed 42 cố định khắp nơi; grid search tất định tuyệt đối. |
| Code tự viết? | Có — `src/{data,models,training,evaluation}/`, 23 unit test. |
| Grid 8 hay random 12? | **Cả hai.** Grid đầy đủ + tất định em chốt. Random quét 4 model. |
| Trên đĩa là ResNet, sao không chọn? | Random 3 epoch quá ngắn. Grid cho simple_cnn thắng 90.53%. ResNet chậm 3.2× mà chỉ kém 0.35%. |

---

## ⚠️ Bẫy — đừng nói nhầm

| ❌ Đừng nói | ✅ Nói thay |
|------------|------------|
| "27 unit test" | **23 unit test** (`tests/training/`) |
| "Train 60.000 ảnh" | **54.000** train, 6.000 val, 10.000 test |
| "~60 s/epoch" | **~75 s/epoch** |
| "Grid 8 trials có ResNet" | Grid chỉ simple_cnn + deep_mlp |
| "Best trên đĩa là ResNet" | Đó là random 3 epoch, em chốt theo grid |

---

*Tạo: 2026-10-21*

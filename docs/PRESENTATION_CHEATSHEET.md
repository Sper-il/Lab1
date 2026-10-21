# KỊCH BẢN TRÌNH BÀY — FashionMNIST (10 phút)

**Người trình bày:** Sper-il

**Cách dùng:** Mở Preview (`Ctrl+Shift+V`), đọc từ trên xuống. Chỗ ghi `→ mở:` là tên file cần mở ra.

> Bản rút gọn: `PRESENTATION_SCRIPT.md` — cùng thứ tự, dùng để luyện.

---

## MỞ TRƯỚC 11 TAB

| # | Mở file này | Dùng ở phần | Chỉ vào |
|---|-------------|-------------|--------|
| 1 | `docs/PRESENTATION_CHEATSHEET.md` | Xuyên suốt | Lời thoại |
| 2 | `results/runs/dataset_info.md` | 2 | Bảng 10 lớp + thông số |
| 3 | `results/runs/per_class_samples.png` | 2 | Lưới 10 ảnh mẫu |
| 4 | `results/runs/class_distribution.png` | 3 | 10 cột cao đều |
| 5 | `results/runs/pixel_histogram.png` | 3 | Hai đỉnh pixel |
| 6 | `results/runs/architecture_selection.md` | 4 | Bảng 4 kiến trúc |
| 7 | `results/runs/hyperopt.md` | 4 | Bảng random 12 trials |
| 8 | `results/runs/loss_curve.png` | 5 | Vùng epoch 11–13 |
| 9 | `results/runs/accuracy_curve.png` | 5 | Train ↑, val ngang |
| 10 | `results/runs/confusion_matrix.png` | 6 | Hàng Shirt × cột T-shirt |
| 11 | `results/runs/predictions.png` | 6 | Xanh = đúng, đỏ = sai |

Dự phòng: `docs/reports/nguoi1_models_report.md` §2.2 (bảng grid 8 trials) · `docs/reports/nguoi3_evaluation_report.md` (metrics test)

---

## PHẦN 1 — Mở đầu · 45s · `0:00`

> "Em chào cô. Em tên là Sper-il. Hôm nay em trình bày đồ án **phân loại ảnh thời trang bằng mạng nơ-ron tích chập trên FashionMNIST**.
>
> Em chọn bài này không phải vì dễ, mà vì nó chạy được **trọn vòng đời một dự án Machine Learning thật**: khám phá dữ liệu → xây mô hình → tinh chỉnh siêu tham số → huấn luyện → đánh giá → phân tích lỗi. Tất cả chạy bằng code, không làm thủ công.
>
> Source em đẩy lên GitHub công khai, cô kiểm tra bất kỳ lúc nào."

---

## PHẦN 2 — Bài toán & mục tiêu · 45s · `0:45`

> "**Bài toán:** một ảnh xám 28×28 pixel, máy phải trả lời là món nào trong 10 loại: áo thun, quần, áo len, váy, áo khoác, dép, áo sơ mi, giày thể thao, túi xách, ủng.
>
> → mở `results/runs/dataset_info.md`
>
> Dataset có **70.000 ảnh**, chia đều 10 lớp — khoảng 6.000 ảnh gốc mỗi lớp.
>
> → mở `results/runs/per_class_samples.png`
>
> Đây là ảnh mẫu từng lớp. Cô thấy không: đều 28×28, đều đơn sắc. Nhiều cặp áo gần như giống nhau — đây sẽ là chỗ khó nhất của bài toán.
>
> **Mục tiêu em đặt trước khi code: test accuracy ≥ 90%.** Có mốc rõ ràng thì em biết khi nào nên dừng tìm."

---

## PHẦN 3 — Chuẩn bị dữ liệu · 1m30 · `1:45`

> "Giai đoạn 1 em làm 4 bước.
>
> **1. Khám phá dữ liệu.** Mean = **0.286**, std = **0.353**, pixel nằm trong [0, 1].
>
> → mở `results/runs/class_distribution.png`
>
> 10 cột cao đều nhau — dataset cân bằng hoàn hảo, không NaN, không ảnh hỏng. Nhờ vậy em không phải xử lý mất cân bằng lớp.
>
> → mở `results/runs/pixel_histogram.png`
>
> Histogram có **hai đỉnh** — nền sáng và vật thể tối. Phân bố sạch, không outlier.
>
> **2. Làm sạch.** Kiểm tra NaN/Inf, kích thước 28×28×1, nhãn 0–9. Kết quả **60.000/60.000 hợp lệ — 100%**.
>
> **3. Chuẩn hóa.** Áp dụng `(x − 0.286) / 0.353`, đúng mean và std đã tính ở bước 1. Cần vì pixel lệch giá trị làm gradient nghiêng, model hội tụ chậm hơn.
>
> **4. Chia dữ liệu — 3 tập, không phải 2:**

| Tập | Số ảnh | Vai trò |
|-----|--------|----------|
| Train | 50.000 | Model học trọng số |
| Validation | 10.000 | Đánh giá khi train, quyết định dừng |
| Test | 10.000 | Giữ kín, mở 1 lần ở cuối |

> Nếu chỉ chia 2 tập, model có thể học thuộc cả test → điểm đẹp hơn thực tế, tức là **đánh giá sai**.
>
> Hai kỹ thuật áp dụng: **stratified split** để validation cân bằng giữa 10 lớp, và **seed cố định 42** để chạy lại ra kết quả y hệt."

---

## PHẦN 4 — Mô hình & siêu tham số · 2m30 · `3:15`

> "**Bước 1 — Baseline.** Trước khi làm gì phức tạp, em dựng MLP đơn giản: 784 → 256 → 128 → 10. Để có **mức sàn** so sánh. Không có baseline thì không biết mô hình phức tạp có thật sự tốt hơn không.
>
> **Bước 2 — So sánh 4 kiến trúc**, đăng ký qua `ModelFactory` nên thêm kiến trúc mới chỉ cần 1 dòng code:
>
> → mở `results/runs/architecture_selection.md`
>
> | Kiến trúc | Tham số | Val Acc | s/epoch |
> |-----------|---------|---------|---------|
> | BaselineMLP | 203.530 | 86.77% | 23 |
> | DeepMLP | 118.650 | 86.27% | 29 |
> | **SimpleCNN** | **421.642** | **88.83%** | **75** |
> | ResNetMini | 174.970 | 88.48% | 239 |
>
> **Vì sao CNN thắng?** Vấn đề gốc của MLP với ảnh: nó coi **mỗi pixel độc lập**, không thấy cấu trúc 2D, không có tính chất dịch chuyển. CNN trượt filter 3×3 và **chia sẻ trọng số**, nên học được đặc trưng không gian — cạnh, góc, texture.
>
> ResNetMini kém 0.35% nhưng **chậm gấp 3,2×** — không đáng đánh đổi.
>
> Kiến trúc em chọn: **SimpleCNN** — Conv(1→32) → ReLU → MaxPool → Conv(32→64) → ReLU → MaxPool → Flatten → FC(3136→128) → Dropout → FC(128→10).
>
> **Bước 3 — Tinh chỉnh siêu tham số, hai lần thử nghiệm.**
>
> **Lần 1 — Grid search 8 trials × 3 epoch**, quét đủ 2 model × 2 learning rate × 2 optimizer, mất khoảng 19 phút CPU:
>
> | Model | LR | Optim | Val Acc |
> |-------|-----|-------|---------|
> | **simple_cnn** | **0.001** | **adam** | **90.53%** |
> | simple_cnn | 0.0005 | adam | 88.97% |
> | deep_mlp | 0.001 | adam | 87.13% |
> | deep_mlp | 0.0005 | adam | 87.07% |
> | deep_mlp | 0.001 | sgd | 83.98% |
> | deep_mlp | 0.0005 | sgd | 82.25% |
> | simple_cnn | 0.001 | sgd | 80.83% |
> | simple_cnn | 0.0005 | sgd | 78.13% |
>
> Ba kết luận từ bảng:
> - **Adam thắng SGD ở cùng learning rate, chênh 7–10%** — Adam tự chỉnh learning rate từng tham số, SGD thì không.
> - **simple_cnn thắng deep_mlp ở mọi tổ hợp** — CNN khai thác cấu trúc 2D.
> - **SGD lr=0.0005 học rất chậm**, chỉ 78% sau 3 epoch.
>
> **Lần 2 — Random search 12 trials × 3 epoch**, quét cả 4 kiến trúc với siêu tham số ngẫu nhiên.
>
> → mở `results/runs/hyperopt.md`
>
> Trial tốt nhất là **resnet_mini + adam + lr=0.01, đạt 89.15%**.
>
> Em **chốt theo grid search**, vì grid quét **đầy đủ** mọi tổ hợp còn random có thể trượt, và grid **tất định** — cùng config cho cùng kết quả. ResNet điểm thấp hơn trong bảng này không phải do nó tệ, mà vì **3 epoch quá ngắn để hội tụ**.
>
> **Bước 4 — Chốt cấu hình:** SimpleCNN + Adam + lr=0.001 + batch 128 + dropout 0.5.
>
> Em chọn mô hình dựa trên **số liệu validation**, không phải cảm tính. Tập test giữ kín tới cuối."

---

## PHẦN 5 — Huấn luyện · 1m45 · `5:45`

> "Em viết một lớp `Trainer`, mỗi epoch gồm 2 vòng:
>
> **Train loop** — 5 bước: `zero_grad` → **forward** (dự đoán) → **loss** (đo sai số) → **backward** (tính gradient) → **step** (cập nhật trọng số).
>
> **Validate loop** — dùng `model.eval()` và `torch.no_grad()`: tắt dropout, dùng statistics đã tích lũy của BatchNorm, không tính gradient — tiết kiệm khoảng **50% bộ nhớ**.
>
> Loss là **CrossEntropyLoss** — gộp cả Softmax và Negative Log Likelihood vào một hàm, nhận logits thay vì xác suất để ổn định số học. Optimizer là **Adam**.
>
> **Hai callback:**
> - **EarlyStopping** theo `val_loss`, patience = 5. Không cải thiện trong 5 epoch liên tiếp là dừng, vì model đã overfit.
> - **Checkpoint** lưu cả **best model** (theo val_loss thấp nhất, dùng để đánh giá) và **last model** (để resume nếu crash).
>
> Em chạy **15 epoch**:
>
> | Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
> |-------|-----------|----------|----------|---------|
> | 1 | 0.585 | 0.351 | 79.03% | 87.08% |
> | 5 | 0.265 | 0.237 | 90.41% | 91.42% |
> | **11 (best)** | 0.178 | **0.215** | 93.47% | 92.85% |
> | 15 (last) | 0.141 | 0.230 | 94.72% | 92.80% |
>
> → mở `results/runs/loss_curve.png`
>
> Cô thấy ngay: **train loss giảm đều, nhưng val loss đạt đỉnh ở epoch 11 — 0.215 — rồi tăng lại**.
>
> → mở `results/runs/accuracy_curve.png`
>
> Và **train accuracy tăng đều lên 94.72%, còn validation đi ngang quanh 92.8%**.
>
> Đây là dấu hiệu **overfit**, và nó chứng minh **EarlyStopping hoạt động đúng** — dừng đúng lúc tốt nhất thay vì để model học thuộc.
>
> Em lưu theo `val_loss` chứ không phải `val_acc`, vì loss chuẩn hơn: loss phạt cả trường hợp model mơ hồ — đoán sai nhưng vẫn trả xác suất cao."

---

## PHẦN 6 — Đánh giá & phân tích lỗi · 1m45 · `7:30`

> "Đây là phần quan trọng nhất về mặt kỹ thuật: cho biết model **thực sự tốt bao nhiêu, và sai ở đâu**.
>
> Em mở tập test — **lần đầu tiên sau cả dự án** — và tính 4 chỉ số:
>
> | Chỉ số | Giá trị |
> |--------|---------|
> | **Accuracy** | **92.10%** |
> | Precision (macro) | 92.11% |
> | Recall (macro) | 92.10% |
> | F1-Score (macro) | 92.09% |
>
> **92.10% — vượt mục tiêu 90% em đặt ra ban đầu.**
>
> Ba chỉ số còn lại gần như bằng nhau. Nghĩa là vì dataset **cân bằng**, model **không thiên lệch** lớp nào. Với dữ liệu mất cân bằng thì sự chênh này mới đáng lo.
>
> Nhưng em không dừng ở con số tổng. Em làm **ma trận nhầm lẫn** để tìm ra model sai ở đâu:
>
> | Lớp thực | Bị nhầm thành | Số lần |
> |-----------|---------------|--------|
> | Shirt | T-shirt/top | 95 |
> | T-shirt/top | Shirt | 90 |
> | Shirt | Pullover | 64 |
> | Coat | Pullover | 63 |
> | Coat | Shirt | 55 |
>
> → mở `results/runs/confusion_matrix.png`
>
> **Shirt và T-shirt/top nhầm nhau nhiều nhất — 185 lần, gần 20% tổng số lỗi.**
>
> Điều này hợp lý về mặt thị giác: cả hai đều là áo, cùng cổ tròn, cùng tay ngắn, chỉ khác ở chi tiết nhỏ về cổ áo và độ dài — mà ở 28×28 thì gần như không thấy nổi.
>
> → mở `results/runs/predictions.png`
>
> Em vẽ thêm **lưới ảnh dự đoán** — xanh là đúng, đỏ là sai. Phần đỏ tập trung đúng vào các cặp áo, khớp với phân tích trên.
>
> Cuối cùng em lưu **final model kèm metrics và config**, để người khác mở ra là biết ngay model đạt bao nhiêu, dùng siêu tham số gì, và **tái tạo lại được kết quả**."

---

## PHẦN 7 — Kết luận · 45s · `9:15`

> "**Kết quả:** Test accuracy **92.10%**, precision/recall/F1 đều trên 92% — **vượt mục tiêu 90%** đặt ra trước khi bắt đầu.
>
> **Về mặt kỹ thuật**, em triển khai trọn chuỗi: EDA → làm sạch → chuẩn hóa → chia 3 tập stratified → 4 kiến trúc qua factory → grid + random search → training loop với early stopping và checkpoint → đánh giá 4 chỉ số → confusion matrix. **23 unit test phần training đều pass**.
>
> **Điều em học được lớn nhất** không phải thuật toán cụ thể, mà là **tư duy thực nghiệm**: mọi quyết định về kiến trúc và siêu tham số đều đến từ **số liệu đo được**, không phải cảm tính. Nhờ đó em biết chính xác model giỏi ở đâu và yếu ở đâu.
>
> **Hướng tiếp:** thêm Data Augmentation, thử ResNet18, thêm learning rate scheduler — em đã ghi kế hoạch trong tài liệu phương pháp.
>
> Cảm ơn cô!"

---

## TIMELINE

| Phần | Nội dung | Từ – đến |
|------|----------|-----------|
| 1 | Mở đầu | 0:00 – 0:45 |
| 2 | Bài toán & mục tiêu | 0:45 – 1:45 |
| 3 | Chuẩn bị dữ liệu | 1:45 – 3:15 |
| 4 | Mô hình & siêu tham số | 3:15 – 5:45 |
| 5 | Huấn luyện | 5:45 – 7:30 |
| 6 | Đánh giá & phân tích lỗi | 7:30 – 9:15 |
| 7 | Kết luận | 9:15 – 10:00 |

---

## 12 CON SỐ PHẢI NHỚ

| Con số | Ý nghĩa |
|--------|----------|
| **92.10%** | Test accuracy — kết quả cuối |
| **92.11 / 92.10 / 92.09%** | Precision / Recall / F1 (macro) |
| **50.000 / 10.000 / 10.000** | Train / Val / Test |
| **0.286 / 0.353** | Mean / Std của pixel |
| **90.53%** | Best val acc — Grid 8 trials |
| **89.15%** | Best val acc — Random 12 trials |
| **75 s/epoch** | Tốc độ train SimpleCNN (CPU) |
| **0.215** | Val loss thấp nhất @ epoch 11 |
| **185** | Tổng nhầm Shirt ↔ T-shirt |
| **23** | Số unit test (`tests/training/`) |
| **421.642** | Params SimpleCNN |
| **239 / 75** | s/epoch ResNetMini / SimpleCNN = 3,2× chậm hơn |

---

## BẪY HAY GẶP — ĐỪNG NÓI NHẦM

| Đừng nói | Nói thay |
|----------|---------|
| "27 unit test" | **23 unit test** |
| "Train 60.000 ảnh" | **50.000** train, 10.000 val, 10.000 test |
| "khoảng 60 s/epoch" | **75 s/epoch** |
| "Grid 8 trials có ResNet" | Grid chỉ **simple_cnn + deep_mlp** |
| "Best trial là ResNet" | Đó là **random 3 epoch**; em chốt theo **grid = simple_cnn 90.53%** |

---

## NẾU CÔ YÊU CẦU DEMO

```powershell
cd e:\lab1
$env:PYTHONPATH = "."
python main.py                          # Train 15 epoch + sinh đồ thị
python src/evaluation/run_evalution.py   # Đánh giá test + confusion matrix
pytest tests/training -v                # 23 unit test
```

---

*Tạo: 2026-10-21 · Bản rút gọn: `PRESENTATION_SCRIPT.md`*
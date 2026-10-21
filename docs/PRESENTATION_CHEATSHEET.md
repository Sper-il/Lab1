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

> "Bốn bước, mỗi bước một quyết định có căn cứ từ số liệu.
>
> **Bước 1 — Baseline.** Em dựng MLP 1 tầng ẩn đúng theo `PHASE2_PLAN.md`: **Flatten 784 → Linear(784→256) → ReLU → Dropout(0.2) → Linear(256→10)** — **203.530 tham số**, train lr=1e-3 đạt val_acc **86.77%** trong 2 epoch. Đây là **mức sàn** — cao hơn 10% ngẫu nhiên nên pipeline chạy đúng.
>
> Hai ý nghĩa khi dựng baseline:
> 1. **Sanity check pipeline.** Nếu baseline không học được (>10%) thì hỏng ở data, loss hoặc optimizer — không phải kiến trúc. Đạt 86.77% chứng minh data, loss, optimizer và vòng train đều đúng.
> 2. **Mức sàn để so sánh.** Mọi kiến trúc phức tạp hơn phải **vượt** 86.77% mới đáng giữ. ResNet 88.48% — cao hơn baseline 1.71%, tức kiến trúc mới có giá trị. Nếu chỉ đạt 87.0% thì 174.970 tham số và 239s/epoch đều lãng phí.
>
> **Bước 2 — So sánh 4 kiến trúc**
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
> **Vì sao CNN thắng?** MLP coi **mỗi pixel độc lập**, không thấy cấu trúc 2D. CNN trượt filter 3×3 và **chia sẻ trọng số**, nên học được đặc trưng không gian — cạnh, góc, texture.
>
> **Em chốt SimpleCNN** vì nó **đẹp nhất về hiệu năng trên mỗi giây**: 88.83% trong 75s/epoch, nhanh gấp 3,2× ResNetMini mà còn chính xác hơn. ResNet cần tập dữ liệu lớn hơn nhiều (ImageNet 1.2M ảnh) mới phát huy, còn ở 60k ảnh thì skip connection chỉ làm chậm chứ không thêm giá trị.
>
> Cấu trúc: Conv(1→32) → ReLU → MaxPool → Conv(32→64) → ReLU → MaxPool → Flatten → Linear(3136→128) → ReLU → Dropout(0.5) → Linear(128→10).
>
> **Bước 3 — Tinh chỉnh siêu tham số, hai lần thử nghiệm.**
>
> **Lần 1 — Grid search 8 trials × 3 epoch**, quét 2 model × 2 learning rate × 2 optimizer, ~19 phút CPU. Grid **tất định** và **phủ kín**.
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
> Ba kết luận: **Adam thắng SGD, chênh 7–10%** ở cùng learning rate — Adam duy trì **một learning rate riêng cho từng tham số** nên tự giảm tốc khi gradient dao động mạnh, còn SGD dễ lạc khỏi minimum trên loss surface gồ ghề của CNN. **`simple_cnn` thắng `deep_mlp` ở *mọi* tổ hợp** — nhắc lại lý do cấu trúc 2D. Và **SGD học chậm**: lr=0.0005 chỉ 78–82% sau 3 epoch.
>
> **Lần 2 — Random search 12 trials × 3 epoch**, quét cả **4 kiến trúc** — cho ResNet cơ hội cạnh tranh công bằng.
>
> → mở `results/runs/hyperopt.md`
>
> Trial tốt nhất **resnet_mini + adam + lr=0.01 = 89.15%**. Em vẫn **chốt theo grid**, vì grid **phủ đầy đủ** còn random **có thể trượt** ô tốt, và grid **tất định** — quan trọng cho reproducibility. ResNet thua 1.4% vì **3 epoch quá ngắn**.
>
> **Bước 4 — Chốt cấu hình:** SimpleCNN + Adam + lr=0.001 + batch 128 + dropout 0.5 + 15 epoch + CrossEntropyLoss.
>
> Mọi quyết định em chốt dựa trên **số liệu validation**, không phải cảm tính. Tập test giữ kín tới Phần 6."

---

## PHẦN 5 — Huấn luyện · 1m45 · `5:45`

> "Giờ em train với cấu hình đã chốt.
>
> **Cấu trúc code.** Em viết lớp `Trainer`, mỗi epoch gồm 2 vòng. **Vòng train** là 5 bước chuẩn PyTorch: `zero_grad` → **forward** (ra logits B×10) → **loss** → **backward** → **step** (cập nhật trọng số theo Adam).
>
> **Vòng validate** khác 3 điểm: `model.eval()` tắt dropout, dùng **statistics tích lũy** của BatchNorm — quên thì validation sai hoàn toàn; `torch.no_grad()` không tính gradient, **tiết kiệm ~50% bộ nhớ**; model **không học** trên validation.
>
> Loss là **CrossEntropyLoss** — gộp cả Softmax và Negative Log Likelihood vào một hàm, nhận thẳng **logits** thay vì xác suất để ổn định số học.
>
> **Hai callback:** *EarlyStopping* theo `val_loss`, patience = 5, min_delta = 0.0001 — không giảm thêm 0.0001 trong 5 epoch liên tiếp thì dừng vì model đã **overfit**. *Checkpoint* lưu `best_model.pt` + `last_model.pt` kèm weights, optimizer state và metrics.
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
> **Train loss giảm đều không ngừng**, còn **val loss đạt đỉnh ở epoch 11 — 0.215 — rồi tăng lại**. Train chạm đáy 0.141 ở epoch 15 trong khi val loss tăng — hình dạng kinh điển của **overfitting**.
>
> → mở `results/runs/accuracy_curve.png`
>
> **Train accuracy tăng đều lên 94.72%, còn validation đi ngang quanh 92.8%** từ epoch 13. Khoảng cách ~2% là model học thuộc.
>
> Ba điều em muốn nói rõ:
> 1. **EarlyStopping hoạt động đúng.** Sau epoch 11 chỉ có 4 epoch không cải thiện nên nó **chưa kích hoạt** — dừng sớm thì em mất 4 epoch cải thiện nhỏ.
> 2. **Em lưu theo `val_loss` chứ không phải `val_acc`.** Val_acc cao nhất là epoch 13 (92.88%), cao hơn epoch 11 (92.85%) chỉ 0.03% — không có ý nghĩa. Chọn loss chuẩn hơn vì **loss phạt cả trường hợp model "mơ hồ"**, còn accuracy thì không.
> 3. **Overfit ở mức nhẹ** — val accuracy chỉ giảm 0.05% (92.85% → 92.80%).

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
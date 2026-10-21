# KỊCH BẢN TRÌNH BÀY — FashionMNIST (10 phút)

**Người trình bày:** Sper-il

**Cách dùng:** Mở Preview (`Ctrl+Shift+V`), đọc từ trên xuống. Chỗ ghi `→ mở:` là tên file cần mở ra.

> Bản rút gọn: `PRESENTATION_SCRIPT.md` — cùng thứ tự, dùng để luyện.

---

## MỞ TRƯỚC 11 TAB

| # | Mở file này | Dùng ở phần | Chỉ vào |
|---|-------------|-------------|--------|
| 1 | `docs/PRESENTATION_CHEATSHEET.md` | Xuyên suốt | Lời thoại |
| 2 | `results/runs/dataset_info.md` | 3 | Bảng 10 lớp + thông số |
| 3 | `results/runs/per_class_samples.png` | 3 | Lưới 10 ảnh mẫu |
| 4 | `results/runs/class_distribution.png` | 4 | 10 cột cao đều |
| 5 | `results/runs/pixel_histogram.png` | 4 | Hai đỉnh pixel |
| 6 | `results/runs/architecture_selection.md` | 5 | Bảng 4 kiến trúc |
| 7 | `results/runs/hyperopt.md` | 5 | Bảng random 12 trials |
| 8 | `results/runs/loss_curve.png` | 6 | Vùng epoch 11–13 |
| 9 | `results/runs/accuracy_curve.png` | 6 | Train ↑, val ngang |
| 10 | `results/runs/confusion_matrix.png` | 7 | Hàng Shirt × cột T-shirt |
| 11 | `results/runs/predictions.png` | 7 | Xanh = đúng, đỏ = sai |

Dự phòng: `docs/reports/nguoi1_models_report.md` §2.2 (bảng grid 8 trials) · `docs/reports/nguoi3_evaluation_report.md` (metrics test)

---

## PHẦN 1 — Mở đầu · 45s · `0:00`

> "Em chào cô. Em tên là Sper-il. Hôm nay em trình bày đồ án **phân loại ảnh thời trang bằng mạng nơ-ron tích chập trên FashionMNIST**.
>
> Em chọn bài này không phải vì dễ, mà vì nó chạy được **trọn vòng đời một dự án Machine Learning thật**: khám phá dữ liệu → xây mô hình → tinh chỉnh siêu tham số → huấn luyện → đánh giá → phân tích lỗi. Tất cả chạy bằng code, không làm thủ công.
>
> Source em đẩy lên GitHub công khai, cô kiểm tra bất kỳ lúc nào."

---

## PHẦN 2 — Pipeline tổng quan · 1m00 · `0:45`

> "Trước khi vào từng bước, em xin trình bày **bản đồ của cả dự án** — pipeline chạy từ dữ liệu thô ra tới kết quả cuối.
>
> Em thiết kế theo nguyên tắc **mỗi tầng một trách nhiệm**, các tầng thay thế độc lập:
>
```
 FashionMNIST thô (70.000 ảnh)
         │
         ▼
 ┌───────────────┐  src/data/eda.py — mean/std, phân bố lớp
 │ EDA + Làm sạch│  60.000 ảnh hợp lệ 100%
 └───────┬───────┘
         ▼
 ┌───────────────┐  src/data/split.py — stratified, seed 42
 │   Chia 3 tập  │  50.000 train / 10.000 val / 10.000 test
 └───────┬───────┘
         ▼
 ┌───────────────┐  src/models/model_factory.py — registry
 │  ModelFactory │  4 kiến trúc → chốt SimpleCNN
 └───────┬───────┘
         ▼
 ┌───────────────┐  src/training/trainer.py + callbacks.py
 │    Trainer    │  Adam · CrossEntropy · EarlyStopping · Checkpoint
 └───────┬───────┘
         ▼
 ┌───────────────┐  src/evaluation/ — 4 chỉ số, confusion matrix
 │   Đánh giá    │  test accuracy 92.10%
 └───────┬───────┘
         ▼
   Artifacts: results/runs/ + models/checkpoints/
```

> **Điểm vào là `main.py`** — đọc `configs/model_config.yaml` rồi gọi `run_training.py`, nơi **ghép tất cả lại**: set seed, chia dữ liệu, dựng model qua factory, tạo loss và optimizer, rồi đưa vào `Trainer`.
>
> `Trainer` là **trái tim** — chỉ lo vòng train/validate và callback, **không biết** model nào, dữ liệu ở đâu. Nhờ vậy thêm kiến trúc mới đều **không phải sửa `Trainer`**.
>
> Cuối cùng mọi thứ được **lưu thành artifact** — checkpoint, history, đồ thị — để người khác mở ra là biết ngay kết quả và **tái tạo lại được**.

---

## PHẦN 3 — Bài toán & mục tiêu · 40s · `1:45`

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

## PHẦN 4 — Chuẩn bị dữ liệu · 1m10 · `2:25`

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
> **3. Chuẩn hóa.** Áp dụng `(x − 0.286) / 0.353`, đúng mean và std ở bước 1. Cần vì pixel lệch giá trị làm gradient nghiêng, model hội tụ chậm hơn.
>
> **4. Chia dữ liệu — 3 tập, không phải 2:**

| Tập | Số ảnh | Vai trò |
|-----|--------|----------|
| Train | 50.000 | Model học trọng số |
| Validation | 10.000 | Đánh giá khi train, quyết định dừng |
| Test | 10.000 | Giữ kín, mở 1 lần ở cuối |

> Nếu chỉ chia 2 tập, model có thể học thuộc cả test → điểm đẹp hơn thực tế, tức là **đánh giá sai**.
>
> Hai kỹ thuật áp dụng: **stratified split** để validation cân bằng giữa 10 lớp, và **seed 42** để chạy lại ra kết quả y hệt."

---

## PHẦN 5 — Mô hình & siêu tham số · 2m20 · `3:35`

> "Bốn bước, mỗi bước một quyết định có căn cứ từ số liệu.
>
> **Bước 1 — Baseline.** Em dựng MLP 1 tầng ẩn: **Flatten 784 → Linear(784→256) → ReLU → Dropout(0.2) → Linear(256→10)** — **203.530 tham số**, đạt val_acc **86.77%** trong 2 epoch. Đây là **mức sàn** — cao hơn 10% ngẫu nhiên nên pipeline chạy đúng.
>
> Hai ý nghĩa khi dựng baseline:
> 1. **Sanity check.** Nếu baseline không học được (>10%) thì hỏng ở data, loss hoặc optimizer — không phải kiến trúc. Đạt 86.77% chứng minh cả chuỗi đúng.
> 2. **Mức sàn để so sánh.** Kiến trúc phức tạp hơn phải **vượt** 86.77% mới đáng giữ. ResNet 88.48% — cao hơn 1.71%, tức kiến trúc mới có giá trị.
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
> Em chốt **SimpleCNN** vì nó **cao nhất trong 4 kiến trúc** (88.83%) mà lại **nhanh gấp 3,2× so với ResNet** — 75s/epoch. Đổi lấy 0.35%, ResNet tốn thêm 164s mỗi epoch, không đáng.
>
> **Bước 3 — Tinh chỉnh siêu tham số.**
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
> Ba kết luận: **Adam thắng SGD, chênh 7–10%** ở cùng learning rate — Adam tự điều chỉnh bước đi theo độ lớn gradient từng tham số nên ổn định hơn trên loss surface gồ ghề của CNN. **`simple_cnn` thắng `deep_mlp` ở *mọi* tổ hợp**. Và **SGD học chậm**: lr=0.0005 chỉ 78–82% sau 3 epoch.
>
> **Lần 2 — Random search 12 trials × 3 epoch**, quét cả **4 kiến trúc** — cho ResNet cơ hội cạnh tranh.
>
> → mở `results/runs/hyperopt.md`
>
> Trial tốt nhất **resnet_mini + adam + lr=0.01 = 89.15%**. Em vẫn **chốt theo grid**, vì grid **phủ đầy đủ** còn random **có thể trượt** ô tốt, và grid **tất định** — quan trọng cho reproducibility. ResNet thua 1.4% vì **3 epoch quá ngắn**.
>
> **Bước 4 — Chốt cấu hình:** SimpleCNN + Adam + lr=0.001 + batch 128 + dropout 0.5 + 15 epoch + CrossEntropyLoss.
>
> Mọi quyết định em chốt dựa trên **số liệu validation**, không phải cảm tính. Tập test giữ kín tới Phần 7."

---

## PHẦN 6 — Huấn luyện · 1m40 · `5:55`

> "Giờ em train với cấu hình đã chốt.
>
> **Cấu trúc code.** Mỗi epoch gồm 2 vòng. **Vòng train** là 5 bước chuẩn PyTorch: `zero_grad` → **forward** (ra logits B×10) → **loss** → **backward** → **step** (cập nhật trọng số theo Adam).
>
> **Vòng validate** khác 3 điểm: `model.eval()` tắt dropout, dùng **statistics tích lũy** của BatchNorm — quên thì validation sai hoàn toàn; `torch.no_grad()` không tính gradient, **tiết kiệm ~50% bộ nhớ**; model **không học** trên validation.
>
> Loss là **CrossEntropyLoss** — gộp Softmax và Negative Log Likelihood, nhận thẳng **logits** thay vì xác suất để ổn định số học.
>
> **Hai callback:** *EarlyStopping* theo `val_loss`, patience = 5, min_delta = 0.0001 — không giảm thêm 0.0001 trong 5 epoch liên tiếp thì dừng vì model đã **overfit**. *Checkpoint* lưu `best_model.pt` + `last_model.pt` kèm weights và metrics.
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
> 1. **EarlyStopping hoạt động đúng.** Sau epoch 11 chỉ có 4 epoch không cải thiện nên nó **chưa kích hoạt** — dừng sớm thì mất 4 epoch cải thiện nhỏ.
> 2. **Em lưu theo `val_loss` chứ không phải `val_acc`.** Val_acc cao nhất là epoch 13 (92.88%), cao hơn epoch 11 (92.85%) chỉ 0.03% — không có ý nghĩa. Chọn loss chuẩn hơn vì **loss phạt cả trường hợp model "mơ hồ"**, còn accuracy thì không.
> 3. **Overfit ở mức nhẹ** — val accuracy chỉ giảm 0.05% (92.85% → 92.80%).

---

## PHẦN 7 — Đánh giá & phân tích lỗi · 1m35 · `7:35`

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
> Ba chỉ số còn lại gần như bằng nhau — vì dataset **cân bằng**, model **không thiên lệch** lớp nào. Với dữ liệu mất cân bằng thì sự chênh này mới đáng lo.
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
> Hợp lý về mặt thị giác: cả hai đều là áo, cùng cổ tròn, cùng tay ngắn, chỉ khác ở chi tiết nhỏ về cổ áo và độ dài — mà ở 28×28 thì gần như không thấy nổi.
>
> → mở `results/runs/predictions.png`
>
> Em vẽ thêm **lưới ảnh dự đoán** — xanh là đúng, đỏ là sai. Phần đỏ tập trung đúng vào các cặp áo, khớp với phân tích trên.
>
> Cuối cùng em lưu **final model kèm metrics và config**, để người khác mở ra là biết ngay model đạt bao nhiêu, dùng siêu tham số gì, và **tái tạo lại được**."

---

## PHẦN 8 — Kết luận · 40s · `9:10`

> "**Kết quả:** Test accuracy **92.10%**, precision/recall/F1 đều trên 92% — **vượt mục tiêu 90%** đặt ra trước khi bắt đầu.
>
> **Về mặt kỹ thuật**, em triển khai trọn chuỗi: EDA → chuẩn hóa → chia 3 tập stratified → 4 kiến trúc qua factory → grid + random search → training loop có early stopping và checkpoint → đánh giá 4 chỉ số → confusion matrix. **23 unit test đều pass**.
>
> **Điều em học được lớn nhất** không phải thuật toán cụ thể, mà là **tư duy thực nghiệm**: mọi quyết định về kiến trúc và siêu tham số đều đến từ **số liệu đo được**, không phải cảm tính.
>
> **Hướng tiếp:** thêm Data Augmentation, thử ResNet18, thêm learning rate scheduler — em đã ghi trong tài liệu phương pháp.
>
> Cảm ơn cô!"

---

## TIMELINE

| Phần | Nội dung | Từ – đến |
|------|----------|-----------|
| 1 | Mở đầu | 0:00 – 0:45 |
| 2 | Pipeline tổng quan | 0:45 – 1:45 |
| 3 | Bài toán & mục tiêu | 1:45 – 2:25 |
| 4 | Chuẩn bị dữ liệu | 2:25 – 3:35 |
| 5 | Mô hình & siêu tham số | 3:35 – 5:55 |
| 6 | Huấn luyện | 5:55 – 7:35 |
| 7 | Đánh giá & phân tích lỗi | 7:35 – 9:10 |
| 8 | Kết luận | 9:10 – 9:50 |

---

## 13 CON SỐ PHẢI NHỚ

| Con số | Ý nghĩa |
|--------|----------|
| **92.10%** | Test accuracy — kết quả cuối |
| **92.11 / 92.10 / 92.09%** | Precision / Recall / F1 (macro) |
| **50.000 / 10.000 / 10.000** | Train / Val / Test |
| **0.286 / 0.353** | Mean / Std của pixel |
| **86.77%** | BaselineMLP — mức sàn để so sánh |
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
| "Baseline là 2 tầng ẩn" | BaselineMLP là **1 tầng ẩn 256 + Dropout 0.2**, 203.530 params |
| "Chạy `run_training.py`" | Lệnh là **`python main.py`** — `run_training.py` là module được import |

---

## NẾU CÔ YÊU CẦU DEMO

```powershell
cd e:\lab1
$env:PYTHONPATH = "."
python main.py                          # Train 15 epoch + sinh đồ thị
python src/evaluation/run_evalution.py   # Đánh giá test + confusion matrix
pytest tests/training -v                # 23 unit test
```
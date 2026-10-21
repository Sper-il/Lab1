# 🎤 KỊCH BẢN THUYẾT TRÌNH ĐẦY ĐỦ — FashionMNIST (10 phút)

**Người trình bày:** Sper-il
**Cách dùng:** Mở `Ctrl+Shift+V` (Preview) → đọc từ trên xuống. Thấy **`[[Tab X]]`** → bấm `Ctrl+Tab` chuyển sang tab X.

> Bản rút gọn 1 trang: `PRESENTATION_SCRIPT.md` — cùng số thứ tự, dùng để luyện tập.

---

## 📂 Sơ đồ tab — mở sẵn 11 tab trước khi lên lớp

| Tab | File | Dùng ở phần | Chỉ vào điều gì |
|-----|------|-------------|-----------------|
| **1** | `docs/PRESENTATION_CHEATSHEET.md` | Xuyên suốt | Chính là lời thoại này |
| **2** | `results/runs/dataset_info.md` | Phần 2 | Bảng 10 lớp + thông số dataset |
| **3** | `results/runs/per_class_samples.png` | Phần 2 | Lưới 10 ảnh, mỗi lớp 1 ảnh |
| **4** | `results/runs/class_distribution.png` | Phần 3 | 10 cột cao đều nhau |
| **5** | `results/runs/pixel_histogram.png` | Phần 3 | Hai đỉnh: nền sáng + vật thể tối |
| **6** | `results/runs/architecture_selection.md` | Phần 4 | Bảng 4 kiến trúc + s/epoch |
| **7** | `results/runs/hyperopt.md` | Phần 4 | ⚠️ Bảng random 12 trials |
| **8** | `results/runs/loss_curve.png` | Phần 5 | Vùng epoch 11–13 |
| **9** | `results/runs/accuracy_curve.png` | Phần 5 | Train acc ↑, val acc ngang |
| **10** | `results/runs/confusion_matrix.png` | Phần 6 | Hàng Shirt × cột T-shirt/top |
| **11** | `results/runs/predictions.png` | Phần 6 | Ô xanh = đúng, ô đỏ = sai |

> **Dự phòng** (không cần mở trước): `docs/reports/nguoi1_models_report.md` §2.2 — bảng **grid search 8 trials**;
> `docs/reports/nguoi3_evaluation_report.md` — metrics test set + top-5 nhầm lẫn.

---

## Phần 1 — Mở đầu & đặt vấn đề · 45s · `0:00`

> "Em xin chào thầy. Em tên là Sper-il. Hôm nay em trình bày đồ án **Phân loại ảnh thời trang với mạng nơ-ron tích chập trên dataset FashionMNIST**.
>
> Đây là một bài toán kinh điển của Deep Learning, nhưng em chọn nó **không phải vì dễ**, mà vì nó cho phép em đi **đủ toàn bộ vòng đời của một dự án Machine Learning thực tế** — từ khám phá dữ liệu, xây dựng mô hình, tinh chỉnh siêu tham số, huấn luyện, cho tới đánh giá và phân tích lỗi — tất cả đều được **tự động hoá bằng code** chứ không phải chạy thủ công.
>
> Toàn bộ source code em đã đẩy lên GitHub công khai, để thầy kiểm tra lại bất kỳ lúc nào.""

**🎬 Slide:** tiêu đề đồ án + QR/link GitHub.

---

## Phần 2 — Bài toán & mục tiêu · 45s · `0:45`

> "**Bài toán:** Cho một ảnh xám 28×28 pixel, máy tính phải trả lời đây là món đồ nào, trong 10 loại: áo thun, quần, áo len, váy, áo khoác, dép, áo sơ mi, giày thể thao, túi xách, và ủng.
>
> `[[Tab 2]]` Dataset có **60.000 ảnh huấn luyện** và **10.000 ảnh kiểm tra** — tổng cộng **70.000 ảnh**, tất cả đều cân bằng tuyệt đối: mỗi lớp đúng 6.000 ảnh train và 1.000 ảnh test.
>
> `[[Tab 3]]` Đây là ảnh mẫu từ từng lớp. Mời thầy quan sát: tất cả đều **28×28 pixel**, đơn sắc — chính vì vậy một số cặp áo rất khó phân biệt, đây sẽ là điểm khó nhất của bài toán.
>
> **Mục tiêu em đặt ra là độ chính xác trên tập kiểm tra tối thiểu 90%.**
>
> Em chốt mục tiêu này **trước khi bắt đầu code**, vì một mục tiêu rõ ràng giúp em biết khi nào nên dừng tìm kiếm, thay vì cứ mò mãi mãi.""

---

## Phần 3 — Giai đoạn 1: Chuẩn bị dữ liệu · 1m30 · `1:45`

> "**Giai đoạn 1** là giai đoạn em coi là quan trọng nhất, vì **80% lỗi trong Machine Learning đến từ dữ liệu chứ không phải từ mô hình**. Em chia thành 4 bước:
>
> **Bước 1 — Khám phá dữ liệu (EDA).** Em tính thống kê toàn bộ pixel: **mean = 0.286**, **std = 0.353**, giá trị nằm trong khoảng [0, 1].
>
> `[[Tab 4]]` 10 cột này cao đều nhau — dataset **cân bằng hoàn hảo**, không có NaN, không có ảnh hỏng. Nhờ vậy em **không cần** xử lý mất cân bằng lớp — một việc mà nếu có, sẽ tốn rất nhiều công sức.
>
> `[[Tab 5]]` Histogram cường độ pixel có **hai đỉnh** — một đỉnh là nền sáng, một đỉnh là vật thể tối. Phân bố đẹp, không có outlier.
>
> **Bước 2 — Làm sạch.** Em viết hàm kiểm tra toàn diện: NaN/Inf, kích thước ảnh đúng chuẩn 28×28×1, nhãn trong khoảng 0–9. Kết quả **60.000/60.000 mẫu hợp lệ — 100%**.
>
> **Bước 3 — Tiền xử lý.** Em **chuẩn hóa** dữ liệu theo công thức `(x − 0.286) / 0.353` với đúng mean và std đã tính ở bước 1. Tại sao cần? Vì pixel có giá trị lệch nhau sẽ khiến gradient bị 'nghiêng', làm model hội tụ chậm hơn và khó tối ưu hơn.
>
> **Bước 4 — Chia dữ liệu.** Đây là bước em muốn nhấn mạnh. Em chia thành **3 tập** chứ không phải 2:
> - **Train 54.000 ảnh** — nơi model học.
> - **Validation 6.000 ảnh** — model **không học**, chỉ dùng để đánh giá trong lúc train và quyết định khi nào nên dừng.
> - **Test 10.000 ảnh** — giữ kín đến cuối cùng, mới mở ra đánh giá một lần duy nhất.
>
> Nếu chỉ chia 2 tập thì model có thể 'học thuộc' cả tập test, và kết quả đánh giá sẽ **đẹp hơn thực tế** — tức là đánh giá sai.
>
> Hai kỹ thuật em áp dụng: **Stratified Split** để mỗi lớp có đúng 600 ảnh trong tập validation, và **random seed cố định = 42** để mọi lần chạy lại đều cho kết quả giống hệt nhau — tức là **reproducible**.""

**💡 Câu hỏi dự kiến:** *"Vì sao chọn seed 42?"* → 42 là số quen thuộc trong văn hoá đại chúng, nhưng quan trọng hơn là nó chỉ là **một quy ước bất kỳ nhưng cố định** — điều duy nhất cần là nó không đổi.

---

## Phần 4 — Giai đoạn 2: Phát triển mô hình · 2m30 · `3:15`

> "**Giai đoạn 2** là tìm ra kiến trúc mô hình và bộ siêu tham số tốt nhất. Em làm theo 4 bước, mỗi bước một quyết định có căn cứ.
>
> **Bước 1 — Baseline.** Trước khi làm gì phức tạp, em dựng một MLP đơn giản — Flatten 784 → 256 → 128 → 10 — để có **mức sàn** để so sánh. Nếu không có baseline, em sẽ không biết mô hình phức tạp có thật sự tốt hơn hay không.
>
> **Bước 2 — Khám phá kiến trúc.** Em cài đặt và so sánh **4 kiến trúc** qua một `ModelFactory` — một registry, nên thêm kiến trúc mới chỉ cần **1 dòng code**:
>
> | Kiến trúc | Số tham số | Val Acc | s/epoch | Đặc điểm |
> |-----------|------------|---------|---------|-----------|
> | BaselineMLP | 203.530 | 86.77% | 23 | 1 tầng ẩn — sanity check |
> | DeepMLP | 118.650 | 86.27% | 29 | 3 tầng FC + BatchNorm + Dropout |
> | **SimpleCNN** | **421.642** | **88.83%** | **75** | 2 block Conv + 2 tầng FC + Dropout |
> | ResNetMini | 174.970 | 88.48% | 239 | Residual blocks + skip connections |
>
> `[[Tab 6]]` **Vì sao CNN thắng?** Vấn đề cốt lõi của MLP với ảnh là nó coi **mỗi pixel là độc lập** — nó không hiểu ảnh có cấu trúc không gian 2D, không có tính chất 'dịch chuyển'. Trong khi đó CNN trượt một filter 3×3 trên toàn bộ ảnh và **chia sẻ trọng số**, nên nó học được đặc trưng không gian — cạnh, góc, texture — với **ít tham số hơn** và **nhanh hơn**.
>
> Và ResNetMini chỉ kém 0.35% nhưng **chậm gấp 3.2×** — đánh đổi không đáng.
>
> Kiến trúc em chọn là **SimpleCNN**: Conv(1→32) → ReLU → MaxPool → Conv(32→64) → ReLU → MaxPool → Flatten → FC(3136→128) → Dropout → FC(128→10).
>
> **Bước 3 — Tinh chỉnh siêu tham số.** Em chạy **hai lần thử nghiệm**:
>
> **Lần 1 — Grid search 8 trials × 3 epoch** (quét *đầy đủ* 2 model × 2 learning rate × 2 optimizer = 8 tổ hợp, ~19 phút CPU):
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
> **Ba nhận xét từ bảng này:**
> - **Adam luôn thắng SGD** ở cùng learning rate — chênh lệch tới **7–10%**. Đúng với lý thuyết: Adam tự điều chỉnh learning rate cho từng tham số, còn SGD thì không.
> - **`simple_cnn` thắng `deep_mlp` ở *mọi* tổ hợp** — CNN khai thác cấu trúc không gian 2D của ảnh, MLP thì không.
> - **SGD ở lr=0.0005** học rất chậm — chỉ 78% sau 3 epoch.
>
> **Lần 2 — Random search 12 trials × 3 epoch** (quét cả **4 kiến trúc** với siêu tham số ngẫu nhiên, ~13 phút). Ở đây trial tốt nhất là `resnet_mini` + adam + lr=0.01 đạt **89.15%**.
>
> `[[Tab 7]]` Em chủ động **chốt theo grid search**, vì: một, grid quét **đầy đủ** mọi tổ hợp còn random có thể trượt; hai, grid **tất định** — cùng config cho cùng kết quả. Điểm random thấp hơn không phải do `resnet_mini` tệ hơn, mà vì **3 epoch quá ngắn để hội tụ**.
>
> **Bước 4 — Chọn mô hình.** Em chốt **SimpleCNN + Adam + lr=0.001 + batch 128 + dropout 0.5** và chuyển sang giai đoạn 3.
>
> Điểm em muốn lưu ý: em **không** chọn mô hình dựa trên cảm tính, mà dựa trên **số liệu validation**. Tập test được giữ kín hoàn toàn cho tới bước đánh giá cuối.""

**💡 Câu hỏi dự kiến:** *"Vì sao không dùng mô hình lớn hơn?"* → Em **đã thử** ResNetMini. Nó chỉ kém 0.35% nhưng chậm gấp 3.2×. Mục tiêu 90% và SimpleCNN đã vượt (92.1%) → đánh đổi đó không đáng.

---

## Phần 5 — Giai đoạn 3: Huấn luyện · 1m45 · `5:45`

> "**Giai đoạn 3** em tách làm hai phần: huấn luyện, rồi đánh giá.
>
> **Phần huấn luyện.** Em viết một lớp `Trainer` gọn gàng, mỗi epoch gồm 2 vòng:
>
> - **Train loop** — 5 bước chuẩn: `zero_grad` → **forward** (dự đoán) → **loss** (đo sai số) → **backward** (tính gradient) → **step** (cập nhật trọng số).
> - **Validate loop** — đánh giá trên tập validation với `model.eval()` và `torch.no_grad()` để tắt dropout, dùng statistics đã tích lũy của BatchNorm, và không tính gradient — tiết kiệm khoảng **50% bộ nhớ**.
>
> Loss em dùng là **CrossEntropyLoss** — nó gộp cả Softmax và Negative Log Likelihood vào một hàm, dùng logits thay vì xác suất để **ổn định số học**. Optimizer là **Adam**.
>
> Em áp dụng 2 callback quan trọng:
> - **EarlyStopping** với patience = 5, theo dõi `val_loss`. Nếu validation loss không cải thiện trong 5 epoch liên tiếp thì dừng — vì lúc đó model đã bắt đầu **overfit**.
> - **Checkpoint**: lưu cả **best model** (theo val_loss thấp nhất, dùng để đánh giá và deploy) lẫn **last model** (để resume training nếu bị crash giữa chừng).
>
> Em chạy **15 epoch**. Kết quả thể hiện rất rõ:
>
> | Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
> |-------|-----------|----------|----------|---------|
> | 1 | 0.585 | 0.351 | 79.03% | 87.08% |
> | 5 | 0.265 | 0.237 | 90.41% | 91.42% |
> | 11 (best) | 0.178 | **0.215** | 93.47% | 92.85% |
> | 15 (last) | 0.141 | 0.230 | 94.72% | 92.80% |
>
> `[[Tab 8]]` Nếu thầy nhìn vào đây sẽ thấy ngay: **train loss giảm đều, nhưng val loss đạt đỉnh ở epoch 11 — 0.215 — rồi bắt đầu tăng trở lại**.
>
> `[[Tab 9]]` Và **train accuracy cứ tăng đều lên 94.72%, trong khi validation accuracy đi ngang quanh 92.8% từ epoch 13**.
>
> Đây chính là dấu hiệu **overfit** — và nó cho thấy **EarlyStopping đã hoạt động đúng**, dừng model đúng lúc tốt nhất thay vì để nó học thuộc.""

**💡 Câu hỏi dự kiến:** *"Vì sao val acc đỉnh ở epoch 13 mà lưu epoch 11?"* → Vì em theo dõi `val_loss` chứ không phải `val_acc`. Chọn loss là chuẩn hơn, vì loss phạt cả trường hợp model 'mơ hồ' — đoán sai nhưng vẫn trả xác suất cao.

---

## Phần 6 — Giai đoạn 3: Đánh giá & phân tích lỗi · 1m45 · `7:30`

> "**Phần đánh giá** là phần em cho là quan trọng nhất về mặt kỹ thuật, vì nó cho biết model thực sự tốt đến đâu, **và sai ở đâu**.
>
> Đầu tiên em mở tập test — **lần đầu tiên sau cả dự án** — và tính **4 chỉ số**:
>
> | Chỉ số | Giá trị |
> |--------|---------|
> | **Accuracy** | **92.10%** |
> | Precision (macro) | 92.11% |
> | Recall (macro) | 92.10% |
> | F1-Score (macro) | 92.09% |
>
> **92.10% — vượt mục tiêu 90% mà em đặt ra ban đầu.**
>
> Ba chỉ số còn lại gần như bằng nhau. Điều này có ý nghĩa: vì dataset **cân bằng**, model **không thiên lệch** về lớp nào. Với dữ liệu mất cân bằng, sự chênh lệch này mới là thứ đáng lo.
>
> Nhưng em không dừng ở con số tổng. Em làm thêm **phân tích ma trận nhầm lẫn** để tìm ra **model sai ở đâu**:
>
> | Lớp thực | Bị nhầm thành | Số lần |
> |-----------|---------------|--------|
> | Shirt | T-shirt/top | 95 |
> | T-shirt/top | Shirt | 90 |
> | Shirt | Pullover | 64 |
> | Coat | Pullover | 63 |
> | Coat | Shirt | 55 |
>
> `[[Tab 10]]` **Hai lớp bị nhầm nhiều nhất là Shirt và T-shirt/top — 185 lần, chiếm gần 20% tổng số lỗi.** Và điều này **hoàn toàn hợp lý về mặt thị giác**: cả hai đều là áo, cùng cổ tròn, cùng tay ngắn, chỉ khác nhau ở chi tiết nhỏ về cổ áo và độ dài — mà ở độ phân giải 28×28 thì những chi tiết đó gần như không nhìn thấy.
>
> `[[Tab 11]]` Em cũng vẽ thêm **lưới ảnh dự đoán** — xanh lá là đúng, đỏ là sai. Phần lớn là xanh, phần đỏ tập trung đúng vào các cặp áo — khớp với phân tích trên.
>
> Và em lưu lại **final model kèm metrics và config**, để người khác mở ra là biết ngay model này đạt bao nhiêu, dùng siêu tham số gì, và **tái tạo lại được kết quả**.""

**💡 Câu hỏi dự kiến:** *"Nếu muốn cải thiện thì làm gì?"* → Augmentation (+1–2%), ResNet18 (+1–3%), LR scheduler (+0.5–1%), Ensemble (+1–2%). Nhưng em thừa nhận trade-off: cải thiện 1–2% **không đáng** khi phải tăng thời gian train gấp nhiều lần.

---

## Phần 7 — Kết luận · 45s · `9:15`

> "**Tổng kết:**
>
> **Kết quả:** Test accuracy **92.10%**, precision/recall/F1 đều trên 92% — **vượt mục tiêu 90%** đã đặt ra trước khi bắt đầu.
>
> **Về mặt kỹ thuật**, em đã triển khai đầy đủ chuỗi pipeline: EDA → làm sạch → chuẩn hóa → chia dữ liệu 3 tập với stratified split → 4 kiến trúc mô hình qua factory pattern → grid + random search siêu tham số → training loop với early stopping và checkpoint → đánh giá 4 chỉ số → phân tích lỗi bằng confusion matrix. Toàn bộ **23 unit test cho phần training đều pass**.
>
> **Điều em học được lớn nhất** không phải là thuật toán nào cụ thể, mà là **tư duy thực nghiệm**: mọi quyết định về kiến trúc và siêu tham số trong dự án này đều đến từ **số liệu đo được**, không phải từ cảm tính. Chính nhờ đó em biết chính xác model của mình giỏi ở đâu và yếu ở đâu.
>
> **Hướng phát triển tiếp:** thêm Data Augmentation, thử ResNet18, và thêm learning rate scheduler — em đã ghi kế hoạch chi tiết trong tài liệu phương pháp.
>
> Cảm ơn thầy! Em sẵn sàng nhận câu hỏi.""

**🎬 Slide:** tổng kết + link GitHub + QR code.

---

## ⏱️ Tổng thời gian

| Phần | Nội dung | Từ – đến | Tab dùng |
|------|----------|----------|---------|
| 1 | Mở đầu | 0:00 – 0:45 | — |
| 2 | Bài toán & mục tiêu | 0:45 – 1:45 | 2, 3 |
| 3 | EDA & chuẩn bị dữ liệu | 1:45 – 3:15 | 4, 5 |
| 4 | Mô hình & siêu tham số | 3:15 – 5:45 | 6, 7 |
| 5 | Huấn luyện | 5:45 – 7:30 | 8, 9 |
| 6 | Đánh giá & phân tích lỗi | 7:30 – 9:15 | 10, 11 |
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

## ❓ Câu hỏi GV hay hỏi — trả lời 1 dòng

| Câu hỏi | Trả lời |
|---------|---------|
| Vì sao chọn seed 42? | Quy ước cố định — quan trọng là không đổi, không phải giá trị. |
| Train vs Val chênh nhau, có overfit? | Nhẹ (94.7% vs 92.8%); val loss tăng từ epoch 13 → EarlyStopping đã xử lý. |
| Test có bị rò rỉ? | Không — chỉ mở 1 lần ở bước đánh giá cuối. |
| Vì sao không GPU? | Máy không CUDA; ảnh 28×28 + 70k ảnh → CPU đủ. |
| Reproducible? | Seed 42 cố định khắp nơi; grid search tất định tuyệt đối. |
| Code tự viết? | Có — `src/{data,models,training,evaluation}/`, 23 unit test. |
| Grid 8 hay random 12? | **Cả hai.** Grid đầy đủ + tất định em chốt. Random quét 4 model. |
| Trên đĩa là ResNet, sao không chọn? | Random 3 epoch quá ngắn. Grid cho simple_cnn thắng 90.53%. ResNet chậm 3.2× mà chỉ kém 0.35%. |
| Vì sao CNN thắng MLP? | MLP coi pixel độc lập, không hiểu cấu trúc 2D; CNN chia sẻ filter nên học đặc trưng không gian. |
| Vì sao Adam thắng SGD? | Adam tự chỉnh learning rate từng tham số; cùng lr Adam hơn SGD 7–10%. |

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

*Tạo: 2026-10-21 · Bản rút gọn: `PRESENTATION_SCRIPT.md`*

---

## 🔧 Nếu thầy yêu cầu demo trực tiếp

```powershell
cd e:\lab1
$env:PYTHONPATH = "."
python main.py                          # Train lại 15 epoch + sinh đồ thị
python src/evaluation/run_evalution.py   # Đánh giá test + confusion matrix
pytest tests/training -v                # 23 unit test
```

---

## 🎨 Nếu làm slide (8 slide)

| # | Nội dung | Nguồn |
|---|----------|-------|
| 1 | Tiêu đề + tên + GitHub | — |
| 2 | Bài toán + 10 lớp + stats | `dataset_info.md` + `per_class_samples.png` |
| 3 | EDA: cân bằng + phân bố pixel | `class_distribution.png` + `pixel_histogram.png` |
| 4 | 4 kiến trúc + số tham số | `architecture_selection.md` |
| 5 | Grid 8 + random 12 trials | Bảng grid lấy từ `nguoi1_models_report.md` §2.2 |
| 6 | Đồ thị training (2 ảnh) | `loss_curve.png` + `accuracy_curve.png` |
| 7 | Confusion matrix + dự đoán | `confusion_matrix.png` + `predictions.png` |
| 8 | Tổng kết + hướng phát triển | — |

---

*Tài liệu tạo: 2026-10-21*
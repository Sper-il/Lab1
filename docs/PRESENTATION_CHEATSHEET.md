# LỜI THOẠI TRÌNH BÀY — FashionMNIST (10 phút)

---

## PHẦN 1 — Mở đầu · 40s · `0:00`

> "Em chào cô. Em tên là Sper-il.
>
> Hôm nay em trình bày đồ án phân loại ảnh thời trang bằng mạng nơ-ron tích chập, trên tập dữ liệu FashionMNIST.
>
> Em chọn bài này không phải vì nó dễ. Em chọn vì nó chạy được trọn vòng đời của một dự án machine learning thật: khám phá dữ liệu, dựng mô hình, tinh chỉnh siêu tham số, huấn luyện, đánh giá, rồi phân tích lỗi. Tất cả chạy bằng code, không làm thủ công.
>
> Source em đẩy lên GitHub công khai, cô kiểm tra bất cứ lúc nào ạ."

---

## PHẦN 2 — Bài toán & mục tiêu · 50s · `0:40`

> "Bài toán của em rất đơn giản: một ảnh xám 28×28 pixel, máy phải trả lời là món nào trong 10 loại. Áo thun, quần, áo len, váy, áo khoác, dép, áo sơ mi, giày thể thao, túi xách, ủng.
>
> → mở `results/runs/dataset_info.md`
>
> Tập dữ liệu có 70 nghìn ảnh, chia đều cho 10 lớp, tức là mỗi lớp khoảng 6 nghìn ảnh gốc.
>
> → mở `results/runs/per_class_samples.png`
>
> Đây là ảnh mẫu của từng lớp. Cô thấy không, đều 28×28 và đều đơn sắc. Em để ý là có nhiều cặp áo nhìn gần như giống nhau — đây sẽ là chỗ khó nhất của bài toán, và em sẽ quay lại nó ở phần 6.
>
> Trước khi viết dòng code đầu tiên, em đặt mục tiêu luôn: test accuracy tối thiểu 90%. Có mốc rõ ràng thì em biết khi nào nên dừng tìm, và khi nào là đang lãng phí thời gian."

---

## PHẦN 3 — Chuẩn bị dữ liệu · 1m30 · `1:30`

> "Giai đoạn một em làm bốn bước.
>
> **Bước một, khám phá dữ liệu.** Em tính được mean bằng 0.286 và standard deviation bằng 0.353. Pixel nằm trong khoảng 0 đến 1.
>
> → mở `results/runs/class_distribution.png`
>
> Mười cột này cao đều nhau. Nghĩa là dataset cân bằng hoàn hảo, không có NaN, không có ảnh hỏng. Nhờ vậy em không phải xử lý vấn đề mất cân bằng lớp — một việc nữa tiết kiệm.
>
> → mở `results/runs/pixel_histogram.png`
>
> Histogram có hai đỉnh: một đỉnh là nền sáng, một đỉnh là vật thể tối. Phân bố sạch, không có outlier.
>
> **Bước hai, làm sạch.** Em kiểm tra NaN, Inf, kích thước ảnh và nhãn. Kết quả là 60 nghìn trên 60 nghìn ảnh hợp lệ, tức 100%.
>
> **Bước ba, chuẩn hóa.** Em áp dụng đúng công thức trừ mean rồi chia std — 0.286 và 0.353, chính là hai con số em vừa tính ở bước một. Việc này cần, vì pixel lệch giá trị làm gradient bị nghiêng, model hội tụ chậm hơn.
>
> **Bước bốn, chia dữ liệu — ba tập, không phải hai:**

| Tập | Số ảnh | Vai trò |
|-----|--------|----------|
| Train | 50.000 | Model học trọng số |
| Validation | 10.000 | Đánh giá khi train, quyết định khi nào dừng |
| Test | 10.000 | Giữ kín, chỉ mở một lần ở cuối |

> Nếu chỉ chia hai tập thì có một cái bẫy: model có thể học thuộc luôn phần test. Khi đó điểm đẹp lên, nhưng đẹp sai nghĩa — tức là em đánh giá sai model của mình.
>
> Hai kỹ thuật em áp dụng ở đây. Một là stratified split, để tập validation cân bằng giữa cả 10 lớp. Hai là seed cố định 42, để chạy lại là ra đúng kết quả cũ."

---

## PHẦN 4 — Mô hình & siêu tham số · 2m15 · `3:00`

> "Phần này gồm bốn bước.
>
> Bước 1 – Dựng baseline. Em xây dựng MLP gồm Flatten 784 → Linear 256 → ReLU → Dropout 0.2 → Linear 10, với 203.530 tham số. Sau 2 epoch, mô hình đạt 87,05% validation accuracy, làm mốc đánh giá các kiến trúc khác. Mô hình phức tạp hơn phải cải thiện độ chính xác đủ đáng kể so với chi phí tính toán thì mới được giữ lại.
>
> **Bước hai, so sánh bốn kiến trúc.**
>
> → mở `results/runs/architecture_selection.md`
>
> | Kiến trúc | Tham số | Val Acc | s/epoch |
> |-----------|---------|---------|---------|
> | BaselineMLP | 203.530 | 87.05% | 31 |
> | DeepMLP | 118.650 | 86.10% | 33 |
> | **SimpleCNN** | **421.642** | **89.48%** | **96** |
> | ResNetMini | 174.970 | 88.63% | 259 |
>
> Em chọn SimpleCNN vì đạt accuracy cao nhất (89,48%) và chỉ mất 96 giây mỗi epoch, nhanh gấp 2,7 lần ResNetMini. ResNetMini chỉ đạt 88,63% nhưng tốn 259 giây mỗi epoch, nên không hiệu quả bằng trong thử nghiệm này.
>
> **Bước ba, tinh chỉnh siêu tham số.** Em chạy grid search 16 trials, mỗi trial 10 epoch, tổng cộng 160 epoch.
>
> Em chọn grid search vì nó **tất định** — chạy lại là ra đúng kết quả cũ, không có tổ hợp nào bị bỏ sót. Grid này phủ kín toàn bộ không gian: 4 kiến trúc × 2 learning rate × 2 optimizer, đúng bằng 16 trials.
>
> → mở `results/runs/hyperopt.md`
>
> Kết quả cuối cùng cấu hình tốt nhất là SimpleCNN + Adam + learning rate 0.001, đạt 92,33% validation accuracy; ResNet đạt 92,20%, gần tương đương.
>
> **Bước bốn, chốt cấu hình:** SimpleCNN, Adam, learning rate 0.001, batch 128, dropout 0.5, 15 epoch, hàm lỗi CrossEntropyLoss."

---

## PHẦN 5 — Huấn luyện · 1m45 · `5:15`

> "Giờ em train theo cấu hình vừa chốt.
>
> Về code, em có một lớp `Trainer` trong `src/training/trainer.py`. Mỗi epoch gồm hai vòng: một vòng train, một vòng validate.
>
> **Vòng train** là năm bước: zero gradient, forward, tính loss, backward, rồi bước optimizer cập nhật trọng số theo Adam.
>
> **Vòng validate** khác ở ba điểm. Một: gọi `model.eval()` để tắt dropout. Hai: dùng số liệu tích lũy của BatchNorm — nếu quên thì kết quả validation sai hoàn toàn. Ba: bọc trong `torch.no_grad()` để không tính gradient, tiết kiệm khoảng một nửa bộ nhớ. Và tất nhiên model không học gì trên tập validation.
>
> Hàm lỗi em dùng là CrossEntropyLoss. Nó gộp cả Softmax và Negative Log Likelihood vào một, và nhận thẳng logits thay vì xác suất, nên ổn định hơn về mặt số học.
>
> Callback thì em có một cái tên là *EarlyStopping*. Nó làm hai việc trong cùng một chỗ. Một: theo dõi `val_loss` với patience bằng 5, min_delta 0.0001 — nghĩa là 5 epoch liên tiếp mà loss không giảm thêm được 0.0001 thì dừng, vì lúc đó model đã overfit rồi. Hai: mỗi epoch lưu checkpoint ra file, một bản `last` là epoch vừa rồi, một bản `best` là khi loss vừa giảm, kèm trọng số, trạng thái optimizer và metrics.
>
> Một chi tiết em thêm nữa: sau khi train xong, chương trình tự nạp lại trọng số của epoch tốt nhất. Nghĩa là model em dùng để đánh giá là model tốt nhất, không phải model ở epoch cuối.
>
> Em chạy 15 epoch. Em kể cô nghe bốn mốc quan trọng:

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|----------|---------|
| 1 | 0.585 | 0.351 | 79.03% | 87.08% |
| 5 | 0.265 | 0.237 | 90.41% | 91.42% |
| **11 (best)** | 0.178 | **0.215** | 93.47% | 92.85% |
| 15 (last) | 0.141 | 0.230 | 94.72% | 92.80% |

> → mở `results/runs/loss_curve.png`
>
> Train loss giảm đều qua các epoch, cho thấy mô hình học ngày càng tốt trên tập huấn luyện. Validation loss giảm mạnh lúc đầu, đạt mức thấp nhất khoảng epoch 11, sau đó tăng nhẹ. Điều này cho thấy mô hình bắt đầu có dấu hiệu overfitting nhẹ, nên epoch 11 có thể là thời điểm phù hợp để chọn model tốt nhất.
>
> → mở `results/runs/accuracy_curve.png`
>
> Train accuracy tăng đều và đạt khoảng 94,7%, trong khi validation accuracy đạt đỉnh khoảng 92,9% rồi gần như chững lại. Điều này cho thấy mô hình học tốt trên tập train nhưng khả năng tổng quát hóa không tăng thêm nhiều, cũng là dấu hiệu của overfitting nhẹ ở các epoch cuối.

> **Tra cứu (KHÔNG đọc ra) — vị trí code trong lời thoại trên:**
>
> | Nói gì trong lời thoại | File | Dòng |
> |----------------------|------|------|
> | lớp `Trainer`, hai vòng mỗi epoch | `src/training/trainer.py` | class 17 · `train_epoch` 54 · `validate` 80 · gọi ở 111–112 |
> | 5 bước vòng train | `src/training/trainer.py` | zero_grad 65 · forward 66 · loss 67 · backward 68 · step 69 |
> | `model.eval()` | `src/training/trainer.py` | 83 |
> | `torch.no_grad()` | `src/training/trainer.py` | 88 |
> | CrossEntropyLoss | `src/training/losses.py` | 12 |
> | callback `EarlyStopping` | `src/training/callbacks.py` | class 12 · `__init__` 15 |
> | lưu checkpoint `last` / `best` | `src/training/callbacks.py` | 105 / 108 |
> | nạp lại epoch tốt nhất | `src/training/trainer.py` | 156 → `callbacks.py` 112 |

---

## PHẦN 6 — Đánh giá & phân tích lỗi · 1m30 · `6:45`

> "Đây là phần đánh giá hiệu quả thực tế của mô hình trên tập test, với bốn chỉ số.
>
> → mở `results/runs/test_metrics.md`

| Chỉ số | Giá trị |
|--------|---------|
| **Accuracy** | **92.10%** |
| Precision (macro) | 92.11% |
| Recall (macro) | 92.10% |
| F1-Score (macro) | 92.09% |

> Kết quả cho thấy Accuracy đạt 92,10%, vượt mục tiêu 90% em đặt ra ban đầu. Precision, Recall và F1-Score cũng đều xấp xỉ 92,1%, cho thấy mô hình hoạt động khá đồng đều giữa các lớp.
>
> Tuy nhiên, để hiểu rõ hơn mô hình dự đoán sai ở đâu, em tiếp tục phân tích ma trận nhầm lẫn. Kết quả cho thấy mô hình thường nhầm giữa các loại quần áo có hình dáng tương tự, đặc biệt là Shirt với T-shirt/top, và Coat với Pullover.
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
> Nhìn vào ma trận nhầm lẫn, đường chéo chính thể hiện tỷ lệ mô hình dự đoán đúng ở từng lớp. Kết quả cho thấy mô hình nhận diện tốt nhất lớp Bag với 99%, trong khi lớp Shirt thấp nhất, khoảng 76%. Cụ thể, mô hình thường nhầm Shirt thành T-shirt/top khoảng 10%, và nhầm T-shirt/top thành Shirt khoảng 9%. Điều này cho thấy mô hình khó phân biệt những loại quần áo có hình dáng tương tự.
>
> → mở `results/runs/predictions.png`
>
> Cuối cùng, em trực quan hóa một số ảnh trong tập test để so sánh nhãn thực tế với nhãn mô hình dự đoán. Màu xanh là dự đoán đúng, màu đỏ là dự đoán sai. Qua đó, em có thể kiểm tra trực quan khả năng phân loại của mô hình và hiểu rõ hơn những trường hợp mô hình dễ nhầm lẫn."

| Con số | Ý nghĩa |
|--------|----------|
| **92.10%** | Test accuracy — kết quả cuối |
| **92.11 / 92.10 / 92.09%** | Precision / Recall / F1 (macro) |
| **50.000 / 10.000 / 10.000** | Train / Val / Test |
| **0.286 / 0.353** | Mean / Std của pixel |
| **92.33%** | Best val acc — Grid search 16 trials × 10 epoch |
| **96 s/epoch** | Tốc độ train SimpleCNN (CPU) |
| **0.215** | Val loss thấp nhất @ epoch 11 |
| **185** | Tổng nhầm Shirt ↔ T-shirt |
| **27** | Số unit test (`tests/`, tất cả đều pass) |
| **421.642** | Params SimpleCNN |
| **87.05%** | Baseline MLP — mức sàn 2 epoch |
| **259 / 96** | s/epoch ResNetMini / SimpleCNN = 2,7× chậm hơn |
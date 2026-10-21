# Kịch Bản Thuyết Trình — FashionMNIST Classification (10 phút)

**Người trình bày:** Sper-il
**Thời lượng:** 10 phút
**Tài liệu đi kèm:** `docs/PROJECT_METHODOLOGY.md`, `docs/reports/`

---

## 🚀 MỞ FILE NÀO Ở ĐÂU — BẢNG CHỈ DẪN THEO THỨ TỰ

> Mở sẵn 8 cửa sổ này trước khi lên lớp, xếp theo thứ tự sẽ nói.

| # | Mở file | Dùng khi nào | Nói gì khi chỉ vào |
|---|----------|--------------|-------------------|
| **1** | `docs/PRESENTATION_SCRIPT.md` | Xuyên suốt | Chính là lời thoại này — để ở chế độ Reader (F11) |
| **1b** | `results/runs/dataset_info.md` | Phần 2 (1:00) | **Bảng 10 lớp + thông số dataset** ← file này là bảng đó |
| **2** | `results/runs/per_class_samples.png` | Phần 2 (1:00) | 10 lớp, mỗi lớp 1 ảnh mẫu |
| **3** | `results/runs/class_distribution.png` | Phần 3 (1:45) | 10 cột đều nhau = cân bằng |
| **4** | `results/runs/pixel_histogram.png` | Phần 3 (2:00) | Histogram phân bố cường độ pixel |
| **5** | `results/runs/hyperopt.md` | Phần 4 (4:30) | ⚠️ **random search 12 trials** — không phải grid 8 trials |
| **5b** | `results/runs/architecture_selection.md` | Phần 4 | So sánh 4 kiến trúc + s/epoch |
| **6** | `results/runs/loss_curve.png` | Phần 5 (6:30) | Vùng epoch 11–13: val loss bắt đầu tăng |
| **7** | `results/runs/accuracy_curve.png` | Phần 5 (6:45) | Train acc vẫn tăng, val acc đi ngang |
| **8** | `results/runs/confusion_matrix.png` | Phần 6 (8:00) | Hàng Shirt & cột T-shirt sáng nhất |
| **9** | `results/runs/predictions.png` | Phần 6 (8:45) | Ô xanh = đúng, ô đỏ = sai |
| **10** | `docs/reports/nguoi1_models_report.md` | Dự phòng | Bảng so sánh 4 kiến trúc + params |
| **11** | `docs/reports/nguoi3_evaluation_report.md` | Dự phòng | Bảng metrics test set + top-5 nhầm lẫn |
| **12** | `docs/PROJECT_METHODOLOGY.md` | Dự phòng | Tài liệu phương pháp đầy đủ |

### Thứ tự tab nên mở (bấm `Ctrl+Tab` để chuyển nhanh)

```
Tab 1: PRESENTATION_SCRIPT.md     ← đang đọc
Tab 2: dataset_info.md            ← Phần 2 (bảng 10 lớp + stats)
Tab 3: per_class_samples.png      ← Phần 2
Tab 4: class_distribution.png     ← Phần 3
Tab 5: pixel_histogram.png        ← Phần 3
Tab 6: hyperopt.md                ← Phần 4 (⚠️ random 12 trials)
Tab 7: loss_curve.png             ← Phần 5
Tab 8: accuracy_curve.png         ← Phần 5
Tab 9: confusion_matrix.png       ← Phần 6
Tab 10: predictions.png           ← Phần 6
```

> ⚠️ **Trước khi lên lớp:** đọc `docs/PRESENTATION_FACTS.md` — file này ghi lại mọi con số
> đã kiểm chứng và **4 chỗ sai đã sửa** trong kịch bản.

> 💡 **Mẹo thao tác:** Trong VSCode, dùng `Ctrl+P` → gõ tên file → Enter để mở nhanh không cần rời bàn phím. Với file `.png`, bấm chuột phải → **Open Preview** để phóng to fullscreen.

---

## 📋 Tóm tắt phân bố thời gian

| Phần | Nội dung | Thời gian | Mốc |
|------|----------|-----------|------|
| 1 | Mở đầu & đặt vấn đề | 1:00 | 0:00 – 1:00 |
| 2 | Bài toán & mục tiêu | 0:45 | 1:00 – 1:45 |
| 3 | Giai đoạn 1 — Chuẩn bị dữ liệu | 1:30 | 1:45 – 3:15 |
| 4 | Giai đoạn 2 — Phát triển mô hình | 2:30 | 3:15 – 5:45 |
| 5 | Giai đoạn 3 — Huấn luyện | 1:45 | 5:45 – 7:30 |
| 6 | Giai đoạn 3 — Đánh giá & phân tích lỗi | 1:45 | 7:30 – 9:15 |
| 7 | Kết luận & hướng phát triển | 0:45 | 9:15 – 10:00 |

> **Mẹo:** Ghi chú những con số in đậm, đó là những thứ giảng viên hay hỏi.

---

## PHẦN 1 — MỞ ĐẦU (1 phút) `0:00`

> "Em xin chào thầy. Em tên là Sper-il. Hôm nay em trình bày đồ án **Phân loại ảnh thời trang với mạng nơ-ron tích chập trên dataset FashionMNIST**.

> Đây là một bài toán kinh điển của Deep Learning, nhưng em chọn nó không phải vì dễ, mà vì nó cho phép em **đi đủ toàn bộ vòng đời của một dự án Machine Learning thực tế** — từ khám phá dữ liệu, xây dựng mô hình, tinh chỉnh siêu tham số, huấn luyện, cho tới đánh giá và phân tích lỗi — tất cả đều được **tự động hoá bằng code** chứ không phải chạy thủ công.

> Toàn bộ source code của em đã đẩy lên GitHub công khai, để thầy có thể kiểm tra lại bất kỳ lúc nào."

**📎 Mở slide: tiêu đề đồ án + GitHub repo**

---

## PHẦN 2 — BÀI TOÁN & MỤC TIÊU (45 giây) `1:00`

> "**Bài toán:** Cho một ảnh xám 28×28 pixel, máy tính phải trả lời đây là món đồ nào, trong 10 loại: áo thun, quần, áo len, váy, áo khoác, dép, áo sơ mi, giày thể thao, túi xách, và ủng.

> Dataset có **60.000 ảnh huấn luyện** và **10.000 ảnh kiểm tra**, tất cả đều cân bằng tuyệt đối — mỗi lớp đúng 6.000 ảnh train và 1.000 ảnh test.

> **Mục tiêu em đặt ra là đạt độ chính xác trên tập kiểm tra tối thiểu 90%.**

> Em đặt mục tiêu này trước khi bắt đầu code, vì một mục tiêu rõ ràng giúp em biết khi nào nên dừng tìm kiếm, thay vì cứ mò mãi mãi."

**📎 Mở slide: bảng 10 lớp + dataset stats** → lấy từ `results/runs/dataset_info.md`

---

## PHẦN 3 — GIAI ĐOẠN 1: CHUẨN BỊ DỮ LIỆU (1 phút 30 giây) `1:45`

> "**Giai đoạn 1** là giai đoạn em coi là quan trọng nhất, vì **80% lỗi trong Machine Learning đến từ dữ liệu chứ không phải từ mô hình**.

> Em chia nó thành 4 bước:

**Bước 1 — Khám phá dữ liệu (EDA).** Em tính thống kê toàn bộ pixel: mean = 0.286, std = 0.353, giá trị nằm trong khoảng [0, 1]. Rồi em vẽ biểu đồ phân bố lớp và trực quan hóa mẫu ảnh từ từng lớp. Kết quả: dữ liệu **cân bằng hoàn hảo**, không có NaN, không có ảnh hỏng. Nhờ vậy em **không cần** xử lý mất cân bằng lớp — một việc mà nếu có, sẽ tốn rất nhiều công sức.

**Bước 2 — Làm sạch dữ liệu.** Em viết hàm kiểm tra toàn diện: NaN/Inf, kích thước ảnh đúng chuẩn 28×28×1, nhãn trong khoảng 0–9. Kết quả **60.000/60.000 mẫu hợp lệ — 100%**.

**Bước 3 — Tiền xử lý.** Em **chuẩn hóa** dữ liệu theo công thức `(x − mean) / std` với đúng mean và std đã tính ở bước 1. Tại sao cần? Vì pixel có giá trị lệch nhau sẽ khiến gradient bị "nghiêng", làm model hội tụ chậm hơn và khó tối ưu hơn.

**Bước 4 — Chia dữ liệu.** Đây là bước em muốn nhấn mạnh. Em chia thành **3 tập** chứ không phải 2:
> - **Train** 54.000 ảnh — nơi model học.
> - **Validation** 6.000 ảnh — model **không học**, chỉ dùng để đánh giá trong lúc train và quyết định khi nào nên dừng.
> - **Test** 10.000 ảnh — giữ kín đến cuối cùng, mới mở ra đánh giá một lần duy nhất.

> Nếu chỉ chia 2 tập thì model có thể 'học thuộc' cả tập test, và kết quả đánh giá sẽ **đẹp hơn thực tế** — tức là đánh giá sai.

> Hai kỹ thuật em áp dụng ở đây: **Stratified Split** để mỗi lớp có đúng 600 ảnh trong tập validation như dataset gốc, và **random seed cố định = 42** để mọi lần chạy lại đều cho kết quả giống hệt nhau — tức là kết quả **reproducible**."

**📎 Mở tab 4: `results/runs/class_distribution.png`** — 10 cột cao đều nhau → chứng minh "cân bằng hoàn hảo" bằng mắt
**📎 Mở tab 4b: `results/runs/pixel_histogram.png`** — histogram phân bố cường độ pixel

**📎 Mở tab 2: `results/runs/dataset_info.md`** — bảng 10 lớp (kèm số lượng train/val/test từng lớp) + bảng thông số dataset (mean 0.286, std 0.353, shape (1,28,28), 60.000 mẫu hợp lệ)
**📎 Mở tab 3: `results/runs/per_class_samples.png`** — lưới 10 ảnh mẫu, mỗi ảnh một lớp

**💡 Câu hỏi dự kiến:** *"Vì sao chọn seed 42?"* → 42 là số quen thuộc trong văn hóa đại chúng (trong "Hitchhiker's Guide to the Galaxy" 42 là câu trả lời cho mọi câu hỏi), và quan trọng hơn là nó chỉ là **một quy ước bất kỳ nhưng cố định** — điều duy nhất cần là nó không đổi.

---

## PHẦN 4 — GIAI ĐOẠN 2: PHÁT TRIỂN MÔ HÌNH (2 phút 30 giây) `3:15`

> "**Giai đoạn 2** là tìm ra kiến trúc mô hình và bộ siêu tham số tốt nhất. Em làm theo 4 bước, mỗi bước một quyết định có căn cứ.

**Bước 1 — Baseline.** Trước khi làm gì phức tạp, em dựng một MLP đơn giản — Flatten 784 → 256 → 128 → 10 — để có **mức sàn** để so sánh. Nếu không có baseline, em sẽ không biết mô hình phức tạp có thật sự tốt hơn hay không.

**Bước 2 — Khám phá kiến trúc.** Em cài đặt và so sánh **4 kiến trúc** qua một `ModelFactory` — một registry, nên thêm kiến trúc mới chỉ cần **1 dòng code**:

| Kiến trúc | Số tham số | Đặc điểm |
|-----------|------------|-----------|
| BaselineMLP | 203.530 | 1 tầng ẩn — sanity check |
| DeepMLP | 118.650 | 3 tầng FC + BatchNorm + Dropout |
| **SimpleCNN** | **421.642** | 2 block Conv + 2 tầng FC + Dropout |
| ResNetMini | 174.970 | Residual blocks + skip connections |

> **Vì sao CNN thắng?** Vấn đề cốt lõi của MLP với ảnh là nó coi **mỗi pixel là độc lập** — nó không hiểu ảnh có cấu trúc không gian 2D, không có tính chất "dịch chuyển" (translation invariance). Trong khi đó CNN trượt một filter 3×3 trên toàn bộ ảnh và **chia sẻ trọng số**, nên nó học được đặc trưng không gian — cạnh, góc, texture — với **ít tham số hơn** và **tốc độ train nhanh hơn**.

> Kiến trúc em chọn là **SimpleCNN**: Conv(1→32) → ReLU → MaxPool → Conv(32→64) → ReLU → MaxPool → Flatten → FC(3136→128) → Dropout → FC(128→10).

**Bước 3 — Tinh chỉnh siêu tham số.** Em chạy **hai lần thử nghiệm**:

**Lần 1 — Grid search 8 trials × 3 epoch** (quét *đầy đủ* 2 model × 2 learning rate × 2 optimizer = 8 tổ hợp, ~19 phút CPU):

| Model | LR | Optim | Val Acc |
|-------|-----|-------|---------|
| **simple_cnn** | **0.001** | **adam** | **90.53%** |
| simple_cnn | 0.0005 | adam | 88.97% |
| deep_mlp | 0.001 | adam | 87.13% |
| deep_mlp | 0.0005 | adam | 87.07% |
| deep_mlp | 0.001 | sgd | 83.98% |
| deep_mlp | 0.0005 | sgd | 82.25% |
| simple_cnn | 0.001 | sgd | 80.83% |
| simple_cnn | 0.0005 | sgd | 78.13% |

> **Ba nhận xét từ bảng này:**
> - **Adam luôn thắng SGD** ở cùng learning rate — chênh lệch tới **7–10%**. Đúng với lý thuyết: Adam tự điều chỉnh learning rate cho từng tham số, còn SGD thì không.
> - **`simple_cnn` thắng `deep_mlp` ở *mọi* tổ hợp** — CNN khai thác cấu trúc không gian 2D của ảnh, MLP thì không.
> - **SGD ở lr=0.0005** học rất chậm — chỉ 78% sau 3 epoch.

**Lần 2 — Random search 12 trials × 3 epoch** (quét cả **4 kiến trúc** với siêu tham số ngẫu nhiên, ~13 phút). Trial tốt nhất ở đây là `resnet_mini` + adam + lr=0.01 đạt **89.15%**.

> Em chủ động **chốt theo grid search**, vì: (1) grid quét **đầy đủ** mọi tổ hợp, random có thể trượt; (2) grid **tất định** — cùng config cho cùng kết quả. Điểm random thấp hơn không phải do `resnet_mini` tệ hơn, mà vì **3 epoch quá ngắn để hội tụ** — benchmark riêng cho thấy `resnet_mini` chỉ kém `simple_cnn` **0.35%** nhưng chậm **3.2×**.

**Bước 4 — Chọn mô hình.** Em chốt **SimpleCNN + Adam + lr=0.001 + batch 128 + dropout 0.5** và chuyển sang giai đoạn 3.

> Điểm em muốn lưu ý: em **không** chọn mô hình dựa trên cảm tính, mà dựa trên **số liệu validation**. Tập test được giữ kín hoàn toàn cho tới bước đánh giá cuối."

**📎 Mở tab 5: `results/runs/hyperopt.md`** — ⚠️ đây là bảng **random search 12 trials**. Muốn trình bày bảng **grid 8 trials** thì lấy từ `docs/reports/nguoi1_models_report.md` §2.2
**📎 Mở tab 5b: `results/runs/architecture_selection.md`** — so sánh 4 kiến trúc: `simple_cnn` 88.83% / 75 s-per-epoch vs `resnet_mini` 88.48% / 239 s-per-epoch

**💡 Câu hỏi dự kiến:** *"Vì sao không dùng mô hình lớn hơn để đạt điểm cao hơn?"* → Em **đã thử** ResNetMini. Nó chỉ kém 0.35% (88.48% vs 88.83%) nhưng chậm gấp 3.2× (239 vs 75 s/epoch). Với mục tiêu 90% và SimpleCNN đã vượt (92.1%), đánh đổi đó không đáng.

---

## PHẦN 5 — GIAI ĐOẠN 3: HUẤN LUYỆN (1 phút 45 giây) `5:45`

> "**Giai đoạn 3** em tách làm hai phần: huấn luyện, rồi đánh giá.

**Phần huấn luyện.** Em viết một lớp `Trainer` gọn gàng, mỗi epoch gồm 2 vòng:

> - **Train loop** — 5 bước chuẩn: `zero_grad` → **forward** (dự đoán) → **loss** (đo sai số) → **backward** (tính gradient) → **step** (cập nhật trọng số).
> - **Validate loop** — đánh giá trên tập validation với `model.eval()` và `torch.no_grad()` để tắt dropout, dùng statistics đã tích lũy của BatchNorm, và không tính gradient — tiết kiệm khoảng **50% bộ nhớ**.

> Loss em dùng là **CrossEntropyLoss** — nó gộp cả Softmax và Negative Log Likelihood vào một hàm, dùng logits thay vì xác suất để **ổn định số học**. Optimizer là **Adam**.

> Em áp dụng 2 callback quan trọng:
> - **EarlyStopping** với patience = 5, theo dõi `val_loss`. Nếu validation loss không cải thiện trong 5 epoch liên tiếp thì dừng — vì lúc đó model đã bắt đầu **overfit**.
> - **Checkpoint**: lưu cả **best model** (theo val_loss thấp nhất, dùng để đánh giá và deploy) lẫn **last model** (để resume training nếu bị crash giữa chừng).

> Em chạy **15 epoch**. Kết quả thể hiện rất rõ:

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|----------|---------|
| 1 | 0.585 | 0.351 | 79.03% | 87.08% |
| 5 | 0.265 | 0.237 | 90.41% | 91.42% |
| 11 (best) | 0.178 | **0.215** | 93.47% | 92.85% |
| 15 (last) | 0.141 | 0.230 | 94.72% | 92.80% |

> Nếu thầy nhìn vào đây sẽ thấy ngay: **train accuracy cứ tăng đều lên 94.72%, nhưng validation accuracy đi ngang từ epoch 13 và validation loss bắt đầu tăng trở lại**. Đây chính là dấu hiệu overfit — và nó cho thấy **EarlyStopping đã hoạt động đúng**, dừng model đúng lúc tốt nhất thay vì để nó học thuộc.

> Em cũng vẽ biểu đồ loss và accuracy để trực quan hóa quá trình này."

**📎 Mở tab 5: `results/runs/loss_curve.png`** — chỉ vào khoảng epoch 11–13 (val loss bắt đầu tăng lại)
**📎 Mở tab 6: `results/runs/accuracy_curve.png`** — chỉ vào khoảng epoch 13 (train acc vẫn tăng, val acc đi ngang)

---

## PHẦN 6 — GIAI ĐOẠN 3: ĐÁNH GIÁ & PHÂN TÍCH LỖI (1 phút 45 giây) `7:30`

> "**Phần đánh giá** là phần em cho là quan trọng nhất về mặt kỹ thuật, vì nó cho biết model thực sự tốt đến đâu, **và sai ở đâu**.

> Đầu tiên em mở tập test — lần đầu tiên sau cả dự án — và tính **4 chỉ số**:

| Chỉ số | Giá trị |
|--------|---------|
| **Accuracy** | **92.10%** |
| Precision (macro) | 92.11% |
| Recall (macro) | 92.10% |
| F1-Score (macro) | 92.09% |

> **92.10% — vượt mục tiêu 90% mà em đặt ra ban đầu.**

> Ba chỉ số còn lại gần như bằng nhau. Điều này có ý nghĩa: vì dataset **cân bằng**, model **không thiên lệch** về lớp nào — không phải dự đoán thiên về lớp lớn, cũng không bỏ sót lớp nào. Với dữ liệu mất cân bằng, sự chênh lệch này mới là thứ đáng lo.

> Nhưng em không dừng ở con số tổng. Em làm thêm **phân tích ma trận nhầm lẫn** để tìm ra **model sai ở đâu**:

| Lớp thực | Bị nhầm thành | Số lần |
|-----------|---------------|--------|
| Shirt | T-shirt/top | 95 |
| T-shirt/top | Shirt | 90 |
| Shirt | Pullover | 64 |
| Coat | Pullover | 63 |
| Coat | Shirt | 55 |

> Kết luận từ đây rất rõ: **hai lớp bị nhầm nhiều nhất là Shirt và T-shirt/top — 185 lần, chiếm gần 20% tổng số lỗi**. Và điều này **hoàn toàn hợp lý về mặt thị giác**: cả hai đều là áo, cùng cổ tròn, cùng tay ngắn, chỉ khác nhau ở chi tiết nhỏ về cổ áo và độ dài — mà ở độ phân giải 28×28 thì những chi tiết đó gần như không nhìn thấy.

> Em cũng vẽ thêm **lưới ảnh dự đoán** — xanh lá là đúng, đỏ là sai — để kiểm chứng trực quan.

> Và em lưu lại **final model kèm metrics và config**, để người khác mở ra là biết ngay model này đạt bao nhiêu, dùng siêu tham số gì, và **tái tạo lại được kết quả**."

**📎 Mở tab 7: `results/runs/confusion_matrix.png`** — chỉ vào HÀNG "Shirt" và CỘT "T-shirt/top" (ô sáng nhất), rồi đến hàng "T-shirt/top" cột "Shirt"
**📎 Mở tab 8: `results/runs/predictions.png`** — lưới ảnh, ô xanh lá = dự đoán đúng, ô đỏ = sai

**💡 Câu hỏi dự kiến:** *"Nếu muốn cải thiện thì làm gì?"* → Em đã tính sẵn trong tài liệu: Data Augmentation (+1–2%), ResNet18 (+1–3%), Learning rate scheduler (+0.5–1%), Ensemble (+1–2%). Nhưng em cũng thừa nhận trade-off: cải thiện 1–2% là **không đáng** khi phải tăng thời gian train gấp nhiều lần và độ phức tạp vận hành.

---

## PHẦN 7 — KẾT LUẬN (45 giây) `9:15`

> "**Tổng kết:**

> **Kết quả:** Test accuracy **92.10%**, precision/recall/F1 đều trên 92% — **vượt mục tiêu 90%** đã đặt ra trước khi bắt đầu.

> **Về mặt kỹ thuật**, em đã triển khai đầy đủ chuỗi pipeline: EDA → làm sạch → chuẩn hóa → chia dữ liệu 3 tập với stratified split → 4 kiến trúc mô hình qua factory pattern → grid search + random search siêu tham số → training loop với early stopping và checkpoint → đánh giá 4 chỉ số → phân tích lỗi bằng confusion matrix. Toàn bộ **23 unit test cho phần training đều pass**.

> **Điều em học được lớn nhất** không phải là thuật toán nào cụ thể, mà là **tư duy thực nghiệm**: mọi quyết định về kiến trúc và siêu tham số trong dự án này đều đến từ **số liệu đo được**, không phải từ cảm tính. Chính nhờ đó em biết chính xác model của mình giỏi ở đâu và yếu ở đâu.

> **Hướng phát triển tiếp:** thêm Data Augmentation, thử ResNet18, và thêm learning rate scheduler — em đã ghi kế hoạch chi tiết trong tài liệu phương pháp.

> Cảm ơn thầy! Em sẵn sàng nhận câu hỏi."

**📎 Mở slide: tổng kết + link GitHub + QR code (nếu có)**

---

## 🎯 PHỤ LỤC: CHUẨN BỊ TRƯỚC KHI TRÌNH BÀY

### Danh sách slide nên có (8 slide)

| # | Nội dung slide | Nguồn hình ảnh (copy từ file) |
|---|----------------|-------------------------------|
| 1 | Tiêu đề + tên + GitHub | — |
| 2 | Bài toán + 10 lớp + dataset stats | Copy bảng từ `results/runs/dataset_info.md` + ảnh `per_class_samples.png` |
| 3 | EDA: cân bằng lớp + phân bố pixel | `results/runs/class_distribution.png` + `pixel_histogram.png` |
| 4 | 4 kiến trúc + số tham số | Bảng trong `docs/reports/nguoi1_models_report.md` §1 |
| 5 | Grid search 8 trials + random 12 trials | `results/runs/hyperopt.md` (⚠️ random) hoặc copy bảng grid từ `nguoi1_models_report.md` §2.2 |
| 6 | Đồ thị training (2 ảnh cạnh nhau) | `results/runs/loss_curve.png` + `accuracy_curve.png` |
| 7 | Confusion matrix + dự đoán | `results/runs/confusion_matrix.png` + `predictions.png` |
| 8 | Tổng kết + hướng phát triển | — |

### File nào cần mở trước (checklist in ra giấy)

```
[x] docs/PRESENTATION_SCRIPT.md         — lời thoại
[x] docs/PRESENTATION_FACTS.md          — số đã kiểm chứng + 4 lỗi đã sửa
[x] results/runs/dataset_info.md         — Phần 2 (bảng 10 lớp + stats)
[x] results/runs/per_class_samples.png   — Phần 2
[x] results/runs/class_distribution.png  — Phần 3
[x] results/runs/pixel_histogram.png     — Phần 3
[x] results/runs/hyperopt.md             — Phần 4 (⚠️ random 12 trials)
[x] results/runs/architecture_selection.md — Phần 4 (dự phòng)
[x] results/runs/loss_curve.png          — Phần 5
[x] results/runs/accuracy_curve.png      — Phần 5
[x] results/runs/confusion_matrix.png    — Phần 6
[x] results/runs/predictions.png         — Phần 6
[ ] docs/reports/nguoi1_models_report.md — dự phòng (bảng grid 8 trials)
[ ] docs/reports/nguoi3_evaluation_report.md — dự phòng
```

### Nếu không dùng PowerPoint — trình bày trực tiếp bằng VSCode

Mở 8 tab theo thứ tự, dùng `Ctrl+Tab` chuyển tab. Với file `.png`:
- Bấm chuột phải → **Open Preview** → phóng to fullscreen bằng `F11`
- Hoặc kéo file `.png` vào cửa sổ trình duyệt để xem full màn hình

### Demo trực tiếp (nếu thầy yêu cầu chạy thử)

```powershell
cd e:\lab1
$env:PYTHONPATH = "."
python main.py                          # Train lại từ đầu
python src/evaluation/run_evalution.py   # Đánh giá + sinh confusion matrix
pytest tests/training -v                # 23 unit tests
```

### Những con số phải nhớ (in đậm trong script)

| Số | Ý nghĩa |
|----|---------|
| **92.10%** | Test accuracy — kết quả cuối cùng |
| **92.11 / 92.10 / 92.09** | Precision / Recall / F1 (macro) |
| **54.000 / 6.000 / 10.000** | Train / Val / Test |
| **0.286 / 0.353** | Mean / Std của pixel |
| **90.53%** | Val accuracy — grid search (best trial) |
| **89.15%** | Val accuracy — random search (best trial, 3 epoch) |
| **75 s/epoch** | Tốc độ train `simple_cnn` (CPU) |
| **0.215** | Val loss thấp nhất (epoch 11) |
| **185** | Tổng số lần nhầm Shirt ↔ T-shirt |
| **4** | Số kiến trúc mô hình |
| **8 trials (grid) + 12 trials (random)** | Quy mô tìm kiếm siêu tham số |
| **23** | Số unit test (`tests/training/`) |

### Câu hỏi giảng viên thường hỏi & cách trả lời

| Câu hỏi | Gợi ý trả lời |
|---------|---------------|
| "Vì sao không dùng augmentation?" | Giai đoạn 1 tập trung vào pipeline cơ bản; augmentation đã ghi trong hướng phát triển, kỳ vọng +1–2% |
| "Train vs Val chênh nhau nhiều, có overfit không?" | Có dấu hiệu nhẹ từ epoch 13; val loss tăng trở lại → EarlyStopping đã xử lý |
| "Test set có bị dùng khi train không?" | Không — chỉ mở 1 lần ở bước đánh giá cuối, sau khi mọi quyết định đã chốt |
| "Tại sao không dùng GPU?" | Máy không có CUDA; CPU vẫn đủ vì ảnh 28×28 và dataset chỉ 60K ảnh |
| "Reproducibility?" | Seed 42 cố định ở config, split, và training; cùng config → cùng kết quả |
| "Code có phải của bạn không?" | Có — chia theo 3 module `data/` / `models/` / `training/` + `evaluation/`, có **23 unit test** cho phần training, log chi tiết trong `docs/reports/` |
| "Grid 8 trials hay random 12 trials?" | **Cả hai.** Grid quét 2 model × 2 lr × 2 optimizer = 8 tổ hợp đầy đủ, tất định. Random quét cả 4 model, 12 trials. Em chốt theo grid vì không bỏ sót tổ hợp. |
| "Vì sao best trial trên đĩa là resnet_mini mà em chọn simple_cnn?" | Đó là lần chạy **random 3 epoch** — quá ngắn nên nhiễu cao. Grid cho thấy `simple_cnn` thắng 90.53%. Ngoài ra `resnet_mini` chậm gấp 3.2× mà chỉ kém 0.35%. |

---

*Tài liệu tạo: 2026-10-21*

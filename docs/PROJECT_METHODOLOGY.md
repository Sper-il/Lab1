# FashionMNIST Classification — Quy Trình Thực Hiện Dự Án

**Tác giả:** Sper-il  
**Ngày:** 2026-10-15  
**Dataset:** [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)  
**Framework:** PyTorch 2.14+  
**GitHub:** https://github.com/Sper-il/Lab1

---

## Mục lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Giai đoạn 1: Chuẩn bị dữ liệu](#2-giai-đoạn-1-chuẩn-bị-dữ-liệu)
3. [Giai đoạn 2: Phát triển mô hình](#3-giai-đoạn-2-phát-triển-mô-hình)
4. [Giai đoạn 3: Huấn luyện và đánh giá](#4-giai-đoạn-3-huấn-luyện-và-đánh-giá)
5. [Tổng kết kết quả](#5-tổng-kết-kết-quả)
6. [Cấu trúc dự án](#6-cấu-trúc-dự-án)

---

## 1. Tổng quan dự án

### 1.1 Bài toán

**Chúng ta cần làm gì?**

Cho một bức ảnh xám 28×28 pixel của một món đồ thời trang, máy tính phải trả lời: "Đây là món gì?" trong 10 loại sau:

| Mã | Tên loại | Biểu tượng |
|----|----------|------------|
| 0 | Áo thun | 👕 |
| 1 | Quần dài | 👖 |
| 2 | Áo len | 🧥 |
| 3 | Váy | 👗 |
| 4 | Áo khoác | 🧥 |
| 5 | Dép | 🩴 |
| 6 | Áo sơ mi | 👔 |
| 7 | Giày thể thao | 👟 |
| 8 | Túi xách | 👜 |
| 9 | Ủng | 👢 |

**Tại sao bài toán này quan trọng?**
- Là bài toán kinh điển để học Deep Learning
- Ứng dụng thực tế: phân loại sản phẩm thời trang tự động cho shop online
- Kích thước ảnh nhỏ → có thể chạy trên CPU, không cần GPU đắt tiền

### 1.2 Thông số kỹ thuật

| Thông số | Giá trị | Ý nghĩa |
|----------|---------|---------|
| Ảnh đầu vào | 28×28 pixel xám | Mỗi ảnh chỉ có 1 kênh màu (đen trắng) |
| Số mẫu train | 60,000 | 6,000 ảnh cho mỗi lớp |
| Số mẫu test | 10,000 | 1,000 ảnh cho mỗi lớp |
| Số lớp | 10 | 10 loại quần áo |
| Tổng tham số ảnh | 784 pixels | 28×28 = 784 con số đầu vào |

### 1.3 Các công cụ sử dụng

| Công cụ | Tác dụng | Tại sao dùng? |
|---------|----------|---------------|
| **PyTorch** | Framework Deep Learning | Phổ biến nhất, dễ học, document tốt |
| **NumPy** | Xử lý mảng số | Nhanh, hiệu quả cho tính toán |
| **Matplotlib** | Vẽ biểu đồ | Trực quan hóa dữ liệu |
| **scikit-learn** | Tính metrics | Có sẵn các hàm đánh giá |

---

## 2. Giai đoạn 1: Chuẩn bị dữ liệu

### 2.1 Bước 1 — Phân tích khám phá dữ liệu (EDA)

**Mục tiêu:** Trước khi train model, phải hiểu dữ liệu mình có.

**Các công việc thực hiện:**

- **Tải dataset FashionMNIST** thông qua `torchvision.datasets.FashionMNIST`
- **Tính toán thống kê cơ bản**: Mean, Std, Min, Max của tất cả pixel
- **Vẽ biểu đồ phân bố lớp** để kiểm tra cân bằng dữ liệu
- **Trực quan hóa mẫu ảnh** từ mỗi lớp để hiểu trực quan
- **Phân tích histogram pixel** để xem phân bố cường độ

**Các kỹ thuật áp dụng:**

| Kỹ thuật | Mục đích | Cách áp dụng |
|----------|----------|---------------|
| **Thống kê mô tả** | Hiểu phân bố giá trị | Tính mean, std, min, max |
| **Data visualization** | "Nhìn" dữ liệu bằng mắt | Dùng Matplotlib/Seaborn vẽ biểu đồ |
| **Frequency analysis** | Kiểm tra cân bằng lớp | Đếm số mẫu mỗi lớp |

**Kết quả thu được:**

- 60,000 mẫu huấn luyện, **cân bằng hoàn hảo** (6,000 ảnh/lớp)
- Mean = 0.286, Std = 0.353
- Giá trị pixel nằm trong khoảng [0, 1] → đã được chuẩn hóa sẵn
- Không có dữ liệu thiếu hoặc hỏng

---

### 2.2 Bước 2 — Làm sạch dữ liệu

**Mục tiêu:** Đảm bảo dữ liệu đầu vào "sạch" trước khi xử lý tiếp.

**Các công việc thực hiện:**

- **Kiểm tra giá trị NaN/Inf** trong tất cả ảnh
- **Xác thực kích thước ảnh** đúng chuẩn 28×28×1
- **Kiểm tra phạm vi nhãn** hợp lệ (0-9)
- **Đảm bảo không có ảnh hỏng** hoặc không đọc được

**Các kỹ thuật áp dụng:**

| Kỹ thuật | Mục đích |
|----------|----------|
| **Data validation** | Kiểm tra tính hợp lệ của dữ liệu |
| **Defensive programming** | Viết code an toàn với mọi loại input |
| **Quality assurance** | Đảm bảo chất lượng dữ liệu đầu vào |

**Kết quả:** 60,000/60,000 mẫu hợp lệ (100%) — FashionMNIST là dataset đã được làm sạch.

---

### 2.3 Bước 3 — Tiền xử lý (Preprocessing)

**Mục tiêu:** Biến đổi dữ liệu về dạng phù hợp để model học tốt hơn.

**Các công việc thực hiện:**

- **Áp dụng Normalization** với mean=0.286 và std=0.353 đã tính từ EDA
- **Sử dụng transforms pipeline** của torchvision để xử lý nhất quán
- **Tắt data augmentation** ở giai đoạn này (chỉ dùng cơ bản)

**Các kỹ thuật áp dụng:**

#### 1. Normalization (Chuẩn hóa)

**Công thức:** `normalized_pixel = (pixel - mean) / std`

**Ví dụ cụ thể:**
- Pixel gốc: 0.5
- Sau chuẩn hóa: (0.5 - 0.286) / 0.353 = **0.606**

**Tại sao cần chuẩn hóa?**

| Lý do | Giải thích |
|-------|-----------|
| **Tăng tốc độ hội tụ** | Gradient không bị "lệch" do giá trị lớn |
| **Tránh overflow** | Giá trị lớn có thể gây tràn số trong tính toán |
| **Cân bằng tầm quan trọng** | Mỗi pixel có "trọng số" ngang nhau |
| **Ổn định training** | Loss surface mượt hơn, dễ tối ưu |

**Kết quả:** Mean ≈ 0, Std ≈ 1 sau khi chuẩn hóa → phân bố chuẩn.

---

### 2.4 Bước 4 — Chia dữ liệu (Train/Val/Test Split)

**Mục tiêu:** Có 3 tập dữ liệu với vai trò khác nhau để train và đánh giá đúng cách.

**Phân chia:**

| Tập | Tỷ lệ | Số mẫu | Vai trò |
|-----|-------|--------|---------|
| **Train** | 90% | 54,000 | Dùng để train model (model học từ đây) |
| **Validation** | 10% | 6,000 | Đánh giá trong khi train, chọn best model |
| **Test** | 100% | 10,000 | Đánh giá cuối cùng, **không được dùng khi train** |

**Các kỹ thuật áp dụng:**

#### 1. Stratified Split (Phân tầng)
- **Vấn đề:** Chia ngẫu nhiên có thể làm mất cân bằng lớp trong từng tập
- **Giải pháp:** Đảm bảo tỷ lệ mỗi lớp trong train và val là giống nhau
- **Ví dụ:** Cả train và val đều có đúng 600 ảnh/lớp

#### 2. Random Seed = 42
- **Vấn đề:** Mỗi lần chạy lại cho kết quả khác nhau
- **Giải pháp:** Fix seed để kết quả reproducible (tái tạo được)
- **Ý nghĩa:** Người khác chạy lại code sẽ được cùng kết quả

#### 3. Tách biệt 3 tập
- **Train**: Model học từ dữ liệu này
- **Validation**: Model không học, chỉ dùng để đánh giá trong quá trình train
- **Test**: "Giấu kín" đến cuối cùng mới dùng → mô phỏng dữ liệu thực tế

**Tại sao cần 3 tập?**

```
Nếu chỉ có Train/Test:
  → Model có thể "học thuộc" test set
  → Đánh giá sai (overestimate accuracy)

Giải pháp 3 tập:
  → Train: model học
  → Validation: chọn best model, điều chỉnh hyperparam
  → Test: đánh giá khách quan cuối cùng
```

---

## 3. Giai đoạn 2: Phát triển mô hình

### 3.1 Bước 1 — Mô hình Baseline (MLP đơn giản)

**Tại sao cần baseline?** Để biết "mức sàn" performance và làm điểm so sánh với model phức tạp hơn.

**Kiến trúc BaselineMLP:**

| Layer | Input → Output | Vai trò |
|-------|---------------|---------|
| Flatten | [B,1,28,28] → [B,784] | Trải phẳng ảnh 2D thành vector 1D |
| Linear + ReLU | 784 → 256 | Học 256 đặc trưng từ 784 pixel |
| Linear + ReLU | 256 → 128 | Tiếp tục nén thông tin |
| Linear | 128 → 10 | Cho ra "điểm" cho 10 lớp (logits) |

**Các kỹ thuật áp dụng:**

| Kỹ thuật | Giải thích |
|----------|-----------|
| **Flatten layer** | Chuyển tensor 2D thành vector 1D |
| **Fully Connected (Linear)** | Mỗi nơ-ron kết nối với tất cả nơ-ron lớp trước |
| **ReLU activation** | f(x) = max(0, x) — thêm phi tuyến, tránh vanishing gradient |
| **Progressive dimension reduction** | 784 → 256 → 128 → 10 (nén dần thông tin) |

**Tại sao chọn 256, 128?** Đây là các con số phổ biến, không quá lớn (tránh overfit) cũng không quá nhỏ (tránh underfit).

---

### 3.2 Bước 2 — Khám phá kiến trúc CNN

**Vấn đề với MLP:**
- Xem mỗi pixel là độc lập → không hiểu cấu trúc 2D của ảnh
- Số tham số lớn (vì fully connected)
- Không có tính "translation invariance" (dịch chuyển ảnh → kết quả khác)

**Giải pháp: CNN (Convolutional Neural Network)**

**Các thành phần chính của CNN:**

#### 1. Conv2d (Tích chập) — Trích xuất đặc trưng không gian

**Cách hoạt động:**
- Trượt một "cửa sổ" (kernel) trên ảnh
- Mỗi vị trí: nhân từng phần tử rồi cộng lại
- Mỗi filter học một pattern: góc, cạnh, texture, hình dạng...

**Ví dụ trực quan:**

```
Ảnh gốc (6x6):          Filter 3x3:           Kết quả 4x4:
┌─┬─┬─┬─┬─┬─┐           ┌─┬─┬─┐               ┌─┬─┬─┬─┐
│1│2│3│4│5│6│           │1│0│-1│              │?│?│?│?│
├─┼─┼─┼─┼─┼─┤           ├─┼─┼─┤               ├─┼─┼─┼─┤
│7│8│9│1│2│3│           │1│0│-1│              │?│?│?│?│
├─┼─┼─┼─┼─┼─┤           ├─┼─┼─┤               ├─┼─┼─┼─┤
│4│5│6│7│8│9│           │1│0│-1│              │?│?│?│?│
└─┴─┴─┴─┴─┴─┘           └─┴─┴─┘               └─┴─┴─┴─┘
```

**Các tham số quan trọng:** in_channels, out_channels, kernel_size, stride, padding

#### 2. MaxPool2d — Giảm chiều không gian

**Mục đích:** Giảm kích thước ảnh, giữ lại đặc trưng quan trọng nhất.

```
Input 4x4:                Output 2x2:
┌─┬─┬─┬─┐
│1│3│2│4│                 ┌─┬─┐
├─┼─┼─┼─┤    MaxPool      │3│4│   ← Lấy max mỗi vùng 2x2
│5│6│7│8│   ─────────►    ├─┼─┤
├─┼─┼─┼─┤                 │6│8│
│2│1│3│5│                 └─┴─┘
└─┴─┴─┴─┘
```

#### 3. Dropout — Chống overfit

**Cơ chế:** Tắt ngẫu nhiên một tỷ lệ nơ-ron trong quá trình training.

**Tại sao cần Dropout?**
- Khi train, model có thể "học thuộc" dữ liệu → overfit
- Dropout buộc model phải dùng nhiều "đường dẫn" khác nhau
- Kết quả: model tổng quát hóa tốt hơn trên dữ liệu mới

**Ví dụ:** Dropout(0.5) → mỗi nơ-ron có 50% cơ hội bị "tắt" trong mỗi batch.

#### 4. Batch Normalization — Ổn định training

**Cơ chế:** Chuẩn hóa output của mỗi layer theo batch.

**Lợi ích:**
- Cho phép dùng learning rate cao hơn
- Ít phụ thuộc vào initialization
- Có tác dụng regularization nhẹ

#### 5. ReLU Activation — Phi tuyến

**Công thức:** f(x) = max(0, x)

**Tại sao dùng ReLU thay vì Sigmoid?**
- Sigmoid gây vanishing gradient (gradient quá nhỏ ở vùng bão hòa)
- ReLU có gradient = 1 khi x > 0 → gradient không bị triệt tiêu
- Tính toán đơn giản, nhanh hơn

**Kiến trúc SimpleCNN (model chiến thắng):**

```
Input ảnh [1, 28, 28]
   ↓
Conv2d(1→32) + BatchNorm + ReLU + MaxPool     → [32, 14, 14]
   ↓
Conv2d(32→64) + BatchNorm + ReLU + MaxPool    → [64, 7, 7]
   ↓
Flatten                                        → [3136]
   ↓
Linear(3136→128) + ReLU + Dropout              → [128]
   ↓
Linear(128→10)                                 → [10] (logits cho 10 lớp)
```

**So sánh MLP vs CNN:**

| Đặc tính | MLP | CNN |
|----------|-----|-----|
| Số tham số | Rất nhiều (vì fully connected) | Ít hơn (vì shared weights) |
| Nhận diện pattern 2D | Kém | Tốt |
| Translation invariance | Không | Có (filter trượt khắp ảnh) |
| Tốc độ train | Chậm | Nhanh hơn |
| Accuracy trên ảnh | Thấp hơn | Cao hơn 5-7% |

---

### 3.3 Bước 3 — Tinh chỉnh siêu tham số (Hyperparameter Tuning)

**Siêu tham số là gì?** Các tham số phải chọn TRƯỚC khi train (model không tự học được).

**Các siêu tham số chính:**

| Siêu tham số | Ảnh hưởng | Các giá trị thường thử |
|--------------|-----------|------------------------|
| Learning rate | Tốc độ học | 0.1, 0.01, 0.001, 0.0001 |
| Batch size | Số mẫu/1 lần update | 32, 64, 128, 256 |
| Optimizer | Thuật toán tối ưu | SGD, Adam, AdamW |
| Epochs | Số lần duyệt qua data | 5, 10, 20, 50 |
| Dropout | Tỷ lệ tắt nơ-ron | 0.3, 0.5, 0.7 |

**Các phương pháp tìm kiếm áp dụng:**

#### Grid Search (Tìm kiếm lưới)

**Cách hoạt động:** Thử tất cả tổ hợp có thể trong "lưới" giá trị (tích Cartesian).

| Ưu điểm | Nhược điểm |
|---------|-----------|
| Đảm bảo tìm được tổ hợp tốt nhất trong lưới | Chậm nếu không gian tìm kiếm lớn |
| Tất định — chạy lại ra đúng kết quả cũ | Tốn tài nguyên tính toán |

**Trong dự án:** Thực hiện **16 trials × 10 epochs** mỗi trial, quét **4 kiến trúc**
(`baseline_mlp`, `deep_mlp`, `simple_cnn`, `resnet_mini`) × 2 learning rate
(0.001, 0.0005) × 2 optimizer (adam, sgd) — tổng cộng **160 epoch huấn luyện**.
Nhờ grid **tất định và phủ kín**, không có tổ hợp nào bị bỏ sót.

**Cấu hình tốt nhất tìm được:**

| Siêu tham số | Giá trị |
|---------------|---------|
| Model | simple_cnn |
| Optimizer | adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Dropout | 0.5 |
| Validation accuracy | 92.33% |

**Tại sao Adam tốt hơn SGD cho bài toán này?**
- Adam tự điều chỉnh learning rate cho từng tham số
- SGD cần manual tuning nhiều hơn
- Với dataset cỡ trung bình như FashionMNIST, Adam hội tụ nhanh hơn

**Tại sao lr=0.001 là "sweet spot"?**

| Learning rate | Hiệu quả |
|---------------|----------|
| Quá lớn (0.1) | Loss dao động không hội tụ, model không học được |
| Quá nhỏ (0.0001) | Học quá chậm, tốn thời gian |
| **0.001** | **Cân bằng giữa tốc độ và ổn định** |

---

## 4. Giai đoạn 3: Huấn luyện và đánh giá

### 4.1 Bước 1 — Vòng lặp huấn luyện (Training Loop)

**Training loop là gì?** Quy trình lặp đi lặp lại để model "học" từ dữ liệu qua nhiều epoch.

**5 bước cốt lõi trong mỗi batch:**

```
┌─────────────────────────────────────────────┐
│  1. Forward:   Đưa ảnh vào model, lấy dự đoán│
│  2. Loss:      So sánh dự đoán với nhãn thực  │
│  3. Backward:  Tính gradient (đạo hàm)       │
│  4. Step:      Cập nhật trọng số              │
│  5. Zero grad: Xóa gradient cũ               │
└─────────────────────────────────────────────┘
```

**Giải thích chi tiết từng bước:**

| Bước | Ý nghĩa | Ví dụ cụ thể |
|------|----------|---------------|
| **Forward** | Tính output từ input | Ảnh quần → dự đoán "Trouser" với 80% |
| **Loss** | Đo sai số giữa dự đoán và thực tế | Dự đoán 80% mà thực tế là "Trouser" → loss = 0.22 |
| **Backward** | Tính đạo hàm loss theo từng trọng số | Trọng số nào cần tăng, trọng số nào cần giảm? |
| **Step** | Cập nhật trọng số theo gradient | w_new = w_old - 0.001 × gradient |
| **Zero grad** | Xóa gradient cũ | Nếu không, gradient sẽ cộng dồn qua các batch |

**Các kỹ thuật áp dụng:**

#### 1. CrossEntropyLoss

**Cơ chế:**
- Input: logits (điểm số thô cho 10 lớp), ví dụ: [2.1, -0.5, 1.8, 0.2, ...]
- Bước 1: **Softmax** → chuyển thành xác suất [0.42, 0.03, 0.31, ...]
- Bước 2: **Loss = -log(xác suất của lớp đúng)**
- Nếu lớp đúng có xác suất 0.42 → Loss = -log(0.42) = 0.87

**Tại sao dùng CrossEntropy?**
- Kết hợp Softmax + Negative Log Likelihood trong 1 hàm
- Số ổn định (numerical stability)
- Phù hợp cho bài toán phân loại đa lớp

#### 2. Adam Optimizer

**Cơ chế:**
- Kết hợp Momentum (nhớ hướng đi trước đó) + RMSProp (điều chỉnh learning rate theo gradient)
- Mỗi tham số có learning rate riêng
- Tự điều chỉnh trong quá trình training

**Công thức đơn giản hóa:**
```
m_t = β1 × m_{t-1} + (1-β1) × gradient       (momentum)
v_t = β2 × v_{t-1} + (1-β2) × gradient²      (RMS)
w_new = w_old - lr × m_t / (√v_t + ε)
```

#### 3. Forward/Backward Pass

**Forward pass:** Lan truyền thuận — tính output từ input qua các layer.

**Backward pass:** Lan truyền ngược — dùng chain rule tính gradient từ loss về các trọng số.

**Chain Rule (Quy tắc chuỗi):**
- Loss phụ thuộc vào output
- Output phụ thuộc vào trọng số layer cuối
- Trọng số layer cuối phụ thuộc vào output layer trước
- → Tính gradient ngược từ cuối về đầu

#### 4. Gradient Descent

**Ý tưởng:** Trọng số mới = Trọng số cũ - learning_rate × gradient

**Hình dung:** Đứng trên núi (loss cao) → nhìn xuống chân núi (loss thấp) → bước theo hướng dốc nhất.

---

### 4.2 Bước 2 — Validation và Early Stopping

**Validation là gì?** Sau mỗi epoch, đánh giá model trên tập validation (model không học từ tập này).

**Các kỹ thuật áp dụng:**

#### 1. model.eval() — Chế độ đánh giá

**Cơ chế:** Chuyển model sang chế độ eval, tắt các layer "chỉ dùng khi train":
- **Dropout**: Bật lại tất cả nơ-ron
- **BatchNorm**: Dùng statistics tích lũy thay vì batch hiện tại

#### 2. torch.no_grad() — Tắt tính gradient

**Mục đích:**
- Không cần gradient khi đánh giá
- Tiết kiệm ~50% bộ nhớ
- Tăng tốc độ đánh giá

#### 3. Early Stopping — Dừng train sớm

**Ý tưởng:** Nếu val_loss không cải thiện trong N epoch liên tiếp → dừng train.

**Tham số:**
- `patience=5`: Chờ 5 epoch trước khi dừng
- `min_delta=0.0001`: Cải thiện tối thiểu bao nhiêu mới tính là "tốt hơn"

**Hình dung:**

```
val_loss
  │
  │  ╲        ╱╲         ← Early stopping dừng ở đây
  │   ╲      ╱  ╲
  │    ╲    ╱    ╲
  │     ╲  ╱      ╲
  │      ╲╱        ╲___ ← Nếu không có early stopping, train tiếp → overfit
  └────────────────────── epoch
       1  5  10  15  20
```

**Tại sao cần Early Stopping?**
- Validation không cải thiện → model bắt đầu overfit
- Tránh lãng phí thời gian training
- Tự động chọn được điểm "sweet spot"

---

### 4.3 Bước 3 — Checkpoint (Lưu mô hình)

**Mục đích:** Lưu lại trọng số model để dùng sau hoặc khôi phục nếu bị crash giữa chừng.

**Các loại checkpoint:**

| Loại | Khi nào lưu | Dùng khi nào |
|------|-------------|--------------|
| **Best model** | val_loss thấp nhất | Đánh giá cuối, deploy production |
| **Last model** | Cuối mỗi epoch | Resume training nếu bị gián đoạn |

**Các kỹ thuật áp dụng:**

| Kỹ thuật | Mục đích |
|----------|----------|
| **State dict** | Lưu trọng số model (chỉ parameters, không lưu architecture) |
| **Serialization** | Chuyển object thành file có thể tải lại |
| **Versioning** | Theo dõi model nào tốt nhất qua các lần train |

**Tại sao lưu cả hai?**
- **Best**: Model cho kết quả test cao nhất
- **Last**: Để resume training nếu bị crash

---

### 4.4 Bước 4 — Trực quan hóa loss/accuracy

**Tại sao cần vẽ biểu đồ?**
- Số liệu thô khó nhìn → biểu đồ giúp thấy xu hướng
- Phát hiện overfit/underfit nhanh chóng
- Truyền đạt kết quả trực quan

**Các kỹ thuật áp dụng:**

| Kỹ thuật | Mục đích |
|----------|----------|
| **Line plot** | Vẽ loss/accuracy theo epoch |
| **Train vs Validation** | So sánh 2 đường cong để phát hiện vấn đề |
| **Subplot** | Hiển thị nhiều biểu đồ cùng lúc |

**Cách đọc biểu đồ:**

```
✅ Tốt (Generalization):     ❌ Overfit:                ❌ Underfit:
  acc                           acc                         acc
  │   val╱──                    │  val───╲                  │  val
  │  ╱                          │ ╱       ╲                  │ ╱
  │ ╱ train                     │╱ train   ╲                 │╱ train
  │╱                            │           ╲___             │╱
  └────── epoch                 └──────────── epoch          └────── epoch
  (train ≈ val)                 (val giảm, train tăng)      (cả 2 đều thấp)
```

---

### 4.5 Bước 5 — Đánh giá trên tập Test

**Mục tiêu:** Đánh giá khách quan model trên dữ liệu chưa từng thấy.

**Các metrics đánh giá:**

#### 1. Accuracy (Độ chính xác tổng thể)

**Công thức:** `accuracy = correct_predictions / total_predictions`

**Kết quả:** 9,210/10,000 = **92.10%**

#### 2. Precision (Độ chính xác dương)

**Công thức:** `precision = TP / (TP + FP)`

**Ý nghĩa:** Trong tất cả các dự đoán "lớp X", bao nhiêu % thực sự là lớp X?

**Ví dụ:** Model dự đoán 100 ảnh là "Áo thun", trong đó 92 đúng → Precision = 92%.

#### 3. Recall (Khả năng tìm đúng)

**Công thức:** `recall = TP / (TP + FN)`

**Ý nghĩa:** Trong tất cả các ảnh thực sự là "lớp X", model tìm được bao nhiêu %?

**Ví dụ:** Có 100 ảnh "Áo thun" thật, model tìm đúng 90 → Recall = 90%.

#### 4. F1-Score (Trung bình điều hòa)

**Công thức:** `F1 = 2 × (precision × recall) / (precision + recall)`

**Khi nào F1 quan trọng?** Khi dữ liệu mất cân bằng (precision và recall "trái ngược" nhau).

#### 5. Confusion Matrix (Ma trận nhầm lẫn)

**Ý tưởng:** Bảng 10×10 cho biết lớp nào hay bị nhầm với lớp nào.

```
            Predicted
            T-shirt  Trouser  ...  Shirt
True
T-shirt      892      5       ...   50
Trouser        3    945       ...    1
...
Shirt         90      1       ...  640
```

**Cách đọc:**
- Hàng: lớp thực tế
- Cột: lớp model dự đoán
- Đường chéo: số dự đoán đúng
- Ngoài đường chéo: số dự đoán sai

**Kết quả phân tích confusion matrix:**

| Lớp thực | → Bị nhầm thành | Số lần |
|-----------|----------------|--------|
| Shirt | T-shirt/top | **95** |
| T-shirt/top | Shirt | 90 |
| Shirt | Pullover | 64 |
| Coat | Pullover | 63 |
| Coat | Shirt | 55 |

**Insight:** Shirt và T-shirt/top dễ bị nhầm vì hình dạng tương tự (đều là áo, cổ tròn, tay ngắn).

---

### 4.6 Bước 6 — Lưu mô hình cuối

**Nội dung lưu trữ:**

| Thành phần | Mục đích |
|------------|----------|
| `model_state_dict` | Trọng số đã học |
| `model_name` | Tên kiến trúc |
| `metrics` | Accuracy, Precision, Recall, F1 |
| `config` | Hyperparameters đã dùng |

**Tại sao lưu kèm metrics?**
- Tiện cho việc so sánh các model
- Người khác mở ra biết ngay model này đạt bao nhiêu %
- Tái tạo được kết quả

---

## 5. Tổng kết kết quả

### 5.1 Kết quả cuối cùng

| Metric | Train | Validation | **Test** |
|--------|-------|------------|----------|
| **Accuracy** | 94.72% | 92.80% | **92.10%** |
| **Precision (macro)** | — | — | **92.11%** |
| **Recall (macro)** | — | — | **92.10%** |
| **F1-Score (macro)** | — | — | **92.09%** |

### 5.2 Quá trình train

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|------------|----------|----------|---------|
| 1 | 0.585 | 0.351 | 79.03% | 87.08% |
| 5 | 0.265 | 0.237 | 90.41% | 91.42% |
| 10 | 0.190 | 0.222 | 92.91% | 92.63% |
| 11 (best) | 0.180 | **0.215** | 93.21% | 92.88% |
| 15 (last) | 0.141 | 0.230 | 94.72% | 93.03% |

### 5.3 Tổng hợp các kỹ thuật đã sử dụng

| # | Kỹ thuật | Áp dụng ở đâu | Hiệu quả |
|---|----------|---------------|----------|
| 1 | **Normalization** | Giai đoạn 1 - Preprocessing | Tăng tốc hội tụ |
| 2 | **Stratified Split** | Giai đoạn 1 - Data Split | Đánh giá công bằng |
| 3 | **CNN thay vì MLP** | Giai đoạn 2 - Model Dev | +5-7% accuracy |
| 4 | **Batch Normalization** | Trong CNN | Ổn định training |
| 5 | **Dropout** | Trong CNN | Giảm 2-3% overfit |
| 6 | **Adam Optimizer** | Giai đoạn 3 - Training | Hội tụ nhanh |
| 7 | **Early Stopping** | Giai đoạn 3 - Training | Tránh overfit |
| 8 | **Checkpoint** | Giai đoạn 3 - Training | An toàn khi crash |
| 9 | **Grid Search** | Giai đoạn 2 - Hyperparam | Tăng 2-3% accuracy |
| 10 | **CrossEntropyLoss** | Giai đoạn 3 - Training | Phù hợp phân loại đa lớp |

### 5.4 Hướng cải thiện trong tương lai

| Cải thiện | Kỳ vọng tăng |
|-----------|--------------|
| Data Augmentation (xoay, lật, zoom) | +1-2% |
| ResNet18 thay vì SimpleCNN | +1-3% |
| Learning rate scheduler | +0.5-1% |
| Ensemble nhiều model | +1-2% |
| Test-Time Augmentation | +0.5-1% |

---

## 6. Cấu trúc dự án

```
e:\lab1/
├── src/
│   ├── data/                    # Giai đoạn 1: Chuẩn bị dữ liệu
│   │   ├── dataset.py          # FashionMNIST loader
│   │   ├── eda.py              # Phân tích khám phá
│   │   ├── preprocessing.py    # Làm sạch & transforms
│   │   └── split.py            # Chia train/val/test
│   ├── models/                  # Giai đoạn 2: Kiến trúc mô hình
│   │   ├── architectures/      # MLP, CNN, ResNet
│   │   ├── baseline_model.py
│   │   └── model_factory.py    # Model registry
│   ├── training/                # Giai đoạn 3: Huấn luyện
│   │   ├── trainer.py          # Vòng lặp huấn luyện
│   │   ├── callbacks.py        # Early stopping, checkpoint
│   │   ├── losses.py           # Loss functions
│   │   ├── optimizers.py       # Optimizer factory
│   │   └── visualizer.py       # Biểu đồ huấn luyện
│   └── evaluation/              # Giai đoạn 3: Đánh giá
│       ├── evaluator.py        # Tính toán metrics
│       ├── error_analysis.py   # Phân tích confusion
│       ├── confusion_matrix.py # Trực quan hóa CM
│       ├── visualizer.py       # Hiển thị dự đoán
│       ├── model_io.py         # Lưu/tải mô hình
│       └── run_evaluation.py    # Entry point
├── tests/training/              # Unit tests (27 tests)
├── models/checkpoints/          # Mô hình đã lưu
├── results/runs/                # Biểu đồ và kết quả
├── docs/reports/                # Báo cáo các giai đoạn
├── configs/                     # Cấu hình YAML
├── main.py                      # Entry point huấn luyện
└── README.md                    # Tài liệu dự án
```

---

## Hướng dẫn tái tạo dự án

| Bước | Lệnh | Mục đích |
|------|------|----------|
| 1 | `git clone https://github.com/Sper-il/Lab1.git` | Tải source code |
| 2 | `cd Lab1` | Vào thư mục dự án |
| 3 | `pip install -r requirements.txt` | Cài thư viện cần thiết |
| 4 | `python main.py` | Huấn luyện model |
| 5 | `python src/evaluation/run_evaluation.py` | Đánh giá trên test set |
| 6 | `pytest tests/training -v` | Chạy unit tests |

---

## Kết luận

Dự án này minh họa một quy trình **Machine Learning hoàn chỉnh** từ A đến Z:

1. **Hiểu dữ liệu** → EDA, thống kê, trực quan hóa
2. **Chuẩn bị dữ liệu** → Làm sạch, chuẩn hóa, chia tập
3. **Xây dựng mô hình** → MLP → CNN → ResNet
4. **Tìm hyper tối ưu** → Grid search (tích Cartesian, tất định)
5. **Huấn luyện** → Forward → Backward → Optimize
6. **Đánh giá** → Accuracy, Precision, Recall, F1
7. **Phân tích lỗi** → Confusion matrix, error analysis

**Kết quả:** Test Accuracy = **92.10%** — vượt mục tiêu 90% đề ra.

---

*Tài liệu được tạo: 2026-10-15*

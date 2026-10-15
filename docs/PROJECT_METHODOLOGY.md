# FashionMNIST Classification — Hướng Dẫn Chi Tiết Từ A Đến Z

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

| Mã | Tên loại | Ví dụ |
|----|----------|-------|
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

**Cách làm:**

#### Bước 1.1: Tải dữ liệu

```python
from torchvision import datasets, transforms

# Tải FashionMNIST
train_data = datasets.FashionMNIST(
    root='./data/raw',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
```

**Giải thích:**
- `root='./data/raw'`: Lưu dữ liệu vào thư mục `data/raw`
- `train=True`: Lấy tập huấn luyện
- `download=True`: Tự động tải nếu chưa có
- `transforms.ToTensor()`: Chuyển ảnh từ dạng PIL Image sang Tensor (dạng số để máy tính xử lý)

#### Bước 1.2: Thống kê cơ bản

```python
import torch

# Lấy tất cả ảnh
all_images = torch.stack([img for img, _ in train_data])

# Tính toán
print(f"Kích thước ảnh: {all_images.shape}")  # [60000, 1, 28, 28]
print(f"Giá trị nhỏ nhất: {all_images.min()}")  # 0.0
print(f"Giá trị lớn nhất: {all_images.max()}")  # 1.0
print(f"Trung bình: {all_images.mean():.4f}")   # 0.2860
print(f"Độ lệch chuẩn: {all_images.std():.4f}") # 0.3530
```

**Tại sao quan trọng?**
- Biết được giá trị pixel nằm trong khoảng [0, 1] → đã được chuẩn hóa
- Mean và Std giúp ta biết cách chuẩn hóa dữ liệu mới

#### Bước 1.3: Vẽ biểu đồ phân bố lớp

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Đếm số ảnh mỗi lớp
labels = [label for _, label in train_data]
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 5))
sns.countplot(x=labels)
plt.title('Số lượng ảnh mỗi lớp')
plt.show()
```

**Kỹ thuật áp dụng:**
- **Data visualization**: Dùng biểu đồ để "nhìn" dữ liệu
- **Statistical analysis**: Tính mean, std, count để hiểu phân bố

**Kết quả:** 6,000 ảnh/lớp → **dữ liệu cân bằng hoàn hảo**, không cần xử lý mất cân bằng.

---

### 2.2 Bước 2 — Làm sạch dữ liệu

**Vấn đề:** Dữ liệu thực tế thường có lỗi (ảnh hỏng, nhãn sai, giá trị NaN).

**Cách làm:**

```python
import numpy as np

def clean_data(images, labels):
    """Kiểm tra và loại bỏ dữ liệu lỗi"""
    valid_indices = []
    
    for i in range(len(images)):
        img = images[i]
        label = labels[i]
        
        # Kiểm tra NaN/Inf
        if torch.isnan(img).any() or torch.isinf(img).any():
            continue
        
        # Kiểm tra kích thước ảnh
        if img.shape != (1, 28, 28):
            continue
        
        # Kiểm tra nhãn hợp lệ (0-9)
        if label < 0 or label > 9:
            continue
        
        valid_indices.append(i)
    
    return images[valid_indices], labels[valid_indices]
```

**Kỹ thuật áp dụng:**
- **Data validation**: Kiểm tra tính hợp lệ của dữ liệu
- **Defensive programming**: Viết code an toàn với mọi loại input

**Kết quả:** 60,000/60,000 mẫu hợp lệ (100%) — FashionMNIST là dataset sạch.

---

### 2.3 Bước 3 — Tiền xử lý (Preprocessing)

**Mục tiêu:** Biến đổi dữ liệu về dạng phù hợp để model học tốt hơn.

**Kỹ thuật 1: Chuẩn hóa (Normalization)**

**Vấn đề:** Giá trị pixel thô có phân bố không đều, làm model học chậm.

**Giải pháp:** Đưa về phân bố chuẩn (mean=0, std=1):

```python
# Công thức: normalized = (pixel - mean) / std
# mean = 0.2860, std = 0.3530

normalize = transforms.Normalize(
    mean=[0.2860],
    std=[0.3530]
)
```

**Ví dụ cụ thể:**
- Pixel gốc: 0.5
- Sau chuẩn hóa: (0.5 - 0.286) / 0.353 = **0.606**

**Tại sao cần chuẩn hóa?**
1. **Tăng tốc độ hội tụ**: Gradient không bị "lệch" do giá trị lớn
2. **Tránh overflow**: Giá trị lớn có thể gây tràn số
3. **Cân bằng tầm quan trọng**: Mỗi pixel có "trọng số" ngang nhau

**Pipeline đầy đủ:**

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.ToTensor(),              # PIL → Tensor [0, 1]
    transforms.Normalize(               # Chuẩn hóa về N(0, 1)
        mean=[0.2860],
        std=[0.3530]
    )
])
```

---

### 2.4 Bước 4 — Chia dữ liệu (Train/Val/Test Split)

**Mục tiêu:** Có 3 tập dữ liệu để train, đánh giá trong quá trình train, và test cuối cùng.

**Phân chia:**

| Tập | Tỷ lệ | Số mẫu | Vai trò |
|-----|-------|--------|---------|
| **Train** | 90% | 54,000 | Dùng để train model |
| **Validation** | 10% | 6,000 | Đánh giá trong khi train, chọn best model |
| **Test** | 100% | 10,000 | Đánh giá cuối cùng, **không được dùng khi train** |

**Code triển khai:**

```python
from sklearn.model_selection import train_test_split

# Chia train và val
train_idx, val_idx = train_test_split(
    np.arange(60000),
    test_size=0.1,           # 10% cho validation
    random_state=42,         # Đảm bảo reproducibility
    stratify=labels          # Giữ tỷ lệ các lớp
)
```

**Kỹ thuật áp dụng:**

#### 1. Stratified Split (Phân tầng)
- **Vấn đề:** Chia ngẫu nhiên có thể làm mất cân bằng lớp
- **Giải pháp:** Đảm bảo tỷ lệ mỗi lớp trong train và val là như nhau
- **Ví dụ:** Cả train và val đều có 600 ảnh/lớp

#### 2. Random Seed = 42
- **Vấn đề:** Mỗi lần chạy lại cho kết quả khác nhau
- **Giải pháp:** Fix seed để kết quả reproducible
- **42** là số "ma thuật" trong văn hóa pop (cũng là trong The Hitchhiker's Guide to the Galaxy)

#### 3. Train/Val/Test tách biệt
- **Train**: Model học từ dữ liệu này
- **Validation**: Model không học, chỉ dùng để đánh giá trong quá trình train
- **Test**: Giấu kín đến cuối cùng mới dùng → mô phỏng dữ liệu thực tế

**Tại sao cần 3 tập?**
- Nếu chỉ có train/test → model có thể "học thuộc" test set → đánh giá sai
- Validation giúp chọn best model trong quá trình train
- Test là "bài thi cuối kỳ" thực sự

---

## 3. Giai đoạn 2: Phát triển mô hình

### 3.1 Bước 1 — Mô hình Baseline (MLP đơn giản)

**Tại sao cần baseline?** Để biết "mức sàn" và so sánh với model phức tạp hơn.

**MLP là gì?** Multi-Layer Perceptron — mạng nơ-ron truyền thống, mỗi tầng kết nối đầy đủ.

**Code:**

```python
import torch.nn as nn

class BaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()         # 28x28 → 784
        self.fc1 = nn.Linear(784, 256)      # Ẩn 1
        self.fc2 = nn.Linear(256, 128)      # Ẩn 2
        self.fc3 = nn.Linear(128, 10)       # Đầu ra (10 lớp)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.flatten(x)    # [B, 1, 28, 28] → [B, 784]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)        # Logits, chưa softmax
        return x
```

**Giải thích từng thành phần:**

| Layer | Input → Output | Vai trò |
|-------|---------------|---------|
| `Flatten` | [B,1,28,28] → [B,784] | Trải phẳng ảnh 2D thành vector 1D |
| `Linear(784,256)` | Vector 784 → 256 | Học 256 đặc trưng từ 784 pixel |
| `ReLU` | f(x) = max(0, x) | Thêm phi tuyến (không dùng sigmoid vì vanishing gradient) |
| `Linear(256,128)` | 256 → 128 | Tiếp tục nén thông tin |
| `Linear(128,10)` | 128 → 10 | Cho ra "điểm" cho 10 lớp (logits) |

**Tại sao chọn 256, 128?** Đây là các con số phổ biến, không quá lớn (tránh overfit) cũng không quá nhỏ (tránh underfit).

---

### 3.2 Bước 2 — Khám phá kiến trúc CNN

**Vấn đề với MLP:** Xem mỗi pixel là độc lập, không hiểu cấu trúc 2D của ảnh.

**CNN là gì?** Convolutional Neural Network — mạng tích chập, học đặc trưng không gian.

**Các thành phần chính của CNN:**

#### Conv2d (Tích chập)

```python
nn.Conv2d(
    in_channels=1,      # Ảnh xám: 1 kênh
    out_channels=32,    # Học 32 đặc trưng
    kernel_size=3,      # Cửa sổ 3x3
    stride=1,           # Bước nhảy 1
    padding=1           # Padding để giữ kích thước
)
```

**Conv2d hoạt động thế nào?**
- Trượt một "cửa sổ" 3×3 trên ảnh
- Mỗi vị trí: nhân từng phần tử rồi cộng lại
- Mỗi filter học một pattern: góc, cạnh, texture...

**Ví dụ trực quan:**

```
Ảnh gốc (6x6):          Filter (3x3):         Kết quả (4x4):
┌─┬─┬─┬─┬─┬─┐          ┌─┬─┬─┐              ┌─┬─┬─┬─┐
│1│2│3│4│5│6│          │1│0│-1│             │?│?│?│?│
├─┼─┼─┼─┼─┼─┤          ├─┼─┼─┤             ├─┼─┼─┼─┤
│7│8│9│1│2│3│          │1│0│-1│             │?│?│?│?│
├─┼─┼─┼─┼─┼─┤          ├─┼─┼─┤             ├─┼─┼─┼─┤
│4│5│6│7│8│9│          │1│0│-1│             │?│?│?│?│
├─┼─┼─┼─┼─┼─┤          └─┴─┴─┘             ├─┼─┼─┼─┤
│1│2│3│4│5│6│                                │?│?│?│?│
├─┼─┼─┼─┼─┼─┤                                └─┴─┴─┴─┘
│7│8│9│1│2│3│
├─┼─┼─┼─┼─┼─┤
│4│5│6│7│8│9│
└─┴─┴─┴─┴─┴─┘
```

**Tại sao CNN tốt hơn MLP cho ảnh?**

| Đặc tính | MLP | CNN |
|----------|-----|-----|
| Số tham số | Nhiều (vì fully connected) | Ít hơn (vì shared weights) |
| Nhận diện pattern 2D | Kém | Tốt |
| Translation invariance | Không | Có (filter trượt khắp ảnh) |
| Tốc độ train | Chậm | Nhanh hơn |

#### MaxPool2d (Lấy mẫu xuống)

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

**Mục đích:** Giảm kích thước ảnh, giữ lại đặc trưng quan trọng nhất.

```
Input (4x4):              Output (2x2):
┌─┬─┬─┬─┐
│1│3│2│4│                ┌─┬─┐
├─┼─┼─┼─┤     MaxPool    │3│4│   ← Max mỗi vùng 2x2
│5│6│7│8│    ────────►   ├─┼─┤
├─┼─┼─┼─┤                │6│8│
│2│1│3│5│                └─┴─┘
└─┴─┴─┴─┘
```

**Dropout (chống overfit):**

```python
nn.Dropout(p=0.5)  # Tắt ngẫu nhiên 50% nơ-ron khi train
```

**Tại sao cần Dropout?**
- Khi train, model có thể "học thuộc" dữ liệu → overfit
- Dropout buộc model phải dùng nhiều đường dẫn khác nhau
- Kết quả: model tổng quát hóa tốt hơn

**Code SimpleCNN đầy đủ:**

```python
class SimpleCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        # Block 1: Ảnh xám → 32 features
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 → 14x14
        
        # Block 2: 32 → 64 features
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 → 7x7
        
        # Classifier
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
```

**Giải thích BatchNorm:** Chuẩn hóa output mỗi batch → ổn định quá trình train, cho phép dùng learning rate cao hơn.

---

### 3.3 Bước 3 — Tinh chỉnh siêu tham số (Hyperparameter Tuning)

**Siêu tham số là gì?** Các tham số ta phải chọn TRƯỚC khi train (model không tự học).

**Các siêu tham số chính:**

| Siêu tham số | Ảnh hưởng | Thường thử |
|--------------|-----------|-----------|
| Learning rate | Tốc độ học | 0.1, 0.01, 0.001, 0.0001 |
| Batch size | Số mẫu/1 lần update | 32, 64, 128, 256 |
| Optimizer | Thuật toán tối ưu | SGD, Adam, AdamW |
| Epochs | Số lần duyệt qua data | 5, 10, 20, 50 |
| Dropout | Tỷ lệ tắt nơ-ron | 0.3, 0.5, 0.7 |

**Phương pháp 1: Grid Search (Tìm kiếm lưới)**

```python
# Thử tất cả tổ hợp
learning_rates = [0.001, 0.0001]
optimizers     = ['adam', 'sgd']
batch_sizes    = [64, 128]

for lr in learning_rates:
    for opt in optimizers:
        for bs in batch_sizes:
            # Train và đánh giá
            score = train_and_evaluate(lr, opt, bs)
            print(f"lr={lr}, opt={opt}, bs={bs} → {score}")
```

**Ưu điểm:** Đảm bảo tìm được tổ hợp tốt nhất trong lưới.
**Nhược điểm:** Chậm nếu không gian tìm kiếm lớn (8 trials trong dự án này).

**Phương pháp 2: Random Search**

```python
# Chọn ngẫu nhiên tổ hợp
for trial in range(8):
    lr = random.choice([0.001, 0.0001, 0.01])
    opt = random.choice(['adam', 'sgd'])
    bs = random.choice([64, 128])
    
    score = train_and_evaluate(lr, opt, bs)
```

**Ưu điểm:** Nhanh hơn, vẫn có cơ hội tìm được cấu hình tốt.
**Nhược điểm:** Không đảm bảo tìm được tối ưu.

**Kết quả tìm được:**

| Cấu hình | Validation Accuracy |
|----------|---------------------|
| simple_cnn + adam + lr=0.001 + bs=128 + dropout=0.5 | **90.53%** ✅ |
| simple_cnn + sgd + lr=0.001 | 87.21% |
| resnet_mini + adam + lr=0.0001 | 89.15% |
| ... | ... |

**Tại sao Adam tốt hơn SGD?**
- Adam tự điều chỉnh learning rate cho từng tham số
- SGD cần manual tuning nhiều hơn
- Với dataset nhỏ như FashionMNIST, Adam hội tụ nhanh hơn

**Tại sao lr=0.001?**
- Quá lớn (0.1): loss dao động không hội tụ
- Quá nhỏ (0.0001): học quá chậm
- 0.001 là "sweet spot" cho Adam với bài toán này

---

## 4. Giai đoạn 3: Huấn luyện và đánh giá

### 4.1 Bước 1 — Vòng lặp huấn luyện (Training Loop)

**Training loop là gì?** Quy trình lặp đi lặp lại để model "học" từ dữ liệu.

**5 bước cốt lõi:**

```
┌─────────────────────────────────────────────┐
│  1. Forward:   Đưa ảnh vào model, lấy dự đoán│
│  2. Loss:      So sánh dự đoán với nhãn thực  │
│  3. Backward:  Tính gradient (đạo hàm)       │
│  4. Step:      Cập nhật trọng số              │
│  5. Zero grad: Xóa gradient cũ               │
└─────────────────────────────────────────────┘
```

**Code chi tiết:**

```python
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()  # Chuyển sang chế độ train
    total_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        # Bước 1: Forward pass
        optimizer.zero_grad()    # Xóa gradient cũ
        output = model(data)      # Dự đoán
        
        # Bước 2: Tính loss
        loss = criterion(output, target)
        
        # Bước 3: Backward pass
        loss.backward()           # Tính gradient
        
        # Bước 4: Cập nhật trọng số
        optimizer.step()          # w = w - lr * gradient
        
        # Thống kê
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
    
    return total_loss / len(loader), correct / len(loader.dataset)
```

**Giải thích chi tiết:**

| Bước | Ý nghĩa | Ví dụ |
|------|----------|-------|
| `forward` | Tính output từ input | Ảnh quần → dự đoán "Trouser" với 80% |
| `loss` | Đo sai số | Dự đoán 80% mà thực tế là "Trouser" → loss = 0.22 |
| `backward` | Tính "đạo hàm" của loss theo từng trọng số | Trọng số nào cần tăng/giảm? |
| `step` | Cập nhật trọng số theo gradient | w_new = w_old - 0.001 * gradient |
| `zero_grad` | Xóa gradient cũ | Nếu không, gradient sẽ cộng dồn qua các batch |

**Hàm Loss — CrossEntropyLoss:**

```python
criterion = nn.CrossEntropyLoss()
```

**CrossEntropy hoạt động thế nào?**
- Input: logits (điểm số thô cho 10 lớp), ví dụ: [2.1, -0.5, 1.8, 0.2, 0.1, -1.2, 1.5, 0.3, -0.8, 0.7]
- Bước 1: Softmax → xác suất [0.42, 0.03, 0.31, 0.06, 0.06, 0.02, 0.23, 0.07, 0.02, 0.10]
- Bước 2: Loss = -log(xác suất của lớp đúng)
- Nếu đúng là lớp 0: Loss = -log(0.42) = 0.87

**Tại sao dùng CrossEntropy?**
- Kết hợp Softmax + Negative Log Likelihood
- Số ổn định (numerical stability)
- Phù hợp cho bài toán phân loại đa lớp

---

### 4.2 Bước 2 — Validation và Early Stopping

**Validation là gì?** Sau mỗi epoch, đánh giá model trên tập validation (model không học từ tập này).

**Code:**

```python
def validate(model, loader, criterion, device):
    model.eval()  # Chuyển sang chế độ eval (tắt dropout)
    val_loss = 0
    correct = 0
    
    with torch.no_grad():  # Không tính gradient → tiết kiệm bộ nhớ
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    return val_loss / len(loader), correct / len(loader.dataset)
```

**Tại sao cần `model.eval()` và `torch.no_grad()`?**
- `eval()`: Tắt dropout và batchnorm update → kết quả ổn định
- `no_grad()`: Không tính gradient → tiết kiệm ~50% bộ nhớ và nhanh hơn

**Early Stopping — kỹ thuật chống overfit:**

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            # Có cải thiện → reset counter
            self.best_loss = val_loss
            self.counter = 0
            return False  # Chưa dừng
        else:
            # Không cải thiện → tăng counter
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Dừng train
            return False
```

**Logic:**
- Nếu val_loss không giảm trong 5 epoch liên tiếp → dừng train
- Lý do: validation không cải thiện nghĩa là model bắt đầu overfit

**Hình dung:**

```
val_loss
  │
  │  ╲        ╱╲         ← Early stopping dừng ở đây
  │   ╲      ╱  ╲
  │    ╲    ╱    ╲
  │     ╲  ╱      ╲
  │      ╲╱        ╲___ ← Nếu không có early stopping, sẽ tiếp tục train
  └────────────────────── epoch
       1  5  10  15  20
```

---

### 4.3 Bước 3 — Checkpoint (Lưu mô hình)

**Mục đích:** Lưu lại trọng số model để dùng sau hoặc khôi phục nếu crash.

**Code:**

```python
def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, path)

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_loss']
```

**Best vs Last checkpoint:**

| Loại | Khi nào lưu | Dùng khi nào |
|------|-------------|--------------|
| `best_model.pt` | val_loss thấp nhất | Đánh giá cuối, deploy |
| `last_model.pt` | Cuối mỗi epoch | Resume training |

**Tại sao lưu cả hai?**
- Best: model tốt nhất (cho kết quả test cao nhất)
- Last: để resume training nếu bị gián đoạn

---

### 4.4 Bước 4 — Trực quan hóa loss/accuracy

**Tại sao cần vẽ biểu đồ?**
- Số liệu thô khó nhìn → biểu đồ giúp thấy xu hướng
- Phát hiện overfit/underfit nhanh chóng

**Code:**

```python
import matplotlib.pyplot as plt

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Loss theo epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_title('Accuracy theo epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('results/runs/training_curves.png')
    plt.show()
```

**Cách đọc biểu đồ:**

```
✅ Tốt:               ❌ Overfit:           ❌ Underfit:
  acc                     acc                    acc
  │   val╱──              │  val───╲            │  val
  │  ╱                    │ ╱       ╲            │ ╱
  │ ╱ train               │╱ train   ╲          │╱ train
  │╱                      │           ╲___      │╱
  └────── epoch           └──────────── epoch   └────── epoch
```

---

### 4.5 Bước 5 — Đánh giá trên tập Test

**Các metrics chính:**

#### Accuracy (Độ chính xác tổng thể)

```python
accuracy = correct_predictions / total_predictions
```

**Ví dụ:** 9,210/10,000 = **92.10%**

#### Precision & Recall (Per Class)

```python
# True Positive: dự đoán đúng là "Áo thun" và thực tế là "Áo thun"
# False Positive: dự đoán là "Áo thun" nhưng thực tế không phải
# False Negative: thực tế là "Áo thun" nhưng dự đoán sai

precision = TP / (TP + FP)  # Trong các dự đoán "Áo thun", bao nhiêu % đúng?
recall    = TP / (TP + FN)  # Trong các "Áo thun" thực tế, tìm được bao nhiêu %?
```

**Ví dụ dễ hiểu:**
- Precision cao: Model "cẩn thận", chỉ dự đoán khi chắc chắn
- Recall cao: Model "thoáng", không bỏ sót trường hợp nào

#### F1-Score (Trung bình điều hòa)

```python
f1 = 2 * (precision * recall) / (precision + recall)
```

**Khi nào F1 quan trọng?** Khi dữ liệu mất cân bằng (precision và recall "trái ngược" nhau).

#### Confusion Matrix (Ma trận nhầm lẫn)

**Ý tưởng:** Bảng 10×10 cho biết lớp nào hay bị nhầm với lớp nào.

```
         Predicted
         T-shirt  Trouser  Pullover  ...  Shirt
True
T-shirt    892      5        10            50
Trouser     3      945        2             1
Pullover    8       2       870           60
...
Shirt      90       1        64          640
```

**Cách đọc:**
- Hàng: lớp thực tế
- Cột: lớp model dự đoán
- Đường chéo: số dự đoán đúng
- Ngoài đường chéo: số dự đoán sai

**Kết quả confusion matrix của dự án:**

| Lớp thực | → Bị nhầm thành | Số lần |
|-----------|----------------|--------|
| Shirt | T-shirt/top | **95** |
| T-shirt/top | Shirt | 90 |
| Shirt | Pullover | 64 |
| Coat | Pullover | 63 |
| Coat | Shirt | 55 |

**Insight:** Shirt và T-shirt/top dễ nhầm vì hình dạng giống nhau (đều là áo, cổ tròn, tay ngắn).

---

### 4.6 Bước 6 — Lưu mô hình cuối

**Code:**

```python
# Lưu model với metrics
torch.save({
    'model_state_dict': model.state_dict(),
    'model_name': 'simple_cnn',
    'metrics': {
        'test_accuracy': 0.9210,
        'test_precision': 0.9211,
        'test_recall': 0.9210,
        'test_f1': 0.9209
    },
    'config': {
        'dropout': 0.5,
        'batch_size': 128
    }
}, 'models/checkpoints/final_model.pt')
```

**Tại sao lưu kèm metrics?**
- Tiện cho việc so sánh các model
- Người khác mở ra biết ngay model này đạt bao nhiêu %

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

### 5.3 Những kỹ thuật đã sử dụng (Tóm tắt)

| # | Kỹ thuật | Mục đích | Hiệu quả |
|---|----------|----------|----------|
| 1 | **Normalization** | Đưa về cùng scale | Tăng tốc hội tụ |
| 2 | **Stratified Split** | Giữ tỷ lệ lớp | Đánh giá công bằng |
| 3 | **CNN thay vì MLP** | Học pattern 2D | +5-7% accuracy |
| 4 | **Batch Normalization** | Ổn định train | Cho phép lr cao hơn |
| 5 | **Dropout** | Chống overfit | Giảm 2-3% overfit |
| 6 | **Adam Optimizer** | Tự điều chỉnh lr | Hội tụ nhanh |
| 7 | **Early Stopping** | Dừng đúng lúc | Tránh overfit |
| 8 | **Checkpoint** | Lưu model tốt | An toàn khi crash |
| 9 | **Grid Search** | Tìm hyper tối ưu | Tăng 2-3% accuracy |
| 10 | **Data Augmentation** | Tăng data (chưa dùng) | Có thể tăng thêm 1-2% |

### 5.4 Hướng cải thiện

| Cải thiện | Kỳ vọng |
|-----------|---------|
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
│       └── run_evalution.py    # Entry point
├── tests/training/              # Unit tests (27 tests)
├── models/checkpoints/          # Mô hình đã lưu
├── results/runs/                # Biểu đồ và kết quả
├── docs/reports/                # Báo cáo các giai đoạn
├── configs/                     # Cấu hình YAML
├── main.py                      # Entry point huấn luyện
└── README.md                    # Tài liệu dự án
```

---

## Hướng dẫn tái tạo

### Bước 1: Clone repository
```bash
git clone https://github.com/Sper-il/Lab1.git
cd Lab1
```

### Bước 2: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 3: Huấn luyện model
```bash
python main.py
```

### Bước 4: Đánh giá model
```bash
$env:PYTHONPATH = "."
python src/evaluation/run_evalution.py
```

### Bước 5: Chạy tests
```bash
pytest tests/training -v
```

---

## Kết luận

Dự án này minh họa một quy trình **Machine Learning hoàn chỉnh** từ A đến Z:

1. **Hiểu dữ liệu** → EDA, thống kê, trực quan hóa
2. **Chuẩn bị dữ liệu** → Làm sạch, chuẩn hóa, chia tập
3. **Xây dựng mô hình** → MLP → CNN → ResNet
4. **Tìm hyper tối ưu** → Grid/Random search
5. **Huấn luyện** → Forward → Backward → Optimize
6. **Đánh giá** → Accuracy, Precision, Recall, F1
7. **Phân tích lỗi** → Confusion matrix, error analysis

**Kết quả:** Test Accuracy = **92.10%** — vượt mục tiêu 90% đề ra.

---

*Tài liệu được tạo: 2026-10-15*

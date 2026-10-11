# Data Flow

Sơ đồ luồng dữ liệu sẽ được cập nhật sau khi M1 hoàn thành.

```
FashionMNIST (torchvision)
        ↓
    ToTensor + Normalize
        ↓
     DataLoader
        ↓
      FashionCNN
        ↓
   CrossEntropyLoss + Adam
        ↓
  save model + plot metrics
```
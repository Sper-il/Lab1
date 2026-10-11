# Phase 2 Plan - FashionMNIST PyTorch Classification

**Ngày tạo:** 2026-10-11  
**Cập nhật:** 2026-10-11 (chia lại cho 3 người, align với đề bài)  
**Phase:** 2 - Model Development, Training & Evaluation  
**Số lượng người:** 3 người  
**Dự kiến hoàn thành:** Sau Phase 1  
**Dataset:** FashionMNIST (60k train / 10k test, 28x28 grayscale, 10 classes)

---

## 📋 YÊU CẦU TỪ ĐỀ BÀI (`đề.txt`)

### 1. Learn (Nghiên cứu)
- PyTorch Tensors
- Datasets / DataLoaders
- Transforms
- Model Building
- Autograd
- Optimization
- Model Saving/Loading

### 2. Implement
- ✅ Load FashionMNIST dataset *(Phase 1 đã xong)*
- ✅ Apply transforms *(Phase 1 đã xong)*
- ⏳ **Build a neural network** *(Người 1)*
- ⏳ **Create a training loop** (forward, loss, backward, optimize) *(Người 2)*
- ⏳ **Evaluate model accuracy** *(Người 3)*
- ⏳ **Save and load the model** *(Người 3)*

### 3. Tasks
- ⏳ **Experiment with network/hyperparameters** *(Người 1 + Người 2)*
- ⏳ **Visualize loss** *(Người 2)*
- ⏳ **Display predicted vs actual images** *(Người 3)*

### 4. Deliver
- ✅ Python code *(đang build)*
- ⏳ **Brief report** *(3 reports của 3 người + 1 final report)*
- ⏳ **Loss graphs** *(Người 2)*
- ⏳ **Image displays** *(Người 3)*

### 5. Use
- PyTorch docs/tutorials
- scikit-learn (cho metrics)

---

## Tổng quan Workflow Phase 2

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  Build Network +    │───→│  Training Loop +    │───→│  Eval + Save/Load + │
│  Hyperparameter     │    │  Loss Visualization │    │  Image Display      │
│  Experiments        │    │                     │    │                     │
│   (Người 1)         │    │   (Người 2)         │    │   (Người 3)         │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
        │                          │                          ▲
        │    best model +          │    trained model +      │
        │    best params           │    loss history         │
        └──────────────────────────┴──────────────────────────┘
```

### Phân chia theo đề bài

| Nhóm | Người | Phụ trách | Files |
|------|-------|-----------|-------|
| **Network** | Người 1 | Build neural network + Experiment với architectures/hyperparameters | `src/models/` |
| **Training** | Người 2 | Training loop (forward, loss, backward, optimize) + Visualize loss | `src/training/` |
| **Evaluation** | Người 3 | Evaluate accuracy + Save/Load model + Display predicted vs actual | `src/evaluation/` |

---

## PHÂN CÔNG CÔNG VIỆC

---

## 👤 NGƯỜI 1: Build Network + Hyperparameter Experiments

### Nhiệm vụ (theo đề)
1. **Build a neural network** - Xây dựng neural network cho FashionMNIST
2. **Experiment with network/hyperparameters** - Thử nhiều architectures và hyperparameters

### File cần tạo/sửa

| File | Chức năng | Trạng thái |
|------|-----------|------------|
| `src/models/baseline_model.py` | **Chính** - Baseline MLP | Tạo mới |
| `src/models/architectures/mlp.py` | MLP architecture | Tạo mới |
| `src/models/architectures/simple_cnn.py` | Simple CNN | Tạo mới |
| `src/models/architectures/advanced_cnn.py` | Advanced CNN (ResNet-like) | Tạo mới |
| `src/models/architectures/__init__.py` | Init architectures module | Tạo mới |
| `src/models/model_factory.py` | Factory pattern để tạo models | Tạo mới |
| `src/models/model_selection.py` | So sánh các models | Tạo mới |
| `src/models/__init__.py` | Export tất cả models | Sửa thêm |
| `src/training/hyperparameter_tuning.py` | **Chính** - Hyperparameter search | Tạo mới |
| `src/training/hyperopt_scheduler.py` | Scheduler cho hyperparameter search | Tạo mới |
| `configs/hyperopt_config.yaml` | Search space definition | Tạo mới |
| `run_experiments.py` | Script chạy thử nghiệm (optional) | Tạo mới |

### Chi tiết implementation

#### 1.1. Baseline MLP

```python
# src/models/baseline_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineMLP(nn.Module):
    """Simple MLP for FashionMNIST baseline.
    
    The simplest neural network cho FashionMNIST - 1 hidden layer.
    Input: (batch, 1, 28, 28) → flatten → (batch, 784)
    Output: (batch, 10) class logits
    """
    
    def __init__(self, input_size=784, hidden_size=256, 
                 num_classes=10, dropout=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Input shape: (batch, 1, 28, 28)
        x = self.flatten(x)          # → (batch, 784)
        x = self.fc1(x)              # → (batch, 256)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)              # → (batch, 10)
        return x  # Logits (chưa softmax - dùng CrossEntropyLoss)
```

#### 1.2. Architectures khác

```python
# src/models/architectures/mlp.py - MLP sâu hơn

class DeepMLP(nn.Module):
    """Deeper MLP với 2-3 hidden layers."""
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], 
                 num_classes=10, dropout=0.3):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers += [
                nn.Linear(prev_size, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev_size = h
        layers.append(nn.Linear(prev_size, num_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


# src/models/architectures/simple_cnn.py - CNN cơ bản

class SimpleCNN(nn.Module):
    """Simple CNN: 2 conv blocks + FC."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Input: (batch, 1, 28, 28)
        x = self.pool(self.relu(self.conv1(x)))  # → (batch, 32, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))  # → (batch, 64, 7, 7)
        x = x.view(x.size(0), -1)                # → (batch, 3136)
        x = self.relu(self.fc1(x))               # → (batch, 128)
        x = self.dropout(x)
        x = self.fc2(x)                          # → (batch, 10)
        return x


# src/models/architectures/advanced_cnn.py - ResNet-like

class BasicBlock(nn.Module):
    """Residual block."""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetMini(nn.Module):
    """Mini ResNet for FashionMNIST."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
```

#### 1.3. Model Factory

```python
# src/models/model_factory.py

class ModelFactory:
    """Factory pattern để tạo models theo tên."""
    
    _models = {
        'baseline_mlp': BaselineMLP,
        'deep_mlp': DeepMLP,
        'simple_cnn': SimpleCNN,
        'resnet_mini': ResNetMini,
    }
    
    @classmethod
    def create(cls, name, **kwargs):
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls._models.keys())}")
        return cls._models[name](**kwargs)
    
    @classmethod
    def list_models(cls):
        return list(cls._models.keys())
```

#### 1.4. Hyperparameter Tuning

```python
# src/training/hyperparameter_tuning.py

class HyperparameterTuner:
    """Hyperparameter tuning cho FashionMNIST."""
    
    def __init__(self, model_name, train_loader, val_loader, device):
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
    
    def grid_search(self, param_grid):
        """Grid search - thử tất cả combinations."""
        # Ví dụ: lr=[0.001, 0.01], batch_size=[32, 64], ...
        # → 2*2*... combinations
        pass
    
    def random_search(self, param_distributions, n_trials=10):
        """Random search - thử random combinations."""
        # Faster than grid search
        pass
    
    def train_with_params(self, params):
        """Train model với specific hyperparameters, return val accuracy."""
        # 1. Create model
        # 2. Setup optimizer, criterion
        # 3. Train for N epochs
        # 4. Return val accuracy
        pass
```

### Search Space đề xuất

```yaml
# configs/hyperopt_config.yaml
model:
  - baseline_mlp
  - deep_mlp
  - simple_cnn
  - resnet_mini

learning_rate: [0.0001, 0.001, 0.01]
batch_size: [32, 64, 128]
optimizer: [adam, sgd]
weight_decay: [0.0, 0.0001, 0.001]
dropout: [0.2, 0.3, 0.5]
epochs: 10  # Quick test
```

### Requirements (theo đề - "Experiment with network/hyperparameters")

✅ Phải thử được nhiều architectures:
- Baseline MLP (đơn giản nhất)
- Deep MLP (sâu hơn)
- Simple CNN
- Advanced CNN (ResNet)

✅ Phải thử được nhiều hyperparameters:
- Learning rate
- Batch size
- Optimizer (Adam, SGD)
- Weight decay
- Dropout

### Test
- Forward pass test với input shape `(batch, 1, 28, 28)`
- Verify output shape `(batch, 10)`
- Train 1 epoch và verify loss decreases
- Grid search với 2-3 params, verify return results

### Output cho Người 2
- **Best model architecture** (chuyển cho Người 2 để train full)
- **Best hyperparameters** (chuyển cho Người 2 để train full)
- **Trained model ở 1 epoch** để verify integration

### Report cần viết (lưu vào `docs/reports/nguoi1_models_report.md`)

```markdown
# Network & Hyperparameter Report

## 1. Models Implemented
- BaselineMLP: ...
- DeepMLP: ...
- SimpleCNN: ...
- ResNetMini: ...

## 2. Hyperparameter Search Results
### Search Space
- ...

### Best Hyperparameters
| Parameter | Best Value |
|-----------|------------|
| model | ... |
| learning_rate | ... |
| batch_size | ... |
| optimizer | ... |
| dropout | ... |

### Search Results
| Trial | Model | LR | Batch | Val Acc |
|-------|-------|-----|-------|---------|

## 3. Selected Model
- Architecture: ...
- Hyperparameters: ...
- Expected accuracy: ...

## 4. Files Created
- List all files
```

---

## 👤 NGƯỜI 2: Training Loop + Loss Visualization

### Nhiệm vụ (theo đề)
1. **Create a training loop (forward, loss, backward, optimize)** - Training loop hoàn chỉnh
2. **Visualize loss** - Vẽ đồ thị loss

### File cần tạo/sửa

| File | Chức năng | Trạng thái |
|------|-----------|------------|
| `src/training/trainer.py` | **Chính** - Training loop | Tạo mới |
| `src/training/callbacks.py` | Callbacks (early stopping, checkpoint) | Tạo mới |
| `src/training/losses.py` | Custom loss functions | Tạo mới |
| `src/training/optimizers.py` | Optimizer setup | Tạo mới |
| `src/training/visualizer.py` | **Chính** - Visualize loss | Tạo mới |
| `src/training/__init__.py` | Export training modules | Sửa thêm |
| `main.py` | Main training script | Sửa |
| `run_training.py` | Script chạy training | Tạo mới |

### Chi tiết implementation

#### 2.1. Training Loop (forward, loss, backward, optimize)

```python
# src/training/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    """Training loop cho FashionMNIST classification.
    
    Implements: forward → loss → backward → optimize
    """
    
    def __init__(self, model, train_loader, val_loader, 
                 criterion, optimizer, device, callbacks=None):
        """
        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            criterion: Loss function (e.g., CrossEntropyLoss)
            optimizer: Optimizer (e.g., Adam)
            device: 'cpu' or 'cuda'
            callbacks: List of callbacks
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.callbacks = callbacks or []
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
    
    def train_epoch(self):
        """Train 1 epoch - implements forward, loss, backward, optimize."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Forward
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_loss = total_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self):
        """Validate 1 epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in self.val_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            
            total_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_loss = total_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs, early_stopping_patience=10):
        """Full training loop."""
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch()
            # Validate
            val_loss, val_acc = self.validate()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        return self.history
```

#### 2.2. Loss Visualization (theo đề - "Visualize loss")

```python
# src/training/visualizer.py

import matplotlib.pyplot as plt

class TrainingVisualizer:
    """Visualize training progress."""
    
    def __init__(self, save_dir='results/runs/'):
        self.save_dir = save_dir
    
    def plot_loss_curve(self, history, save_path=None):
        """Plot train vs val loss (yêu cầu của đề)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history['train_loss'], label='Train Loss', marker='o')
        ax.plot(history['val_loss'], label='Val Loss', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss - FashionMNIST')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Loss curve saved to {save_path}")
        plt.show()
    
    def plot_accuracy_curve(self, history, save_path=None):
        """Plot train vs val accuracy."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history['train_acc'], label='Train Accuracy', marker='o')
        ax.plot(history['val_acc'], label='Val Accuracy', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Training and Validation Accuracy - FashionMNIST')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
    
    def plot_combined(self, history, save_path=None):
        """Plot cả loss và accuracy cùng lúc."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(history['train_loss'], label='Train Loss', marker='o')
        ax1.plot(history['val_loss'], label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(history['train_acc'], label='Train Acc', marker='o')
        ax2.plot(history['val_acc'], label='Val Acc', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
```

#### 2.3. Main Script

```python
# main.py

import torch
import torch.nn as nn
from src.data.split import DataSplitManager
from src.models.model_factory import ModelFactory
from src.training.trainer import Trainer
from src.training.visualizer import TrainingVisualizer

def main():
    # Setup
    device = 'cpu'
    
    # Load data (Phase 1)
    data_manager = DataSplitManager()
    train_loader, val_loader, test_loader = data_manager.get_loaders(batch_size=64)
    
    # Create model (best from Người 1)
    model = ModelFactory.create('simple_cnn')
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, 
                     criterion, optimizer, device)
    history = trainer.train(num_epochs=20, early_stopping_patience=5)
    
    # Visualize (yêu cầu của đề)
    visualizer = TrainingVisualizer()
    visualizer.plot_combined(history, save_path='results/runs/loss_curve.png')
    
    # Save model (Người 3 sẽ load)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': 'simple_cnn',
        'history': history
    }, 'models/checkpoints/trained_model.pt')

if __name__ == '__main__':
    main()
```

### Requirements (theo đề)

✅ **Training loop** đầy đủ 4 bước:
- **Forward pass**: `output = model(data)`
- **Loss**: `loss = criterion(output, target)`
- **Backward**: `loss.backward()`
- **Optimize**: `optimizer.step()` (sau `optimizer.zero_grad()`)

✅ **Visualize loss**:
- Train loss curve
- Val loss curve
- Combined loss + accuracy plot
- Save to `results/runs/`

### Test
- Train 2-3 epochs với baseline model
- Verify forward → loss → backward → optimize flow
- Verify loss decreases
- Verify loss curve saved correctly
- Verify checkpoint saved

### Input từ Người 1
- Best model name (e.g., `'simple_cnn'`)
- Best hyperparameters (lr, batch_size, optimizer, etc.)

### Output cho Người 3
- Trained model checkpoint: `models/checkpoints/trained_model.pt`
- Training history (loss/accuracy arrays)
- Loss curve: `results/runs/loss_curve.png`

### Report cần viết (lưu vào `docs/reports/nguoi2_training_report.md`)

```markdown
# Training Report

## 1. Training Configuration
- Model: ...
- Batch size: ...
- Learning rate: ...
- Optimizer: ...
- Loss: ...
- Epochs: ...

## 2. Training Progress
| Epoch | Train Loss | Val Loss | Val Acc |
|-------|------------|----------|---------|

## 3. Loss Visualization
- File: results/runs/loss_curve.png
- Mô tả đồ thị: ...

## 4. Best Model
- Epoch: ...
- Val Accuracy: ...
- Checkpoint: models/checkpoints/trained_model.pt

## 5. Training Time
- Total: ...
- Per epoch: ...

## 6. Files Created
- List all files
```

---

## 👤 NGƯỜI 3: Evaluation + Save/Load + Image Display

### Nhiệm vụ (theo đề)
1. **Evaluate model accuracy** - Đánh giá accuracy
2. **Save and load the model** - Save và load model
3. **Display predicted vs actual images** - Hiển thị ảnh predicted vs actual

### File cần tạo/sửa

| File | Chức năng | Trạng thái |
|------|-----------|------------|
| `src/evaluation/evaluator.py` | **Chính** - Evaluate accuracy | Tạo mới |
| `src/evaluation/metrics.py` | Accuracy, Precision, Recall, F1 | Tạo mới |
| `src/evaluation/model_io.py` | **Chính** - Save/Load model | Tạo mới |
| `src/evaluation/visualizer.py` | **Chính** - Display predicted vs actual | Tạo mới |
| `src/evaluation/confusion_matrix.py` | Confusion matrix | Tạo mới |
| `src/evaluation/error_analysis.py` | Error analysis | Tạo mới |
| `src/evaluation/__init__.py` | Export evaluation modules | Sửa thêm |
| `run_evaluation.py` | Script chạy evaluation | Tạo mới |

### Chi tiết implementation

#### 3.1. Model Evaluation (theo đề - "Evaluate model accuracy")

```python
# src/evaluation/evaluator.py

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

class ModelEvaluator:
    """Evaluate trained model trên test set."""
    
    CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    def __init__(self, model, test_loader, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.device = device
    
    def predict(self):
        """Get all predictions trên test set."""
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                output = self.model(data)
                probs = torch.softmax(output, dim=1)
                _, predicted = output.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def evaluate(self):
        """Đánh giá model - trả về dict of metrics."""
        preds, labels, probs = self.predict()
        
        metrics = {
            'accuracy': accuracy_score(labels, preds) * 100,
            'precision_macro': precision_score(labels, preds, average='macro') * 100,
            'recall_macro': recall_score(labels, preds, average='macro') * 100,
            'f1_macro': f1_score(labels, preds, average='macro') * 100,
        }
        
        print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
        print(f"Precision (macro): {metrics['precision_macro']:.2f}%")
        print(f"Recall (macro): {metrics['recall_macro']:.2f}%")
        print(f"F1-Score (macro): {metrics['f1_macro']:.2f}%")
        
        return metrics, preds, labels, probs
```

#### 3.2. Save/Load Model (theo đề - "Save and load the model")

```python
# src/evaluation/model_io.py

import torch
import os

class ModelIO:
    """Save và load model."""
    
    @staticmethod
    def save_model(model, save_path, model_name=None, 
                   optimizer=None, history=None, metrics=None):
        """Save model to file.
        
        Args:
            model: PyTorch model
            save_path: Path to save (e.g., 'models/checkpoints/model.pt')
            model_name: Name of model architecture
            optimizer: Optional - to resume training
            history: Optional - training history
            metrics: Optional - evaluation metrics
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_name': model_name or model.__class__.__name__,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if history is not None:
            checkpoint['history'] = history
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")
    
    @staticmethod
    def load_model(model, load_path, device='cpu', 
                   optimizer=None, strict=True):
        """Load model from file.
        
        Args:
            model: Model instance (cần tạo trước với đúng architecture)
            load_path: Path to checkpoint
            device: 'cpu' or 'cuda'
            optimizer: Optional - to resume training
            strict: Whether to strictly enforce state dict matching
        
        Returns:
            model: Loaded model
            checkpoint: Full checkpoint dict
        """
        checkpoint = torch.load(load_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        model.to(device)
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Model loaded from {load_path}")
        print(f"Model name: {checkpoint.get('model_name', 'unknown')}")
        if 'metrics' in checkpoint:
            print(f"Saved metrics: {checkpoint['metrics']}")
        
        return model, checkpoint
    
    @staticmethod
    def save_pretrained(model, save_path):
        """Save chỉ weights (cho deployment)."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Pretrained weights saved to {save_path}")
```

#### 3.3. Image Display (theo đề - "Display predicted vs actual images")

```python
# src/evaluation/visualizer.py

import matplotlib.pyplot as plt
import numpy as np
import torch

class PredictionVisualizer:
    """Visualize predicted vs actual images."""
    
    CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    def __init__(self, class_names=None):
        if class_names:
            self.CLASS_NAMES = class_names
    
    def plot_predictions(self, images, true_labels, pred_labels, 
                        num_samples=20, save_path=None):
        """Display predicted vs actual images.
        
        Yêu cầu của đề: "Display predicted vs actual images"
        """
        num_samples = min(num_samples, len(images))
        rows = (num_samples + 4) // 5
        cols = min(5, num_samples)
        
        fig, axes = plt.subplots(rows, cols, figsize=(2.5*cols, 2.5*rows))
        axes = np.array(axes).flatten() if num_samples > 1 else [axes]
        
        for i in range(num_samples):
            ax = axes[i]
            img = images[i]
            
            # Convert to numpy nếu là tensor
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            
            # Denormalize nếu cần (mean=0.286, std=0.353)
            if img.ndim == 3 and img.shape[0] == 1:
                img = img.squeeze(0)  # (1, 28, 28) → (28, 28)
            elif img.ndim == 3 and img.shape[0] == 3:
                img = img.transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
            
            ax.imshow(img, cmap='gray')
            
            true_label = self.CLASS_NAMES[true_labels[i]]
            pred_label = self.CLASS_NAMES[pred_labels[i]]
            
            color = 'green' if true_labels[i] == pred_labels[i] else 'red'
            ax.set_title(f"True: {true_label}\nPred: {pred_label}",
                        color=color, fontsize=8)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Predicted vs Actual (Green=Correct, Red=Wrong)', 
                     fontsize=12, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Predictions saved to {save_path}")
        plt.show()
    
    def plot_confusion_matrix(self, true_labels, pred_labels, save_path=None):
        """Plot confusion matrix."""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(true_labels, pred_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.CLASS_NAMES, 
                    yticklabels=self.CLASS_NAMES, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix (Normalized)')
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
```

### Requirements (theo đề)

✅ **Evaluate model accuracy**:
- Test accuracy (overall)
- Per-class metrics (precision, recall, F1)
- Confusion matrix

✅ **Save and load the model**:
- `save_model()` - lưu checkpoint
- `load_model()` - load checkpoint
- `save_pretrained()` - lưu weights only

✅ **Display predicted vs actual images**:
- Grid ảnh với label predicted vs actual
- Màu xanh cho đúng, đỏ cho sai
- Save to `results/runs/`

### Test
- Load model từ checkpoint
- Verify accuracy được compute đúng
- Verify save → load → predict cycle hoạt động
- Verify images được display correctly
- Verify confusion matrix saved

### Input từ Người 2
- Trained model checkpoint: `models/checkpoints/trained_model.pt`
- Training history

### Output (Final Deliverables theo đề)
- **Python code** ✅
- **Brief report** (3 reports này + 1 final report)
- **Loss graphs** (từ Người 2: `results/runs/loss_curve.png`)
- **Image displays** (từ Người 3: `results/runs/predictions.png`)

### Report cần viết (lưu vào `docs/reports/nguoi3_evaluation_report.md`)

```markdown
# Evaluation & Display Report

## 1. Test Set Accuracy
- Accuracy: ...%
- Precision: ...%
- Recall: ...%
- F1-Score: ...%

## 2. Per-Class Metrics
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|

## 3. Save/Load Test
- Model saved to: ...
- Model loaded from: ...
- Verify same predictions: ...

## 4. Image Display
- File: results/runs/predictions.png
- Mô tả: Hiển thị 20 ảnh với predicted vs actual labels
- Correct/Wrong counts: ...

## 5. Confusion Matrix
- File: results/runs/confusion_matrix.png
- Most confused classes: ...

## 6. Files Created
- List all files
```

---

## 📋 FILES SUMMARY

### Theo người phụ trách

| Người | Module (theo đề) | File chính | File phụ |
|-------|------------------|------------|----------|
| 1 | Build Network + Hyperparameter Experiment | `baseline_model.py`, `model_factory.py`, `model_selection.py`, `hyperparameter_tuning.py` | `architectures/*.py`, `hyperopt_config.yaml` |
| 2 | Training Loop + Visualize Loss | `trainer.py`, `visualizer.py` | `callbacks.py`, `losses.py`, `optimizers.py`, `main.py` |
| 3 | Evaluate + Save/Load + Display | `evaluator.py`, `model_io.py`, `visualizer.py` | `metrics.py`, `confusion_matrix.py`, `error_analysis.py` |

### Mapping files với yêu cầu đề

| Yêu cầu đề | Người | File |
|------------|-------|------|
| Load FashionMNIST dataset | (Phase 1) | `src/data/dataset.py` ✅ |
| Apply transforms | (Phase 1) | `src/data/dataset.py` ✅ |
| **Build a neural network** | **Người 1** | `src/models/baseline_model.py`, `architectures/*.py` |
| **Create a training loop** (forward, loss, backward, optimize) | **Người 2** | `src/training/trainer.py` |
| **Evaluate model accuracy** | **Người 3** | `src/evaluation/evaluator.py` |
| **Save and load the model** | **Người 3** | `src/evaluation/model_io.py` |
| **Experiment with network/hyperparameters** | **Người 1** | `hyperparameter_tuning.py` |
| **Visualize loss** | **Người 2** | `src/training/visualizer.py` |
| **Display predicted vs actual images** | **Người 3** | `src/evaluation/visualizer.py` |

---

## 📁 CẤU TRÚC THƯ MỤC

```
src/
├── data/                              # Phase 1 ✅
│   ├── __init__.py
│   ├── dataset.py                     # Load FashionMNIST ✅
│   ├── eda.py                         # EDA ✅
│   ├── preprocessing.py               # Preprocessing ✅
│   └── split.py                       # Train/val/test split ✅
│
├── models/                            # 👤 Người 1
│   ├── __init__.py
│   ├── baseline_model.py              # BaselineMLP
│   ├── model_factory.py               # Factory pattern
│   ├── model_selection.py             # So sánh models
│   └── architectures/                 # Các architectures
│       ├── __init__.py
│       ├── mlp.py                     # DeepMLP
│       ├── simple_cnn.py              # SimpleCNN
│       └── advanced_cnn.py            # ResNetMini
│
├── training/                          # 👤 Người 1 (tuning) + 👤 Người 2
│   ├── __init__.py
│   ├── hyperparameter_tuning.py       # Người 1
│   ├── hyperopt_scheduler.py          # Người 1
│   ├── trainer.py                     # Người 2
│   ├── callbacks.py                   # Người 2
│   ├── losses.py                      # Người 2
│   ├── optimizers.py                  # Người 2
│   └── visualizer.py                  # Người 2 (Visualize loss)
│
└── evaluation/                        # 👤 Người 3
    ├── __init__.py
    ├── evaluator.py                   # Evaluate accuracy
    ├── metrics.py                     # Accuracy, Precision, Recall, F1
    ├── model_io.py                    # Save/Load model
    ├── visualizer.py                  # Display predicted vs actual
    ├── confusion_matrix.py            # Confusion matrix
    └── error_analysis.py              # Error analysis

configs/
├── model_config.yaml                  # (đã có, có thể sửa)
└── hyperopt_config.yaml               # Người 1

docs/reports/                          # Báo cáo
├── nguoi1_models_report.md            # Người 1
├── nguoi2_training_report.md          # Người 2
└── nguoi3_evaluation_report.md        # Người 3

models/checkpoints/                    # Trained models
├── trained_model.pt                   # Trained model từ Người 2
└── final_model.pt                     # Final model từ Người 3

results/runs/                          # Outputs
├── loss_curve.png                     # Loss graph (Người 2)
├── accuracy_curve.png                 # Accuracy graph (Người 2)
├── predictions.png                    # Predicted vs actual (Người 3)
└── confusion_matrix.png               # Confusion matrix (Người 3)
```

---

## ✅ CHECKLIST HOÀN THÀNH (theo đề)

### 👤 Người 1: Build Network + Hyperparameter
- [ ] `src/models/baseline_model.py` tạo xong
- [ ] `src/models/architectures/{mlp,simple_cnn,advanced_cnn}.py` tạo xong
- [ ] `src/models/model_factory.py` tạo xong
- [ ] `src/models/model_selection.py` tạo xong
- [ ] `src/training/hyperparameter_tuning.py` tạo xong
- [ ] `configs/hyperopt_config.yaml` tạo xong
- [ ] Forward pass test passed với 4 architectures
- [ ] Grid/random search test passed
- [ ] **Best model + best params identified**
- [ ] `docs/reports/nguoi1_models_report.md` viết xong

### 👤 Người 2: Training Loop + Visualize Loss
- [ ] `src/training/trainer.py` tạo xong (forward, loss, backward, optimize)
- [ ] `src/training/callbacks.py` tạo xong
- [ ] `src/training/losses.py` tạo xong
- [ ] `src/training/optimizers.py` tạo xong
- [ ] `src/training/visualizer.py` tạo xong
- [ ] `main.py` cập nhật
- [ ] Training loop test passed
- [ ] **Loss curve saved to `results/runs/loss_curve.png`**
- [ ] **Trained model saved to `models/checkpoints/trained_model.pt`**
- [ ] `docs/reports/nguoi2_training_report.md` viết xong

### 👤 Người 3: Evaluate + Save/Load + Display
- [ ] `src/evaluation/evaluator.py` tạo xong
- [ ] `src/evaluation/metrics.py` tạo xong
- [ ] `src/evaluation/model_io.py` tạo xong
- [ ] `src/evaluation/visualizer.py` tạo xong
- [ ] `src/evaluation/confusion_matrix.py` tạo xong
- [ ] `src/evaluation/error_analysis.py` tạo xong
- [ ] **Test accuracy ≥ 90%**
- [ ] **Save/Load test passed**
- [ ] **Predicted vs actual images saved to `results/runs/predictions.png`**
- [ ] `docs/reports/nguoi3_evaluation_report.md` viết xong

### 📦 Final Deliverables (theo đề)
- [ ] Python code hoàn chỉnh
- [ ] 3 reports từ 3 người
- [ ] `results/runs/loss_curve.png` (từ Người 2)
- [ ] `results/runs/predictions.png` (từ Người 3)
- [ ] `docs/PHASE2_FINAL_REPORT.md` (tổng hợp)

---

## 🎯 BỔ SUNG

### Optional Enhancements
1. **Data Augmentation** - Người 1 có thể thêm vào transforms
2. **TensorBoard Integration** - Người 2 có thể thêm
3. **GradCAM** - Người 3 có thể thêm visualization cho model decisions
4. **Unit Tests** - Mỗi người tự viết test

### Integration Test
- Chạy full pipeline: Load → Build → Train → Eval → Save/Load → Display
- Tạo `docs/reports/PHASE2_INTEGRATION_REPORT.md`

### Final Report
Tạo `docs/PHASE2_FINAL_REPORT.md`:
- Executive summary
- Best model architecture
- Final performance metrics
- Lessons learned
- Recommendations

---

## 📞 WORKFLOW GIỮA 3 NGƯỜI

```
Người 1 (Network)              Người 2 (Training)              Người 3 (Evaluation)
─────────────────              ─────────────────                ─────────────────
1. Tạo baseline MLP
2. Tạo architectures
3. Compare models
4. Hyperparameter tuning
5. Best model + params ────→   6. Setup trainer
                               7. Train (forward, loss, 
                                  backward, optimize)
                               8. Visualize loss
                               9. Save checkpoint ─────────→  10. Load checkpoint
                                                                   11. Evaluate accuracy
                                                                   12. Display predictions
                                                                   13. Save final model
                               ←─────────────────────────────────  14. Final report
```

### Giao tiếp giữa các người

| Từ | Đến | Giao |
|----|-----|------|
| Người 1 | Người 2 | Best model name, hyperparameters, model factory |
| Người 2 | Người 3 | Trained model path, training history, loss curve path |
| Người 3 | Tất cả | Final accuracy, prediction images, recommendations |

---

**Ngày tạo:** 2026-10-11  
**Cập nhật lần cuối:** 2026-10-11 (align với đề.txt, 3 người)  
**Phiên bản:** 3.0  
**Trạng thái:** Ready for Phase 2 (3 người, aligned với đề bài)

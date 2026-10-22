# Team Guide — Cách chạy FashionMNIST Project

> Hướng dẫn này dành cho **cả 3 thành viên** trong nhóm. Mục tiêu: ai mở project lên cũng có thể tự chạy thử được trong vòng 10 phút.

---

## 0. Tổng quan pipeline

```
Người 1 (đã xong)        Người 2 (đang làm)         Người 3 (làm tiếp)
─────────────────         ─────────────────          ──────────────────
data/processed/  ──────►  trainer.py                  ──────►  evaluator.py
src/models/        ──────►  src/training/trainer.py   ──────►  model_io.py
src/training/      ──────►  callbacks/visualizer      ──────►  visualizer.py
configs/hyperopt.yaml     main.py (loop chính)
run_experiments.py
```

- **Data (Phase 1)** đã có sẵn ở `data/processed/` (54k train / 6k val / 10k test).
- **Người 1**: 4 kiến trúc + hyperparameter tuning → đã xong, chạy được qua `run_experiments.py`.
- **Người 2**: training loop + callbacks + visualize loss → đang code.
- **Người 3**: evaluate + save/load + display predictions.

---

## 1. Setup môi trường (mỗi máy làm 1 lần)

### 1.1. Clone & mở project

```powershell
cd C:\Users\hoait\Downloads\Lab1
```

### 1.2. Tạo virtual environment (venv)

```powershell
# Tạo venv tên .venv ngay trong workspace
python -m venv .venv

# Activate (PowerShell)
.\.venv\Scripts\Activate.ps1
```

> ⚠️ Nếu PowerShell báo lỗi `running scripts is disabled`, chạy **1 lần** với quyền admin:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
> Rồi activate lại.

### 1.3. Cài dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.4. Kiểm tra nhanh

```powershell
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
```

> Project chạy được trên CPU (đã test). CUDA optional.

---

## 3. Chuẩn bị dữ liệu (1 lần)

Data đã được generate sẵn ở `data/processed/`. Nếu clone sang máy mới chưa có, chạy:

```powershell
# Chia train/val/test từ FashionMNIST gốc (chỉ làm 1 lần, ~2-3 phút)
python -m src.data.split --root data/processed --val-split 10000 --seed 42
```

Kiểm tra output:

```
data/
├── raw/                          # dataset gốc (auto download lần đầu)
├── processed/
│   ├── train/<class_id>/*.png    # 54,000 ảnh
│   ├── val/<class_id>/*.png      # 6,000 ảnh
│   └── test/<class_id>/*.png     # 10,000 ảnh
└── splits/
    └── split_info.txt            # metadata split
```

---

## 4. Chạy phần của Người 1 (Hyperparameter tuning) ✅ ĐÃ HOÀN THÀNH

### 4.1. Smoke test cực nhanh (~5s)

Kiểm tra 4 model (`baseline_mlp`, `deep_mlp`, `simple_cnn`, `resnet_mini`) chạy được end-to-end:

```powershell
python run_experiments.py --smoke
```

Kỳ vọng:
```
[smoke] baseline_mlp: shape_ok=True, loss 2.xxxx -> 0.xxxx (OK)
[smoke] deep_mlp:     shape_ok=True, loss ...
[smoke] simple_cnn:   shape_ok=True, loss ...
[smoke] resnet_mini:  shape_ok=True, loss ...
```

### 4.2. Grid search nhanh (smoke, ~3 phút)

```powershell
python run_experiments.py --epochs 3 --batch-size 128
```

Kỳ vọng output cuối (ví dụ từ run `--epochs 3`):
```
========================================================================
NGƯỜI 1 SUMMARY — Best (model, hyperparams)
========================================================================
trial_id     : 1
val_acc      : 90.53%
train_acc    : 88.81%
params       :
  - model_name: simple_cnn
  - learning_rate: 0.001
  - optimizer: adam
  - epochs: 3
========================================================================
```

Artifacts tự động ghi vào:
- `results/runs/hyperopt.json` — toàn bộ trial results
- `results/runs/architecture_selection.md` — so sánh architectures
- `results/runs/best_config.json` — best config tìm được

### 4.3. Grid search đầy đủ — 16 trials × 10 epoch (~4,5 giờ CPU)

```powershell
python run_experiments.py --batch-size 128
```

Grid search **phủ kín** toân bộ: 4 kiến trúc × 2 learning rate × 2 optimizer = 16 trials × 10 epoch = 160 epoch. Tất định, không bỏ sót tổ hợp nào.

Config ở `configs/hyperopt_config.yaml` — muốn đổi search space thì sửa file này.

> **Kết quả chạy thật (2026-10-22):** best = **simple_cnn + adam + lr=0.001 + 10 epoch → val_acc 92.33%** (trial 9). Runner-up: `resnet_mini + adam + lr=0.0005 → 92.20%`. Chi tiết: `results/runs/hyperopt.md`.

---

## 5. Chạy phần của Người 2 (Training loop) 🚧 ĐANG CODE

### 5.1. File cần tạo

```
src/training/trainer.py          # loop chính: forward → loss → backward → step
src/training/callbacks.py        # early stopping, checkpoint, log
src/training/visualizer.py       # vẽ loss/acc curve
main.py                          # entry point cho Người 2 (override stub)
```

### 5.2. Skeleton trainer (tham khảo)

```python
# src/training/trainer.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total
```

### 5.3. Lệnh chạy tạm (khi đã có trainer.py)

```powershell
python main.py --model simple_cnn --epochs 5 --lr 0.001 --optimizer adam
```

### 5.4. Test nhanh 1 model

```powershell
python -c "
import sys; sys.path.insert(0, '.')
import torch, torch.nn.functional as F
from src.models.model_factory import ModelFactory
from src.data.split import split_pipeline

train_loader, val_loader, _, _ = split_pipeline('./data/processed', batch_size=128)

model = ModelFactory.create('simple_cnn')
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(2):
    model.train()
    for x, y in train_loader:
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    print(f'epoch {epoch}: last batch loss = {loss.item():.4f}')
"
```

---

## 6. Chạy Người 3 (Evaluate + Save/Load) — Chưa code

Khi code đã có:

```powershell
python -m src.evaluation.evaluator --checkpoint models/best.pt --split test
python -m src.evaluation.model_io --save models/best.pt
```

---

## 7. Logs & Outputs

| Folder | Chứa gì |
|--------|---------|
| `logs/` | log files (tự tạo khi run) |
| `results/runs/` | hyperopt.json, architecture_selection.md, plots |
| `models/checkpoints/` | file `.pt` save/load model |
| `data/splits/split_info.txt` | metadata train/val/test split |
| `docs/reports/` | báo cáo phase 1/2 |

Để xem log của lần chạy gần nhất:

```powershell
Get-Content _e2e_run.log -Tail 30   # xem 30 dòng cuối
```

---

## 8. Troubleshooting thường gặp

| Lỗi | Nguyên nhân | Cách fix |
|------|-----------|----------|
| `ModuleNotFoundError: No module named 'src'` | Chưa `cd` đúng thư mục gốc | `cd C:\Users\hoait\Downloads\Lab1` |
| `'ResNetMini' object has no attribute 'expansion'` | Bug cũ đã fix ở `src/models/architectures/advanced_cnn.py` | Pull code mới nhất |
| `pin_memory` warning | CPU only | Bỏ qua warning, không ảnh hưởng |
| Loss = 89.97% ở trial 1 nhưng giảm ở trial 2 | Grid cũng tất định — không phải lỗi | Kiểm tra `grid_search` đã expand đúng product |
| PowerShell không activate được venv | Execution policy | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `data/processed` rỗng | Chưa generate split | Chạy bước 3 ở trên |

---

## 9. Quy ước khi commit code

```powershell
git checkout -b feat/<ten-minh>-<viec-dang-lam>
git add .
git commit -m "nguoi-1: them resnet_mini architecture"
git push origin feat/<ten-minh>-<viec-dang-lam>
```

**Không commit**:
- `.venv/` (đã có trong `.gitignore`)
- `data/processed/*.png` (70k files)
- `models/checkpoints/*.pt` (file lớn)
- `_e2e_run.log`, `logs/` local

---

## 10. Liên hệ nhanh

- Docs chi tiết Phase 2 plan: `docs/PHASE2_PLAN.md`
- Báo cáo Phase 1: `docs/PHASE1_REPORT.md`
- Cấu hình hyperopt: `configs/hyperopt_config.yaml`
- Quick start cũ: `docs/QUICK_START.md`

> Cập nhật lần cuối: **2026-10-22** — grid search 16 trials × 10 epoch chạy thật, best val_acc = **92.33%** (simple_cnn + adam + lr=0.001).
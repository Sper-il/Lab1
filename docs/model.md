# Mô tả Model

> Mô tả kiến trúc model cho project — model nào được dùng, vì sao chọn, input/output, loss, params, config và cách tái tạo.
> Nguồn chuẩn: `.cursor/skills/ai-fusion-workflow/modules/model.md` (baseline trước, backbone + head, loss, params/dung lượng, config hóa, naming).
> Báo cáo thí nghiệm chi tiết của Người 1 (pilot + grid 16 trials) xem tại `docs/reports/nguoi1_models_report.md` và `results/runs/hyperopt.md`.

## 1. Bài toán → họ model

- **Bài toán:** multi-class image classification trên FashionMNIST (ảnh xám 28×28, 10 lớp, 50k train / 10k val / 10k test, chi tiết xem `docs/data.md`).
- **Họ kiến trúc được chọn:** CNN nhỏ tự viết (không dùng backbone pretrained) vì ảnh 28×28 đơn kênh, dataset cân bằng (6.000 ảnh/lớp gốc), máy lab CPU-only. CNN tận dụng cấu trúc không gian 2D (filter trượt khắp ảnh, shared weights, translation invariance) nên vượt MLP 5–7% trong mọi thí nghiệm của project.
- **Vì sao không dùng backbone lớn (ResNet-18 pretrained, ViT):** quá nặng so với dữ liệu nhỏ (dễ overfit), chậm trên CPU lab (ResNetMini tự viết đã chậm gấp 2,9× SimpleCNN), giấy phép và trọng số ngoài không cần thiết cho đề Lab 1 (tự build network từ đầu).

## 2. Baseline trước

- **Baseline đơn giản nhất chạy được:** `BaselineMLP` (`src/models/baseline_model.py`) — `Flatten` → `Linear(784→256)` → `ReLU` → `Dropout(0,2)` → `Linear(256→10)`, tổng 203.530 params.
- **Metric baseline:** pilot 2 epochs (Adam lr=1e-3) đạt val **87,05%**; grid 10 epochs (Adam lr=0,001) đạt val **89,75%** (`results/runs/architecture_selection.md`, `results/runs/hyperopt.md`).
- **Ý nghĩa:** mọi kiến trúc phức tạp hơn phải vượt rõ mức sàn này mới được giữ — đúng checklist skill (có baseline chạy trước khi nâng cấp).

## 3. Backbone + head

### 3.1. Model chính thức: `SimpleCNN` (`src/models/architectures/simple_cnn.py`)

Backbone gồm 2 conv blocks, head là 2 FC layers (docstring trong code ghi đầy đủ input/output shape):

```
Input (B, 1, 28, 28)
  → Conv2d(1→32, k=3, p=1) → ReLU → MaxPool(2)   → (B, 32, 14, 14)
  → Conv2d(32→64, k=3, p=1) → ReLU → MaxPool(2)  → (B, 64, 7, 7)
  → Flatten                                       → (B, 3136)
  → Linear(3136→128) → ReLU → Dropout(0,5)        → (B, 128)   (head)
  → Linear(128→10)                                → (B, 10)    (logits, chưa softmax)
```

- **Cố ý không dùng BatchNorm** trong `SimpleCNN` (giữ đơn giản theo đề); BatchNorm chỉ dùng cho `DeepMLP`/`ResNetMini`.
- **Khớp Design:** input `(B,1,28,28)` → output `(B,10)` khớp `Interfaces / APIs` trong `docs/api.md` và `docs/data-flow.md`.

### 3.2. Các kiến trúc còn lại (đã thử và loại)

| Tên đăng ký | Class / file | Cấu trúc | Vai trò trong project |
|-------------|--------------|----------|------------------------|
| `deep_mlp` | `DeepMLP` (`src/models/architectures/mlp.py`) | 3 hidden FC `80→256→128`, mỗi tầng `Linear → BatchNorm1d → ReLU → Dropout(0,3)` + `Linear(128→10)`; 118.650 params | Baseline sâu để chứng minh MLP sâu vẫn thua CNN |
| `resnet_mini` | `ResNetMini` + `BasicBlock` (`src/models/architectures/advanced_cnn.py`) | Stem `Conv(1→16)` + `BN`; 3 residual stages (2 blocks/stage, widths 16/32/64, strides 1/2/2, shortcut `1×1` khi lệch shape) → `AdaptiveAvgPool2d(1,1)` → `FC(64→10)`; 174.970 params | Phương án nâng cấp residual, về nhì grid (val 92,20%) nhưng overfit hơn và chậm gấp 2,9× nên bị loại |

- Mọi model đều nhận `(B,1,28,28)` (riêng `DeepMLP` tự `view` khi nhận tensor 4D), không `softmax` ở cuối để dùng trực tiếp với `CrossEntropyLoss`.
- Tạo model qua điểm tiếp xúc duy nhất: `ModelFactory.create("simple_cnn")` (`src/models/model_factory.py`); dropout mặc định `baseline_mlp=0,2`, `deep_mlp=0,3`, `simple_cnn=0,5`.

## 4. Loss function

- **Loss chính:** `CrossEntropyLoss` (`src/training/losses.py` → `create_loss`), nhận thẳng logits `(B,10)` + nhãn `(B,)` — gộp Softmax + NLL nên ổn định số học, phù hợp phân loại đa lớp.
- **Class weight:** không dùng vì dataset cân bằng hoàn hảo (mỗi lớp đúng 5.000/1.000/1.000 train/val/test, xem `docs/data.md`); metric đánh giá dùng macro-average để công bằng với lớp khó (Shirt, Coat).
- **Metric đi kèm:** accuracy + precision/recall/F1 macro bằng scikit-learn (`src/evaluation/evaluator.py`).

## 5. Số tham số / dung lượng

| Model | Params | Checkpoint thực tế | Tốc độ train (CPU lab) |
|-------|--------|--------------------|-------------------------|
| `SimpleCNN` (chính thức) | 421.642 (~1,6 MB trọng số float32) | `trained_model.pt` 5.071.079 bytes (~4,84 MB, kèm optimizer + history) · `final_model.pt` 1.690.373 bytes (~1,61 MB, kèm metrics) | ~75 s/epoch ở grid (bs=128) · ~96 s/epoch ở pilot (bs=64) |
| `BaselineMLP` | 203.530 | — (không bàn giao) | ~29–31 s/epoch |
| `DeepMLP` | 118.650 | — | ~28–33 s/epoch |
| `ResNetMini` | 174.970 | — | ~202–259 s/epoch (chậm gấp ~2,9× SimpleCNN) |

- Ngân sách deploy: model ~1,6 MB, forward 1 batch 64 ảnh CPU trong vài giây (`python run_experiments.py --smoke`), vừa máy lab CPU-only; không cần GPU hay export ONNX trong phạm vi Lab 1.

## 6. Config hóa

Mọi số kiến trúc nằm trong `configs/model_config.yaml`, không hardcode trong code:

```yaml
model:
  name: simple_cnn
  dropout: 0.5
  num_classes: 10
```

- `run_training.py` validate schema chặt (thiếu trường, dropout ngoài `[0, 1)` đều báo lỗi trước khi train); CLI `main.py` chỉ override giá trị truyền rõ (`--lr`, `--batch-size`, `--epochs`...).
- Search space grid nằm riêng ở `configs/hyperopt_config.yaml` (4 model × 2 lr × 2 optimizer = 16 trials × 10 epochs, `seed=42`).
- Seed cố định 42 (Python + NumPy + PyTorch, `cudnn.deterministic=True`) nên tái lập tuyệt đối.

## 7. Naming + version

- Quy ước: `{domain}_{architecture}_{version}.{ext}` theo skill — áp dụng: `fashion_simplecnn_v1.pt` (tương ứng `trained_model.pt` / `final_model.pt` hiện tại).
- Checkpoint chứa `model_name`, `hyperparameters`, `metrics`, `epoch`, `history` nên mở file là biết version + config, không phụ thuộc notebook.
- Đổi kiến trúc → tag version mới + ghi CHANGELOG + Decisions Log trong `progress_status.md` (quyết định chọn `simple_cnn` + Adam + lr=0,001 đã ghi ngày 2026-10-22).

## 8. Cách build / tái tạo

```python
from src.models.model_factory import ModelFactory

model = ModelFactory.create("simple_cnn")  # dropout=0.5, num_classes=10 mặc định
x = torch.randn(8, 1, 28, 28)
logits = model(x)  # (8, 10) — dùng với nn.CrossEntropyLoss
```

```bash
# Verify 1 batch thật trước khi train dài (khớp checklist skill)
python run_experiments.py --smoke

# Train full với đúng model + config này (15 epochs, CPU)
python main.py
```

Kết quả với đúng model này: best epoch 11 (`val_loss=0,215183`, `val_acc=92,85%`), test **92,10%** — chi tiết xem `docs/training.md`, `docs/evaluation.md`, `docs/EXPERIMENTS.md`.

## 9. Checklist review (theo skill, đã pass)

- [x] Có baseline đơn giản (`BaselineMLP` 87,05%) chạy trước khi nâng cấp lên CNN/ResNet.
- [x] Không dùng pretrained weight ngoài nên không vướng giấy phép.
- [x] Input/output shape `(B,1,28,28) → (B,10)` khớp `docs/api.md` và `docs/data-flow.md` (đã verify bằng 1 batch thật + smoke test).
- [x] Loss không cần class weight vì dữ liệu cân bằng; mask rỗng không áp dụng cho classification.
- [x] Mọi số (`num_classes=10`, `in_channels=1`, `dropout=0,5`) nằm trong `configs/model_config.yaml`.
- [x] Đã ước lượng params (421.642) / dung lượng (~1,6–4,8 MB) / tốc độ (~75 s/epoch) trước khi train dài 15 epochs.
- [x] Quyết định chọn `simple_cnn` / loại `resnet_mini` đã ghi vào Decisions Log (`progress_status.md`).

## 10. Rủi ro đã kiểm soát

- Model quá lớn → overfit: đã bắt đầu từ MLP nhỏ rồi tăng dần, kèm Dropout + `EarlyStopping` (`val_loss`/`min`/`patience=5`) + best-weight restoration.
- Quên `model.eval()` / `torch.no_grad()` khi eval: `Trainer.validate()` và `ModelEvaluator.predict()` đều bọc đúng.
- Mismatch kênh/số lớp: `in_channels=1`, `num_classes=10` cố định trong config + factory; smoke test bắt sai shape ngay.
- Không seed/version: seed 42 cố định + checkpoint kèm config/history/metrics.

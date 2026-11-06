# Báo cáo tổng quan Training

> Báo cáo chính của pipeline huấn luyện — tổng hợp toàn bộ vòng train của project, không phải báo cáo cá nhân.
> Báo cáo cá nhân của Người 2 xem tại `docs/reports/nguoi2_training_report.md`.
> Nguồn code: `src/training/`, `run_training.py`, `main.py`, `configs/model_config.yaml`.
> Kết quả run chính thức: `results/runs/training_history.json`, `loss_curve.png`, `accuracy_curve.png`.

## 1. Mục tiêu

- Huấn luyện classifier FashionMNIST đạt accuracy validation ≥ 90% trên CPU, với quy trình tái lập được 100% từ config + seed.
- Mọi lần train sinh artifact đầy đủ: best/last checkpoint, history JSON, loss/accuracy curves — không ghi đè run cũ.

## 2. Cấu hình hiệu lực (`configs/model_config.yaml`)

| Thành phần | Giá trị |
|------------|---------|
| Model | `simple_cnn` (421.642 params, dropout 0,5, 10 lớp) |
| Loss | `CrossEntropyLoss` (`src/training/losses.py` → `create_loss`) |
| Optimizer | Adam, lr=0,001, weight_decay=0,0 (`src/training/optimizers.py` → `create_optimizer`; SGD momentum 0,9 cũng được hỗ trợ) |
| Batch size | 128 (train 391 batch, val 79 batch) |
| Epochs tối đa | 15 |
| Device | `auto` → thực tế CPU (CUDA không khả dụng) |
| Seed | 42 (seed Python + NumPy + PyTorch CPU/CUDA, `cudnn.deterministic=True`) |
| Num workers | 0 (tương thích Windows) |
| Augment | tắt; chỉ `ToTensor` + normalize mean 0,2860 / std 0,3530 |
| Early stopping | monitor `val_loss`, mode `min`, patience 5, min_delta 0,0001 |

Config được validate schema chặt trong `run_training.py` (`_validate_config`): thiếu trường, lr ≤ 0, dropout ngoài `[0, 1)` đều báo lỗi rõ ràng trước khi train.

## 3. Kiến trúc code

```
configs/model_config.yaml
  → main.py (CLI: --config/--epochs/--batch-size/--lr/--device/--num-workers/--output-dir)
  → run_training.load_config + apply_overrides (chỉ override giá trị CLI truyền rõ)
  → run_training.run_training(config)
      ├── set_random_seed(42) + resolve_device(auto→cpu)
      ├── split_pipeline (train 50k / val 10k / test 10k, stratified, seed=42)
      ├── ModelFactory.create("simple_cnn") + create_loss + create_optimizer
      ├── EarlyStopping (best → trained_model.pt, last → last_model.pt)
      ├── Trainer.train(15) → history {train_loss, val_loss, train_acc, val_acc}
      └── persist training_history.json + loss_curve.png + accuracy_curve.png
```

### 3.1. Training loop (`src/training/trainer.py` → `Trainer`)

- `train_epoch()`: `model.train()`, `zero_grad(set_to_none=True)` → forward logits `(B,10)` → loss → `backward()` → `step()`; loss/accuracy **weighted theo số mẫu** nên batch cuối lẻ vẫn đúng.
- `validate()`: `model.eval()` + `torch.no_grad()`, không cập nhật tham số; cùng cách weighted metrics.
- `train(num_epochs)`: lặp epoch, gọi callbacks mỗi epoch (`callback.step(...)`), log thời gian từng epoch bằng `logging`, early stopping thì dừng; cuối cùng **restore best weights** vào model trả về.
- Guard: loader rỗng → `ValueError`; `num_epochs ≤ 0` → `ValueError`.

### 3.2. Early stopping + checkpoint (`src/training/callbacks.py` → `EarlyStopping`)

- Cải thiện được định nghĩa chặt: mode `min` yêu cầu `value ≤ best − min_delta` (tránh "cải thiện giả" do nhiễu).
- Mỗi epoch đều lưu `last_model.pt`; chỉ khi cải thiện mới ghi đè `trained_model.pt`.
- Checkpoint chứa: `model_state_dict`, `optimizer_state_dict` (để resume), `epoch`, `metrics`, `monitor`, `best_metric`, `best_epoch`, `history`, `model_name`, `hyperparameters`.
- Guard: metric monitor thiếu → `KeyError`; metric non-finite (NaN/Inf) → `ValueError` ngay thay vì lưu checkpoint hỏng.
- `restore_best(model, device)`: load best weights, throw `FileNotFoundError` nếu chưa có best.

### 3.3. Visualization (`src/training/visualizer.py` → `TrainingVisualizer`)

- `plot_loss_curve` / `plot_accuracy_curve`: backend `Agg` (headless), tự tạo thư mục output, validate history đủ 4 key + độ dài bằng nhau, vẽ kèm legend/grid, đóng figure sau khi lưu.
- Output: `results/runs/loss_curve.png`, `results/runs/accuracy_curve.png`.

## 4. Kết quả run chính thức (`python main.py`, 15 epochs, CPU)

| Epoch | Train loss | Val loss | Train acc | Val acc |
|---:|---:|---:|---:|---:|
| 1 | 0,585270 | 0,351479 | 79,03% | 87,08% |
| 2 | 0,379770 | 0,301564 | 86,38% | 89,50% |
| 3 | 0,327278 | 0,274018 | 88,15% | 90,55% |
| 4 | 0,293590 | 0,249961 | 89,42% | 91,30% |
| 5 | 0,265313 | 0,237384 | 90,41% | 91,42% |
| 6 | 0,248659 | 0,240723 | 91,00% | 91,68% |
| 7 | 0,229662 | 0,224409 | 91,73% | 91,98% |
| 8 | 0,214868 | 0,245304 | 92,22% | 91,65% |
| 9 | 0,201467 | 0,223467 | 92,59% | 92,40% |
| 10 | 0,190164 | 0,221764 | 92,91% | 92,63% |
| 11 | 0,178463 | **0,215183** | 93,47% | 92,85% |
| 12 | 0,168702 | 0,222018 | 93,83% | 92,60% |
| 13 | 0,159146 | 0,216467 | 94,03% | **92,88%** (đỉnh val_acc) |
| 14 | 0,151745 | 0,222037 | 94,28% | 92,87% |
| 15 | 0,141153 | 0,229658 | **94,72%** | 92,80% |

- **Best theo `val_loss`:** epoch 11 (val_loss 0,215183, val_acc 92,85%) → weights bàn giao trong `trained_model.pt`.
- **Early stopping không kích hoạt:** sau best chỉ có 4 epoch liên tiếp không cải thiện, chưa đủ patience 5 nên chạy đủ 15 epoch.
- **Nhận xét:** train_acc 94,72% vs val_acc ~92,8% cuối run → overfitting nhẹ; chọn best theo `val_loss` thay vì `val_acc` giúp weights ổn định hơn.
- **Thời gian:** ~75 s/epoch trên CPU.

## 5. Artifact bàn giao

| File | Nội dung |
|------|----------|
| `models/checkpoints/trained_model.pt` | Best weights (epoch 11) + optimizer state + history tại best + hyperparameters |
| `models/checkpoints/last_model.pt` | Epoch 15 cuối cùng (local-only, không track) |
| `results/runs/training_history.json` | Full history 15 epochs (4 series) |
| `results/runs/loss_curve.png` | Train vs val loss |
| `results/runs/accuracy_curve.png` | Train vs val accuracy |

## 6. Kiểm thử

```bash
$env:PYTHONPATH = "."; python -m pytest tests -q   # 27 passed
```

Suite `tests/training/` bao phủ: parameters thay đổi sau train, validation không cập nhật parameters, history đủ 4 key, `min_delta` chính xác, non-finite values, boolean-as-int, YAML schema, CLI override, device/seed.

## 7. Cách chạy lại

```bash
# Full training với config mặc định (15 epochs, CPU ~20 phút)
python main.py

# Override nhanh
python main.py --epochs 10 --batch-size 128 --lr 0.001

# Dùng config khác
python main.py --config configs/model_config.yaml --output-dir results/runs
```

## 8. Quyết định kỹ thuật chính

| Quyết định | Lựa chọn khác | Lý do |
|------------|---------------|-------|
| Chọn best theo `val_loss` | Theo `val_acc` | `val_loss` ổn định hơn khi train/val bắt đầu phân kỳ (overfit nhẹ cuối run) |
| Adam lr=0,001 từ grid Người 1 | SGD / lr khác | Grid 16 trials chứng minh Adam hơn SGD 2–6% mọi kiến trúc; `simple_cnn` + Adam + lr=0,001 đạt val 92,33% |
| Lưu cả best + last | Chỉ lưu một | Best để bàn giao eval; last để resume/debug đủ 15 epoch |
| Weighted metrics theo mẫu | Mean theo batch | Batch cuối lẻ (50.000/128 dư 80) không làm lệch metric |
| `num_workers=0`, seed 42 | Workers > 0 | Tương thích Windows + tái lập tuyệt đối |

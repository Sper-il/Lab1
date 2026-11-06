# Báo cáo Người 2 — Training Loop & Loss Visualization

> **Trạng thái:** Hoàn thành triển khai và full training; chờ nghiệm thu cuối.
> **Ngày cập nhật:** 2026-10-14
> **Nguồn:** `đề.txt`, `docs/PHASE2_PLAN.md`, báo cáo Người 1 và kế hoạch triển khai đã duyệt.

## 1. Phạm vi và phân loại yêu cầu

| Nhãn | Phạm vi |
|---|---|
| ĐỀ BÀI | Training loop: forward, loss, backward, optimize; trực quan hóa loss. |
| KẾ HOẠCH | Validation, early stopping, checkpoint, history, `main.py`, `run_training.py`, public exports và bàn giao artifact. |
| BỔ SUNG | YAML validation, CLI override, seed, device auto, weighted metrics, metadata checkpoint, smoke tests, logging và policy `.gitignore`. |

Các mục đánh giá/evaluation test set, confusion matrix, predicted-vs-actual và save/load module dùng chung thuộc Người 3, không nằm trong phạm vi Người 2.

## 2. Cấu hình nhận từ Người 1 và cấu hình hiệu lực

| Thành phần | Giá trị |
|---|---|
| Model | `simple_cnn` |
| Model factory | `ModelFactory.create("simple_cnn")` |
| Loss | `CrossEntropyLoss` |
| Optimizer | Adam |
| Learning rate | `0.001` |
| Batch size | `128` |
| Weight decay | `0.0` |
| Dropout | `0.5` |
| Epoch tối đa | `15` |
| Device cấu hình | `auto` |
| Device thực tế | CPU (`CUDA` không khả dụng) |
| Seed | `42` |
| Num workers | `0` |
| Augmentation | Tắt; `ToTensor` + normalize FashionMNIST |
| Early stopping | `monitor=val_loss`, `mode=min`, `patience=5`, `min_delta=0.0001` |

Cấu hình mặc định nằm tại `configs/model_config.yaml`; CLI chỉ override các giá trị được truyền rõ ràng.

## 3. Thành phần đã hoàn thành

### 3.1. Training loop

- [x] Forward → CrossEntropyLoss → `zero_grad`/backward → `optimizer.step`.
- [x] Chuyển model, ảnh và nhãn về cùng device.
- [x] `model.train()` trong training và `model.eval()` + `torch.no_grad()` trong validation.
- [x] Weighted loss/accuracy theo số lượng mẫu, kể cả batch cuối nhỏ hơn.
- [x] History gồm `train_loss`, `val_loss`, `train_acc`, `val_acc`.
- [x] Logging kết quả và thời gian từng epoch.

### 3.2. Early stopping và checkpoint

- [x] Early stopping theo `val_loss` với patience/min_delta đã duyệt.
- [x] `trained_model.pt` lưu best weights.
- [x] `last_model.pt` lưu epoch cuối thực sự chạy.
- [x] Best weights được khôi phục vào model trả về sau training.
- [x] Checkpoint chứa model state, optimizer state, epoch, metrics, history, model name và hyperparameters.

### 3.3. Visualization

- [x] Loss curve train/validation có nhãn trục, tiêu đề và chú thích.
- [x] Accuracy curve train/validation.
- [x] Tự tạo thư mục output, lưu PNG và đóng figure.

### 3.4. Entry point và orchestration

- [x] `main.py` là entry point cấp cao nhất.
- [x] `main.py` đọc YAML, validate, áp dụng CLI override rồi gọi `run_training(config)`.
- [x] `run_training.py` lắp ráp data loader, model, loss, optimizer, callback, Trainer và visualizer.
- [x] `src/training/__init__.py` export API Người 2 và giữ nguyên export tuning của Người 1.

## 4. Kết quả kiểm thử và full training

### 4.1. Kiểm thử

Lệnh:

```powershell
python -m pytest tests/training -v
```

Kết quả: **27 passed**.

Smoke test bằng `TensorDataset` xác nhận parameters thay đổi sau training, validation không cập nhật parameters và history đủ bốn key.

### 4.2. Full training FashionMNIST

Lệnh tái tạo:

```powershell
python main.py
```

Kết quả chạy đủ 15 epoch; early stopping không kích hoạt vì sau best epoch 11 chỉ có 4 epoch liên tiếp không cải thiện, chưa đạt patience 5.

| Epoch | Train loss | Val loss | Train acc (%) | Val acc (%) |
|---:|---:|---:|---:|---:|
| 1 | 0.585270 | 0.351479 | 79.03 | 87.08 |
| 2 | 0.379770 | 0.301564 | 86.38 | 89.50 |
| 3 | 0.327278 | 0.274018 | 88.15 | 90.55 |
| 4 | 0.293590 | 0.249961 | 89.42 | 91.30 |
| 5 | 0.265313 | 0.237384 | 90.41 | 91.42 |
| 6 | 0.248659 | 0.240723 | 91.00 | 91.68 |
| 7 | 0.229662 | 0.224409 | 91.73 | 91.98 |
| 8 | 0.214868 | 0.245304 | 92.22 | 91.65 |
| 9 | 0.201467 | 0.223467 | 92.59 | 92.40 |
| 10 | 0.190164 | 0.221764 | 92.91 | 92.63 |
| 11 | 0.178463 | **0.215183** | 93.47 | 92.85 |
| 12 | 0.168702 | 0.222018 | 93.83 | 92.60 |
| 13 | 0.159146 | 0.216467 | 94.03 | 92.88 |
| 14 | 0.151745 | 0.222037 | 94.28 | 92.87 |
| 15 | 0.141153 | 0.229658 | **94.72** | **92.80** |

> Bảng này là **run chính thức** (15 epoch, bs=128, seed=42) — đúng bằng
> `results/runs/training_history.json` và checkpoint `trained_model.pt`.

Best theo tiêu chí `val_loss` là epoch 11 (`0.215183`, validation accuracy `92.85%`). Last checkpoint là epoch 15 (`val_acc` 92.80%). Đỉnh `val_acc` của run là 92.88% ở epoch 13. Chênh lệch giữa train và validation ở các epoch cuối cho thấy có dấu hiệu overfitting nhẹ; việc chọn best theo validation loss giúp bàn giao weights ổn định hơn.

## 5. Artifact bàn giao cho Người 3

- Best checkpoint: `models/checkpoints/trained_model.pt`.
- Last checkpoint local: `models/checkpoints/last_model.pt`.
- Full history: `results/runs/training_history.json`.
- Loss plot: `results/runs/loss_curve.png`.
- Accuracy plot: `results/runs/accuracy_curve.png`.

Best checkpoint load thành công; last checkpoint có metadata epoch 15. History JSON chứa đủ 15 epoch. Best checkpoint lưu history tại thời điểm best (11 epoch), còn last checkpoint lưu history đầy đủ đến epoch 15.

## 6. Danh sách file đã tạo/sửa

### Tạo mới

- `run_training.py`
- `src/training/callbacks.py`
- `src/training/losses.py`
- `src/training/optimizers.py`
- `src/training/visualizer.py`
- `tests/training/`
- `models/checkpoints/trained_model.pt`
- `results/runs/training_history.json`
- `results/runs/loss_curve.png`
- `results/runs/accuracy_curve.png`

### Sửa

- `configs/model_config.yaml`
- `requirements.txt`
- `main.py`
- `src/training/trainer.py`
- `src/training/__init__.py`
- `.gitignore`
- `progress_status.md`

## 7. Trạng thái bàn giao

- [x] Code training và visualization hoàn thành.
- [x] Test suite training đạt 27/27.
- [x] Full training 15 epoch thành công.
- [x] Checkpoint, history và biểu đồ đã sinh và kiểm tra được.
- [x] Báo cáo Người 2 đã cập nhật.
- [x] `progress_status.md` đã cập nhật.
- [x] Nghiệm thu Người 2 được người dùng xác nhận.
- [x] Nội dung thay đổi cần thiết đã chuyển vào báo cáo.
- [x] Spec/plan triển khai đã được xóa sau nghiệm thu.

## 8. Nhật ký thay đổi và nghiệm thu cuối


### 8.1. Các quyết định chính đã thực thi

| Mã | Nội dung | Kết quả |
|---|---|---|
| D-001 | Thực hiện đầy đủ phạm vi Người 2 trong `PHASE2_PLAN.md` và báo cáo công việc. | Đã hoàn thành. |
| D-002 | Dùng cấu hình Người 1: `simple_cnn`, Adam, `lr=0.001`, batch `128`, dropout `0.5`, tối đa `15` epoch. | Đã áp dụng và chạy full training. |
| D-003 | Early stopping theo `val_loss`, `patience=5`, `min_delta=0.0001`. | Đã áp dụng; lần chạy chính hoàn tất đủ 15 epoch. |
| D-004 | Theo dõi thay đổi trong ledger riêng ngoài repository. | Đã ghi nhận xuyên suốt quá trình. |

### 8.2. Tóm tắt thay đổi và bằng chứng

| Mã | Phạm vi | Bằng chứng/bàn giao |
|---|---|---|
| E-002 | Cấu hình YAML và dependency. | Config Người 1 được validate; `PyYAML` đã khai báo. |
| E-003 | Loss/optimizer factories. | CrossEntropyLoss, Adam và SGD momentum được kiểm thử. |
| E-004 | Early stopping và checkpoint callback. | Best/last metadata, restore best weights, `val_loss/min/patience=5/min_delta=0.0001`. |
| E-005 | Trainer core. | Epoch train/validation, weighted metrics, history và callback integration. |
| E-006 | Visualizer. | Loss/accuracy PNG sinh bằng backend headless. |
| E-007 | Config loader, CLI, device và seed. | YAML schema, CLI override, `device=auto`, seed reproducibility. |
| E-008 | Orchestration và public exports. | `run_training.py`, `main.py`, `src/training/__init__.py` và artifact persistence. |
| E-009 | Artifact ignore policy và real-data verification. | Best/history/plots tracked; last checkpoint local-only; smoke run thành công. |
| E-010/E-011 | Remediation sau review. | Xử lý chính xác `min_delta`, non-finite values và boolean-as-int; suite đạt 27 tests. |
| E-012 | Official full training. | 15/15 epoch; best epoch 11, last epoch 15; artifacts load được. |
| E-013 | Documentation handoff. | Báo cáo Người 2 và `progress_status.md` được cập nhật. |
| E-014 | Khôi phục lịch sử tiến độ. | Lịch sử Người 1 được giữ nguyên; cập nhật Người 2 mang tính bổ sung. |
| E-015 | Chuẩn hóa `progress_status.md` theo template skill. | Đã thêm Active Module, Module/Decision Log, Risks, Open Questions và DoD. |

### 8.3. Xác nhận nghiệm thu

- **Người nghiệm thu:** Chủ dự án.
- **Trạng thái:** Người dùng đã xác nhận “Nghiệm thu Người 2”.
- **Phạm vi nghiệm thu:** Training loop, validation, early stopping, checkpoint best/last, best-weight restoration, visualization, tests, full training và artifact bàn giao.

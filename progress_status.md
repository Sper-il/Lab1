# Practice 1 — PyTorch FashionMNIST Classification

> Single source of truth cho **đang làm gì, tại sao, và đã làm được gì**.
> Cập nhật lần cuối: 2026-10-11 17:20

---

## Project Context (Phase 1 — DEFINE)

| Trường | Giá trị |
|--------|---------|
| **Tên đầy đủ** | Practice 1 — PyTorch FashionMNIST Classification |
| **Slug** | `lab1` |
| **Bài toán** | Phân loại ảnh xám 28×28 từ dataset FashionMNIST vào 10 lớp (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot). |
| **Người dùng mục tiêu** | Sinh viên đang học PyTorch end-to-end (mục tiêu chính là **học**, không phải benchmark). |
| **Input** | Ảnh grayscale 1×28×28, giá trị pixel [0, 255]. |
| **Output** | Label dự đoán (1 trong 10 lớp) + xác suất từng lớp. |
| **Tính năng cốt lõi** | **Must**: load data → transform → build model → training loop → evaluate accuracy → save/load. **Should**: visualize loss curve, hiển thị ảnh dự đoán vs thực tế. **Could**: experiment so sánh hyperparameter. |
| **Tech construction** | Python 3.10+, PyTorch 2.x, torchvision, matplotlib, numpy. Auto-detect CUDA → CPU fallback. |
| **Tiêu chí đánh giá** | **Model level**: test accuracy ≥ 88% (chuẩn CNN cơ bản trên FashionMNIST), training loss giảm đều. **Pipeline level**: training xong trong < 10 phút trên CPU, code chạy được từ `python -m src.train`. |

### Bộ câu hỏi bắt buộc (đã trả lời)

**What?**
- Bài toán: image classification 10 lớp, grayscale 28×28.
- Data: FashionMNIST — 60.000 train + 10.000 test, có sẵn qua `torchvision.datasets.FashionMNIST`.
- Output: class index (0–9) + vector xác suất.
- Chia sẵn train/test bởi torchvision; ta thêm validation split 10.000 mẫu từ train (chia 50.000 train / 10.000 val).

**Why?**
- Tại sao CNN? — Ảnh có cấu trúc không gian cục bộ; CNN tận dụng được locality và translation invariance, phù hợp hơn MLP. Vẫn đủ nhỏ để hiểu được từng layer.
- Tại sao FashionMNIST thay vì MNIST? — Đề yêu cầu. FashionMNIST khó hơn MNIST nên phân biệt được model yếu/mạnh.
- Tại sao dùng `torchvision.datasets.FashionMNIST`? — Dataset có sẵn, tải một lần, reproducible, chuẩn hóa format.
- Tại sao normalize mean=0.2860, std=0.3530? — Đây là mean/std thực nghiệm của FashionMNIST tính trên toàn bộ training set (tham khảo docs). Normalize giúp training ổn định hơn so với chia 255.

**Where / When / Which?**
- **Where**: local machine, auto-detect CUDA → fallback CPU.
- **When**: chạy một lần cho baseline, sau đó chạy grid nhỏ cho experiment.
- **Which**:
  - Framework: **PyTorch 2.x** (đề bắt buộc).
  - Data: **`torchvision.datasets.FashionMNIST`** + **`torchvision.transforms`**.
  - Viz: **matplotlib** (cho loss curve + ảnh dự đoán).
  - Tracking: in metric ra console, lưu JSON vào `results/runs/<ts>/` (không dùng W&B để giữ đơn giản).
  - Model: **CNN 2 lớp conv + 2 lớp FC** (tự định nghĩa, không dùng model pretrained).

**How?**
- Pipeline: `raw FashionMNIST → ToTensor → Normalize → DataLoader → CNN → CrossEntropyLoss → Adam → save .pt`.
- Hyperparameter baseline: `lr=1e-3, batch_size=64, epochs=10, optimizer=Adam`.
- Augment: chỉ ToTensor + Normalize (FashionMNIST đã chuẩn hóa, augment phức tạp không cần cho bài học).
- Post-processing: lấy `argmax` của logits, so sánh với label thật.

**Evaluation**
- **Model level**: accuracy trên test set, loss trên train/val mỗi epoch, confusion matrix (bonus).
- **Full pipeline**: thời gian training tổng, thời gian inference / 1000 mẫu, kích thước file .pt.
- **Tiêu chí đạt**: test accuracy ≥ 88%, training loss giảm đều (không NaN, không explode), pipeline chạy ổn định end-to-end 2 lần liên tiếp.

**Điểm số / Score mong đợi (nếu có)**
- TBD — phụ thuộc rubric giảng viên. Mặc định giả định:
  - Code chạy được: 30%.
  - Đúng yêu cầu đề (7 mục trong Deliver): 40%.
  - Loss graphs + ảnh dự đoán: 20%.
  - Report tóm tắt: 10%.

---

## Architecture Snapshot (Phase 2 — DESIGN)

### Pattern
- **Single-script + module structure** — code production theo module (`src/`), nhưng có 1 entry point `python -m src.train` chạy end-to-end.
- Tách rõ **data / model / train / evaluate / visualize** để dễ test từng phần.

### Modules

| Module | Path | Trách nhiệm |
|--------|------|-------------|
| `data` | `src/data/` | Load FashionMNIST, áp transform, trả về DataLoader (train / val / test). |
| `models` | `src/models/` | Định nghĩa kiến trúc CNN (`FashionCNN`). |
| `training` | `src/training/` | Training loop: forward, loss, backward, optimize, log metric, save checkpoint. |
| `evaluation` | `src/evaluation/` | Đánh giá trên test set, tính accuracy, vẽ confusion matrix (optional). |
| `visualization` | `src/visualization/` | Vẽ loss curve, hiển thị ảnh dự đoán vs thực tế. |
| `train` | `src/train.py` | Entry point gọi tất cả module trên theo thứ tự. |

### Folder structure
```
lab1/
├── data/
│   └── raw/                  # FashionMNIST tải về (gitignore)
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py        # build_dataloaders(config) → {train, val, test}
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn.py            # FashionCNN(nn.Module)
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py        # train_one_epoch, validate, fit
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py       # test_accuracy, classification_report
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py          # plot_loss_curve, show_predictions
│   ├── __init__.py
│   └── train.py              # python -m src.train
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_train_smoke.py   # 1 epoch để đảm bảo loop chạy
├── configs/
│   └── baseline.yaml         # lr, batch_size, epochs, optimizer, seed
├── results/
│   └── runs/
│       └── <timestamp>/
│           ├── metrics.json
│           ├── config.yaml
│           ├── loss_curve.png
│           ├── predictions.png
│           └── model.pt
├── notebooks/
│   └── 2026-10-11_eda.ipynb  # EDA: xem 1 vài ảnh mẫu
├── docs/
│   ├── architecture.md
│   ├── data-flow.md
│   └── api.md
├── .gitignore
├── README.md
├── AI_Usage.md
├── progress_status.md
├── requirements.txt
└── đề.txt
```

### Data flow
```
torchvision FashionMNIST (28×28, 1 channel, [0,255])
        │
        ▼
   ToTensor → [0,1] float32
        │
        ▼
Normalize(mean=0.2860, std=0.3530)
        │
        ▼
   DataLoader (batch_size=64, shuffle train)
        │
        ▼
   FashionCNN (forward) → logits [batch, 10]
        │
        ▼
CrossEntropyLoss(logits, labels)
        │
        ▼
   Adam.step() → cập nhật weights
        │
        ▼
   mỗi epoch: validate → lưu metric
        │
        ▼
sau train: test accuracy + save model.pt
        │
        ▼
plot_loss_curve, show_predictions (matplotlib)
        │
        ▼
results/runs/<ts>/metrics.json + ảnh PNG
```

### Interfaces / APIs

| Hàm | Input | Output | Ghi chú |
|-----|-------|--------|---------|
| `build_dataloaders(config)` | dict config | `{train, val, test}` DataLoaders | Download lần đầu vào `data/raw/`. |
| `FashionCNN()` | — | `nn.Module` | 2 conv (32, 64 filter) + 2 FC (128, 10). |
| `train_one_epoch(model, loader, criterion, optimizer, device)` | các thành phần | `(avg_loss, avg_acc)` | 1 epoch train. |
| `validate(model, loader, criterion, device)` | các thành phần | `(avg_loss, avg_acc)` | Không cập nhật weights. |
| `fit(model, train_loader, val_loader, config, device)` | config | history dict, best model | Loop toàn bộ epoch, save best. |
| `test_accuracy(model, loader, device)` | model + loader | float | % accuracy trên test. |
| `plot_loss_curve(history, path)` | history dict | file PNG | Train loss + val loss. |
| `show_predictions(model, loader, n=10, path)` | model + loader | file PNG | Ảnh + label dự đoán vs thật. |

### Milestones

| # | Mốc | Mô tả | Tiêu chí hoàn thành |
|---|-----|-------|---------------------|
| M0 | Scaffold | Tạo folder, `.gitignore`, `requirements.txt`, `progress_status.md` | Folder đúng cấu trúc, import được. |
| M1 | Data module | `build_dataloaders` chạy được, tải FashionMNIST thành công | In được 1 batch shape `[64, 1, 28, 28]`. |
| M2 | Model module | `FashionCNN` forward pass ra đúng shape `[batch, 10]` | Test 1 forward, so sánh shape. |
| M3 | Training loop | `fit` chạy 1 epoch đầu, loss giảm | Test smoke pass, loss epoch 1 < loss ban đầu. |
| M4 | Evaluation + save/load | Test accuracy ≥ 88%, save/load model roundtrip OK | Roundtrip cho cùng accuracy. |
| M5 | Visualization | Loss curve PNG + ảnh dự đoán PNG xuất ra `results/runs/<ts>/` | 2 file PNG tồn tại, mở được. |
| M6 | Hyperparameter experiment | Chạy 2-3 config (vd: lr×optimizer), so sánh | Bảng kết quả trong `results/`. |
| M7 | Docs cuối | `README.md` 10 mục, `AI_Usage.md` 7 mục, report ngắn | Đủ checklist Definition of Done. |

### Technical risks

| Rủi ro | Khả năng | Tác động | Giảm thiểu | Phát hiện |
|--------|----------|----------|------------|-----------|
| CUDA không khả dụng | Trung bình | Train chậm hơn 5-10× | Auto-detect, fallback CPU; giảm epochs cho CPU | In `device` lúc khởi động. |
| Overfitting train/val | Trung bình | Val accuracy chững khi train tiếp tục giảm | Validation mỗi epoch, save best model theo val acc | Log val loss song song train loss. |
| Loss = NaN | Thấp | Train hỏng | Gradient clipping `max_norm=1.0`, log loss mỗi batch | Check `torch.isnan(loss)` trước `backward()`. |
| FashionMNIST tải về chậm / fail | Thấp | Pipeline không chạy | Cache ở `data/raw/`, retry với `download=True` | Bắt exception, hướng dẫn tải thủ công nếu cần. |
| Không reproduce được kết quả | Thấp | Khó debug | Set `torch.manual_seed`, `numpy seed`, lưu config cùng metric | So sánh metric giữa 2 lần chạy. |
| Quên save model / path sai | Thấp | Không load lại được | `results/runs/<ts>/model.pt` là path duy nhất, log lúc save | Test roundtrip save → load. |

### Open Questions
- [ ] Có cần confusion matrix trong report không? (Đề không yêu cầu, nhưng giúp phân tích lỗi.) → Tôi sẽ thêm nếu kịp.
- [ ] Số epochs tối ưu? Baseline 10, có thể tăng 15-20 nếu val acc còn tăng. → Quyết định sau M3.
- [ ] Có dùng learning rate scheduler không? → Có thể thêm `ReduceLROnPlateau` ở M6 nếu cần.
- [ ] Báo cáo PDF hay Markdown? → Markdown cho dễ version control, convert PDF nếu giảng viên yêu cầu.

---

## Active Module: M0 — Scaffold project

- **Describe** — Tạo folder structure, `.gitignore`, `requirements.txt`, các file `__init__.py` rỗng. Chưa code logic.
- **Goal / purpose** — Có khung project sạch, reproducible, sẵn sàng cho M1.
- **Pipeline** — (chưa có pipeline ở module này).
- **How / experiment plan (sub-tasks)**
  - [ ] Tạo `src/{data,models,training,evaluation,visualization}/` với `__init__.py`.
  - [ ] Tạo `tests/`, `configs/`, `results/runs/`, `notebooks/`, `docs/`.
  - [ ] Viết `requirements.txt` (torch, torchvision, matplotlib, numpy, pyyaml, pytest).
  - [ ] Viết `configs/baseline.yaml` (lr, batch_size, epochs, optimizer, seed).
- **Status** — Describe ✅ / Generate ✅ / Review ⏳ / Run ⏳ / Test ⏳ / Refactor ⏳ / Commit ⏳
- **Last run** — chưa có.
- **DoD checklist** — Folder đúng cấu trúc, file YAML hợp lệ, requirements cài được trên máy sạch.

---

## Module Log

| # | Module | Ngày | Status | Commit | Ghi chú |
|---|--------|------|--------|--------|---------|
| 0 | Define + Design | 2026-10-11 | ✅ | — | Phase 1 + Phase 2 xong, scaffold folder. |
| 1 | M0 — Scaffold | 2026-10-11 | ⏳ | — | Tạo folder + file rỗng. |

---

## Decisions Log

| Ngày | Quyết định | Lựa chọn khác | Lý do |
|------|------------|---------------|-------|
| 2026-10-11 | Chọn CNN 2 conv + 2 FC thay vì MLP | MLP, ResNet | Cân bằng giữa dễ hiểu và đủ mạnh; CNN là kiến thức nền tảng phải nắm. |
| 2026-10-11 | Chọn PyTorch thuần, không dùng Lightning/ignite | PyTorch Lightning | Bài học yêu cầu hiểu training loop thủ công (forward/loss/backward/optimize). |
| 2026-10-11 | Normalize theo mean/std FashionMNIST thực nghiệm | Chia 255, ImageNet stats | Ổn định hơn, là chuẩn cho bài classification. |
| 2026-10-11 | Auto-detect device | Ép CPU, ép CUDA | Linh hoạt cho cả 2 môi trường. |
| 2026-10-11 | Experiment grid: lr × optimizer | Batch size, depth | 2 yếu tố ảnh hưởng convergence rõ nhất cho người mới. |

---

## Risks & Mitigations
*(xem bảng Technical risks ở trên)*

---

## Definition of Done (áp dụng mỗi module)
- [ ] Đúng yêu cầu
- [ ] Code chạy end-to-end
- [ ] Tests pass
- [ ] Không có secrets / không có artifact lớn
- [ ] Code đã self-review
- [ ] Có thể giải thích được
- [ ] Docs đã cập nhật
- [ ] Đã committed (Conventional Commit)

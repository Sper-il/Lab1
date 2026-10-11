# Practice 1 — PyTorch FashionMNIST Classification

Phân loại ảnh grayscale 28×28 từ dataset **FashionMNIST** vào **10 lớp**
(T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).

> **Bài tập:** Practice 1 — PyTorch end-to-end (load → transform → model → train → evaluate → save/load).
> **Mục tiêu:** Học PyTorch, không benchmark.

## Cấu trúc project

```
lab1/
├── configs/baseline.yaml          # Hyperparameter (data, model, train)
├── data/raw/                      # 70.000 ảnh .png (sinh bởi M1)
│   ├── train/<class>/*.png        # 50.000
│   ├── val/<class>/*.png          # 10.000
│   └── test/<class>/*.png         # 10.000
├── src/
│   ├── data/                      # M1 ✅
│   │   ├── split.py               # split_fashionmnist_to_disk()
│   │   └── dataset.py             # build_dataloaders(), summarize()
│   ├── models/                    # M2 ⏳
│   ├── training/                  # M3 ⏳
│   ├── evaluation/                # M4 ⏳
│   └── visualization/             # M5 ⏳
├── tests/                         # pytest song song theo module
├── docs/
│   ├── architecture.md            # Kiến trúc tổng quan
│   ├── data-flow.md               # Luồng dữ liệu chi tiết
│   └── api.md                     # API nội bộ
├── progress_status.md             # Nhật ký theo vòng lặp 7 bước
├── requirements.txt
└── README.md                      # File này
```

## Cài đặt

Yêu cầu: **Python 3.10+**, pip.

```bash
# Tạo virtualenv (khuyến nghị)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Cài dependencies
pip install -r requirements.txt
```

Nội dung `requirements.txt`:

```
torch>=2.0
torchvision
numpy
matplotlib
Pillow
pyyaml
pytest
```

## Chạy nhanh

Pipeline đầy đủ gồm 3 bước:

### Bước 1 — Sinh dữ liệu trên disk (1 lần duy nhất, ~2-3 phút)

```bash
python -m src.data.split --root data/raw --val-split 10000 --seed 42
```

Sau khi xong: `data/raw/{train,val,test}/<class_id>/*.png` (tổng 70.000 file).

### Bước 2 — Kiểm tra 3 DataLoader

```bash
python -c "from src.data.dataset import build_dataloaders, summarize; loaders = build_dataloaders('configs/baseline.yaml'); [summarize(l, n) for n, l in loaders.items()]"
```

In ra số mẫu, số batch, classes, shape, dtype, pixel range cho cả 3 loader.

### Bước 3 — Train + Evaluate + Visualize (sau khi M2–M5 sẵn sàng)

```bash
python -m src.train --config configs/baseline.yaml
```

Lưu kết quả vào `results/runs/<timestamp>/`: `metrics.json`, `model.pt`,
`loss_curve.png`, `predictions.png`.

## Chạy test

```bash
pytest -v                          # tất cả test
pytest tests/test_dataset.py -v    # chỉ test data (11 test)
```

**Trạng thái test hiện tại:** 11/11 pass (M1).

## Quy trình phát triển

Mỗi module (M0 → M5) đi qua **vòng lặp 7 bước** trong skill `ai-fusion-workflow`:

1. **Describe** — mô tả trong `progress_status.md`
2. **Generate** — viết code
3. **Review** — self-check logic
4. **Run** — chạy thực tế, in kết quả
5. **Test** — pytest
6. **Refactor** — sửa warning / code smell
7. **Commit** — Conventional Commit, cập nhật `progress_status.md`

| Module | Trạng thái | Commit |
|--------|------------|--------|
| **M0** — Scaffold | ✅ done | (initial) |
| **M1** — Data | ✅ done | `ff2e11a` |
| **M2** — Model | ⏳ sẵn sàng | — |
| **M3** — Training | ⏳ | — |
| **M4** — Evaluation | ⏳ | — |
| **M5** — Visualization | ⏳ | — |
| **M6** — Experiment grid | ⏳ (Phase 2) | — |

## Tài liệu đọc thêm

- **[`progress_status.md`](progress_status.md)** — nhật ký chi tiết từng module,
  decisions log, interfaces/APIs, evaluation criteria.
- **[`docs/architecture.md`](docs/architecture.md)** — kiến trúc 5 module, design conventions.
- **[`docs/data-flow.md`](docs/data-flow.md)** — luồng dữ liệu end-to-end.
- **[`docs/api.md`](docs/api.md)** — API nội bộ từng hàm (input/output/side effect).
- **[`AI_Usage.md`](AI_Usage.md)** — ghi nhận việc sử dụng AI trong quá trình làm bài.

## Tài liệu tham khảo

- [PyTorch Tensors tutorial](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Build Model](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [Autograd](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [Optimization](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Save / Load Model](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)

## License

Dự án phục vụ mục đích học tập (Practice 1).

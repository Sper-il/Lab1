# Quy ước dữ liệu (Data Convention)

> Single source of truth cho mọi quy ước dữ liệu của project. Số liệu thật xem `results/runs/dataset_info.md`.

## 1. Nguồn dữ liệu

- Gốc: `torchvision.datasets.FashionMNIST` (~30 MB), tải về `data/processed/FashionMNIST/raw/` (file `idx3/idx1-ubyte` gốc).
- Không commit `data/` (xem `.gitignore`); mỗi máy tự tải bằng `python -m src.data.split`.
- Metadata text duy nhất được commit: `data/splits/split_info.txt`.

## 2. Định dạng và shape

| Thuộc tính | Giá trị |
|------------|---------|
| Kích thước ảnh | 28 × 28 pixel, 1 kênh xám |
| Shape tensor | `(1, 28, 28)`; batch `(B, 1, 28, 28)` |
| Dtype | `float32`, pixel `[0, 1]` trước chuẩn hóa |
| Label | int `0–9`, mapping trong `src/data/dataset.py` (`CLASS_NAMES`) |
| Output model | logits `(B, 10)` — chưa softmax (dùng với `CrossEntropyLoss`) |

10 lớp: 0 T-shirt/top, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat, 5 Sandal, 6 Shirt, 7 Sneaker, 8 Bag, 9 Ankle boot.

## 3. Chia tập (split)

- 60.000 train gốc → **50.000 train / 10.000 val** (stratified, `seed=42`, `DEFAULT_VAL_RATIO = 1/6` trong `src/data/split.py`).
- Test giữ nguyên **10.000** của FashionMNIST, không chia lại.
- Mỗi lớp đúng **5.000 / 1.000 / 1.000** (train/val/test). Tỷ lệ 5/6–1/6 được chọn vì chỉ nó cho số tròn mỗi lớp từ 6.000 ảnh gốc.
- Entry chuẩn: `from src.data.split import split_pipeline` → `train_loader, val_loader, test_loader, split_info = split_pipeline(batch_size=128)`.
- Số batch (bs=128): train 391, val 79, test 79.

## 4. Transforms và chuẩn hóa

- Train: `ToTensor` + `Normalize(mean=0,2860, std=0,3530)` (giá trị từ EDA `src/data/eda.py`).
- Val/test: cùng normalize, không augmentation (augment tắt ở Phase hiện tại).
- Công thức: `normalized = (pixel − 0,2860) / 0,3530`; ảnh sau chuẩn hóa nằm trong `[-0,8102, 2,0227]`.
- Khi hiển thị predicted vs actual (`src/evaluation/visualizer.py`) cần squeeze `(1, 28, 28) → (28, 28)` và cmap gray.

## 5. Chất lượng dữ liệu

- 60.000/60.000 mẫu hợp lệ (100%), 0 NaN/Inf, 0 ảnh hỏng, 0 label ngoài `[0, 9]` (check bởi `DataCleaner` trong `src/data/preprocessing.py`).
- Cân bằng hoàn hảo: 6.000/lớp trong tập gốc.
- Lưu ý lịch sử: `src/data/split.py` từng có bug `_stratified_split` (thiếu shuffle toàn cục → val lệch 947–1.040/lớp); đã fix bằng `np.random.shuffle(stratified_indices)` — xem báo cáo Người 1.

## 6. Artifact Phase 1 cũ (không còn đọc)

- `data/processed/{train,val,test}/<class_id>/*.png` (70k PNG) là artifact của pipeline Phase 1 cũ, **không còn được đọc** bởi `run_training.py` / `run_experiments.py` (đọc tensor gốc qua `split_pipeline`).
- EDA plots thật: `results/runs/class_distribution.png`, `sample_images.png`, `pixel_histogram.png`, `per_class_samples.png`.
- Embedding 2D (2.000 ảnh test, `python src/data/embedding_viz.py --n 2000 --thumbs 600`): `embedding_pca_color.png`, `embedding_pca_thumbs.png`, `embedding_tsne_color.png`, `embedding_tsne_thumbs.png` (PCA giải thích ~47% phương sai, t-SNE perplexity=30; UMAP bỏ qua vì lab chưa cài `umap-learn`). Cả 4 ảnh đã nhúng trong `results/runs/report.html` mục 8.
- PCA 3D (3.000 ảnh test, `python src/evaluation/pca_3d.py --samples 3000 --seed 42`): `pca_3d.png`, `pca_explained.png`, `pca_3d.json` (3PC giữ 53,3% phương sai; viewer 3D nhúng trực tiếp trong `results/runs/report.html` mục 9, không còn file HTML rời). Ảnh tĩnh đã nhúng trong `results/runs/report.html` mục 8.
- t-SNE 3D (2.000 ảnh test, `python src/evaluation/tsne_3d.py --samples 2000 --seed 42 --perplexity 30`): `tsne_3d.png`, `tsne_3d.json` (cụm tách rõ hơn PCA, đúng kiểu ảnh mẫu; viewer nhúng trong `report.html` mục 9). Ảnh tĩnh đã nhúng trong `results/runs/report.html` mục 8.
- UMAP 3D (2.000 ảnh test, `python src/evaluation/umap_3d.py --samples 2000 --seed 42`): `umap_3d.png`, `umap_3d.json` (n_neighbors=15, min_dist=0,1; viewer nhúng trong `report.html` mục 9). Ảnh tĩnh đã nhúng trong `results/runs/report.html` mục 8.
- PyMDE 3D (2.000 ảnh test, `python src/evaluation/mde_3d.py --samples 2000 --seed 42`): `mde_3d.png`, `mde_3d.json` (preserve_neighbors trên CPU; viewer nhúng trong `report.html` mục 9). Ảnh tĩnh đã nhúng trong `results/runs/report.html` mục 8.

## 7. Tái tạo

```bash
# Tải gốc + chia tập (1 lần, ~2-3 phút)
python -m src.data.split --root data/processed --val-split 10000 --seed 42

# Kiểm tra metadata
cat data/splits/split_info.txt

# Chạy tests
$env:PYTHONPATH = "."; python -m pytest tests -q   # 27 passed
```

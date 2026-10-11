# EDA Report — FashionMNIST

Output dir: `results\eda`
Config normalize: mean=0.286, std=0.353

## Sanity check mỗi split

| Split | n | shape | dtype | min | max | mean | std |
|-------|---|-------|-------|-----|-----|------|-----|
| train | 50,000 | (1, 28, 28) | torch.float32 | -0.810 | 2.023 | 0.0010 | 1.0003 |
| val | 10,000 | (1, 28, 28) | torch.float32 | -0.810 | 2.023 | -0.0044 | 0.9991 |
| test | 10,000 | (1, 28, 28) | torch.float32 | -0.810 | 2.023 | 0.0024 | 0.9984 |

> Nếu `mean ≈ 0` và `std ≈ 1` thì config Normalize khớp với dataset. Khác xa → sai mean/std.

## Class distribution

| Class | Name | train | val | test |
|-------|------|-------|-----|------|
| 0 | T-shirt/top | 4955 | 1045 | 1000 |
| 1 | Trouser | 4938 | 1062 | 1000 |
| 2 | Pullover | 5063 | 937 | 1000 |
| 3 | Dress | 4990 | 1010 | 1000 |
| 4 | Coat | 5041 | 959 | 1000 |
| 5 | Sandal | 4957 | 1043 | 1000 |
| 6 | Shirt | 4994 | 1006 | 1000 |
| 7 | Sneaker | 4996 | 1004 | 1000 |
| 8 | Bag | 5030 | 970 | 1000 |
| 9 | Ankle boot | 5036 | 964 | 1000 |

**Train imbalance ratio (max/min)**: 1.03

## Data quality

| Split | Blank (mean=0) | Near-constant (std<1e-3) | NaN/Inf pixels |
|-------|----------------|--------------------------|----------------|
| train | 0 | 0 | 0 |
| val | 0 | 0 | 0 |
| test | 0 | 0 | 0 |

## Figures

- `figures/sample_grid.png` — 1 ảnh mỗi class (train).
- `figures/class_distribution.png` — phân phối nhãn 3 split.
- `figures/pixel_stats.png` — histogram pixel (train).

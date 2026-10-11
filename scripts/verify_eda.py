"""Verify EDA pipeline — tương đương việc chạy tất cả cells của 01_eda.ipynb
nhưng bằng Python script (ổn định hơn nbconvert, không phụ thuộc Jupyter kernel).

Mục đích: CI/quick-check rằng EDA pipeline vẫn chạy được sau khi thay đổi code/data.

Output:
  - Console: in shape/dtype/distribution cho 3 split
  - artifacts/eda_summary.txt: text summary
  - artifacts/eda_samples.png: 3x10 grid (raw vs train tfm vs eval tfm)
"""
from __future__ import annotations

import sys
from pathlib import Path

# === Setup: giống cell 1 của notebook ===
NOTEBOOK_DIR = Path.cwd()
sys.path.insert(0, str(NOTEBOOK_DIR))

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.data.dataset import (
    build_dataloaders,
    _build_train_transform,
    _build_eval_transform,
    FASHION_MNIST_MEAN,
    FASHION_MNIST_STD,
    summarize,
)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# Override num_workers cho EDA (tránh Windows multiprocessing spawn issue)
def build_eda_loaders(config_path: str = "configs/baseline.yaml"):
    cfg_path = Path(config_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["num_workers"] = 0
    tmp_cfg = NOTEBOOK_DIR / "configs" / "_eda_override.yaml"
    tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return build_dataloaders(tmp_cfg)

print(f"[verify_eda] Project root: {NOTEBOOK_DIR}")
print(f"[verify_eda] PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"[verify_eda] FashionMNIST mean/std: {FASHION_MNIST_MEAN} / {FASHION_MNIST_STD}")

# Load loaders (num_workers=0)
loaders = build_eda_loaders()
print(f"[verify_eda] Loaders: {list(loaders.keys())}\n")

# === Cell 2: summarize ===
print("=" * 60)
print("CELL 2: SHAPE / DTYPE / SIZES")
print("=" * 60)
for name in ("train", "val", "test"):
    summarize(loaders[name], name)

# === Cell 3: class distribution ===
print("=" * 60)
print("CELL 3: CLASS DISTRIBUTION")
print("=" * 60)
from collections import Counter

def count_classes(loader):
    ds = loader.dataset
    return Counter(ds.samples[i][1] for i in range(len(ds)))

counts = {name: count_classes(loaders[name]) for name in ("train", "val", "test")}

# Text summary
summary_lines = []
summary_lines.append("FashionMNIST EDA — Summary\n")
summary_lines.append(f"Mean/std: {FASHION_MNIST_MEAN} / {FASHION_MNIST_STD}\n")
summary_lines.append(f"{'Class':<16}{'Train':>10}{'Val':>10}{'Test':>10}\n")
for i, name in enumerate(CLASS_NAMES):
    n_tr = counts["train"][i]
    n_va = counts["val"][i]
    n_te = counts["test"][i]
    summary_lines.append(f"{name:<16}{n_tr:>10}{n_va:>10}{n_te:>10}\n")
summary_text = "".join(summary_lines)
print(summary_text)

# Bar chart
artifacts_dir = NOTEBOOK_DIR / "artifacts"
artifacts_dir.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
for ax, split in zip(axes, ("train", "val", "test")):
    counts_split = [counts[split][i] for i in range(10)]
    bars = ax.bar(range(10), counts_split, color="steelblue", edgecolor="black")
    ax.set_xticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_title(f"{split} (n={sum(counts_split)})")
    ax.set_ylabel("# samples")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for b, c in zip(bars, counts_split):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 50,
                str(c), ha="center", va="bottom", fontsize=8)
plt.suptitle("Class distribution per split", y=1.02)
plt.tight_layout()
out_dist = artifacts_dir / "eda_class_distribution.png"
plt.savefig(out_dist, dpi=100, bbox_inches="tight")
plt.close()
print(f"[verify_eda] Saved: {out_dist.relative_to(NOTEBOOK_DIR)}")

# === Cell 4: visualization grid ===
print("\n" + "=" * 60)
print("CELL 4: SAMPLE GRID (raw vs train tfm vs eval tfm)")
print("=" * 60)
ROOT = Path("data/raw")
raw_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
train_tfm = _build_train_transform(FASHION_MNIST_MEAN, FASHION_MNIST_STD)
eval_tfm = _build_eval_transform(FASHION_MNIST_MEAN, FASHION_MNIST_STD)

raw_ds = ImageFolder(str(ROOT / "train"), transform=raw_transform)
first_idx_per_class = {}
for idx, (_, label) in enumerate(raw_ds.samples):
    if label not in first_idx_per_class:
        first_idx_per_class[label] = idx
    if len(first_idx_per_class) == 10:
        break

from torchvision.transforms.functional import to_pil_image

def unnormalize(t):
    mean = torch.tensor([FASHION_MNIST_MEAN]).view(1, 1, 1)
    std = torch.tensor([FASHION_MNIST_STD]).view(1, 1, 1)
    return (t * std + mean).clamp(0, 1)

fig, axes = plt.subplots(3, 10, figsize=(20, 6))
for c in range(10):
    idx = first_idx_per_class[c]
    img_raw, label = raw_ds[idx]
    img_train = train_tfm(img_raw)
    img_eval = eval_tfm(img_raw)

    axes[0, c].imshow(np.asarray(img_raw), cmap="gray")
    axes[0, c].set_title(CLASS_NAMES[c], fontsize=9)
    axes[0, c].axis("off")

    axes[1, c].imshow(np.asarray(to_pil_image(unnormalize(img_train))), cmap="gray")
    axes[1, c].axis("off")

    axes[2, c].imshow(np.asarray(to_pil_image(unnormalize(img_eval))), cmap="gray")
    axes[2, c].axis("off")

axes[0, 0].set_ylabel("raw", rotation=0, ha="right", va="center", fontsize=10, labelpad=40)
axes[1, 0].set_ylabel("+ train\n transform", rotation=0, ha="right", va="center", fontsize=10, labelpad=40)
axes[2, 0].set_ylabel("+ eval\n transform", rotation=0, ha="right", va="center", fontsize=10, labelpad=40)
plt.suptitle("1 sample per class — raw vs. after transform", y=1.00, fontsize=12)
plt.tight_layout()
out_grid = artifacts_dir / "eda_samples.png"
plt.savefig(out_grid, dpi=100, bbox_inches="tight")
plt.close()
print(f"[verify_eda] Saved: {out_grid.relative_to(NOTEBOOK_DIR)}")
print(f"[verify_eda] Displayed {len(first_idx_per_class)}/10 classes")

# Save text summary
out_txt = artifacts_dir / "eda_summary.txt"
out_txt.write_text(summary_text, encoding="utf-8")
print(f"[verify_eda] Saved: {out_txt.relative_to(NOTEBOOK_DIR)}")

print("\n[verify_eda] DONE — all cells executed successfully")

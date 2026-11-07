"""PCA 3D cho FashionMNIST (sinh pca_3d.png + pca_explained.png + pca_3d.json).

Nguon du lieu: FashionMNIST test goc qua torchvision (data/processed/), lay N mau
dau tien, flatten 784, chuan hoa z-score (mean=0.2860, std=0.3530) nhu project.

Sinh ra trong results/runs/:
- pca_3d.png        (scatter 3D mau theo 10 lop)
- pca_explained.png (explained variance 10 PC dau)
- pca_3d.json       ({samples, seed, explained_ratio_10, explained_ratio_3, coords, labels})

Cach dung:
    python src/evaluation/pca_3d.py --samples 3000 --seed 42
    python src/evaluation/pca_3d_html.py   # render trang 3D tuong tac tu JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "results" / "runs"

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load_fashion(n: int):
    from torchvision.datasets import FashionMNIST

    for root in ["data/processed", "./data/processed", str(PROJECT_ROOT / "data" / "processed")]:
        try:
            ds = FashionMNIST(root=root, train=False, download=False)
            break
        except Exception:
            continue
    else:
        raise FileNotFoundError("Khong thay FashionMNIST o data/processed (can chay split 1 lan).")
    n = min(n, len(ds))
    imgs = ds.data[:n].numpy().astype(np.float32) / 255.0
    labels = ds.targets[:n].numpy()
    flat = imgs.reshape(n, -1)
    return imgs, labels, flat


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PCA 3D cho FashionMNIST.")
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from sklearn.decomposition import PCA

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _, labels, flat = load_fashion(args.samples)
    flat_z = (flat - 0.2860) / 0.3530

    pca10 = PCA(n_components=10, random_state=args.seed)
    pca10.fit(flat_z)
    ratio10 = [float(v) for v in pca10.explained_variance_ratio_]
    ratio3 = float(np.sum(pca10.explained_variance_ratio_[:3]))
    coords = pca10.transform(flat_z)[:, :3]
    print(f"3PC={ratio3 * 100:.1f}%")

    # Scatter 3D mau theo lop.
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                    c=labels, cmap="tab10", s=4, alpha=0.6)
    fig.colorbar(sc, ax=ax, ticks=range(10))
    ax.set_title(f"PCA 3D Fashion-MNIST test ({len(labels)} anh, 3PC={ratio3 * 100:.1f}%)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.tight_layout()
    fig.savefig(RUNS_DIR / "pca_3d.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Explained variance 10 PC.
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.bar(range(1, 11), np.array(ratio10) * 100, color="#1f77b4")
    ax2.set_xlabel("Thanh phan chinh")
    ax2.set_ylabel("Phuong sai giai thich (%)")
    ax2.set_title("PCA explained variance 10 PC dau")
    for i, v in enumerate(ratio10):
        ax2.text(i + 1, v * 100 + 0.2, f"{v * 100:.1f}", ha="center", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(RUNS_DIR / "pca_explained.png", dpi=110, bbox_inches="tight")
    plt.close(fig2)

    payload = {
        "samples": int(len(labels)),
        "seed": int(args.seed),
        "explained_ratio_10": ratio10,
        "explained_ratio_3": ratio3,
        "coords": [[float(a), float(b), float(c)] for a, b, c in coords.tolist()],
        "labels": [int(v) for v in labels.tolist()],
    }
    (RUNS_DIR / "pca_3d.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote pca_3d.png + pca_explained.png + pca_3d.json ({len(labels)} samples, 3PC={ratio3 * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

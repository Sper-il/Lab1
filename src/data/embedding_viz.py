"""Vẽ embedding 2D cho FashionMNIST (PCA / t-SNE / UMAP nếu có).

Mục tiêu: visualization dữ liệu giống ảnh mẫu (thumbnail scatter + color scatter).
Chỉ dùng matplotlib + numpy + sklearn (+ umap nếu có), không cần seaborn/torch.

Nguồn dữ liệu: FashionMNIST test gốc qua torchvision (data/processed/), lấy N mẫu
đầu tiên, flatten 784, chuẩn hóa z-score nhẹ để PCA/t-SNE ổn định.

Sinh ra trong results/runs/:
- embedding_pca_color.png   (scatter màu theo 10 lớp)
- embedding_pca_thumbs.png  (thumbnail 28x28 tại vị trí PCA)
- embedding_tsne_color.png
- embedding_tsne_thumbs.png
- embedding_umap_color.png  (chỉ khi `import umap` thành công, ngược lại bỏ qua)

Cách dùng:
    python src/data/embedding_viz.py --n 2000 --thumbs 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

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
        raise FileNotFoundError("Không thấy FashionMNIST ở data/processed (cần chạy split 1 lần).")
    n = min(n, len(ds))
    imgs = ds.data[:n].numpy().astype(np.float32) / 255.0  # (N,28,28)
    labels = ds.targets[:n].numpy()
    flat = imgs.reshape(n, -1)
    return imgs, labels, flat


def scatter_color(xy: np.ndarray, labels: np.ndarray, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab10", s=6, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax, ticks=range(10))
    cbar.set_ticklabels(CLASS_NAMES)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def scatter_thumbs(xy: np.ndarray, imgs: np.ndarray, title: str, out: Path, n_thumbs: int = 600) -> None:
    # Chuẩn hóa embedding về [0,1] để đặt thumbnail không đè quá nhiều.
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    norm = (xy - lo) / (hi - lo + 1e-9)
    idx = np.linspace(0, len(norm) - 1, min(n_thumbs, len(norm))).astype(int)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.set_title(title, color="white")
    for i in idx:
        thumb = imgs[i]
        imagebox = OffsetImage(thumb, zoom=0.35, cmap="gray")
        ab = AnnotationBbox(imagebox, (norm[i, 0], norm[i, 1]), frameon=False, pad=0.1)
        ax.add_artist(ab)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vẽ embedding PCA/t-SNE/UMAP cho FashionMNIST.")
    parser.add_argument("--n", type=int, default=2000, help="Số ảnh test dùng (mặc định 2000).")
    parser.add_argument("--thumbs", type=int, default=600, help="Số thumbnail vẽ (mặc định 600).")
    args = parser.parse_args(argv)

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    imgs, labels, flat = load_fashion(args.n)
    print(f"Loaded {len(labels)} test images; flat {flat.shape}".encode("ascii", "ignore").decode())

    # Chuan hoa z-score theo mean/std project de embedding on dinh.
    flat_z = (flat - 0.2860) / 0.3530

    pca = PCA(n_components=2, random_state=42)
    xy_pca = pca.fit_transform(flat_z)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    scatter_color(xy_pca, labels, f"PCA on Fashion-MNIST test ({len(labels)} images)", RUNS_DIR / "embedding_pca_color.png")
    scatter_thumbs(xy_pca, imgs, f"PCA on Fashion-MNIST ({len(labels)} images)", RUNS_DIR / "embedding_pca_thumbs.png", args.thumbs)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, init="pca", learning_rate="auto")
    xy_tsne = tsne.fit_transform(flat_z)
    scatter_color(xy_tsne, labels, f"t-SNE on Fashion-MNIST test ({len(labels)} images, perplexity=30)", RUNS_DIR / "embedding_tsne_color.png")
    scatter_thumbs(xy_tsne, imgs, f"t-SNE on Fashion-MNIST ({len(labels)} images)", RUNS_DIR / "embedding_tsne_thumbs.png", args.thumbs)

    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=42)
        xy_umap = reducer.fit_transform(flat_z)
        scatter_color(xy_umap, labels, f"UMAP on Fashion-MNIST test ({len(labels)} images)", RUNS_DIR / "embedding_umap_color.png")
        print("UMAP done.")
    except Exception as e:  # pragma: no cover - moi truong lab thuong thieu umap
        print(f"Skip UMAP (umap-learn not installed): {type(e).__name__}")

    print(f"Wrote embedding PNGs to {RUNS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""So sanh Fashion-MNIST (trai) vs MNIST goc (phai) kieu Zalando.

Repo goc zalandoresearch/fashion-mnist muc Visualization:
- t-SNE on Fashion-MNIST (left) and original MNIST (right)  -> thumbnail nen den
- PCA  on Fashion-MNIST (left) and original MNIST (right)  -> thumbnail nen den
- UMAP on Fashion-MNIST (left) and original MNIST (right)  -> scatter mau nen trang
- PyMDE on Fashion-MNIST (left) and original MNIST (right) -> scatter mau nen trang + colorbar

Moi ben embedding RIENG (fit rieng tren tung dataset), lay N anh test dau tien,
flatten 784, chuan hoa z-score (mean/std project cho Fashion, mean/std MNIST cho MNIST).

Sinh ra trong results/runs/:
- compare_pca_thumbs.png
- compare_tsne_thumbs.png
- compare_umap_color.png
- compare_mde_color.png

Cach dung:
    python src/data/zalando_compare.py --n 2000 --thumbs 1500
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

FASHION_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load_both(n: int):
    from torchvision.datasets import FashionMNIST, MNIST

    root = str(PROJECT_ROOT / "data" / "processed")
    fds = FashionMNIST(root=root, train=False, download=True)
    mds = MNIST(root=root, train=False, download=True)
    n = min(n, len(fds), len(mds))
    f_imgs = fds.data[:n].numpy().astype(np.float32) / 255.0
    f_labels = fds.targets[:n].numpy()
    m_imgs = mds.data[:n].numpy().astype(np.float32) / 255.0
    m_labels = mds.targets[:n].numpy()
    return (f_imgs, f_labels), (m_imgs, m_labels)


def thumbs_ax(ax, xy: np.ndarray, imgs: np.ndarray, n_thumbs: int, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    n = len(xy)
    if n > n_thumbs:
        idx = rng.choice(n, size=n_thumbs, replace=False)
        idx = idx[np.argsort(xy[idx, 1])]
    else:
        idx = np.argsort(xy[:, 1])
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    norm = (xy - lo) / (hi - lo + 1e-9)
    zoom = 0.22 if len(idx) > 1000 else 0.28
    for i in idx:
        ab = AnnotationBbox(OffsetImage(imgs[i], zoom=zoom, cmap="gray"),
                            (norm[i, 0], norm[i, 1]), frameon=False, pad=0.0)
        ax.add_artist(ab)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([])
    ax.set_yticks([])


def save_thumbs_pair(xy_f, imgs_f, xy_m, imgs_m, title: str, out: Path, n_thumbs: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor("#0b0e14")
    for ax in axes:
        ax.set_facecolor("#0b0e14")
    fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=0.97)
    thumbs_ax(axes[0], xy_f, imgs_f, n_thumbs, seed=42)
    thumbs_ax(axes[1], xy_m, imgs_m, n_thumbs, seed=7)
    axes[0].set_title("Fashion-MNIST", color="white", fontsize=11)
    axes[1].set_title("original MNIST", color="white", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="#0b0e14")
    plt.close(fig)


def save_color_pair(xy_f, lb_f, xy_m, lb_m, title: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")
    for ax, xy, lb, t in zip(axes,
                             [xy_f, xy_m], [lb_f, lb_m],
                             ["Fashion-MNIST", "original MNIST"]):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=lb, cmap="tab10", s=5, alpha=0.7,
                        linewidths=0, edgecolors="none")
        ax.set_title(t, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="So sanh Fashion vs MNIST kieu Zalando.")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--thumbs", type=int, default=1500)
    args = parser.parse_args(argv)

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (f_imgs, f_lb), (m_imgs, m_lb) = load_both(args.n)
    print(f"Loaded Fashion {f_imgs.shape} + MNIST {m_imgs.shape}".encode("ascii", "ignore").decode())
    f_flat = f_imgs.reshape(len(f_imgs), -1)
    m_flat = m_imgs.reshape(len(m_imgs), -1)
    f_z = (f_flat - 0.2860) / 0.3530
    m_z = (m_flat - 0.1307) / 0.3081  # mean/std chuan cua MNIST

    # PCA (fit rieng moi ben, giong Zalando).
    pca = PCA(n_components=2, random_state=42)
    xy_pf = pca.fit_transform(f_z)
    xy_pm = PCA(n_components=2, random_state=42).fit_transform(m_z)
    save_thumbs_pair(xy_pf, f_imgs, xy_pm, m_imgs,
                     f"PCA on Fashion-MNIST (left) and original MNIST (right) — {args.n} test images each",
                     RUNS_DIR / "compare_pca_thumbs.png", args.thumbs)
    print("PCA pair done.")

    # t-SNE.
    xy_tf = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000,
                 init="pca", learning_rate="auto").fit_transform(f_z)
    xy_tm = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000,
                 init="pca", learning_rate="auto").fit_transform(m_z)
    save_thumbs_pair(xy_tf, f_imgs, xy_tm, m_imgs,
                     f"t-SNE on Fashion-MNIST (left) and original MNIST (right) — {args.n} test images each",
                     RUNS_DIR / "compare_tsne_thumbs.png", args.thumbs)
    print("t-SNE pair done.")

    # UMAP.
    try:
        import umap

        xy_uf = umap.UMAP(n_components=2, random_state=42).fit_transform(f_z)
        xy_um = umap.UMAP(n_components=2, random_state=42).fit_transform(m_z)
        save_color_pair(xy_uf, f_lb, xy_um, m_lb,
                        f"UMAP on Fashion-MNIST (left) and original MNIST (right) — {args.n} test images each",
                        RUNS_DIR / "compare_umap_color.png")
        print("UMAP pair done.")
    except Exception as e:
        print(f"Skip UMAP: {type(e).__name__}")

    # PyMDE.
    try:
        import torch
        import pymde

        torch.manual_seed(42)
        ef = pymde.preserve_neighbors(torch.from_numpy(f_z), embedding_dim=2,
                                      constraint=pymde.Standardized(),
                                      n_neighbors=15, verbose=False, device="cpu").embed(verbose=False).detach().cpu().numpy()
        em = pymde.preserve_neighbors(torch.from_numpy(m_z), embedding_dim=2,
                                      constraint=pymde.Standardized(),
                                      n_neighbors=15, verbose=False, device="cpu").embed(verbose=False).detach().cpu().numpy()
        save_color_pair(ef, f_lb, em, m_lb,
                        f"PyMDE on Fashion-MNIST (left) and original MNIST (right) — {args.n} test images each",
                        RUNS_DIR / "compare_mde_color.png")
        print("MDE pair done.")
    except Exception as e:
        print(f"Skip MDE: {type(e).__name__}: {e}")

    print(f"Wrote compare_*.png to {RUNS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

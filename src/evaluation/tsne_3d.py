"""t-SNE 3D cho FashionMNIST (sinh tsne_3d.png + tsne_3d.json).

Tuong tu `src/evaluation/pca_3d.py` nhung dung t-SNE 3 chieu thay vi PCA,
de co ban 3D giong hinh t-SNE 2D thumbnail mau (cum tach ro hon PCA).

Nguon du lieu: FashionMNIST test goc qua torchvision (data/processed/), lay N mau
dau tien, flatten 784, chuan hoa z-score (mean=0.2860, std=0.3530) nhu project.

Sinh ra trong results/runs/:
- tsne_3d.png   (scatter 3D mau theo 10 lop)
- tsne_3d.json  ({samples, seed, perplexity, coords, labels})

Cach dung:
    python src/evaluation/tsne_3d.py --samples 2000 --seed 42 --perplexity 30
    python src/evaluation/embed3d_html.py --input tsne_3d.json --output tsne_3d.html --title "t-SNE 3D"
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
    labels = ds.targets[:n].numpy()
    flat = ds.data[:n].numpy().astype(np.float32) / 255.0
    flat = flat.reshape(n, -1)
    return labels, flat


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="t-SNE 3D cho FashionMNIST.")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    args = parser.parse_args(argv)

    from sklearn.manifold import TSNE

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    labels, flat = load_fashion(args.samples)
    flat_z = (flat - 0.2860) / 0.3530

    tsne = TSNE(n_components=3, random_state=args.seed, perplexity=args.perplexity,
                max_iter=1000, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(flat_z)
    print(f"t-SNE 3D done: {coords.shape}")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                    c=labels, cmap="tab10", s=4, alpha=0.6)
    fig.colorbar(sc, ax=ax, ticks=range(10))
    ax.set_title(f"t-SNE 3D Fashion-MNIST test ({len(labels)} anh, perplexity={args.perplexity:g})")
    ax.set_xlabel("Dim1")
    ax.set_ylabel("Dim2")
    ax.set_zlabel("Dim3")
    fig.tight_layout()
    fig.savefig(RUNS_DIR / "tsne_3d.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "samples": int(len(labels)),
        "seed": int(args.seed),
        "perplexity": float(args.perplexity),
        "coords": [[float(a), float(b), float(c)] for a, b, c in coords.tolist()],
        "labels": [int(v) for v in labels.tolist()],
    }
    (RUNS_DIR / "tsne_3d.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote tsne_3d.png + tsne_3d.json ({len(labels)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

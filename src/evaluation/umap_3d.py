"""UMAP 3D cho FashionMNIST (sinh umap_3d.png + umap_3d.json).

Nguon du lieu: FashionMNIST test goc qua torchvision (data/processed/), lay N mau
dau tien, flatten 784, chuan hoa z-score (mean=0.2860, std=0.3530) nhu project.

Sinh ra trong results/runs/:
- umap_3d.png   (scatter 3D mau theo 10 lop)
- umap_3d.json  ({samples, seed, n_neighbors, min_dist, coords, labels})

Cach dung:
    python src/evaluation/umap_3d.py --samples 2000 --seed 42
    python src/evaluation/embed3d_html.py --input umap_3d.json --output umap_3d.html --title "UMAP 3D"
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
    parser = argparse.ArgumentParser(description="UMAP 3D cho FashionMNIST.")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    args = parser.parse_args(argv)

    import umap

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    labels, flat = load_fashion(args.samples)
    flat_z = (flat - 0.2860) / 0.3530

    reducer = umap.UMAP(n_components=3, random_state=args.seed,
                        n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    coords = reducer.fit_transform(flat_z)
    print(f"UMAP 3D done: {coords.shape}")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                    c=labels, cmap="tab10", s=4, alpha=0.6)
    fig.colorbar(sc, ax=ax, ticks=range(10))
    ax.set_title(f"UMAP 3D Fashion-MNIST test ({len(labels)} anh)")
    ax.set_xlabel("Dim1")
    ax.set_ylabel("Dim2")
    ax.set_zlabel("Dim3")
    fig.tight_layout()
    fig.savefig(RUNS_DIR / "umap_3d.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "samples": int(len(labels)),
        "seed": int(args.seed),
        "n_neighbors": int(args.n_neighbors),
        "min_dist": float(args.min_dist),
        "coords": [[float(a), float(b), float(c)] for a, b, c in coords.tolist()],
        "labels": [int(v) for v in labels.tolist()],
    }
    (RUNS_DIR / "umap_3d.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote umap_3d.png + umap_3d.json ({len(labels)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""PyMDE 3D cho FashionMNIST (sinh mde_3d.png + mde_3d.json).

Nguon du lieu: FashionMNIST test goc qua torchvision (data/processed/), lay N mau
dau tien, flatten 784, chuan hoa z-score (mean=0.2860, std=0.3530) nhu project.

Sinh ra trong results/runs/:
- mde_3d.png   (scatter 3D mau theo 10 lop)
- mde_3d.json  ({samples, seed, coords, labels})

Cach dung:
    python src/evaluation/mde_3d.py --samples 2000 --seed 42
    python src/evaluation/embed3d_html.py --input mde_3d.json --output mde_3d.html --title "PyMDE 3D"
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
    parser = argparse.ArgumentParser(description="PyMDE 3D cho FashionMNIST.")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    import torch
    import pymde

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    labels, flat = load_fashion(args.samples)
    flat_z = (flat - 0.2860) / 0.3530

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = torch.from_numpy(flat_z)
    # neighbors ~15 giong UMAP de so sanh cong bang.
    mde = pymde.preserve_neighbors(data, embedding_dim=3, constraint=pymde.Standardized(),
                                   n_neighbors=15, verbose=False, device="cpu")
    coords = mde.embed(verbose=False).detach().cpu().numpy()
    print(f"PyMDE 3D done: {coords.shape}")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                    c=labels, cmap="tab10", s=4, alpha=0.6)
    fig.colorbar(sc, ax=ax, ticks=range(10))
    ax.set_title(f"PyMDE 3D Fashion-MNIST test ({len(labels)} anh)")
    ax.set_xlabel("Dim1")
    ax.set_ylabel("Dim2")
    ax.set_zlabel("Dim3")
    fig.tight_layout()
    fig.savefig(RUNS_DIR / "mde_3d.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "samples": int(len(labels)),
        "seed": int(args.seed),
        "coords": [[float(a), float(b), float(c)] for a, b, c in coords.tolist()],
        "labels": [int(v) for v in labels.tolist()],
    }
    (RUNS_DIR / "mde_3d.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote mde_3d.png + mde_3d.json ({len(labels)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

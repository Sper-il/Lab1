"""Xuat TensorBoard Projector tuong duong project_zalando.py goc (bản TF1.x da cu).

Code goc dung `tensorflow.contrib.projector` (TF1, nay da bo) de:
- luu Xtest.tsv (pixels) + Ytest.tsv (ten lop)
- ghep sprite zalando-mnist-sprite.png (luoi anh 28x28)
- cau hinh embedding + sprite cho TensorBoard

Ban nay lam dieu tuong duong bang PyTorch + torch.utils.tensorboard:
- doc FashionMNIST test goc (data/processed/), lay N anh dau tien
- ghi results/runs/projector/{Xtest.tsv, Ytest.tsv, sprite.png}
- ghi embedding log (add_embedding) de mo bang `tensorboard --logdir results/runs/projector`

Cach dung:
    python src/data/projector_export.py --n 2000
    tensorboard --logdir results/runs/projector
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
OUT_DIR = RUNS_DIR / "projector"

LABELS = ["t_shirt_top", "trouser", "pullover", "dress", "coat",
          "sandal", "shirt", "sneaker", "bag", "ankle_boots"]


def get_sprite_image(images: np.ndarray) -> np.ndarray:
    """Ghep N anh 28x28 thanh sprite vuong (giong utils/helper.py goc)."""
    n = len(images)
    grid = int(np.ceil(np.sqrt(n)))
    pad = grid * grid - n
    if pad:
        images = np.concatenate([images, np.zeros((pad, 28, 28), dtype=images.dtype)], axis=0)
    rows = []
    for r in range(grid):
        rows.append(np.hstack(images[r * grid:(r + 1) * grid]))
    return np.vstack(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Xuat TensorBoard Projector cho FashionMNIST.")
    parser.add_argument("--n", type=int, default=2000, help="So anh test xuat (mac dinh 2000).")
    args = parser.parse_args(argv)

    import torch
    from torchvision.datasets import FashionMNIST
    from torch.utils.tensorboard import SummaryWriter
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = FashionMNIST(root=str(PROJECT_ROOT / "data" / "processed"), train=False, download=True)
    n = min(args.n, len(ds))
    imgs = ds.data[:n].numpy().astype(np.float32) / 255.0  # (N,28,28)
    labels = ds.targets[:n].numpy()
    flat = imgs.reshape(n, -1)

    # 1. TSV giong ban goc.
    np.savetxt(OUT_DIR / "Xtest.tsv", flat, fmt="%.6e", delimiter="\t")
    y_str = np.array([LABELS[j] for j in labels])
    np.savetxt(OUT_DIR / "Ytest.tsv", y_str, fmt="%s")

    # 2. Sprite.
    sprite = get_sprite_image((imgs * 255).astype(np.uint8))
    plt.imsave(OUT_DIR / "sprite.png", sprite, cmap="gray")

    # 3. Embedding log cho TensorBoard (thay projector_config.pbtxt + ckpt TF1).
    writer = SummaryWriter(log_dir=str(OUT_DIR))
    writer.add_embedding(torch.from_numpy(flat),
                         metadata=y_str.tolist(),
                         label_img=torch.from_numpy(imgs).unsqueeze(1),
                         global_step=n)
    writer.close()
    print(f"Wrote {OUT_DIR} (Xtest.tsv + Ytest.tsv + sprite.png + TB embedding, {n} samples)")
    print(f"Mo bang: tensorboard --logdir {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

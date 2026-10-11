"""Split FashionMNIST ra file .png trên disk theo cấu trúc thư mục.

CHẠY 1 LẦN trước khi dùng data. Idempotent: nếu đã có đủ file thì bỏ qua.

Cấu trúc output:
    <root>/train/<class_id>/<idx>.png
    <root>/val/<class_id>/<idx>.png
    <root>/test/<class_id>/<idx>.png
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Tuple

from torchvision.datasets import FashionMNIST

logger = logging.getLogger(__name__)

# Tên lớp FashionMNIST — dùng để đặt tên file cho dễ debug
CLASS_NAMES = [
    "T-shirt_top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle_boot",
]


def _set_seed(seed: int) -> None:
    """Đặt seed cho Python random để split reproducible."""
    random.seed(seed)


def _expected_counts(total: int, val_split: int) -> Tuple[int, int]:
    """Tính số mẫu train/val từ tổng train gốc, đảm bảo không âm và không vượt tổng."""
    if val_split < 0:
        raise ValueError(f"val_split phải >= 0, nhận {val_split}")
    if val_split > total:
        raise ValueError(f"val_split={val_split} vượt quá total={total}")
    return total - val_split, val_split


def _save_split(dataset, indices, split_name: str, root: Path) -> None:
    """Lưu một tập (train/val/test) ra đĩa theo cấu trúc thư mục.

    `dataset` là FashionMNIST gốc, gọi `dataset[i]` trả về (PIL Image, int).
    """
    split_dir = root / split_name
    for original_idx in indices:
        img, label = dataset[original_idx]  # PIL Image grayscale 28x28, int
        label = int(label)

        # Tên file có class_name để dễ debug khi mở folder bằng File Explorer
        class_name = CLASS_NAMES[label]
        filename = f"{class_name}_{original_idx:06d}.png"
        out_path = split_dir / str(label) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not out_path.exists():
            img.save(out_path, format="PNG")


def split_fashionmnist_to_disk(
    root: str | Path,
    val_split: int = 10000,
    seed: int = 42,
) -> Path:
    """Tải FashionMNIST, chia train/val/test, lưu tất cả ra file .png trên disk.

    Args:
        root: thư mục gốc, sẽ chứa 3 subfolder train/val/test.
        val_split: số mẫu lấy từ train gốc làm validation.
        seed: seed cho random split, đảm bảo reproducible.

    Returns:
        Đường dẫn root sau khi ghi xong.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    # Idempotent: nếu 3 folder đều có file thì bỏ qua
    if train_dir.exists() and val_dir.exists() and test_dir.exists():
        n_train = sum(1 for _ in train_dir.rglob("*.png"))
        n_val = sum(1 for _ in val_dir.rglob("*.png"))
        n_test = sum(1 for _ in test_dir.rglob("*.png"))
        if n_train > 0 and n_val > 0 and n_test > 0:
            logger.info(
                "Đã có sẵn %d train / %d val / %d test .png tại %s — bỏ qua.",
                n_train, n_val, n_test, root,
            )
            return root

    logger.info("Tải FashionMNIST từ torchvision (lần đầu có thể chậm)...")
    # download=True: tải về data gốc của torchvision (cache nội bộ)
    # Chỉ cần 1 lần, sau đó dùng lại từ cache.
    train_set = FashionMNIST(root=str(root / "_torchvision_cache"), train=True, download=True)
    test_set = FashionMNIST(root=str(root / "_torchvision_cache"), train=False, download=True)

    # Tách train / val reproducible
    _set_seed(seed)
    n_train_keep, n_val = _expected_counts(len(train_set), val_split)
    indices = list(range(len(train_set)))
    random.shuffle(indices)
    train_indices = sorted(indices[:n_train_keep])
    val_indices = sorted(indices[n_train_keep:])

    logger.info("Ghi %d ảnh train ra %s ...", len(train_indices), train_dir)
    _save_split(train_set, train_indices, "train", root)

    logger.info("Ghi %d ảnh val ra %s ...", len(val_indices), val_dir)
    _save_split(train_set, val_indices, "val", root)

    logger.info("Ghi %d ảnh test ra %s ...", len(test_set), test_dir)
    test_indices = list(range(len(test_set)))
    _save_split(test_set, test_indices, "test", root)

    logger.info("Xong. Cấu trúc tại %s:", root)
    for split in ("train", "val", "test"):
        n_files = sum(1 for _ in (root / split).rglob("*.png"))
        logger.info("  %s/: %d ảnh .png", split, n_files)

    return root


def main() -> None:
    """CLI: python -m src.data.split --root data/raw --val-split 10000 --seed 42"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    parser = argparse.ArgumentParser(description="Split FashionMNIST ra .png trên disk.")
    parser.add_argument("--root", type=str, default="data/raw", help="Thư mục output.")
    parser.add_argument("--val-split", type=int, default=10000, help="Số mẫu validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    split_fashionmnist_to_disk(args.root, args.val_split, args.seed)


if __name__ == "__main__":
    main()

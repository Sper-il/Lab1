"""Build DataLoader cho 3 tập train/val/test từ cấu trúc thư mục .png đã split.

Mỗi tập dùng transform riêng để:
  - tránh data leakage (split đã xảy ra trước, transform là bước sau)
  - cho phép augmentation khác nhau giữa train vs val/test
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)

# Mean/std thực nghiệm của FashionMNIST (tính trên toàn bộ training set).
# Dùng cố định cho cả 3 tập vì stats chỉ nên tính trên train,
# nhưng ở đây ta dùng giá trị chuẩn đã công bố để tránh phải load toàn bộ ảnh để tính.
FASHION_MNIST_MEAN = 0.2860
FASHION_MNIST_STD = 0.3530


def _load_config(config_path: str | Path) -> dict:
    """Đọc file YAML, raise nếu không tồn tại."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_train_transform(mean: float, std: float) -> transforms.Compose:
    """Transform cho train — hiện tại chỉ ToTensor + Normalize (baseline).

    Augmentation phức tạp (RandomHorizontalFlip, RandomCrop) có thể thêm ở M6 experiment.
    """
    return transforms.Compose([
        # Ảnh .png grayscale 28x28 → Tensor [1, 28, 28] giá trị [0, 1]
        transforms.Grayscale(num_output_channels=1),  # đảm bảo 1 channel dù PNG lưu thế nào
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ])


def _build_eval_transform(mean: float, std: float) -> transforms.Compose:
    """Transform cho val/test — KHÔNG augmentation, chỉ ToTensor + Normalize."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ])


def _make_loader(
    dataset: ImageFolder,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = False,
) -> DataLoader:
    """Tạo DataLoader với cấu hình nhất quán."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def build_dataloaders(
    config_path: str | Path = "configs/baseline.yaml",
) -> Dict[str, DataLoader]:
    """Đọc config, build 3 DataLoader từ cấu trúc thư mục .png.

    Args:
        config_path: đường dẫn tới file YAML config.

    Returns:
        Dict với key "train", "val", "test" → DataLoader.
    """
    config = _load_config(config_path)

    data_cfg = config["data"]
    root = Path(data_cfg["root"])
    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg["num_workers"])
    mean = float(data_cfg.get("mean", FASHION_MNIST_MEAN))
    std = float(data_cfg.get("std", FASHION_MNIST_STD))

    # Kiểm tra split đã chạy chưa
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists() or not any(split_dir.rglob("*.png")):
            raise FileNotFoundError(
                f"Không tìm thấy ảnh .png trong {split_dir}. "
                f"Chạy trước: python -m src.data.split --root {root}"
            )

    # Build ImageFolder cho mỗi tập với transform riêng
    train_ds = ImageFolder(
        root=str(root / "train"),
        transform=_build_train_transform(mean, std),
    )
    val_ds = ImageFolder(
        root=str(root / "val"),
        transform=_build_eval_transform(mean, std),
    )
    test_ds = ImageFolder(
        root=str(root / "test"),
        transform=_build_eval_transform(mean, std),
    )

    logger.info(
        "Dataset sizes — train: %d, val: %d, test: %d, classes: %d",
        len(train_ds), len(val_ds), len(test_ds), len(train_ds.classes),
    )

    # pin_memory chỉ hữu ích khi có CUDA; bật auto để tránh warning trên CPU.
    use_pin_memory = torch.cuda.is_available()

    return {
        "train": _make_loader(train_ds, batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_pin_memory),
        "val": _make_loader(val_ds, batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=use_pin_memory),
        "test": _make_loader(test_ds, batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=use_pin_memory),
    }


def summarize(loader: DataLoader, name: str) -> None:
    """In thông tin 1 loader: số batch, shape, dtype, range pixel — để verify nhanh."""
    ds = loader.dataset
    print(f"[{name}]")
    print(f"  dataset size: {len(ds)}")
    print(f"  num batches:  {len(loader)}")
    print(f"  classes:      {ds.classes}")

    # Lấy 1 batch đầu tiên để in shape
    images, labels = next(iter(loader))
    print(f"  batch shape:  images={tuple(images.shape)}, labels={tuple(labels.shape)}")
    print(f"  dtypes:       images={images.dtype}, labels={labels.dtype}")
    print(f"  pixel range:  min={images.min().item():.4f}, max={images.max().item():.4f}")
    print(f"  label range:  min={labels.min().item()}, max={labels.max().item()}")
    print()

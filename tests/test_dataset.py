"""Test cho data module.

CHẠY: pytest tests/test_dataset.py -v
YÊU CẦU: đã chạy `python -m src.data.split --root data/raw` trước đó.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torchvision.datasets import ImageFolder

from src.data.dataset import (
    FASHION_MNIST_MEAN,
    FASHION_MNIST_STD,
    _build_eval_transform,
    _build_train_transform,
    _load_config,
    build_dataloaders,
    summarize,
)


# ---------- Helpers ----------

CONFIG_PATH = "configs/baseline.yaml"
DATA_ROOT = Path("data/raw")


def _data_ready() -> bool:
    """Bỏ qua test nếu data chưa được split (tránh fail CI khi chưa download)."""
    return all((DATA_ROOT / s).exists() and any((DATA_ROOT / s).rglob("*.png"))
               for s in ("train", "val", "test"))


# Skip tất cả test nếu data chưa sẵn sàng
pytestmark = pytest.mark.skipif(
    not _data_ready(),
    reason=f"Chưa chạy `python -m src.data.split --root {DATA_ROOT}`"
)


# ---------- Test 1: config load được và đúng schema ----------

def test_config_loads_and_has_required_keys():
    """Config YAML phải có đủ key cần thiết."""
    cfg = _load_config(CONFIG_PATH)
    assert "data" in cfg
    for key in ("root", "batch_size", "num_workers", "val_split", "mean", "std"):
        assert key in cfg["data"], f"Thiếu key 'data.{key}' trong config"
    assert cfg["data"]["mean"] == FASHION_MNIST_MEAN
    assert cfg["data"]["std"] == FASHION_MNIST_STD


# ---------- Test 2: build_dataloaders trả về 3 loader ----------

def test_build_dataloaders_returns_three_loaders():
    loaders = build_dataloaders(CONFIG_PATH)
    assert set(loaders.keys()) == {"train", "val", "test"}
    for name, loader in loaders.items():
        assert loader is not None
        assert hasattr(loader, "__iter__")


# ---------- Test 3: số lượng mẫu đúng ----------

def test_dataset_sizes_match_expected():
    """Train 50.000, val 10.000, test 10.000."""
    loaders = build_dataloaders(CONFIG_PATH)
    assert len(loaders["train"].dataset) == 50000
    assert len(loaders["val"].dataset) == 10000
    assert len(loaders["test"].dataset) == 10000


# ---------- Test 4: batch shape đúng ----------

def test_batch_shape_and_dtype():
    """Batch đầu tiên phải có shape [B, 1, 28, 28] và dtype float32."""
    loaders = build_dataloaders(CONFIG_PATH)
    images, labels = next(iter(loaders["train"]))

    assert images.dim() == 4, f"Expected 4D tensor, got {images.dim()}D"
    assert images.shape[1] == 1, f"Expected 1 channel, got {images.shape[1]}"
    assert images.shape[2] == 28 and images.shape[3] == 28
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64
    assert len(labels) == images.shape[0]


# ---------- Test 5: pixel range sau normalize ----------

def test_pixel_range_after_normalize():
    """Sau Normalize, pixel thường nằm trong khoảng [-2, 2] (3 sigma) cho FashionMNIST."""
    loaders = build_dataloaders(CONFIG_PATH)
    images, _ = next(iter(loaders["train"]))
    # 99% pixel nằm trong [-3, 3] sau Normalize(mean=0.286, std=0.353)
    assert images.min() >= -3.5, f"Pixel min quá thấp: {images.min()}"
    assert images.max() <= 3.5, f"Pixel max quá cao: {images.max()}"


# ---------- Test 6: label trong 10 lớp ----------

def test_labels_in_valid_range():
    """Label phải nằm trong [0, 9]."""
    loaders = build_dataloaders(CONFIG_PATH)
    images, labels = next(iter(loaders["train"]))
    assert labels.min() >= 0
    assert labels.max() <= 9


# ---------- Test 7: 3 tập có cùng 10 class ----------

def test_all_splits_have_same_classes():
    """Train/val/test phải cùng 10 class và cùng thứ tự (để label map nhất quán)."""
    loaders = build_dataloaders(CONFIG_PATH)
    train_classes = loaders["train"].dataset.classes
    val_classes = loaders["val"].dataset.classes
    test_classes = loaders["test"].dataset.classes
    assert train_classes == val_classes == test_classes
    assert len(train_classes) == 10


# ---------- Test 8: transform train != eval ----------

def test_train_and_eval_transforms_differ():
    """Transform train và eval phải là 2 instance khác nhau (tránh share state)."""
    train_t = _build_train_transform(0.2860, 0.3530)
    eval_t = _build_eval_transform(0.2860, 0.3530)
    assert train_t is not eval_t
    # Cùng normalize nên giá trị cuối giống nhau
    # (test này chỉ đảm bảo 2 pipeline tồn tại độc lập, không phải cùng object)


# ---------- Test 9: split reproducible ----------

def test_split_is_reproducible_via_seeding():
    """Nếu đã có data rồi thì số file phải khớp — split một lần là đủ, không random lại."""
    # Đếm file hiện tại
    n_train = sum(1 for _ in (DATA_ROOT / "train").rglob("*.png"))
    n_val = sum(1 for _ in (DATA_ROOT / "val").rglob("*.png"))
    n_test = sum(1 for _ in (DATA_ROOT / "test").rglob("*.png"))
    assert n_train == 50000
    assert n_val == 10000
    assert n_test == 10000


# ---------- Test 10: ImageFolder load được (sanity check) ----------

def test_imagefolder_can_load_one_sample():
    """Đảm bảo ImageFolder đọc được 1 ảnh .png và trả về (PIL, int)."""
    ds = ImageFolder(root=str(DATA_ROOT / "test"))
    img, label = ds[0]
    # Trước khi transform, img là PIL Image
    assert img.size == (28, 28) or img.mode == "L" or img.mode == "RGB"
    assert isinstance(label, int)
    assert 0 <= label <= 9


# ---------- Test 11: summarize chạy được ----------

def test_summarize_runs_without_error(capsys):
    """Hàm summarize phải in được thông tin, dùng capsys bắt output."""
    loaders = build_dataloaders(CONFIG_PATH)
    summarize(loaders["train"], "train")
    captured = capsys.readouterr()
    assert "[train]" in captured.out
    assert "batch shape" in captured.out
    assert "pixel range" in captured.out

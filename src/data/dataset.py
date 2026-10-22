"""
FashionMNIST Dataset Module
============================
Handles loading and transformation of FashionMNIST dataset.

Fashion Categories:
    0: T-shirt/top
    1: Trouser
    2: Pullover
    3: Dress
    4: Coat
    5: Sandal
    6: Shirt
    7: Sneaker
    8: Bag
    9: Ankle boot
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Callable

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import Compose

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Dataset configuration
DATASET_CONFIG = {
    "name": "FashionMNIST",
    "num_classes": 10,
    "image_size": (28, 28),
    "channels": 1,
    "train_size": 60000,
    "test_size": 10000,
}

# Fashion item class names
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


class TransformBuilder:
    """Builder class for creating data transforms."""
    
    @staticmethod
    def get_train_transforms(augment: bool = False) -> Compose:
        """
        Get training transforms with optional augmentation.
        
        Args:
            augment: Whether to apply data augmentation
            
        Returns:
            Composed transforms for training data
        """
        if augment:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.2860],
                    std=[0.3530]
                ),
            ])
        
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.2860],  # FashionMNIST mean
                std=[0.3530]     # FashionMNIST std
            ),
        ])
    
    @staticmethod
    def get_test_transforms() -> Compose:
        """
        Get transforms for test/validation data.
        
        Returns:
            Composed transforms for test data
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.2860],
                std=[0.3530]
            ),
        ])


def load_fashion_mnist(
    data_dir: str = "./data/raw",
    train: bool = True,
    transform: Optional[Compose] = None,
    download: bool = True
) -> Dataset:
    """
    Load FashionMNIST dataset.
    
    Args:
        data_dir: Directory to store/load dataset
        train: Whether to load training or test set
        transform: Transforms to apply to images
        download: Whether to download if not present
        
    Returns:
        FashionMNIST dataset object
    """
    if transform is None:
        transform = TransformBuilder.get_test_transforms()
    
    logger.info(f"Loading {'training' if train else 'test'} dataset from {data_dir}")
    
    dataset = datasets.FashionMNIST(
        root=data_dir,
        train=train,
        transform=transform,
        download=download
    )
    
    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


def create_data_loader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a DataLoader from a dataset.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        DataLoader object
    """
    logger.info(f"Creating DataLoader with batch_size={batch_size}, shuffle={shuffle}")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    
    return loader


def get_fashion_mnist_loaders(
    data_dir: str = "./data/raw",
    batch_size: int = 64,
    augment_train: bool = False,
    num_workers: int = 0,
    val_split: float = 1 / 6
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, validation, and test DataLoaders.
    
    Args:
        data_dir: Directory to store/load dataset
        batch_size: Batch size for all loaders
        augment_train: Whether to augment training data
        num_workers: Number of worker processes
        val_split: Fraction of training data for validation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create transforms
    train_transform = TransformBuilder.get_train_transforms(augment=augment_train)
    test_transform = TransformBuilder.get_test_transforms()
    
    # Load training dataset
    train_full = load_fashion_mnist(
        data_dir=data_dir,
        train=True,
        transform=train_transform,
        download=True
    )
    
    # Split training data
    total_train = len(train_full)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size
    
    logger.info(f"Splitting training data: train={train_size}, val={val_size}")
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_full, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Load test dataset
    test_dataset = load_fashion_mnist(
        data_dir=data_dir,
        train=False,
        transform=test_transform,
        download=True
    )
    
    # Create data loaders
    train_loader = create_data_loader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = create_data_loader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = create_data_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    logger.info("Data loaders created successfully")
    
    return train_loader, val_loader, test_loader


def get_class_name(label: int) -> str:
    """
    Get class name for a label.
    
    Args:
        label: Class label (0-9)
        
    Returns:
        Human-readable class name
    """
    return CLASS_NAMES[label]


if __name__ == "__main__":
    # Test data loading
    logger.info("Testing FashionMNIST data loading...")
    
    train_loader, val_loader, test_loader = get_fashion_mnist_loaders(
        batch_size=64,
        augment_train=False
    )
    
    # Get a batch
    images, labels = next(iter(train_loader))
    
    logger.info(f"Batch shape: images={images.shape}, labels={labels.shape}")
    logger.info(f"Image dtype: {images.dtype}, range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Test class names
    for i in range(10):
        logger.info(f"Label {i}: {get_class_name(i)}")
    
    logger.info("Dataset loading test completed successfully!")

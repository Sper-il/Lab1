"""
Data Split Module
=================
Handles train/validation/test data splitting.

Split Strategy:
- Training: 80% of original training data
- Validation: 10% of original training data  
- Test: 100% of original test data
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for data splitting."""
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42
    
    def __post_init__(self):
        """Validate split ratios."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


class DataSplitter:
    """Handles dataset splitting operations."""
    
    def __init__(self, config: Optional[SplitConfig] = None):
        """
        Initialize data splitter.
        
        Args:
            config: Split configuration
        """
        self.config = config or SplitConfig()
        self.generator = torch.Generator().manual_seed(self.config.random_seed)
        
        logger.info(f"DataSplitter initialized with config: {self.config}")
    
    def split_dataset(
        self, 
        dataset: Dataset,
        stratify: bool = False
    ) -> Tuple[Subset, Subset, Subset]:
        """
        Split dataset into train/val/test sets.
        
        Args:
            dataset: Full dataset to split
            stratify: Whether to use stratified splitting
            
        Returns:
            Tuple of (train_subset, val_subset, test_subset)
        """
        n_samples = len(dataset)
        n_train = int(n_samples * self.config.train_ratio)
        n_val = int(n_samples * self.config.val_ratio)
        # n_test = remaining
        
        indices = list(range(n_samples))
        
        if stratify:
            # Stratified splitting by labels
            indices = self._stratified_split(dataset, indices)
        
        # Shuffle indices
        shuffled = indices.copy()
        np.random.seed(self.config.random_seed)
        np.random.shuffle(shuffled)
        
        # Split indices
        train_idx = shuffled[:n_train]
        val_idx = shuffled[n_train:n_train + n_val]
        test_idx = shuffled[n_train + n_val:]
        
        # Create subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        test_subset = Subset(dataset, test_idx)
        
        logger.info(f"Dataset split: train={len(train_subset)}, "
                    f"val={len(val_subset)}, test={len(test_subset)}")
        
        return train_subset, val_subset, test_subset
    
    def _stratified_split(
        self, 
        dataset: Dataset, 
        indices: List[int]
    ) -> List[int]:
        """
        Perform stratified splitting to maintain class distribution.
        
        Args:
            dataset: Dataset with labels
            indices: List of indices to split
            
        Returns:
            Stratified shuffled indices
        """
        # Group indices by label
        label_to_indices = {}
        for idx in indices:
            try:
                _, label = dataset[idx]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                if label not in label_to_indices:
                    label_to_indices[label] = []
                label_to_indices[label].append(idx)
            except Exception as e:
                logger.warning(f"Error getting label for index {idx}: {e}")
        
        # Shuffle within each class
        for label in label_to_indices:
            np.random.shuffle(label_to_indices[label])
        
        # Distribute samples proportionally
        train_ratio = self.config.train_ratio
        val_ratio = self.config.val_ratio

        stratified_indices = []

        for label, idx_list in label_to_indices.items():
            n = len(idx_list)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            stratified_indices.extend(idx_list[:n_train])
            stratified_indices.extend(idx_list[n_train:n_train + n_val])
            stratified_indices.extend(idx_list[n_train + n_val:])

        # CRITICAL: shuffle the final stratified indices. Otherwise the
        # caller would see all-class-1-then-all-class-2 ordering, and a
        # plain ``indices[:n_train]`` slice would steal entire classes
        # into either train or val.
        np.random.shuffle(stratified_indices)
        return stratified_indices
    
    def create_split_from_existing(
        self,
        train_dataset: Dataset,
        val_size: float = 0.1,
        stratify: bool = True
    ) -> Tuple[Subset, Subset]:
        """
        Create validation split from existing training dataset.
        
        Args:
            train_dataset: Full training dataset
            val_size: Fraction of data for validation
            stratify: Whether to use stratified splitting
            
        Returns:
            Tuple of (train_subset, val_subset)
        """
        n_total = len(train_dataset)
        n_val = int(n_total * val_size)
        n_train = n_total - n_val
        
        indices = list(range(n_total))
        
        if stratify:
            indices = self._stratified_split(train_dataset, indices)
        else:
            np.random.seed(self.config.random_seed)
            np.random.shuffle(indices)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        
        logger.info(f"Existing split: train={len(train_subset)}, val={len(val_subset)}")
        
        return train_subset, val_subset


class DataSplitManager:
    """Manages data splits and loaders."""
    
    def __init__(self, config: Optional[SplitConfig] = None):
        """
        Initialize split manager.
        
        Args:
            config: Split configuration
        """
        self.config = config or SplitConfig()
        self.splitter = DataSplitter(self.config)
        
        self.train_subset: Optional[Subset] = None
        self.val_subset: Optional[Subset] = None
        self.test_subset: Optional[Subset] = None
        
        logger.info("DataSplitManager initialized")
    
    def load_and_split(
        self,
        data_dir: str = "./data/processed",
        train_transform=None,
        test_transform=None,
        stratify: bool = True
    ) -> Tuple[Subset, Subset, Dataset]:
        """
        Load FashionMNIST and split into train/val/test.
        
        Args:
            data_dir: Directory for dataset
            train_transform: Transform for training data
            test_transform: Transform for test data
            stratify: Whether to use stratified splitting
            
        Returns:
            Tuple of (train_subset, val_subset, test_dataset)
        """
        from src.data.dataset import load_fashion_mnist
        
        logger.info("Loading and splitting FashionMNIST...")
        
        # Load full training dataset
        train_full = load_fashion_mnist(
            data_dir=data_dir,
            train=True,
            transform=train_transform,
            download=True
        )
        
        # Split training into train/val
        self.train_subset, self.val_subset = self.splitter.create_split_from_existing(
            train_full,
            val_size=self.config.val_ratio / (self.config.train_ratio + self.config.val_ratio),
            stratify=stratify
        )
        
        # Load test dataset
        self.test_subset = load_fashion_mnist(
            data_dir=data_dir,
            train=False,
            transform=test_transform,
            download=True
        )
        
        logger.info(f"Data split complete: "
                    f"train={len(self.train_subset)}, "
                    f"val={len(self.val_subset)}, "
                    f"test={len(self.test_subset)}")
        
        return self.train_subset, self.val_subset, self.test_subset
    
    def create_loaders(
        self,
        batch_size: int = 64,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create DataLoaders for all splits.
        
        Args:
            batch_size: Batch size for all loaders
            num_workers: Number of worker processes
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        from src.data.dataset import create_data_loader
        
        if self.train_subset is None or self.test_subset is None:
            raise ValueError("Must call load_and_split before creating loaders")
        
        train_loader = create_data_loader(
            self.train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        val_loader = create_data_loader(
            self.val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        test_loader = create_data_loader(
            self.test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        logger.info("Data loaders created successfully")
        
        return train_loader, val_loader, test_loader
    
    def save_split_info(self, output_path: str = "data/splits/split_info.txt"):
        """
        Save split information to file.
        
        Args:
            output_path: Path to save split info
        """
        if self.train_subset is None:
            raise ValueError("Must call load_and_split before saving split info")
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            f.write("FashionMNIST Data Split Information\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Random Seed: {self.config.random_seed}\n")
            f.write(f"Train Ratio: {self.config.train_ratio}\n")
            f.write(f"Val Ratio: {self.config.val_ratio}\n")
            f.write(f"Test Ratio: {self.config.test_ratio}\n\n")
            f.write(f"Train Size: {len(self.train_subset)}\n")
            f.write(f"Val Size: {len(self.val_subset)}\n")
            f.write(f"Test Size: {len(self.test_subset)}\n")
        
        logger.info(f"Split info saved to {output_path}")


def split_pipeline(
    data_dir: str = "./data/processed",
    batch_size: int = 64,
    val_ratio: float = 0.1,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, SplitConfig]:
    """
    Complete data splitting pipeline.
    
    Args:
        data_dir: Directory for dataset
        batch_size: Batch size for loaders
        val_ratio: Validation split ratio
        num_workers: Number of workers
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, config)
    """
    logger.info("Starting data split pipeline...")
    
    # Create config
    config = SplitConfig(
        train_ratio=1.0 - val_ratio,
        val_ratio=val_ratio,
        test_ratio=0.0  # Test set is separate
    )
    
    # Create manager
    manager = DataSplitManager(config)
    
    # Load and split
    manager.load_and_split(data_dir=data_dir, stratify=True)
    
    # Create loaders
    train_loader, val_loader, test_loader = manager.create_loaders(
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Save split info
    manager.save_split_info()
    
    logger.info("Data split pipeline complete")
    
    return train_loader, val_loader, test_loader, config


if __name__ == "__main__":
    # Test split module
    from dataset import TransformBuilder
    
    logger.info("Testing data split module...")
    
    train_loader, val_loader, test_loader = split_pipeline(
        data_dir="./data/processed",
        batch_size=64,
        val_ratio=0.1
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get sample batch
    images, labels = next(iter(train_loader))
    print(f"\nSample batch: images={images.shape}, labels={labels.shape}")
    
    logger.info("Data split test completed successfully!")

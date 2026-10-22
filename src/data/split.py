"""
Data Split Module
=================
Handles train/validation/test data splitting.

Split Strategy:
- Training: 50,000 samples (10,000 per class)
- Validation: 10,000 samples (1,000 per class)
- Test: 10,000 samples — the dataset's own official test split,
  loaded as-is and never re-split
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#: Fraction of the 60,000-sample training set held out for validation.
#: ``1/6`` → 10,000 validation / 50,000 train, i.e. clean per-class counts
#: of 1,000 / 5,000 out of 6,000. Used as the default by :func:`split_pipeline`
#: and by every entry point, so the whole project shares one split.
DEFAULT_VAL_RATIO = 1 / 6


@dataclass
class SplitConfig:
    """Configuration for data splitting.

    The official 60,000-sample training set is divided 50,000 train /
    10,000 validation — i.e. exactly 5/6 and 1/6. Those fractions are used
    instead of the round numbers 0.8/0.2 or 0.9/0.1 because only 1/6 lands
    on clean per-class counts (5,000 / 1,000 out of 6,000 per class).
    """

    train_ratio: float = 5 / 6
    val_ratio: float = 1 / 6
    test_ratio: float = 0.0  # test set is the dataset's own split, never re-split
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
    
    def _group_indices_by_label(
        self,
        dataset: Dataset,
        indices: List[int]
    ) -> Dict[int, List[int]]:
        """Group sample indices by their integer label."""
        label_to_indices: Dict[int, List[int]] = {}
        for idx in indices:
            try:
                _, label = dataset[idx]
                if isinstance(label, torch.Tensor):
                    label = int(label.item())
                else:
                    label = int(label)
                label_to_indices.setdefault(label, []).append(idx)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"Error getting label for index {idx}: {e}")
        return label_to_indices

    def _stratified_split(
        self, 
        dataset: Dataset, 
        indices: List[int]
    ) -> List[int]:
        """
        Perform stratified splitting to maintain class distribution.

        The returned list is ordered as ``[train..., val..., test...]`` with
        every class contributing its own quota to each block, so callers can
        slice by ratio and still get a stratified result.

        Args:
            dataset: Dataset with labels
            indices: List of indices to split
            
        Returns:
            Stratified indices laid out train-first, then val, then test
        """
        label_to_indices = self._group_indices_by_label(dataset, indices)
        rng = np.random.default_rng(self.config.random_seed)

        train_ratio = self.config.train_ratio
        val_ratio = self.config.val_ratio

        train_block: List[int] = []
        val_block: List[int] = []
        test_block: List[int] = []

        for label in sorted(label_to_indices):
            idx_list = list(label_to_indices[label])
            rng.shuffle(idx_list)

            n = len(idx_list)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_block.extend(idx_list[:n_train])
            val_block.extend(idx_list[n_train:n_train + n_val])
            test_block.extend(idx_list[n_train + n_val:])

        # Shuffle inside each block so the DataLoader sees a random order
        # even though the blocks stay class-complete.
        for block in (train_block, val_block, test_block):
            rng.shuffle(block)

        return train_block + val_block + test_block
    
    def create_split_from_existing(
        self,
        train_dataset: Dataset,
        val_size: Optional[float] = None,
        stratify: bool = True
    ) -> Tuple[Subset, Subset]:
        """
        Create validation split from existing training dataset.

        Args:
            train_dataset: Full training dataset
            val_size: Fraction of data for validation. Defaults to
                ``config.val_ratio`` (1/6 → 10,000 of 60,000).
            stratify: Whether to use stratified splitting
            
        Returns:
            Tuple of (train_subset, val_subset)
        """
        n_total = len(train_dataset)
        fraction = self.config.val_ratio if val_size is None else val_size
        n_val = int(n_total * fraction)
        n_train = n_total - n_val
        
        indices = list(range(n_total))

        if stratify:
            # ``_stratified_split`` lays the result out as
            # [train..., val..., test...]. Slicing the *blocks* (instead of
            # shuffling the whole list and cutting by index) is what keeps
            # the class distribution intact in both subsets.
            ordered = self._stratified_split(train_dataset, indices)
            train_idx = ordered[:n_train]
            val_idx = ordered[n_train:n_train + n_val]
        else:
            rng = np.random.default_rng(self.config.random_seed)
            shuffled = np.array(indices)
            rng.shuffle(shuffled)
            train_idx = shuffled[:n_train]
            val_idx = shuffled[n_train:n_train + n_val]

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
    val_ratio: float = DEFAULT_VAL_RATIO,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, SplitConfig]:
    """
    Complete data splitting pipeline.

    Args:
        data_dir: Directory for dataset
        batch_size: Batch size for loaders
        val_ratio: Validation split ratio, as a fraction of the 60,000-sample
            training set. Defaults to ``1/6`` → 10,000 validation / 50,000
            train, which keeps clean per-class counts (1,000 / 5,000).
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
        val_ratio=DEFAULT_VAL_RATIO
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get sample batch
    images, labels = next(iter(train_loader))
    print(f"\nSample batch: images={images.shape}, labels={labels.shape}")
    
    logger.info("Data split test completed successfully!")

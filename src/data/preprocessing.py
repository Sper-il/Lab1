"""
Data Preprocessing Module
==========================
Handles data cleaning and preprocessing for FashionMNIST.

Preprocessing Steps:
- Data validation
- Normalization
- Data augmentation
- Feature engineering
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    normalize: bool = True
    mean: float = 0.2860
    std: float = 0.3530
    augment: bool = False
    augmentation_prob: float = 0.5
    rotation_degrees: int = 10
    translation: float = 0.1
    horizontal_flip: bool = True


class DataCleaner:
    """Handles data cleaning operations."""
    
    @staticmethod
    def validate_image(img: Tensor) -> bool:
        """
        Validate that an image tensor is valid.
        
        Args:
            img: Image tensor
            
        Returns:
            True if valid, False otherwise
        """
        if img is None:
            return False
        
        if not isinstance(img, Tensor):
            return False
        
        if torch.isnan(img).any() or torch.isinf(img).any():
            logger.warning("Found NaN or Inf values in image")
            return False
        
        if img.numel() == 0:
            logger.warning("Empty image tensor")
            return False
        
        return True
    
    @staticmethod
    def validate_label(label: int, num_classes: int = 10) -> bool:
        """
        Validate that a label is valid.
        
        Args:
            label: Class label
            num_classes: Number of classes
            
        Returns:
            True if valid, False otherwise
        """
        return 0 <= label < num_classes
    
    @staticmethod
    def clean_dataset(dataset: Dataset) -> Tuple[List[int], List[int]]:
        """
        Identify valid and invalid indices in a dataset.
        
        Args:
            dataset: PyTorch dataset
            
        Returns:
            Tuple of (valid_indices, invalid_indices)
        """
        valid_indices = []
        invalid_indices = []
        
        logger.info(f"Validating dataset with {len(dataset)} samples...")
        
        for idx in range(len(dataset)):
            try:
                img, label = dataset[idx]
                
                if DataCleaner.validate_image(img) and \
                   DataCleaner.validate_label(label):
                    valid_indices.append(idx)
                else:
                    invalid_indices.append(idx)
                    
            except Exception as e:
                logger.warning(f"Error at index {idx}: {e}")
                invalid_indices.append(idx)
        
        valid_pct = len(valid_indices) / len(dataset) * 100
        logger.info(f"Validation complete: {len(valid_indices)} valid ({valid_pct:.2f}%), "
                   f"{len(invalid_indices)} invalid")
        
        return valid_indices, invalid_indices


class DataPreprocessor:
    """Handles data preprocessing operations."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        logger.info(f"Preprocessor initialized with config: {self.config}")
    
    def compute_dataset_statistics(
        self, 
        dataset: Dataset, 
        sample_size: int = 10000
    ) -> Dict[str, float]:
        """
        Compute mean and std of dataset.
        
        Args:
            dataset: PyTorch dataset
            sample_size: Number of samples to use for computation
            
        Returns:
            Dictionary with mean and std values
        """
        logger.info(f"Computing dataset statistics from {sample_size} samples...")
        
        all_values = []
        
        for idx in range(min(sample_size, len(dataset))):
            img, _ = dataset[idx]
            all_values.append(img.flatten())
        
        all_values = torch.cat(all_values)
        
        stats = {
            "mean": all_values.mean().item(),
            "std": all_values.std().item(),
            "min": all_values.min().item(),
            "max": all_values.max().item(),
        }
        
        logger.info(f"Dataset statistics: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        return stats
    
    def normalize_data(self, data: Tensor) -> Tensor:
        """
        Normalize data using configured mean and std.
        
        Args:
            data: Input tensor
            
        Returns:
            Normalized tensor
        """
        if not self.config.normalize:
            return data
        
        return (data - self.config.mean) / self.config.std
    
    def denormalize_data(self, data: Tensor) -> Tensor:
        """
        Denormalize data back to original scale.
        
        Args:
            data: Normalized tensor
            
        Returns:
            Denormalized tensor
        """
        if not self.config.normalize:
            return data
        
        return data * self.config.std + self.config.mean
    
    def clip_values(
        self, 
        data: Tensor, 
        min_val: float = 0.0, 
        max_val: float = 1.0
    ) -> Tensor:
        """
        Clip tensor values to specified range.
        
        Args:
            data: Input tensor
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Clipped tensor
        """
        return torch.clamp(data, min_val, max_val)


class DatasetAnalyzer:
    """Analyzes and reports on dataset characteristics."""
    
    @staticmethod
    def analyze_batch(batch: Tuple[Tensor, Tensor]) -> Dict[str, Any]:
        """
        Analyze a single batch.
        
        Args:
            batch: Tuple of (images, labels)
            
        Returns:
            Dictionary of batch statistics
        """
        images, labels = batch
        
        analysis = {
            "batch_size": len(labels),
            "image_shape": list(images.shape),
            "image_dtype": str(images.dtype),
            "image_mean": images.mean().item(),
            "image_std": images.std().item(),
            "image_min": images.min().item(),
            "image_max": images.max().item(),
            "unique_labels": torch.unique(labels).tolist(),
            "label_counts": {
                int(label): int((labels == label).sum())
                for label in torch.unique(labels)
            },
        }
        
        return analysis
    
    @staticmethod
    def analyze_dataloader(dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Analyze a full dataloader.
        
        Args:
            dataloader: PyTorch DataLoader
            
        Returns:
            Dictionary of dataloader statistics
        """
        logger.info("Analyzing dataloader...")
        
        total_images = 0
        total_labels = 0
        label_counts = {}
        batch_stats = []
        
        for batch_idx, batch in enumerate(dataloader):
            analysis = DatasetAnalyzer.analyze_batch(batch)
            batch_stats.append(analysis)
            
            total_images += len(batch[1])
            total_labels += len(batch[1])
            
            for label, count in analysis["label_counts"].items():
                label_counts[label] = label_counts.get(label, 0) + count
        
        return {
            "num_batches": len(batch_stats),
            "total_samples": total_images,
            "batch_size": batch_stats[0]["batch_size"] if batch_stats else 0,
            "label_distribution": label_counts,
            "avg_batch_stats": {
                "image_mean": np.mean([s["image_mean"] for s in batch_stats]),
                "image_std": np.mean([s["image_std"] for s in batch_stats]),
            },
        }


def preprocess_pipeline(
    dataset: Dataset,
    config: Optional[PreprocessingConfig] = None
) -> Dataset:
    """
    Full preprocessing pipeline.
    
    Args:
        dataset: Input dataset
        config: Preprocessing configuration
        
    Returns:
        Preprocessed dataset
    """
    logger.info("Starting preprocessing pipeline...")
    
    config = config or PreprocessingConfig()
    preprocessor = DataPreprocessor(config)
    
    # Step 1: Clean data
    cleaner = DataCleaner()
    valid_indices, invalid_indices = cleaner.clean_dataset(dataset)
    
    if invalid_indices:
        logger.warning(f"Removing {len(invalid_indices)} invalid samples")
    
    # Step 2: Compute statistics
    stats = preprocessor.compute_dataset_statistics(dataset)
    
    # Step 3: Log results
    logger.info("Preprocessing pipeline complete")
    logger.info(f"  Valid samples: {len(valid_indices)}")
    logger.info(f"  Mean: {stats['mean']:.4f}")
    logger.info(f"  Std: {stats['std']:.4f}")
    
    return dataset


if __name__ == "__main__":
    # Test preprocessing module
    from dataset import load_fashion_mnist, TransformBuilder
    
    logger.info("Testing preprocessing module...")
    
    # Load dataset
    dataset = load_fashion_mnist(
        data_dir="./data/raw",
        train=True,
        download=True
    )
    
    # Clean data
    cleaner = DataCleaner()
    valid_idx, invalid_idx = cleaner.clean_dataset(dataset)
    print(f"Valid: {len(valid_idx)}, Invalid: {len(invalid_idx)}")
    
    # Preprocess
    config = PreprocessingConfig(normalize=True, augment=False)
    preprocessor = DataPreprocessor(config)
    stats = preprocessor.compute_dataset_statistics(dataset)
    
    print(f"Computed stats: {stats}")
    
    logger.info("Preprocessing test completed successfully!")

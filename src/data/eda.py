"""
Exploratory Data Analysis (EDA) Module
======================================
Performs exploratory data analysis on FashionMNIST dataset.

Analysis includes:
- Dataset statistics
- Class distribution
- Sample visualization
- Pixel intensity analysis
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EDAAnalyzer:
    """Exploratory Data Analysis for FashionMNIST."""
    
    def __init__(self, dataset: Dataset, class_names: List[str]):
        """
        Initialize EDA analyzer.
        
        Args:
            dataset: FashionMNIST dataset
            class_names: List of class names
        """
        self.dataset = dataset
        self.class_names = class_names
        self.data_dir = Path("results/runs")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        logger.info(f"EDA Analyzer initialized with {len(dataset)} samples")
    
    def get_dataset_stats(self) -> dict:
        """
        Calculate dataset statistics.
        
        Returns:
            Dictionary of statistics
        """
        logger.info("Calculating dataset statistics...")
        
        # Collect all data
        all_images = []
        all_labels = []
        
        for i in range(len(self.dataset)):
            img, label = self.dataset[i]
            all_images.append(img)
            all_labels.append(label)
        
        images = torch.stack(all_images)
        labels = torch.tensor(all_labels)
        
        stats = {
            "num_samples": len(self.dataset),
            "image_shape": images[0].shape,
            "num_channels": images.shape[1],
            "height": images.shape[2],
            "width": images.shape[3],
            "dtype": str(images.dtype),
            "min_value": images.min().item(),
            "max_value": images.max().item(),
            "mean": images.mean().item(),
            "std": images.std().item(),
        }
        
        # Class distribution
        unique, counts = labels.unique(return_counts=True)
        class_dist = {self.class_names[u]: c.item() for u, c in zip(unique, counts)}
        stats["class_distribution"] = class_dist
        
        logger.info(f"Statistics: {stats}")
        return stats
    
    def plot_class_distribution(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot class distribution.
        
        Args:
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        logger.info("Plotting class distribution...")
        
        # Get class distribution
        stats = self.get_dataset_stats()
        class_dist = stats["class_distribution"]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        classes = list(class_dist.keys())
        counts = list(class_dist.values())
        
        bars = ax.bar(classes, counts, color=sns.color_palette("husl", len(classes)))
        
        ax.set_xlabel("Fashion Category", fontsize=12)
        ax.set_ylabel("Number of Samples", fontsize=12)
        ax.set_title("FashionMNIST Class Distribution", fontsize=14, fontweight="bold")
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha="right")
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 100,
                str(count),
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Class distribution saved to {save_path}")
        
        return fig
    
    def plot_sample_images(
        self, 
        num_samples: int = 25,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot sample images from the dataset.
        
        Args:
            num_samples: Number of images to display
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Plotting {num_samples} sample images...")
        
        # Create figure
        fig, axes = plt.subplots(
            nrows=int(np.sqrt(num_samples)),
            ncols=int(np.sqrt(num_samples)),
            figsize=(12, 12)
        )
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        for i, ax in enumerate(axes):
            if i >= num_samples:
                break
            
            # Get random sample
            idx = np.random.randint(0, len(self.dataset))
            img, label = self.dataset[idx]
            
            # Handle tensor shape (C, H, W) -> (H, W)
            if img.dim() == 3:
                img = img.squeeze(0)
            
            # Denormalize for visualization
            # Using approximate original scale
            img = img * 0.3530 + 0.2860
            img = np.clip(img, 0, 1)
            
            ax.imshow(img, cmap='gray')
            ax.set_title(f"{self.class_names[label]}", fontsize=10)
            ax.axis('off')
        
        plt.suptitle("FashionMNIST Sample Images", fontsize=16, fontweight="bold")
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Sample images saved to {save_path}")
        
        return fig
    
    def plot_pixel_intensity_histogram(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot pixel intensity distribution.
        
        Args:
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        logger.info("Plotting pixel intensity histogram...")
        
        # Collect pixel values
        all_pixels = []
        for i in range(min(5000, len(self.dataset))):
            img, _ = self.dataset[i]
            if img.dim() == 3:
                img = img.squeeze(0)
            all_pixels.extend(img.flatten().tolist())
        
        all_pixels = np.array(all_pixels)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(all_pixels, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel("Pixel Intensity", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].set_title("Pixel Intensity Distribution", fontsize=14, fontweight="bold")
        
        # Box plot
        axes[1].boxplot(all_pixels, vert=True)
        axes[1].set_ylabel("Pixel Intensity", fontsize=12)
        axes[1].set_title("Pixel Intensity Box Plot", fontsize=14, fontweight="bold")
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Pixel histogram saved to {save_path}")
        
        return fig
    
    def plot_sample_per_class(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot one sample image per class.
        
        Args:
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        logger.info("Plotting one sample per class...")
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for class_idx in range(10):
            # Find first sample of this class
            for idx in range(len(self.dataset)):
                img, label = self.dataset[idx]
                if label == class_idx:
                    break
            
            # Handle tensor shape
            if img.dim() == 3:
                img = img.squeeze(0)
            
            # Denormalize for visualization
            img = img * 0.3530 + 0.2860
            img = np.clip(img, 0, 1)
            
            axes[class_idx].imshow(img, cmap='gray')
            axes[class_idx].set_title(f"{class_idx}: {self.class_names[class_idx]}", fontsize=10)
            axes[class_idx].axis('off')
        
        plt.suptitle("One Sample Per Class", fontsize=16, fontweight="bold")
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Per-class samples saved to {save_path}")
        
        return fig
    
    def generate_full_report(self, output_dir: str = "results/runs") -> dict:
        """
        Generate complete EDA report.
        
        Args:
            output_dir: Directory to save report figures
            
        Returns:
            Dictionary containing all statistics
        """
        logger.info("Generating full EDA report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get statistics
        stats = self.get_dataset_stats()
        
        # Generate all plots
        self.plot_class_distribution(output_path / "class_distribution.png")
        self.plot_sample_images(25, output_path / "sample_images.png")
        self.plot_pixel_intensity_histogram(output_path / "pixel_histogram.png")
        self.plot_sample_per_class(output_path / "per_class_samples.png")
        
        logger.info(f"EDA report saved to {output_path}")
        
        return stats


def run_eda(dataset: Dataset, class_names: List[str]) -> dict:
    """
    Run full EDA on FashionMNIST dataset.
    
    Args:
        dataset: FashionMNIST dataset
        class_names: List of class names
        
    Returns:
        Dictionary of statistics
    """
    analyzer = EDAAnalyzer(dataset, class_names)
    return analyzer.generate_full_report()


if __name__ == "__main__":
    # Test EDA module
    from dataset import load_fashion_mnist, CLASS_NAMES, TransformBuilder
    
    logger.info("Running EDA test...")
    
    # Load dataset (without normalization for visualization)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = load_fashion_mnist(
        data_dir="./data/raw",
        train=True,
        transform=test_transform,
        download=True
    )
    
    # Run EDA
    stats = run_eda(dataset, CLASS_NAMES)
    
    print("\n=== Dataset Statistics ===")
    for key, value in stats.items():
        if key != "class_distribution":
            print(f"{key}: {value}")
    
    print("\n=== Class Distribution ===")
    for cls, count in stats["class_distribution"].items():
        print(f"  {cls}: {count}")
    
    logger.info("EDA test completed successfully!")

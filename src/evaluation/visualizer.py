# Prediction Visualizer Module
"""Visualization for predicted vs actual images."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path


def _show_if_interactive():
    """Chỉ gọi plt.show() khi chạy ở chế độ tương tác.

    Khi backend là Agg (headless / CI) hoặc người dùng chỉ muốn lưu file,
    plt.show() sẽ in UserWarning 'FigureCanvasAgg is non-interactive' và
    chặn luồng chạy, nên bỏ qua trong các trường hợp đó.
    """
    backend = plt.get_backend().lower()
    if 'agg' in backend or 'template' in backend or backend == '':
        plt.close()
        return
    plt.show()


class PredictionVisualizer:
    """Visualize model predictions with predicted vs actual labels."""
    
    CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    def plot_predictions(self, images, true_labels, pred_labels, 
                        num_samples=20, save_path=None):
        """
        Plot images with their true and predicted labels.
        
        Args:
            images: Batch of images (N, C, H, W) or (N, H, W)
            true_labels: True labels for each image
            pred_labels: Predicted labels for each image
            num_samples: Number of samples to display
            save_path: Optional path to save the figure
        """
        num_samples = min(num_samples, len(images))
        
        # Calculate grid dimensions
        cols = 5
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        for i in range(num_samples):
            ax = axes[i]
            
            # Handle image format
            img = images[i]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            
            # Handle channel dimension
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = img.transpose(1, 2, 0).squeeze()
            
            # Display image
            if img.ndim == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            
            true_label = true_labels[i]
            pred_label = pred_labels[i] if i < len(pred_labels) else -1
            
            # Color code: green if correct, red if wrong
            color = 'green' if true_label == pred_label else 'red'
            
            true_name = self.CLASS_NAMES[true_label] if 0 <= true_label < len(self.CLASS_NAMES) else str(true_label)
            pred_name = self.CLASS_NAMES[pred_label] if 0 <= pred_label < len(self.CLASS_NAMES) else str(pred_label)
            
            ax.set_title(f'True: {true_name}\nPred: {pred_name}', color=color, fontsize=9)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Predicted vs Actual Labels (Green=Correct, Red=Wrong)', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Predictions plot saved to {save_path}")
        
        _show_if_interactive()
    
    def plot_confidence_distribution(self, probs, true_labels, save_path=None):
        """
        Plot distribution of prediction confidence.
        
        Args:
            probs: Prediction probabilities (N, num_classes)
            true_labels: True labels
            save_path: Optional path to save
        """
        max_probs = np.max(probs, axis=1)
        correct = np.array(true_labels) == np.argmax(probs, axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.hist(max_probs[correct], bins=30, alpha=0.7, label='Correct', color='green')
        ax.hist(max_probs[~correct], bins=30, alpha=0.7, label='Wrong', color='red')
        
        ax.set_xlabel('Prediction Confidence')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Prediction Confidence')
        ax.legend()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        _show_if_interactive()

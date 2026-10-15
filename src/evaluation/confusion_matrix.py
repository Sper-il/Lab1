# src/evaluation/confusion_matrix.py

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

class ConfusionMatrixPlotter:
    """Class chuyên biệt để tính toán và visualize Confusion Matrix."""
    
    CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    def __init__(self, class_names=None):
        if class_names:
            self.CLASS_NAMES = class_names
            
    def compute_matrix(self, true_labels, pred_labels):
        """Tính toán raw confusion matrix."""
        return confusion_matrix(true_labels, pred_labels)
    
    def plot(self, true_labels, pred_labels, normalize=True, save_path=None):
        """Vẽ confusion matrix (có thể chọn normalize theo hàng)."""
        cm = self.compute_matrix(true_labels, pred_labels)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix - FashionMNIST'
        else:
            fmt = 'd'
            title = 'Confusion Matrix - FashionMNIST'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=self.CLASS_NAMES, 
                    yticklabels=self.CLASS_NAMES, ax=ax)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
            
        plt.show()
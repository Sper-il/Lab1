# Model Evaluator Module
"""
Model evaluator module - empty placeholder.

To be implemented in later phase.
"""
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

class ModelEvaluator:
    """Evaluate trained model trên test set."""
    
    CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    def __init__(self, model, test_loader, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.device = device
    
    def predict(self):
        """Get all predictions trên test set."""
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                output = self.model(data)
                probs = torch.softmax(output, dim=1)
                _, predicted = output.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def evaluate(self):
        """Đánh giá model - trả về dict of metrics."""
        preds, labels, probs = self.predict()
        
        metrics = {
            'accuracy': accuracy_score(labels, preds) * 100,
            'precision_macro': precision_score(labels, preds, average='macro') * 100,
            'recall_macro': recall_score(labels, preds, average='macro') * 100,
            'f1_macro': f1_score(labels, preds, average='macro') * 100,
        }
        
        print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
        print(f"Precision (macro): {metrics['precision_macro']:.2f}%")
        print(f"Recall (macro): {metrics['recall_macro']:.2f}%")
        print(f"F1-Score (macro): {metrics['f1_macro']:.2f}%")
        
        return metrics, preds, labels, probs


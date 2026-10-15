# src/evaluation/error_analysis.py

import numpy as np

class ErrorAnalyzer:
    """Phân tích các mẫu dự đoán sai của mô hình."""
    
    CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    def __init__(self, class_names=None):
        if class_names:
            self.CLASS_NAMES = class_names
            
    def get_misclassified_samples(self, images, true_labels, pred_labels, probs=None):
        """Lọc ra danh sách các index bị dự đoán sai."""
        misclassified_indices = np.where(pred_labels != true_labels)[0]
        return misclassified_indices
    
    def analyze_common_confusions(self, true_labels, pred_labels, top_k=5):
        """Tìm ra top các cặp lớp bị nhầm lẫn nhiều nhất (True -> Pred)."""
        confusions = {}
        for t, p in zip(true_labels, pred_labels):
            if t != p:
                pair = (self.CLASS_NAMES[t], self.CLASS_NAMES[p])
                confusions[pair] = confusions.get(pair, 0) + 1
                
        # Sắp xếp theo số lượng nhầm lẫn giảm dần
        sorted_confusions = sorted(confusions.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n--- Top {top_k} Most Confused Class Pairs ---")
        for i, (pair, count) in enumerate(sorted_confusions[:top_k], 1):
            print(f"{i}. True: '{pair[0]}' --> Predicted as '{pair[1]}': {count} times")
            
        return sorted_confusions
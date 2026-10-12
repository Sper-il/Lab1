# run_evaluation.py

import torch
from src.data.split import DataSplitManager
from src.models.model_factory import ModelFactory
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.model_io import ModelIO
from src.evaluation.visualizer import PredictionVisualizer
from src.evaluation.confusion_matrix import ConfusionMatrixPlotter
from src.evaluation.error_analysis import ErrorAnalyzer

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load test data từ Phase 1
    data_manager = DataSplitManager()
    _, _, test_loader = data_manager.get_loaders(batch_size=64)
    
    # 2. Khởi tạo model architecture (ví dụ: simple_cnn)
    model = ModelFactory.create('simple_cnn')
    
    # 3. Load checkpoint model đã train (Yêu cầu đề bài)
    checkpoint_path = 'models/checkpoints/trained_model.pt'
    try:
        model, checkpoint = ModelIO.load_model(model, checkpoint_path, device=device)
    except FileNotFoundError:
        print(f"[Lỗi] Không tìm thấy checkpoint tại {checkpoint_path}. Hãy chạy training của Người 2 trước!")
        return

    # 4. Đánh giá model accuracy, precision, recall, f1 (Yêu cầu đề bài)
    evaluator = ModelEvaluator(model, test_loader, device=device)
    metrics, preds, labels, probs = evaluator.evaluate()
    
    # 5. Phân tích lỗi (Error Analysis)
    error_analyzer = ErrorAnalyzer()
    error_analyzer.analyze_common_confusions(labels, preds, top_k=5)
    
    # Lấy một batch ảnh từ test loader để vẽ hiển thị so sánh
    data_iter = iter(test_loader)
    images, target_labels = next(data_iter)
    
    # 6. Hiển thị predicted vs actual images (Yêu cầu đề bài)
    visualizer = PredictionVisualizer()
    visualizer.plot_predictions(
        images=images, 
        true_labels=target_labels.numpy(), 
        pred_labels=preds[:len(images)], 
        num_samples=20, 
        save_path='results/runs/predictions.png'
    )
    
    # 7. Vẽ Confusion Matrix chi tiết
    cm_plotter = ConfusionMatrixPlotter()
    cm_plotter.plot(labels, preds, normalize=True, save_path='results/runs/confusion_matrix.png')
    
    # 8. Lưu final model kèm các metrics đánh giá hoàn chỉnh
    ModelIO.save_model(
        model=model,
        save_path='models/checkpoints/final_model.pt',
        model_name='simple_cnn',
        metrics=metrics
    )
    print("\nHoàn thành toàn bộ quá trình Evaluation cho Người số 3!")

if __name__ == '__main__':
    main()
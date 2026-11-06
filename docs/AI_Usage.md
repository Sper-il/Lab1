# AI Usage

> Hồ sơ trung thực về cách các công cụ AI được sử dụng trong project này.

## 1. Tools Used

| Vai trò | Tool / Model | Phiên bản | Khi nào |
|---------|--------------|-----------|---------|
| Code generation | Cursor + Claude | 2026-10 | Scaffold module, viết boilerplate PyTorch |
| Documentation | Cursor + Claude | 2026-10 | Nháp README, báo cáo, docs chuẩn skill |
| Debugging | Cursor + Claude | 2026-10 | Phân tích traceback, đề xuất root cause |
| Research | ChatGPT / Claude | 2026-10 | So sánh kiến trúc, gợi ý hyperparameter khởi điểm |
| Review | Cursor + Claude | 2026-10 | Review code AI generate, phát hiện edge case |
| Completion | GitHub Copilot | 2026-10 | Gợi ý completion ngắn |

## 2. Plain Purposes

- **Code generation** — scaffold module từ thiết kế đã duyệt, viết hàm tiện ích (DataLoader, training loop, evaluator).
- **Documentation** — nháp README, `progress_status.md`, báo cáo Người 1/2/3 và docs chuẩn (`description.md`, `data.md`, `api.md`).
- **Debugging** — đề xuất root cause (lỗi split lệch class, BatchNorm flaky với batch nhỏ), gợi ý log cần thêm.
- **Research** — so sánh MLP vs CNN vs ResNet, tóm tắt best practice normalize/regularization cho FashionMNIST.
- **Review** — review code AI đã viết, phát hiện edge case (batch cuối nhỏ hơn, non-finite loss, boolean-as-int).

## 3. Student Contributions (con người làm gì)

- Định nghĩa bài toán, phạm vi và mapping yêu cầu `đề.txt` (chia 3 người trong `PHASE2_PLAN.md`).
- Quyết định kiến trúc cuối cùng: chọn `SimpleCNN` vì đứng đầu grid search (val 92,33%), nhanh gấp 2,9× ResNetMini mà chỉ kém 0,13%.
- Quyết định split 50k/10k/10k (`val_ratio=1/6`, `seed=42`), chọn best theo `val_loss` (`patience=5`, `min_delta=0,0001`).
- Toàn bộ phương pháp đánh giá và chọn metric (accuracy/precision/recall/F1 macro + confusion matrix + error analysis).
- Toàn bộ review cuối cùng cho code AI generate (đặc biệt path CPU/default).
- Toàn bộ commit (AI có thể nháp, người quyết định) và kết luận thí nghiệm (test 92,10%, vượt mục tiêu 90%).
- Chạy thật grid search 16 trials × 10 epochs (~4,5 giờ CPU) và full training 15 epochs trên máy local.

## 4. Important AI-Generated Artifacts

| File | Nội dung | Prompt đã dùng (tóm tắt) |
|------|----------|--------------------------|
| `src/models/architectures/*.py` | `DeepMLP`, `SimpleCNN`, `ResNetMini` + `BasicBlock` | "Generate CNN/ResNet-mini cho FashionMNIST bằng PyTorch, có batch norm và dropout, input (B,1,28,28) → logits (B,10)" |
| `src/training/trainer.py` | Training loop forward → loss → backward → optimize + validation weighted | "Create complete training loop in PyTorch với CrossEntropyLoss, Adam, early stopping theo val_loss, logging từng epoch" |
| `src/evaluation/*.py` | Evaluator, confusion matrix, error analysis, PredictionVisualizer, ModelIO | "Implement evaluation code cho PyTorch classifier gồm accuracy, classification report, confusion matrix heatmap, predicted vs actual grid xanh/đỏ" |
| `docs/*.md`, `docs/reports/*.md` | Nháp README, methodology, báo cáo 3 người, docs chuẩn skill | "Viết báo cáo training/evaluation từ metrics thật trong results/runs, giữ số liệu nguyên gốc, tiếng Việt có dấu" |

## 5. Prompt Purpose (cho mỗi artifact)

- **`src/models/architectures/*.py`** — prompt yêu cầu AI scaffold 4 kiến trúc theo plan đã duyệt, dùng `CrossEntropyLoss`-friendly logits (không softmax cuối), `BatchNorm` chỉ cho MLP sâu + ResNet.
- **`src/training/trainer.py`** — prompt yêu cầu AI viết training loop đầy đủ forward/loss/backward/optimize kèm validation `eval()` + `no_grad()`, weighted metrics cho batch cuối lẻ.
- **`src/evaluation/*.py`** — prompt yêu cầu AI viết evaluator + confusion matrix normalized + top-K confused pairs + grid predicted vs actual, giải thích rõ vì sao train loss = NaN khi gặp non-finite thì phải skip/clip.
- **`docs/`** — prompt yêu cầu AI tóm tắt quyết định thiết kế và kết quả thật từ `hyperopt.json` / `training_history.json` / `test_metrics.md`, không bịa số.

## 6. Student Modifications

- **`src/data/split.py`** — sửa import `from dataset import …` → `from src.data.dataset import …` và thêm `np.random.shuffle(stratified_indices)` vì bản AI thiếu shuffle toàn cục khiến val lệch class (947–1.040/lớp); sau fix đúng 5.000/1.000/1.000 mỗi lớp.
- **`run_experiments.py`** — đổi smoke test từ 4 samples → 64 samples × 5 steps vì BatchNorm với batch=4 quá nhiễu gây flaky fail.
- **`src/training/callbacks.py` + `trainer.py`** — siết `min_delta` chính xác, xử lý non-finite values và boolean-as-int sau review, vì bản AI ban đầu thiếu các guard này (suite sau đó đạt 27/27).
- **`src/training/*`** — giữ optimizer Adam + `SimpleCNN` theo kết quả grid thật, không theo gợi ý generic của AI; đổi sang chọn best theo `val_loss` thay vì `val_acc` vì ổn định hơn khi overfit nhẹ.
- **`docs/`** — con người kiểm tra từng con số (92,33% grid, 92,10% test, epoch 11 best) khớp artifact trước khi commit; bỏ random search khỏi project để giữ grid tất định phủ kín.

## 7. Validation Method

- [x] Chạy code end-to-end trên môi trường sạch (`python main.py`, `python run_experiments.py --smoke`, `python src/evaluation/run_evaluation.py`).
- [x] So sánh output với baseline / đáp án đã biết (pilot `compare_architectures` xác nhận thứ tự simple_cnn > resnet_mini > baseline_mlp > deep_mlp).
- [x] Viết unit/integration test mà không có thay đổi thì sẽ fail (`tests/training`, 27 passed).
- [x] Reproduce kết quả từ fresh clone + ghi lại command (split `seed=42`, grid 16 trials, full train 15 epochs).
- [x] Sanity-check metric theo Definition of Done (test 92,10% ≥ 90%, loss curves hội tụ, confusion matrix giải thích được cặp Shirt ↔ T-shirt/top).

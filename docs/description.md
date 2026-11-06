# Mô tả Project (Define — 6 mục bắt buộc)

> Nguồn chuẩn: `ai-fusion-workflow` — `modules/description.md` + `checklists/define-questions.md`.
> Trạng thái: đã đủ 6/6 mục, không còn TBD. Chi tiết tiến độ xem `progress_status.md`.

## 1. Project name

| Trường | Giá trị |
|--------|---------|
| Tên đầy đủ | Phân loại ảnh thời trang FashionMNIST bằng PyTorch (FashionMNIST Classification with PyTorch) |
| Slug | `lab1` / `fashionmnist-pytorch` |
| Tagline | Phân loại 10 nhóm sản phẩm thời trang từ ảnh xám 28×28 với accuracy test 92,10% trên CPU. |

## 2. Problem (bài toán là gì, vì sao quan trọng?)

- **Bài toán:** Multi-class image classification trên tập FashionMNIST của Zalando (60.000 train + 10.000 test, ảnh xám 28×28, 10 lớp).
- **Yêu cầu theo `đề.txt`:** học Tensors, Datasets/DataLoaders, Transforms, Model Building, Autograd, Optimization, Model Saving/Loading; hiện thực load dataset, apply transforms, build network, training loop (forward, loss, backward, optimize), evaluate accuracy, save/load model; thử nghiệm network/hyperparameters, visualize loss, display predicted vs actual; bàn giao code, brief report, loss graphs, image displays.
- **Vì sao quan trọng:** FashionMNIST khó hơn MNIST chữ số viết tay (nhiều lớp dễ nhầm như Shirt/T-shirt/top, Coat/Pullover), là bài tập chuẩn để nắm vững pipeline PyTorch trước khi sang bài toán CV thực tế.
- **Định lượng:** mục tiêu accuracy ≥ 90% — kết quả đạt **92,10% test** (SimpleCNN, 15 epochs, CPU).

## 3. Target users (ai dùng, bối cảnh nào, kỳ vọng gì?)

| Nhóm | Bối cảnh | Kỳ vọng ở output |
|------|----------|------------------|
| Sinh viên trong team (3 người) | Học PyTorch theo đề, mỗi người phụ trách 1 module | Code modular chạy được, báo cáo riêng + báo cáo tổng hợp |
| Giảng viên / người chấm | Kiểm tra đủ yêu cầu đề + tái lập kết quả | `python main.py` chạy end-to-end, metrics + plots khớp báo cáo |
| Người học PyTorch bên ngoài | Tham khảo pipeline mẫu | README + `QUICK_START.md` + `TEAM_GUIDE.md` đủ để clone và chạy lại |

## 4. Input / Output (dữ liệu vào/ra, shape, nguồn, chia tập)

- **Input:** ảnh xám 1 kênh 28×28, tensor shape `(1, 28, 28)`, giá trị pixel `[0, 1]`, chuẩn hóa về mean `0,2860` / std `0,3530`. Nguồn: `torchvision.datasets.FashionMNIST` tải về `data/processed/FashionMNIST/raw/`.
- **Output:** logits 10 lớp shape `(batch, 10)` (chưa softmax, dùng với `CrossEntropyLoss`); khi inference chuyển thành label `0–9` + confidence (softmax) + thời gian inference.
- **Chia tập:** 60.000 train gốc → **50.000 train / 10.000 val** (stratified, `seed=42`, `val_ratio=1/6`); test giữ nguyên **10.000** của FashionMNIST. Mỗi lớp đúng 5.000/1.000/1.000. Metadata: `data/splits/split_info.txt`. Chi tiết xem `docs/data.md` và `results/runs/dataset_info.md`.

## 5. Core features (Must / Should / Could + phạm vi loại trừ)

| Độ ưu tiên | Tính năng | Lý do |
|------------|-----------|-------|
| Must | Data pipeline: EDA, cleaning, preprocessing, stratified split | Nền tảng cho mọi bước sau (Phase 1) |
| Must | 4 kiến trúc + ModelFactory + grid search 16 trials | Trả lời "Experiment with network/hyperparameters" trong đề |
| Must | Training loop forward → loss → backward → optimize + early stopping + checkpoint | Yêu cầu cốt lõi của đề |
| Must | Evaluate accuracy/precision/recall/F1 + confusion matrix + error analysis | Trả lời "Evaluate model accuracy" |
| Must | Save/load model + display predicted vs actual | Yêu cầu "Save and load", "Display predicted vs actual" |
| Must | Loss/accuracy curves + predictions plot + báo cáo 3 người + báo cáo tổng hợp | Deliverables của đề |
| Should | Early stopping theo `val_loss`, best-weight restoration, CLI override, seed reproducibility | Chống overfitting, dễ tái lập |
| Should | Unit tests 27 cases cho training | Bảo đảm refactor không vỡ pipeline |
| Could | TensorBoard / W&B, augmentation, GradCAM, export ONNX, REST API | Nêu trong Limitations & Future, chưa làm |
| Loại trừ | Không deploy production, không xử lý ảnh màu/domain khác, không tối ưu GPU/edge | Vượt phạm vi đề Lab 1 |

## 6. Technology criteria (Where / When / Which + tiêu chí chọn)

- **Where (chạy ở đâu):** máy local Windows + PowerShell, CPU-only (CUDA không khả dụng). Vì vậy chọn model nhỏ (SimpleCNN 421.642 params, ~75 s/epoch) thay vì ResNetMini chậm gấp 2,9× mà chỉ hơn 0,13% val_acc ở grid search.
- **When (khi nào):** tháng 10/2026, Phase 1 (data) → Phase 2 (3 người song song: network / training / evaluation) → tổng hợp. Timeline chi tiết xem `progress_status.md` mục Milestones.
- **Which (bằng gì):** PyTorch 2.14.1+cpu, Python 3.8+, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn (metrics), PyYAML (config), pytest (tests). Không dùng framework nặng (Lightning, HF Trainer) để sinh viên nắm rõ từng dòng forward/backward.
- **Tiêu chí chọn:** (1) đúng yêu cầu đề từng dòng, (2) chạy được trên CPU lab, (3) giấy phép mở (BSD/MIT/Apache), (4) kỹ năng team là Python cơ bản + PyTorch mới học, (5) tái lập bằng 1 lệnh (`python main.py`, `python run_experiments.py --smoke`, `python src/evaluation/run_evaluation.py`).
- **Tiêu chí đánh giá:** accuracy test ≥ 90% (đạt 92,10%); precision/recall/F1 macro ~92,1%; loss curves hội tụ; confusion matrix giải thích được cặp nhầm chính (Shirt ↔ T-shirt/top).

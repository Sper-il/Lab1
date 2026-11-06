# Experiment Log

## Format

Each experiment entry follows:
- **Date:** YYYY-MM-DD
- **Name:** Experiment name
- **Model:** Model type used
- **Hyperparameters:** Key settings
- **Result:** Final accuracy/loss
- **Notes:** Observations

---

## Experiments

### Phase 1: Data Preparation (Completed 2026-10-11)

- **Name:** Data Pipeline Setup
- **Model:** N/A
- **Result:** All 4 data prep steps complete
- **Notes:** EDA, cleaning, preprocessing, splitting implemented

### Phase 2 — Người 1: Architecture pilot + Grid search (2026-10-12 → 2026-10-22)

- **Date:** 2026-10-12
- **Name:** `compare_architectures` pilot (2 epochs, lr=1e-3)
- **Model:** baseline_mlp / deep_mlp / simple_cnn / resnet_mini
- **Hyperparameters:** epochs=2, lr=0,001, Adam (mặc định pilot)
- **Result:** simple_cnn val 89,48% (421.642 params, 96,30 s/epoch) > resnet_mini 88,63% > baseline_mlp 87,05% > deep_mlp 86,10%. Artifact: `results/runs/architecture_selection.md`
- **Notes:** Thứ tự kiến trúc ổn định, CNN vượt MLP ngay từ pilot.

- **Date:** 2026-10-22
- **Name:** Grid search full 16 trials × 10 epochs (4 model × 2 lr × 2 optimizer, 160 epochs CPU ~4,5 giờ)
- **Model:** cả 4 kiến trúc qua `ModelFactory`
- **Hyperparameters:** xem `configs/hyperopt_config.yaml`; batch 128, epochs 10/trial, seed 42
- **Result:** best trial 9 — `simple_cnn` + Adam + lr=0,001 → train 92,79% / val **92,33%**; runner-up trial 15 — `resnet_mini` + Adam + lr=0,0005 → val 92,20% (train 95,60%, overfit gap 3,4%). Adam hơn SGD 2–6% mọi kiến trúc. Artifacts: `results/runs/hyperopt.json`, `hyperopt.md`, `hyperopt_best.json`
- **Notes:** `simple_cnn` nhanh gấp 2,9× `resnet_mini` (~75 vs ~218 s/epoch) nên được chọn train full. Lệnh tái tạo: `python run_experiments.py --batch-size 128`.

- **Date:** 2026-10-12
- **Name:** Real training 3 epochs best config (integration verify)
- **Model:** simple_cnn
- **Hyperparameters:** Adam, lr=0,001, batch 128, dropout 0,5
- **Result:** epoch 3 val **90,12%** (train 88,79%, val_loss 0,2689); checkpoint `models/checkpoints/best_model.pt` (1.690.359 bytes)
- **Notes:** Chứng minh `ModelFactory.create("simple_cnn")` train được bằng trainer chuẩn, bàn giao cho Người 2 resume.

### Phase 2 — Người 2: Full training (2026-10-14)

- **Date:** 2026-10-14
- **Name:** Official full run 15 epochs (`python main.py`, bs=128, seed=42, CPU)
- **Model:** simple_cnn (421.642 params)
- **Hyperparameters:** Adam lr=0,001, wd=0,0, dropout 0,5, CrossEntropyLoss, early stopping `val_loss/min/patience=5/min_delta=0,0001`
- **Result:** best epoch 11 (val_loss **0,215183**, val_acc 92,85%); đỉnh val_acc 92,88% epoch 13; last epoch 15 (train 94,72%, val 92,80%). Early stopping không kích hoạt (4 epoch không cải thiện < patience 5). Artifacts: `models/checkpoints/trained_model.pt` (best), `last_model.pt` (local-only), `results/runs/training_history.json`, `loss_curve.png`, `accuracy_curve.png`
- **Notes:** Overfitting nhẹ cuối run; chọn best theo `val_loss` cho weights ổn định. Tests: `pytest tests -q` → 27 passed.

### Phase 2 — Người 3: Evaluation (2026-10-15)

- **Date:** 2026-10-15
- **Name:** Test evaluation (`python src/evaluation/run_evaluation.py`, checkpoint `trained_model.pt`)
- **Model:** simple_cnn (best weights epoch 11)
- **Hyperparameters:** N/A (inference trên 10.000 test)
- **Result:** test accuracy **92,10%**, precision 92,11%, recall 92,10%, F1 92,09% (macro). Top confused: Shirt→T-shirt/top 95, T-shirt/top→Shirt 90, Shirt→Pullover 64, Coat→Pullover 63, Coat→Shirt 55. Artifacts: `models/checkpoints/final_model.pt`, `results/runs/confusion_matrix.png`, `predictions.png`, `results/runs/test_metrics.md`
- **Notes:** Vượt mục tiêu 90%. Nhóm Shirt/T-shirt/top/Pullover/Coat là nguồn lỗi chính do hình dáng tương tự.

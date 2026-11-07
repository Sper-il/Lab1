# Documentation Index

Danh sách tất cả tài liệu trong project FashionMNIST.

## Project Documentation (Root Level)

| File | Description |
|------|-------------|
| [README.md](../README.md) | Project overview & getting started |
| [progress_status.md](../progress_status.md) | Project progress tracking |
| [requirements.txt](../requirements.txt) | Python dependencies |
| [.gitignore](../.gitignore) | Git ignore patterns |
| [data/README.md](../data/README.md) | Data folder guide (cách sinh lại data) |

## Documentation (This Folder)

### Guides & Reports

| File | Description |
|------|-------------|
| [PHASE1_REPORT.md](./PHASE1_REPORT.md) | Phase 1 summary report |
| [PHASE2_PLAN.md](./PHASE2_PLAN.md) | Phase 2 work plan (3 people, aligned với đề) |
| [PHASE2_FINAL_REPORT.md](./PHASE2_FINAL_REPORT.md) | Phase 2 final summary (best model, metrics, lessons) |
| [QUICK_START.md](./QUICK_START.md) | Quick start guide |
| [AI_Usage.md](./AI_Usage.md) | AI assistance report (chuẩn 7 mục) |
| [audit_code.md](./audit_code.md) | Code audit report |
| [CHANGELOG.md](./CHANGELOG.md) | Version history |

### Reference Documents

| File | Description |
|------|-------------|
| [architecture.md](./architecture.md) | System architecture |
| [model.md](./model.md) | Mô tả Model (baseline, backbone + head, loss, params, config) |
| [description.md](./description.md) | Mô tả project (Define 6 mục) |
| [data.md](./data.md) | Quy ước dữ liệu |
| [data-flow.md](./data-flow.md) | Luồng dữ liệu |
| [api.md](./api.md) | API nội bộ giữa các module |
| [training.md](./training.md) | Báo cáo tổng quan Training (báo cáo chính, không phải cá nhân) |
| [evaluation.md](./evaluation.md) | Báo cáo tổng quan Evaluation (báo cáo chính, không phải cá nhân) |
| [demo.md](./demo.md) | Báo cáo tổng quan Demo (kịch bản + cách chạy + roadmap 1 lệnh) |
| [TEAM_GUIDE.md](./TEAM_GUIDE.md) | Hướng dẫn chạy cho cả team (3 người) |
| [PROJECT_METHODOLOGY.md](./PROJECT_METHODOLOGY.md) | Quy trình thực hiện + giải thích kỹ thuật |
| [PRESENTATION_CHEATSHEET.md](./PRESENTATION_CHEATSHEET.md) | Kịch bản thuyết trình 10 phút |
| [EXPERIMENTS.md](./EXPERIMENTS.md) | Experiment log |

## Phase Reports (Subfolder: `reports/`)

### Individual Reports (Phase 2 - 3 người)

| File | Person | Module (theo đề) |
|------|--------|------------------|
| [reports/nguoi1_models_report.md](./reports/nguoi1_models_report.md) | Người 1 | Build Network + Hyperparameter Experiment |
| [reports/nguoi2_training_report.md](./reports/nguoi2_training_report.md) | Người 2 | Training Loop + Visualize Loss |
| [reports/nguoi3_evaluation_report.md](./reports/nguoi3_evaluation_report.md) | Người 3 | Evaluate + Save/Load + Display Predictions |

### Templates

| File | Description |
|------|-------------|
| [reports/REPORT_TEMPLATE.md](./reports/REPORT_TEMPLATE.md) | Template for new reports |

### Phase Integration Reports

| File | Description |
|------|-------------|
| [PHASE2_FINAL_REPORT.md](./PHASE2_FINAL_REPORT.md) | Phase 2 final summary (best model, metrics, lessons, recommendations) |

---

## Reading Order

### For New Team Members
1. Start with [QUICK_START.md](./QUICK_START.md)
2. Read [PHASE1_REPORT.md](./PHASE1_REPORT.md)
3. Read [PHASE2_PLAN.md](./PHASE2_PLAN.md)
4. Pick your assigned module

### For Documentation Writers
1. Read [REPORT_TEMPLATE.md](./reports/REPORT_TEMPLATE.md)
2. Fill in your assigned report (e.g., `nguoi1_models_report.md`)

### For Project Managers
1. Read [PHASE2_PLAN.md](./PHASE2_PLAN.md)
2. Check [progress_status.md](../progress_status.md)
3. Review individual reports in [reports/](./reports/)

---

## Contributing

When adding new documentation:
1. Place in appropriate location (root or `docs/`)
2. Update this index
3. Add an entry in [CHANGELOG.md](./CHANGELOG.md)

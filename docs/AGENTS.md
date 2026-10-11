# AGENTS.md — Working rules cho AI agent

> File này Cursor agent tự đọc khi làm việc trong project. Mục đích: đảm bảo
> AI agent tuân thủ convention + Definition of Done mà không cần hỏi lại.
> Cập nhật lần cuối: 2026-10-11 18:13

## Nguyên tắc bắt buộc

1. **AI là công cụ, người là chủ.** Mọi quyết định kiến trúc phải được tác giả duyệt.
2. **Define → Design → Build MVP → Document → Eval** — không bao giờ code trước khi định nghĩa.
3. **Một module một lần, một thay đổi một lần** — tránh mega-commit.
4. **Describe → Generate → Review → Run → Test → Refactor → Commit** — vòng lặp chuẩn.
5. **CONTEXT là vua** — luôn cập nhật `progress_status.md` sau mỗi module.

## Naming convention

| Loại | Format | Ví dụ |
|------|--------|-------|
| Model | `{domain}_{architecture}_{version}.pt` | `fashion_cnn_v1.pt` |
| Config | `{component}_config.yaml` hoặc `baseline.yaml` | `baseline.yaml` |
| Script | `{action}_{target}.py` | `train_model.py`, `eval_pipeline.py` |
| Test | `test_{module}.py` | `test_dataset.py` |
| Run folder | `results/runs/YYYYMMDD-HHMM/` | `results/runs/20261011-1708/` |

## Folder convention (đã thiết lập)

```
lab1/
├── data/raw/{train,val,test}/<class_id>/*.png    # 70.000 ảnh, gitignore
├── src/{data,models,training,evaluation,visualization}/
├── tests/test_{module}.py                          # 1 file / module
├── configs/baseline.yaml
├── docs/{architecture,data-flow,api}.md
├── results/runs/<ts>/{metrics.json,model.pt,*.png}
├── progress_status.md                              # single source of truth
├── README.md                                       # 10 mục chuẩn
├── AI_Usage.md                                     # 7 mục chuẩn
├── CHANGELOG.md                                    # model/dataset version
└── AGENTS.md                                       # file này
```

## Discussion rule (quyết định kiến trúc)

Mọi quyết định quan trọng phải ghi vào `progress_status.md` mục "Decisions Log":

```
| Ngày | Quyết định | Lựa chọn khác | Lý do |
```

Đã có sẵn 6 quyết định (xem file `progress_status.md` dòng ~244-260).

## Edition rule (version + tag)

- **Model**: mỗi version lưu `<run-folder>/model.pt` + commit hash tương ứng.
- **Dataset**: 70.000 ảnh là cố định (FashionMNIST gốc), commit hash đầu tiên
  tạo ra data là "v1.0".
- **CHANGELOG.md**: ghi lại mỗi version model (test_acc, ngày, commit).

## Definition of Done (áp dụng mỗi module)

Trước khi commit, **AI agent PHẢI tự kiểm tra 8 mục** và báo cáo:

- [ ] Đúng yêu cầu (đáp án What/Why/Where-When-Which/How).
- [ ] Code chạy được end-to-end trên môi trường sạch.
- [ ] Test pass (unit + integration).
- [ ] Không có secrets, không commit file lớn.
- [ ] Code đã self-review (không có TODO/FIXME trừ khi có ticket).
- [ ] Có thể giải thích được từng dòng cho người khác.
- [ ] Docs đã cập nhật (`README.md` / `progress_status.md` / `AI_Usage.md`).
- [ ] Đã commit theo Conventional Commit.

## Conventional Commit (chuẩn bắt buộc)

Format: `<type>(<scope>): <subject>` — subject tiếng Việt, không quá 72 ký tự.

| Type | Khi nào |
|------|---------|
| `feat` | Thêm tính năng / module mới |
| `fix` | Sửa bug |
| `docs` | Chỉ thay đổi docs |
| `refactor` | Sửa code nhưng không thay đổi behavior |
| `test` | Thêm/sửa test |
| `chore` | Cấu hình, .gitignore, requirements |

**Ví dụ commit đã dùng trong project:**
- `ff2e11a feat(data): M1 data module — split FashionMNIST ra .png, build DataLoaders`
- `e32241d docs: điền đầy đủ architecture, data-flow, api dựa trên M0+M1 đã ship`

## Prompt template chuẩn

Có 4 file trong `.cursor/skills/ai-fusion-workflow/prompts/`:

1. `plan.md` — lập kế hoạch cho 1 module
2. `implement.md` — viết code cho 1 module
3. `review.md` — review code AI đã generate
4. `debug.md` — gỡ lỗi khi test fail

Mỗi khi bắt đầu hành động tương ứng, copy file prompt, điền thông tin module cụ thể.

## Lệnh thường dùng (đã chạy thực tế)

```bash
# Sinh dữ liệu trên disk (1 lần, ~2-3 phút)
python -m src.data.split --root data/raw --val-split 10000 --seed 42

# Build 3 DataLoader + tóm tắt
python -c "from src.data.dataset import build_dataloaders, summarize; loaders = build_dataloaders('configs/baseline.yaml'); [summarize(l, n) for n, l in loaders.items()]"

# Chạy test
pytest -v
pytest tests/test_dataset.py -v
```

## Không được làm

- ❌ Commit code chưa chạy được trên máy thật.
- ❌ Commit secrets, API key, credential.
- ❌ Commit data lớn (data/, *.pt, results/.../model.pt đã gitignore).
- ❌ Tạo dataset giả / fake accuracy.
- ❌ Bỏ qua test fail.
- ❌ Sửa code ngoài phạm vi module đang làm (mega-commit).
- ❌ Tự ý thay đổi convention đã ghi trong file này.

## Khi nào hỏi user

AI agent nên dùng `AskQuestion` khi:

- Có ≥ 2 cách hợp lý và cần user chọn (vd: optimizer Adam vs SGD).
- Phát hiện rủi ro mới chưa có trong `Risks & Mitigations`.
- Test fail mà không rõ root cause.
- Cần xác nhận trước khi commit phá vỡ convention.

## Liên kết

- `progress_status.md` — mục Project Context, Architecture Snapshot, Active Module, Module Log, Decisions Log, Risks, DoD
- `README.md` — 10 mục chuẩn
- `AI_Usage.md` — 7 mục chuẩn
- `.cursor/skills/ai-fusion-workflow/SKILL.md` — workflow tổng thể

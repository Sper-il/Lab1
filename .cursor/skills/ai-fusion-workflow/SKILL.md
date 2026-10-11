---
name: ai-fusion-workflow
description: Quy trình chuẩn để làm project AI/ML từ Define → Design → Build MVP → Document → Eval. Áp dụng khi người dùng bắt đầu một project AI mới, scaffold thư mục, viết README.md, AI_Usage.md, progress_status.md, hoặc khi cần trả lời bộ câu hỏi What/Why/Where-When-Which/How trước khi code. Auto-invoke khi thấy project AI/ML hoặc yêu cầu lập kế hoạch.
---

# AI Fusion Workflow

Quy trình chuẩn để làm project AI/ML có trách nhiệm, có tài liệu, và đo lường được.

## Nguyên tắc cốt lõi

1. **Define trước, code sau** — không bao giờ viết dòng code nào khi chưa trả lời được bộ câu hỏi bắt buộc.
2. **MVP trước, hoàn thiện sau** — build module nhỏ nhất chạy được, rồi mới nâng cấp.
3. **Một module một lần, một thay đổi một lần** — tránh mega-commit.
4. **Describe → Generate → Review → Run → Test → Refactor → Commit** — vòng lặp chuẩn.
5. **CONTEXT là vua** — mọi quyết định, lý do, kết quả đều phải được ghi vào `progress_status.md`.
6. **AI là công cụ, người là chủ** — mọi output AI phải được review, sửa, và xác minh trước khi merge.

## Khi nào dùng skill này

- Người dùng mở một project AI/ML mới và muốn scaffold.
- Người dùng yêu cầu lập kế hoạch, định nghĩa yêu cầu, thiết kế kiến trúc.
- Người dùng viết hoặc cập nhật `progress_status.md`, `README.md`, `AI_Usage.md`.
- Người dùng nói: "định nghĩa yêu cầu", "thiết kế hệ thống", "đánh giá model", "làm MVP", "tổ chức project".

## Khi nào KHÔNG dùng

- Project không phải AI/ML (web thuần, CRUD đơn giản).
- Câu hỏi debug lỗi đơn lẻ không liên quan cấu trúc project.
- Đang trong phiên sửa bug nóng, cần fix ngay trong vài phút.

## Quy trình tổng thể

```
Business Understanding (User's requirements)
        |
        v
  Features  +  Tech Solution  +  Logic (AI)  +  Implementation
        |
        v
  Demo  +  Test (Model level  +  Full pipeline)
```

Áp dụng theo 5 giai đoạn:

### Giai đoạn 1 — DEFINE
Trả lời bộ câu hỏi bắt buộc (xem `checklists/define-questions.md`):
- **What?** — bài toán là gì? Dữ liệu gì? Output mong đợi?
- **Why?** — tại sao chọn cách này? Tại sao chọn model/feature/tech này?
- **Where / When / Which?** — chạy ở đâu? Khi nào? Thư viện nào?
- **How?** — dữ liệu chảy thế nào? Train/deploy thế nào? Đo lường ra sao?

Sau khi trả lời, điền vào bảng "Project Context" trong `progress_status.md`.

### Giai đoạn 2 — DESIGN
Điền mục "Architecture Snapshot" trong `progress_status.md`:
- **System architecture** — sơ đồ tổng thể.
- **Modules** — danh sách module và trách nhiệm.
- **Folder structure** — cấu trúc thư mục theo chuẩn (xem `templates/folder-structure.md`).
- **Data flow** — dữ liệu đi từ đâu đến đâu.
- **Interfaces / APIs** — đầu vào, đầu ra từng module.
- **Milestones** — mốc thời gian theo tuần hoặc theo module.
- **Technical risks** — rủi ro kỹ thuật và cách giảm thiểu.

### Giai đoạn 3 — BUILD MVP (theo module)
Với mỗi module, lặp lại vòng:
1. **Describe** — module làm gì, input/output, tiêu chí thành công.
2. **Generate** — dùng prompt "Implement" (xem `prompts/implement.md`).
3. **Review** — tự review, sửa theo nhận xét AI (xem `prompts/review.md`).
4. **Run** — chạy thử, ghi log vào `results/runs/<ts>/`.
5. **Test** — unit test cho flow, integration test cho pipeline (xem `prompts/debug.md` nếu fail).
6. **Refactor** — dọn code, thêm docstring, cập nhật README.
7. **Commit** — dùng Conventional Commit, một module = một commit (hoặc một nhóm commit nhỏ có liên quan).

### Giai đoạn 4 — DOCUMENT
Luôn cập nhật song song:
- `README.md` — 10 mục chuẩn (xem `templates/readme.md`).
- `AI_Usage.md` — 7 mục chuẩn (xem `templates/ai-usage.md`).
- `progress_status.md` — module log, decisions log, risks.

### Giai đoạn 5 — EVAL
Hai lớp đánh giá bắt buộc:
- **Model level** — accuracy / precision / recall / F1 / mAP / IoU / inference time, tùy bài toán.
- **Full pipeline** — latency end-to-end, throughput, resource usage, error handling.

## Quy tắc làm việc với AI (AI Reference)

### Naming convention
- Model: `{domain}_{architecture}_{version}.{ext}` — ví dụ `face_unet_v1.pt`.
- Dataset: `{purpose}_{date}.{ext}` — ví dụ `train_faces_20261015.zip`.
- Script: `{action}_{target}.py` — ví dụ `train_model.py`, `eval_pipeline.py`.
- Config: `{component}_config.yaml` — ví dụ `model_config.yaml`.
- Run folder: `results/runs/YYYYMMDD-HHMM/`.

### Folder convention
Theo `templates/folder-structure.md`. Tóm tắt:
```
project-root/
├── data/{raw, processed, splits}
├── src/{data, models, training, evaluation}
├── models/{checkpoints, trained, configs}
├── notebooks/
├── tests/
├── docs/
├── configs/
├── results/runs/<ts>/
├── README.md
├── AI_Usage.md
└── progress_status.md
```

### Discussion rule
Mọi quyết định quan trọng phải ghi vào `progress_status.md` mục "Decisions Log":
```
[YYYY-MM-DD] Decision: <what> | Reasoning: <why> | Alternatives: <what else>
```

### Edition rule
- Version mọi model và dataset.
- Tag release trong git.
- Giữ `CHANGELOG.md` cho model version.
- Ghi rõ breaking change.

## Definition of Done (áp dụng mỗi module)
- [ ] Đúng yêu cầu (đáp án của bộ câu hỏi What/Why/Where-When-Which/How).
- [ ] Code chạy được end-to-end trên môi trường sạch.
- [ ] Test pass (unit + integration).
- [ ] Không có secrets, không commit file lớn.
- [ ] Code đã được self-review.
- [ ] Có thể giải thích được từng dòng cho người khác.
- [ ] Docs đã cập nhật (README, progress_status, AI_Usage).
- [ ] Đã commit theo Conventional Commit.

## Prompt template chuẩn

Có 4 prompt đã soạn sẵn trong thư mục `prompts/`:
1. `plan.md` — lập kế hoạch cho một module.
2. `implement.md` — viết code cho một module.
3. `review.md` — review code AI đã generate.
4. `debug.md` — gỡ lỗi khi test fail.

Mỗi khi bắt đầu một hành động tương ứng, copy file prompt, điền thông tin module cụ thể, rồi gửi cho AI.

## Lớp chuyên môn (Domain expertise)

Khi làm project AI, hãy tự đặt mình vào các vai trò:
- **Product Owner** — định nghĩa yêu cầu, ưu tiên tính năng.
- **Domain Expert** — cung cấp kiến thức chuyên ngành, validate output.
- **AI Architect** — thiết kế kiến trúc, chọn tech stack.
- **AI Engineer** — code model, build pipeline, tối ưu.
- **AI Operator** — deploy, monitor, maintain.
- **Security** — đảm bảo privacy, model security, compliance.

AI agent hoạt động theo 3 lớp:
```
Prompting (mô tả + luật)
        |
        v
Skill / Behavioral (ra quyết định)
        |
        v
Brain / Knowledge base (context dài hạn)
```

## CONTEXT — tại sao quan trọng

CONTEXT là thứ duy nhất khiến AI trả lời đúng. Khi mất context, AI sẽ:
- Đoán sai ý bạn.
- Lặp lại quyết định đã bác bỏ.
- Vi phạm convention đã thống nhất.

Cách giữ context:
- `progress_status.md` luôn được cập nhật sau mỗi module.
- Mỗi prompt phải chứa: mục tiêu module, ràng buộc, file liên quan, kết quả mong đợi.
- Không để AI "đoán" — ghi rõ input/output shape, metric, threshold.

## Tài nguyên trong skill

- `checklists/define-questions.md` — bộ câu hỏi What/Why/Where-When-Which/How đầy đủ.
- `checklists/dod.md` — Definition of Done chi tiết.
- `prompts/plan.md` — template prompt lập kế hoạch.
- `prompts/implement.md` — template prompt code module.
- `prompts/review.md` — template prompt review code.
- `prompts/debug.md` — template prompt gỡ lỗi.
- `templates/folder-structure.md` — cấu trúc thư mục chuẩn.
- `templates/readme.md` — README 10 mục.
- `templates/ai-usage.md` — AI_Usage 7 mục.
- `templates/progress-status.md` — progress_status đầy đủ.

## Cách dùng nhanh

Khi user bảo "bắt đầu project AI mới":
1. Đọc `checklists/define-questions.md` và hỏi user các câu hỏi chưa rõ.
2. Điền `progress_status.md` mục Project Context.
3. Đọc `templates/folder-structure.md` và scaffold thư mục.
4. Đọc `templates/readme.md`, `templates/ai-usage.md`, `templates/progress-status.md` để user có file mẫu.
5. Sau đó mới đến giai đoạn Design → MVP → Eval.

Khi user bảo "code module X":
1. Đọc `prompts/implement.md`, điền thông tin module, gửi cho AI.
2. Sau khi nhận code, dùng `prompts/review.md` để review.
3. Chạy và commit theo vòng lặp.

Khi user bảo "đánh giá model" / "đánh giá pipeline":
1. Tách rõ model level và full pipeline.
2. Ghi kết quả vào `results/runs/<ts>/metrics.json`.
3. Cập nhật `progress_status.md` mục "Module Log".
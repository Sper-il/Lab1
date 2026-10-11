# AI Usage — ghi nhận việc sử dụng AI

> **Mục đích:** Minh bạch việc dùng công cụ AI trong quá trình làm Practice 1.
> Cập nhật liên tục theo tiến độ project.

## Tóm tắt

| Mục | Giá trị |
|-----|---------|
| **Công cụ chính** | Cursor IDE với model **Claude Sonnet 4.5** (`claude-sonnet-4-5`) |
| **Phạm vi sử dụng** | Generate boilerplate code, self-review, debug, viết docs, planning |
| **Quyết định thiết kế** | Do tác giả đưa ra, AI hỗ trợ đánh giá ưu/nhược |
| **Kiểm chứng output** | Tất cả code chạy thực tế + pytest; không commit code không pass test |
| **Lưu ý** | AI đã giúp lên đến **~40%** thời gian làm bài (estimate từ số commit) |

## Nguyên tắc sử dụng

1. **AI là công cụ, không phải tác giả.** Mọi quyết định kiến trúc (tách 5 module,
   split trước / transform sau, dùng `.png` thay vì tensor trong RAM) đều do tác giả
   đề xuất; AI hỗ trợ review ưu/nhược điểm.
2. **Không commit code chưa chạy được.** Mỗi module đều chạy thực tế trên máy và
   pass test trước khi commit.
3. **Ghi rõ nguồn trong commit** nếu đoạn code lấy từ AI (xem bảng dưới).
4. **Không gửi data nhạy cảm.** Project không có API key / credential / data cá nhân.

## Công cụ AI đã dùng

### Cursor IDE (môi trường chính)
- **Model mặc định:** Claude Sonnet 4.5 (`claude-sonnet-4-5`)
- **Chế độ dùng:** Agent (tự chạy shell, đọc file) + Ask (giải thích concept)
- **Cách dùng điển hình:**
  - "Tạo skeleton module M1" → AI generate code → tác giả review → chạy thực tế
  - "Tại sao test fail" → AI đọc traceback + code → đề xuất fix
  - "Điền docs cho architecture" → AI viết dựa trên code thực tế

### Công cụ khác
- **Không dùng** ChatGPT / Gemini / Copilot trong project này (chỉ Cursor).
- **Không dùng** AI để tạo dataset hoặc data giả — toàn bộ data là FashionMNIST
  chính thức từ `torchvision`.

## Sử dụng theo giai đoạn

### Phase 1 — DEFINE (ngày 1)
| Việc | AI hỗ trợ | AI tự quyết |
|------|-----------|-------------|
| Viết `progress_status.md` (Project Context) | — | ✅ (do tác giả) |
| Chia 5 module M0–M5 | ✅ gợi ý | ✅ quyết định cuối |
| Quyết định CNN architecture cho M2 | ✅ đề xuất FashionCNN | ✅ quyết định layer count |

### Phase 2 — BUILD
#### M0 — Scaffold (commit initial)
- **AI hỗ trợ:** Generate cấu trúc thư mục, `__init__.py` rỗng, `requirements.txt`.
- **Output:** ~6 file Python skeleton.
- **Review:** Tác giả thêm `.gitignore` (data/, results/, .venv/), thêm
  `configs/baseline.yaml` theo schema thống nhất.

#### M1 — Data module (commit `ff2e11a`)
- **AI hỗ trợ:**
  - Generate `src/data/split.py` (~90 dòng): logic split + ghi `.png` ra disk.
  - Generate `src/data/dataset.py` (~80 dòng): `build_dataloaders()` + `summarize()`.
  - Generate 11 test trong `tests/test_dataset.py`.
- **AI tự quyết:** Cách tách split (1 lần) và transform (mỗi lần load) — AI đề xuất,
  tác giả chấp nhận sau khi đánh giá ưu điểm (chống data leakage, reproducible).
- **Verify:** Chạy 3 lần shell (lệnh `python -c "import torch..."`,
  `summarize(loaders)`, `summarize(loaders)` unbuffered) — tất cả success.
- **Test:** 11/11 pytest pass.

#### Docs (commit `e32241d`)
- **AI hỗ trợ:** Điền `docs/architecture.md` (137 dòng), `docs/data-flow.md` (132 dòng),
  `docs/api.md` (161 dòng) dựa trên code thực tế đã commit.
- **AI tự quyết:** Bố cục sơ đồ ASCII, bảng so sánh, danh sách API. Tác giả
  review + bổ sung phần "Quyết định kiến trúc".
- **File này (AI_Usage.md) + README.md:** Tác giả yêu cầu AI generate dựa trên
  thực tế project, không bịa.

## Tỉ lệ đóng góp (estimate)

| Loại công việc | % do AI | % do tác giả |
|----------------|---------|--------------|
| Boilerplate code (split, dataset, tests) | ~60% | ~40% review |
| Decision kiến trúc | ~10% | ~90% |
| Debug / fix test fail | ~50% | ~50% |
| Viết docs (architecture, data-flow, api) | ~80% | ~20% review |
| Viết `progress_status.md` | ~5% | ~95% |
| Chạy thực tế + verify | 0% (AI không chạy máy tác giả) | 100% |
| Git commit | 0% | 100% |

## Không dùng AI cho

- ❌ Tạo data giả / fake accuracy
- ❌ Copy code từ repo khác mà không hiểu
- ❌ Commit code chưa chạy được
- ❌ Bỏ qua test fail
- ❌ Tăng accuracy bằng cách "tweak số" không có cơ sở

## Kiểm chứng

- Tất cả code đã commit có thể reproduce: `git clone` → `pip install -r requirements.txt`
  → `python -m src.data.split` → `pytest -v` đều pass.
- Commit history tại `e:\lab1\.git\log` có thể kiểm tra bằng `git log --oneline`.
- 4 commit chính:
  1. `M0 — scaffold`
  2. `ff2e11a feat(data): M1 data module — split FashionMNIST ra .png, build DataLoaders`
  3. `ca59d91 docs(progress): mark M1 done, activate M2 model module`
  4. `e32241d docs: điền đầy đủ architecture, data-flow, api dựa trên M0+M1 đã ship`

## Cập nhật

| Ngày | Thay đổi |
|------|----------|
| 2026-10-11 | Tạo file, ghi nhận 4 commit đầu (M0, M1, mark M1 done, điền docs). |

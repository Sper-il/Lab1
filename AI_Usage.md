# AI Usage

> Hồ sơ trung thực về cách các công cụ AI được sử dụng trong project này.
> Format theo template 7 mục trong `.cursor/skills/ai-fusion-workflow/templates/ai-usage.md`.
> Cập nhật lần cuối: 2026-10-11 18:13.

## 1. Tools Used

| Vai trò | Tool / Model | Phiên bản | Khi nào dùng |
|---------|--------------|-----------|--------------|
| **Code generation** | Cursor IDE + Claude Sonnet 4.5 | `claude-sonnet-4-5` | Scaffold module, viết boilerplate (M0, M1) |
| **Documentation** | Cursor IDE + Claude Sonnet 4.5 | `claude-sonnet-4-5` | Điền docs/architecture, data-flow, api; README; AI_Usage |
| **Debugging** | Cursor IDE + Claude Sonnet 4.5 | `claude-sonnet-4-5` | Phân tích pytest fail, đề xuất root cause |
| **Research** | (không dùng AI — dùng PyTorch docs chính thức) | — | So sánh thư viện, tóm tắt API |
| **Review** | Cursor IDE + Claude Sonnet 4.5 | `claude-sonnet-4-5` | Self-review code, kiểm tra convention |

**Lý do chọn Cursor + Claude Sonnet 4.5:**
- Cursor là IDE chính của dự án (môi trường làm việc).
- Claude Sonnet 4.5 có cửa sổ context dài (200K token) → đọc được toàn bộ project.
- Hỗ trợ `Agent` mode (tự chạy shell, đọc file) phù hợp với vòng lặp
  Describe → Generate → Review → Run → Test → Refactor → Commit.

**Công cụ AI KHÔNG dùng trong project:**
- ❌ ChatGPT (OpenAI) — để giữ single source.
- ❌ GitHub Copilot — đã tắt trong IDE.
- ❌ Gemini / Claude.ai web — không tải code lên server bên ngoài.
- ❌ Cursor Tab autocomplete — tắt để buộc mình viết tay, học kỹ hơn.

## 2. Plain Purposes

- **Code generation** — AI scaffold module data loader theo plan đã duyệt
  trong `progress_status.md`. Tác giả review + sửa trước khi commit.
- **Documentation** — AI điền `docs/architecture.md`, `data-flow.md`, `api.md`,
  `README.md`, `AI_Usage.md`, `CHANGELOG.md`, `data/README.md`, `AGENTS.md`
  dựa trên code thực tế đã có. Tác giả review bố cục + bổ sung phần
  "Quyết định kiến trúc".
- **Debugging** — Khi pytest fail, AI đọc traceback + code → đề xuất root cause.
  Tác giả tự verify bằng cách chạy lại test trước khi áp dụng fix.
- **Review** — Tác giả dán code vào AI để hỏi "có edge case nào chưa cover?",
  "có convention nào bị vi phạm không?". AI đề xuất, tác giả quyết.

## 3. Student Contributions (con người làm gì)

| Công việc | Tác giả | AI hỗ trợ |
|-----------|---------|-----------|
| Định nghĩa bài toán (Practice 1 — FashionMNIST) | ✅ 100% | — |
| Quyết định 5 module M0–M5 | ✅ 100% | ✅ gợi ý chia module |
| Quyết định CNN architecture (2 conv + 2 FC) | ✅ 100% | ✅ đề xuất FashionCNN |
| Quyết định split-trước-transform-sau (`.png` trên disk) | ✅ 100% | ✅ gợi ý tách bước |
| Quyết định mean=0.2860, std=0.3530 | ✅ 100% (tham khảo PyTorch docs) | — |
| Quyết định chọn Adam, lr=1e-3, batch=64, epochs=10 | ✅ 100% | — |
| Viết `progress_status.md` (Project Context, Decisions Log) | ✅ 95% | ✅ 5% gợi ý format |
| Self-review code trước commit | ✅ 100% | ✅ checklist từ prompts/review.md |
| Tất cả commit (chọn message, push) | ✅ 100% | — |
| Chạy code thực tế + verify (pytest, summarize) | ✅ 100% | — |
| Đánh giá kết quả (accuracy, loss curve) | ✅ 100% | — |

**Tỉ lệ tổng thể (estimate):**
- AI: ~30% thời gian (boilerplate, docs, gợi ý)
- Tác giả: ~70% thời gian (quyết định, code, verify, commit, học)

## 4. Important AI-Generated Artifacts

| File | Nội dung | Prompt đã dùng (tóm tắt) |
|------|----------|--------------------------|
| `src/data/split.py` (90 dòng) | `split_fashionmnist_to_disk()` | "Viết hàm split FashionMNIST gốc thành 3 tập, lưu ra .png, idempotent, seed=42, CLI argparse." |
| `src/data/dataset.py` (80 dòng) | `build_dataloaders()`, `summarize()` | "Viết DataLoader cho 3 tập từ ImageFolder, transform ToTensor+Normalize, đọc config YAML, in summarize." |
| `tests/test_dataset.py` (130 dòng) | 11 test | "Viết 11 pytest cho M1: số file, class folder, batch shape, dtype, pixel range, idempotency." |
| `docs/architecture.md` (147 dòng) | Sơ đồ 5 module + design conventions | "Điền architecture.md dựa trên code thực tế đã có, đừng bịa, đánh dấu M2-M5 chưa code." |
| `docs/data-flow.md` (161 dòng) | Sơ đồ 2 bước split → transform | "Điền data-flow.md cho M1, có sơ đồ ASCII end-to-end M1→M5, ví dụ 1 ảnh." |
| `docs/api.md` (196 dòng) | API public + đặc tả M2–M5 | "Điền api.md với input/output/side effect cho mỗi hàm, error contracts." |
| `AGENTS.md` (130 dòng) | Working rules cho AI agent | "Viết AGENTS.md tổng hợp convention + DoD + commit chuẩn cho Cursor agent." |
| `CHANGELOG.md` (60 dòng) | Lịch sử version 0.1.0 | "Viết CHANGELOG.md format Keep a Changelog, liệt kê 6 commit." |
| `data/README.md` (75 dòng) | Hướng dẫn sinh data | "Viết data/README.md hướng dẫn chạy split.py, bảng 10 lớp, idempotency." |
| `README.md` (212 dòng) | 10 mục chuẩn | "Viết README.md đủ 10 mục theo template, có limitation + future." |
| `AI_Usage.md` (file này) | 7 mục chuẩn | "Viết AI_Usage.md theo template 7 mục, estimate % đóng góp." |

## 5. Prompt Purpose (cho mỗi artifact)

- **`src/data/split.py`** — prompt yêu cầu AI viết hàm split với ràng buộc:
  dùng `torchvision.datasets.FashionMNIST`, seed=42, ghi `.png` (không ghi
  tensor), idempotent (skip nếu đủ file), CLI argparse, docstring đầy đủ.
  Tác giả review: thêm check `val_split <= 60000` và logging.
- **`src/data/dataset.py`** — prompt yêu cầu AI viết DataLoader đọc config
  YAML (`configs/baseline.yaml`), 3 transform riêng từng tập, auto `pin_memory`
  nếu CUDA. Tác giả review: thêm fallback default nếu key không có trong YAML.
- **`tests/test_dataset.py`** — prompt yêu cầu AI viết 11 test cover:
  số ảnh mỗi tập, class folder tồn tại, batch shape, dtype float32/int64,
  pixel range, số batch, shuffle train, không shuffle val/test, idempotency.
  Tác giả review: thêm test config path tồn tại.
- **`docs/architecture.md`** — prompt yêu cầu AI điền file dựa trên code thực tế,
  KHÔNG bịa M2–M5 (đánh dấu ⏳), có sơ đồ ASCII tổng quan. Tác giả review: bổ
  sung mục "Quyết định kiến trúc" ở cuối.
- **`docs/data-flow.md`** — prompt yêu cầu AI vẽ sơ đồ 2 bước (split 1 lần,
  transform on-the-fly) + end-to-end M1→M5, có ví dụ 1 ảnh đi qua pipeline.
- **`docs/api.md`** — prompt yêu cầu AI liệt kê API public: input/output/side
  effect/raises/CLI cho mỗi hàm. Tác giả review: chuẩn hóa bảng error contract.
- **`AGENTS.md`** — prompt yêu cầu AI tổng hợp convention từ
  `.cursor/skills/ai-fusion-workflow/SKILL.md` thành file ngắn gọn cho agent
  đọc nhanh, có DoD 8 mục + Conventional Commit format.
- **`CHANGELOG.md`** — prompt yêu cầu AI liệt kê 6 commit + dataset v1.0 +
  configuration + limitations + reproducibility command.
- **`data/README.md`** — prompt yêu cầu AI viết hướng dẫn ngắn cho người mới,
  giải thích tại sao dùng `.png` thay vì tensor, tại sao gitignore.
- **`README.md`** — prompt yêu cầu AI viết đủ 10 mục theo template
  `templates/readme.md`, có limitation + future improvements, link tới docs khác.
- **`AI_Usage.md` (file này)** — prompt yêu cầu AI tự viết về chính mình theo
  template 7 mục, có estimate % đóng góp.

## 6. Student Modifications

- **`src/data/split.py`** — thêm check `if val_split < 0 or val_split > 60000:
  raise ValueError(...)` để bắt input lỗi sớm, thay vì để Python tự lỗi
  ambiguous. AI gốc chỉ có `random.shuffle` không validate.
- **`src/data/dataset.py`** — thêm fallback `cfg.get("data", {})` để nếu
  config thiếu key `data` thì dùng dict rỗng thay vì `KeyError`. AI gốc
  dùng `cfg["data"]` trực tiếp.
- **`tests/test_dataset.py`** — thêm test `test_config_exists` kiểm tra
  `configs/baseline.yaml` tồn tại, vì đây là input bắt buộc của
  `build_dataloaders`. AI gốc chỉ test 10 case liên quan data.
- **`configs/baseline.yaml`** — sửa `num_workers: 2` thành `num_workers: 0`
  trên Windows (AI đề xuất 2 theo Linux convention, nhưng Windows có thể
  bị hang nếu fork nhiều). Hiện để 2 vì máy tác giả chạy OK, nhưng ghi
  chú "set 0 nếu bị lỗi trên Windows".
- **`docs/architecture.md`** — thêm mục "Quyết định kiến trúc" ở cuối với
  6 quyết định + lý do, vì AI gốc chỉ liệt kê convention chung.
- **`docs/data-flow.md`** — thêm "Ví dụ cụ thể 1 ảnh" đi từ file `.png`
  đến logits, vì phần này quan trọng cho người mới hiểu pipeline.
- **`README.md`** — thay thế placeholder "TBD" trong Results bằng bảng với
  status thật (✅ M1 đã có, ⏳ M3–M5 chưa). AI gốc để placeholder trống.
- **Quyết định KHÔNG dùng AI cho** `progress_status.md` — vì đây là nơi ghi
  nhận quyết định của tác giả, để AI tự viết sẽ dẫn đến circular reference
  (AI viết về việc AI đã dùng ở đâu). Tác giả viết 95% file này.

## 7. Validation Method

Đã thực hiện trong project:

- [x] **Chạy code end-to-end trên môi trường sạch** — `pip install -r
  requirements.txt` → `python -m src.data.split` → `pytest -v` đều pass.
- [x] **So sánh output với baseline** — FashionMNIST gốc có 60.000 train +
  10.000 test, mình reproduce đúng (50k train + 10k val + 10k test).
- [x] **Viết unit/integration test** — `tests/test_dataset.py` 11 test
  cover happy path + edge case (idempotency, config thiếu).
- [x] **Reproduce kết quả từ fresh clone** — sau mỗi commit, đã xóa
  `data/raw/` rồi chạy lại split.py, kết quả giống hệt.
- [x] **Sanity-check metric theo DoD** — batch shape `[64, 1, 28, 28]`,
  dtype float32/int64, pixel range `[-0.81, 2.02]`, label range `[0, 9]`
  đều đúng như spec.

Sẽ thực hiện ở M3+:

- [ ] Test accuracy ≥ 88% (sau M4).
- [ ] Loss giảm đều qua 10 epoch (sau M3).
- [ ] Confusion matrix không có class nào < 70% (sau M4, optional).
- [ ] Save/load model roundtrip cho cùng accuracy (sau M4).

---

**Nguyên tắc cuối cùng:**

> AI là công cụ, người là chủ. Mọi code AI generate đều được tác giả đọc
> lại từng dòng, chạy thực tế trên máy, và commit bằng tay. Không có dòng
> code nào trong repo này được commit mà tác giả không hiểu.

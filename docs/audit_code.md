# Code Audit Report — FashionMNIST PyTorch Project

**Audit Date:** 2026-10-22
**Auditor:** Human Review (có đối chiếu bằng công cụ: `pytest`, `git ls-files`, `Select-String`)
**Project Phase:** Phase 2 — hoàn tất (data → model → training → evaluation)
**Scope:** Toàn bộ source Python đang được Git track (39 file, 4.074 dòng) + 6 file test

> **Ghi chú:** Bản audit trước (2026-10-11) viết ở giai đoạn khởi tạo, liệt kê nhiều module là "chưa
> triển khai". Các module đó nay đã tồn tại và chạy thật, nên bản này **viết lại toàn bộ** dựa trên
> trạng thái code hiện tại, không sửa lại bản cũ.

---

## 1. Audit Summary

| Hạng mục | Trạng thái | Ghi chú |
|----------|-----------|---------|
| Kiến trúc & tổ chức | ✅ Đạt | Phân tầng rõ `data → models → training → evaluation`, có `ModelFactory` làm điểm tiếp xúc duy nhất giữa các module |
| Chất lượng code | ✅ Đạt | Đọc dễ, có docstring, không còn TODO/FIXME trong source |
| Test | ⚠️ Mới đạt | 27 test pass, nhưng **chỉ phủ `src/training/`** — `data/`, `models/`, `evaluation/` chưa có test |
| Xử lý lỗi | ⚠️ Đạt một phần | Có xử lý lỗi có chủ đích, nhưng còn 3 chỗ bắt `except Exception` quá rộng |
| Type safety | ⚠️ Đạt một phần | 67/150 hàm có ghi chú kiểu trả về; kiểu tham số đầy đủ ở khoảng 40% |
| Logging | ⚠️ Đạt một phần | 9 module dùng `logging`, nhưng còn **28 lệnh `print()`** trong source |
| Tính tái lập | ✅ Đạt | Grid search tất định, seed cố định ở split/EDA, checkpoint kèm metadata |
| Bảo mật | ⚠️ Cần xem lại | `torch.load(weights_only=False)` trên checkpoint không tin cậy |
| Quản lý dependency | ⚠️ Cần xem lại | `requirements.txt` khai báo dependency theo dạng `>=` (không có file lock), chưa ghim phiên bản cụ thể. `loguru` được cài nhưng không module nào dùng (L-9) |
| Tài liệu | ⚠️ Cần xem lại | `docs/EXPERIMENTS.md` bị xoá khỏi ổ đĩa nhưng vẫn còn trong index và trong lời thoại trình bày |
| CI / Tự động hoá | ❌ Chưa có | Không có `.github/workflows`, không có `pre-commit`, không có `pyproject.toml` |

**Kết luận:** Code đạt chất lượng tốt và **sẵn sàng trình bày**. Các vấn đề còn lại đều là *độc lập*
với việc bảo vệ đề tài — không có lỗi nào chặn pipeline chạy. Khuyến nghị xử lý **M-1, M-2, H-1**
trước khi nộp.

---

## 2. Code Quality Metrics

Đo bằng công cụ trên trạng thái hiện tại:

| Chỉ số | Giá trị | Mục tiêu | Đánh giá |
|--------|---------|----------|----------|
| Số file Python (không tính test) | 33 | — | — |
| Tổng số dòng (không tính test) | ~3.326 | — | — |
| Tổng số dòng (tính cả test) | 4.074 | — | — |
| Hàm + class (không tính test) | 150 | — | — |
| Hàm có ghi chú kiểu trả về (`->`) | 67 / 150 = **45%** | 80% | ⚠️ |
| Module có docstring cấp module | 29 / 33 = **88%** | 100% | ⚠️ |
| Hàm/class có docstring hoặc comment | 90 / 150 = **60%** | 90% | ⚠️ |
| Test | 27 pass / 27 = **100% pass** | — | ✅ |
| Test coverage theo thư mục | `training/` ✅ · `data/` ❌ · `models/` ❌ · `evaluation/` ❌ | — | ⚠️ |
| `print()` còn lại trong source | 28 | 0 | ⚠️ |
| `TODO` / `FIXME` / `HACK` | **0** | 0 | ✅ |
| `except Exception` / `except:` | 3 | 0 | ⚠️ |
| Kiểm tra kiểu tĩnh (mypy/pylint) | Chưa chạy trong CI | — | ❌ |
| Format tự động (black/isort) | Chưa chạy trong CI | — | ❌ |

---

## 3. Kết quả theo từng module

### 3.1 `src/data/` — Chuẩn bị dữ liệu

| File | LOC | Đánh giá |
|------|-----|----------|
| `dataset.py` | 270 | ✅ Tách `load_fashion_mnist` + `CLASS_NAMES`, có `DataSplitManager` tích hợp. Kiểm tra `dataset[idx]` trả về đúng `(image, label)` shape. |
| `eda.py` | 347 | ✅ Sinh đủ 3 loại biểu đồ (sample / pixel histogram / per-class), tất cả tái sinh được bằng 1 lệnh. |
| `preprocessing.py` | 340 | ⚠️ Có `except Exception` tại dòng 111 — nuốt lỗi rồi log rồi tiếp tục, có thể che lỗi thật. |
| `split.py` | 411 | ✅ **Điểm mạnh.** Tách tầng (random → stratified), seed cố định, tự tính lại `val_ratio` cho khớp tổng. Đã verify bằng script: `50.000 / 10.000 / 10.000`. |

✅ **Split đã được kiểm chứng bằng số liệu thực tế** — không phải chỉ đọc code.

### 3.2 `src/models/` — Kiến trúc & chọn model

| File | LOC | Đánh giá |
|------|-----|----------|
| `model_factory.py` | 76 | ✅ Registry pattern, `create()` ném `ValueError` kèm danh sách model hợp lệ. `register()` kiểm tra subclass. Đây là điểm tiếp xúc duy nhất giữa Người 1 và Người 2. |
| `architectures/simple_cnn.py` | 61 | ✅ Docstring vẽ đầy đủ trace kích thước `(B,1,28,28) → (B,3136) → (B,10)`. Có `extra_repr()`. |
| `architectures/mlp.py` | 69 | ✅ |
| `architectures/advanced_cnn.py` | 117 | ✅ |
| `baseline_model.py` | 82 | ✅ |
| `model_selection.py` | 174 | ⚠️ `except Exception` tại dòng 175 — **có chủ đích**: một model fail không nên làm hỏng cả lượt so sánh. Đã có `# noqa: BLE001`. Đánh giá: chấp nhận được, nhưng nên log cả traceback. |

### 3.3 `src/training/` — Vòng huấn luyện

| File | LOC | Đánh giá |
|------|-----|----------|
| `trainer.py` | 137 | ✅ Có `train_epoch()` / `validate()`, loss được sample-weighted (đúng cách tính khi batch cuối nhỏ hơn). `_ensure_non_empty()` chặn loader rỗng. |
| `callbacks.py` | 105 | ✅ `ModelCheckpoint` dùng `weights_only=True` — ✅ đúng. |
| `hyperparameter_tuning.py` | 276 | ✅ Grid search tất định, có `hyperopt_scheduler.py` cho Bayesian search. |
| `visualizer.py` | 63 | ✅ Vẽ loss/accuracy curve. |
| `losses.py` | 10 | ✅ Nhỏ gọn, đúng. |
| `optimizers.py` | 32 | ✅ |

### 3.4 `src/evaluation/` — Đánh giá

| File | LOC | Đánh giá |
|------|-----|----------|
| `evaluator.py` | 57 | ✅ Trả về metrics + preds + labels + probs. |
| `model_io.py` | 68 | ⚠️ **`torch.load(..., weights_only=False)`** tại dòng 29 — xem H-1. |
| `visualizer.py` | 117 | ✅ *(vừa sửa)* Thêm guard `_show_if_interactive()` để không spam warning `FigureCanvasAgg is non-interactive` khi chạy headless. |
| `confusion_matrix.py` | 48 | ✅ *(vừa sửa)* Dùng chung helper `_show_if_interactive`. |
| `error_analysis.py` | 33 | ✅ |
| `run_evaluation.py` | 72 | ✅ *(đã đổi tên)* `run_evalution.py` → `run_evaluation.py`, cập nhật toàn bộ 10 tham chiếu. Nội dung file OK sau khi sửa lỗi encoding UTF-8. |
| ~~`experiment_tracker.py`~~ | — | ✅ *(đã xoá)* Placeholder rỗng, không được import ở đâu. Đề bài không yêu cầu experiment tracking; nhật ký thử nghiệm đã có ở `progress_status.md` + `docs/EXPERIMENTS.md`. |

---

## 4. Issues Found

### 🔴 Medium — nên xử lý trước khi nộp

| # | Issue | File | Mô tả | Đề xuất |
|---|-------|------|-------|---------|
| **M-1** | ✅ **Đã sửa** — Sai chính tả tên file | `src/evaluation/run_evaluation.py` | Đã `git mv` từ `run_evalution.py` → `run_evaluation.py` và cập nhật toàn bộ 10 tham chiếu (`README.md`, `PRESENTATION_CHEATSHEET.md`, `PROJECT_METHODOLOGY.md`, `progress_status.md`, `docs/reports/nguoi3_evaluation_report.md`). |
| **M-2** | ✅ **Đã sửa** — File bị xoá khỏi ổ đĩa nhưng còn trong Git index | `docs/EXPERIMENTS.md` | Đã `git checkout HEAD -- docs/EXPERIMENTS.md` khôi phục lại file. Nội dung log thử nghiệm được giữ và liệt kê trong `docs/README.md`. |
| **M-3** | ✅ **Đã sửa** — File rỗng | `docs/reports/REPORT_TEMPLATE.md` | Đã điền template đầy đủ 7 mục (phạm vi, cấu hình nhận, thành phần, kết quả, cách chạy, tự rà soát, bàn giao) theo đúng cấu trúc 3 báo cáo hiện có. |
| **M-4** | ✅ **Đã sửa** — Module placeholder chưa hoàn thiện | ~~`src/evaluation/experiment_tracker.py`~~ | Đề bài (`đề.txt`) không yêu cầu experiment tracking. Đã `git rm` file placeholder rỗng và gỡ mọi tham chiếu trong `README.md`, `progress_status.md`, `docs/PHASE1_REPORT.md`. Nhật ký thử nghiệm vẫn còn ở `progress_status.md` + `docs/EXPERIMENTS.md`. |
| **M-5** | ✅ **Đã sửa** — Sai casing tên file → link hỏng trên Linux/GitHub | `docs/architecture.md` | Đã đổi link `ARCHITECTURE.md` → `architecture.md` trong `docs/README.md`, đồng thời sửa trong `docs/QUICK_START.md` và `docs/PHASE1_REPORT.md`. |

### 🟡 Low — nên xử lý nếu còn thời gian

| # | Issue | File | Mô tả | Đề xuất |
|---|-------|------|-------|---------|
| **L-1** | `torch.load` không an toàn | `src/evaluation/model_io.py:29` | `weights_only=False` cho phép thực thi code tùy ý khi nạp checkpoint. Hiện tại chỉ nạp file do chính mình tạo nên **chưa thực sự nguy hiểm**, nhưng là mẫu không tốt. | Với checkpoint có metadata, `weights_only=True` không dùng được. Cân nhắc: thử `weights_only=True` trước, fallback về `False` kèm cảnh báo rõ ràng. |
| **L-2** | `print()` còn lại trong source | 28 chỗ | Đã có `logging` ở 9 module nhưng vẫn dùng `print` cho output người dùng. Trong `model_io.py`, `visualizer.py`, `confusion_matrix.py`... | Chấp nhận được nếu đây là output CLI có chủ đích. Nên thống nhất: `print` cho thông báo người dùng, `logger` cho thông tin chẩn đoán. |
| **L-3** | Module thiếu docstring cấp module | `src/evaluation/__init__.py`, `confusion_matrix.py`, `error_analysis.py`, `run_evalution.py` | 4/33 module. | Bổ sung 1 dòng mô tả. |
| **L-4** | `except Exception` quá rộng | `preprocessing.py:111`, `split.py:131`, `model_selection.py:175` | Xem đánh giá chi tiết ở §3. `split.py` và `model_selection.py` có chủ đích; `preprocessing.py` nên hẹp lại thành exception cụ thể. |
| **L-5** | Thiếu test cho 3 module lớn | `src/data/`, `src/models/`, `src/evaluation/` | Chỉ `src/training/` có test (6 file, 27 test). `split.py` là 411 dòng có logic phức tạp (stratified split) mà **chưa có test nào** — đây là khoảng trống đáng chú ý nhất. | Thêm `tests/data/test_split.py` (kiểm tra tỉ lệ, phân bố lớp, tính tất định với cùng seed) và `tests/models/test_model_factory.py`. |
| **L-6** | Còn `print()` trong `run_evalution.py` cần reconfig | `src/evaluation/run_evalution.py` | Đã sửa bằng `stream.reconfigure(encoding="utf-8", errors="replace")` — đây là fix đúng và cần giữ. Nên chuyển vào helper dùng chung nếu thêm script khác. | Tạo `src/utils/console.py` với hàm `setup_utf8_console()`. |
| **L-7** | Filename không phải ASCII | `đề.txt` (file gốc đề bài) | Git lưu dưới dạng escape octal `"d\303\244.txt"`. Mở trong terminal khác encoding có thể hiển thị lỗi. | Chấp nhận (đây là file đề gốc, nên giữ nguyên tên). Cần biết khi trình bày. |
| **L-8** | Checkpoint `.pt` lớn đang được track | `models/checkpoints/trained_model.pt` (4,84 MB), `best_model.pt` (1,61 MB) | `.gitignore` có `models/checkpoints/*` + `!trained_model.pt`, nhưng **`best_model.pt` đã được commit từ trước nên vẫn còn trong index** — `.gitignore` không có tác dụng với file đã track (`git check-ignore` trả exit 1, tức *không* bị ignore). Phải `git rm --cached` mới gỡ được. Tổng file track trong repo: **6,91 MB**. | `git rm --cached models/checkpoints/best_model.pt` nếu không cần giữ. Cân nhắc Git LFS nếu muốn giữ nhiều checkpoint. |
| **L-9** | `loguru` khai báo nhưng không dùng | `requirements.txt` | `loguru>=0.6.0` có trong requirements và **đã cài** (0.7.3), nhưng **không file `.py` nào import `loguru`** — toàn bộ project dùng `logging` của thư viện chuẩn. | Hoặc gỡ khỏi `requirements.txt` (giảm dependency thừa), hoặc chuyển sang dùng loguru cho nhất quán. |

### 🟢 Đã kiểm tra — không có vấn đề

| Hạng mục | Kết quả |
|----------|---------|
| Secret / credentials trong repo | ✅ Không tìm thấy |
| `.gitignore` có đủ thư mục sinh ra (`results/runs/`, `models/checkpoints/`) | ✅ Có, có `!` un-ignore cho file cần track |
| `torch.save`/`torch.load` tương thích | ✅ Dùng `state_dict`, không pickle model object |
| Chia sẻ model giữa 3 người qua `ModelFactory` | ✅ Rõ ràng, không coupling |
| Reproducibility | ✅ Seed cố định ở split/EDA; grid search tất định |
| Callback lưu checkpoint | ✅ Dùng `weights_only=True` — an toàn |
| Test suite | ✅ 27/27 pass, chạy 15,17 s |

---

## 5. Recommendations

**Ưu tiên cao (làm trước khi nộp, ~30 phút):**
1. ✅ ~~Sửa M-1~~ — Đã xong: `git mv` + cập nhật 10 tham chiếu.
2. ✅ ~~Xử lý M-2~~ — Đã xong: khôi phục `docs/EXPERIMENTS.md`.
3. ✅ ~~Sửa M-5~~ — Đã xong: link `architecture.md` lowercase trong `docs/README.md`, `QUICK_START.md`, `PHASE1_REPORT.md`.

**Ưu tiên trung bình:**
4. Viết `tests/data/test_split.py` — logic stratified split là phần dễ sai nhất trong dự án (L-5).
5. ✅ ~~M-3 + M-4~~ — Đã xong: điền `REPORT_TEMPLATE.md`, `git rm` `experiment_tracker.py`.

**Cải thiện dài hạn:**
6. Thêm `.github/workflows/ci.yml` chạy `pytest` + `black --check` + `mypy` trên mỗi PR.
7. Thêm `pre-commit` hook để format tự động trước khi commit.
8. Thêm `pyproject.toml` gom cấu hình `black`/`isort`/`mypy`/`pytest` vào một chỗ (hiện chưa có file nào).
9. Chuyển 28 `print()` sang `logging` cho phần logic, giữ `print` cho thông báo CLI.
10. Dọn dependency thừa: gỡ `loguru` khỏi `requirements.txt` hoặc chuyển sang dùng thật (L-9); `git rm --cached models/checkpoints/best_model.pt` để giảm repo (L-8).

### Đã sửa thêm trong đợt này (ngoài danh sách trên)

| Vấn đề | Sửa tại |
|--------|---------|
| Báo cáo grid search hiển thị `?` ở cột Batch / WD | `src/training/hyperparameter_tuning.py` — thêm `_effective_params()` ghi lại batch size, weight decay, dropout **có hiệu lực** của từng trial; `results/runs/hyperopt.md` regenerate lại |
| Lời thoại trình bày nói "em có **hai** callback" | Chỉ có **một** callback (`EarlyStopping`) vừa dừng sớm vừa lưu checkpoint — đã sửa trong `PRESENTATION_CHEATSHEET.md` PHẦN 5 |
| Lời thoại nói simple_cnn thắng deep_mlp "ở mọi tổ hợp" | Sai — thắng 2/4 (thua cả 2 tổ hợp SGD, chênh 0,05–0,66%). Đã sửa thành "ở cả hai tổ hợp dùng Adam, thắng 2,5–3 điểm" |
| Thiếu số dòng code trong lời thoại PHẦN 5 | Đã thêm file + số dòng cho từng chi tiết (`trainer.py`, `losses.py`, `callbacks.py`) |

---

## 6. Sign-off

| Hạng mục | Trạng thái |
|----------|-----------|
| Code đạt yêu cầu đề bài | ✅ |
| Pipeline chạy end-to-end trên CPU | ✅ Test accuracy 92,10% · F1 92,09% |
| Test suite pass | ✅ 27/27 |
| Tài liệu đầy đủ | ✅ M-2, M-3, M-5 đã xử lý |
| Sẵn sàng trình bày | ✅ |

- [x] Code đạt để trình bày
- [x] M-1, M-2, M-5 đã xử lý (khuyến nghị trước khi nộp)
- [x] Tài liệu đã cập nhật theo trạng thái code hiện tại

**Auditor:** Human Review · **Ngày:** 2026-10-7
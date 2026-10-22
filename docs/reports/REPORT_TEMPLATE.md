# Báo cáo Module — <Tên Module> (Người N)

> **Deliverable:** <Một câu tóm tắt: module này đáp ứng phần nào của đề bài>
> **Người phụ trách:** <Họ tên> (Người N)
> **Ngày chạy:** <YYYY-MM-DD>
> **Dataset:** FashionMNIST (50k train / 10k val / 10k test, 28×28 grayscale, 10 classes)
> **Device:** CPU (CUDA không khả dụng)
> **Nguồn:** `đề.txt`, `docs/PHASE2_PLAN.md`, báo cáo của các module trước đó.

---

## 1. Phạm vi và phân loại yêu cầu

| Nhãn | Phạm vi |
|------|---------|
| ĐỀ BÀI | <Các yêu cầu trực tiếp từ `đề.txt` mà module này gánh> |
| KẾ HOẠCH | <Các hạng mục trong `docs/PHASE2_PLAN.md` dành cho người này> |
| BỔ SUNG | <Phần mở rộng tự thêm> |

<Phạm vi của các module khác không nằm ở báo cáo này — nêu rõ để tránh trùng lặp.>

## 2. Cấu hình nhận từ module trước

| Thành phần | Giá trị |
|------------|---------|
| Model | `<tên model>` |
| Loss | `<loss>` |
| Optimizer | `<optimizer>` |
| Learning rate | `<lr>` |
| Batch size | `<batch>` |
| Weight decay | `<wd>` |
| Dropout | `<dropout>` |
| Epoch | `<n>` |
| Device | `<auto/cpu/cuda>` |

## 3. Thành phần đã hoàn thành

| File | Trách nhiệm | Trạng thái |
|------|-------------|:-----------:|
| `<path/to/file.py>` | <1 dòng> | ✅ |

> Ghi chú về các quyết định thiết kế đáng nói (vì sao chọn cách này, cách nào bị loại và vì sao).

## 4. Kết quả

### 4.1 Kết quả chính

| Chỉ số | Giá trị |
|--------|---------|
| <Accuracy> | <xx.xx%> |
| <F1 (macro)> | <xx.xx%> |

> Giải thích con số: nó so với cái gì, và kết luận rút ra là gì.

### 4.2 Artifact bàn giao

| File | Nội dung |
|------|----------|
| `<path>` | <mô tả> |

### 4.3 Test

| Test | Kết quả |
|------|:-------:|
| <tên test> | ✅ pass |

## 5. Cách chạy lại

```powershell
cd e:\lab1
$env:PYTHONPATH = "."
python <entry_point>.py
```

## 6. Tự rà soát & hạn chế

- <Điểm đã làm tốt>
- <Hạn chế / nợ kỹ thuật / chỗ có thể cải thiện>
- <Việc chưa làm và lý do>

## 7. Bàn giao cho module tiếp theo

- **Sản phẩm bàn giao:** <file / giá trị cụ thể>
- **Cách dùng lại:** <snippet hoặc câu mô tả>

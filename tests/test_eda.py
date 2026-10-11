"""Test EDA: file output tồn tại + nội dung hợp lệ + cảnh báo config sai."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _output_dir() -> Path:
    return REPO_ROOT / "results" / "eda"


@pytest.mark.skipif(
    not (_output_dir() / "summary.json").exists(),
    reason="EDA chưa chạy — chạy `python notebooks/eda.py` trước.",
)
def test_eda_outputs_exist():
    """Các file output bắt buộc phải tồn tại."""
    out = _output_dir()
    assert (out / "report.md").exists(), "thiếu report.md"
    assert (out / "summary.json").exists(), "thiếu summary.json"
    for fname in ("sample_grid.png", "class_distribution.png", "pixel_stats.png"):
        f = out / "figures" / fname
        assert f.exists(), f"thiếu figure {fname}"
        # 1 PNG tối thiểu ~ 5KB, nếu < 1KB thì coi như lỗi.
        assert f.stat().st_size > 1024, f"{fname} quá nhỏ ({f.stat().st_size} B)"


@pytest.mark.skipif(
    not (_output_dir() / "summary.json").exists(),
    reason="EDA chưa chạy",
)
def test_eda_summary_json_structure():
    """summary.json có đủ keys và 3 splits."""
    data = json.loads((_output_dir() / "summary.json").read_text(encoding="utf-8"))
    assert "splits" in data
    assert set(data["splits"].keys()) == {"train", "val", "test"}
    for name, s in data["splits"].items():
        assert s["num_samples"] > 0, f"{name} rỗng"
        assert tuple(s["shape_per_image"]) == (1, 28, 28), f"{name} shape lạ: {s['shape_per_image']}"
        assert s["dtype"] == "torch.float32", f"{name} dtype lạ: {s['dtype']}"
        # 10 class, count > 0 cho mỗi class.
        cc = s["class_counts"]
        assert set(cc.keys()) == {str(i) for i in range(10)}, f"{name} thiếu class: {cc.keys()}"
        assert min(cc.values()) > 0, f"{name} có class rỗng: {cc}"


@pytest.mark.skipif(
    not (_output_dir() / "summary.json").exists(),
    reason="EDA chưa chạy",
)
def test_eda_normalize_stats_match_config():
    """Sau Normalize, mean ≈ 0 và std ≈ 1 (sai lệch < 0.05) cho train split."""
    data = json.loads((_output_dir() / "summary.json").read_text(encoding="utf-8"))
    train = data["splits"]["train"]
    assert abs(train["pixel_mean"]) < 0.05, (
        f"train pixel_mean={train['pixel_mean']:.4f} lệch xa 0 — "
        "sai config mean/std hoặc transform không áp Normalize."
    )
    assert abs(train["pixel_std"] - 1.0) < 0.05, (
        f"train pixel_std={train['pixel_std']:.4f} lệch xa 1 — "
        "sai config mean/std hoặc transform không áp Normalize."
    )


@pytest.mark.skipif(
    not (_output_dir() / "summary.json").exists(),
    reason="EDA chưa chạy",
)
def test_eda_class_balance_in_train():
    """Train split phải cân bằng (FashionMNIST gốc cân bằng, không xáo trộn thì ratio ≤ 1.05)."""
    data = json.loads((_output_dir() / "summary.json").read_text(encoding="utf-8"))
    cc = data["splits"]["train"]["class_counts"]
    vals = list(cc.values())
    ratio = max(vals) / max(min(vals), 1)
    assert ratio < 1.1, f"train imbalance ratio {ratio:.2f} quá lớn — split có thể bị xáo trộn."
    assert sum(vals) == 50000, f"train phải có 50.000 ảnh, có {sum(vals)}"


@pytest.mark.skipif(
    not (_output_dir() / "summary.json").exists(),
    reason="EDA chưa chạy",
)
def test_eda_no_load_failures():
    """Không có NaN/Inf pixel — nếu có thì loader hoặc dataset có vấn đề."""
    data = json.loads((_output_dir() / "summary.json").read_text(encoding="utf-8"))
    for name, s in data["splits"].items():
        assert s["num_load_failures"] == 0, f"{name} có {s['num_load_failures']} NaN/Inf pixel"

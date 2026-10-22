"""Sinh báo cáo metrics đánh giá ra file để đối chiếu được.

Các con số trước đây chỉ tồn tại trong checkpoint nhị phân
(``models/checkpoints/final_model.pt``) hoặc trong log console — không có
file nào để mở ra chỉ vào khi thuyết trình. Module này ghi ra
``results/runs/test_metrics.md`` (đọc được) và ``results/runs/test_metrics.json``
(để máy đọc).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

# Tên metric trong checkpoint (evaluator.py) -> tên hiển thị.
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision_macro": "Precision (macro)",
    "recall_macro": "Recall (macro)",
    "f1_macro": "F1-Score (macro)",
}


def build_metrics_markdown(
    metrics: Mapping[str, Any],
    confusions: Sequence[tuple[tuple[str, str], int]] = (),
    top_k: int = 5,
) -> str:
    """Dựng bảng markdown cho metrics + các cặp lớp bị nhầm nhiều nhất."""
    lines = [
        "# Test Set Evaluation — SimpleCNN",
        "",
        "Nguồn: `python src/evaluation/run_evaluation.py` "
        "(checkpoint `models/checkpoints/trained_model.pt`)",
        "",
        "## 1. Metrics trên tập test (10.000 ảnh)",
        "",
        "| Chỉ số | Giá trị |",
        "|--------|---------|",
    ]
    for key, label in METRIC_LABELS.items():
        if key not in metrics:
            continue
        value = float(metrics[key])
        bold = "**" if key == "accuracy" else ""
        lines.append(f"| {bold}{label}{bold} | {bold}{value:.2f}%{bold} |")

    if confusions:
        lines += [
            "",
            f"## 2. Top {top_k} cặp lớp bị nhầm nhiều nhất",
            "",
            "| # | Lớp thực | Bị nhầm thành | Số lần |",
            "|---|-----------|--------------|---------|",
        ]
        for i, (pair, count) in enumerate(confusions[:top_k], 1):
            true_name, pred_name = pair
            lines.append(f"| {i} | {true_name} | {pred_name} | {count} |")

    lines.append("")
    return "\n".join(lines)


def save_metrics_report(
    metrics: Mapping[str, Any],
    confusions: Sequence[tuple[tuple[str, str], int]] = (),
    output_dir: str | Path = "results/runs",
    top_k: int = 5,
) -> Path:
    """Ghi ``test_metrics.md`` + ``test_metrics.json``, trả về đường dẫn .md."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / "test_metrics.md"
    md_path.write_text(
        build_metrics_markdown(metrics, confusions, top_k=top_k), encoding="utf-8"
    )

    json_path = out_dir / "test_metrics.json"
    json_path.write_text(
        json.dumps(
            {
                "metrics": {k: float(v) for k, v in metrics.items()},
                "top_confusions": [
                    {"true": p[0], "predicted": p[1], "count": int(c)}
                    for p, c in confusions[:top_k]
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return md_path


__all__ = ["METRIC_LABELS", "build_metrics_markdown", "save_metrics_report"]

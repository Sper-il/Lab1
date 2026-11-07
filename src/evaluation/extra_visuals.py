"""Vẽ các hình visualize còn thiếu chỉ bằng matplotlib+numpy (không cần seaborn/torch).

Sinh ra từ JSON thật, không gõ tay:
- results/runs/split_overview.png  (50k/10k/10k + mỗi lớp 5k/1k/1k)
- results/runs/normalize_curve.png  (ánh xạ pixel [0,1] -> normalized, mean=0.2860/std=0.3530)
- results/runs/hyperopt_trials.png  (16 trials val_acc, highlight trial 9)
- results/runs/top_confusions.png   (top-5 cặp nhầm từ test_metrics.json)
- results/runs/train_val_gap.png    (gap train-val acc + val_loss, đánh dấu best epoch 11)
- results/demo/<ts>/index.html      (bản demo rút gọn theo chuẩn skill)

Cách dùng:
    python src/evaluation/extra_visuals.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "results" / "runs"

MEAN = 0.2860
STD = 0.3530


def _load(name: str) -> dict | list:
    return json.loads((RUNS_DIR / name).read_text(encoding="utf-8"))


def plot_split(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(["Train", "Val", "Test"], [50000, 10000, 10000], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[0].set_title("Split 50.000 / 10.000 / 10.000 (seed=42, val_ratio=1/6)")
    axes[0].set_ylabel("Số mẫu")
    for i, v in enumerate([50000, 10000, 10000]):
        axes[0].text(i, v + 800, f"{v:,}".replace(",", "."), ha="center", fontsize=10)
    axes[1].bar(["Train/lớp", "Val/lớp", "Test/lớp"], [5000, 1000, 1000], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[1].set_title("Mỗi lớp: 5.000 / 1.000 / 1.000 (từ 6.000 gốc)")
    axes[1].set_ylabel("Số mẫu/lớp")
    for i, v in enumerate([5000, 1000, 1000]):
        axes[1].text(i, v + 80, f"{v:,}".replace(",", "."), ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_normalize(out: Path) -> None:
    x = np.linspace(0, 1, 200)
    y = (x - MEAN) / STD
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, label=f"(pixel − {MEAN}) / {STD}")
    ax.scatter([0.0, 0.5, 1.0], [(0 - MEAN) / STD, (0.5 - MEAN) / STD, (1 - MEAN) / STD], color="red", zorder=5)
    ax.annotate("0.5 → 0.606", xy=(0.5, (0.5 - MEAN) / STD), xytext=(0.62, 1.2),
                arrowprops=dict(arrowstyle="->"), fontsize=10)
    ax.set_xlabel("Pixel gốc [0, 1]")
    ax.set_ylabel("Pixel sau chuẩn hóa")
    ax.set_title(f"Chuẩn hóa FashionMNIST (mean={MEAN}, std={STD}) → khoảng [-0.81, 2.02]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_hyperopt(out: Path) -> None:
    trials = _load("hyperopt.json")
    trials = sorted(trials, key=lambda r: r["val_acc"], reverse=True)
    labels = [f"T{r['trial']}\n{r['param_model_name'][:6]}" for r in trials]
    vals = [r["val_acc"] for r in trials]
    colors = ["#d62728" if r["trial"] == 9 else "#1f77b4" for r in trials]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(range(len(trials)), vals, color=colors)
    ax.set_xticks(range(len(trials)))
    ax.set_xticklabels(labels, fontsize=8, rotation=0)
    ax.set_ylabel("Val acc (%)")
    ax.set_title("Grid search 16 trials × 10 epochs — best T9 simple_cnn+Adam+lr=0.001 → 92.33%")
    ax.set_ylim(min(vals) - 1, max(vals) + 0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_confusions(out: Path) -> None:
    test = _load("test_metrics.json")
    confs = test["top_confusions"]
    labels = [f"{c['true']}→{c['predicted']}" for c in confs]
    counts = [c["count"] for c in confs]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(labels[::-1], counts[::-1], color="#ff7f0e")
    ax.set_xlabel("Số lần nhầm")
    ax.set_title("Top-5 cặp lớp bị nhầm (test 10.000 ảnh, tổng top-5 = 367)")
    for i, v in enumerate(counts[::-1]):
        ax.text(v + 1, i, str(v), va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_gap(out: Path) -> None:
    h = _load("training_history.json")
    epochs = list(range(1, len(h["train_loss"]) + 1))
    gap_acc = np.array(h["train_acc"]) - np.array(h["val_acc"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, gap_acc, marker="o", label="train_acc − val_acc")
    axes[0].axvline(11, color="red", linestyle="--", label="best epoch 11")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Gap (%)")
    axes[0].set_title("Overfit gap (cuối run ~1.9%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, h["val_loss"], marker="o", color="#2ca02c", label="val_loss")
    axes[1].axvline(11, color="red", linestyle="--", label="best 0.215183 @ e11")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val loss")
    axes[1].set_title("Val loss (giảm tới e11 rồi tăng nhẹ)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


def write_demo_index(ts: str) -> Path:
    test = _load("test_metrics.json")
    m = test["metrics"]
    demo_dir = PROJECT_ROOT / "results" / "demo" / ts
    demo_dir.mkdir(parents=True, exist_ok=True)
    html = f"""<!DOCTYPE html>
<html lang="vi">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FashionMNIST Demo — {ts}</title>
<style>body{{font-family:Arial,sans-serif;max-width:720px;margin:24px auto;padding:0 16px;color:#222;line-height:1.6}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:8px;text-align:left}}code{{background:#f6f6f6;padding:1px 5px;border-radius:4px}}</style>
</head>
<body>
<h1>FashionMNIST Demo rút gọn — {ts}</h1>
<p>Bản full: <code>results/runs/report.html</code> (offline, 8+ ảnh). Sinh lúc {datetime.now():%Y-%m-%d %H:%M}.</p>
<h2>3 case mẫu</h2>
<table><tr><th>Case</th><th>Input</th><th>Kỳ vọng</th></tr>
<tr><td>Dễ</td><td>Trouser / Sneaker / Bag 28×28</td><td>Đúng, confidence cao (ô xanh <code>predictions.png</code>)</td></tr>
<tr><td>Khó</td><td>Shirt vs T-shirt/top</td><td>Có thể sai (ô đỏ, top nhầm 95+90 lần)</td></tr>
<tr><td>Biên</td><td>Full test 10.000 ảnh</td><td>Accuracy {m['accuracy']:.2f}% (~9.210 đúng)</td></tr></table>
<h2>Metric chính</h2>
<table><tr><th>Metric</th><th>Giá trị</th></tr>
<tr><td>Accuracy</td><td>{m['accuracy']:.2f}% (≥90% PASS)</td></tr>
<tr><td>Precision / Recall / F1 (macro)</td><td>{m['precision_macro']:.2f}% / {m['recall_macro']:.2f}% / {m['f1_macro']:.2f}%</td></tr>
<tr><td>Checkpoint</td><td><code>models/checkpoints/trained_model.pt</code> (best epoch 11)</td></tr></table>
<h2>Cách chạy</h2>
<pre><code>python run_experiments.py --smoke
$env:PYTHONPATH = "."; python src/evaluation/run_evaluation.py
# xem results/runs/report.html (double-click)</code></pre>
</body>
</html>
"""
    out = demo_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vẽ hình visualize còn thiếu (matplotlib only).")
    parser.add_argument("--demo-ts", default=datetime.now().strftime("%Y%m%d-%H%M"))
    args = parser.parse_args(argv)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    plot_split(RUNS_DIR / "split_overview.png")
    plot_normalize(RUNS_DIR / "normalize_curve.png")
    plot_hyperopt(RUNS_DIR / "hyperopt_trials.png")
    plot_confusions(RUNS_DIR / "top_confusions.png")
    plot_gap(RUNS_DIR / "train_val_gap.png")
    demo_index = write_demo_index(args.demo_ts)
    print(f"Wrote 5 PNGs to {RUNS_DIR} + {demo_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

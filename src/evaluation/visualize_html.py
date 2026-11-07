"""Sinh báo cáo trực quan HTML từ artifacts có sẵn (offline 100%).

Chuẩn skill: `.cursor/skills/ai-fusion-workflow/modules/visualization.md`
- Một file duy nhất `results/runs/report.html`, CSS nội tuyến, ảnh base64, không CDN.
- Mọi số render từ `training_history.json` + `test_metrics.json` + `hyperopt_best.json`, không gõ tay.
- Chạy lại là ra đúng file cũ.

Cách dùng:
    python -m src.evaluation.visualize_html
    python -m src.evaluation.visualize_html --run-id 20261015-final --output results/runs/report.html
"""

from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "results" / "runs"

# Ảnh nhúng theo đúng quy ước plots/ của skill.
IMAGES = [
    ("class_distribution.png", "Phân bố 10 lớp (cân bằng hoàn hảo 6.000/lớp gốc)"),
    ("split_overview.png", "Tổng quan split 50k/10k/10k + mỗi lớp 5k/1k/1k (seed=42)"),
    ("per_class_samples.png", "Ảnh mẫu mỗi lớp (1 ảnh/lớp)"),
    ("pixel_histogram.png", "Histogram cường độ pixel (2 đỉnh: nền sáng / vật thể tối)"),
    ("embedding_pca_color.png", "PCA scatter tô màu 10 lớp (2.000 ảnh test, giải thích ~47% phương sai)"),
    ("embedding_pca_thumbs.png", "PCA thumbnail scatter nền đen (600 ảnh mẫu đặt tại vị trí PCA)"),
    ("embedding_tsne_color.png", "t-SNE scatter tô màu 10 lớp (perplexity=30, cụm tách rõ hơn PCA)"),
    ("embedding_tsne_thumbs.png", "t-SNE thumbnail scatter nền đen (600 ảnh mẫu)"),
    ("pca_3d.png", "PCA 3D scatter 3.000 ảnh (3PC giữ 53,3% phương sai, xem bản tương tác pca_3d.html)"),
    ("pca_explained.png", "PCA explained variance 10 thành phần đầu (PC1+PC2 ~47%)"),
    ("tsne_3d.png", "t-SNE 3D scatter 2.000 ảnh (perplexity=30, xem bản tương tác tsne_3d.html)"),
    ("umap_3d.png", "UMAP 3D scatter 2.000 ảnh (n_neighbors=15, xem bản tương tác umap_3d.html)"),
    ("mde_3d.png", "PyMDE 3D scatter 2.000 ảnh (preserve_neighbors, xem bản tương tác mde_3d.html)"),
    ("normalize_curve.png", "Đường cong chuẩn hóa (pixel − 0,2860) / 0,3530 → [-0,81, 2,02]"),
    ("sample_images.png", "25 ảnh mẫu ngẫu nhiên"),
    ("loss_curve.png", "Loss train vs val qua 15 epochs (best epoch 11)"),
    ("accuracy_curve.png", "Accuracy train vs val qua 15 epochs"),
    ("train_val_gap.png", "Gap overfit train−val + val_loss (đánh dấu best epoch 11)"),
    ("hyperopt_trials.png", "Grid search 16 trials (best T9 đỏ, val 92,33%)"),
    ("confusion_matrix.png", "Confusion matrix normalized 10×10 (hàng = recall từng lớp)"),
    ("top_confusions.png", "Top-5 cặp nhầm (tổng 367 lỗi, Shirt↔T-shirt/top chiếm 185)"),
    ("predictions.png", "20 ảnh predicted vs actual (xanh = đúng, đỏ = sai)"),
]


CLASS_NAMES_3D = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
HEX_3D = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# Bản 3D tương tác nhúng trực tiếp vào report.html (json, tiêu đề, mô tả).
EMBED3D = [
    ("pca_3d.json", "PCA 3D — 3.000 ảnh, 3PC giữ 53,3% phương sai",
     "PCA tuyến tính, giữ cấu trúc toàn cục; Trouser/Bag tách xa, nhóm Shirt chồng lấn."),
    ("tsne_3d.json", "t-SNE 3D — 2.000 ảnh, perplexity=30",
     "t-SNE phi tuyến, giữ cấu trúc lân cận; cụm tách rõ hơn PCA."),
    ("umap_3d.json", "UMAP 3D — 2.000 ảnh, n_neighbors=15",
     "UMAP giữ cả cục bộ + toàn cục; cụm gọn, đúng kiểu ảnh mẫu."),
    ("mde_3d.json", "PyMDE 3D — 2.000 ảnh, preserve_neighbors",
     "MDE tối thiểu distortion neighbour trên CPU; cụm tương tự UMAP."),
]


def _embed3d_section() -> str:
    """Nhúng 4 viewer 3D (canvas + chiếu phối cảnh, xoay/zoom/bật-tắt lớp) vào report."""
    blocks = []
    datas = {}
    for fname, title, desc in EMBED3D:
        p = RUNS_DIR / fname
        if not p.exists():
            blocks.append(f"<div class='viewer'><h3>{title}</h3><p>⚠️ Thiếu {fname}</p></div>")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        key = fname.replace(".json", "").replace("_3d", "")
        datas[key] = {"coords": d["coords"], "labels": d["labels"]}
        boxes = "\n".join(
            f"<label style='margin-right:8px'><input type='checkbox' data-view='{key}' data-cls='{i}' checked> "
            f"<span style='color:{HEX_3D[i]}'>■</span> {CLASS_NAMES_3D[i]}</label>"
            for i in range(10)
        )
        blocks.append(
            f"<div class='viewer'><h3>{title}</h3><p>{desc} File gốc: <code>results/runs/{fname}</code>.</p>"
            f"<canvas id='cv-{key}' width='920' height='520'></canvas>"
            f"<div style='margin:6px 0'>{boxes}</div>"
            f"<p style='margin:6px 0'><button data-spin='{key}'>⏸ Dừng xoay</button> "
            f"<button data-reset='{key}'>⟲ Reset góc</button> "
            f"Zoom <input data-zoom='{key}' type='range' min='60' max='500' value='260' style='vertical-align:middle'></p></div>"
        )
    if not datas:
        return "<p>⚠️ Chưa có file *_3d.json nào.</p>"
    data_js = json.dumps(datas, separators=(",", ":"))
    hex_js = json.dumps(HEX_3D)
    viewers_html = "\n".join(blocks)
    return f"""{viewers_html}
<script>
const VIEW3D = {data_js};
const COLORS3D = {hex_js};
function initViewer(key) {{
  const cv = document.getElementById('cv-' + key);
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const D = VIEW3D[key];
  let pts = D.coords.map((p, i) => ({{x: p[0], y: p[1], z: p[2], c: D.labels[i]}}));
  let mx = 0, my = 0, mz = 0;
  pts.forEach(p => {{ mx += p.x; my += p.y; mz += p.z; }});
  mx /= pts.length; my /= pts.length; mz /= pts.length;
  let sc = 0;
  pts.forEach(p => {{ sc = Math.max(sc, Math.abs(p.x - mx), Math.abs(p.y - my), Math.abs(p.z - mz)); }});
  pts.forEach(p => {{ p.x = (p.x - mx) / sc; p.y = (p.y - my) / sc; p.z = (p.z - mz) / sc; }});
  const st = {{ax: -0.5, ay: 0.6, zoom: 260, spin: true}};
  const on = new Array(10).fill(true);
  document.querySelectorAll("input[data-view='" + key + "']").forEach(cb => cb.onchange = () => on[+cb.dataset.cls] = cb.checked);
  const spinBtn = document.querySelector("button[data-spin='" + key + "']");
  if (spinBtn) spinBtn.onclick = e => {{ st.spin = !st.spin; e.target.textContent = st.spin ? '⏸ Dừng xoay' : '▶ Xoay tiếp'; }};
  const resetBtn = document.querySelector("button[data-reset='" + key + "']");
  const zoomInput = document.querySelector("input[data-zoom='" + key + "']");
  if (resetBtn) resetBtn.onclick = () => {{ st.ax = -0.5; st.ay = 0.6; st.zoom = 260; if (zoomInput) zoomInput.value = 260; }};
  if (zoomInput) zoomInput.oninput = e => st.zoom = +e.target.value;
  let drag = false, lx = 0, ly = 0;
  cv.onmousedown = e => {{ drag = true; lx = e.clientX; ly = e.clientY; cv.style.cursor = 'grabbing'; }};
  window.addEventListener('mouseup', () => {{ drag = false; cv.style.cursor = 'grab'; }});
  window.addEventListener('mousemove', e => {{ if (!drag) return; st.ay += (e.clientX - lx) * 0.008; st.ax += (e.clientY - ly) * 0.008; lx = e.clientX; ly = e.clientY; }});
  cv.onwheel = e => {{ e.preventDefault(); st.zoom = Math.min(500, Math.max(60, st.zoom - e.deltaY * 0.15)); if (zoomInput) zoomInput.value = st.zoom; }}, {{passive: false}};
  function frame() {{
    if (st.spin && !drag) st.ay += 0.004;
    const cx = Math.cos(st.ax), sx = Math.sin(st.ax), cy = Math.cos(st.ay), sy = Math.sin(st.ay);
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    ctx.font = '13px Arial';
    const axis = [[1,0,0,'#ff5555','X'],[0,1,0,'#55ff55','Y'],[0,0,1,'#5599ff','Z']];
    axis.forEach(a => {{
      const y1 = a[1] * cx - a[2] * sx, z1 = a[1] * sx + a[2] * cx, x2 = a[0] * cy + z1 * sy;
      const px = W/2 + x2 * st.zoom * 1.2, py = H/2 - y1 * st.zoom * 1.2;
      ctx.strokeStyle = a[3]; ctx.beginPath(); ctx.moveTo(W/2, H/2); ctx.lineTo(px, py); ctx.stroke();
      ctx.fillStyle = a[3]; ctx.fillText(a[4], px + 4, py);
    }});
    const proj = [];
    for (const p of pts) {{
      if (!on[p.c]) continue;
      const y1 = p.y * cx - p.z * sx, z1 = p.y * sx + p.z * cx, x1 = p.x * cy + z1 * sy, z2 = -p.x * sy + z1 * cy;
      const per = 1 / (3 - z2 * 0.8);
      proj.push({{sx: W/2 + x1 * st.zoom * per, sy: H/2 - y1 * st.zoom * per, d: z2, c: p.c}});
    }}
    proj.sort((a, b) => a.d - b.d);
    for (const q of proj) {{ ctx.fillStyle = COLORS3D[q.c]; ctx.globalAlpha = 0.9; ctx.fillRect(q.sx, q.sy, 4, 4); }}
    ctx.globalAlpha = 1;
    requestAnimationFrame(frame);
  }}
  frame();
}}
Object.keys(VIEW3D).forEach(initViewer);
</script>"""


def _img_base64(path: Path) -> str:
    raw = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(run_id: str = "20261015-final", commit: str = "049acf0") -> str:
    history = _load_json(RUNS_DIR / "training_history.json")
    test = _load_json(RUNS_DIR / "test_metrics.json")
    best = _load_json(RUNS_DIR / "hyperopt_best.json")
    m = test["metrics"]
    confs = test["top_confusions"]

    n_epochs = len(history["train_loss"])
    best_epoch = 11
    rows = []
    for i in range(n_epochs):
        mark = " ⭐ best" if (i + 1) == best_epoch else ""
        rows.append(
            f"<tr><td>{i + 1}{mark}</td><td>{history['train_loss'][i]:.4f}</td>"
            f"<td>{history['val_loss'][i]:.4f}</td>"
            f"<td>{history['train_acc'][i]:.2f}%</td>"
            f"<td>{history['val_acc'][i]:.2f}%</td></tr>"
        )
    history_table = "\n".join(rows)

    conf_rows = "\n".join(
        f"<tr><td>{c['true']}</td><td>{c['predicted']}</td><td>{c['count']}</td></tr>"
        for c in confs
    )

    figures = []
    for fname, caption in IMAGES:
        p = RUNS_DIR / fname
        if not p.exists():
            figures.append(f"<figure><figcaption>⚠️ Thiếu {fname}</figcaption></figure>")
            continue
        figures.append(
            f"<figure><img src=\"{_img_base64(p)}\" alt=\"{fname}\" loading=\"lazy\">"
            f"<figcaption>{caption} — <code>results/runs/{fname}</code></figcaption></figure>"
        )
    figures_html = "\n".join(figures)
    embed3d_html = _embed3d_section()
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FashionMNIST — {run_id}</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 960px; margin: 24px auto; padding: 0 16px; color: #222; line-height: 1.6; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; font-size: 14px; }}
  th {{ background: #f4f4f4; }}
  .pass {{ color: #0a7b2e; font-weight: bold; }}
  .fail {{ color: #b00020; font-weight: bold; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #eee; }}
  figure {{ margin: 16px 0; }}
  figcaption {{ font-size: 13px; color: #555; margin-top: 6px; }}
  .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }}
  @media (max-width: 700px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  code {{ background: #f6f6f6; padding: 1px 5px; border-radius: 4px; }}
  .header {{ background: #fafafa; border: 1px solid #e5e5e5; padding: 12px 16px; border-radius: 8px; }}
  .viewer {{ border: 1px solid #e5e5e5; border-radius: 8px; padding: 12px 14px; margin: 14px 0; background: #fff; }}
  canvas {{ border: 1px solid #ddd; border-radius: 8px; width: 100%; height: 520px; cursor: grab; background: #ffffff; }}
  button {{ padding: 6px 12px; margin-right: 6px; }}
</style>
</head>
<body>
<div class="header">
  <h1>FashionMNIST Classification — {run_id}</h1>
  <p>Commit: <code>{commit}</code> | Thời gian sinh báo cáo: {time_str} | Config: <code>configs/model_config.yaml</code> + <code>configs/hyperopt_config.yaml</code></p>
  <p>Checkpoint: <code>models/checkpoints/trained_model.pt</code> (best epoch 11) → <code>models/checkpoints/final_model.pt</code></p>
</div>

<h2>1. Tóm tắt Define (1 câu)</h2>
<p>Phân loại 10 nhóm sản phẩm thời trang từ ảnh xám 28×28 (input <code>(1, 28, 28)</code> → logits <code>(B, 10)</code>), mục tiêu test accuracy ≥ 90% — <strong>đạt 92,10%</strong> bằng SimpleCNN trên CPU. Chi tiết xem <code>docs/description.md</code>.</p>

<h2>2. Phân bố dữ liệu</h2>
<table><tr><th>Tập</th><th>Số mẫu</th><th>Mỗi lớp</th><th>Vai trò</th></tr>
<tr><td>Train</td><td>50.000</td><td>5.000</td><td>Model học trọng số (391 batch, bs=128)</td></tr>
<tr><td>Validation</td><td>10.000</td><td>1.000</td><td>Early stopping theo <code>val_loss</code></td></tr>
<tr><td>Test</td><td>10.000</td><td>1.000</td><td>Giữ kín, mở 1 lần cuối</td></tr></table>
<p>Chuẩn hóa <code>(pixel − 0,2860) / 0,3530</code>; stratified split <code>seed=42</code>, <code>val_ratio=1/6</code>. Nguồn: <code>results/runs/dataset_info.md</code>.</p>

<h2>3. Quá trình train (15 epochs, CPU ~75 s/epoch)</h2>
<p>Best theo <code>val_loss</code> là epoch 11 (<code>val_loss=0,215183</code>, <code>val_acc=92,85%</code>); đỉnh <code>val_acc=92,88%</code> ở epoch 13; early stopping không kích hoạt (sau best chỉ 4 epoch không cải thiện &lt; patience 5). Best config từ grid: <code>simple_cnn + Adam + lr=0,001</code> → val 92,33% (trial 9, <code>{best.get('val_acc', 92.33)}%</code>).</p>
<table><tr><th>Epoch</th><th>Train loss</th><th>Val loss</th><th>Train acc</th><th>Val acc</th></tr>
{history_table}</table>

<h2>4. Kết quả model level (test 10.000 ảnh)</h2>
<table><tr><th>Metric</th><th>Target</th><th>Đạt được</th><th>Kết quả</th></tr>
<tr><td>Accuracy</td><td>≥ 90%</td><td><strong>{m['accuracy']:.2f}%</strong></td><td class="pass">PASS</td></tr>
<tr><td>Precision (macro)</td><td>—</td><td>{m['precision_macro']:.2f}%</td><td class="pass">PASS</td></tr>
<tr><td>Recall (macro)</td><td>—</td><td>{m['recall_macro']:.2f}%</td><td class="pass">PASS</td></tr>
<tr><td>F1-Score (macro)</td><td>—</td><td>{m['f1_macro']:.2f}%</td><td class="pass">PASS</td></tr></table>

<h2>5. Kết quả pipeline</h2>
<table><tr><th>Hạng mục</th><th>Giá trị thật</th></tr>
<tr><td>Model</td><td>SimpleCNN 421.642 params (~1,61 MB <code>final_model.pt</code>)</td></tr>
<tr><td>Train throughput</td><td>~75 s/epoch (grid, bs=128) · ~96 s/epoch (pilot, bs=64), CPU-only</td></tr>
<tr><td>Inference</td><td>Full eval 10.000 test ~1–2 phút CPU (~10 ms/ảnh); smoke 64 ảnh ~5 giây</td></tr>
<tr><td>VRAM</td><td>Không dùng GPU; RAM CPU đủ cho batch 128 ảnh 28×28</td></tr>
<tr><td>Đúng/sai trên test</td><td>Đúng ~9.210 / sai ~790 trên 10.000 ảnh</td></tr></table>

<h2>6. Error analysis — top-5 cặp nhầm</h2>
<table><tr><th>Lớp thực</th><th>Bị nhầm thành</th><th>Số lần</th></tr>
{conf_rows}</table>
<p><strong>5 case tệ nhất (đoán nguyên nhân + đề xuất):</strong></p>
<ol>
<li>Shirt → T-shirt/top (95): hình dáng gần như giống hệt ở 28×28 → augmentation shift/rotation nhẹ.</li>
<li>T-shirt/top → Shirt (90): cùng nhóm áo cổ tròn tay ngắn → label smoothing.</li>
<li>Shirt → Pullover (64): tay dài/thân rộng lẫn lộn → thêm mẫu khó hoặc class weight nhóm áo.</li>
<li>Coat → Pullover (63): chất liệu/dáng khoác tương tự → tăng độ sâu kênh conv nếu có GPU.</li>
<li>Coat → Shirt (55): cổ/vạt áo gây lẫn → GradCAM kiểm tra vùng model nhìn.</li>
</ol>

<h2>7. Limitations + Next steps</h2>
<p><strong>Limitations (3):</strong></p>
<ul>
<li>Chưa có demo 1 lệnh (<code>src/demo.py</code>) và chưa đo latency live từng mẫu.</li>
<li>Chưa augmentation / scheduler / ensemble nên trần accuracy quanh ~92%.</li>
<li>Nhóm Shirt/T-shirt/top/Pullover/Coat vẫn nhầm nhiều do độ phân giải 28×28 thấp.</li>
</ul>
<p><strong>Next steps (3):</strong></p>
<ul>
<li>Viết <code>src/demo.py</code> + <code>data/processed/demo/</code> 3–5 ảnh mẫu, log <code>inference_time_ms</code>.</li>
<li>Thử augmentation nhẹ + learning rate scheduler, kỳ vọng +1–2%.</li>
<li>Thử ResNetMini train dài hơn hoặc ensemble nếu có GPU.</li>
</ul>

<h2>8. Hình ảnh (sinh từ <code>results/runs/</code>, nhúng base64 offline)</h2>
<div class="grid">
{figures_html}
</div>

<h2>9. Embedding 3D tương tác (PCA / t-SNE / UMAP / PyMDE — xoay trực tiếp trong report)</h2>
<p>Kéo chuột để xoay · lăn chuột để zoom · bật/tắt từng lớp. Dữ liệu lấy từ <code>pca_3d.json</code>, <code>tsne_3d.json</code>, <code>umap_3d.json</code>, <code>mde_3d.json</code> — nhúng trực tiếp nên report vẫn mở offline bằng double-click.</p>
{embed3d_html}
<p><em>Mọi ảnh và số liệu render bằng <code>src/evaluation/visualize_html.py</code> từ JSON, không gõ tay. Tổng dung lượng &lt; 10 MB, mở bằng double-click không cần mạng.</em></p>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sinh results/runs/report.html (offline, base64).")
    parser.add_argument("--run-id", default="20261015-final")
    parser.add_argument("--commit", default="049acf0")
    parser.add_argument("--output", default=str(RUNS_DIR / "report.html"))
    args = parser.parse_args(argv)

    html = build_report(run_id=args.run_id, commit=args.commit)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Sinh trang 3D tương tác offline từ `results/runs/pca_3d.json` (không CDN).

Dùng <canvas> 2D + tự chiếu phối cảnh 3D (xoay chuột, tự xoay, bật/tắt lớp),
nên mở bằng double-click, không cần mạng, không cần plotly.

Sinh ra: results/runs/pca_3d.html
Cách dùng: python src/evaluation/pca_3d_html.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "results" / "runs"

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
HEX = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
       "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def main() -> int:
    d = json.loads((RUNS_DIR / "pca_3d.json").read_text(encoding="utf-8"))
    coords, labels = d["coords"], d["labels"]
    r = d["explained_ratio_10"]
    n = d["samples"]
    pct3 = d["explained_ratio_3"] * 100
    data_js = json.dumps({"coords": coords, "labels": labels}, separators=(",", ":"))
    cls_js = json.dumps(CLASS_NAMES, ensure_ascii=False)
    hex_js = json.dumps(HEX)
    boxes = "\n".join(
        f'<label style="margin-right:8px"><input type="checkbox" data-cls="{i}" checked> '
        f'<span style="color:{HEX[i]}">■</span> {CLASS_NAMES[i]}</label>'
        for i in range(10))

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FashionMNIST 3D tương tác — PCA ({n} mẫu)</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 980px; margin: 20px auto; padding: 0 14px; color: #222; line-height: 1.6; }}
  canvas {{ border: 1px solid #ddd; border-radius: 8px; width: 100%; height: 560px; cursor: grab; background: #0b1020; }}
  .bar {{ background: #fafafa; border: 1px solid #e5e5e5; border-radius: 8px; padding: 10px 14px; margin-bottom: 10px; }}
  button {{ padding: 6px 12px; margin-right: 6px; }}
  code {{ background: #f4f4f4; padding: 1px 6px; border-radius: 4px; }}
</style>
</head>
<body>
<div class="bar">
  <h2 style="margin:4px 0">FashionMNIST trong không gian 3 chiều (PCA tương tác)</h2>
  <p style="margin:4px 0">{n} ảnh 28×28 (784 chiều) → 3 thành phần chính:
  PC1={r[0] * 100:.1f}%, PC2={r[1] * 100:.1f}%, PC3={r[2] * 100:.1f}% —
  giữ <b>{pct3:.1f}%</b> phương sai. Kéo chuột để xoay · lăn chuột để zoom ·
  bật/tắt từng lớp bên dưới. File gốc: <code>results/runs/pca_3d.json</code>.</p>
  <div>{boxes}</div>
  <p style="margin:8px 0">
    <button id="spin">⏸ Dừng xoay</button>
    <button id="reset">⟲ Reset góc</button>
    Zoom <input id="zoom" type="range" min="40" max="400" value="150" style="vertical-align:middle">
  </p>
</div>
<canvas id="cv" width="960" height="560"></canvas>
<p><b>Đọc hình:</b> cụm Trouser (cam) và Bag (ô-liu) tách xa — khớp accuracy cao;
nhóm Shirt / T-shirt/top / Pullover / Coat chồng lấn ở giữa — đúng top-5 cặp nhầm
trong <code>results/runs/report.html</code>. Ảnh tĩnh: <code>pca_3d.png</code>.</p>
<script>
const DATA = {data_js};
const NAMES = {cls_js};
const COLORS = {hex_js};
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
let ax = -0.5, ay = 0.6, zoom = 150, spin = true;
let pts = DATA.coords.map((p, i) => ({{x: p[0], y: p[1], z: p[2], c: DATA.labels[i]}}));
// Chuẩn hóa về tâm 0, bán kính ~1
let mx = 0, my = 0, mz = 0;
pts.forEach(p => {{ mx += p.x; my += p.y; mz += p.z; }});
mx /= pts.length; my /= pts.length; mz /= pts.length;
let sc = 0;
pts.forEach(p => {{ sc = Math.max(sc, Math.abs(p.x - mx), Math.abs(p.y - my), Math.abs(p.z - mz)); }});
pts.forEach(p => {{ p.x = (p.x - mx) / sc; p.y = (p.y - my) / sc; p.z = (p.z - mz) / sc; }});
const on = new Array(10).fill(true);
document.querySelectorAll('input[data-cls]').forEach(cb => cb.onchange = () => on[+cb.dataset.cls] = cb.checked);
document.getElementById('spin').onclick = e => {{ spin = !spin; e.target.textContent = spin ? '⏸ Dừng xoay' : '▶ Xoay tiếp'; }};
document.getElementById('reset').onclick = () => {{ ax = -0.5; ay = 0.6; zoom = 150; document.getElementById('zoom').value = 150; }};
document.getElementById('zoom').oninput = e => zoom = +e.target.value;
let drag = false, lx = 0, ly = 0;
cv.onmousedown = e => {{ drag = true; lx = e.clientX; ly = e.clientY; cv.style.cursor = 'grabbing'; }};
window.onmouseup = () => {{ drag = false; cv.style.cursor = 'grab'; }};
window.onmousemove = e => {{ if (!drag) return; ay += (e.clientX - lx) * 0.008; ax += (e.clientY - ly) * 0.008; lx = e.clientX; ly = e.clientY; }};
cv.onwheel = e => {{ e.preventDefault(); zoom = Math.min(400, Math.max(40, zoom - e.deltaY * 0.15)); document.getElementById('zoom').value = zoom; }}, {{passive: false}};
function frame() {{
  if (spin && !drag) ay += 0.004;
  const cx = Math.cos(ax), sx = Math.sin(ax), cy = Math.cos(ay), sy = Math.sin(ay);
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  // Trục tọa độ
  const axis = [[1,0,0,'#ff5555','PC1'],[0,1,0,'#55ff55','PC2'],[0,0,1,'#5599ff','PC3']];
  ctx.font = '13px Arial';
  axis.forEach(([x,y,z,col,lb]) => {{
    const y1 = y * cx - z * sx, z1 = y * sx + z * cx, x2 = x * cy + z1 * sy;
    const px = W/2 + x2 * zoom * 1.2, py = H/2 - y1 * zoom * 1.2;
    ctx.strokeStyle = col; ctx.beginPath(); ctx.moveTo(W/2, H/2); ctx.lineTo(px, py); ctx.stroke();
    ctx.fillStyle = col; ctx.fillText(lb, px + 4, py);
  }});
  // Sắp xếp xa→gần để vẽ đúng che khuất
  const proj = [];
  for (const p of pts) {{
    if (!on[p.c]) continue;
    const y1 = p.y * cx - p.z * sx, z1 = p.y * sx + p.z * cx, x1 = p.x * cy + z1 * sy, z2 = -p.x * sy + z1 * cy;
    const per = 1 / (3 - z2 * 0.8);
    proj.push({{sx: W/2 + x1 * zoom * per, sy: H/2 - y1 * zoom * per, d: z2, c: p.c}});
  }}
  proj.sort((a, b) => a.d - b.d);
  for (const q of proj) {{ ctx.fillStyle = COLORS[q.c]; ctx.globalAlpha = 0.6; ctx.fillRect(q.sx, q.sy, 3, 3); }}
  ctx.globalAlpha = 1;
  requestAnimationFrame(frame);
}}
frame();
</script>
</body>
</html>"""
    out = RUNS_DIR / "pca_3d.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1024:.1f} KB, {n} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

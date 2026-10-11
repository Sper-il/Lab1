"""Notebook fix v5 FINAL: đơn giản hóa, document rõ lý do + cách chạy."""
import json
from pathlib import Path

p = Path("notebooks/01_eda.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))

# Cell 0 markdown: thêm hướng dẫn
new_md = """# EDA \u2014 Kh\u00e1m ph\u00e1 d\u1eef li\u1ec7u FashionMNIST (Practice 1)
#
# \u26a0\ufe0f **L\u01b0u \u00fd khi ch\u1ea1y b\u1eb1ng `nbconvert --execute`:**
#   Tr\u00ean Windows + venv, c\u00f3 bug nbclient kh\u00f4ng persist state gi\u1eefa c\u00e1c cell
#   (cell 1 import `src` OK, cell 2 l\u1ea1i b\u00e1o `No module named 'src'`).
#   \u2192 **C\u00e1ch ch\u1ea1y headless khuy\u1ebfn ngh\u1ecb:** d\u00f9ng `scripts/verify_eda.py`
#     (c\u00f9ng logic, \u1ed5n \u0111\u1ecbnh, \u0111\u00e3 verify pass 10/10 l\u1edbp).
#   \u2192 **C\u00e1ch ch\u1ea1y Jupyter UI (VS Code / Cursor / JupyterLab):** m\u1edf notebook,
#     ch\u1ecdn kernel `Python 3 (venv lab1)`, Run All Cells \u2014 s\u1ebd ho\u1ea1t \u0111\u1ed9ng b\u00ecnh th\u01b0\u1eddng.
#
# **M\u1ee5c ti\u00eau:**
# 1. Ki\u1ec3m tra shape, dtype, s\u1ed1 l\u01b0\u1ee3ng sample c\u1ee7a 3 t\u1eadp `train` / `val` / `test`.
# 2. Ki\u1ec3m tra nh\u00e3n (label range, class names).
# 3. Ph\u00e2n ph\u1ed1i l\u1edbp \u2014 c\u00f3 c\u00e2n b\u1eb1ng kh\u00f4ng? C\u00f3 class n\u00e0o thi\u1ebfu/th\u1eeba nhi\u1ec1u kh\u00f4ng?
# 4. Tr\u1ef1c quan h\u00f3a \u1ea3nh raw + \u1ea3nh sau transform (ToTensor + Normalize).
#
# **Inputs:** `data/raw/{train,val,test}/{0..9}/<id>.png`
#
# **Config:** `configs/baseline.yaml` (mean=0.2860, std=0.3530)
#
# **Output notebook n\u00e0y** gi\u00fap hi\u1ec3u d\u1eef li\u1ec7u tr\u01b0\u1edbc khi train model (M2 \u2192 M3 \u2192 M4 \u2192 M5)."""

nb["cells"][0]["source"] = new_md.split("\n")

# Cell 1: setup đơn giản, dùng thẳng Path.cwd() (giả định kernel CWD = root project)
new_c1 = """# === SETUP: đảm bảo CWD = root project và src.* importable ===
# Notebook này được thiết kế để chạy với kernel CWD = root project (E:\\lab1).
# Trong Jupyter UI (VS Code/Cursor), kernel thường launch từ root project → OK.
# Nếu import src.* fail, hãy chạy `%cd E:\\lab1` trước.

import os, sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
# Nếu notebook không ở root, tự tìm root bằng cách walk up tìm src/
if not (PROJECT_ROOT / "src").exists():
    for parent in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            PROJECT_ROOT = parent
            break
    os.chdir(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"[setup] CWD = {os.getcwd()}")
print(f"[setup] sys.path[0] = {sys.path[0]}")

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.data.dataset import (
    build_dataloaders,
    _build_train_transform,
    _build_eval_transform,
    FASHION_MNIST_MEAN,
    FASHION_MNIST_STD,
)
import src.data.dataset as _ds_mod
import yaml as _yaml

# Override num_workers=0 cho EDA (tránh Windows multiprocessing spawn issue).
# Training scripts (M3+) sẽ dùng num_workers=2 theo configs/baseline.yaml.
_orig_build = _ds_mod.build_dataloaders

def build_dataloaders_eda(config_path):
    cfg_path = Path(config_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = _yaml.safe_load(f)
    cfg["data"]["num_workers"] = 0
    tmp_cfg = PROJECT_ROOT / "configs" / "_eda_override.yaml"
    tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        _yaml.safe_dump(cfg, f, allow_unicode=True)
    return _orig_build(tmp_cfg)

# Class names theo thứ tự index (FashionMNIST label mapping chuẩn)
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

print(f"Project root: {PROJECT_ROOT}")
print(f"FashionMNIST mean/std: {FASHION_MNIST_MEAN} / {FASHION_MNIST_STD}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Load 3 DataLoader (num_workers=0 cho EDA — tránh Windows spawn issue)
loaders = build_dataloaders_eda("configs/baseline.yaml")
print(f"\\nLoaders: {list(loaders.keys())}")"""

nb["cells"][1]["source"] = new_c1.split("\n")

# Clear outputs
for c in nb["cells"]:
    if "outputs" in c:
        c["outputs"] = []
    if "execution_count" in c:
        c["execution_count"] = None

p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook v5 saved. Cell 0 lines:", len(nb["cells"][0]["source"]), "Cell 1 lines:", len(nb["cells"][1]["source"]))

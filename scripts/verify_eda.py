"""Execute notebook 01_eda.ipynb từng cell để verify code chạy thực tế.

Chạy: python scripts/verify_eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Force UTF-8 stdout cho Windows console (cp1252 không in được tiếng Việt)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import nbformat
from nbclient import NotebookClient

NB_PATH = Path("notebooks/01_eda.ipynb")


def main() -> int:
    if not NB_PATH.exists():
        print(f"[FAIL] Không tìm thấy {NB_PATH}")
        return 1

    print(f"[INFO] Đọc notebook: {NB_PATH}")
    nb = nbformat.read(NB_PATH, as_version=4)

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    md_cells = [c for c in nb.cells if c.cell_type == "markdown"]
    print(f"[INFO] Tổng: {len(nb.cells)} cells  (code: {len(code_cells)}, markdown: {len(md_cells)})")

    print("[INFO] Bắt đầu execute (timeout 600s/cell)...")
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        # Ẩn output stdout khỏi terminal, chỉ in lỗi để tóm tắt gọn
        allow_errors=False,
    )
    try:
        client.execute()
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] Execute lỗi: {type(e).__name__}: {e}")
        # In cell nào lỗi
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == "code" and cell.get("outputs"):
                for out in cell.outputs:
                    if out.get("output_type") == "error":
                        print(f"  Cell #{i} ERROR: {out.get('ename')}: {out.get('evalue')}")
                        print("    Traceback:")
                        for line in out.get("traceback", [])[:5]:
                            print(f"      {line}")
        return 1

    # Tóm tắt output
    print("\n[OK] Notebook execute thành công!")
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        n_out = len(cell.get("outputs", []))
        n_stream = sum(1 for o in cell.get("outputs", []) if o.get("output_type") == "stream")
        n_disp = sum(1 for o in cell.get("outputs", []) if o.get("output_type") in ("display_data", "execute_result"))
        n_err = sum(1 for o in cell.get("outputs", []) if o.get("output_type") == "error")
        print(f"  Cell {i:>2}  outputs={n_out} (stream={n_stream}, display={n_disp}, error={n_err})")

    return 0


if __name__ == "__main__":
    sys.exit(main())

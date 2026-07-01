"""
generate_master_doc.py
Render Master_Writeup.md (Week 1 to Week 4 combined) to Master_Writeup.docx.
Figures are resolved relative to the repo root, so the weekN/figures/ links
in the master all resolve.  Run after build_master.py, from the repo root.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))          # project root, for md2docx
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from md2docx import render_markdown

MD  = os.path.join(HERE, "Master_Writeup.md")
OUT = os.path.join(HERE, "Master_Writeup.docx")

if __name__ == "__main__":
    out = render_markdown(MD, fig_dir=HERE, out_path=OUT, base_dir=HERE)
    print(f"Saved -> {out}")

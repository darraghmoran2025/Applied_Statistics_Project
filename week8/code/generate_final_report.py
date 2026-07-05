"""
generate_final_report.py
Render week8/writeup/Final_Report.md to Final_Report.docx.
Figure links in the report are written relative to the repo root
(weekN/figures/...), so the renderer resolves them from there.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, ".."))          # project root, for md2docx
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from md2docx import render_markdown

MD  = os.path.join(HERE, "..", "writeup", "Final_Report.md")
OUT = os.path.join(HERE, "..", "Final_Report.docx")

if __name__ == "__main__":
    out = render_markdown(MD, fig_dir=ROOT, out_path=OUT, base_dir=ROOT)
    print(f"Saved -> {out}")

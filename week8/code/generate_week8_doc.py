"""
generate_week8_doc.py
Render week8/writeup/Week8_BlackSwan.md to Week8_BlackSwan.docx in the same
folder, using the shared md2docx renderer at the project root.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))   # project root, for md2docx
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from md2docx import render_markdown

WRITEUP = os.path.join(HERE, "..", "writeup")
MD  = os.path.join(WRITEUP, "Week8_BlackSwan.md")
OUT = os.path.join(WRITEUP, "Week8_BlackSwan.docx")
FIG = os.path.join(HERE, "..", "figures")

if __name__ == "__main__":
    out = render_markdown(MD, fig_dir=FIG, out_path=OUT, base_dir=WRITEUP)
    print(f"Saved -> {os.path.abspath(out)}")

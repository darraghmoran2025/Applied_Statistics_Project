"""
build_master.py
Assemble a single master markdown of all project writeups to date, in order:
Week 1 literature review -> Week 5 posterior predictive checks.
The Week 3 lesson is excluded.

Figure links in each source use paths relative to that week's writeup folder
(../figures/...); they are rewritten to be relative to the repo root
(weekN/figures/...) so they resolve from the master's location.

Run: python build_master.py   (from the repo root)
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Master_Writeup.md")

# (markdown source, week folder used to rewrite ../figures/ links)
PARTS = [
    ("week1/writeup/Week1_Literature_Review.md", "week1"),
    ("week2/writeup/Week2_Results.md",           "week2"),
    ("week3/writeup/Week3_Results.md",           "week3"),
    ("week3/writeup/Week3_Regression.md",        "week3"),
    ("week4/writeup/Week4_Bayesian.md",          "week4"),
    ("week5/writeup/Week5_PPC.md",               "week5"),
]

HEADER = (
    "# Beyond Black-Scholes: Fitting Lévy Processes to Stock Returns\n\n"
    "*Master writeup, Weeks 1 to 5 combined. Compiled from the committed "
    "markdown sources; regenerate with `build_master.py`.*\n"
)


def main():
    chunks = [HEADER]
    for rel, week in PARTS:
        path = os.path.join(HERE, rel)
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        # ../figures/foo.png -> weekN/figures/foo.png
        text = text.replace("](../figures/", f"]({week}/figures/")
        chunks.append(text)
    master = "\n\n---\n\n".join(chunks) + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(master)
    print(f"Saved -> {OUT}  ({len(master):,} chars, {master.count(chr(10))+1} lines)")


if __name__ == "__main__":
    main()

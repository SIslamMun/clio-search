#!/usr/bin/env python3
"""Generate the L1 three-path comparison figure for the paper."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "eval" / "eval_final" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

with open(REPO / "eval" / "eval_final" / "outputs" / "L1_three_path.json") as f:
    data = json.load(f)

methods = ["Claude\nNative Agent", "NDP-MCP", "CLIO"]
tokens = [
    data["summary"]["run0_total_tokens"] / 1000,
    data["summary"]["run1_total_tokens"] / 1000,
    data["summary"]["run2_total_tokens"] / 1000,
]
times = [
    data["summary"]["run0_total_time_s"] / 60,
    data["summary"]["run1_total_time_s"] / 60,
    data["summary"]["run2_total_time_s"] / 60,
]

colors = ["#5B9BD5", "#ED7D31", "#70AD47"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

# --- Token bars ---
bars1 = ax1.bar(methods, tokens, color=colors, edgecolor="white", width=0.6)
for bar, val in zip(bars1, tokens):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
             f"{val:.0f}K", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax1.set_ylabel("Total Tokens (K)", fontsize=11)
ax1.set_title("Token Consumption", fontsize=12, fontweight="bold")
ax1.set_ylim(0, max(tokens) * 1.18)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Add reduction annotation
ax1.annotate(
    "−52.1%",
    xy=(2, tokens[2]),
    xytext=(1.5, tokens[0] * 0.75),
    fontsize=11,
    fontweight="bold",
    color="#70AD47",
    arrowprops=dict(arrowstyle="->", color="#70AD47", lw=1.5),
    ha="center",
)

# --- Time bars ---
bars2 = ax2.bar(methods, times, color=colors, edgecolor="white", width=0.6)
for bar, val in zip(bars2, times):
    label = f"{val:.1f} min"
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
             label, ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2.set_ylabel("Wall Time (minutes)", fontsize=11)
ax2.set_title("Execution Time", fontsize=12, fontweight="bold")
ax2.set_ylim(0, max(times) * 1.18)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Add reduction annotation
ax2.annotate(
    "−91.3%",
    xy=(2, times[2]),
    xytext=(1.5, times[0] * 0.65),
    fontsize=11,
    fontweight="bold",
    color="#70AD47",
    arrowprops=dict(arrowstyle="->", color="#70AD47", lw=1.5),
    ha="center",
)

fig.suptitle(
    "L1: Three-Path Agentic Discovery (10 queries, claude-sonnet-4-20250514)",
    fontsize=11,
    y=0.98,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])

out_path = OUT / "fig_L1_three_path.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")

# Also save PDF for LaTeX
fig.savefig(OUT / "fig_L1_three_path.pdf", bbox_inches="tight")
print(f"Saved: {OUT / 'fig_L1_three_path.pdf'}")

plt.close()

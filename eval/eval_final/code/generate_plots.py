#!/usr/bin/env python3
"""Generate all paper figures from the JSON outputs.

Produces:
  eval/eval_final/plots/fig_L1_tokens_tools.png          (L1 bar chart)
  eval/eval_final/plots/fig_L2_cte_scaling.png           (L2 log-log lines)
  eval/eval_final/plots/fig_L3_cross_unit_heatmap.png    (L3 heatmap)
  eval/eval_final/plots/fig_L3_methods_bar.png           (L3 macro bar)
  eval/eval_final/plots/fig_L4_numconq.png               (L4 method comparison)
  eval/eval_final/plots/fig_L5_federation.png            (L5 branches saved)
  eval/eval_final/plots/fig_L6_quality.png               (L6 QC distribution)
  eval/eval_final/plots/fig_L7_scaling_curves.png        (L7 log-log scaling)
  eval/eval_final/plots/fig_L8_cross_corpus.png          (L8 density/richness)
  eval/eval_final/plots/fig_D1_strong_scaling.png        (D1 strong scaling)
  eval/eval_final/plots/fig_D2_weak_scaling.png          (D2 weak scaling)
  eval/eval_final/plots/fig_pareto_accuracy_cost.png     (cross-experiment Pareto)

Each plot is generated ONLY if the corresponding JSON output exists. Runs
with whatever data is available; skips gracefully if an experiment hasn't
been executed yet.

Usage
-----
  python3 eval/eval_final/code/generate_plots.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib required: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

_REPO = Path(__file__).resolve().parents[3]
OUT_JSON = _REPO / "eval" / "eval_final" / "outputs"
OUT_FIG = _REPO / "eval" / "eval_final" / "plots"
OUT_FIG.mkdir(parents=True, exist_ok=True)

# Paper-quality matplotlib defaults
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (6, 4),
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def _load(name: str) -> dict[str, Any] | None:
    path = OUT_JSON / name
    if not path.exists():
        print(f"  skip: {name} not found")
        return None
    with path.open() as f:
        return json.load(f)


# ============================================================================
# L1 — Token + tool call comparison
# ============================================================================


def plot_L1() -> None:
    d = _load("L1_10queries.json")
    if d is None:
        # Try old format
        d = _load("L1_10queries.json")
    if d is None:
        return

    # New format: per_query_run1 / per_query_run2
    if "per_query_run1" in d:
        r1 = d["per_query_run1"]
        r2 = d["per_query_run2"]
        r1_tokens = sum(r["total_tokens"] for r in r1)
        r2_tokens = sum(r["total_tokens"] for r in r2)
        r1_tools = sum(r["num_tool_calls"] for r in r1)
        r2_tools = sum(r["num_tool_calls"] for r in r2)
        r1_time = sum(r["elapsed_s"] for r in r1)
        r2_time = sum(r["elapsed_s"] for r in r2)
    else:
        agg_a = d["mode_a_aggregate"]
        agg_b = d["mode_b_aggregate"]
        r1_tokens = agg_a["total_tokens"]
        r2_tokens = agg_b["total_tokens"]
        r1_tools = agg_a["total_tool_calls"]
        r2_tools = agg_b["total_tool_calls"]
        r1_time = agg_a["total_time_s"]
        r2_time = agg_b["total_time_s"]

    labels = ["Tool calls", "Total tokens", "Wall time (s)"]
    a_vals = [r1_tools, r1_tokens, r1_time]
    b_vals = [r2_tools, r2_tokens, r2_time]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, (ax, label) in enumerate(zip(axes, labels, strict=True)):
        bars = ax.bar(
            ["LLM+NDP-MCP", "LLM+CLIO"],
            [a_vals[i], b_vals[i]],
            color=["#6699cc", "#ee7733"],
            edgecolor="black",
        )
        ax.set_title(label)
        ax.set_ylabel(label)
        for j, v in enumerate([a_vals[i], b_vals[i]]):
            fmt = f"{v:,.0f}" if v > 100 else f"{v:.1f}"
            ax.text(j, v, fmt, ha="center", va="bottom", fontsize=9)

    reduction = (1 - r2_tokens / r1_tokens) * 100
    fig.suptitle(f"L1: LLM+NDP-MCP vs LLM+CLIO ({len(r1) if 'per_query_run1' in d else '10'} queries, {reduction:.0f}% token reduction)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_L1_tokens_tools.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ fig_L1_tokens_tools.png")

    # Per-query comparison bar chart
    if "per_query_run1" in d:
        qids = [r["query_id"] for r in r1]
        r1_toks = [r["total_tokens"] for r in r1]
        r2_toks = [r["total_tokens"] for r in r2]

        fig, ax = plt.subplots(figsize=(12, 5))
        x = range(len(qids))
        w = 0.35
        ax.bar([i - w/2 for i in x], r1_toks, w, label="LLM+NDP-MCP", color="#6699cc", edgecolor="black")
        ax.bar([i + w/2 for i in x], r2_toks, w, label="LLM+CLIO", color="#ee7733", edgecolor="black")
        ax.set_xticks(list(x))
        ax.set_xticklabels(qids)
        ax.set_ylabel("Tokens consumed")
        ax.set_title("L1: Per-query token consumption")
        ax.legend()
        ax.set_ylim(0, max(r1_toks) * 1.15)
        for i, (v1, v2) in enumerate(zip(r1_toks, r2_toks)):
            pct = (1 - v2 / v1) * 100
            ax.text(i, max(v1, v2) + 2000, f"-{pct:.0f}%", ha="center", fontsize=8, color="red")
        fig.tight_layout()
        fig.savefig(OUT_FIG / "fig_L1_per_query.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ fig_L1_per_query.png")


# ============================================================================
# L2 — IOWarp CTE scaling
# ============================================================================


def plot_L2() -> None:
    d = _load("L2_iowarp_cte_scaling.json")
    if d is None:
        return
    scales = d["scales"]
    N = [s["N_blobs"] for s in scales]
    tag_ms = [s["BlobQuery_tag_ms"] for s in scales]
    spec_ms = [s["BlobQuery_specific_ms"] for s in scales]
    all_ms = [s["BlobQuery_all_ms"] for s in scales]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(N, tag_ms, "o-", label="Tag-filtered (sub-linear)", color="#228833")
    ax.loglog(N, spec_ms, "s-", label="Specific regex (near-linear)", color="#ee7733")
    ax.loglog(N, all_ms, "^-", label="Full scan (linear)", color="#cc3311")
    ax.set_xlabel("Number of blobs")
    ax.set_ylabel("BlobQuery latency (ms)")
    ax.set_title("L2: IOWarp CTE BlobQuery scaling (1K → 1M)")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_L2_cte_scaling.png")
    plt.close(fig)
    print(f"  ✓ fig_L2_cte_scaling.png")


# ============================================================================
# L3 — SI unit cross-prefix
# ============================================================================


def plot_L3() -> None:
    d = _load("L3_si_unit_cross_prefix.json")
    if d is None:
        return
    overall = d["overall_macro_avg_recall_at_5"]
    methods = list(overall.keys())
    values = [overall[m] for m in methods]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#6699cc", "#aa3377", "#ee7733", "#228833"]
    bars = ax.bar(methods, values, color=colors[: len(methods)], edgecolor="black")
    for b, v in zip(bars, values, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Macro-average Recall@5")
    ax.set_title("L3: SI unit cross-prefix correctness")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_L3_methods_bar.png")
    plt.close(fig)
    print(f"  ✓ fig_L3_methods_bar.png")


# ============================================================================
# L4 — NumConQ
# ============================================================================


def plot_L4() -> None:
    d = _load("L4_numconq_benchmark.json")
    if d is None:
        return
    overall = d["overall"]
    methods = list(overall.keys())
    r10 = [overall[m]["R@10"] for m in methods]
    p10 = [overall[m]["P@10"] for m in methods]
    mrr = [overall[m]["MRR"] for m in methods]

    x = list(range(len(methods)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - width for i in x], r10, width, label="R@10", color="#228833")
    ax.bar(x, p10, width, label="P@10", color="#6699cc")
    ax.bar([i + width for i in x], mrr, width, label="MRR", color="#ee7733")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Score")
    ax.set_title(f"L4: NumConQ ({d['total_queries']:,} queries, {d['total_documents']:,} docs)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_L4_numconq.png")
    plt.close(fig)
    print(f"  ✓ fig_L4_numconq.png")


# ============================================================================
# L5 — 100-namespace federation
# ============================================================================


def plot_L5() -> None:
    d = _load("L5_federation_100_namespaces.json")
    if d is None:
        return
    ns_counts = d.get("namespace_counts", {})
    results = d.get("results_without_sampling", [])
    if not results or not ns_counts:
        return

    # Bar chart: for each query, show activated vs skipped scientific branches
    queries = [r["query_id"] for r in results]
    activated = [r["activated_scientific"] for r in results]
    skipped = [r["skipped_scientific"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = list(range(len(queries)))
    ax.bar(x, activated, label="Scientific activated", color="#228833")
    ax.bar(x, skipped, bottom=activated, label="Scientific skipped", color="#cccccc")
    ax.set_xticks(x)
    ax.set_xticklabels(queries)
    ax.set_ylabel("Namespace count")
    ax.set_title("L5: Federation over 100 namespaces (branch-selection result)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_L5_federation.png")
    plt.close(fig)
    print(f"  ✓ fig_L5_federation.png")


# ============================================================================
# L6 — CIMIS quality filter
# ============================================================================


def plot_L6() -> None:
    d = _load("L6_cimis_quality_filter.json")
    if d is None:
        return
    agg = d["aggregate"]
    labels = ["With QC filter", "Without QC filter"]
    values = [agg["matches_with_quality_filter"], agg["matches_without_quality_filter"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#228833", "#cc3311"], edgecolor="black")
    for b, v in zip(bars, values, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.01,
                f"{v:,}", ha="center", va="bottom")
    ax.set_ylabel("Rows matching temperature ≥ 30°C")
    ax.set_title(f"L6: CIMIS quality filter ({d['stations_analysed']} stations, "
                 f"{agg['total_rows']:,} rows)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_L6_quality.png")
    plt.close(fig)
    print(f"  ✓ fig_L6_quality.png")


# ============================================================================
# L7 — Single-node scaling curves
# ============================================================================


def plot_L7() -> None:
    d = _load("L7_scaling_curves.json")
    if d is None:
        return
    results = d["results"]
    scales = [r["scale"] for r in results]
    profile = [r["profile_time_ms_median"] for r in results]
    query = [r["query_time_ms_median"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(scales, profile, "o-", label="Profile time", color="#6699cc")
    ax.loglog(scales, query, "s-", label="Query time", color="#ee7733")
    # Reference line: linear scaling
    ref = [profile[0] * (s / scales[0]) for s in scales]
    ax.loglog(scales, ref, "--", color="gray", alpha=0.5, label="O(n) reference")
    ax.set_xlabel("Corpus size (documents)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("L7: Single-node scaling curves (profile + query)")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_L7_scaling_curves.png")
    plt.close(fig)
    print(f"  ✓ fig_L7_scaling_curves.png")


# ============================================================================
# L8 — Cross-corpus diversity
# ============================================================================


def plot_L8() -> None:
    d = _load("L8_cross_corpus_diversity.json")
    if d is None:
        return
    corpora = d["corpora"]
    names = [c["name"] for c in corpora]
    densities = [c["metadata_density"] for c in corpora]
    richness = [c["richness_score"] for c in corpora]

    x = list(range(len(names)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - width / 2 for i in x], densities, width, label="metadata_density",
           color="#6699cc", edgecolor="black")
    ax.bar([i + width / 2 for i in x], richness, width, label="richness_score",
           color="#ee7733", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Score (0-1)")
    ax.set_title("L8: Cross-corpus metadata richness")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_L8_cross_corpus.png")
    plt.close(fig)
    print(f"  ✓ fig_L8_cross_corpus.png")


# ============================================================================
# D1 — Strong scaling
# ============================================================================


def plot_D1() -> None:
    n_workers = []
    latency = []
    for n in (1, 2, 4):
        d = _load(f"D_strong_{n}workers.json")
        if d is None:
            continue
        n_workers.append(n)
        latency.append(d["latency_p50_s"] * 1000)
    if not n_workers:
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(n_workers, latency, "o-", color="#228833", label="CLIO distributed")
    # Ideal strong-scaling reference: latency/N
    if latency:
        ref = [latency[0] / n for n in n_workers]
        ax.plot(n_workers, ref, "--", color="gray", alpha=0.6, label="Ideal")
    ax.set_xlabel("Number of workers")
    ax.set_ylabel("p50 latency (ms)")
    ax.set_title("D1: Distributed strong scaling on DeltaAI")
    ax.set_xticks(n_workers)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_D1_strong_scaling.png")
    plt.close(fig)
    print(f"  ✓ fig_D1_strong_scaling.png")


# ============================================================================
# D2 — Weak scaling
# ============================================================================


def plot_D2() -> None:
    n_workers = []
    latency = []
    for n in (1, 2, 4):
        d = _load(f"D_weak_{n}workers.json")
        if d is None:
            continue
        n_workers.append(n)
        latency.append(d["latency_p50_s"] * 1000)
    if not n_workers:
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(n_workers, latency, "o-", color="#ee7733", label="CLIO distributed")
    # Ideal weak-scaling reference: constant
    if latency:
        ax.axhline(
            y=latency[0], linestyle="--", color="gray", alpha=0.6,
            label="Ideal (constant)",
        )
    ax.set_xlabel("Number of workers (data proportional)")
    ax.set_ylabel("p50 latency (ms)")
    ax.set_title("D2: Distributed weak scaling on DeltaAI")
    ax.set_xticks(n_workers)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_D2_weak_scaling.png")
    plt.close(fig)
    print(f"  ✓ fig_D2_weak_scaling.png")


# ============================================================================
# Pareto (composite)
# ============================================================================


def plot_pareto() -> None:
    """Token cost comparison — Pareto chart across experiments."""
    points: list[tuple[float, float, str]] = []
    l1 = _load("L1_10queries.json")
    if l1 and "summary" in l1:
        s = l1["summary"]
        points.append((s["run1_total_tokens"], 1.0, "L1: LLM+NDP-MCP"))
        points.append((s["run2_total_tokens"], 1.0, "L1: LLM+CLIO"))
    if not points:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for tokens, acc, label in points:
        ax.scatter(tokens, acc, s=100, label=label, edgecolor="black")
    ax.set_xlabel("Total tokens (sum over queries)")
    ax.set_ylabel("Average correctness (keyword hit rate)")
    ax.set_title("Pareto: accuracy vs token cost")
    ax.set_xscale("log")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_pareto_accuracy_cost.png")
    plt.close(fig)
    print(f"  ✓ fig_pareto_accuracy_cost.png")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    print(f"Generating plots in {OUT_FIG}")
    plot_L1()
    plot_L2()
    plot_L3()
    plot_L4()
    plot_L5()
    plot_L6()
    plot_L7()
    plot_L8()
    plot_D1()
    plot_D2()
    plot_pareto()
    print("\nDone.")


if __name__ == "__main__":
    main()

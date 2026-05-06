#!/usr/bin/env python3
"""
13_make_figures.py
==================
Generate figures and tables for the CompressedRescue paper.

Figures:
  fig1_pipeline.png          — two-row pipeline flowchart
  fig2_main_result.png       — genus recall: OPQ vs PQ vs Kraken
  fig3_per_genus.png         — per-genus recall comparison (OPQ vs PQ)
  fig4_rescue_outcome.png    — % rescued correctly vs still wrong

Tables (PNG + TSV):
  table1_data_summary
  table2_key_metrics
  table3_index_stats

Usage:
  conda activate bioenv
  cd ~/CompressedRescue
  python scripts/13_make_figures.py
"""

import argparse
import logging
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D

# ── defaults ────────────────────────────────────────────────────────────
ROOT        = os.path.expanduser("~/CompressedRescue")
META_DIR    = os.path.join(ROOT, "metadata")
METRICS_DIR = os.path.join(ROOT, "results", "metrics")
INDEX_DIR   = os.path.join(ROOT, "indices")
FIG_DIR     = os.path.join(ROOT, "results", "figures")
LOG_DIR     = os.path.join(ROOT, "logs")

C_OPQ   = "#c0392b"
C_PQ    = "#2980b9"
C_OK    = "#27ae60"
C_WRONG = "#e74c3c"
C_GREY  = "#95a5a6"

plt.rcParams.update({
    "figure.dpi":         110,
    "savefig.dpi":        200,
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.titlesize":     14,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})


def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "13_make_figures.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(),
        ],
    )


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Pipeline flowchart (two rows, straight connector)
# ═════════════════════════════════════════════════════════════════════════

def fig_pipeline(out: str) -> None:
    row1 = [
        ("Genomes\n(NCBI download)",        "#dfe6e9"),
        ("Simulate reads\n(ART, 150 bp)",   "#dfe6e9"),
        ("Build Kraken2 DB\n(train strains)", "#b2bec3"),
        ("Classify reads\n(Kraken2)",        "#b2bec3"),
        ("Extract rescue\nreads",            "#fdcb6e"),
    ]
    row2 = [
        ("Fragment refs\n+ k-mer embed",    "#f39c12"),
        ("Build FAISS\nindex (PQ / OPQ)",   "#e67e22"),
        ("Search + weighted\nvote → genus",  "#d35400"),
        ("Evaluate\n(P / R / F1)",           "#c0392b"),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(-0.2, max(len(row1), len(row2)) + 0.2)
    ax.set_ylim(-0.5, 4.5)
    ax.axis("off")

    bw, bh = 0.88, 1.1

    def draw_row(row, y, start_x=0):
        centers = []
        for i, (label, color) in enumerate(row):
            cx = start_x + i + 0.5
            box = FancyBboxPatch(
                (cx - bw/2, y - bh/2), bw, bh,
                boxstyle="round,pad=0.03,rounding_size=0.1",
                linewidth=0.8, edgecolor="#2d3436", facecolor=color,
            )
            ax.add_patch(box)
            tc = "white" if color in ("#d35400", "#c0392b") else "#2d3436"
            ax.text(cx, y, label, ha="center", va="center",
                    fontsize=9, color=tc, linespacing=1.3)
            centers.append((cx, y))
        for (x0, y0), (x1, y1) in zip(centers[:-1], centers[1:]):
            ax.add_patch(FancyArrowPatch(
                (x0 + bw/2 + 0.01, y0), (x1 - bw/2 - 0.01, y1),
                arrowstyle="-|>", mutation_scale=14, lw=1.2, color="#636e72"))
        return centers

    c1 = draw_row(row1, 3.2)
    c2 = draw_row(row2, 1.2, start_x=0.5)

    # straight vertical connector: bottom of last row1 → top of first row2
    # two straight segments: down then left
    end_x = c1[-1][0]
    end_y = c1[-1][1] - bh/2 - 0.02
    start_x = c2[0][0]
    start_y = c2[0][1] + bh/2 + 0.02
    mid_y = (end_y + start_y) / 2

    # draw as three line segments: down, across, down
    ax.plot([end_x, end_x], [end_y, mid_y],
            color="#636e72", lw=1.2, solid_capstyle="butt")
    ax.plot([end_x, start_x], [mid_y, mid_y],
            color="#636e72", lw=1.2, solid_capstyle="butt")
    ax.annotate("", xy=(start_x, start_y),
                xytext=(start_x, mid_y),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#636e72"))

    ax.set_title("Figure 1 — CompressedRescue pipeline", pad=15)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  ✓ {out}")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Main result (consistent % everywhere)
# ═════════════════════════════════════════════════════════════════════════

def fig_main_result(summary: pd.DataFrame, out: str) -> None:
    s = summary.sort_values("hybrid_genus_F1", ascending=False)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    labels = ["Kraken\nonly"] + [f"Hybrid\n({r['index'].upper()})"
                                  for _, r in s.iterrows()]
    values = [0.0] + [r["hybrid_genus_F1"] for _, r in s.iterrows()]
    colors = [C_GREY] + [C_OPQ if r["index"] == "opq" else C_PQ
                          for _, r in s.iterrows()]

    bars = ax.bar(labels, values, width=0.6, color=colors,
                  edgecolor="#2d3436", linewidth=0.7)

    for bar, val in zip(bars, values):
        label = f"{val:.1%}" if val > 0 else "0%"
        y_pos = val + 0.008 if val > 0 else 0.008
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                label, ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="#2d3436")

    ax.set_ylabel("Genus-level F1 score")
    ax.set_title("Figure 2 — Rescue layer performance\non previously unclassified reads")
    ax.set_ylim(0, max(values) * 1.3)
    # format y-axis as percentages to match bar labels
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logging.info(f"  ✓ {out}")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Per-genus recall (legend below plot, no overlap)
# ═════════════════════════════════════════════════════════════════════════

def fig_per_genus(per_genus: dict, out: str) -> None:
    if "opq" not in per_genus:
        logging.warning("  skip per_genus — no OPQ data")
        return

    opq = per_genus["opq"][["genus", "hybrid_recall", "n_reads"]].copy()
    opq.columns = ["genus", "recall_opq", "n_reads"]

    if "pq" in per_genus:
        pq = per_genus["pq"][["genus", "hybrid_recall"]].copy()
        pq.columns = ["genus", "recall_pq"]
        df = opq.merge(pq, on="genus", how="left")
    else:
        df = opq.copy()
        df["recall_pq"] = np.nan

    df = df.sort_values("recall_opq", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.55 * len(df))))

    y = np.arange(len(df))
    h = 0.35

    ax.barh(y + h/2, df["recall_opq"], h, color=C_OPQ,
            edgecolor="#2d3436", linewidth=0.5, label="OPQ")

    if df["recall_pq"].notna().any():
        ax.barh(y - h/2, df["recall_pq"], h, color=C_PQ,
                edgecolor="#2d3436", linewidth=0.5, label="PQ")

    # OPQ percentage labels
    for i, (val, n) in enumerate(zip(df["recall_opq"], df["n_reads"])):
        ax.text(val + 0.01, y[i] + h/2,
                f"{val:.0%}", va="center", fontsize=9, fontweight="bold",
                color=C_OPQ)
        ax.text(1.08, y[i], f"n={n:,}", va="center", fontsize=8,
                color="#636e72")

    ax.set_yticks(y)
    ax.set_yticklabels(df["genus"], fontsize=10)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Genus recall (fraction of rescue reads correctly classified)")
    ax.set_title("Figure 3 — Per-genus rescue performance")
    # legend below the plot, horizontal — never overlaps bars or n= labels
    ax.legend(
        frameon=False, fontsize=10,
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=2, handlelength=1.8, columnspacing=2.5,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  ✓ {out}")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Rescue outcome
# ═════════════════════════════════════════════════════════════════════════

def fig_rescue_outcome(breakdowns: dict, out: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))

    tags = sorted(breakdowns.keys(), reverse=True)  # OPQ first
    y_pos = np.arange(len(tags))

    for i, tag in enumerate(tags):
        bd = breakdowns[tag]
        total = bd["count"].sum()

        improved = 0
        wrong    = 0
        for _, row in bd.iterrows():
            b = row["bucket"]
            c = row["count"]
            if b in ("rescue_improved", "rescue_fixed", "both_correct"):
                improved += c
            else:
                wrong += c

        pct_ok   = improved / total
        pct_bad  = wrong / total

        ax.barh(i, pct_ok, color=C_OK, edgecolor="#2d3436",
                linewidth=0.5, height=0.5)
        ax.barh(i, pct_bad, left=pct_ok, color=C_WRONG, edgecolor="#2d3436",
                linewidth=0.5, height=0.5)

        ax.text(pct_ok / 2, i,
                f"Correctly rescued\n{pct_ok:.1%}",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white")
        ax.text(pct_ok + pct_bad / 2, i,
                f"Still wrong\n{pct_bad:.1%}",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([t.upper() for t in tags], fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of rescue reads")
    ax.set_title("Figure 4 — Rescue outcome breakdown")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logging.info(f"  ✓ {out}")


# ═════════════════════════════════════════════════════════════════════════
# TABLES
# ═════════════════════════════════════════════════════════════════════════

def collect_data_summary(meta_dir: str) -> pd.DataFrame:
    rows = []
    gpath = os.path.join(meta_dir, "genomes.tsv")
    if os.path.exists(gpath):
        g = pd.read_csv(gpath, sep="\t")
        rows.append(("Total genomes", f"{len(g):,}"))
        if "genus" in g.columns:
            rows.append(("Distinct genera", f"{g['genus'].nunique():,}"))
        if "species" in g.columns:
            rows.append(("Distinct species", f"{g['species'].nunique():,}"))
    spath = os.path.join(meta_dir, "splits.tsv")
    if os.path.exists(spath):
        s = pd.read_csv(spath, sep="\t")
        if "split" in s.columns:
            for name, count in s["split"].value_counts().items():
                rows.append((f"Genomes — {name}", f"{count:,}"))
    rpath = os.path.join(meta_dir, "read_truth.tsv")
    if os.path.exists(rpath):
        with open(rpath) as fh:
            n = sum(1 for _ in fh) - 1
        rows.append(("Total simulated reads", f"{n:,}"))
    xpath = os.path.join(meta_dir, "rescue_reads.tsv")
    if os.path.exists(xpath):
        x = pd.read_csv(xpath, sep="\t")
        rows.append(("Rescue reads", f"{len(x):,}"))
        if "rescue_reason" in x.columns:
            for reason, c in x["rescue_reason"].value_counts().items():
                rows.append((f"  — {reason}", f"{c:,}"))
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def build_table2(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    first = summary.iloc[0]
    rows.append({
        "Mode": "Kraken only", "Genus P": f"{first['kraken_genus_P']:.4f}",
        "Genus R": f"{first['kraken_genus_R']:.4f}",
        "Genus F1": f"{first['kraken_genus_F1']:.4f}",
        "Species P": "—", "Species R": "—", "Species F1": "—",
    })
    for _, r in summary.iterrows():
        rows.append({
            "Mode": f"Hybrid ({r['index'].upper()})",
            "Genus P":    f"{r['hybrid_genus_P']:.4f}",
            "Genus R":    f"{r['hybrid_genus_R']:.4f}",
            "Genus F1":   f"{r['hybrid_genus_F1']:.4f}",
            "Species P":  f"{r['hybrid_species_P']:.4f}",
            "Species R":  f"{r['hybrid_species_R']:.4f}",
            "Species F1": f"{r['hybrid_species_F1']:.4f}",
        })
    return pd.DataFrame(rows)


def parse_query_times(log_path: str) -> dict:
    times = {}
    if not os.path.exists(log_path):
        return times
    current = None
    with open(log_path) as fh:
        for line in fh:
            m = re.search(r"Loading index:\s+(faiss_\w+\.index)", line)
            if m:
                current = m.group(1)
                continue
            m = re.search(
                r"Search done\s+time\s*=\s*([\d\.]+)s\s+\(([\d,]+)\s+queries/sec\)",
                line)
            if m and current:
                times[current] = {
                    "total_s": float(m.group(1)),
                    "qps":     float(m.group(2).replace(",", "")),
                }
                times[current]["ms_per_query"] = (
                    1000.0 / times[current]["qps"]
                    if times[current]["qps"] > 0 else None)
                current = None
    return times


def build_table3(index_dir: str, log_path: str,
                 n_vectors: int = 3_711_052, d: int = 512) -> pd.DataFrame:
    rows = []
    times = parse_query_times(log_path)
    flat_mb = 4 * d * n_vectors / 1e6

    for f in sorted(os.listdir(index_dir)):
        if not f.endswith(".index"):
            continue
        size_mb = os.path.getsize(os.path.join(index_dir, f)) / 1e6
        t = times.get(f, {})
        ratio = f"{flat_mb / size_mb:.0f}x" if size_mb > 0 else "—"
        rows.append({
            "Index":       f.replace("faiss_", "").replace(".index", "").upper(),
            "Size (MB)":   f"{size_mb:.1f}",
            "Compression": ratio,
            "Time (s)":    f"{t.get('total_s'):.1f}" if t else "n/a",
            "q/sec":       f"{t.get('qps'):.0f}"     if t else "n/a",
            "ms/query":    f"{t.get('ms_per_query'):.2f}" if t else "n/a",
        })
    rows.append({
        "Index":       "FLAT (est.)",
        "Size (MB)":   f"{flat_mb:,.0f}",
        "Compression": "1x",
        "Time (s)":    "n/a",
        "q/sec":       "n/a",
        "ms/query":    "n/a",
    })
    return pd.DataFrame(rows)


def render_table(df: pd.DataFrame, title: str, out: str) -> None:
    if df is None or len(df) == 0:
        logging.warning(f"  skip {title} (no data)")
        return

    n_cols = len(df.columns)
    n_rows = len(df)
    # wider figure to fit text, auto-size columns
    fig_w = max(8, 1.5 * n_cols)
    fig_h = 1.0 + 0.45 * (n_rows + 1)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(cellText=df.values,
                   colLabels=list(df.columns),
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(list(range(n_cols)))
    tbl.scale(1.0, 1.5)

    # header styling
    for j in range(n_cols):
        tbl[0, j].set_facecolor("#2d3436")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # alternate row shading
    for i in range(1, n_rows + 1):
        bg = "#f5f6fa" if i % 2 == 0 else "white"
        for j in range(n_cols):
            tbl[i, j].set_facecolor(bg)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  ✓ {out}")


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Make all paper figures/tables")
    p.add_argument("--meta_dir",    default=META_DIR)
    p.add_argument("--metrics_dir", default=METRICS_DIR)
    p.add_argument("--index_dir",   default=INDEX_DIR)
    p.add_argument("--fig_dir",     default=FIG_DIR)
    p.add_argument("--log_dir",     default=LOG_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_dir)

    logging.info("=" * 60)
    logging.info("13_make_figures.py   — CompressedRescue")
    logging.info("=" * 60)

    os.makedirs(args.fig_dir, exist_ok=True)

    summary_path = os.path.join(args.metrics_dir, "summary.tsv")
    if not os.path.exists(summary_path):
        sys.exit(f"ERROR: {summary_path} not found — run 12_evaluate.py first")
    summary = pd.read_csv(summary_path, sep="\t")
    logging.info(f"Summary: {len(summary)} rows")

    index_sizes = {
        f: os.path.getsize(os.path.join(args.index_dir, f)) / 1e6
        for f in os.listdir(args.index_dir) if f.endswith(".index")
    }

    per_genus = {}
    breakdowns = {}
    for _, row in summary.iterrows():
        tag = row["index"]
        p1 = os.path.join(args.metrics_dir, f"per_genus_{tag}.tsv")
        p2 = os.path.join(args.metrics_dir, f"rescue_breakdown_{tag}.tsv")
        if os.path.exists(p1):
            per_genus[tag] = pd.read_csv(p1, sep="\t")
        if os.path.exists(p2):
            breakdowns[tag] = pd.read_csv(p2, sep="\t")

    search_log = os.path.join(args.log_dir, "10_search_index.log")

    # ── figures ──
    logging.info("")
    logging.info("Figures...")
    fig_pipeline(      os.path.join(args.fig_dir, "fig1_pipeline.png"))
    fig_main_result(   summary,
                       os.path.join(args.fig_dir, "fig2_main_result.png"))
    if per_genus:
        fig_per_genus( per_genus,
                       os.path.join(args.fig_dir, "fig3_per_genus.png"))
    if breakdowns:
        fig_rescue_outcome(breakdowns,
                       os.path.join(args.fig_dir, "fig4_rescue_outcome.png"))

    # ── tables ──
    logging.info("")
    logging.info("Tables...")
    t1 = collect_data_summary(args.meta_dir)
    render_table(t1, "Table 1 — Data summary",
                 os.path.join(args.fig_dir, "table1_data_summary.png"))
    t2 = build_table2(summary)
    render_table(t2, "Table 2 — Key metrics (rescue subset)",
                 os.path.join(args.fig_dir, "table2_key_metrics.png"))
    t3 = build_table3(args.index_dir, search_log)
    render_table(t3, "Table 3 — Index sizes and query times",
                 os.path.join(args.fig_dir, "table3_index_stats.png"))

    for label, df in [("table1_data_summary", t1),
                      ("table2_key_metrics", t2),
                      ("table3_index_stats", t3)]:
        if len(df):
            df.to_csv(os.path.join(args.metrics_dir, f"{label}.tsv"),
                      sep="\t", index=False)

    # ── list ──
    logging.info("")
    logging.info("Output files:")
    for f in sorted(os.listdir(args.fig_dir)):
        kb = os.path.getsize(os.path.join(args.fig_dir, f)) / 1024
        logging.info(f"  {f:35s}  {kb:7.1f} KB")
    logging.info("")
    logging.info("Done.  Pipeline complete.")


if __name__ == "__main__":
    main()

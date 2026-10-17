"""Front-page (teaser) figure: why similarity cannot carry identity.

(a) Query-centred view of one postcode block's bi-encoder embeddings.
(b) Cosine similarity of each present query to its true property vs. to its
    nearest identity-contradicting neighbour (full benchmark).
(c) What the top-1 answer is: neural-only stacks vs. PropertyLM-UK.

Inputs: candidate dumps from scripts/dump_candidates.py and the block vectors
for the panel-(a) example (evaluation/camera_ready/veitch_ex2_4af.json).

Usage (cwd = repo root):
    python scripts/make_teaser_figure.py OUT.pdf [--fonts DIR]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Circle, Patch

from camera_ready_analysis import load_dumps
from offline_rescore import Params, contradicts, decide

BLUE, RED, GRAY = "#2A78D6", "#C62D2D", "#8C939B"  # validated: CVD dE 26, contrast >= 3:1
INK, MUTED, GRID = "#1D1D1F", "#5F6368", "#E3E5E8"


def setup_fonts(font_dir: str | None) -> None:
    family = "DejaVu Sans"
    if font_dir:
        for f in Path(font_dir).glob("LinBiolinum_R*.otf"):
            font_manager.fontManager.addfont(str(f))
        family = "Linux Biolinum O"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
        }
    )


def panel_block(ax, example: dict) -> None:
    rows = example["rows"]
    gold = example["gold_uprn"]
    rows = sorted(rows, key=lambda r: int("".join(ch for ch in r["number"] if ch.isdigit()) or 0))
    cos_hi, cos_lo = 0.88, 0.74
    r_in, r_out = 0.32, 1.0

    def radius(c: float) -> float:
        return r_in + (cos_hi - c) / (cos_hi - cos_lo) * (r_out - r_in)

    for c in (0.85, 0.80):
        ax.add_patch(Circle((0, 0), radius(c), fill=False, lw=0.5, ec=GRID, zorder=0))
    gold_row = next(r for r in rows if r["uprn"] == gold)
    ax.add_patch(Circle((0, 0), radius(gold_row["cos"]), fill=False, lw=0.8, ec=BLUE, ls=(0, (2.5, 2)), zorder=1))

    n = len(rows)
    for i, r in enumerate(rows):
        theta = math.pi / 2 - 2 * math.pi * (i + 0.5) / n
        rad = radius(r["cos"])
        x, y = rad * math.cos(theta), rad * math.sin(theta)
        is_gold = r["uprn"] == gold
        ax.plot([0, x], [0, y], color=BLUE if is_gold else GRID, lw=0.8 if is_gold else 0.5, zorder=1)
        ax.scatter(x, y, s=46 if is_gold else 30, color=BLUE if is_gold else RED, ec="white", lw=1.0, zorder=3)
        lx, ly = (rad + 0.12) * math.cos(theta), (rad + 0.12) * math.sin(theta)
        ax.text(lx, ly, r["number"], ha="center", va="center", fontsize=6.6 if not is_gold else 7.2,
                color=INK, fontweight="bold" if is_gold else "normal", zorder=4)
    ax.scatter(0, 0, marker="*", s=95, color=INK, ec="white", lw=0.8, zorder=5)
    ax.text(0, -0.16, "query", ha="center", va="top", fontsize=6.2, color=INK)
    ax.text(-1.22, 1.18, f"rings: cosine 0.85, 0.80\ndashed: No. 18 ({gold_row['cos']:.3f})", ha="left", va="top",
            fontsize=5.6, color=MUTED, linespacing=1.25)
    closer = sum(r["cos"] > gold_row["cos"] for r in rows)
    ax.text(0, -1.30, f"“{example['query']}”\n{closer} different houses sit closer than No. 18",
            ha="center", va="top", fontsize=6.6, color=INK, linespacing=1.3)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.62, 1.22)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) One postcode block, seen from the query", loc="left", color=INK, pad=3)


def pair_similarities(recs, present, ok) -> tuple[np.ndarray, np.ndarray]:
    gold_s, contra_s = [], []
    for x in recs:
        if x["benchmark_id"] not in ok or not present[x["benchmark_id"]]:
            continue
        gold = [c for c in x["candidates"] if c["uprn"] == x["expected_uprn"]]
        con = [c for c in x["candidates"] if contradicts(x["keys"], c)]
        if gold and con:
            gold_s.append(gold[0]["retrieval_score"])
            contra_s.append(max(c["retrieval_score"] for c in con))
    return np.array(gold_s), np.array(contra_s)


def panel_hist(ax, gold_s: np.ndarray, contra_s: np.ndarray) -> None:
    bins = np.arange(0.40, 1.0001, 0.015)
    for data, color, z in ((contra_s, RED, 2), (gold_s, BLUE, 3)):
        h, e = np.histogram(data, bins=bins)
        h = h / len(data) * 100
        ax.fill_between(e[:-1], h, step="post", color=color, alpha=0.14, lw=0, zorder=z)
        ax.step(e[:-1], h, where="post", color=color, lw=1.2, zorder=z + 1)
    thr = np.percentile(gold_s, 5)
    admitted = (contra_s >= thr).mean() * 100
    ax.axvline(thr, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=4)
    top = ax.get_ylim()[1]
    ax.text(thr + 0.010, top * 0.72,
            f"95%-recall\nthreshold\nadmits\n{admitted:.1f}% of\ncontradicting\nneighbours",
            ha="left", va="top", fontsize=6.2, color=INK, linespacing=1.25)
    ax.set_xlim(0.40, 1.0)
    ax.set_xlabel("Bi-encoder cosine similarity to query")
    ax.set_ylabel("% of present queries")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor=BLUE, alpha=0.6, label=f"true property (median {np.median(gold_s):.3f})"),
            Patch(facecolor=RED, alpha=0.6, label=f"nearest contradicting neighbour ({np.median(contra_s):.3f})"),
        ],
        loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=True, fontsize=6.2, handlelength=1.0,
        handletextpad=0.4, borderaxespad=0.1, facecolor="white", edgecolor="none", framealpha=1.0,
    ).set_zorder(10)
    ax.set_title(f"(b) Similarity cannot separate them (n = {len(gold_s):,})", loc="left", color=INK, pad=3)


def outcome_shares(recs, present, ok, p: Params) -> dict:
    counts = {"correct": 0, "contra": 0, "other": 0}
    n = 0
    for x in recs:
        if x["benchmark_id"] not in ok or not present[x["benchmark_id"]]:
            continue
        n += 1
        d = decide(x, p)
        if not d:
            counts["other"] += 1
        elif d[0]["uprn"] == x["expected_uprn"]:
            counts["correct"] += 1
        elif d[0]["contradicts"]:
            counts["contra"] += 1
        else:
            counts["other"] += 1
    return {k: 100 * v / n for k, v in counts.items()}


def panel_outcomes(ax, shares: list[tuple[str, dict]]) -> None:
    y = np.arange(len(shares))[::-1]
    for yi, (name, s) in zip(y, shares):
        left = 0.0
        for key, color in (("correct", BLUE), ("contra", RED), ("other", GRAY)):
            w = s[key]
            ax.barh(yi, w, left=left, height=0.56, color=color, ec="white", lw=1.0)
            if w >= 7:
                ax.text(left + w / 2, yi, f"{w:.1f}%", ha="center", va="center", fontsize=6.6, color="white",
                        fontweight="bold")
            left += w
    ax.set_yticks(y, [n for n, _ in shares], fontsize=6.8, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.set_xlabel("Top-1 answer on present queries")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    final = shares[-1][1]
    ax.annotate(f"{final['contra']:.1f}% cross a key —\nall forced to NO_MATCH",
                xy=(final["correct"] + final["contra"] / 2, y[-1] - 0.30), xytext=(68, y[-1] - 0.78),
                fontsize=6.2, color=INK, ha="center", va="top",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_ylim(y[-1] - 1.62, y[0] + 0.95)
    ax.legend(
        handles=[Patch(color=BLUE, label="correct"),
                 Patch(color=RED, label="wrong, crosses a parsed key"),
                 Patch(color=GRAY, label="wrong, other")],
        loc="upper left", bbox_to_anchor=(-0.02, 1.0), ncol=3, frameon=False, fontsize=6.0,
        handlelength=0.9, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.0,
    )
    ax.set_title("(c) Adjudication removes the dangerous error", loc="left", color=INK, pad=3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--fonts")
    ap.add_argument("--dumps", nargs="+", default=["evaluation/camera_ready/a", "evaluation/camera_ready/b"])
    ap.add_argument("--example", default="evaluation/camera_ready/veitch_ex2_4af.json")
    ap.add_argument("--ok", default="evaluation/camera_ready/ok_9992.csv")
    args = ap.parse_args()
    setup_fonts(args.fonts)

    recs, _ = load_dumps(args.dumps)
    cov = pd.read_csv("evaluation/SX9090_coverage_current.csv", dtype=str)
    present = dict(zip(cov.benchmark_id, cov.uprn_in_corpus == "1"))
    ok = set(pd.read_csv(args.ok, dtype=str).benchmark_id)

    example = json.loads(Path(args.example).read_text())
    example["gold_uprn"] = "10023118351"

    gold_s, contra_s = pair_similarities(recs, present, ok)
    shares = [
        ("Bi-encoder only", outcome_shares(recs, present, ok, Params(use_rules=False, use_reranker=False))),
        ("+ cross-encoder", outcome_shares(recs, present, ok, Params(use_rules=False))),
        ("PropertyLM-UK", outcome_shares(recs, present, ok, Params())),
    ]

    fig = plt.figure(figsize=(7.0, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.86, 1.18, 1.0], wspace=0.42, left=0.005, right=0.985, top=0.86,
                          bottom=0.17)
    panel_block(fig.add_subplot(gs[0]), example)
    panel_hist(fig.add_subplot(gs[1]), gold_s, contra_s)
    panel_outcomes(fig.add_subplot(gs[2]), shares)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(str(Path(args.out).with_suffix(".png")), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(json.dumps({"pairs": len(gold_s), "shares": shares}, indent=1))


if __name__ == "__main__":
    main()

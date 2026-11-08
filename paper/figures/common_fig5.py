"""Data and drawing for Fig. 5. Every number is read here from saved results, except the hand-set layout of the
drawings (positions only).
- CLRS-30, the 186k-parameter OR-join layer: data/clrs/scores_c13.json (12 tasks x 3 seeds, test score at K = 1..16
  loops; K = 8 reported), data/clrs/scores_c14.json (DFS, K = 1..34; K = 34 reported).
- Published, no hints (data/clrs/published.json): FloydNet (Yu et al., 73M parameters, Table 10) and the two node-state
  models (Bevilacqua et al. 2023 'No Hints'; Rodionov and Prokhorenkova 2023 'No hint'), best of the two per task.
- data/fig5_margins.json (every decision's relative margin, from the saved predictions).
- Language: data/lab/results/p27-*.
"""
import json
import math
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from style import *  # noqa: E402,F401,F403
from style import house, HERE  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Polygon, Rectangle, Wedge  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from scipy.interpolate import splev, splprep  # noqa: E402

DATA = HERE / "data"
CLRS = DATA / "clrs"
WATER, SHORE, WOOD, WOOD_E = "#E6ECF8", "#C3CEE5", "#F2F1EE", "#DEDBD3"   # mapdraw's water and woods
PINK, PINK_E = "#FBE3EB", "#F1A3BC"                                      # cr_exact's pale rose

TASKS = ["bfs", "topological_sort", "articulation_points", "bridges", "mst_kruskal", "mst_prim", "bellman_ford",
         "dijkstra", "dag_shortest_paths", "floyd_warshall", "dfs", "strongly_connected_components"]
NAME = {"bfs": "BFS", "topological_sort": "topological sort", "articulation_points": "articulation points",
        "bridges": "bridges", "mst_kruskal": "Kruskal", "mst_prim": "Prim", "bellman_ford": "Bellman–Ford",
        "dijkstra": "Dijkstra", "dag_shortest_paths": "DAG shortest paths", "floyd_warshall": "Floyd–Warshall",
        "dfs": "depth-first search", "strongly_connected_components": "strong components"}
# The framework's classes: an NC closed form over walks, trees or a semiring closure; the
# answer indexed by a pair (node-state models fail it); the output fixed by lexicographic DFS order (P-complete).
CLASS = {t: "closed form" for t in TASKS}
CLASS["floyd_warshall"] = "answer is a pair"
CLASS["dfs"] = CLASS["strongly_connected_components"] = "sequential order"


# ------------------------------------------------------------------------------------------------------------ data
def c13():
    """{task: array (3 seeds, 16 loops)} of test scores in %, our layer (C13)."""
    s = json.loads((CLRS / "scores_c13.json").read_text())
    out = {}
    for v in s.values():
        out.setdefault(v["task"], {})[v["seed"]] = v["test"]
    return {t: 100 * np.array([d[k] for k in sorted(d)]) for t, d in out.items()}


def c14():
    """array (3 seeds, 34 loops) of DFS test scores in %, trained with K from 4 to 24 (C14)."""
    s = json.loads((CLRS / "scores_c14.json").read_text())
    return 100 * np.array([v["test"] for v in sorted(s.values(), key=lambda v: v["seed"])])


def published():
    p = json.loads((CLRS / "published.json").read_text())
    fl = {t: p["floydnet_no_hint"][t][0] for t in TASKS}
    ns = {t: max(p["bevilacqua2023_no_hints"][t][0], p["rodionov2023_no_hint_ours"][t][0]) for t in TASKS}
    return fl, ns


def margins():
    return json.loads((DATA / "fig5_margins.json").read_text())


def language_gains():
    loss = lambda a, s: json.loads((DATA / "lab" / "results" / f"p27-{a}-s{s}" / "result.json").read_text())["result"]["test_loss"]  # noqa: E731,E501
    return {a: float(np.mean([loss("r", s) - loss(a, s) for s in (0, 1)])) for a in ("every2", "r10")}


# ------------------------------------------------------------------------------------------------------------ drawing


# ------------------------------------------------------------------------------------------------ shared panels
def dumbbell(fig, x0, y0, w, h, lo=60.0, label_w=24.0, class_w=17.0, inline34=True):
    """CLRS per task: our layer (C13, K = 8) against FloydNet without hints, grouped by the framework's class, with
    the best node-state model as a hollow mark (off-scale ones at the axis edge with their value) and DFS at 34 loops
    (C14). Returns the axes; data x = accuracy (%), data y = mm downwards."""
    ours, ours14 = c13(), c14()
    fl, ns = published()
    mm_per = (w - label_w - class_w) / (100 - lo)
    xl, xr = lo - label_w / mm_per, 100 + class_w / mm_per
    ax = axes_mm(fig, x0, y0, w, h)
    ax.set_xlim(xl, xr)
    ax.set_ylim(h, 0)
    row, pad, gap = 3.25, 0.9, 1.0
    groups = [("closed form", WATER, [t for t in TASKS if CLASS[t] == "closed form"]),
              ("answer is a pair", WOOD, ["floyd_warshall"]),
              ("sequential order", PINK, ["dfs", "strongly_connected_components"])]
    y = 0.0
    ys = {}
    for name, col, ts in groups:
        top = y
        y += pad
        for t in ts:
            ys[t] = y + row / 2
            y += row
        y += pad
        ax.add_patch(Rectangle((xl, top), xr - xl, y - top, fc=col, ec="none", zorder=0))
        ax.text(xr - 0.6 / mm_per, (top + y) / 2, name, ha="right", va="center", fontsize=SMALL, color=SOFT,
                style="italic")
        y += gap
    bottom = y - gap
    for t in TASKS:
        yy = ys[t]
        m = float(ours[t][:, 7].mean())
        ax.text(xl + 1.2 / mm_per, yy, NAME[t], ha="left", va="center", fontsize=SMALL, color=INK)
        a, b = sorted([fl[t], m])
        ax.plot([a, b], [yy, yy], color=INK, lw=0.75, zorder=2, solid_capstyle="butt")
        if ns[t] >= lo:
            ax.plot(ns[t], yy, "o", ms=3.0, mfc="white", mec=GREY, mew=0.6, zorder=3)
        else:
            ax.plot(lo + 0.35 / mm_per * 2, yy, "<", ms=3.0, mfc="white", mec=GREY, mew=0.6, zorder=3)
            ax.text(lo + 2.7 / mm_per, yy, f"{ns[t]:.0f}", ha="left", va="center", fontsize=SMALL, color=GREY)
        ax.plot(fl[t], yy, "o", ms=4.6, color=GREY, mew=0, zorder=4)
        ax.plot(m, yy, "o", ms=4.6, color=ROSE, mew=0, zorder=5)
        if t == "dfs":
            m34 = float(ours14[:, 33].mean())
            ax.plot([m, m34], [yy, yy], color=ROSE, lw=0.6, ls=(0, (1.2, 1.2)), zorder=2)
            ax.plot(m34, yy, "o", ms=4.4, mfc="white", mec=ROSE, mew=0.9, zorder=5)
            if inline34:
                ax.text(m34 + 1.4 / mm_per, yy, "34 loops", ha="left", va="center", fontsize=SMALL, color=ROSE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_bounds(lo, 100)
    ax.spines["bottom"].set_position(("data", bottom + 1.2))
    ax.spines["bottom"].set_color(INK)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.set_yticks([])
    ticks = list(range(int(lo), 101, 10))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.tick_params(axis="x", colors=INK, length=2.0, width=0.5, pad=1.5)
    ax.set_facecolor("none")
    return ax, mm_per, xl, bottom


def loops_c14(ax, show_c13=True, show_floydnet=True, logx=True):
    """DFS accuracy against the number of loops at test: C14 (trained with 4-24 loops) and C13 (4-8)."""
    d14 = c14()
    K = np.arange(1, d14.shape[1] + 1)
    m = d14.mean(0)
    ax.fill_between(K, d14.min(0), d14.max(0), color=ROSE, alpha=0.16, lw=0, zorder=2)
    ax.plot(K, m, color=ROSE, lw=1.0, zorder=4)
    sel = [k for k in (1, 2, 4, 8, 16, 32) if k <= len(K)]
    ax.plot(sel, [m[k - 1] for k in sel], "o", color=ROSE, ms=3.6, mew=0, zorder=5)
    if show_c13:
        d13 = c13()["dfs"]
        K13 = np.arange(1, d13.shape[1] + 1)
        ax.plot(K13, d13.mean(0), color=ROSE_L, lw=0.9, zorder=3)
    if show_floydnet:
        fl, _ = published()
        ax.axhline(fl["dfs"], color=GREY, lw=0.6, ls=(0, (3.0, 1.6)), zorder=1)
    if logx:
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.85, 40)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
        ax.minorticks_off()
    return d14, m


def margin_hists(ax):
    """Relative decision margins: all decisions of four tasks against the errors of the three near-tie tasks, and
    Prim's errors apart (C13, K = 8, 3 seeds)."""
    mg = margins()
    edges = np.logspace(np.log10(3e-4), np.log10(40), 25)
    allm, err, prim = [], [], []
    for t, v in mg.items():
        m = np.array([np.nan if x is None else x for x in v["margin"]])
        w = np.array(v["wrong"])
        ok = np.isfinite(m)
        allm += list(m[ok])
        (prim if t == "mst_prim" else err).extend(m[ok & w])
    ha, _ = np.histogram(allm, edges)
    he, _ = np.histogram(err, edges)
    hp, _ = np.histogram(prim, edges)

    def step(h, col, fc, z, ls="-", lw=0.8):
        xs = np.repeat(edges, 2)[1:-1]
        ys = np.repeat(np.maximum(h, 0.0), 2).astype(float)
        yb = np.where(ys > 0, ys, 0.5)
        if fc:
            ax.fill_between(xs, 0.5, yb, color=fc, lw=0, zorder=z)
        ax.plot(np.r_[xs[0], xs, xs[-1]], np.r_[0.5, yb, 0.5], color=col, lw=lw, ls=ls, zorder=z + 0.5)

    step(ha, COB, WATER, 1)
    step(he, ROSE, PINK, 2)
    ax.fill_between([], [], [])
    step(hp, ROSE_D, None, 3, ls=(0, (1.0, 1.1)), lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0.8, 2.5e4)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10])
    ax.set_xticklabels(["0.1%", "1%", "10%", "100%", "1,000%"])
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.set_yticklabels(["1", "10", "100", "1,000", "10,000"])
    ax.minorticks_off()
    return dict(n_all=len(allm), n_err=len(err), n_prim=len(prim), edges=edges, ha=ha, he=he, hp=hp,
                med_all=float(np.median(allm)), med_err=float(np.median(err)), med_prim=float(np.median(prim)))



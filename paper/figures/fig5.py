"""Fig. 5, what the principles predict: each principle predicts a pattern on tasks the layers were not designed for.

a  CLRS against FloydNet. Per task, the 186k-parameter OR-join layer (C13, K = 8, mean of 3 seeds, official test file)
   against FloydNet without hints (73M, published), in three classes (P2: closed form; P1: answer is a pair; P2's
   limit: sequential order); open marks: the best other published model without hints (Triplet-GMPNN based:
   Bevilacqua et al. 2023 or Rodionov and Prokhorenkova 2023), off-scale values written beside an arrow at the axis
   edge; DFS also at 34 loops (C14).
b  Loops for sequential order. DFS accuracy against the number of loops at test (log scale): trained with 4 to 24
   loops (C14, mean and range of 3 seeds) and with 4 to 8 (C13); FloydNet's score dashed.
c  Errors sit at near-ties. Relative decision margins (how much worse the best alternative is) of every decision of
   Bellman-Ford, Dijkstra, Kruskal and Prim, against the errors of the first three, and Prim's errors apart (C13,
   K = 8, 3 seeds; data/fig5_margins.json).
d  Where joins add little. Top: interaction share of trained joins and of the same joins at initialisation, SET
   (42 trained, 3 initial) against chess (6 trained, 12 initial) (data/shares.json; 1 = as much pair structure as
   random scores). Language has no interaction share in this definition, so it is left out. Bottom: language,
   test-loss gain over the 8-layer baseline of a join over the 16 previous tokens against two more layers, mean of
   2 seeds (data/lab/results/p27-*).

    cd paper/figures && uv run --no-project --with numpy --with scipy --with matplotlib==3.10.7 python -I fig5.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common_fig5 import *  # noqa: E402,F401,F403
import common_fig5 as fc  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

H = 120.0
fig = figure(H)
L0, R0 = 14.0, 177.0                 # left edge of a and c, right edge of b and d
AW = 90.0                            # a's band width
BL = 118.0                           # left spine of b and d
TOP1, TOP2 = 17.0, 82.0              # top spines of rows 1 and 2
BOT2 = 110.0
TY1, TY2 = 7.2, 75.6                 # title baselines (va top) of rows 1 and 2


def title(x, y, L, s):
    text(fig, x, y, L, LETTER, fontweight="bold")
    text(fig, x + 4.6, y, s, LETTER)


text(fig, 1.0, 0.8, "What the principles predict", LETTER, fontweight="bold")

# a: CLRS against FloydNet --------------------------------------------------------------------------------------
title(1.0, TY1, "a", "CLRS against FloydNet")
legend_at(fig, L0 + 0.5, TY1 + 4.4,
          [(hkey(ROSE, ls="None", marker="o", ms=4.6), "ours, 186k"),
           (hkey(ROSE, ls="None", marker="o", ms=4.4, mfc="white", mew=0.9), "ours, 34 loops"),
           (hkey(GREY, ls="None", marker="o", ms=4.6), "FloydNet, ≥ 73M"),
           (hkey(GREY, ls="None", marker="o", ms=3.0, mfc="white", mew=0.6), "best other no-hint model")], ncol=4)
axa, mmp, xl, bottom = fc.dumbbell(fig, L0, TOP1, AW, 49.0, label_w=22.0, class_w=19.5, inline34=False)
axa.set_xlabel("accuracy (%)", labelpad=2.0)
BOT1 = TOP1 + bottom + 1.2                                          # a's axis line

# b: Loops for sequential order ---------------------------------------------------------------------------------
title(BL - 13.0, TY1, "b", "Loops for sequential order")
axb = axes_mm(fig, BL, TOP1, R0 - BL, BOT1 - TOP1)
d14, m = fc.loops_c14(axb)
fl, _ = fc.published()
axb.set_ylim(30, 100)
axb.set_yticks([40, 60, 80, 100])
axb.set_yticklabels(["40", "60", "80", "100"])
axb.set_xlabel("loops at test", labelpad=2.0)
axb.set_ylabel("DFS accuracy (%)")
clean(axb)
legend(axb, [(hkey(ROSE, lw=1.0, marker="o", ms=3.0), "trained with 4–24 loops"),
             (hkey(ROSE_L, lw=0.9), "trained with 4–8 loops"),
             (hkey(GREY, lw=0.6, ls=(0, (3.0, 1.6))), "FloydNet, ≥ 73M")], loc="upper left")

# c: Errors sit at near-ties ------------------------------------------------------------------------------------
title(1.0, TY2, "c", "Errors sit at near-ties")
axc = axes_mm(fig, L0, TOP2, 88.0 - L0, BOT2 - TOP2)
st = fc.margin_hists(axc)
axc.set_ylim(0.8, 3e5)
axc.set_yticks([1, 10, 100, 1000, 10000])
axc.set_yticklabels(["1", "10", "100", "1,000", "10,000"])
axc.set_xlabel("decision margin (best alternative, relative)", labelpad=2.0)
axc.set_ylabel("decisions")
clean(axc)
legend(axc, [(Patch(fc=PINK, ec=ROSE, lw=0.8), "errors: Bellman–Ford, Dijkstra, Kruskal"),
             (Patch(fc=WATER, ec=COB, lw=0.8), "all decisions, four tasks"),
             (hkey(ROSE_D, lw=0.8, ls=(0, (1.0, 1.1))), "Prim's errors")], loc="upper left", ncol=2)

# d: Where joins add little -------------------------------------------------------------------------------------
title(BL - 13.0, TY2, "d", "Where joins add little")
sh = json.loads((DATA / "shares.json").read_text())
shares = {("SET", "trained"): [r["share"] for r in sh["set"]],
          ("SET", "init"): [r["share"] for r in sh["set_init"]],
          ("chess", "trained"): [r["share"] for r in sh["chess"] if r["state"] == "trained"],
          ("chess", "init"): [r["share"] for r in sh["chess"] if r["state"] == "init"]}
D1H, GAPD = 9.0, 9.0
axd = axes_mm(fig, BL, TOP2, R0 - BL, D1H)
rng = np.random.default_rng(1)
for y, (task, col) in enumerate((("SET", ROSE), ("chess", SOFT))):
    for state in ("init", "trained"):
        v = np.array(shares[(task, state)])
        jit = rng.uniform(-0.2, 0.2, len(v))
        if state == "trained":
            axd.plot(v, y + jit, "o", ms=2.2, color=col, mew=0, alpha=0.85, zorder=3)
        else:
            axd.plot(v, y + jit, "o", ms=2.2, mfc="white", mec=col, mew=0.5, zorder=3)
    axd.text(-0.06, y, task, ha="right", va="center", fontsize=SMALL, color=INK)
axd.axvline(1.0, color=VIOLET, lw=0.6, ls=DOT, zorder=1)
axd.set_ylim(1.55, -0.55)
axd.set_xlim(-0.02, 1.12)
axd.set_xticks([0, 0.5, 1])
axd.set_xticklabels(["0", "0.5", "1"])
axd.set_xlabel("interaction share (random scores = 1)", labelpad=2.0)
clean(axd, left=False)
legend_at(fig, BL + 22.0, TY2 + 0.1, [(hkey(INK, ls="None", marker="o", ms=2.6), "trained"),
                                      (hkey(INK, ls="None", marker="o", ms=2.6, mfc="white", mew=0.5),
                                       "at initialisation")], ncol=2)
lg = fc.language_gains()
axe = axes_mm(fig, BL, TOP2 + D1H + GAPD, R0 - BL, BOT2 - (TOP2 + D1H + GAPD))
for y, (name, key, col) in enumerate((("join, 16 tokens", "every2", ROSE), ("two more layers", "r10", COB))):
    axe.barh(y, lg[key], height=0.5, color=col, lw=0, zorder=3)
    axe.text(-0.0011, y, name, ha="right", va="center", fontsize=SMALL, color=INK)
    axe.text(lg[key] + 0.0009, y, f"{lg[key]:.3f}", ha="left", va="center", fontsize=SMALL, color=col)
axe.set_ylim(1.6, -0.6)
axe.set_xlim(0, 0.033)
axe.set_xticks([0, 0.01, 0.02, 0.03])
axe.set_xticklabels(["0", "0.01", "0.02", "0.03"])
axe.set_xlabel("language: loss reduction (nats per token)", labelpad=2.0)
clean(axe, left=False)


# shared edges, measured at 300 dpi -------------------------------------------------------------------------------
def px(v_mm):
    return v_mm / 25.4 * 300


fig.canvas.draw()
inv = fig.dpi / 300.0


def spine_px(ax, which):
    b = ax.get_window_extent()
    return {"left": b.x0, "right": b.x1, "top": fig.bbox.height - b.y1, "bottom": fig.bbox.height - b.y0}[which] / inv


a_axis = px(BOT1)
a_top = px(TOP1)
a_left = px(L0)
checks = {"row1 top (a band top, b)": (a_top, spine_px(axb, "top")),
          "row1 bottom (a axis line, b)": (a_axis, spine_px(axb, "bottom")),
          "row2 top (c, d top)": (spine_px(axc, "top"), spine_px(axd, "top")),
          "row2 bottom (c, d bottom)": (spine_px(axc, "bottom"), spine_px(axe, "bottom")),
          "left (a band, c)": (a_left, spine_px(axc, "left")),
          "right (b, d top, d bottom)": (spine_px(axb, "right"), spine_px(axd, "right"), spine_px(axe, "right"))}
save(fig, "fig5")
for k, v in checks.items():
    print(f"{k}: " + ", ".join(f"{x:.1f}" for x in v) + f"  (spread {max(v) - min(v):.1f} px)")
print("language gains", {k: round(v, 4) for k, v in lg.items()}, "margins", {k: v for k, v in st.items()
                                                                            if k.startswith(("n_", "med"))})
